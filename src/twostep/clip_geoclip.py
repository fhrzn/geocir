"""Two-step retrieval: image-filter -> GeoCLIP GPS prediction -> reverse geocode.

Mirror image of ``src/evaluate_two_step.py`` with the stages reversed:

    evaluate_two_step.py : GPS pre-filter (text->location) -> re-rank in image space
    this module          : text->image filter (any backbone) -> per-candidate GPS
                           prediction (GeoCLIP) -> reverse-geocode -> keep
                           candidates whose predicted country matches the query

Pipeline
--------
1. **Image filter.** Encode the query text with ``--model`` (clip / siglip / blip /
   blip2 / uniir / ...) and search its FAISS image index for the top
   ``--topk-first`` candidates.
2. **GPS prediction (GeoCLIP).** Predicted country per distinct candidate image.
   Read from ``--gps-cache`` (parquet from ``precompute_geoclip_gps.py``) when
   given; any id missing from the cache falls back to a live GeoCLIP pass.
3. **Reverse geocode.** Predicted coordinate -> ISO-3166 alpha-2 country code
   (already materialised in the cache; done with ``reverse_geocoder`` on fallback).
4. **Geo filter.** Drop candidates whose predicted country != the query country.
   Survivors keep their Stage-1 ranking, truncated to ``--topk-second``.
   ``--backfill`` refills the tail from the remaining Stage-1 candidates so recall
   is never worse than the ``--model``-only baseline.

Since GeoCLIP predicts from the image alone, one ``--gps-cache`` built over the DB
serves every Stage-1 backbone.

Reporting (`--breakdown` / `--marginal`) is shared with ``src/evaluate.py`` via
``src.eval_report.eval_and_report``, so scores are computed in DB-row-index space
(``relevant_ids`` are mapped to index rows on load).

Example
-------
    python -m src.twostep.clip_geoclip \
        --model siglip --index-dir index/baseline-siglip \
        --test-query .../test_templated_queries.json \
        --gps-cache outputs/geoclip_gps_test.parquet \
        --topk-first 1000 --topk-second 100 --backfill \
        --breakdown --marginal \
        --output outputs/twostep-siglip-geoclip.json
"""

import os
from argparse import ArgumentParser

import country_converter as coco
import numpy as np
import polars as pl
import reverse_geocoder as rg
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.eval_report import eval_and_report
from src.evaluate import _load_index_and_queries
from src.model.backbones import MODEL_NAMES, load_backbone
from src.utils import get_device


def _image_path(record: dict, image_root: str, ext: str) -> str:
    return os.path.join(image_root, record["src"], f"{record['id']}{ext}")


def _retrieve(index, query_emb: np.ndarray, topk: int) -> np.ndarray:
    """Stage 1: text->image search. Returns (num_queries, topk) row indices."""
    results = []
    for embed in tqdm(query_emb, desc="stage-1 filter", leave=False):
        _, indices = index.search(embed.reshape(1, -1), topk)
        results.append(indices[0])
    return np.stack(results, axis=0)


@torch.no_grad()
def _predict_gps(geoclip, paths: list[str], device, batch_size: int) -> np.ndarray:
    """GeoCLIP top-1 GPS prediction for a list of image paths.

    Returns an (N, 2) float array of (lat, lon). Rows whose image could not be
    read are left as NaN so the caller can skip them.
    """
    model = geoclip.model
    gallery = model.gps_gallery.to(device)
    loc_feat = F.normalize(model.location_encoder(gallery), dim=1)

    coords = np.full((len(paths), 2), np.nan, dtype="float64")
    buf_idx: list[int] = []
    buf_img: list[torch.Tensor] = []

    def flush():
        if not buf_idx:
            return
        batch = torch.cat(buf_img, dim=0).to(device)
        img_feat = F.normalize(model.image_encoder(batch), dim=1)
        top = (img_feat @ loc_feat.t()).argmax(dim=-1).cpu()
        picked = model.gps_gallery[top].numpy()
        for j, row in zip(buf_idx, picked):
            coords[j] = row
        buf_idx.clear()
        buf_img.clear()

    for i, path in enumerate(tqdm(paths, desc="geoclip predict", leave=False)):
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, ValueError):
            continue
        buf_img.append(model.image_encoder.preprocess_image(img))
        buf_idx.append(i)
        if len(buf_idx) >= batch_size:
            flush()
    flush()
    return coords


def _target_country_codes(query: list[dict]) -> list[str | None]:
    names = [q.get("country", "") or "" for q in query]
    # Only ask coco about non-blank names; a blank `country` (e.g. category-only
    # --marginal queries) just means "no geo filter" and would otherwise spam
    # coco's "<name> not found in regex" log once per query.
    uniq = sorted({n for n in names if n})
    codes = coco.CountryConverter().convert(uniq, to="ISO2", not_found=None) if uniq else []
    if isinstance(codes, str):
        codes = [codes]
    # coco echoes the input name back when it can't resolve it; a real ISO2 code
    # is always two upper-case letters.
    lookup = {
        n: (c if isinstance(c, str) and len(c) == 2 and c.isupper() else None)
        for n, c in zip(uniq, codes)
    }
    return [lookup.get(n) for n in names]


def _predicted_country_codes(rows, coords) -> dict[int, str]:
    valid = ~np.isnan(coords[:, 0])
    if not valid.any():
        return {}
    hits = rg.search([tuple(c) for c in coords[valid]])
    return {int(r): h["cc"] for r, h in zip(np.asarray(rows)[valid], hits)}


def _resolve_pred_cc(uniq_rows, records, args, device) -> dict[int, str]:
    """row index -> predicted ISO2 country, from the cache first, GeoCLIP for misses."""
    cache_cc: dict[str, str | None] = {}
    cache_prob: dict[str, float] = {}
    if args.gps_cache:
        df = pl.read_parquet(args.gps_cache)
        ids = df["id"].to_list()
        cache_cc = dict(zip(ids, df["cc_pred"].to_list()))
        if "prob" in df.columns:
            cache_prob = dict(zip(ids, df["prob"].to_list()))

    pred_cc: dict[int, str] = {}
    missing: list[int] = []
    for r in uniq_rows:
        rid = str(records[r]["id"])
        if rid not in cache_cc:
            missing.append(r)
            continue
        cc, prob = cache_cc[rid], cache_prob.get(rid)
        if cc and (prob is None or prob >= args.prob_threshold):
            pred_cc[r] = cc  # else: left unknown -> excluded from the geo match

    if args.gps_cache:
        print(f"[gps] cache hit {len(uniq_rows) - len(missing)}/{len(uniq_rows)}")
    if missing:
        print(f"[gps] predicting {len(missing)} candidates with GeoCLIP")
        geoclip = load_backbone("geoclip", device)
        paths = [_image_path(records[r], args.image_root, args.image_ext) for r in missing]
        coords = _predict_gps(geoclip, paths, device, args.geoclip_batch_size)
        pred_cc.update(_predicted_country_codes(missing, coords))
    return pred_cc


def _make_search_fn(index, records, bb, args, device):
    """Build ``query dicts -> list[list[int]]`` (retrieved DB row indices).

    Runs the full pipeline: Stage-1 text->image filter, then the GeoCLIP geo
    filter. Predicted country codes are resolved once per row and memoised across
    calls, so ``--marginal`` reuses the work done for the main pass.
    """
    pred_cc: dict[int, str] = {}
    resolved: set[int] = set()

    def _ensure_pred_cc(rows):
        missing = sorted(r for r in rows if r not in resolved)
        if missing:
            pred_cc.update(_resolve_pred_cc(missing, records, args, device))
            resolved.update(missing)

    def search_fn(queries):
        texts = [q["text"] for q in queries]
        query_emb = bb.encode_text(texts, args.batch_size).numpy()
        first_pass = _retrieve(index, query_emb, args.topk_first)
        _ensure_pred_cc({int(r) for row in first_pass for r in row if r >= 0})

        preds = []
        for row, tcc in zip(first_pass, _target_country_codes(queries)):
            cands = [int(r) for r in row if r >= 0]
            if tcc is None:
                kept = cands[: args.topk_second]
            else:
                kept = [r for r in cands if pred_cc.get(r) == tcc][: args.topk_second]
                if args.backfill and len(kept) < args.topk_second:
                    seen = set(kept)
                    kept += [r for r in cands if r not in seen][: args.topk_second - len(kept)]
            preds.append(kept)
        return preds

    return search_fn


def main(args):
    device = get_device()

    index, queries, records = _load_index_and_queries(args)
    print(f"Queries: {len(queries)}")
    if not queries:
        print("No queries meet --min-relevant. Exiting.")
        return

    bb = load_backbone(args.model, device, args.ckpt_path)
    search_fn = _make_search_fn(index, records, bb, args, device)

    all_preds = search_fn(queries)
    all_gt = [q["relevant_indices"] for q in queries]

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    eval_and_report(all_preds, all_gt, queries, args, search_fn=search_fn, records=records)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="clip", choices=list(MODEL_NAMES),
                        help="Stage-1 text->image backbone")
    parser.add_argument("--index-dir", required=True,
                        help="FAISS image index dir for --model (index.index + metadata.json)")
    parser.add_argument("--ckpt-path", default=None, help="checkpoint for geotir/uniir")
    parser.add_argument("--test-query", type=str, required=True)
    parser.add_argument("--gps-cache", type=str, default=None,
                        help="parquet from precompute_geoclip_gps.py (id, cc_pred, prob); "
                             "ids missing from it fall back to a live GeoCLIP pass")
    parser.add_argument("--prob-threshold", type=float, default=0.0,
                        help="ignore cached GeoCLIP predictions below this softmax prob")
    parser.add_argument("--image-root", type=str, default="/mnt/yokoyamalab-nas/gldv2-full",
                        help="fallback image root: <root>/<record.src>/<record.id><ext>")
    parser.add_argument("--image-ext", type=str, default=".jpg")
    parser.add_argument("--batch-size", type=int, default=128, help="text encode batch")
    parser.add_argument("--geoclip-batch-size", type=int, default=64,
                        help="images per GeoCLIP GPS-prediction batch (fallback only)")
    parser.add_argument("--topk-first", type=int, default=1000,
                        help="candidates kept from the Stage-1 image filter")
    parser.add_argument("--topk-second", type=int, default=100,
                        help="final results kept after the geo filter")
    parser.add_argument("--min-relevant", type=int, default=5)
    parser.add_argument("--breakdown", action="store_true",
                        help="slice the joint (category, country) queries by category / country / region")
    parser.add_argument("--marginal", action="store_true",
                        help="also evaluate single-attribute queries: one per category "
                             "(relevance = all rows of that category) and one per country")
    parser.add_argument("--marginal-cat-template", default="a {}",
                        help="text template for category-only queries")
    parser.add_argument("--marginal-country-template", default="a landmark located in {}",
                        help="text template for country-only queries")
    parser.add_argument("--backfill", action="store_true",
                        help="refill the tail from Stage-1 candidates when the geo "
                             "filter leaves fewer than --topk-second")
    parser.add_argument("--output", type=str, default=None)

    # eval_and_report() is shared with src/evaluate.py, whose overall-line print
    # branches on args.two_step; this pipeline has no plain one-step mode.
    parser.set_defaults(two_step=False)

    args = parser.parse_args()
    main(args)
