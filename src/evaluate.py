import argparse
import json

import numpy as np
from tqdm.auto import tqdm

from src.eval_report import effective_k1, eval_and_report
from src.metrics import DEFAULT_KS
from src.model.backbones import MODEL_NAMES, load_backbone
from src.utils import get_device, read_index


def _load_index_and_queries(args):
    faiss_index, meta = read_index(args.index_dir)
    records = meta["metadata"]

    with open(args.test_query) as f:
        payload = json.load(f)
    raw = payload["data"] if isinstance(payload, dict) else payload

    id_to_row = {str(rec["id"]): i for i, rec in enumerate(records) if "id" in rec}
    queries, n_missing, n_dropped = [], 0, 0
    for q in raw:
        if q.get("relevant_indices") is not None:
            gt = [int(x) for x in q["relevant_indices"]]
        else:
            rel_ids = [str(x) for x in q.get("relevant_ids", [])]
            gt = [id_to_row[x] for x in rel_ids if x in id_to_row]
            n_missing += len(rel_ids) - len(gt)
        if len(gt) < args.min_relevant:
            n_dropped += 1
            continue
        queries.append({**q, "relevant_indices": gt})

    if n_missing:
        print(f"[warn] {n_missing} relevant_ids absent from the index (skipped)")
    if n_dropped:
        print(f"[info] {n_dropped} queries below --min-relevant={args.min_relevant} (skipped)")
    return faiss_index, queries, records


def _retrieve(faiss_index, query_embeds_np, topk):
    results = []
    for embed in tqdm(query_embeds_np, desc="retrieve", leave=False):
        _, I = faiss_index.search(embed.reshape(1, -1), topk)
        results.append(I)
    return np.concatenate(results, axis=0)


def _two_step_search(gps_index, img_matrix, q_txt, q_gps, k1, topk):
    """Two-step retrieval: geo pre-filter, then rank in image space.

    `gps_index` rows are aligned with `img_matrix` rows (row i = same DB image).
    For each query: take the top-`k1` rows by GPS-embedding similarity, then
    re-rank just those with the image-space query embedding and keep `topk`.
    """
    k1 = min(k1, gps_index.ntotal)
    preds = []
    for i in tqdm(range(len(q_txt)), desc="two-step retrieve", leave=False):
        _, cand = gps_index.search(q_gps[i].reshape(1, -1), k1)
        cand = cand[0]
        cand = cand[cand >= 0]  # faiss pads with -1 when k1 > ntotal
        scores = img_matrix[cand] @ q_txt[i]
        order = np.argsort(-scores)[:topk]
        preds.append(cand[order].tolist())
    return preds


def run_eval(args):
    device = args.device or get_device()

    faiss_index, queries, records = _load_index_and_queries(args)
    print(f"Queries: {len(queries)}")

    if not queries:
        print("No queries meet the min_relevant threshold. Exiting.")
        return

    bb = load_backbone(args.model, device, args.ckpt_path)
    topk = max(DEFAULT_KS) + 1

    gps_index = img_matrix = None
    if args.two_step:
        if not (bb.supports_two_step and bb.encode_text_gps is not None):
            raise SystemExit(
                f"--two-step is not supported for --model {args.model} "
                "(no text->location path)"
            )
        try:
            gps_index, _ = read_index(args.index_dir, prefix="gps")
        except (FileNotFoundError, RuntimeError) as e:
            raise SystemExit(
                f"--two-step needs a GPS index (gps_index.index / gps_metadata.json) in "
                f"{args.index_dir}. Re-run: python -m src.index --model {args.model} ... "
                f"[{e}]"
            ) from e
        img_matrix = faiss_index.reconstruct_n(0, faiss_index.ntotal)
        print(f"two-step: gps index {gps_index.ntotal} rows, k1={effective_k1(args)}")

    def search_fn(qs):
        """query dicts -> list[list[int]] of retrieved row indices (topk each)."""
        texts = [q["text"] for q in qs]
        q_txt = bb.encode_text(texts, args.batch_size).numpy()
        if args.two_step:
            q_gps = bb.encode_text_gps(texts, args.batch_size).numpy()
            return _two_step_search(gps_index, img_matrix, q_txt, q_gps,
                                    effective_k1(args), topk)
        return [row.tolist() for row in _retrieve(faiss_index, q_txt, topk)]

    all_pred = search_fn(queries)
    all_gt = [q["relevant_indices"] for q in queries]

    eval_and_report(all_pred, all_gt, queries, args, search_fn=search_fn, records=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--ckpt-path")
    parser.add_argument("--test-query", type=str, required=True,
                        help="query JSON: list (or {'data': [...]}) of "
                             "{text, category, country, region, relevant_ids}")
    parser.add_argument("--min-relevant", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--breakdown", action="store_true",
                        help="slice the joint (category, country) queries by category / country / region")
    parser.add_argument("--marginal", action="store_true",
                        help="also evaluate single-attribute queries: one per category "
                             "(relevance = all rows of that category) and one per country")
    parser.add_argument("--marginal-cat-template", default="a {}",
                        help="text template for category-only queries")
    parser.add_argument("--marginal-country-template", default="a landmark located in {}",
                        help="text template for country-only queries")
    parser.add_argument("--two-step", action="store_true",
                        help="two-step retrieval (two-step backbones only): geo pre-filter "
                             "via the GPS index, then re-rank candidates in image space")
    parser.add_argument("--two-step-k1", type=int, default=1000,
                        help="candidate pool size from the GPS pre-filter stage")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_eval(args)
