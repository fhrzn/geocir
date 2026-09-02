import argparse
import json

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, CLIPProcessor

from src.data.countries import display_country
from src.data.query_builder import build_queries
from src.metrics import evaluate
from src.model.g3 import G3
from src.model.geoclip import GeoCLIP
from src.model.geotir.model import GeoTIRModel
from src.utils import get_device, read_index

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
KS = [5, 10, 25, 50, 100]


def _setup_clip(args, device):
    model = AutoModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    model = torch.compile(model)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    def encode(queries, batch_size):
        texts = [q["text"] for q in queries]
        all_embeds = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for i in tqdm(
                range(0, len(texts), batch_size), desc="encode queries", leave=False
            ):
                inputs = processor(
                    text=texts[i : i + batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                ).to(device)
                feats = model.get_text_features(**inputs).pooler_output
                all_embeds.append(F.normalize(feats.float(), dim=-1).cpu())
        return torch.cat(all_embeds, dim=0)

    return encode


def _setup_g3(args, device):
    model = G3().to(device).eval()

    def encode(queries, batch_size):
        texts = [q["text"] for q in queries]
        all_embeds = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for i in tqdm(
                range(0, len(texts), batch_size), desc="encode queries", leave=False
            ):
                inputs = model.preprocess_text(texts[i : i + batch_size])
                inputs = {k: v.to(device) for k, v in inputs.items()}
                text_emb = model.text_proj(model.text_model(**inputs)[1])
                txt2img = F.normalize(model.txt2img_proj(text_emb).float(), dim=-1)
                text_n = F.normalize(text_emb.float(), dim=-1)
                loc_pad = torch.zeros_like(text_n)
                out = F.normalize(torch.cat([txt2img, text_n, loc_pad], dim=1), dim=-1)
                all_embeds.append(out.cpu())
        return torch.cat(all_embeds, dim=0)

    return encode


def _setup_geoclip(args, device):
    model = GeoCLIP().to(device).eval()

    def encode(queries, batch_size):
        texts = [q["text"] for q in queries]
        all_embeds = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for i in tqdm(
                range(0, len(texts), batch_size), desc="encode queries", leave=False
            ):
                inputs = model.text_encoder.preprocess_text(texts[i : i + batch_size])
                inputs = {k: v.to(device) for k, v in inputs.items()}
                feats = model.text_encoder(**inputs)
                all_embeds.append(F.normalize(feats.float(), dim=-1).cpu())
        return torch.cat(all_embeds, dim=0)

    return encode


def _setup_geotir(args, device):
    if not args.ckpt_path:
        raise ValueError("--ckpt-path is required for --model geotir")
    model = GeoTIRModel(clip_model_name=CLIP_MODEL_NAME).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.eval()
    model = torch.compile(model)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    def encode(queries, batch_size):
        texts = [q["text"] for q in queries]
        all_embeds = []
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            for i in tqdm(
                range(0, len(texts), batch_size), desc="encode queries", leave=False
            ):
                inputs = processor(
                    text=texts[i : i + batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                ).to(device)
                embeds = model.encode_texts(inputs["input_ids"], inputs["attention_mask"])
                all_embeds.append(embeds.cpu().float())
        return torch.cat(all_embeds, dim=0)

    return encode


_MODEL_REGISTRY = {
    "clip": _setup_clip,
    "g3": _setup_g3,
    "geoclip": _setup_geoclip,
    "geotir": _setup_geotir,
}


def _load_index_and_queries(args):
    faiss_index, meta = read_index(args.index_dir)
    records = meta["metadata"]

    if args.test_query:
        with open(args.test_query) as f:
            payload = json.load(f)
        raw = payload["data"] if isinstance(payload, dict) else payload

        # ground truth may be given as FAISS row indices ("relevant_indices")
        # or as image ids ("relevant_ids"); normalise everything to row indices.
        id_to_row = {str(rec["id"]): i for i, rec in enumerate(records) if "id" in rec}
        queries, n_missing, n_dropped = [], 0, 0
        for q in raw:
            if q.get("relevant_indices") is not None:
                gt = [int(x) for x in q["relevant_indices"]]
            else:
                gt = []
                for _id in q.get("relevant_ids", []):
                    row = id_to_row.get(str(_id))
                    if row is None:
                        n_missing += 1
                    else:
                        gt.append(row)
            if len(gt) < args.min_relevant:
                n_dropped += 1
                continue
            queries.append({**q, "relevant_indices": gt})

        if n_missing:
            print(f"[warn] {n_missing} relevant_ids absent from the index (skipped)")
        if n_dropped:
            print(f"[info] {n_dropped} queries below --min-relevant={args.min_relevant} (skipped)")
        return faiss_index, queries, records

    for i, rec in enumerate(records):
        rec["row_idx"] = i

    if records and "predicted_label" in records[0] and "category" not in records[0]:
        for rec in records:
            rec["category"] = rec.pop("predicted_label")

    meta_df = pl.DataFrame(records)
    queries = build_queries(meta_df, min_relevant=args.min_relevant)
    return faiss_index, queries, records


def _retrieve(faiss_index, query_embeds_np, topk):
    results = []
    for embed in tqdm(query_embeds_np, desc="retrieve", leave=False):
        _, I = faiss_index.search(embed.reshape(1, -1), topk)
        results.append(I)
    return np.concatenate(results, axis=0)


def _print_breakdown(label, scores):
    print(f"\n  [{label}]")
    for group_key, group_scores in sorted(scores.items()):
        line = "  " + str(group_key) + ":"
        for metric, val in group_scores.items():
            line += f"  {metric}={val:.4f}"
        print(line)


def _marginal_queries(records, axis, template, min_relevant):
    """One query per distinct `axis` value; relevance = every index row with that value.

    axis: "category" or "country". This measures the model on the single-attribute
    task in isolation, to see whether the joint (category AND country) score is
    bottlenecked by the category axis, the country axis, or their conjunction.
    """
    groups = {}
    for i, rec in enumerate(records):
        key = rec.get(axis)
        if key is None or key == "":
            continue
        groups.setdefault(key, []).append(i)

    disp = display_country if axis == "country" else (lambda x: x)
    out = []
    for key, idxs in sorted(groups.items()):
        if len(idxs) < min_relevant:
            continue
        out.append({axis: key, "text": template.format(disp(key)), "relevant_indices": idxs})
    return out


def _score_query_set(faiss_index, queries, encode_fn, args, axis, label):
    if not queries:
        print(f"\n=== Marginal: {label} — no groups meet --min-relevant ===")
        return {}
    embeds = encode_fn(queries, args.batch_size).numpy()
    topk = max(KS) + 1
    pred = [row.tolist() for row in _retrieve(faiss_index, embeds, topk)]
    gt = [q["relevant_indices"] for q in queries]

    overall = evaluate(pred, gt, query_img_ids=None, ks=KS)
    per_key = {
        q[axis]: evaluate([pred[i]], [gt[i]], query_img_ids=None, ks=KS)
        for i, q in enumerate(queries)
    }
    print(f"\n=== Marginal: {label} ({len(queries)} queries) ===")
    for metric, val in overall.items():
        print(f"  {metric}: {val:.4f}")
    _print_breakdown(f"marginal {label}", per_key)
    return {"overall": overall, "per_key": per_key}


def _eval_and_report(all_pred, all_gt, queries, args, faiss_index=None, encode_fn=None, records=None):
    results = evaluate(all_pred, all_gt, query_img_ids=None, ks=KS)

    print(f"\n=== Overall ({args.model}) ===")
    for metric, val in results.items():
        print(f"  {metric}: {val:.4f}")

    if args.breakdown:
        categories = sorted(set(q["category"] for q in queries))
        cat_results = {
            cat: evaluate(
                [all_pred[i] for i, q in enumerate(queries) if q["category"] == cat],
                [all_gt[i] for i, q in enumerate(queries) if q["category"] == cat],
                query_img_ids=None, ks=KS,
            )
            for cat in categories
        }
        _print_breakdown("by category", cat_results)
        results["by_category"] = cat_results

        countries = sorted(set(q["country"] for q in queries))
        ctr_results = {
            ctr: evaluate(
                [all_pred[i] for i, q in enumerate(queries) if q["country"] == ctr],
                [all_gt[i] for i, q in enumerate(queries) if q["country"] == ctr],
                query_img_ids=None, ks=KS,
            )
            for ctr in countries
        }
        _print_breakdown("by country", ctr_results)
        results["by_country"] = ctr_results

        if any("region" in q for q in queries):
            continents = sorted(set(q["region"] for q in queries if "region" in q and q["region"]))
            cont_results = {
                cont: evaluate(
                    [all_pred[i] for i, q in enumerate(queries) if q.get("region") == cont],
                    [all_gt[i] for i, q in enumerate(queries) if q.get("region") == cont],
                    query_img_ids=None, ks=KS,
                )
                for cont in continents
            }
            _print_breakdown("by continent", cont_results)
            results["by_continent"] = cont_results

    if args.marginal and records is not None and encode_fn is not None:
        cat_q = _marginal_queries(records, "category", args.marginal_cat_template, args.min_relevant)
        ctr_q = _marginal_queries(records, "country", args.marginal_country_template, args.min_relevant)
        results["marginal_category"] = _score_query_set(
            faiss_index, cat_q, encode_fn, args, "category", "category-only"
        )
        results["marginal_country"] = _score_query_set(
            faiss_index, ctr_q, encode_fn, args, "country", "country-only"
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


def run_eval(args):
    device = args.device or get_device()

    faiss_index, queries, records = _load_index_and_queries(args)
    print(f"Queries: {len(queries)}")

    if not queries:
        print("No queries meet the min_relevant threshold. Exiting.")
        return

    encode_fn = _MODEL_REGISTRY[args.model](args, device)
    query_embeds_np = encode_fn(queries, args.batch_size).numpy()

    topk = max(KS) + 1
    I = _retrieve(faiss_index, query_embeds_np, topk)

    all_pred = [row.tolist() for row in I]
    all_gt = [q["relevant_indices"] for q in queries]

    _eval_and_report(
        all_pred, all_gt, queries, args,
        faiss_index=faiss_index, encode_fn=encode_fn, records=records,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(_MODEL_REGISTRY))
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--ckpt-path")
    parser.add_argument("--test-query", type=str, default=None)
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
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_eval(args)
