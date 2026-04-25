import argparse
import json

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from src.datasets.query_builder import build_queries
from src.g3 import G3
from src.metrics import evaluate
from src.utils import get_device, read_index

KS = [5, 10, 25, 50, 100]


def encode_queries(
    model: G3,
    queries: list[dict],
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    texts = [q["text"] for q in queries]
    all_embeds = []

    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        for i in range(0, len(texts), batch_size):
            inputs = model.preprocess_text(texts[i : i + batch_size])
            inputs = {k: v.to(device) for k, v in inputs.items()}

            text_emb = model.text_proj(model.text_model(**inputs)[1])  # (B, 768)

            # mirror the 3-component index structure
            txt2img = F.normalize(model.txt2img_proj(text_emb).float(), dim=-1)   # text→image space
            text_n  = F.normalize(text_emb.float(), dim=-1)                        # raw text (matches img2txt space)
            loc_pad = torch.zeros_like(text_n)                                      # no text→location mapping

            out = F.normalize(torch.cat([txt2img, text_n, loc_pad], dim=1), dim=-1)
            all_embeds.append(out.cpu())

    return torch.cat(all_embeds, dim=0)


def print_breakdown(label: str, scores: dict[str, dict]):
    print(f"\n  [{label}]")
    for group_key, group_scores in sorted(scores.items()):
        line = "  " + group_key + ":"
        for metric, val in group_scores.items():
            line += f"  {metric}={val:.4f}"
        print(line)


def run_eval(args):
    device = args.device or get_device()

    faiss_index, meta = read_index(args.index_dir)
    records = meta["metadata"]

    for i, rec in enumerate(records):
        rec["row_idx"] = i

    if records and "predicted_label" in records[0] and "category" not in records[0]:
        for rec in records:
            rec["category"] = rec.pop("predicted_label")

    meta_df = pl.DataFrame(records)

    queries = build_queries(meta_df, min_relevant=args.min_relevant)
    print(f"Queries: {len(queries)}")

    if not queries:
        print("No queries meet the min_relevant threshold. Exiting.")
        return

    print("Loading G3...")
    model = G3().to(device).eval()

    query_embeds = encode_queries(model, queries, device, args.batch_size)
    query_embeds_np = query_embeds.numpy()

    topk = max(KS) + 1
    retrieved = []
    for embed in query_embeds_np:
        _, idx = faiss_index.search(embed.reshape(1, -1), topk)
        retrieved.append(idx)
    retrieved = np.concatenate(retrieved, axis=0)

    all_pred = [row.tolist() for row in retrieved]
    all_gt = [q["relevant_indices"] for q in queries]

    results = evaluate(all_pred, all_gt, query_img_ids=None, ks=KS)

    print("\n=== Overall (G3 baseline) ===")
    for metric, val in results.items():
        print(f"  {metric}: {val:.4f}")

    if args.breakdown:
        categories = sorted(set(q["category"] for q in queries))
        cat_results = {}
        for cat in categories:
            mask = [i for i, q in enumerate(queries) if q["category"] == cat]
            cat_results[cat] = evaluate(
                [all_pred[i] for i in mask], [all_gt[i] for i in mask],
                query_img_ids=None, ks=KS,
            )
        print_breakdown("by category", cat_results)
        results["by_category"] = cat_results

        countries = sorted(set(q["country"] for q in queries))
        ctr_results = {}
        for ctr in countries:
            mask = [i for i, q in enumerate(queries) if q["country"] == ctr]
            ctr_results[ctr] = evaluate(
                [all_pred[i] for i in mask], [all_gt[i] for i in mask],
                query_img_ids=None, ks=KS,
            )
        print_breakdown("by country", ctr_results)
        results["by_country"] = ctr_results

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G3 baseline evaluation")
    parser.add_argument("--index-dir", required=True, help="Directory with index.index + metadata.json")
    parser.add_argument("--min-relevant", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_eval(args)
