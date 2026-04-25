import argparse
import json

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, CLIPProcessor

from src.datasets.query_builder import build_queries
from src.metrics import evaluate
from src.utils import get_device, read_index

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
KS = [5, 10, 25, 50, 100]


def encode_queries(
    model,
    processor: CLIPProcessor,
    queries: list[dict],
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    texts = [q["text"] for q in queries]
    all_embeds = []

    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)
            feats = model.get_text_features(**inputs).pooler_output
            feats = F.normalize(feats.float(), dim=-1)
            all_embeds.append(feats.cpu())

    return torch.cat(all_embeds, dim=0)  # (num_queries, D)


def print_breakdown(label: str, scores: dict[str, dict]):
    print(f"\n  [{label}]")
    for group_key, group_scores in sorted(scores.items()):
        line = "  " + group_key + ":"
        for metric, val in group_scores.items():
            line += f"  {metric}={val:.4f}"
        print(line)


def run_eval(args):
    device = args.device or get_device()

    # Load index and build metadata DataFrame
    faiss_index, meta = read_index(args.index_dir)
    records = meta["metadata"]

    for i, rec in enumerate(records):
        rec["row_idx"] = i

    if records and "predicted_label" in records[0] and "category" not in records[0]:
        for rec in records:
            rec["category"] = rec.pop("predicted_label")

    meta_df = pl.DataFrame(records)

    # Build queries
    queries = build_queries(meta_df, min_relevant=args.min_relevant)
    print(f"Queries: {len(queries)}")

    if not queries:
        print("No queries meet the min_relevant threshold. Exiting.")
        return

    # Load vanilla CLIP (no checkpoint, no LoRA)
    print(f"Loading CLIP: {CLIP_MODEL_NAME}")
    model = AutoModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    model = torch.compile(model)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    # Encode all query texts
    query_embeds = encode_queries(model, processor, queries, device, args.batch_size)
    query_embeds_np = query_embeds.numpy()

    # Retrieve from FAISS
    topk = max(KS) + 1
    all_I = []
    for embed in query_embeds_np:
        _, I = faiss_index.search(embed.reshape(1, -1), topk)
        all_I.append(I)
    I = np.concatenate(all_I, axis=0)

    all_pred = [row.tolist() for row in I]
    all_gt = [q["relevant_indices"] for q in queries]

    results = evaluate(all_pred, all_gt, query_img_ids=None, ks=KS)

    print("\n=== Overall (CLIP baseline) ===")
    for metric, val in results.items():
        print(f"  {metric}: {val:.4f}")

    if args.breakdown:
        categories = sorted(set(q["category"] for q in queries))
        cat_results = {}
        for cat in categories:
            cat_mask = [i for i, q in enumerate(queries) if q["category"] == cat]
            sub_pred = [all_pred[i] for i in cat_mask]
            sub_gt = [all_gt[i] for i in cat_mask]
            cat_results[cat] = evaluate(sub_pred, sub_gt, query_img_ids=None, ks=KS)
        print_breakdown("by category", cat_results)
        results["by_category"] = cat_results

        countries = sorted(set(q["country"] for q in queries))
        ctr_results = {}
        for ctr in countries:
            ctr_mask = [i for i, q in enumerate(queries) if q["country"] == ctr]
            sub_pred = [all_pred[i] for i in ctr_mask]
            sub_gt = [all_gt[i] for i in ctr_mask]
            ctr_results[ctr] = evaluate(sub_pred, sub_gt, query_img_ids=None, ks=KS)
        print_breakdown("by country", ctr_results)
        results["by_country"] = ctr_results

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP baseline evaluation (no fine-tuning)")
    parser.add_argument("--index-dir", required=True, help="Directory with index.index + metadata.json")
    parser.add_argument("--min-relevant", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_eval(args)
