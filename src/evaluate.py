import argparse
import json

import numpy as np
import polars as pl
import torch
from transformers import CLIPProcessor

from src.data.query_builder import build_queries
from src.geotir.model import GeoTIRModel
from src.metrics import evaluate
from src.utils import get_device, read_index

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
KS = [5, 10, 25, 50, 100]


def encode_queries(
    model: GeoTIRModel,
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
            embeds = model.encode_texts(inputs["input_ids"], inputs["attention_mask"])
            all_embeds.append(embeds.cpu().float())

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

    # Attach FAISS row position as row_idx
    for i, rec in enumerate(records):
        rec["row_idx"] = i

    # Rename predicted_label → category if present
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

    # Load model + processor
    model = GeoTIRModel(clip_model_name=CLIP_MODEL_NAME).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.eval()
    model = torch.compile(model)

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    # Encode all query texts
    query_embeds = encode_queries(model, processor, queries, device, args.batch_size)
    query_embeds_np = query_embeds.numpy()

    # Retrieve from FAISS in batches to avoid memory exhaustion
    topk = max(KS) + 1  # +1 to allow self-hit removal
    all_I = []
    for embed in query_embeds_np:
        _, I = faiss_index.search(embed.reshape(1, -1), topk)
        all_I.append(I)
    I = np.concatenate(all_I, axis=0)

    all_pred = [row.tolist() for row in I]
    all_gt = [q["relevant_indices"] for q in queries]

    # Evaluate — pass None for query_img_ids since text queries have no self-hit
    results = evaluate(all_pred, all_gt, query_img_ids=None, ks=KS)

    print("\n=== Overall ===")
    for metric, val in results.items():
        print(f"  {metric}: {val:.4f}")

    # Per-category breakdown
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

        # Per-country breakdown
        countries = sorted(set(q["country"] for q in queries))
        ctr_results = {}
        for ctr in countries:
            ctr_mask = [i for i, q in enumerate(queries) if q["country"] == ctr]
            sub_pred = [all_pred[i] for i in ctr_mask]
            sub_gt = [all_gt[i] for i in ctr_mask]
            ctr_results[ctr] = evaluate(sub_pred, sub_gt, query_img_ids=None, ks=KS)
        print_breakdown("by country", ctr_results)
        results["by_country"] = ctr_results

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True, help="Directory with index.index + metadata.json")
    parser.add_argument("--ckpt-path", required=True, help="Model checkpoint (.pt file)")
    parser.add_argument("--min-relevant", type=int, default=5, help="Min images per (category, country) group to form a query")
    parser.add_argument("--batch-size", type=int, default=256, help="Text encoding batch size")
    parser.add_argument("--breakdown", action="store_true", help="Also print per-category and per-country mAP")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g. cuda, cpu). Auto-detected if omitted.")
    args = parser.parse_args()

    run_eval(args)
