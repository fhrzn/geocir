import argparse
import json

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, CLIPProcessor

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
            for i in range(0, len(texts), batch_size):
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
            for i in range(0, len(texts), batch_size):
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
            for i in range(0, len(texts), batch_size):
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
            for i in range(0, len(texts), batch_size):
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

    if args.test_query:
        with open(args.test_query) as f:
            payload = json.load(f)
        queries = payload["data"] if isinstance(payload, dict) else payload
        return faiss_index, queries

    records = meta["metadata"]

    for i, rec in enumerate(records):
        rec["row_idx"] = i

    if records and "predicted_label" in records[0] and "category" not in records[0]:
        for rec in records:
            rec["category"] = rec.pop("predicted_label")

    meta_df = pl.DataFrame(records)
    queries = build_queries(meta_df, min_relevant=args.min_relevant)
    return faiss_index, queries


def _retrieve(faiss_index, query_embeds_np, topk):
    results = []
    for embed in query_embeds_np:
        _, I = faiss_index.search(embed.reshape(1, -1), topk)
        results.append(I)
    return np.concatenate(results, axis=0)


def _print_breakdown(label, scores):
    print(f"\n  [{label}]")
    for group_key, group_scores in sorted(scores.items()):
        line = "  " + group_key + ":"
        for metric, val in group_scores.items():
            line += f"  {metric}={val:.4f}"
        print(line)


def _eval_and_report(all_pred, all_gt, queries, args):
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

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


def run_eval(args):
    device = args.device or get_device()

    faiss_index, queries = _load_index_and_queries(args)
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

    _eval_and_report(all_pred, all_gt, queries, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(_MODEL_REGISTRY))
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--ckpt-path")
    parser.add_argument("--test-query", type=str, default=None)
    parser.add_argument("--min-relevant", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_eval(args)
