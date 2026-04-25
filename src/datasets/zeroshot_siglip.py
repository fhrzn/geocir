import argparse
import hashlib
import json
import os

import polars as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SiglipImageProcessor, SiglipModel, SiglipProcessor, SiglipTokenizer

HF_CHECKPOINT_NAME = "google/siglip-so400m-patch14-384"
BASE_IMG_PATH = "/mnt/yokoyamalab-nas/gldv2-full/index"
CSV_PATH = "../datasets/google-landmark/index_set_fixed.csv"
OUTPUT_PATH = "/mnt/yokoyamalab-nas/gldv2-full/index_set_fixed_w_category.csv"
CHECKPOINT_PATH = "../datasets/google-landmark/index_set_fixed_category_ckpt.csv"
LABEL_ENSEMBLE_PATH = "./label_ensemble.json"
EMBEDDINGS_CACHE_DIR = "./checkpoints"


class ImageDataset(Dataset):
    def __init__(self, df: pl.DataFrame, img_col: str = "id", img_base_path: str = ""):
        self.df = df
        self.img_col = img_col
        self.img_base_path = img_base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id = self.df[index][self.img_col].item()
        path = os.path.join(self.img_base_path, f"{img_id}.jpg")
        img = Image.open(path).convert("RGB")
        return {"image": img, "img_id": img_id}


def make_collate_fn(processor):
    def collate_fn(batch):
        images = [b["image"] for b in batch]
        img_ids = [b["img_id"] for b in batch]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        return inputs, img_ids
    return collate_fn


def _label_cache_path(cache_dir: str, checkpoint_name: str, label_ensemble: dict) -> str:
    key = checkpoint_name + json.dumps(label_ensemble, sort_keys=True)
    digest = hashlib.md5(key.encode()).hexdigest()[:10]
    safe_name = checkpoint_name.replace("/", "_")
    return os.path.join(cache_dir, f"label_embeddings_{safe_name}_{digest}.pt")


def precompute_label_embeddings(
    model, processor, label_ensemble: dict[str, list[str]], device: str,
    cache_dir: str | None = None,
) -> tuple[list[str], torch.Tensor]:
    """Encode each label's prompts, average into one embedding per label, and cache the result."""
    if cache_dir:
        cache_path = _label_cache_path(cache_dir, model.config.name_or_path, label_ensemble)
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location=device, weights_only=True)
            print(f"Loaded label embeddings from cache: {cache_path}")
            return cached["labels"], cached["embeddings"]

    labels = list(label_ensemble.keys())
    averaged = []

    for prompts in label_ensemble.values():
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        # transformers 5.x returns BaseModelOutputWithPooling; .pooler_output is the pooled rep
        feats = model.get_text_features(**inputs).pooler_output
        feats = F.normalize(feats, dim=-1).mean(dim=0)
        averaged.append(feats)

    # Re-normalize after averaging so dot product == cosine similarity
    label_embeddings = F.normalize(torch.stack(averaged), dim=-1)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({"labels": labels, "embeddings": label_embeddings.cpu()}, cache_path)
        print(f"Saved label embeddings to cache: {cache_path}")

    return labels, label_embeddings  # (num_labels, dim)


def _save_to_path(new_results: list[dict], already_done: set, path: str) -> None:
    new_df = pl.DataFrame(new_results)
    if already_done and os.path.exists(path):
        new_df = pl.concat([pl.read_csv(path), new_df])
    new_df.write_csv(path)


def main(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.label_ensemble_path) as f:
        label_ensemble = json.load(f)

    df = pl.read_csv(args.csv_path)
    print(f"Total images: {len(df)}")

    already_done: set[str] = set()
    if args.resume and os.path.exists(args.checkpoint_path):
        ckpt = pl.read_csv(args.checkpoint_path)
        already_done = set(ckpt["id"].to_list())
        print(f"Resuming — skipping {len(already_done)} already processed images")
        df = df.filter(~pl.col("id").is_in(already_done))

    if len(df) == 0:
        print("Nothing left to process. Exiting.")
        return

    # SiglipProcessor.from_pretrained is broken in transformers 5.x (sentencepiece tokenizer
    # lookup returns None); build the processor manually from its two components instead.
    tokenizer = SiglipTokenizer.from_pretrained(args.checkpoint_name)
    image_processor = SiglipImageProcessor.from_pretrained(args.checkpoint_name)
    processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = SiglipModel.from_pretrained(args.checkpoint_name).to(device)
    model.eval()
    model = torch.compile(model)

    dataset = ImageDataset(df, img_col="id", img_base_path=args.img_base_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
        collate_fn=make_collate_fn(processor),
    )

    results: list[dict] = []

    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        labels, label_embeddings = precompute_label_embeddings(
            model, processor, label_ensemble, device, cache_dir=args.embeddings_cache_dir
        )
        label_embeddings = label_embeddings.to(device)
        print(f"Loaded embeddings for {len(labels)} labels: {labels}")

        for img_inputs, img_ids in tqdm(loader, desc="Classifying"):
            inputs = {k: v.to(device, non_blocking=True) for k, v in img_inputs.items()}
            image_feats = F.normalize(model.get_image_features(**inputs).pooler_output, dim=-1)

            scores = image_feats @ label_embeddings.T  # (batch, num_labels)
            pred_indices = scores.argmax(dim=-1).tolist()
            scores_list = scores.tolist()

            for img_id, pred_idx, score_row in zip(img_ids, pred_indices, scores_list):
                results.append({
                    "id": img_id,
                    "predicted_label": labels[pred_idx],
                    **{f"score_{label}": score for label, score in zip(labels, score_row)},
                })

            if len(results) % args.checkpoint_every == 0:
                _save_to_path(results, already_done, args.checkpoint_path)

    _save_to_path(results, already_done, args.output_path)
    print(f"Done. Results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-name", type=str, default=HF_CHECKPOINT_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--img-base-path", type=str, default=BASE_IMG_PATH)
    parser.add_argument("--csv-path", type=str, default=CSV_PATH)
    parser.add_argument("--output-path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--checkpoint-path", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--label-ensemble-path", type=str, default=LABEL_ENSEMBLE_PATH)
    parser.add_argument("--embeddings-cache-dir", type=str, default=EMBEDDINGS_CACHE_DIR)
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g. cpu, cuda, cuda:1). Auto-detected if omitted.")
    args = parser.parse_args()

    main(args)
