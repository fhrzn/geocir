import argparse
import os

import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

HF_CHECKPOINT_NAME = "facebook/mask2former-swin-large-ade-semantic"
BASE_IMG_PATH = "/mnt/yokoyamalab-nas/gldv2-full/index"
CSV_PATH = "../datasets/google-landmark/index_set_fixed.csv"
OUTPUT_PATH = "/mnt/yokoyamalab-nas/gldv2-full/index_set_fixed_w_segment.csv"
CHECKPOINT_PATH = "../datasets/google-landmark/index_set_fixed_segment_ckpt.csv"

# ADE20K class IDs (0-indexed)
LANDMARK_SEGMENT_IDS = torch.tensor([
    1, 25, 48, 84,       # building, house, skyscraper, tower
    61,                  # bridge
    132, 42, 40, 104,   # sculpture, column, pedestal, fountain
    53, 59, 95, 38,     # stairs, stairway, balustrade, railing
    140, 51, 113,       # pier, grandstand, waterfall
])


class ImageDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        img_col: str = "id",
        img_base_path: str = "",
        src_col: str = "folder",
    ):
        self.df = df.to_dicts()
        self.img_col = img_col
        self.img_base_path = img_base_path
        self.src_col = src_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df[index]
        img_id = row[self.img_col]
        filename = img_id if ".jpg" in img_id else f"{img_id}.jpg"
        if self.src_col:
            path = os.path.join(self.img_base_path, row[self.src_col], filename)
        else:
            path = os.path.join(self.img_base_path, filename)
        img = Image.open(path).convert("RGB")
        size = img.size[::-1]  # (H, W) expected by post_process_semantic_segmentation
        return {"image": img, "size": size, "img_id": img_id}


def make_collate_fn(processor):
    def collate_fn(batch):
        images = [b["image"] for b in batch]
        sizes = [b["size"] for b in batch]
        img_ids = [b["img_id"] for b in batch]
        inputs = processor(images=images, return_tensors="pt")
        return inputs, sizes, img_ids
    return collate_fn


def score_batch(segments: list[torch.Tensor], landmark_ids: torch.Tensor) -> list[float]:
    return [torch.isin(seg, landmark_ids).sum().item() / seg.numel() for seg in segments]


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    landmark_ids = LANDMARK_SEGMENT_IDS.to(device)

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

    processor = AutoImageProcessor.from_pretrained(HF_CHECKPOINT_NAME)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(HF_CHECKPOINT_NAME).to(device)
    model.eval()
    model = torch.compile(model)

    dataset = ImageDataset(df, img_col="id", img_base_path=args.img_base_path, src_col="src")
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
    threshold = args.threshold

    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        for img_processed, target_sizes, img_ids in tqdm(loader, desc="Segmenting"):
            inputs = {k: v.to(device, non_blocking=True) for k, v in img_processed.items()}
            out = model(**inputs)
            segments = processor.post_process_semantic_segmentation(out, target_sizes=target_sizes)
            ratios = score_batch([s.to(device) for s in segments], landmark_ids)

            for img_id, ratio in zip(img_ids, ratios):
                results.append({
                    "id": img_id,
                    "ratio": ratio,
                    "threshold": threshold,
                    "is_landmark": ratio >= threshold,
                })

            if len(results) % args.checkpoint_every == 0:
                _save_to_path(results, already_done, args.checkpoint_path)

    _save_to_path(results, already_done, args.output_path)
    print(f"Done. Results saved to {args.output_path}")


def _save_to_path(new_results: list[dict], already_done: set, path: str) -> None:
    new_df = pl.DataFrame(new_results)
    if already_done and os.path.exists(path):
        new_df = pl.concat([pl.read_csv(path), new_df])
    new_df.write_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--img-base-path", type=str, default=BASE_IMG_PATH)
    parser.add_argument("--csv-path", type=str, default=CSV_PATH)
    parser.add_argument("--output-path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--checkpoint-path", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--src-col", type=str, default="src")
    args = parser.parse_args()

    main(args)
