import os
import random

import polars as pl
from PIL import Image
from torch.utils.data import BatchSampler, Dataset
from transformers import AutoImageProcessor, CLIPProcessor


class ImageDataset(Dataset):
    def __init__(
        self,
        processor=None,
        img_paths: list[str] = None,
        img_ids: list[str] = None,
    ):
        self.processor = processor
        self.img_paths = img_paths
        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int):
        img = Image.open(self.img_paths[index]).convert("RGB")
        if self.processor is not None:
            img = self.processor(img)
        return img, self.img_ids[index]


class MP16Dataset(Dataset):
    def __init__(
        self,
        df,
        img_col: str = "IMG_ID",
        id_col: str = "IMG_ID",
        img_base_path: str = "",
        model_name: str = "",
    ):
        self.df = df
        self.img_col = img_col
        self.id_col = id_col
        self.img_base_path = img_base_path
        self.processor = (
            AutoImageProcessor.from_pretrained(model_name) if model_name else None
        )

        print(self.img_col, self.id_col)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df[index]
        path = row[self.img_col].item()
        path = path if ".jpg" in path else f"{path}.jpg"
        path = os.path.join(self.img_base_path, path)
        img_id = row[self.id_col].item()

        img = Image.open(path).convert("RGB")

        if self.processor:
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            return inputs, img_id

        return {"image": img, "size": img.size[::-1]}, img_id


class GeoTIRDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        base_img_path: str,
        processor: CLIPProcessor,
        img_col: str = "id",
        id_col: str = "id",
    ):
        super().__init__()
        self.df = df.to_dicts()
        self.base_img_path = base_img_path
        self.processor = processor
        self.img_col = img_col
        self.id_col = id_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df[index]
        path = row[self.img_col]
        path = path if ".jpg" in path else f"{path}.jpg"
        path = os.path.join(self.base_img_path, path)

        img = Image.open(path).convert("RGB")
        caption = f"A {row['category']} landmark located in {row['country']}"

        processed = self.processor(
            images=img,
            text=caption,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "category": row["category"],
            "country": row["country"],
            **{k: v.squeeze(0) for k, v in processed.items()},
        }


class PairAwareBatchSampler(BatchSampler):
    """
    Stores group -> indices mapping (O(N) memory).
    Pairs are interleaved across groups each epoch to maximise batch diversity.
    Each batch contains batch_size//2 positive pairs.

    cap: max pairs taken per group per epoch (caps large groups so they don't
         dominate; set to the 75th-percentile group size // 2 by default).
    """

    def __init__(
        self,
        df: pl.DataFrame,
        batch_size: int,
        drop_last: bool = True,
        cap: int = 1157,
    ):
        assert batch_size % 2 == 0, "batch_size must be even"
        self.batch_size = batch_size
        self.pairs_per_batch = batch_size // 2
        self.drop_last = drop_last
        self.cap = cap

        grouped = (
            df.with_row_index("_idx")
            .group_by(["category", "country"])
            .agg(pl.col("_idx").alias("indices"))
        )
        self.groups = [
            row["indices"]
            for row in grouped.iter_rows(named=True)
            if len(row["indices"]) >= 2
        ]

    def __iter__(self):
        # Build per-group pair streams, shuffled and capped
        pair_streams = []
        for group in self.groups:
            shuffled = random.sample(group, len(group))
            stream = [
                (shuffled[k], shuffled[k + 1])
                for k in range(0, len(shuffled) - 1, 2)
            ]
            pair_streams.append(stream[: self.cap])

        random.shuffle(pair_streams)

        # Interleave streams so batches get diverse groups
        interleaved = []
        max_len = max(len(s) for s in pair_streams)
        for i in range(max_len):
            for stream in pair_streams:
                if i < len(stream):
                    interleaved.append(stream[i])

        batch = []
        for i, j in interleaved:
            batch += [i, j]
            if len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size :]

        if not self.drop_last and batch:
            yield batch

    def __len__(self):
        total_pairs = sum(min(len(g) // 2, self.cap) for g in self.groups)
        return total_pairs // self.pairs_per_batch


def geo_collate_fn(batch: list[dict]) -> dict:
    import torch
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "category": [b["category"] for b in batch],
        "country": [b["country"] for b in batch],
    }
