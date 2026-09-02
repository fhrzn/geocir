import os
import random
from concurrent.futures import ThreadPoolExecutor

import polars as pl
import torch
from PIL import Image
from torch.utils.data import BatchSampler, Dataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, CLIPProcessor

from src.data.countries import display_country


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
        img_shape = img.size[::-1]
        if self.processor is not None:
            img = self.processor(img, return_tensors="pt")
        return self.img_ids[index], img_shape, img


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
        caption_col: str = "caption",
        use_template: bool = True,
        max_text_length: int = 77,
        src_col: str = "src",
        cache_dir: str | None = None,
        cache_size: int = 256,
        cell_weights: dict | None = None,
    ):
        super().__init__()
        self.df = df.to_dicts()
        self.base_img_path = base_img_path
        self.image_processor = processor.image_processor
        self.img_col = img_col
        self.id_col = id_col
        self.caption_col = caption_col
        self.use_template = use_template
        self.src_col = src_col
        # local NVMe cache of downscaled JPEGs -> avoids re-reading the NAS every epoch
        self.cache_dir = cache_dir
        self.cache_size = cache_size

        # integer id per (category, country) cell -> in-batch positive mask
        cells = sorted({(r["category"], r["country"]) for r in self.df})
        self.cell_to_id = {c: i for i, c in enumerate(cells)}
        self.cell_ids = torch.tensor(
            [self.cell_to_id[(r["category"], r["country"])] for r in self.df]
        )
        # optional per-sample loss weight (class-balanced inverse cell frequency)
        if cell_weights is not None:
            self.weights = torch.tensor(
                [float(cell_weights.get((r["category"], r["country"]), 1.0)) for r in self.df],
                dtype=torch.float32,
            )
        else:
            self.weights = None

        # tokenize text ONCE: dedupe -> tokenize unique -> scatter back per row
        texts = [self._caption(r) for r in self.df]
        uniq = sorted(set(texts))
        tok = processor.tokenizer(
            uniq,
            max_length=max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        pos = {t: i for i, t in enumerate(uniq)}
        gather = torch.tensor([pos[t] for t in texts])
        self.input_ids = tok["input_ids"][gather].contiguous()
        self.attention_mask = tok["attention_mask"][gather].contiguous()

    def _caption(self, row: dict) -> str:
        if self.use_template:
            # oracle text; must match the eval query template exactly, including the
            # natural country phrasing used in the query JSON (COUNTRY_DISPLAY).
            return f"a {row['category']} located in {display_country(row['country'])}"
        return row[self.caption_col]

    def _rel_path(self, row: dict) -> str:
        name = str(row[self.img_col])
        name = name if name.endswith(".jpg") else f"{name}.jpg"
        return os.path.join(row[self.src_col], name)

    def ensure_cached(self, index: int) -> None:
        """Populate the local cache entry for one row (used by warm_image_cache)."""
        if self.cache_dir is None:
            return
        rel = self._rel_path(self.df[index])
        dst = os.path.join(self.cache_dir, rel)
        if os.path.exists(dst):
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        src = os.path.join(self.base_img_path, rel)
        with Image.open(src) as im:
            im.draft("RGB", (self.cache_size, self.cache_size))
            im = im.convert("RGB")
            im.thumbnail((self.cache_size, self.cache_size), Image.BILINEAR)
            tmp = f"{dst}.{os.getpid()}.tmp"
            im.save(tmp, format="JPEG", quality=90)
        os.replace(tmp, dst)

    def _load_image(self, row: dict) -> Image.Image:
        rel = self._rel_path(row)
        if self.cache_dir is not None:
            dst = os.path.join(self.cache_dir, rel)
            if os.path.exists(dst):
                with Image.open(dst) as im:
                    return im.convert("RGB")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with Image.open(os.path.join(self.base_img_path, rel)) as im:
                im.draft("RGB", (self.cache_size, self.cache_size))
                im = im.convert("RGB")
                im.thumbnail((self.cache_size, self.cache_size), Image.BILINEAR)
                tmp = f"{dst}.{os.getpid()}.tmp"
                im.save(tmp, format="JPEG", quality=90)
            os.replace(tmp, dst)
            return im
        with Image.open(os.path.join(self.base_img_path, rel)) as im:
            im.draft("RGB", (self.cache_size, self.cache_size))
            return im.convert("RGB")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df[index]
        img = self._load_image(row)
        pixel_values = self.image_processor(images=img, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        item = {
            "category": row["category"],
            "country": row["country"],
            "pixel_values": pixel_values,
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "cell_id": self.cell_ids[index],
        }
        if self.weights is not None:
            item["weight"] = self.weights[index]
        return item


def warm_image_cache(dataset: "GeoTIRDataset", workers: int = 32) -> None:
    """Pre-fill the local cache from the NAS in parallel before training starts."""
    if dataset.cache_dir is None:
        return
    n = len(dataset)
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = ex.map(_safe_ensure_cached, (dataset,) * n, range(n))
        for ok in tqdm(futs, total=n, desc="warm cache", unit="img"):
            errors += not ok
    print(f"cache warm done: {n - errors}/{n} cached ({errors} errors) -> {dataset.cache_dir}")


def _safe_ensure_cached(dataset: "GeoTIRDataset", index: int) -> bool:
    try:
        dataset.ensure_cached(index)
        return True
    except Exception:  # noqa: BLE001 - a few unreadable images shouldn't abort the warm-up
        return False


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


class PKBatchSampler(BatchSampler):
    """P cells x K images per batch (batch_size = P * K).

    Every anchor is guaranteed K-1 same-cell positives. Each epoch a cell
    contributes ~ ceil(min(size, cap) / K) draws, so head cells are down-
    weighted to `cap` and every cell with >= 2 images is seen at least once.
    Cells with fewer than K images pad by sampling with replacement.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        batch_size: int,
        images_per_cell: int = 4,
        cap: int | None = None,
        drop_last: bool = True,
        seed: int = 0,
    ):
        assert batch_size % images_per_cell == 0, "batch_size must be divisible by images_per_cell"
        self.batch_size = batch_size
        self.k = images_per_cell
        self.p = batch_size // images_per_cell
        self.cap = cap
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        grouped = (
            df.with_row_index("_idx")
            .group_by(["category", "country"])
            .agg(pl.col("_idx").alias("indices"))
        )
        self.cells = [
            list(row["indices"])
            for row in grouped.iter_rows(named=True)
            if len(row["indices"]) >= 2
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _draws(self, size: int) -> int:
        take = size if self.cap is None else min(size, self.cap)
        return max(1, take // self.k)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools = [rng.sample(c, len(c)) for c in self.cells]
        cursors = [0] * len(self.cells)

        slots = []
        for ci, c in enumerate(self.cells):
            slots += [ci] * self._draws(len(c))
        rng.shuffle(slots)

        batch = []
        for ci in slots:
            pool, cur = pools[ci], cursors[ci]
            if cur + self.k > len(pool):
                rng.shuffle(pool)
                cur = 0
            chosen = pool[cur : cur + self.k]
            cursors[ci] = cur + self.k
            if len(chosen) < self.k:  # tiny cell: keep distinct, pad with replacement
                chosen = (chosen + [rng.choice(pool) for _ in range(self.k)])[: self.k]
            batch += chosen
            if len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size :]
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        return sum(self._draws(len(c)) for c in self.cells) // self.p


def geo_collate_fn(batch: list[dict]) -> dict:
    import torch
    out = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "cell_id": torch.stack([b["cell_id"] for b in batch]),
        "category": [b["category"] for b in batch],
        "country": [b["country"] for b in batch],
    }
    if "weight" in batch[0]:
        out["weight"] = torch.stack([b["weight"] for b in batch])
    return out
