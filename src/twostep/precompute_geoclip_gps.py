"""Precompute GeoCLIP top-1 GPS prediction (+ reverse geocode) for every DB record.

GeoCLIP predicts location from the image alone, so this table is independent of the
Stage-1 retrieval backbone: build it once over ``test.csv`` and reuse it for every
CLIP / SigLIP / BLIP / BLIP-2 two-step run. See ``src/twostep/clip_geoclip.py``.
"""

import os
from argparse import ArgumentParser

import numpy as np
import polars as pl
import reverse_geocoder as rg
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.model.backbones import load_backbone
from src.utils import get_device


class _ImageDataset(Dataset):
    def __init__(self, rows, img_base_path, image_processor, src_col, image_ext,
                 cache_dir=None, cache_size=256):
        self.rows = rows
        self.base = img_base_path
        self.image_processor = image_processor
        self.src_col = src_col
        self.ext = image_ext
        self.cache_dir = cache_dir
        self.cache_size = cache_size

    def __len__(self):
        return len(self.rows)

    def _rel(self, row):
        name = str(row["id"])
        return os.path.join(str(row[self.src_col]), name if name.endswith(self.ext) else name + self.ext)

    def _load(self, row):
        rel = self._rel(row)
        if self.cache_dir:
            dst = os.path.join(self.cache_dir, rel)
            if os.path.exists(dst):
                with Image.open(dst) as im:
                    return im.convert("RGB")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with Image.open(os.path.join(self.base, rel)) as im:
                im.draft("RGB", (self.cache_size, self.cache_size))
                im = im.convert("RGB")
                im.thumbnail((self.cache_size, self.cache_size), Image.BILINEAR)
                tmp = f"{dst}.{os.getpid()}.tmp"
                im.save(tmp, format="JPEG", quality=90)
            os.replace(tmp, dst)
            return im
        with Image.open(os.path.join(self.base, rel)) as im:
            im.draft("RGB", (self.cache_size, self.cache_size))
            return im.convert("RGB")

    def __getitem__(self, i):
        try:
            px = self.image_processor(
                images=self._load(self.rows[i]), return_tensors="pt"
            )["pixel_values"].squeeze(0)
            ok = True
        except (OSError, ValueError, KeyError):
            px = torch.zeros(3, 224, 224)
            ok = False
        return {"idx": i, "pixel_values": px, "ok": ok}


@torch.no_grad()
def _predict(model, loader, device, n):
    gallery = model.gps_gallery.to(device)
    loc_feat = F.normalize(model.location_encoder(gallery), dim=1)
    logit_scale = model.logit_scale.exp().to(device)
    coords_gallery = model.gps_gallery.cpu().numpy()

    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    prob = np.full(n, np.nan, dtype="float32")
    ok_mask = np.zeros(n, dtype=bool)

    for batch in tqdm(loader, desc="geoclip predict"):
        ok = batch["ok"].numpy().astype(bool)
        if not ok.any():
            continue
        idx = batch["idx"].numpy()[ok]
        feats = F.normalize(model.image_encoder(batch["pixel_values"].to(device)), dim=1)
        p = (logit_scale * feats @ loc_feat.t()).softmax(dim=-1)
        top = p.max(dim=-1)
        gi = top.indices.cpu().numpy()[ok]
        lat[idx] = coords_gallery[gi, 0]
        lon[idx] = coords_gallery[gi, 1]
        prob[idx] = top.values.cpu().numpy()[ok]
        ok_mask[idx] = True

    return lat, lon, prob, ok_mask


def main(args):
    device = args.device or get_device()

    df = pl.read_csv(args.data_path)
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dicts()
    n = len(rows)

    model = load_backbone("geoclip", device).model
    ds = _ImageDataset(
        rows, args.img_base_path, model.image_encoder.image_processor,
        args.src_col, args.image_ext, args.img_cache_dir, args.cache_size,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    lat, lon, prob, ok_mask = _predict(model, loader, device, n)

    cc = np.full(n, None, dtype=object)
    admin1 = np.full(n, None, dtype=object)
    place = np.full(n, None, dtype=object)
    valid = np.where(ok_mask)[0]
    if len(valid):
        hits = rg.search([(float(lat[i]), float(lon[i])) for i in valid])
        for i, h in zip(valid, hits):
            cc[i], admin1[i], place[i] = h["cc"], h["admin1"], h["name"]

    out = pl.DataFrame({
        "id": [str(r["id"]) for r in rows],
        "lat_pred": lat,
        "lon_pred": lon,
        "prob": prob,
        "cc_pred": cc.tolist(),
        "admin1_pred": admin1.tolist(),
        "place_pred": place.tolist(),
    })
    for src, dst in (("latitude", "lat_true"), ("longitude", "lon_true"), ("country_code", "cc_true")):
        if src in df.columns:
            out = out.with_columns(df.get_column(src).alias(dst))

    print(f"{n} records, {n - int(ok_mask.sum())} image-load failures")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    (out.write_json if args.output.endswith(".json") else out.write_parquet)(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--img-base-path", required=True)
    parser.add_argument("--output", required=True, help=".parquet (recommended) or .json")
    parser.add_argument("--src-col", default="src")
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--img-cache-dir", default=None)
    parser.add_argument("--cache-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)

    main(parser.parse_args())
