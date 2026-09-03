"""Pre-fill the local downscaled-JPEG cache from the NAS, without training.

Caches the exact same files (same relative paths, same downscale) that
train.py / index.py read through GeoTIRDataset, so a later run starts hot.
Works for any CSV that has an id column and a src (subdir) column -- the
train/val/test splits and the retrieval index CSV alike.

Example:
    python -u -m src.model.geotir.warm_cache \
        --data /mnt/yokoyamalab-nas/gldv2-full/csv/train-test-split/train_top5_countries.csv \
        --val-data /mnt/yokoyamalab-nas/gldv2-full/csv/train-test-split/val_top5_countries.csv \
        --index-data /mnt/yokoyamalab-nas/gldv2-full/csv/raw_train_index.csv \
        --img-path /mnt/yokoyamalab-nas/gldv2-full \
        --img-cache-dir /home/affahrizain/gldv2-img-cache \
        --cache-size 256 \
        --workers 48
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from PIL import Image
from tqdm.auto import tqdm


def _rel_path(img_id: str, src: str) -> str:
    name = str(img_id)
    name = name if name.endswith(".jpg") else f"{name}.jpg"
    return os.path.join(str(src), name)


def _cache_one(rel: str, base_img_path: str, cache_dir: str, cache_size: int) -> bool:
    """Mirror GeoTIRDataset.ensure_cached for a single relative path."""
    dst = os.path.join(cache_dir, rel)
    if os.path.exists(dst):
        return True
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with Image.open(os.path.join(base_img_path, rel)) as im:
            im.draft("RGB", (cache_size, cache_size))
            im = im.convert("RGB")
            im.thumbnail((cache_size, cache_size), Image.BILINEAR)
            tmp = f"{dst}.{os.getpid()}.tmp"
            im.save(tmp, format="JPEG", quality=90)
        os.replace(tmp, dst)
        return True
    except Exception:  # noqa: BLE001 - a few unreadable images shouldn't abort the run
        return False


def warm_csv(
    path: str,
    base_img_path: str,
    cache_dir: str,
    cache_size: int,
    workers: int,
    id_col: str,
    src_col: str,
) -> None:
    df = pl.read_csv(path)
    rels = [
        _rel_path(i, s)
        for i, s in zip(df[id_col].to_list(), df[src_col].to_list())
    ]
    n = len(rels)
    print(f"[{os.path.basename(path)}] {n:,} images")
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = ex.map(
            _cache_one,
            rels,
            (base_img_path,) * n,
            (cache_dir,) * n,
            (cache_size,) * n,
        )
        for ok in tqdm(futs, total=n, desc="warm cache", unit="img"):
            errors += not ok
    print(f"  -> {n - errors:,}/{n:,} cached ({errors} errors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="train split CSV")
    parser.add_argument("--val-data", help="val split CSV")
    parser.add_argument("--test-data", help="test split CSV")
    parser.add_argument("--index-data", help="retrieval index CSV")
    parser.add_argument(
        "--csv",
        nargs="+",
        default=[],
        help="any additional CSVs to cache",
    )
    parser.add_argument("--img-path", required=True, help="base dir the src paths are relative to")
    parser.add_argument("--img-cache-dir", required=True, help="local dir for downscaled JPEGs")
    parser.add_argument("--cache-size", type=int, default=256, help="longer-side pixel size")
    parser.add_argument("--workers", type=int, default=48, help="parallel NAS-reading threads")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--src-col", default="src")
    args = parser.parse_args()

    paths = [
        p
        for p in (args.data, args.val_data, args.test_data, args.index_data, *args.csv)
        if p
    ]
    if not paths:
        parser.error("pass at least one of --data / --val-data / --test-data / --index-data / --csv")

    for path in paths:
        warm_csv(
            path,
            base_img_path=args.img_path,
            cache_dir=args.img_cache_dir,
            cache_size=args.cache_size,
            workers=args.workers,
            id_col=args.id_col,
            src_col=args.src_col,
        )


if __name__ == "__main__":
    main()
