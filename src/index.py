import polars as pl
import randomname
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data import GeoTIRDataset, warm_image_cache
from src.model.backbones import MODEL_NAMES, load_backbone
from src.utils import add_record_to_index, build_index, get_device, save_index


def ingest(args):
    device = get_device()
    bb = load_backbone(args.model, device, args.ckpt_path)

    df = pl.read_csv(args.data_path)
    if "category" not in df.columns:
        for alt in ("pred_label", "predicted_label"):
            if alt in df.columns:
                df = df.rename({alt: "category"})
                break

    dataset = GeoTIRDataset(
        df,
        base_img_path=args.img_base_path,
        processor=bb.processor,
        src_col=args.src_col,
        cache_dir=args.img_cache_dir,
        cache_size=args.cache_size,
    )
    if args.warm_cache_workers > 0:
        warm_image_cache(dataset, workers=args.warm_cache_workers)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    index = build_index(bb.image_dim, args.index_type)
    gps_index = build_index(bb.gps_dim, args.index_type) if bb.supports_two_step else None

    with torch.no_grad():
        for batch in tqdm(loader, desc="encode"):
            add_record_to_index(index, bb.encode_image(batch))
            if bb.supports_two_step:
                add_record_to_index(gps_index, bb.encode_gps(batch))

    target_dir = (
        f"index/{args.output_dir if args.output_dir else randomname.generate(sep='_')}"
    )
    save_index(index, df.to_dicts(), target_dir=target_dir)
    if bb.supports_two_step:
        save_index(gps_index, df.to_dicts(), target_dir=target_dir, prefix="gps")

    print(f"index and metadata saved successfully to {target_dir}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--ckpt-path")
    parser.add_argument("--img-base-path", default="../datasets/mp16-reason/images")
    parser.add_argument("--src-col", default="src")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--img-cache-dir",
        default=None,
        help="reuse the local downscaled-JPEG cache built during training "
        "(same dir + same --cache-size)",
    )
    parser.add_argument("--cache-size", type=int, default=256)
    parser.add_argument(
        "--warm-cache-workers",
        type=int,
        default=0,
        help="threads to pre-fill the cache from the NAS before encoding "
        "(0 = fill lazily during the pass)",
    )
    parser.add_argument("--index-type", default="flat_ip")
    parser.add_argument("--output-dir")

    args = parser.parse_args()

    ingest(args)
