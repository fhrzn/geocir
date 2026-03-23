from functools import partial

import polars as pl
import randomname
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.mp16 import MP16Dataset
from src.geoclip import GeoCLIP
from src.utils import (
    add_record_to_index,
    build_index,
    clip_collate_fn,
    get_device,
    save_index,
)


def ingest(args):
    # prepare
    index = build_index(args.index_size)

    device = get_device()
    geoclip_model = GeoCLIP().to(device)
    geoclip_model = geoclip_model.eval()
    geoclip_processor = geoclip_model.image_encoder.image_processor

    # dataset
    df = pl.read_csv(args.data_path)
    dataset = MP16Dataset(df, img_col=args.img_col, img_base_path=args.img_base_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=partial(clip_collate_fn, geoclip_processor),
    )

    # encode
    with torch.no_grad():
        for batch in tqdm(loader, desc="encode"):
            out = geoclip_model.image_encoder(batch["pixel_values"].to(device))
            out = out.cpu().numpy()
            add_record_to_index(index, out)

    metadatas = df.to_dicts()

    # save
    target_dir = args.output_dir if args.output_dir else randomname.generate(sep="_")
    save_index(index, metadatas, target_dir=target_dir)

    print(f"index and metadata saved successfully to {target_dir}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--img-base-path", default="../datasets/mp16-reason/images")
    parser.add_argument("--img-col", default="IMG_ID")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--index-size", type=int, default=768)
    parser.add_argument("--output-dir")

    args = parser.parse_args()

    ingest(args)
