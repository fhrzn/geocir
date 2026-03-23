import os

import numpy as np
import polars as pl
import randomname
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

from src.utils import (
    add_record_to_index,
    build_index,
    get_device,
    save_index,
)


def ingest(args):
    # prepare
    index = build_index(args.index_size)
    device = get_device()
    pipe = pipeline(
        model="facebook/dinov2-base",
        device=device,
        pool=True,
        task="image-feature-extraction",
    )

    # dataset
    df = pl.read_csv(args.data_path)
    img_paths = [
        os.path.join(args.img_base_path, f"{id}.jpg") for id in df["id"].to_list()
    ]
    metadatas = df.to_dicts()

    # encode
    for i in tqdm(range(0, len(img_paths), args.batch_size), desc="batch encode"):
        paths = img_paths[i : i + args.batch_size]
        images = [Image.open(p).convert("RGB") for p in paths]
        embeddings = pipe(images, batch_size=args.batch_size)
        embeddings = np.array(embeddings, dtype=np.float32).squeeze(1)
        add_record_to_index(index, embeddings)

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
