import polars as pl
import randomname
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from data.data import GeoTIRDataset
from src.geotir.model import GeoTIRModel
from src.utils import (
    add_record_to_index,
    build_index,
    get_device,
    save_index,
)

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


def ingest(args):
    # prepare
    index = build_index(args.index_size, args.index_type)

    device = get_device()
    model = GeoTIRModel(clip_model_name=CLIP_MODEL_NAME).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.eval()
    model = torch.compile(model)

    processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)

    # dataset
    df = pl.read_csv(args.data_path)
    try:
        df = df.rename({"pred_label": "category"})
    except Exception:
        df = df.rename({"predicted_label": "category"})
    dataset = GeoTIRDataset(
        df=df, base_img_path=args.img_base_path, processor=processor, src_col=args.src_col
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
    )

    # encode
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        for batch in tqdm(loader, desc="encode"):
            output = model.encode_images(pixel_values=batch["pixel_values"].to(device)).cpu().float().numpy()
            add_record_to_index(index, output)

    metadatas = df.to_dicts()

    # save
    target_dir = args.output_dir if args.output_dir else randomname.generate(sep="_")
    save_index(index, metadatas, target_dir=target_dir)

    print(f"index and metadata saved successfully to {target_dir}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--img-base-path", default="../datasets/mp16-reason/images")
    parser.add_argument("--img-col", default="IMG_ID")
    parser.add_argument("--id-col", default="IMG_ID")
    parser.add_argument("--src-col", default="folder")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--index-size", type=int, default=768)
    parser.add_argument("--index-type", default="flat_ip")
    parser.add_argument("--output-dir")

    args = parser.parse_args()

    ingest(args)
