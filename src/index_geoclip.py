import polars as pl
import randomname
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from data.data import GeoTIRDataset
from src.geoclip import GeoCLIP
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
    geoclip_model = GeoCLIP().to(device)
    geoclip_model = geoclip_model.eval()
    geoclip_processor = geoclip_model.image_encoder.image_processor

    # dataset
    df = pl.read_csv(args.data_path)
    try:
        df = df.rename({"pred_label": "category"})
    except Exception:
        df = df.rename({"predicted_label": "category"})
    dataset = GeoTIRDataset(
        df,
        base_img_path=args.img_base_path,
        processor=geoclip_processor,
        src_col=args.src_col,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)

    # encode
    with torch.no_grad():
        for batch in tqdm(loader, desc="encode"):
            out = geoclip_model.image_encoder(batch["pixel_values"].to(device))
            out = out.cpu()
            out = F.normalize(out, dim=-1).numpy()
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
    parser.add_argument("--id-col", default="IMG_ID")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--index-size", type=int, default=768)
    parser.add_argument("--index-type", default="flat_ip")
    parser.add_argument("--src-col", default="folder")
    parser.add_argument("--output-dir")

    args = parser.parse_args()

    ingest(args)
