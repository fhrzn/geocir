import polars as pl
import randomname
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data import GeoTIRDataset
from src.g3 import G3
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
    g3_model = G3().to(device)
    g3_model = g3_model.eval()
    g3_processor = g3_model._processor

    # dataset
    df = pl.read_csv(args.data_path)
    try:
        df = df.rename({"pred_label": "category"})
    except Exception:
        df = df.rename({"predicted_label": "category"})
    dataset = GeoTIRDataset(
        df,
        base_img_path=args.img_base_path,
        processor=g3_processor,
        src_col=args.src_col,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)

    # encode
    with torch.no_grad():
        for batch in tqdm(loader, desc="encode"):
            img_emb = g3_model.vision_proj(
                g3_model.vision_model(batch["pixel_values"].to(device)).pooler_output
            )
            img_emb_norm = F.normalize(img_emb, dim=-1)

            img2txt_emb = g3_model.img2txt_proj(img_emb)
            img2txt_emb_norm = F.normalize(img2txt_emb, dim=-1)

            img2loc_emb = g3_model.img2loc_proj(img_emb)
            img2loc_emb_norm = F.normalize(img2loc_emb, dim=-1)

            out = torch.cat([img_emb_norm, img2txt_emb_norm, img2loc_emb_norm], dim=1)
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
