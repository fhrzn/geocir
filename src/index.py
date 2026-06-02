import polars as pl
import randomname
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from src.data.data import GeoTIRDataset
from src.model.g3 import G3
from src.model.geoclip import GeoCLIP
from src.model.geotir.model import GeoTIRModel
from src.utils import add_record_to_index, build_index, get_device, save_index

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


def _setup_clip(args, device):
    model = AutoModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    model = torch.compile(model)
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)

    def encode(batch):
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            out = model.get_image_features(batch["pixel_values"].to(device))
        return F.normalize(out.pooler_output.float(), dim=-1).cpu().numpy()

    return processor, encode


def _setup_g3(args, device):
    model = G3().to(device).eval()
    processor = model._processor

    def encode(batch):
        img_emb = model.vision_proj(model.vision_model(batch["pixel_values"].to(device)).pooler_output)
        img_emb_n = F.normalize(img_emb, dim=-1)
        img2txt_n = F.normalize(model.img2txt_proj(img_emb), dim=-1)
        img2loc_n = F.normalize(model.img2loc_proj(img_emb), dim=-1)
        out = F.normalize(torch.cat([img_emb_n, img2txt_n, img2loc_n], dim=1), dim=-1)
        return out.cpu().numpy()

    return processor, encode


def _setup_geoclip(args, device):
    model = GeoCLIP().to(device).eval()
    processor = model.image_encoder.image_processor

    def encode(batch):
        out = model.image_encoder(batch["pixel_values"].to(device))
        return F.normalize(out, dim=-1).cpu().numpy()

    return processor, encode


def _setup_geotir(args, device):
    if not args.ckpt_path:
        raise ValueError("--ckpt-path is required for --model geotir")
    model = GeoTIRModel(clip_model_name=CLIP_MODEL_NAME).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.eval()
    model = torch.compile(model)
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)

    def encode(batch):
        with torch.autocast(device, dtype=torch.bfloat16):
            out = model.encode_images(pixel_values=batch["pixel_values"].to(device))
        return out.cpu().float().numpy()

    return processor, encode


_MODEL_REGISTRY = {
    "clip": _setup_clip,
    "g3": _setup_g3,
    "geoclip": _setup_geoclip,
    "geotir": _setup_geotir,
}

_INDEX_SIZES = {
    "clip": 768,
    "g3": 2304,  # 3 × 768: vision_proj + img2txt_proj + img2loc_proj
    "geoclip": 512,
    "geotir": 768,
}


def ingest(args):
    device = get_device()
    processor, encode_fn = _MODEL_REGISTRY[args.model](args, device)

    df = pl.read_csv(args.data_path)
    if "category" not in df.columns:
        try:
            df = df.rename({"pred_label": "category"})
        except Exception:
            df = df.rename({"predicted_label": "category"})

    dataset = GeoTIRDataset(df, base_img_path=args.img_base_path, processor=processor, src_col=args.src_col)
    loader = DataLoader(dataset, batch_size=args.batch_size)
    index = build_index(_INDEX_SIZES[args.model], args.index_type)

    with torch.no_grad():
        for batch in tqdm(loader, desc="encode"):
            embeddings = encode_fn(batch)
            add_record_to_index(index, embeddings)

    target_dir = f"index/{args.output_dir if args.output_dir else randomname.generate(sep='_')}"
    save_index(index, df.to_dicts(), target_dir=target_dir)
    print(f"index and metadata saved successfully to {target_dir}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(_MODEL_REGISTRY))
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--ckpt-path")
    parser.add_argument("--img-base-path", default="../datasets/mp16-reason/images")
    parser.add_argument("--src-col", default="folder")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--index-type", default="flat_ip")
    parser.add_argument("--output-dir")

    args = parser.parse_args()

    ingest(args)
