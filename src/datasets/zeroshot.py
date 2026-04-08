import json

import polars as pl
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.mp16 import ImageDataset
from src.datasets.classify import DATASET_PATH_BUILDERS, batch_agg_score

MODEL_NAME = "hf-hub:timm/ViT-SO400M-14-SigLIP2"


def build_text_embeddings(
    model, tokenizer, desc_flatten: list[str], device: str
) -> torch.Tensor:
    """Pre-compute normalized text embeddings for all label descriptions."""
    text = tokenizer(desc_flatten, context_length=model.context_length).to(device)

    with torch.no_grad(), torch.amp.autocast(device), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        text_embs = model.encode_text(text, normalize=True)

    return text_embs  # [N_desc, D]


def run_zero_shot(args):
    device = "cuda"

    # Load model + preprocessor
    model, preprocess = create_model_from_pretrained(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)

    model = model.to(device).eval()
    model = torch.compile(model)

    # Load labels
    labels = json.loads(open(args.label_path).read())
    categories = list(labels.keys())
    desc_flatten = [d for descs in labels.values() for d in descs]
    desc_to_cat = {d: c for c, descs in labels.items() for d in descs}

    # Pre-compute text embeddings once
    print("Computing text embeddings...")
    text_embs = build_text_embeddings(model, tokenizer, desc_flatten, device)

    # Build dataset & dataloader
    df = pl.read_csv(args.data_path)
    img_ids = df[args.id_col].to_list()
    img_paths = DATASET_PATH_BUILDERS[args.dataset](df, args.img_col, args.img_base_path)

    dataset = ImageDataset(preprocess, img_paths=img_paths, img_ids=img_ids)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Inference loop
    all_outputs: list[list[dict]] = []
    all_img_ids: list[str] = []

    with torch.no_grad(), torch.amp.autocast(device), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        for images, ids in tqdm(loader, desc="classify"):
            img_embs = model.encode_image(images.to(device), normalize=True)
            scores = torch.sigmoid(
                img_embs @ text_embs.T * model.logit_scale.exp() + model.logit_bias
            )

            for row in scores.cpu().float().tolist():
                all_outputs.append([
                    {"label": desc, "score": score}
                    for desc, score in zip(desc_flatten, row)
                ])
            all_img_ids.extend(ids)

    # Aggregate per-category scores and pick top class
    pred_labels = batch_agg_score(all_outputs, categories, desc_to_cat, all_img_ids, args.id_col)

    # Join with original df and save
    df_pred = pl.DataFrame(pred_labels)
    df.join(df_pred, on=args.id_col).write_csv(args.output_path)
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    from src.datasets.classify import DATASET_PATH_BUILDERS

    parser = ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--label-path", default="./notebooks/labels.json")
    parser.add_argument("--img-base-path", default="../datasets/google-landmark/train-img")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_PATH_BUILDERS))
    parser.add_argument("--img-col", default="id")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--output-path", required=True)

    args = parser.parse_args()
    run_zero_shot(args)
