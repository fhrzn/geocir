import polars as pl
import timm
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.mp16 import ImageDataset
from src.datasets.classify import DATASET_PATH_BUILDERS

TIMM_MODEL_NAME = "timm/convnext_xxlarge.clip_laion2b_soup_ft_in12k"


def run_zero_shot(args):
    device = "cuda"

    # Load timm model + preprocessing transforms
    model = timm.create_model(TIMM_MODEL_NAME, pretrained=True)
    model = model.to(device).eval()

    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)

    # Load ImageNet-12k label descriptions (11,821 classes)
    info = timm.data.ImageNetInfo("imagenet-12k")
    descriptions = info.label_descriptions()  # list[str], indexed by class id

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
        persistent_workers=True,
        prefetch_factor=args.num_workers // 2,
    )

    # Inference loop
    pred_labels = []

    with torch.no_grad(), torch.amp.autocast(device):
        for images, ids in tqdm(loader, desc="classify"):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=-1)
            top_scores, top_idx = probs.topk(1, dim=-1)
            top_scores = top_scores.squeeze(-1).cpu().tolist()
            top_idx = top_idx.squeeze(-1).cpu().tolist()

            for img_id, idx, score in zip(ids, top_idx, top_scores):
                pred_labels.append({
                    args.id_col: img_id,
                    "pred_label": descriptions[idx],
                    "pred_score": score,
                })

    # Join with original df and save
    df_pred = pl.DataFrame(pred_labels)
    df.join(df_pred, on=args.id_col).write_csv(args.output_path)
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    from src.datasets.classify import DATASET_PATH_BUILDERS

    parser = ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--img-base-path", default="../datasets/google-landmark/train-img")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_PATH_BUILDERS))
    parser.add_argument("--img-col", default="id")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--output-path", required=True)

    args = parser.parse_args()
    run_zero_shot(args)
