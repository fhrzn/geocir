import json
import os

import numpy as np
import polars as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline, Siglip2Tokenizer

from src.datasets.mp16 import ImageDataset
from src.utils import add_record_to_index, build_index, get_device, save_index


def collate_fn(batch):
    images, ids = zip(*batch)
    return list(images), list(ids)


TASK_DEFAULTS = {
    "zero-shot-image-classification": "google/siglip2-so400m-patch14-384",
    "image-feature-extraction": "facebook/dinov2-base",
}


def build_img_paths_gld(df, img_col, img_base_path):
    return [os.path.join(img_base_path, f"{id}.jpg") for id in df[img_col].to_list()]


def build_img_paths_mp16(df, img_col, img_base_path):
    return [os.path.join(img_base_path, id) for id in df[img_col].to_list()]


DATASET_PATH_BUILDERS = {
    "gld": build_img_paths_gld,
    "mp16": build_img_paths_mp16,
}


def batch_agg_score(
    output: list[list[dict]],
    categories: list[str],
    desc_to_cat: dict[str, str],
    img_ids: list[str],
    id_col: str,
):
    """
    Aggregate per-description scores into per-category scores by taking the max,
    then return the top predicted category + confidence.
    """
    result = []
    for img_id, clip_item in zip(img_ids, output):
        cat_scores: dict[str, float] = {cat: 0.0 for cat in categories}

        for pred in clip_item:
            cat = desc_to_cat[pred["label"]]
            cat_scores[cat] = max(cat_scores[cat], pred["score"])

        sorted_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_score = sorted_cats[0]

        result.append({id_col: img_id, "category": top_cat, "confidence": top_score})

    return result


def run_zero_shot_classification(pipe, img_paths, img_ids, args, df):
    labels = json.loads(open(args.label_path, "r").read())
    categories = list(labels.keys())
    desc_flatten = [d for desc in labels.values() for d in desc]
    desc_to_cat = {d: c for c, desc in labels.items() for d in desc}

    dataset = ImageDataset(img_paths=img_paths, img_ids=img_ids)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, collate_fn=collate_fn)

    pred_labels = []
    for images, ids in tqdm(loader, desc="classify"):
        output = pipe(images, batch_size=args.batch_size, candidate_labels=desc_flatten)
        pred_labels.extend(batch_agg_score(output, categories, desc_to_cat, ids, args.id_col))

    df_pred = pl.DataFrame(pred_labels)
    # backup
    df_pred.write_csv("/home/affahrizain/project/datasets/siglip_predict.csv")
    df.join(df_pred, on=pl.col(args.id_col)).write_csv(args.output_path)
    print(f"Saved to: {args.output_path}")


def run_feature_extraction(pipe, img_paths, img_ids, args, df):
    index = build_index(args.index_size)
    metadatas = df.to_dicts()

    dataset = ImageDataset(img_paths=img_paths, img_ids=img_ids)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, collate_fn=collate_fn)

    for images, _ in tqdm(loader, desc="extract"):
        embeddings = pipe(images, batch_size=args.batch_size)
        embeddings = np.array(embeddings, dtype=np.float32).squeeze(1)
        add_record_to_index(index, embeddings)

    save_index(index, metadatas, target_dir=args.output_path)
    print(f"Index and metadata saved to: {args.output_path}")


def main(args):
    device = get_device()

    if args.task not in TASK_DEFAULTS:
        raise ValueError(
            f"Unsupported task '{args.task}'. Choose from: {list(TASK_DEFAULTS)}"
        )

    df = pl.read_csv(args.data_path)
    img_ids = df[args.id_col].to_list()
    img_paths = DATASET_PATH_BUILDERS[args.dataset](df, args.img_col, args.img_base_path)

    model = args.model_name or TASK_DEFAULTS[args.task]
    pipe_kwargs = dict(model=model, task=args.task, device=device)

    if args.task == "zero-shot-image-classification":
        pipe_kwargs["tokenizer"] = Siglip2Tokenizer.from_pretrained(model)
        pipe = pipeline(**pipe_kwargs)
        run_zero_shot_classification(pipe, img_paths, img_ids, args, df)
    elif args.task == "image-feature-extraction":
        pipe_kwargs["pool"] = True
        pipe = pipeline(**pipe_kwargs)
        run_feature_extraction(pipe, img_paths, img_ids, args, df)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--data-path", required=True)
    parser.add_argument("--label-path", default="./notebooks/labels.json")
    parser.add_argument("--img-base-path", default="../datasets/google-landmark/train-img")
    parser.add_argument(
        "--task", default="zero-shot-image-classification", choices=list(TASK_DEFAULTS)
    )
    parser.add_argument(
        "--model-name", default=None, help="Override default model for the chosen task"
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASET_PATH_BUILDERS))
    parser.add_argument("--img-col", default="id")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--index-size", type=int, default=768)
    parser.add_argument("--output-path", required=True)

    args = parser.parse_args()

    main(args)
