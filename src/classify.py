import json
import os
from PIL import Image
import polars as pl
from tqdm import tqdm
from transformers import pipeline

from src.utils import get_device


def batch_agg_score(
    output: list[list[dict]],
    categories: list[str],
    desc_to_cat: dict[str, str],
    img_ids: list[str],
):
    """
    The pipeline returns one score per description string.
    Aggregate by summing scores for all descriptions belonging to the same
    category, then return top predicted category + top-3 breakdown.
    """
    result = []
    for img_id, clip_item in zip(img_ids, output):
        cat_scores: dict[str, float] = {cat: 0.0 for cat in categories}
        cat_counts: dict[str, int]   = {cat: 0    for cat in categories}

        for pred in clip_item:
            cat = desc_to_cat[pred["label"]]
            cat_scores[cat] = max(cat_scores[cat], pred["score"])
            cat_counts[cat] += 1

        sorted_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_score = sorted_cats[0]

        result.append({"id": img_id, "category": top_cat, "confidence": top_score})

    return result

def main(args):
    # prepare
    device = get_device()
    pipe = pipeline(
        model="google/siglip2-so400m-patch16-naflex",
        task="zero-shot-image-classification",
        device=device,
    )

    # dataset
    labels = json.loads(open(args.label_path, "r").read())
    categories = list(labels.keys())
    desc_flatten = [d for desc in labels.values() for d in desc]
    desc_to_cat = {d: c for c, desc in labels.items() for d in desc}

    df = pl.read_csv(args.data_path)
    img_paths = [
        os.path.join(args.img_base_path, f"{id}.jpg")
        for id in df[args.img_col].to_list()
    ]
    img_ids = df["id"].to_list()

    # batch inference
    pred_labels = []
    for i in tqdm(range(0, len(img_paths), args.batch_size), desc="batch classify"):
        paths = img_paths[i : i + args.batch_size]
        images = [Image.open(p).convert("RGB") for p in paths]
        ids = img_ids[i : i + args.batch_size]
        output = pipe(images, batch_size=args.batch_size, candidate_labels=desc_flatten)
        agg_score = batch_agg_score(output, categories, desc_to_cat, ids)
        pred_labels.extend(agg_score)

    # merge
    df_pred = pl.DataFrame(pred_labels)
    df.join(df_pred, on=pl.col("id")).write_csv(args.output_path)
    print(f"Saved the file successfully to: {args.output_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--data-path", required=True)
    parser.add_argument("--label-path", default="./notebooks/labels.json")
    parser.add_argument(
        "--img-base-path", default="../datasets/google-landmark/train-img"
    )
    parser.add_argument("--img-col", default="id")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-path", required=True)

    args = parser.parse_args()

    main(args)
