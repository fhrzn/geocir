import asyncio
import base64
import csv
import glob
import io
import json
import os
import re
from argparse import ArgumentParser

import polars as pl
from openai import AsyncOpenAI
from PIL import Image
from tqdm import tqdm

MODEL_ID = "OpenGVLab/InternVL3_5-2B-HF"
IMG_MAX_SIZE = 336
NUM_WORKERS = 8
BATCH_SIZE = 2000
NON_LANDMARK = "non_landmark"

SYSTEM_PROMPT = """\
You are a visual landmark classifier and image captioner.
For each image:
1. Decide if it shows a recognizable landmark or built heritage site.
2. Assign exactly one category and write a concise visual caption.

Caption rules:
- Describe visual appearance: architecture style, materials, setting, notable features.
- Naturally include the country/region context provided.
- Do NOT name the landmark or repeat the category label.
- Keep it under 50 words (fed to CLIP with 77-token limit).

Confidence calibration — be honest, use the full range:
- 0.9–1.0: unmistakable, no ambiguity at all
- 0.7–0.9: clearly fits, minor uncertainty
- 0.5–0.7: plausible but notable doubt
- 0.3–0.5: weak evidence, could easily be wrong
- 0.0–0.3: mostly guessing\
"""


def load_labels(path: str) -> list[str]:
    with open(path) as f:
        return list(json.load(f).keys()) + [NON_LANDMARK]


def build_prompt(labels: list[str], country: str = "") -> str:
    category_block = "\n".join(f"  {chr(65+i)}. {lbl}" for i, lbl in enumerate(labels))
    last_letter = chr(65 + len(labels) - 1)
    non_landmark_letter = chr(65 + labels.index(NON_LANDMARK))
    country_line = f"Country: {country}\n\n" if country else ""
    return f"""{country_line}Analyze the image and output ONLY a JSON object — no extra text:
{{
  "category": "<single letter {chr(65)}–{last_letter}>",
  "confidence": <float 0.0–1.0>,
  "caption": "<concise visual description including country context, under 50 words>"
}}

Available categories:
{category_block}

If no recognizable landmark is visible, set "category" to "{non_landmark_letter}" ({NON_LANDMARK})."""


def encode_image(img_path: str) -> str:
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((IMG_MAX_SIZE, IMG_MAX_SIZE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text: str) -> dict:
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"No JSON in: {text!r}")
    return json.loads(m.group())


async def classify_image(
    client: AsyncOpenAI,
    img_path: str,
    labels: list[str],
    letter_to_label: dict[str, str],
    country: str = "",
) -> tuple[str, float, str]:
    b64 = await asyncio.get_running_loop().run_in_executor(None, encode_image, img_path)
    prompt = build_prompt(labels, country)
    response = await client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]},
        ],
        max_tokens=256,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()

    try:
        result = parse_response(raw)
    except (ValueError, json.JSONDecodeError):
        return NON_LANDMARK, 0.0, f"[parse error] {raw}"

    letter = str(result.get("category", "")).strip().upper()[:1]
    label = letter_to_label.get(letter, NON_LANDMARK)
    confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    caption = result.get("caption", "").strip()
    return label, confidence, caption


async def classify_one(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    img_id: str,
    img_path: str,
    labels: list[str],
    letter_to_label: dict[str, str],
    id_col: str,
    country: str = "",
) -> dict:
    async with sem:
        try:
            label, score, caption = await classify_image(client, img_path, labels, letter_to_label, country)
        except Exception as e:
            label, score, caption = NON_LANDMARK, 0.0, f"[error] {e}"
        return {id_col: img_id, "pred_label": label, "pred_score": score, "caption": caption}



async def run(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key="none")
    sem = asyncio.Semaphore(args.num_workers)

    labels = load_labels(args.labels_path)
    letter_to_label = {chr(65 + i): lbl for i, lbl in enumerate(labels)}
    print(f"Categories ({len(labels)}): {labels}")

    df = pl.read_csv(args.data_path)
    rows = df.to_dicts()
    row_lookup = {row[args.id_col]: row for row in rows}
    img_ids = [row[args.id_col] for row in rows]
    img_paths = [os.path.join(args.img_root, row[args.src_col], f"{row[args.id_col]}.jpg") for row in rows]
    countries = [row.get("country", "") or "" for row in rows]

    done_ids: set = set()
    file_exists = os.path.exists(args.output_path)
    if file_exists:
        with open(args.output_path, newline="") as f:
            done_ids = {row[args.id_col] for row in csv.DictReader(f)}
        print(f"Resuming — {len(done_ids)} done, {len(img_ids) - len(done_ids)} remaining.")

    todo = [(id_, p, c) for id_, p, c in zip(img_ids, img_paths, countries) if id_ not in done_ids]
    if not todo:
        print("All images already processed.")
        return

    todo_ids, todo_paths, todo_countries = zip(*todo)
    fieldnames = df.columns + ["pred_label", "pred_score", "caption"]
    success_count = error_count = 0

    with open(args.output_path, "a", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        with tqdm(total=len(todo_ids), desc="classify", unit="img") as pbar:
            for batch_start in range(0, len(todo_ids), BATCH_SIZE):
                tasks = [
                    asyncio.create_task(classify_one(
                        sem, client, id_, path, labels, letter_to_label, args.id_col, country
                    ))
                    for id_, path, country in zip(
                        todo_ids[batch_start:batch_start + BATCH_SIZE],
                        todo_paths[batch_start:batch_start + BATCH_SIZE],
                        todo_countries[batch_start:batch_start + BATCH_SIZE],
                    )
                ]
                for fut in asyncio.as_completed(tasks):
                    record = await fut
                    writer.writerow({**row_lookup[record[args.id_col]], **record})
                    out_f.flush()
                    if "[error]" in record["caption"]:
                        error_count += 1
                    else:
                        success_count += 1
                    pbar.set_postfix(success=success_count, error=error_count)
                    pbar.update(1)

    print(f"Done — {success_count} succeeded, {error_count} errors — saved to {args.output_path}")


async def smoke_test(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key="none")
    labels = load_labels(args.labels_path)
    letter_to_label = {chr(65 + i): lbl for i, lbl in enumerate(labels)}

    sample_paths = sorted(glob.glob(os.path.join(args.img_root, "train", "*.jpg")))[:5]
    if not sample_paths:
        print("No images found for smoke test.")
        return

    print(f"Smoke test — {len(sample_paths)} images\n")
    for path in sample_paths:
        label, conf, caption = await classify_image(client, path, labels, letter_to_label)
        print(f"File      : {os.path.basename(path)}")
        print(f"Label     : {label}  ({conf:.4f})")
        print(f"Caption   : {caption}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Zero-shot landmark label + caption via vLLM VLM")
    parser.add_argument("--base-url", default="http://localhost:3456/v1")
    parser.add_argument("--labels-path", default="./label_ensemble.json")
    parser.add_argument("--img-root", default="/mnt/yokoyamalab-nas/gldv2-full")
    parser.add_argument("--data-path", help="CSV with image IDs and src column (required unless --smoke-test)")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--src-col", default="src")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output-path", default="./predictions_caption.csv")
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    if args.smoke_test:
        asyncio.run(smoke_test(args))
    else:
        if not args.data_path:
            parser.error("--data-path is required unless --smoke-test is used")
        asyncio.run(run(args))
