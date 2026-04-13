"""
Zero-shot landmark classification using a vLLM-served vision-language model.

Pipeline (single step per image):
  Step 1 — Eligibility check + chain-of-thought: verify the image shows a
            landmark (using the eligibility definition in the system prompt),
            then describe the image, reason through categories, and report
            a confidence score — all in one pass.

Outputs: pred_label, pred_score, rationale columns joined to the input CSV.
"""

import asyncio
import base64
import io
import json
import os
import re
from argparse import ArgumentParser

import polars as pl
from openai import AsyncOpenAI
from PIL import Image
from tqdm import tqdm


# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_ID = "OpenGVLab/InternVL3_5-14B-HF"
IMG_MAX_SIZE = 336      # resize longest side before encoding
NUM_WORKERS = 8         # concurrent async requests to vLLM
BATCH_SIZE = 2000       # max tasks created at once to bound memory usage
NON_LANDMARK = "non-landmark"


# ─── Taxonomy helpers ─────────────────────────────────────────────────────────

def load_taxonomy(taxonomy_path: str) -> dict[str, dict[str, str]]:
    """
    Load the taxonomy JSON file.

    Expected schema per entry:
      {
        "<category_key>": {
          "description": "...",
          "not":         "...",   # exclusion rules (may be empty string)
          "examples":    "..."
        }, ...
      }
    """
    with open(taxonomy_path) as f:
        return json.load(f)


def build_category_map(taxonomy: dict) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """
    Returns:
      categories     – ordered list of category keys
      letter_to_cat  – {'A': 'religious_christian', 'B': 'religious_islamic', ...}
      cat_to_letter  – reverse mapping
    """
    categories = list(taxonomy.keys())
    assert len(categories) <= 26, "Too many categories for single-letter encoding"
    letter_to_cat = {chr(65 + i): cat for i, cat in enumerate(categories)}
    cat_to_letter = {cat: letter for letter, cat in letter_to_cat.items()}
    return categories, letter_to_cat, cat_to_letter


# ─── Prompt builders ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
STEP 1 — LANDMARK ELIGIBILITY CHECK
A landmark is defined as a geographically fixed, permanently built or naturally
formed structure that meets ALL THREE criteria:
  A. Physically bounded and fixed in space (not a crowd, event, or interior)
  B. Publicly recognizable through cultural, historical, or architectural significance
     (has a Wikipedia entry or is widely photographed as a named place)
  C. Geographic point reference: its visual appearance carries information about
     WHERE in the world it is — the same type of structure looks different across regions

If the named entity does NOT meet all three criteria, output:
  {"eligible": false, "reason": "<which criterion fails>"}

If it meets all three, proceed to STEP 2 — CATEGORY CLASSIFICATION.\
"""


def build_category_block(taxonomy: dict, categories: list[str]) -> str:
    lines = []
    for i, cat in enumerate(categories):
        letter = chr(65 + i)
        entry = taxonomy[cat]
        lines.append(f"  {letter}. {cat}")
        lines.append(f"     {entry['description']}")
        if entry.get("not"):
            lines.append(f"     NOT: {entry['not']}")
        if entry.get("examples"):
            lines.append(f"     Examples: {entry['examples']}")
        lines.append("")   # blank line between entries
    return "\n".join(lines).rstrip()


def build_step1_prompt(taxonomy: dict, categories: list[str]) -> str:
    last_letter = chr(65 + len(categories) - 1)
    category_block = build_category_block(taxonomy, categories)
    return f"""Carefully analyze the image.

First apply the landmark eligibility check from the system prompt.
- If the image does NOT qualify as a landmark, output only:
  {{"eligible": false, "reason": "<which criterion fails>"}}

- If it DOES qualify, proceed with STEP 2 — CATEGORY CLASSIFICATION.

Available landmark categories:
{category_block}

For a qualifying landmark, output ONLY a JSON object in this exact format — no extra text:
{{
  "eligible": true,
  "description": "<concise description of what you see>",
  "reasoning": "<step-by-step reasoning leading to your label>",
  "label": "<single letter {chr(65)}–{last_letter}>",
  "confidence": <float 0.0–1.0>
}}"""


# ─── Image encoding ───────────────────────────────────────────────────────────

def encode_image(img_path: str) -> str:
    """Load, resize, and base64-encode an image as a JPEG data-URI string."""
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((IMG_MAX_SIZE, IMG_MAX_SIZE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def image_content(b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


# ─── JSON parsing ─────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict:
    """Extract and parse the first JSON object from a model response."""
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"No JSON found in response: {text!r}")
    return json.loads(match.group())


# ─── Core classification (async) ──────────────────────────────────────────────

async def classify_image_async(
    client: AsyncOpenAI,
    img_path: str,
    prompt: str,
    letter_to_cat: dict[str, str],
) -> tuple[str, float, str]:
    """
    Run the single-step eligibility check + CoT classification for one image.

    encode_image is CPU-bound (PIL), so it runs in the default thread pool via
    run_in_executor. The vLLM API call is I/O-bound and runs natively async.

    Returns:
      pred_label   – category key (e.g. 'bridge') or 'non-landmark'
      pred_score   – confidence in [0, 1]
      rationale    – description and reasoning text
    """
    loop = asyncio.get_running_loop()
    b64 = await loop.run_in_executor(None, encode_image, img_path)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [image_content(b64), {"type": "text", "text": prompt}],
        },
    ]
    response = await client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=512,
        temperature=0.0,
    )
    response_text = response.choices[0].message.content.strip()

    try:
        result = parse_json(response_text)
    except (ValueError, json.JSONDecodeError):
        return NON_LANDMARK, 0.0, f"[parse error] {response_text}"

    if not result.get("eligible", True):
        reason = result.get("reason", "failed eligibility check")
        return NON_LANDMARK, 1.0, f"Eligibility check failed: {reason}"

    label_key = _resolve_label(str(result.get("label", "")).strip().upper(), letter_to_cat)
    confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    rationale = _build_rationale(result)
    return label_key, confidence, rationale


async def _classify_one(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    img_id: str,
    img_path: str,
    prompt: str,
    letter_to_cat: dict[str, str],
    id_col: str,
) -> dict:
    """Semaphore-guarded wrapper — mirrors _scrape_one_with_semaphore."""
    async with sem:
        try:
            label, score, rationale = await classify_image_async(
                client, img_path, prompt, letter_to_cat
            )
            return {
                id_col: img_id,
                "pred_label": label,
                "pred_score": score,
                "rationale": rationale,
            }
        except Exception as e:
            return {
                id_col: img_id,
                "pred_label": NON_LANDMARK,
                "pred_score": 0.0,
                "rationale": f"[error] {e}",
            }


def _resolve_label(raw: str, letter_to_cat: dict[str, str]) -> str:
    """Convert a raw label token (letter or 'NON-LANDMARK') to a category key."""
    if raw in ("NON-LANDMARK", "NON_LANDMARK", NON_LANDMARK.upper()):
        return NON_LANDMARK
    letter = raw[0] if raw else ""
    return letter_to_cat.get(letter, NON_LANDMARK)


def _build_rationale(result: dict) -> str:
    """Concatenate description and reasoning into one rationale string."""
    parts = []
    if result.get("description"):
        parts.append(f"Observation: {result['description']}")
    if result.get("reasoning"):
        parts.append(f"Reasoning: {result['reasoning']}")
    return " | ".join(parts)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

async def run(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key="none")
    sem = asyncio.Semaphore(args.num_workers)

    taxonomy = load_taxonomy(args.taxonomy_path)
    categories, letter_to_cat, _ = build_category_map(taxonomy)
    prompt = build_step1_prompt(taxonomy, categories)

    df = pl.read_csv(args.data_path)
    img_ids = df[args.id_col].to_list()
    img_paths = [os.path.join(args.img_base_path, f"{id_}.jpg") for id_ in img_ids]

    valid = [(id_, p) for id_, p in zip(img_ids, img_paths) if os.path.exists(p)]
    missing = len(img_ids) - len(valid)
    if missing:
        print(f"Warning: {missing} images not found on disk — skipped.")
    if not valid:
        print("No valid images found. Exiting.")
        return
    img_ids, img_paths = zip(*valid)

    # ── Checkpoint / resume ───────────────────────────────────────────────────
    done_ids: set = set()
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)[args.id_col])
        print(f"Resuming — {len(done_ids)} already done, {len(img_ids) - len(done_ids)} remaining.")

    todo = [(id_, p) for id_, p in zip(img_ids, img_paths) if id_ not in done_ids]
    if not todo:
        print("All images already processed.")
        return
    todo_ids, todo_paths = zip(*todo)

    # ── Run in batches to bound memory (BATCH_SIZE tasks at a time) ───────────
    success_count = 0
    error_count = 0

    with open(args.output_path, "a") as out_f:
        with tqdm(total=len(todo_ids), desc="classify", unit="img") as pbar:
            for batch_start in range(0, len(todo_ids), BATCH_SIZE):
                batch = list(zip(
                    todo_ids[batch_start:batch_start + BATCH_SIZE],
                    todo_paths[batch_start:batch_start + BATCH_SIZE],
                ))
                tasks = [
                    asyncio.create_task(
                        _classify_one(sem, client, id_, path, prompt, letter_to_cat, args.id_col)
                    )
                    for id_, path in batch
                ]
                for fut in asyncio.as_completed(tasks):
                    record = await fut
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    if "[error]" in record.get("rationale", ""):
                        error_count += 1
                    else:
                        success_count += 1
                    pbar.set_postfix(success=success_count, error=error_count)
                    pbar.update(1)

    print(f"Done — {success_count} succeeded, {error_count} errors — appended to {args.output_path}")


# ─── Smoke test ───────────────────────────────────────────────────────────────

async def smoke_test(args):
    """Classify 5 sample images and print results — no CSV written."""
    import glob

    client = AsyncOpenAI(base_url=args.base_url, api_key="none")
    taxonomy = load_taxonomy(args.taxonomy_path)
    categories, letter_to_cat, _ = build_category_map(taxonomy)
    prompt = build_step1_prompt(taxonomy, categories)

    sample_paths = sorted(glob.glob(os.path.join(args.img_base_path, "*.jpg")))[:5]
    if not sample_paths:
        print("No images found for smoke test.")
        return

    print(f"Smoke test — classifying {len(sample_paths)} images\n")
    for path in sample_paths:
        label, conf, rationale = await classify_image_async(
            client, path, prompt, letter_to_cat
        )
        print(f"File      : {os.path.basename(path)}")
        print(f"Label     : {label}")
        print(f"Confidence: {conf:.4f}")
        print(f"Rationale : {rationale}")
        print()

    print("Smoke test passed.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser(description="Zero-shot landmark classification via vLLM VLM")
    parser.add_argument("--base-url", default="http://localhost:3456/v1")
    parser.add_argument("--taxonomy-path", default="./taxonomy.json")
    parser.add_argument("--img-base-path", default="/mnt/yokoyamalab-nas/gldv2-full/train")
    parser.add_argument("--data-path", help="CSV with image IDs (required unless --smoke-test)")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output-path", default="./predictions_llm.jsonl")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Classify 5 sample images and print results without writing a CSV",
    )

    args = parser.parse_args()

    if args.smoke_test:
        asyncio.run(smoke_test(args))
    else:
        if not args.data_path:
            parser.error("--data-path is required unless --smoke-test is used")
        asyncio.run(run(args))
