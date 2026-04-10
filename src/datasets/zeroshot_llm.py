"""
Zero-shot landmark classification using a vLLM-served vision-language model.

Pipeline (single step per image):
  Step 1 — Eligibility check + chain-of-thought: verify the image shows a
            landmark (using the eligibility definition in the system prompt),
            then describe the image, reason through categories, and report
            a confidence score — all in one pass.

Outputs: pred_label, pred_score, rationale columns joined to the input CSV.
"""

import base64
import io
import json
import os
import re
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from openai import OpenAI
from PIL import Image
from tqdm import tqdm


# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_ID = "OpenGVLab/InternVL3_5-14B-HF"
IMG_MAX_SIZE = 336      # resize longest side before encoding
NUM_WORKERS = 4         # parallel HTTP requests
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


# ─── Core classification ───────────────────────────────────────────────────────

def classify_image(
    client: OpenAI,
    img_path: str,
    prompt: str,
    letter_to_cat: dict[str, str],
) -> tuple[str, float, str]:
    """
    Run the single-step eligibility check + CoT classification for one image.

    Returns:
      pred_label   – category key (e.g. 'bridge') or 'non-landmark'
      pred_score   – confidence in [0, 1]
      rationale    – description and reasoning text
    """
    b64 = encode_image(img_path)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [image_content(b64), {"type": "text", "text": prompt}],
        },
    ]
    response = client.chat.completions.create(
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

    # ── Eligibility early-exit ────────────────────────────────────────────────
    if not result.get("eligible", True):
        reason = result.get("reason", "failed eligibility check")
        return NON_LANDMARK, 1.0, f"Eligibility check failed: {reason}"

    # ── Resolve label and confidence ──────────────────────────────────────────
    label_key = _resolve_label(str(result.get("label", "")).strip().upper(), letter_to_cat)
    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    rationale = _build_rationale(result)
    return label_key, confidence, rationale


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

def run(args):
    client = OpenAI(base_url=args.base_url, api_key="none")

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

    results: list[dict] = [None] * len(img_ids)

    def _task(idx: int, img_id: str, img_path: str) -> tuple[int, dict]:
        label, score, rationale = classify_image(
            client, img_path, prompt, letter_to_cat
        )
        return idx, {
            args.id_col: img_id,
            "pred_label": label,
            "pred_score": score,
            "rationale": rationale,
        }

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {
            pool.submit(_task, i, id_, path): i
            for i, (id_, path) in enumerate(zip(img_ids, img_paths))
        }
        with tqdm(total=len(futures), desc="classify") as pbar:
            for future in as_completed(futures):
                idx, record = future.result()
                results[idx] = record
                pbar.update(1)

    df_pred = pl.DataFrame(results)
    df.join(df_pred, on=args.id_col).write_csv(args.output_path)
    print(f"Saved {len(results)} predictions → {args.output_path}")


# ─── Smoke test ───────────────────────────────────────────────────────────────

def smoke_test(args):
    """Classify 5 sample images and print results — no CSV written."""
    import glob

    client = OpenAI(base_url=args.base_url, api_key="none")
    taxonomy = load_taxonomy(args.taxonomy_path)
    categories, letter_to_cat, _ = build_category_map(taxonomy)
    prompt = build_step1_prompt(taxonomy, categories)

    sample_paths = sorted(glob.glob(os.path.join(args.img_base_path, "*.jpg")))[:5]
    if not sample_paths:
        print("No images found for smoke test.")
        return

    print(f"Smoke test — classifying {len(sample_paths)} images\n")
    for path in sample_paths:
        label, conf, rationale = classify_image(
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
    parser.add_argument("--taxonomy-path", default="./notebooks/taxonomy.json")
    parser.add_argument("--img-base-path", default="/mnt/yokoyamalab-nas/gldv2-full/train")
    parser.add_argument("--data-path", help="CSV with image IDs (required unless --smoke-test)")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output-path", default="./predictions_llm.csv")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Classify 5 sample images and print results without writing a CSV",
    )

    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(args)
    else:
        if not args.data_path:
            parser.error("--data-path is required unless --smoke-test is used")
        run(args)
