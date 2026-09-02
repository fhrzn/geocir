"""Build and augment templated retrieval queries.

Run as a standalone script to augment a templated-query JSON file with an
Ollama-served ``qwen3.5`` model (same backend used in
``notebooks/wikidata.ipynb``)::

    python -m src.data.query_builder -i test_templated_queries.json -o augmented_queries.json

Or import :func:`build_queries` / :func:`augment_queries` directly.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Literal

import httpx
import polars as pl
from pydantic import BaseModel
from tqdm.asyncio import tqdm


# --- Ollama config (mirrors notebooks/wikidata.ipynb) -------------------------
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3.5:latest")


_GEO_COL: dict[str, str] = {
    "country": "country",
    "region": "region",
    "subregion": "subregion",
    "city": "city",
}

_TEXT_TEMPLATE: dict[str, str] = {
    "country":   "A {category} landmark located in {geo}",
    "region":    "A {category} landmark located in the {geo} region",
    "subregion": "A {category} landmark located in {geo}",
    "city":      "A {category} landmark located in the city of {geo}",
}

_AUGMENT_PROMPT: str = (
    "You are generating search queries for evaluating a geographic landmark image retrieval system."
    " You will be given various captions containing landmark category and geographic attributes."
    " Your task is to generate {N} variations of the given captions."
    " You may combine some of them and formulate new and distinct queries."
    " Keep the generated queries generic - do not mention overly specific visual features."
    ' Respond with ONLY a JSON object of the form {{"augmented_captions": ["...", "..."]}}'
    " containing exactly {N} strings and nothing else."
    "\n\nCaptions:\n{query}"
)


def build_queries(
    df: pl.DataFrame,
    min_relevant: int = 5,
    granularity: Literal["country", "region", "subregion", "city"] = "country",
) -> list[dict]:
    """
    Build query objects from a geo-balanced index DataFrame.

    Each query represents one (category, <geo>) group where <geo> is
    determined by *granularity* ("country", "region", "subregion", or "city").

    Args:
        df: DataFrame with at minimum columns [category, <granularity_col>].
            Row indices must align with the FAISS index.
        min_relevant: Minimum relevant images required to include a query.
        granularity: Geographic level to group by.

    Returns:
        List of query dicts, one per (category, geo) group meeting the threshold.
    """
    geo_col = _GEO_COL[granularity]
    if geo_col not in df.columns:
        raise ValueError(f"Column '{geo_col}' not found in DataFrame for granularity='{granularity}'")

    template = _TEXT_TEMPLATE[granularity]
    queries = []

    for (category, geo), group in df.group_by(["category", geo_col]):
        if geo is None:
            continue
        indices = group["row_idx"].to_list()
        if len(indices) < min_relevant:
            print(f"[WARN] ({category}, {geo}) has fewer than {min_relevant} relevant indices")
            continue
        entry = {
            "stratify_key": f"{category}|{geo}",
            "text": template.format(category=category, geo=geo),
            "category": category,
            geo_col: geo,
            "granularity": granularity,
            "relevant_indices": indices,
        }
        queries.append(entry)

    return queries


class AugmentCaption(BaseModel):
    augmented_captions: list[str]


def _ollama_augment(
    prompt: str,
    *,
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    retries: int = 3,
    timeout: float = 1800.0,
) -> AugmentCaption:
    """Call the Ollama ``/api/chat`` endpoint and parse a structured response.

    Uses the same request shape as ``notebooks/wikidata.ipynb`` (``think`` and
    ``stream`` disabled, ``temperature=0``) plus a JSON schema ``format`` so the
    model is constrained to emit ``{"augmented_captions": [...]}``.
    """
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": prompt}],
        "think": False,
        "stream": False,
        "format": AugmentCaption.model_json_schema(),
        "options": {"temperature": 0},
    }
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            resp = httpx.post(f"{host}/api/chat", json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["message"]["content"].strip()
            if content:
                return AugmentCaption.model_validate_json(content)
        except Exception as e:  # noqa: BLE001 - retried, then reported
            last_err = e
    print(f"[WARN] ollama augment failed after {retries} attempts: {last_err}")
    return AugmentCaption(augmented_captions=[])


async def augment_queries(
    df: pl.DataFrame,
    N: int = 5,
    granularity: Literal["country", "region", "subregion", "city"] = "country",
    *,
    caption_col: str = "caption",
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    concurrency: int = 4,
    limit: int = 0,
) -> list[dict]:
    """Generate paraphrased query variations per (category, geo) group.

    Args:
        df: DataFrame with at least ``[category, <geo>, <caption_col>]``.
        N: Number of augmented captions to request per group.
        granularity: Geographic level to group by.
        caption_col: Column holding the source caption/text.
        model: Ollama model tag (defaults to ``qwen3.5:latest``).
        host: Ollama base URL.
        concurrency: Max in-flight requests to the Ollama server.
        limit: If > 0, only process the first ``limit`` groups (debugging).

    Returns:
        One dict per group with ``augmented_captions``.
    """
    geo_col = _GEO_COL[granularity]
    if geo_col not in df.columns:
        raise ValueError(f"Column '{geo_col}' not found in DataFrame for granularity='{granularity}'")
    if caption_col not in df.columns:
        raise ValueError(f"Column '{caption_col}' not found in DataFrame")

    groups = (
        df.filter(pl.col(geo_col).is_not_null())
        .group_by(["category", geo_col])
        .agg(pl.col(caption_col))
        .to_dicts()
    )
    for g in groups:
        g[caption_col] = "\n".join(str(c) for c in g[caption_col])
    if limit > 0:
        groups = groups[:limit]

    sem = asyncio.Semaphore(concurrency)

    async def worker(item: dict) -> dict:
        prompt = _AUGMENT_PROMPT.format(query=item[caption_col], N=N)
        async with sem:
            parsed = await asyncio.to_thread(
                _ollama_augment, prompt, model=model, host=host
            )
        return {
            "stratify_key": f"{item['category']}|{item[geo_col]}",
            "category": item["category"],
            geo_col: item[geo_col],
            "granularity": granularity,
            "augmented_captions": parsed.augmented_captions,
        }

    tasks = [worker(g) for g in groups]
    return await tqdm.gather(*tasks, desc=f"augmenting captions ({granularity})")


def load_input_frame(path: str | Path, caption_col: str = "caption") -> pl.DataFrame:
    """Load a templated-query JSON file into a DataFrame.

    Accepts either a top-level list of records or ``{"data": [...]}`` (the shape
    of ``test_templated_queries.json``). A ``text`` column is renamed to
    ``caption_col`` so it can be fed straight into :func:`augment_queries`.
    """
    raw = json.loads(Path(path).read_text())
    records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    df = pl.DataFrame(records)
    if "text" in df.columns and caption_col not in df.columns:
        df = df.rename({"text": caption_col})
    return df


def explode_to_input_schema(
    df: pl.DataFrame,
    results: list[dict],
    *,
    granularity: Literal["country", "region", "subregion", "city"] = "country",
    caption_col: str = "caption",
) -> list[dict]:
    """Fan augmented captions back onto the original rows, one row per variation.

    Every emitted record keeps the exact keys of an input record (``text``,
    ``category``, ``<geo>``, ``region``, ``relevant_ids``, ...) with ``text``
    replaced by an augmented variant. A group's variations are attached to each
    input row that fed the group (matched on ``(category, <geo>)``).
    """
    geo_col = _GEO_COL[granularity]
    captions_by_group: dict[tuple, list[str]] = {
        (r["category"], r[geo_col]): r["augmented_captions"] for r in results
    }

    exploded: list[dict] = []
    for row in df.iter_rows(named=True):
        base = {k: v for k, v in row.items() if k != caption_col}
        for variant in captions_by_group.get((row["category"], row[geo_col]), []):
            exploded.append({"text": variant, **base})
    return exploded


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Augment templated retrieval queries with an Ollama (qwen3.5) model."
    )
    p.add_argument(
        "-i", "--input", type=Path, default=Path("test_templated_queries.json"),
        help="Input JSON: a list of records or {\"data\": [...]} with text/category/<geo>.",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=Path("augmented_queries.json"),
        help="Output JSON path.",
    )
    p.add_argument("-n", "--num-variations", type=int, default=5,
                   help="Augmented captions to request per (category, geo) group.")
    p.add_argument("-g", "--granularity", choices=list(_GEO_COL), default="country")
    p.add_argument("--caption-col", default="caption",
                   help="Column holding the source caption (default: caption; 'text' is auto-renamed).")
    p.add_argument("--model", default=OLLAMA_MODEL, help=f"Ollama model tag (default: {OLLAMA_MODEL}).")
    p.add_argument("--host", default=OLLAMA_HOST, help=f"Ollama base URL (default: {OLLAMA_HOST}).")
    p.add_argument("--concurrency", type=int, default=4, help="Max in-flight Ollama requests.")
    p.add_argument("--limit", type=int, default=0, help="Only process the first N groups (debug).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    df = load_input_frame(args.input, caption_col=args.caption_col)
    print(f"[INFO] loaded {df.height} rows from {args.input}")

    results = asyncio.run(
        augment_queries(
            df,
            N=args.num_variations,
            granularity=args.granularity,
            caption_col=args.caption_col,
            model=args.model,
            host=args.host,
            concurrency=args.concurrency,
            limit=args.limit,
        )
    )

    exploded = explode_to_input_schema(
        df, results, granularity=args.granularity, caption_col=args.caption_col
    )
    args.output.write_text(
        json.dumps({"data": exploded}, indent=2, ensure_ascii=False)
    )
    print(
        f"[INFO] wrote {len(exploded)} augmented rows "
        f"(from {len(results)} groups) to {args.output}"
    )


if __name__ == "__main__":
    main()
