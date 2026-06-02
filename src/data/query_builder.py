import asyncio
import os
from typing import List, Literal

import polars as pl
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm


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


async def augment_queries(
    df: pl.DataFrame,
    N: int = 5,
    granularity: Literal["country", "region", "subregion", "city"] = "country",
):
    geo_col = _GEO_COL[granularity]
    if geo_col not in df.columns:
        raise ValueError(f"Column '{geo_col}' not found in DataFrame for granularity='{granularity}'")

    class AugmentCaption(BaseModel):
        augmented_captions: List[str]

    async def call_oai(client: AsyncOpenAI, prompt: str, item: dict, N: int):
        await asyncio.sleep(2)
        output = await client.responses.parse(
            model="gpt-4o-mini",
            text_format=AugmentCaption,
            input=[
                {
                    "role": "system",
                    "content": prompt.format(query=item["caption"], N=N),
                },
            ],
        )
        return {
            "stratify_key": f"{item['category']}|{item[geo_col]}",
            "category": item["category"],
            geo_col: item[geo_col],
            "granularity": granularity,
            "augmented_captions": output.output[1].content[0].parsed.augmented_captions,
        }

    oai_cl = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = (
        "You are generating search queries for evaluating a geographic landmark image retrieval system."
        " You will be given various captions containing landmark category and geographic attributes."
        " Your task is to generate {N} variations of the given captions."
        " You may combine some of them and formulate new and distinct queries."
        " Keep the generated queries generic — do not mention overly specific visual features."
        "\n\nCaptions:\n{query}"
    )

    groups = (
        df.filter(pl.col(geo_col).is_not_null())
        .group_by(["category", geo_col])
        .agg(pl.col("caption"))
        .to_dicts()
    )
    for g in groups:
        g["caption"] = "\n".join(g["caption"])

    tasks = [call_oai(oai_cl, prompt, g, N) for g in groups]
    results = await tqdm.gather(*tasks, desc=f"augmenting captions ({granularity})")

    return results
