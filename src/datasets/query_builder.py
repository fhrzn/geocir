import polars as pl


def build_queries(
    df: pl.DataFrame,
    min_relevant: int = 5,
) -> list[dict]:
    """
    Build query objects from a geo-balanced index DataFrame.

    Each query represents one (category, country) group and contains:
      - text: natural-language query string
      - category: the landmark category
      - country: the country name
      - relevant_indices: row indices in the original DataFrame that are relevant
                          (same category + country)

    Args:
        df: DataFrame with at minimum columns [category, country].
            Row indices must align with the FAISS index (i.e. df is the index metadata source).
        min_relevant: Minimum number of relevant images required to include a query.

    Returns:
        List of query dicts, one per (category, country) group meeting the threshold.
    """
    queries = []

    for (category, country), group in df.group_by(["category", "country"]):
        indices = group["row_idx"].to_list()
        if len(indices) < min_relevant:
            continue
        queries.append({
            "text": f"A {category} landmark located in {country}",
            "category": category,
            "country": country,
            "relevant_indices": indices,
        })

    return queries
