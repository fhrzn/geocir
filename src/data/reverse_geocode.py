"""
Reverse geocoding utility to extract country, country code, region, and subregion
from latitude and longitude coordinates.

Offline pass: reverse_geocoder for the coordinate lookup, with a GeoNames-derived
country table (geonamescache) for country/region/subregion so contested entries
without an ISO 3166-1 code (e.g. XK / Kosovo) are not dropped. Optional online
fallback via OpenStreetMap Nominatim for coordinates the offline pass cannot map.
"""

import argparse
import asyncio
import logging
from pathlib import Path

import country_converter as coco
import geonamescache
import httpx
import polars as pl
import pycountry
import pycountry_convert as pc
import reverse_geocoder as rg
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

_NOT_FOUND = {"not found", "not found in iso2", "not found in regex"}


def _build_country_df() -> pl.DataFrame:
    """Build a country_code → country/region/subregion mapping.

    Enumeration is driven by GeoNames (via geonamescache), not pycountry:
    reverse_geocoder emits GeoNames `cc` codes, and GeoNames covers entries
    pycountry omits (e.g. XK / Kosovo, which has no assigned ISO 3166-1 code).
    Iterating pycountry instead silently dropped those coordinates on the join.

    - country:    official name from pycountry, falling back to the GeoNames name
    - region:     continent from pycountry-convert, falling back to GeoNames
    - subregion:  UN region from country-converter (e.g. "Western Europe")
    """
    # GeoNames carries a few non-ISO codes (e.g. AN, CS) that country_converter
    # warns about; we handle the misses explicitly, so silence the noise.
    logging.getLogger("country_converter").setLevel(logging.ERROR)

    _cc = coco.CountryConverter()
    gc_countries = geonamescache.GeonamesCache().get_countries()

    records = []
    for code, info in gc_countries.items():
        pc_country = pycountry.countries.get(alpha_2=code)
        country = pc_country.name if pc_country else info.get("name")

        # Region (continent)
        try:
            continent_code = pc.country_alpha2_to_continent_code(code)
            region = pc.convert_continent_code_to_continent_name(continent_code)
        except Exception:
            try:
                region = pc.convert_continent_code_to_continent_name(
                    info["continentcode"]
                )
            except Exception:
                region = None

        # Subregion (UN M.49 sub-region)
        try:
            subregion = _cc.convert(code, to="UNregion")
            subregion = None if str(subregion).lower() in _NOT_FOUND else subregion
        except Exception:
            subregion = None

        records.append(
            {
                "country_code": code,
                "country": country,
                "region": region,
                "subregion": subregion,
            }
        )
    return pl.DataFrame(records)


# Country lookup table built at import time
df_cc = _build_country_df()


_GEO_COLS = ("country_code", "country", "region", "subregion", "city")


def reverse_geocode_df(
    df: pl.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    chunk_size: int = 250_000,
    rg_mode: int = 2,
) -> pl.DataFrame:
    # Keep null rows; only reverse geocode valid coordinates.
    valid = df.filter(
        pl.col(lat_col).is_not_null() & pl.col(lon_col).is_not_null()
    ).with_columns(
        pl.col(lat_col).cast(pl.Float64),
        pl.col(lon_col).cast(pl.Float64),
    )

    if valid.is_empty():
        return df.with_columns(
            [pl.lit(None).cast(pl.String).alias(c) for c in _GEO_COLS]
        )

    # Reverse geocode only distinct coordinates; datasets of photos/landmarks
    # repeat locations heavily, so this is the dominant speed-up. Order here is
    # arbitrary but internally consistent: `.rows()`, the per-chunk frames and
    # `rg.search` all preserve it, so results align positionally with a hstack.
    unique_coords = valid.select([lat_col, lon_col]).unique()
    coord_rows = unique_coords.rows()

    chunks: list[pl.DataFrame] = []
    for i in tqdm(
        range(0, len(coord_rows), chunk_size),
        desc="Reverse geocoding batches",
        unit="batch",
    ):
        # mode=2 uses multiprocessing in reverse_geocoder. Only `cc` and `name`
        # are consumed downstream; the rest of each record is discarded here so
        # nothing large accumulates across chunks.
        res = rg.search(coord_rows[i : i + chunk_size], mode=rg_mode, verbose=False)
        chunks.append(
            pl.DataFrame(
                {
                    "country_code": [r["cc"] for r in res],
                    "city": [r["name"] for r in res],
                }
            )
        )

    lookup = (
        unique_coords.hstack(pl.concat(chunks, how="vertical"))
        .join(df_cc, on="country_code", how="left")
        .select([lat_col, lon_col, *_GEO_COLS])
    )

    # Both key columns are Float64 cast from the same source, so the hash join
    # matches on identical bit patterns.
    return df.with_columns(
        pl.col(lat_col).cast(pl.Float64),
        pl.col(lon_col).cast(pl.Float64),
    ).join(
        lookup,
        on=[lat_col, lon_col],
        how="left",
    )


_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
_NOMINATIM_HEADERS = {"User-Agent": "GeoTIR/1.0"}
# OSM usage policy: at most one request per second.
_NOMINATIM_MIN_INTERVAL = 1.0


def _extract_city(address: dict) -> str | None:
    # Nominatim uses different keys depending on settlement type.
    for key in ("city", "town", "village", "hamlet", "suburb", "county"):
        if address.get(key):
            return address[key]
    return None


async def _nominatim_reverse(
    client: httpx.AsyncClient, lat: float, lon: float
) -> dict:
    response = await client.get(
        _NOMINATIM_URL,
        params={"lat": lat, "lon": lon, "format": "json"},
        headers=_NOMINATIM_HEADERS,
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()


async def get_city_nominatim(img_id: str, lat: float, lon: float) -> dict:
    async with httpx.AsyncClient() as client:
        data = await _nominatim_reverse(client, lat, lon)
    return {"img_id": img_id, "city": _extract_city(data.get("address", {}))}


async def nominatim_fallback_df(
    df: pl.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    min_interval: float = _NOMINATIM_MIN_INTERVAL,
) -> pl.DataFrame:
    """Fill rows still missing country info by reverse geocoding with OSM Nominatim.

    Expects the output schema of `reverse_geocode_df`. Only unique coordinates
    that have a non-null lat/lon but a null `country_code` are looked up, one
    request per second per OSM's usage policy. Region/subregion for a resolved
    country come from the same `df_cc` table used by the offline pass.
    """
    missing = (
        df.filter(
            pl.col(lat_col).is_not_null()
            & pl.col(lon_col).is_not_null()
            & pl.col("country_code").is_null()
        )
        .select([lat_col, lon_col])
        .unique(maintain_order=True)
    )

    if missing.is_empty():
        return df

    resolved: list[dict] = []
    async with httpx.AsyncClient() as client:
        for lat, lon in atqdm(
            list(missing.iter_rows()),
            desc="Nominatim fallback",
            unit="coord",
        ):
            try:
                data = await _nominatim_reverse(client, lat, lon)
                address = data.get("address", {})
                code = (address.get("country_code") or "").upper() or None
                resolved.append(
                    {
                        lat_col: float(lat),
                        lon_col: float(lon),
                        "nom_country_code": code,
                        "nom_country": address.get("country"),
                        "nom_city": _extract_city(address),
                    }
                )
            except (httpx.HTTPError, ValueError):
                pass
            await asyncio.sleep(min_interval)

    if not resolved:
        return df

    nom_df = pl.DataFrame(resolved).join(
        df_cc.select(["country_code", "country", "region", "subregion"]).rename(
            {
                "country_code": "nom_country_code",
                "country": "nom_country_std",
                "region": "nom_region",
                "subregion": "nom_subregion",
            }
        ),
        on="nom_country_code",
        how="left",
    )

    return (
        df.join(nom_df, on=[lat_col, lon_col], how="left")
        .with_columns(
            pl.coalesce("country_code", "nom_country_code").alias("country_code"),
            # Prefer the canonical df_cc name; fall back to the raw OSM string.
            pl.coalesce("country", "nom_country_std", "nom_country").alias("country"),
            pl.coalesce("region", "nom_region").alias("region"),
            pl.coalesce("subregion", "nom_subregion").alias("subregion"),
            pl.coalesce("city", "nom_city").alias("city"),
        )
        .drop(
            "nom_country_code",
            "nom_country",
            "nom_country_std",
            "nom_city",
            "nom_region",
            "nom_subregion",
        )
    )


def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Validate input file exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {input_path}...")
    df = pl.read_csv(input_path)

    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns}")

    # Reverse geocode
    print(f"\nReverse geocoding using columns: {args.lat_col}, {args.lon_col}")
    df_geocoded = reverse_geocode_df(df, lat_col=args.lat_col, lon_col=args.lon_col)

    if args.nominatim_fallback:
        missing = (
            df_geocoded.filter(
                pl.col(args.lat_col).is_not_null()
                & pl.col(args.lon_col).is_not_null()
                & pl.col("country_code").is_null()
            )
            .select([args.lat_col, args.lon_col])
            .n_unique()
        )
        if missing:
            eta_min = missing * _NOMINATIM_MIN_INTERVAL / 60
            print(
                f"\n{missing} unique coordinate(s) unresolved; querying Nominatim "
                f"at ~1 req/s (~{eta_min:.0f} min)..."
            )
            df_geocoded = asyncio.run(
                nominatim_fallback_df(
                    df_geocoded, lat_col=args.lat_col, lon_col=args.lon_col
                )
            )

    # Save
    print(f"\nSaving to {output_path}...")
    df_geocoded.write_csv(output_path)

    print("Done!")
    print("\nSample results:")
    print(df_geocoded.head())


if __name__ == "__main__":
    """Reverse geocode CSV file with latitude/longitude columns."""
    parser = argparse.ArgumentParser(
        description="Reverse geocode coordinates to get country, region, and subregion."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input CSV file path with latitude and longitude columns",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--lat-col",
        default="latitude",
        help="Name of latitude column (default: latitude)",
    )
    parser.add_argument(
        "--lon-col",
        default="longitude",
        help="Name of longitude column (default: longitude)",
    )
    parser.add_argument(
        "--nominatim-fallback",
        action="store_true",
        help="Resolve coordinates the offline pass could not map by querying "
        "OpenStreetMap Nominatim (network, ~1 request/sec).",
    )

    args = parser.parse_args()
    main(args)
