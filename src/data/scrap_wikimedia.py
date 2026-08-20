import argparse
import asyncio
import csv
import json
import os
import re
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Optional
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse

import httpx
import polars as pl
from bs4 import BeautifulSoup
from tqdm import tqdm

GEO_DOMAINS = ("geohack.toolforge.org", "tools.wmflabs.org")
MOVED_KEYWORDS = (
    "moved",
    "renamed",
    "redirect",
    "now at",
    "see",
    "has moved",
    "replaced by",
    "use category",
)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://commons.wikimedia.org/",
}
RETRY_STATUS_CODES = {403, 429, 500, 502, 503, 504}
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
# OSM entity type props in priority order: relation > node > way
WIKIDATA_OSM_PROPS = [("P402", "relation"), ("P11693", "node"), ("P10689", "way")]


@dataclass
class CommonsCoordinateResult:
    input_url: str
    resolved_url: str
    geohack_url: str
    latitude: float
    longitude: float


def _to_decimal(parts: list[str], hemisphere: str) -> float:
    vals = [float(p.replace(",", ".")) for p in parts if p]
    if not vals:
        raise ValueError("No DMS parts found")

    deg = vals[0]
    minutes = vals[1] if len(vals) > 1 else 0.0
    seconds = vals[2] if len(vals) > 2 else 0.0
    decimal = abs(deg) + minutes / 60.0 + seconds / 3600.0

    if hemisphere in ("S", "W"):
        decimal *= -1
    if deg < 0:
        decimal *= -1

    return decimal


def parse_geohack_params(params_value: str) -> tuple[float, float]:
    tokens = [t for t in unquote(params_value).split("_") if t]

    lat_idx = next((i for i, t in enumerate(tokens) if t in ("N", "S")), None)
    if lat_idx is None:
        raise ValueError(f"Cannot find latitude hemisphere in params={params_value}")

    lon_idx = next(
        (
            i
            for i, t in enumerate(tokens[lat_idx + 1 :], start=lat_idx + 1)
            if t in ("E", "W")
        ),
        None,
    )
    if lon_idx is None:
        raise ValueError(f"Cannot find longitude hemisphere in params={params_value}")

    lat_parts = tokens[:lat_idx]
    lon_parts = tokens[lat_idx + 1 : lon_idx]

    lat = _to_decimal(lat_parts, tokens[lat_idx])
    lon = _to_decimal(lon_parts, tokens[lon_idx])
    return lat, lon


def parse_geohack_url(geohack_url: str) -> tuple[float, float]:
    query = parse_qs(urlparse(geohack_url).query)
    params_values = query.get("params")
    if not params_values:
        raise ValueError(f"No params query in geohack URL: {geohack_url}")
    return parse_geohack_params(params_values[0])


def _is_category_wiki_link(full_url: str) -> bool:
    p = urlparse(full_url)
    return (p.hostname or "").lower() == "commons.wikimedia.org" and p.path.startswith(
        "/wiki/Category:"
    )


def _find_geohack_link(soup: BeautifulSoup, base_url: str) -> Optional[str]:
    for a in soup.select("a[href]"):
        full = urljoin(base_url, a.get("href", ""))
        host = (urlparse(full).hostname or "").lower()
        if any(dom in host for dom in GEO_DOMAINS) and "params=" in full:
            return full
    return None


def _extract_candidate_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    scored: list[tuple[int, str]] = []
    selectors = [
        ".redirectMsg a[href]",
        ".hatnote a[href]",
        ".mbox-text a[href]",
        ".mw-parser-output .notice a[href]",
        ".mw-parser-output .plainlinks a[href]",
    ]
    for sel in selectors:
        for a in soup.select(sel):
            full = urljoin(base_url, a.get("href", ""))
            if _is_category_wiki_link(full):
                scored.append((0, full))

    for block in soup.select(
        "#mw-content-text p, #mw-content-text li, #mw-content-text div, #mw-content-text td"
    ):
        if any(k in block.get_text(" ", strip=True).lower() for k in MOVED_KEYWORDS):
            for a in block.select("a[href]"):
                full = urljoin(base_url, a.get("href", ""))
                if _is_category_wiki_link(full):
                    scored.append((1, full))

    for a in soup.select('#mw-content-text a[href^="/wiki/Category:"]'):
        full = urljoin(base_url, a.get("href", ""))
        if _is_category_wiki_link(full):
            scored.append((2, full))

    canonical = soup.select_one('link[rel="canonical"]')
    if canonical and canonical.get("href"):
        full = canonical["href"]
        if _is_category_wiki_link(full):
            scored.append((1, full))

    scored.sort(key=lambda x: x[0])
    out: list[str] = []
    seen: set[str] = set()
    for _, u in scored:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


async def _fetch_with_retries(
    client: httpx.AsyncClient, url: str, timeout: float, retries: int = 8
) -> httpx.Response:
    base_delay = 1.0
    for attempt in range(retries):
        try:
            resp = await client.get(url, timeout=timeout)
            if resp.status_code in RETRY_STATUS_CODES and attempt < retries - 1:
                if resp.status_code == 429:
                    # Respect server-specified backoff; fall back to exponential
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else base_delay * (2 ** attempt)
                    await asyncio.sleep(max(wait, base_delay * (2 ** attempt)))
                else:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                continue
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
            if attempt >= retries - 1:
                raise
            await asyncio.sleep(base_delay * (2 ** attempt))
    raise RuntimeError("Retry loop exhausted")


async def scrape_commons_coordinates_async(
    client: httpx.AsyncClient,
    category_url: str,
    max_pages: int = 12,
    timeout: float = 20.0,
) -> CommonsCoordinateResult:
    queue: deque[str] = deque([category_url.replace("http://", "https://")])
    seen: set[str] = set()

    while queue and len(seen) < max_pages:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)

        resp = await _fetch_with_retries(client, url=url, timeout=timeout)
        resolved = str(resp.url)
        if resolved not in seen and len(seen) < max_pages:
            queue.appendleft(resolved)

        soup = BeautifulSoup(resp.text, "html.parser")
        geohack = _find_geohack_link(soup, resolved)
        if geohack:
            lat, lon = parse_geohack_url(geohack)
            return CommonsCoordinateResult(category_url, resolved, geohack, lat, lon)

        for next_url in _extract_candidate_links(soup, resolved):
            if next_url not in seen and next_url not in queue:
                queue.append(next_url)

    raise ValueError(
        f"No GeoHack coordinates found after scanning {len(seen)} page(s) for {category_url}"
    )


async def scrape_commons_meta_async(
    client: httpx.AsyncClient,
    category_url: str,
    max_pages: int = 12,
    timeout: float = 20.0,
) -> dict:
    """Scrape poi_name and wikidata_id from a Wikimedia Commons category page.

    Follows wiki-level redirects (e.g. renamed/moved categories) the same way
    the geohack scraper does, so encoding-mangled URLs that redirect to the real
    page are handled correctly.
    """
    queue: deque[str] = deque([category_url.replace("http://", "https://")])
    seen: set[str] = set()
    best: dict = {"wikimedia_url": category_url, "poi_name": None, "wikidata_id": None, "instance_tag": None}

    while queue and len(seen) < max_pages:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)

        resp = await _fetch_with_retries(client, url, timeout)
        resolved = str(resp.url)
        if resolved not in seen and len(seen) < max_pages:
            queue.appendleft(resolved)

        soup = BeautifulSoup(resp.text, "html.parser")

        h1 = soup.find("h1", id="firstHeading")
        if h1:
            raw = h1.get_text(strip=True)
            poi_name = re.sub(r"^Category:\s*", "", raw, flags=re.IGNORECASE)
        else:
            poi_name = None

        wikidata_id = None
        for a in soup.select('a[href*="wikidata.org"]'):
            m = re.search(r"/(Q\d+)(?:[^0-9]|$)", a.get("href", ""))
            if m:
                wikidata_id = m.group(1)
                break

        # Extract instance/subclass labels from the Wikidata infobox rendered on the page
        instance_tags: list[str] = []
        infobox = soup.select_one("#wdinfobox")
        if infobox:
            target_rows = {"instance of", "subclass of"}
            for row in infobox.select("tr"):
                th = row.select_one("th.wikidatainfobox-lcell")
                if th and th.get_text(strip=True).lower() in target_rows:
                    td = row.select_one("td")
                    if td:
                        instance_tags.extend(
                            a.get_text(strip=True) for a in td.select("a") if a.get_text(strip=True)
                        )

        best["wikimedia_url"] = resolved
        if poi_name:
            best["poi_name"] = poi_name
        if wikidata_id:
            best["wikidata_id"] = wikidata_id
        if instance_tags:
            # best["instance_tag"] = json.dumps(instance_tags, ensure_ascii=False)
            best["instance_tag"] = ",".join(instance_tags)

        if best["poi_name"] and best["wikidata_id"]:
            return best

        for next_url in _extract_candidate_links(soup, resolved):
            if next_url not in seen and next_url not in queue:
                queue.append(next_url)

    return best


async def scrape_wikidata_async(
    client: httpx.AsyncClient,
    wikidata_id: str,
    sem: asyncio.Semaphore,
    timeout: float = 20.0,
) -> dict:
    """Fetch multilingual name tags and OSM ID from the Wikidata API."""
    async with sem:
        params = urlencode({
            "action": "wbgetentities",
            "ids": wikidata_id,
            "format": "json",
            "props": "labels|aliases|claims",
        })
        resp = await _fetch_with_retries(client, f"{WIKIDATA_API}?{params}", timeout)
        data = resp.json()

        entity = data.get("entities", {}).get(wikidata_id, {})

        # Merge labels and aliases per language: {lang: [primary_name, alias1, ...]}
        name_tags: dict[str, list[str]] = {}
        for lang, info in entity.get("labels", {}).items():
            name_tags[lang] = [info["value"]]
        for lang, alias_list in entity.get("aliases", {}).items():
            name_tags.setdefault(lang, []).extend(a["value"] for a in alias_list)

        claims = entity.get("claims", {})
        osm_id = None
        for prop, osm_type in WIKIDATA_OSM_PROPS:
            claim_list = claims.get(prop, [])
            if claim_list:
                val = claim_list[0].get("mainsnak", {}).get("datavalue", {}).get("value")
                if val is not None:
                    osm_id = f"{osm_type}/{val}"
                    break

        return {
            "poi_name_tags": json.dumps(name_tags, ensure_ascii=False) if name_tags else None,
            "osm_id": osm_id,
        }


def _build_fieldnames(scrape_modes: set[str], input_columns: list[str]) -> list[str]:
    """Start from the input CSV's own columns (unchanged names/order), then append
    only the derived columns this run will produce that aren't already present."""
    do_all = "all" in scrape_modes
    fields = list(input_columns)

    def add(*names: str) -> None:
        for name in names:
            if name not in fields:
                fields.append(name)

    if do_all or "geohack" in scrape_modes:
        add("geohack_url", "latitude", "longitude")
    if do_all or "wikimedia" in scrape_modes or "osm" in scrape_modes:
        add("poi_name", "wikidata_id", "instance_tag")
    if do_all or "osm" in scrape_modes:
        add("poi_name_tags", "osm_id")
    add("error")
    return fields


async def _scrape_one_with_semaphore(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    input_url: str,
    scrape_url: str,
    scrape_modes: set[str],
    wikidata_sem: asyncio.Semaphore,
    max_pages: int,
    timeout: float,
    delay: float,
) -> dict:
    async with sem:
        result: dict = {"input_url": input_url, "error": None}
        try:
            if delay > 0:
                await asyncio.sleep(delay)

            do_all = "all" in scrape_modes
            do_geohack = do_all or "geohack" in scrape_modes
            # osm requires wikidata_id, so always fetch wikimedia when osm is selected
            do_wikimedia = do_all or "wikimedia" in scrape_modes or "osm" in scrape_modes
            do_osm = do_all or "osm" in scrape_modes

            if do_geohack:
                r = await scrape_commons_coordinates_async(
                    client, scrape_url, max_pages=max_pages, timeout=timeout
                )
                result.update({
                    "wikimedia_url": r.resolved_url,
                    "geohack_url": r.geohack_url,
                    "latitude": r.latitude,
                    "longitude": r.longitude,
                })

            if do_wikimedia:
                meta = await scrape_commons_meta_async(client, scrape_url, max_pages=max_pages, timeout=timeout)
                result.setdefault("wikimedia_url", meta["wikimedia_url"])
                result["poi_name"] = meta["poi_name"]
                result["wikidata_id"] = meta["wikidata_id"]
                result["instance_tag"] = meta["instance_tag"]

            if do_osm:
                wikidata_id = result.get("wikidata_id")
                if wikidata_id:
                    wd = await scrape_wikidata_async(client, wikidata_id, wikidata_sem, timeout)
                    result["poi_name_tags"] = wd["poi_name_tags"]
                    result["osm_id"] = wd["osm_id"]
                else:
                    result["poi_name_tags"] = None
                    result["osm_id"] = None

        except Exception as e:
            result["error"] = str(e)

        return result


async def scrape_many(
    url_pairs: list[tuple[str, str]],
    scrape_modes: set[str],
    concurrency: int = 20,
    wikidata_concurrency: int = 5,
    max_pages: int = 12,
    timeout: float = 20.0,
    delay: float = 1.0,
) -> AsyncIterator[dict]:
    sem = asyncio.Semaphore(concurrency)
    wikidata_sem = asyncio.Semaphore(wikidata_concurrency)
    limits = httpx.Limits(
        max_connections=concurrency * 2, max_keepalive_connections=concurrency
    )

    async with httpx.AsyncClient(
        headers=DEFAULT_HEADERS, follow_redirects=True, limits=limits, http2=True
    ) as client:
        tasks = [
            asyncio.create_task(
                _scrape_one_with_semaphore(
                    client, sem, input_url, scrape_url, scrape_modes, wikidata_sem,
                    max_pages=max_pages, timeout=timeout, delay=delay,
                )
            )
            for input_url, scrape_url in url_pairs
        ]

        success_count = 0
        error_count = 0
        modes_str = "+".join(sorted(scrape_modes))
        with tqdm(total=len(tasks), desc=f"Scraping [{modes_str}]", unit="url") as pbar:
            for fut in asyncio.as_completed(tasks):
                item = await fut
                if item.get("error"):
                    error_count += 1
                else:
                    success_count += 1
                pbar.set_postfix(success=success_count, error=error_count)
                pbar.update(1)
                yield item


def _dedupe_output(output_path: str, fieldnames: list[str]) -> None:
    """Collapse duplicate rows per wikimedia_url left behind by resumed/retried runs.

    Retries are appended rather than overwritten in place, so a landmark that failed
    once and later succeeded ends up with multiple rows sharing the same wikimedia_url.
    For each wikimedia_url, keep the most recent row without an error, falling back to
    the most recent error row if every attempt failed.
    """
    existing = pl.read_csv(output_path, infer_schema_length=0)
    if "wikimedia_url" not in existing.columns or existing.is_empty():
        return

    ok = (
        pl.col("error").is_null() | (pl.col("error") == "")
        if "error" in existing.columns
        else pl.lit(True)
    )

    deduped = (
        existing
        .with_row_index("_row_idx")
        .with_columns(ok.alias("_ok"))
        .sort(["wikimedia_url", "_ok", "_row_idx"])
        .unique(subset=["wikimedia_url"], keep="last")
        .sort("_row_idx")
    )
    deduped = deduped.select([c for c in fieldnames if c in deduped.columns])
    deduped.write_csv(output_path)


async def main(args):
    scrape_modes = set(args.scrape)

    df = pl.read_csv(args.data_path)

    # Deduplicate by landmark_id before scraping — multiple rows may share the same landmark
    df = df.unique(subset=["landmark_id"], keep="first", maintain_order=True)
    print(f"Loaded {len(df)} unique landmarks after dedup by landmark_id")

    input_urls = df["wikimedia_url"].to_list()
    # Keep every original input column, keyed by wikimedia_url, so it passes through untouched.
    url_meta = {row["wikimedia_url"]: row for row in df.to_dicts()}
    url_pairs = [(url, url) for url in input_urls]

    fieldnames = _build_fieldnames(scrape_modes, list(df.columns))

    done_urls: set[str] = set()
    if os.path.exists(args.output_path):
        existing = pl.read_csv(args.output_path, infer_schema_length=0)
        if "wikimedia_url" in existing.columns:
            if "error" in existing.columns:
                succeeded = existing.filter(pl.col("error").is_null() | (pl.col("error") == ""))
            else:
                succeeded = existing
            done_urls = set(succeeded["wikimedia_url"].to_list())
            error_count = len(existing) - len(succeeded)
            print(f"Resuming: skipping {len(done_urls)} succeeded, retrying {error_count} errored URLs")
        else:
            print("Output file exists but has no 'wikimedia_url' column — starting fresh (output will be appended)")

    pairs_to_scrape = [(inp, scr) for inp, scr in url_pairs if inp not in done_urls]
    if not pairs_to_scrape:
        print("All URLs already scraped.")
    else:
        mode = "a" if done_urls else "w"
        with open(args.output_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not done_urls:
                writer.writeheader()

            async for res in scrape_many(
                pairs_to_scrape,
                scrape_modes=scrape_modes,
                concurrency=args.worker,
                wikidata_concurrency=args.wikidata_worker,
                delay=args.delay,
            ):
                meta = url_meta.get(res["input_url"], {})
                writer.writerow({**meta, **res})
                f.flush()

    _dedupe_output(args.output_path, fieldnames)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--worker", type=int, default=50)
    parser.add_argument("--wikidata-worker", type=int, default=5, help="Max concurrent Wikidata API requests")
    parser.add_argument("--delay", type=float, default=1.0, help="Per-request delay in seconds")
    parser.add_argument(
        "--scrape",
        nargs="+",
        choices=["geohack", "wikimedia", "osm", "all"],
        default=["all"],
        metavar="MODE",
        help=(
            "Which data sources to scrape (space-separated). "
            "geohack: lat/lon coordinates. "
            "wikimedia: poi_name and wikidata_id from Commons. "
            "osm: osm_id and poi_name_tags from Wikidata (implies wikimedia). "
            "all: everything. "
            "Default: geohack."
        ),
    )
    args = parser.parse_args()

    asyncio.run(main(args))
