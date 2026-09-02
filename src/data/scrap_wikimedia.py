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
class CommonsPageResult:
    input_url: str
    resolved_url: str
    geohack_url: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    poi_name: Optional[str] = None
    wikidata_id: Optional[str] = None
    instance_tag: Optional[str] = None


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


def _parse_commons_page(html: str, base_url: str) -> dict:
    """Parse a Commons category page once, pulling every field we might need.

    Pure-CPU work (lxml tree build + selectors). Run via ``asyncio.to_thread`` so
    it overlaps network I/O instead of blocking the event loop.
    """
    soup = BeautifulSoup(html, "lxml")

    geohack_url = _find_geohack_link(soup, base_url)

    h1 = soup.find("h1", id="firstHeading")
    poi_name = (
        re.sub(r"^Category:\s*", "", h1.get_text(strip=True), flags=re.IGNORECASE)
        if h1
        else None
    )

    wikidata_id = None
    for a in soup.select('a[href*="wikidata.org"]'):
        m = re.search(r"/(Q\d+)(?:[^0-9]|$)", a.get("href", ""))
        if m:
            wikidata_id = m.group(1)
            break

    # Instance/subclass labels from the Wikidata infobox rendered on the page
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

    return {
        "geohack_url": geohack_url,
        "poi_name": poi_name,
        "wikidata_id": wikidata_id,
        "instance_tag": ",".join(instance_tags) if instance_tags else None,
        "candidate_links": _extract_candidate_links(soup, base_url),
    }


async def scrape_commons_page_async(
    client: httpx.AsyncClient,
    category_url: str,
    *,
    need_geohack: bool,
    need_meta: bool,
    max_pages: int = 4,
    timeout: float = 20.0,
) -> CommonsPageResult:
    """Single BFS crawl of a Commons category page for coordinates and/or metadata.

    Fetches each page once and extracts everything from the same parse, following
    wiki-level redirects (renamed/moved categories, encoding-mangled URLs) via the
    same candidate-link heuristics. Raises ``ValueError`` if ``need_geohack`` is set
    but no GeoHack coordinates are found; metadata fields are best-effort.
    """
    queue: deque[str] = deque([category_url.replace("http://", "https://")])
    seen: set[str] = set()
    result = CommonsPageResult(input_url=category_url, resolved_url=category_url)

    while queue and len(seen) < max_pages:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)

        resp = await _fetch_with_retries(client, url, timeout=timeout)
        resolved = str(resp.url)
        result.resolved_url = resolved
        if resolved not in seen and len(seen) < max_pages:
            queue.appendleft(resolved)

        parsed = await asyncio.to_thread(_parse_commons_page, resp.text, resolved)

        if need_geohack and result.latitude is None and parsed["geohack_url"]:
            lat, lon = parse_geohack_url(parsed["geohack_url"])
            result.geohack_url = parsed["geohack_url"]
            result.latitude = lat
            result.longitude = lon

        if need_meta:
            if parsed["poi_name"]:
                result.poi_name = parsed["poi_name"]
            if parsed["wikidata_id"]:
                result.wikidata_id = parsed["wikidata_id"]
            if parsed["instance_tag"]:
                result.instance_tag = parsed["instance_tag"]

        geohack_ok = (not need_geohack) or result.latitude is not None
        meta_ok = (not need_meta) or (result.poi_name and result.wikidata_id)
        if geohack_ok and meta_ok:
            return result

        for next_url in parsed["candidate_links"]:
            if next_url not in seen and next_url not in queue:
                queue.append(next_url)

    if need_geohack and result.latitude is None:
        raise ValueError(
            f"No GeoHack coordinates found after scanning {len(seen)} page(s) for {category_url}"
        )
    return result


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


def _build_fieldnames(
    scrape_modes: set[str], input_columns: list[str], keep_input_url: bool = False
) -> list[str]:
    """Start from the input CSV's own columns (unchanged names/order), then append
    only the derived columns this run will produce that aren't already present."""
    do_all = "all" in scrape_modes
    fields = list(input_columns)

    def add(*names: str) -> None:
        for name in names:
            if name not in fields:
                fields.append(name)

    if keep_input_url:
        # Original input URL, untouched; `wikimedia_url` still holds the resolved URL.
        add("input_wikimedia_url")
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

            if do_geohack or do_wikimedia:
                page = await scrape_commons_page_async(
                    client, scrape_url,
                    need_geohack=do_geohack, need_meta=do_wikimedia,
                    max_pages=max_pages, timeout=timeout,
                )
                result["wikimedia_url"] = page.resolved_url
                if do_geohack:
                    result["geohack_url"] = page.geohack_url
                    result["latitude"] = page.latitude
                    result["longitude"] = page.longitude
                if do_wikimedia:
                    result["poi_name"] = page.poi_name
                    result["wikidata_id"] = page.wikidata_id
                    result["instance_tag"] = page.instance_tag

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
    max_pages: int = 4,
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


def _dedupe_output(output_path: str, fieldnames: list[str], key_col: str = "wikimedia_url") -> None:
    """Collapse duplicate rows per ``key_col`` left behind by resumed/retried runs.

    Retries are appended rather than overwritten in place, so a landmark that failed
    once and later succeeded ends up with multiple rows sharing the same key. For each
    key, keep the most recent row without an error, falling back to the most recent
    error row if every attempt failed.
    """
    existing = pl.read_csv(output_path, infer_schema_length=0)
    if key_col not in existing.columns or existing.is_empty():
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
        .sort([key_col, "_ok", "_row_idx"])
        .unique(subset=[key_col], keep="last")
        .sort("_row_idx")
    )
    deduped = deduped.select([c for c in fieldnames if c in deduped.columns])
    deduped.write_csv(output_path)


async def main(args):
    scrape_modes = set(args.scrape)

    df = pl.read_csv(args.data_path)

    # Deduplicate by wikimedia_url before scraping — the URL is the scrape key and is
    # finer-grained than landmark_id (multiple rows may share the same landmark_id, and
    # a single-column `wikimedia_url` CSV has no landmark_id at all).
    df = df.unique(subset=["wikimedia_url"], keep="first", maintain_order=True)
    print(f"Loaded {len(df)} unique URLs after dedup by wikimedia_url")

    input_urls = df["wikimedia_url"].to_list()
    # Keep every original input column, keyed by wikimedia_url, so it passes through untouched.
    url_meta = {row["wikimedia_url"]: row for row in df.to_dicts()}
    url_pairs = [(url, url) for url in input_urls]

    keep_input_url = args.keep_input_url
    # Which column identifies a row for resume/dedupe: the original input URL when
    # preserved, otherwise the redirect-resolved URL written to `wikimedia_url`.
    key_col = "input_wikimedia_url" if keep_input_url else "wikimedia_url"

    fieldnames = _build_fieldnames(scrape_modes, list(df.columns), keep_input_url)

    done_urls: set[str] = set()
    if os.path.exists(args.output_path):
        existing = pl.read_csv(args.output_path, infer_schema_length=0)
        if key_col in existing.columns:
            if "error" in existing.columns:
                succeeded = existing.filter(pl.col("error").is_null() | (pl.col("error") == ""))
            else:
                succeeded = existing
            done_urls = set(succeeded[key_col].to_list())
            error_count = len(existing) - len(succeeded)
            print(f"Resuming: skipping {len(done_urls)} succeeded, retrying {error_count} errored URLs")
        else:
            print(f"Output file exists but has no '{key_col}' column — starting fresh (output will be appended)")

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
                max_pages=args.max_pages,
                delay=args.delay,
            ):
                meta = url_meta.get(res["input_url"], {})
                row = {**meta, **res}
                if keep_input_url:
                    row["input_wikimedia_url"] = res["input_url"]
                writer.writerow(row)
                f.flush()

    _dedupe_output(args.output_path, fieldnames, key_col)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--worker", type=int, default=50)
    parser.add_argument("--wikidata-worker", type=int, default=5, help="Max concurrent Wikidata API requests")
    parser.add_argument("--delay", type=float, default=1.0, help="Per-request delay in seconds")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=4,
        help="Max Commons pages to crawl per URL while following wiki redirects",
    )
    parser.add_argument(
        "--keep-input-url",
        action="store_true",
        help=(
            "Keep the original input URL in a separate 'input_wikimedia_url' column. "
            "By default 'wikimedia_url' is overwritten with the redirect-resolved URL "
            "and the original is lost. When set, resume/dedupe key on the original URL "
            "instead of the resolved one (so redirected rows are not re-scraped)."
        ),
    )
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
