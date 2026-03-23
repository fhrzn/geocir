import argparse
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
import polars as pl
from bs4 import BeautifulSoup
from tqdm import tqdm
import csv

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
    client: httpx.AsyncClient, url: str, timeout: float, retries: int = 5
) -> httpx.Response:
    delay = 0.5
    for attempt in range(retries):
        try:
            resp = await client.get(url, timeout=timeout)
            if resp.status_code in RETRY_STATUS_CODES and attempt < retries - 1:
                await asyncio.sleep(delay * (2**attempt))
                continue
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
            if attempt >= retries - 1:
                raise
            await asyncio.sleep(delay * (2**attempt))
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


async def _scrape_one_with_semaphore(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    url: str,
    max_pages: int,
    timeout: float,
) -> dict:
    async with sem:
        try:
            await asyncio.sleep(3)
            r = await scrape_commons_coordinates_async(
                client, url, max_pages=max_pages, timeout=timeout
            )
            return {
                "input_url": r.input_url,
                "resolved_url": r.resolved_url,
                "geohack_url": r.geohack_url,
                "latitude": r.latitude,
                "longitude": r.longitude,
                "error": None,
            }
        except Exception as e:
            return {
                "input_url": url,
                "resolved_url": None,
                "geohack_url": None,
                "latitude": None,
                "longitude": None,
                "error": str(e),
            }


async def scrape_many_coordinates(
    urls: list[str], concurrency: int = 20, max_pages: int = 12, timeout: float = 20.0
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(
        max_connections=concurrency * 2, max_keepalive_connections=concurrency
    )

    async with httpx.AsyncClient(
        headers=DEFAULT_HEADERS, follow_redirects=True, limits=limits, http2=True
    ) as client:
        tasks = [
            asyncio.create_task(
                _scrape_one_with_semaphore(
                    client, sem, url, max_pages=max_pages, timeout=timeout
                )
            )
            for url in urls
        ]

        results: list[dict] = []
        success_count = 0
        error_count = 0
        with tqdm(total=len(tasks), desc="Scraping Wikimedia", unit="url") as pbar:
            for fut in asyncio.as_completed(tasks):
                item = await fut
                results.append(item)
                if item.get("error"):
                    error_count += 1
                else:
                    success_count += 1
                pbar.set_postfix(success=success_count, error=error_count)
                pbar.update(1)
        return results
    
async def main(args):
    df = pl.read_csv(args.data_path)
    urls = df["category"].to_list()
    
    results = await scrape_many_coordinates(urls, concurrency=20)

    with open(args.output_path, "w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "landmark_id",
                "input_url",
                "resolved_url",
                "geohack_url",
                "latitude",
                "longitude",
                "error",
            ],
        )

        writer.writeheader()
        for res in results:
            writer.writerow(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--worker", type=int, default=100)
    args = parser.parse_args()
    
    asyncio.run(main(args))
    