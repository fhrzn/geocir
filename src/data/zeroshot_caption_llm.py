import argparse
import asyncio
import base64
import csv
import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import polars as pl
from openai import (
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    UnprocessableEntityError,
)
from PIL import Image, ImageFile
from tqdm import tqdm

# tolerate slightly corrupt / truncated JPEGs from the crawl instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 300_000_000

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
# deterministic client errors - retrying these never helps, so fail fast
NON_RETRYABLE = (
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
)
_SENTINEL = object()

SYSTEM_PROMPT = """\
You write ONE short visual caption per image, to be encoded by a CLIP text encoder (~77-token limit).
You are given the landmark CATEGORY and the COUNTRY for the image.
Rules:
- Describe what is visible: architectural style, materials, colours, surroundings, notable features.
- Mention the given category and country naturally in the sentence
  (e.g. "A weathered stone bridge over a river gorge, a heritage site in Norway").
- Do NOT name the specific landmark or use proper nouns other than the country.
- A single sentence, at most 30 words. Output only the caption text - no quotes, no preamble.\
"""

_LEAD_RE = re.compile(
    r"^(caption\s*[:\-]\s*|here(?:'s| is)[^:]*:\s*|(?:this|the)\s+image\s+"
    r"(?:shows|depicts|features|is|presents)\s+)",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# caption post-processing (string-only, no tokenizer)                         #
# --------------------------------------------------------------------------- #
def tidy_caption(text: str) -> str:
    text = _WS_RE.sub(" ", text).strip().strip("\"'“”`")
    text = _LEAD_RE.sub("", text).strip()
    text = text.split("\n", 1)[0].strip()  # keep only the first line the model emitted
    if text:
        text = text[0].upper() + text[1:]
    return text


def _finish(s: str) -> str:
    s = s.rstrip(" ,;:-")
    if s and s[-1] not in ".!?":
        s += "."
    return s


def finalize_caption(text: str, max_words: int) -> tuple[str, int, str]:
    """Return (caption, n_words, truncated) - word cap only, cheap."""
    text = tidy_caption(text)
    if not text:
        return "", 0, "no"
    words = text.split()
    if len(words) <= max_words:
        return text, len(words), "no"
    return _finish(" ".join(words[:max_words])), max_words, "yes"


# --------------------------------------------------------------------------- #
# image loading (runs in worker threads)                                      #
# --------------------------------------------------------------------------- #
def encode_image(path: str, max_size: int, quality: int) -> str:
    with Image.open(path) as im:
        # draft() lets libjpeg decode straight to a reduced scale -> big speedup
        im.draft("RGB", (max_size, max_size))
        im = im.convert("RGB")
        im.thumbnail((max_size, max_size), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# work items                                                                  #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Job:
    id: str
    path: str
    country: str
    category: str


@dataclass(slots=True)
class Prepared:
    job: Job
    b64: str | None
    error: str | None


# --------------------------------------------------------------------------- #
# stages                                                                      #
# --------------------------------------------------------------------------- #
async def loader_worker(job_iter, out_q: asyncio.Queue, pool: ThreadPoolExecutor,
                        max_size: int, quality: int, img_retries: int) -> None:
    loop = asyncio.get_running_loop()
    for job in job_iter:  # next() on a shared iterator is atomic between awaits
        err = "unknown"
        for attempt in range(img_retries + 1):
            try:
                b64 = await loop.run_in_executor(
                    pool, encode_image, job.path, max_size, quality
                )
                await out_q.put(Prepared(job, b64, None))
                err = None
                break
            except FileNotFoundError:
                err = "file not found"  # deterministic - do not retry
                break
            except Exception as e:  # noqa: BLE001 - transient NAS read / decode hiccup
                err = f"decode: {type(e).__name__}: {e}"
                if attempt < img_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
        if err is not None:
            await out_q.put(Prepared(job, None, err))


def build_messages(prep: Prepared) -> list[dict]:
    lines = []
    if prep.job.category:
        lines.append(f"Category: {prep.job.category}")
    if prep.job.country:
        lines.append(f"Country: {prep.job.country}")
    lines.append("Write the caption.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{prep.b64}"}},
                {"type": "text", "text": "\n".join(lines)},
            ],
        },
    ]


async def caption_call(client: AsyncOpenAI, model: str, prep: Prepared,
                       max_words: int, gen_tokens: int, retries: int) -> tuple[str, int, str]:
    """Call the model, retrying transient failures and empty output with backoff."""
    messages = build_messages(prep)
    delay = 1.0
    last_err = "unknown"
    for attempt in range(retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=gen_tokens, temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            cap, nwords, trunc = finalize_caption(raw, max_words)
            if cap:
                return cap, nwords, trunc
            last_err = "empty response" if not raw else "empty after tidy"
        except NON_RETRYABLE as e:  # malformed request etc. - will never succeed
            return f"[error] {type(e).__name__}: {e}", 0, "no"
        except Exception as e:  # noqa: BLE001 - timeout / conn / 5xx / rate limit / parse: retry
            last_err = f"{type(e).__name__}: {e}"
        if attempt < retries:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 20.0)
    return f"[error] {last_err} (after {retries + 1} attempts)", 0, "no"


async def infer_worker(in_q: asyncio.Queue, res_q: asyncio.Queue, client: AsyncOpenAI,
                       model: str, id_col: str, max_words: int,
                       gen_tokens: int, retries: int) -> None:
    while True:
        prep = await in_q.get()
        if prep is _SENTINEL:
            in_q.task_done()
            return
        if prep.error:
            rec = {id_col: prep.job.id, "caption": f"[error] {prep.error}",
                   "caption_words": 0, "caption_truncated": "no"}
        else:
            cap, nwords, trunc = await caption_call(
                client, model, prep, max_words, gen_tokens, retries
            )
            rec = {id_col: prep.job.id, "caption": cap,
                   "caption_words": nwords, "caption_truncated": trunc}
        await res_q.put(rec)
        in_q.task_done()


async def writer_worker(res_q: asyncio.Queue, path: str, fieldnames: list[str],
                        write_header: bool, pbar: tqdm, flush_every: int) -> tuple[int, int]:
    ok = err = since_flush = 0
    with open(path, "a", newline="") as f:  # noqa: ASYNC230 - long-lived append handle; writes are cheap vs inference
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        while True:
            rec = await res_q.get()
            if rec is _SENTINEL:
                f.flush()
                res_q.task_done()
                return ok, err
            w.writerow(rec)
            if str(rec["caption"]).startswith("[error]"):
                err += 1
            else:
                ok += 1
            since_flush += 1
            if since_flush >= flush_every:
                f.flush()
                os.fsync(f.fileno())
                since_flush = 0
            pbar.update(1)
            if (ok + err) % 50 == 0:
                pbar.set_postfix(ok=ok, err=err, refresh=False)
            res_q.task_done()


# --------------------------------------------------------------------------- #
# orchestration                                                               #
# --------------------------------------------------------------------------- #
def build_jobs(args) -> tuple[list[Job], int]:
    df = pl.read_csv(args.data_path, infer_schema_length=20000)
    df = df.with_columns(pl.col(args.id_col).cast(pl.String))

    if args.num_shards > 1:
        df = (
            df.with_row_index("_ridx")
            .filter(pl.col("_ridx") % args.num_shards == args.shard_id)
            .drop("_ridx")
        )
        print(f"shard {args.shard_id}/{args.num_shards}: {df.height:,} rows")

    if args.sample_n and args.sample_n < df.height:
        df = df.sample(n=args.sample_n, seed=args.sample_seed, shuffle=True)
        print(f"random sample: {df.height:,} rows (seed={args.sample_seed})")

    has_country = args.country_col in df.columns
    has_category = args.category_col in df.columns
    has_src = args.src_col in df.columns
    if not has_country:
        print(f"[warn] no '{args.country_col}' column; captions omit country context")
    if not has_category:
        print(f"[warn] no '{args.category_col}' column; captions omit category context")

    done, n_prior_errors = load_done_ids(args.output_path, args.id_col, args.retry_errors)

    jobs: list[Job] = []
    for r in df.iter_rows(named=True):
        _id = r[args.id_col]
        if _id in done:
            continue
        src = (r[args.src_col] if has_src else "") or ""
        country = (r[args.country_col] if has_country else "") or ""
        category = (r[args.category_col] if has_category else "") or ""
        jobs.append(Job(
            _id,
            os.path.join(args.img_root, src, f"{_id}.jpg"),
            str(country),
            str(category),
        ))

    if args.limit:
        jobs = jobs[: args.limit]
    return jobs, n_prior_errors


def load_done_ids(output_path: str, id_col: str, retry_errors: bool) -> tuple[set[str], int]:
    """Return (ids to skip, count of prior [error] rows that will be re-attempted)."""
    if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
        return set(), 0
    try:
        df = pl.read_csv(
            output_path, columns=[id_col, "caption"],
            schema_overrides={id_col: pl.String}, ignore_errors=True,
        ).drop_nulls(id_col)
        is_err = df["caption"].cast(pl.String).fill_null("").str.starts_with("[error]")
        n_err = int(is_err.sum())
        if retry_errors:
            done = set(df.filter(~is_err)[id_col].to_list())
        else:
            done = set(df[id_col].to_list())
            n_err = 0
    except Exception as e:  # noqa: BLE001
        print(f"[warn] fast resume parse failed ({e}); scanning with csv reader")
        done, n_err = set(), 0
        with open(output_path, newline="") as f:
            for row in csv.DictReader(f):
                rid = row.get(id_col)
                if not rid:
                    continue
                if retry_errors and str(row.get("caption", "")).startswith("[error]"):
                    n_err += 1
                else:
                    done.add(rid)
    msg = f"resume: {len(done):,} ids done in {output_path}"
    if n_err:
        msg += f"; {n_err:,} prior [error] rows will be retried"
    print(msg)
    return done, n_err


def dedupe_output(output_path: str, id_col: str) -> None:
    """Collapse duplicate ids from retried rows, preferring a successful caption."""
    df = pl.read_csv(output_path, infer_schema_length=20000,
                     schema_overrides={id_col: pl.String})
    before = df.height
    df = (
        df.with_columns(
            pl.col("caption").cast(pl.String).fill_null("").str.starts_with("[error]").alias("_e")
        )
        .sort("_e", descending=True)  # errors first -> keep="last" prefers a real caption
        .unique(subset=[id_col], keep="last", maintain_order=True)
        .drop("_e")
    )
    if df.height != before:
        df.write_csv(output_path)
        print(f"deduped output: {before:,} -> {df.height:,} rows")


async def run(args) -> None:
    jobs, n_prior_errors = build_jobs(args)
    if not jobs:
        print("nothing to do - all rows already processed.")
        return
    print(f"to process: {len(jobs):,} images "
          f"| infer-concurrency={args.infer_concurrency} loader-workers={args.loader_workers}")

    client = AsyncOpenAI(
        base_url=args.base_url, api_key="EMPTY",
        timeout=args.request_timeout, max_retries=0,  # we retry ourselves
    )
    fieldnames = [args.id_col, "caption", "caption_words", "caption_truncated"]
    write_header = not (os.path.exists(args.output_path) and os.path.getsize(args.output_path) > 0)

    in_q: asyncio.Queue = asyncio.Queue(maxsize=args.prefetch)
    res_q: asyncio.Queue = asyncio.Queue(maxsize=args.prefetch)
    job_iter = iter(jobs)
    pool = ThreadPoolExecutor(max_workers=args.loader_workers, thread_name_prefix="img")
    started = time.monotonic()
    ok = err = 0

    with tqdm(total=len(jobs), desc="caption", unit="img", smoothing=0.05) as pbar:
        loaders = [
            asyncio.create_task(
                loader_worker(job_iter, in_q, pool, args.img_max_size,
                              args.jpeg_quality, args.img_retries)
            )
            for _ in range(args.loader_workers)
        ]
        infers = [
            asyncio.create_task(
                infer_worker(in_q, res_q, client, args.model, args.id_col,
                             args.max_words, args.gen_tokens, args.max_retries)
            )
            for _ in range(args.infer_concurrency)
        ]
        writer = asyncio.create_task(
            writer_worker(res_q, args.output_path, fieldnames, write_header,
                          pbar, args.flush_every)
        )

        try:
            await asyncio.gather(*loaders)          # every image decoded & enqueued
            for _ in range(args.infer_concurrency):
                await in_q.put(_SENTINEL)
            await asyncio.gather(*infers)           # every request answered
            await res_q.put(_SENTINEL)
            ok, err = await writer
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            await client.close()

    if n_prior_errors:
        dedupe_output(args.output_path, args.id_col)

    elapsed = time.monotonic() - started
    rate = len(jobs) / elapsed if elapsed else 0
    print(f"done - {ok:,} ok, {err:,} errors in {elapsed / 3600:.2f} h "
          f"({rate:.1f} img/s) -> {args.output_path}")
    if err:
        print(f"{err:,} rows still failed - rerun the same command to retry them.")
    print(f"join back with: pl.read_csv(source).join(pl.read_csv(output), on='{args.id_col}')")


async def smoke_test(args) -> None:
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY",
                         timeout=args.request_timeout, max_retries=0)
    jobs = build_jobs(args)[0][: args.smoke_n]
    if not jobs:
        print("no rows to smoke-test.")
        return
    print(f"smoke test - {len(jobs)} images\n")
    loop = asyncio.get_running_loop()
    for job in jobs:
        try:
            b64 = await loop.run_in_executor(
                None, encode_image, job.path, args.img_max_size, args.jpeg_quality
            )
            cap, nwords, trunc = await caption_call(
                client, args.model, Prepared(job, b64, None),
                args.max_words, args.gen_tokens, args.max_retries,
            )
        except Exception as e:  # noqa: BLE001
            cap, nwords, trunc = f"[error] {e}", 0, "no"
        print(f"id       : {job.id}")
        print(f"category : {job.category or '-'}")
        print(f"country  : {job.country or '-'}")
        print(f"words    : {nwords}  truncated={trunc}")
        print(f"caption  : {cap}\n")
    await client.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Caption-only VLM inference via vLLM (country + category aware)")
    p.add_argument("--base-url", default="http://localhost:3456/v1")
    p.add_argument("--model", default=DEFAULT_MODEL, help="must match vLLM --served-model-name")
    p.add_argument("--data-path", help="source CSV (required)")
    p.add_argument("--output-path", default="./captions.csv")
    p.add_argument("--img-root", default="/mnt/yokoyamalab-nas/gldv2-full")
    p.add_argument("--id-col", default="id")
    p.add_argument("--src-col", default="src")
    p.add_argument("--country-col", default="country")
    p.add_argument("--category-col", default="category")

    p.add_argument("--infer-concurrency", type=int, default=64,
                   help="in-flight chat completions; the vLLM continuous-batching lever")
    p.add_argument("--loader-workers", type=int, default=32,
                   help="threads decoding/resizing images from disk")
    p.add_argument("--prefetch", type=int, default=1024, help="bounded queue depth (backpressure)")
    p.add_argument("--img-max-size", type=int, default=384)
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--gen-tokens", type=int, default=96, help="max new tokens per caption")
    p.add_argument("--max-words", type=int, default=40,
                   help="hard word cap on the caption (CLIP ~77-token safety net)")
    p.add_argument("--max-retries", type=int, default=4,
                   help="per-request retries (backoff) on timeout/5xx/rate-limit/empty output")
    p.add_argument("--img-retries", type=int, default=2,
                   help="retries on a transient image read/decode failure")
    p.add_argument("--retry-errors", action=argparse.BooleanOptionalAction, default=True,
                   help="on resume, re-attempt rows previously written as [error]")
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--flush-every", type=int, default=200, help="rows between fsync of output")

    p.add_argument("--limit", type=int, default=0, help="cap rows this run (after resume filter)")
    p.add_argument("--sample-n", type=int, default=0, help="random-sample N rows before processing")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)

    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-n", type=int, default=6)
    args = p.parse_args(argv)
    if not args.data_path:
        p.error("--data-path is required")
    if not 0 <= args.shard_id < args.num_shards:
        p.error("--shard-id must be in [0, --num-shards)")
    return args


if __name__ == "__main__":
    _args = parse_args()
    asyncio.run(smoke_test(_args) if _args.smoke_test else run(_args))
