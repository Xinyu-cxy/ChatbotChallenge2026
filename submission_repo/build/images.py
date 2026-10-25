"""Image pipeline. STUB. Workshop 2 blocks 2 and 3.

Describe every collected image once, cache the result, and add the
descriptions to the same store as the scraped text. Run this file
directly, after build/scrape.py and build/index.py, to add images to
data/chroma.

Never describe an image while answering a question. Ingestion time is
unlimited; runtime is 30 seconds.
"""
import json
import mimetypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from openai import AuthenticationError

from build.scrape import classify_page, extract_embedded_year
from bot.llm_ import describe_image_bytes
from bot.store import add_to_store, get_store

DESCRIPTION_PROMPT = """
Create a factual, searchable description of the whole image. Inspect the full frame, including the edges and background. 

Write the answer in English using exactly these headings and order:
Location: Try your best to infer the location of this area using signs and write your answer. If it's too hard and you can't think of any answer, write [unknown].
Objects: List every object and its counts. Make sure you cover every item in the image and don't miss even a single one. Create a dictionary of all the names of the objects you discovered and their corresponding counts. Before searching, set the dictionary to empty. Scan the entire picture and if you detect a new item that you haven't seen before, identify whether its name is recorded in the list. If its name is recorded, add the name's count in the dictionary by one. If not, add its name to the dictionary and set its count to 1. Write your answer using the dictionary but don't print your dictionary. Count 3 times, start all ovevr again from count 1 if your new count differs from previous ones. Write each object as 'Name: count' on its own line. Do not add item numbers, bullets, labels, explanations, or any other text; for example, write 'Laptops: 5', not '1. Laptops: 5'.
Visible text: Copy readable text exactly, preserve capitalization. Identify as much texts as possible. Use [unreadable] for texts that you think are completely unidentifiable. Do not infer missing words.
Spatial relations: Describe the positions and relationships between objects in your previous dictionary. Cover as much positions and relationships as possible. These notes do not change the object counts.
Colours & materials: Include only visible, distinguishing details. Record as much as possible.
"""

CACHE_PATH = Path("data/descriptions.json")
FAILURE_PATH = Path("data/failures.json")

# The vision API rejects these outright; never download or send them.
UNSUPPORTED_EXT = {".svg", ".svgz"}
# Transient errors (timeout, SSL, 5xx) are retried across runs, then given up.
MAX_ATTEMPTS = 3
WORKERS = int(os.environ.get("DESCRIBE_WORKERS", "8"))
# Skip anything larger; base64 of huge files just bloats the API call.
MAX_BYTES = 10 * 1024 * 1024

def get_image_year(src: str) -> int | None:
    """Find a year encoded in the image URL or filename."""
    return extract_embedded_year(src)

def get_image_type(u_: str) -> str:
    label = f'{u_}'.lower()
    if "logo" in label:
        return "logo"
    return classify_page({"url": u_, "title": ""})


def _is_permanent(exc: Exception) -> bool:
    """True when retrying can never help (dead URL, unsupported image)."""
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", 0)
        # 4xx except rate-limit/timeout statuses: the resource is simply gone.
        if 400 <= status < 500 and status not in (408, 425, 429):
            return True
    text = str(exc)
    return (
        "invalid_parameter_error" in text
        or "invalid_request_error" in text
        or "image format is illegal" in text
    )


def _guess_mime(src: str, content_type: str | None) -> str:
    if content_type and content_type.lower().startswith("image/"):
        return content_type.split(";")[0].strip()
    ext = Path(urlparse(src).path).suffix.lower()
    return mimetypes.guess_type("image" + ext)[0] or "image/jpeg"


def _describe_one(src: str):
    """Download and describe a single image.

    Returns (src, description | None, error | None, permanent).
    """
    try:
        response = requests.get(src, timeout=(10, 30))
        response.raise_for_status()
        content = response.content
        if len(content) > MAX_BYTES:
            return src, None, f"image too large ({len(content)} bytes)", True
        if not content:
            return src, None, "empty response body", True
        mime = _guess_mime(src, response.headers.get("Content-Type"))
        description = describe_image_bytes(content, mime, DESCRIPTION_PROMPT)
        return src, description, None, False
    except AuthenticationError:
        raise
    except Exception as exc:
        return src, None, f"{type(exc).__name__}: {exc}"[:300], _is_permanent(exc)


def describe_all(images: list[dict]) -> dict:
    """Describe every image once, cache the result, never regenerate.

    Ingestion is free, but not if you redo it every time you change a
    line downstream. Key the cache by image URL. Failed URLs are recorded
    in data/failures.json so reruns never waste time on dead links:
    permanent failures (404, unsupported format) are skipped forever,
    transient ones (timeout, SSL, 5xx) get up to MAX_ATTEMPTS tries.
    """
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    cache = {
        src: text for src, text in cache.items()
        if isinstance(text, str) and text.strip()
    }
    failures = (
        json.loads(FAILURE_PATH.read_text()) if FAILURE_PATH.exists() else {}
    )

    # Deduplicate sources and split into todo vs. known-dead.
    todo, dead = [], 0
    seen = set()
    for im in images:
        src = im["src"]
        if src in seen or src in cache:
            continue
        seen.add(src)
        if Path(urlparse(src).path).suffix.lower() in UNSUPPORTED_EXT:
            failures[src] = {
                "attempts": MAX_ATTEMPTS,
                "permanent": True,
                "error": "unsupported format (svg)",
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            dead += 1
            continue
        record = failures.get(src)
        if record and (
            record.get("permanent") or record.get("attempts", 0) >= MAX_ATTEMPTS
        ):
            dead += 1
            continue
        todo.append(src)

    print(
        f"{len(cache)} cached, {dead} known-failed (skipped), "
        f"{len(todo)} to fetch on {WORKERS} threads"
    )

    lock = threading.Lock()
    state = {"ok": 0, "fail": 0, "since_save": 0}

    def save() -> None:
        CACHE_PATH.parent.mkdir(exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, indent=1))
        FAILURE_PATH.write_text(json.dumps(failures, indent=1))

    def process(src: str) -> None:
        src, description, error, permanent = _describe_one(src)
        with lock:
            if description is not None:
                cache[src] = description
                failures.pop(src, None)
                state["ok"] += 1
            else:
                record = failures.get(src, {})
                record["attempts"] = record.get("attempts", 0) + 1
                record["permanent"] = permanent
                record["error"] = error
                record["time"] = datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )
                failures[src] = record
                state["fail"] += 1
            state["since_save"] += 1
            if state["since_save"] >= 20:
                state["since_save"] = 0
                save()
            done = state["ok"] + state["fail"]
            if done % 10 == 0 or done == len(todo):
                print(
                    f"  progress {done}/{len(todo)} "
                    f"(+{state['ok']} ok, {state['fail']} failed)",
                    flush=True,
                )

    pool = ThreadPoolExecutor(max_workers=WORKERS)
    try:
        futures = [pool.submit(process, src) for src in todo]
        for future in futures:
            future.result()
    except AuthenticationError as exc:
        pool.shutdown(wait=False, cancel_futures=True)
        with lock:
            save()
        print(f"{len(cache)} descriptions cached")
        raise RuntimeError(
            "The image-description API rejected its credentials. "
            "Check DASHSCOPE_API_KEY in submission_repo/.env and ensure "
            "it is a Model Studio key for the configured API endpoint."
        ) from exc
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        print("\ninterrupted; saving progress...")
        with lock:
            save()
        print(f"{len(cache)} descriptions cached")
        raise
    else:
        pool.shutdown()

    save()
    print(f"{len(cache)} descriptions cached")

    retryable = sum(
        1 for r in failures.values()
        if not r.get("permanent") and r.get("attempts", 0) < MAX_ATTEMPTS
    )
    if state["fail"]:
        print(
            f"{state['fail']} images failed this run "
            f"({retryable} still retryable); see {FAILURE_PATH}. "
            "Continuing with the descriptions that succeeded."
        )
    return cache


def index_descriptions(descriptions: dict, images: list[dict]) -> None:
    """Add descriptions to the same store as the scraped text."""
    by_src = {im["src"]: im for im in images}
    texts, metas = [], []
    for src, text in descriptions.items():
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Refusing to index an empty image description: {src}")
        im = by_src.get(src, {})
        # Alt text and captions are already text and cost nothing to index.
        full = " ".join(filter(None, [im.get("alt"), im.get("caption"), text]))
        texts.append(full)
        u_ = im.get("page", src)
        metadata = {
            "url": u_,
            "image": src,
            "kind": "image",
        }
        t_ = get_image_type(u_)
        if t_ == "page":
            t_ = im.get("page_type") or "page"
        metadata["page_type"] = t_
        year_ = get_image_year(src)
        if im.get("year") is not None:
            metadata["year"] = im["year"]

        elif year_ is not None:
            metadata["year"] = year_
        metas.append(metadata)
    store = get_store(reset=False)
    add_to_store(store, texts, metas, ids=[f"img_{i}" for i in range(len(texts))])
    print(f"indexed {len(texts)} image descriptions")


if __name__ == "__main__":
    images = json.loads(Path("data/images.json").read_text())
    descriptions = describe_all(images)
    index_descriptions(descriptions, images)