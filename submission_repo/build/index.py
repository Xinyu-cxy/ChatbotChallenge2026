"""Indexer. Workshop 1 block 4.

Chunk the scraped text and write it to the store. Run this file
directly, after build/scrape.py, to (re)build data/chroma.

Chunking is heading-first: scrape.py keeps each section heading in
page["text"] as a markdown-style "## Heading" line, split_sections()
cuts the page at those lines, and every chunk carries its heading (or
the page title) as a prefix. Fixed-size character splitting with
overlap is only the fallback for sections too long to fit one chunk.

Store metadata now. Level 4 questions need to filter by year and page
type, and adding a field later means rebuilding everything.
"""
import json
import re
from pathlib import Path

from build.scrape import classify_page, extract_page_year
from bot.store import add_to_store, get_store


# Tuned by answer-span recall (9 dev questions + 47 synthetic verbatim
# QA, top-5 cosine retrieval) over the 806-page corpus; see
# build/_tune_chunk.py. Size sweep peaked clearly at 1000 (0.742 vs
# 0.697 at 800 and 0.711 at 1200); at size 1000, overlap 125 beat
# overlap 0 (0.742 vs 0.734). ~1000 chars also fits the data: median
# section is 247 chars, so most sections stay whole and only ~14%
# fall through to the character split.
CHUNK_SIZE, OVERLAP = 1000, 125

HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")

# Some WordPress pages wrap whole paragraphs in <h2>/<h3> tags, producing
# "headings" thousands of characters long. Those are content, not labels:
# prepending one to every chunk of its section would bury the actual text.
MAX_HEADING = 200


def split_sections(text: str) -> list[tuple[str, str]]:
    """Cut page text into (heading, body) sections at heading lines.

    Text before the first heading gets heading "", and pages with no
    heading markers at all (e.g. old flat pages.json files) come back as
    a single section, so callers always fall through to the character
    split rather than crashing.
    """
    sections: list[tuple[str, str]] = []
    heading, buf = "", []
    for line in text.split("\n"):
        match = HEADING_RE.match(line.strip())
        if match and len(match.group(1)) > MAX_HEADING:
            buf.append(match.group(1))   # mistagged paragraph: keep as body
            continue
        if match:
            body = " ".join(" ".join(buf).split())
            if body:
                sections.append((heading, body))
                heading = match.group(1)
            elif len(f"{heading} > {match.group(1)}") <= MAX_HEADING and heading:
                # Heading with no body of its own: keep it as a
                # breadcrumb for the next one ("Facilities > 3D Printers")
                # instead of dropping the parent.
                heading = f"{heading} > {match.group(1)}"
            else:
                heading = match.group(1)
            buf = []
        else:
            buf.append(line)
    body = " ".join(" ".join(buf).split())
    if body:
        sections.append((heading, body))
    return sections


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP,
          title: str = "") -> list[str]:
    """Heading-first chunks; character split with overlap is the fallback.

    Why headings first: a flat character split cuts wherever the counter
    lands, so a chunk routinely holds facts with no trace of the section
    that gives them meaning ("27 February 2026" with no "Funding Scheme"
    nearby). Sections keep each fact under its own label, and prepending
    the heading to every chunk means a chunk lifted out of the page by
    retrieval still says what it is about.

    Three cases per section:
      - small sections are packed together up to `size`, each with its
        own "Heading: " prefix inline, so siblings stay adjacent;
      - sections longer than `size` fall back to fixed-size character
        windows;
      - the character windows keep `overlap`, because a fact split
        across a boundary is lost: a date at char 998 and its event
        name at char 1002 land in different chunks and neither answers
        the question.

    Pages with no headings use `title` as the prefix instead, and old
    flat text (no heading lines) degrades to the pure character split.
    """
    chunks: list[str] = []
    packed = ""

    def flush() -> None:
        nonlocal packed
        if packed:
            chunks.append(packed)
        packed = ""

    for heading, body in split_sections(text):
        label = heading or title
        prefix = f"{label}: " if label else ""
        budget = max(size - len(prefix), 100)   # prefix counts toward size
        if len(body) <= budget:
            piece = f"{prefix}{body}"
            if packed and len(packed) + 1 + len(piece) > size:
                flush()
            packed = f"{packed} {piece}" if packed else piece
            continue
        # Section too long for one chunk: character windows with overlap.
        flush()
        step = max(budget - min(overlap, budget // 2), 1)
        for i in range(0, len(body), step):
            part = body[i:i + budget]
            if part.strip():
                chunks.append(f"{prefix}{part}")
    flush()
    return chunks


def build_index(pages: list[dict], reset: bool = True):
    texts, metas = [], []
    for page in pages:
        year = page.get("year")
        if year is None:
            year = extract_page_year(page)
        type_ = page.get("page_type") or classify_page(page)
        pieces = chunk(page["text"], title=page.get("title", ""))
        for i, piece in enumerate(pieces):
            metadata = {
                "url":      page["url"],
                "title":    page.get("title", ""),
                "position": i,
                "kind": "text",
                "page_type": type_,
            }
            if year is not None:
                metadata["year"] = year
            texts.append(piece)
            metas.append(metadata)
    store = get_store(reset=reset)
    add_to_store(store, texts, metas)
    print(f"indexed {len(texts)} chunks")
    return store


if __name__ == "__main__":
    pages = json.loads(Path("data/pages.json").read_text(encoding="utf-8"))
    build_index(pages)
