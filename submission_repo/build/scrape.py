"""Scraper. STUB. Workshop 1 block 4.

Crawl both websites, extract the readable text, and collect image
records in the same pass. Run this file directly to (re)build
data/pages.json and data/images.json.

Check for /sitemap.xml before writing a crawler. If it exists it lists
every page and you can skip the crawl entirely.
"""
import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

SITES = [
    # TODO: the two Inno Wing sites you were given
    "https://innowings.engg.hku.hk",
    "https://innoacademy.engg.hku.hk",
]

def extract_embedded_year(value: str) -> int | None:
    """Find a year, including compact YYYYMM and YYYYMMDD URL tokens."""
    pattern = r"(?<!\d)(20\d{2})(?:(0[1-9]|1[0-2])([0-2]\d|3[01])?)?(?!\d)"
    for match in re.finditer(pattern, value):
        year, month, day = match.groups()
        if day is not None:
            try:
                date(int(year), int(month), int(day))
            except ValueError:
                continue
        return int(year)
    return None


def extract_page_year(page: dict) -> int | None:
    """Find a year explicitly associated with the page."""
    for value in (page.get("url", ""), page.get("title", "")):
        year = extract_embedded_year(value)
        if year is not None:
            return year

    text = page.get("text", "")
    for match in re.finditer(
        r"\b(?:date|year|academic year|ongoing since|exhibition period|"
        r"exhibition started from)\b",
        text,
        re.IGNORECASE,
    ):
        context = text[match.end():match.end() + 80].split(".", 1)[0]
        year = extract_embedded_year(context)
        if year is not None:
            return year
    return None


def classify_page(page: dict) -> str:
    """Infer a useful content type from a page's URL and title."""
    label = f'{page.get("url", "")} {page.get("title", "")}'.lower()
    if "poster" in label:
        return "poster"
    if "workshop" in label:
        return "workshop"
    if any(term in label for term in ("techtalk", "tech talk", "edgetalk", "edge talk")):
        return "techtalk"
    if "exhibition" in label:
        return "exhibition"
    if any(term in label for term in (
        "competition", "challenge", "contest", "robomaster", "robocon",
        "duckietown", "imeche",
    )):
        return "competition"
    if re.search(r"\bcourses?\b|\b(?:engg|comp|elec)\s?\d{4}\b", label):
        return "course"
    if re.search(r"\b(?:gallery|photo(?:s)?|photo album)\b", label):
        return "gallery"
    if any(term in label for term in ("equipment", "facility", "facilities", "venue")):
        return "facility"
    if "event" in label or "activity" in label:
        return "event"
    if any(word in label for word in ("project", "research")):
        return "project"
    return "page"


ASSET_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".zip", ".mp4", ".css", ".js",
)


def is_page(link: str) -> bool:
    """False for asset URLs. Feeding a JPEG to BeautifulSoup raises
    ParserRejectedMarkup full of undecodable bytes; not queueing assets
    at all skips hundreds of pointless fetches and that whole failure
    mode."""
    return not urlparse(link).path.lower().endswith(ASSET_SUFFIXES)


def crawl(start_url: str, max_pages: int = 500) -> list[str]:
    """Return every page URL on the same site as start_url."""
    seen, queue, out = set(), [start_url], []
    domain = urlparse(start_url).netloc

    while queue and len(out) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            html = requests.get(url, timeout=20).text
        except Exception:
            continue
        out.append(url)

        # TODO find the links on this page and add the internal ones to
        # TODO queue, something like:
        # for a in BeautifulSoup(html, "html.parser").select("a[href]"):
        #     link = urljoin(url, a["href"]).split("#")[0]
        #     if urlparse(link).netloc == domain and link not in seen:
        #         queue.append(link)
        for a in BeautifulSoup(html, "html.parser").select("a[href]"):
            link = urljoin(url, a["href"]).split("#")[0]
            if urlparse(link).netloc == domain and link not in seen and is_page(link):
                queue.append(link)
    return out


HEADING_LINE = re.compile(r"^#{1,6} \S")


def text_with_headings(body) -> str:
    """Flatten a content element to text, but keep heading markers on
    their own lines. extract() replaces every <h1>-<h6> with a
    "## Heading" line; everything else collapses to single spaces exactly
    like get_text(" ", strip=True) did, so the only new structure in
    page["text"] is one "\n## Heading\n" per section heading.
    """
    lines = []
    for line in body.get_text(" ").split("\n"):
        line = " ".join(line.split())
        if line:
            lines.append(f"\n{line}\n" if HEADING_LINE.match(line) else line)
    return " ".join(lines).strip()


def extract(html: str, url: str) -> dict:
    """Return {"url", "title", "text", "images": [...]} for one page.

    soup.get_text() on the whole page returns the navigation menu and
    footer on every page. Those near-identical fragments become chunks
    that look moderately similar to every query and crowd real results
    out of your top five. Open the site, right-click the content, choose
    Inspect, and find the element that actually wraps it.
    """
    soup = BeautifulSoup(html, "html.parser")

    # TODO replace this with the element that holds the content, e.g.
    # TODO soup.select_one("main") or soup.select_one("#content")
    body = soup.select_one("#content")
    if body is None:
        # Not a content page (asset, redirect stub, odd template). Raise a
        # clean ASCII message so the caller's skip line prints safely on
        # consoles that cannot encode bs4's replacement characters.
        raise ValueError("no #content element")

    # Keep section headings in the text as markdown-style lines ("## X"
    # on its own line). build/index.py chunks on those lines and
    # prepends the heading to every chunk, so a fact never loses the
    # section that gives it meaning.
    for h in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        label = " ".join(h.get_text(" ", strip=True).split())
        if label:
            h.replace_with(f"\n{'#' * int(h.name[1])} {label}\n")

    images = []
    for img in soup.select("img"):
        src = img.get("src")
        if not src:
            continue
        fig = img.find_parent("figure")
        images.append({
            "src":     urljoin(url, src),     # relative -> absolute
            "alt":     img.get("alt", ""),
            "caption": (fig.find("figcaption").get_text(strip=True)
                        if fig and fig.find("figcaption") else ""),
            "page":    url,
        })

    page = {
        "url":    url,
        "title":  soup.title.get_text(strip=True) if soup.title else "",
        "text":   text_with_headings(body),
        "images": images,
    }
    page["year"] = extract_page_year(page)
    page["page_type"] = classify_page(page)
    for image in images:
        image["year"] = (
            page["year"] if page["year"] is not None
            else extract_embedded_year(image["src"])
        )
        image["page_type"] = page["page_type"]
    return page


if __name__ == "__main__":
    # A skipped page's exception text can contain bytes the GBK console
    # cannot encode; errors="replace" keeps one bad URL from aborting the
    # whole crawl with a UnicodeEncodeError.
    import sys
    sys.stdout.reconfigure(errors="replace")

    pages = []
    for site in SITES:
        for url in crawl(site):
            try:
                pages.append(extract(requests.get(url, timeout=20).text, url))
            except Exception as exc:
                print("skipped", url, exc)

    Path("data").mkdir(exist_ok=True)
    Path("data/pages.json").write_text(json.dumps(pages, indent=1))

    images = [im for p in pages for im in p["images"]]
    Path("data/images.json").write_text(json.dumps(images, indent=1))

    print(f"{len(pages)} pages, {len(images)} images")