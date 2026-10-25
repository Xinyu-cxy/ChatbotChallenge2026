"""Generate synthetic QA pairs for chunk-size tuning.

The dev set has 9 questions - too few to reliably rank chunking
configs. This samples real sections from data/pages.json and asks the
chat model for one factual question per section plus a short VERBATIM
answer span. Only spans that actually appear in the source section are
kept, so scoring is a simple substring check: a config hits when
retrieval surfaces a chunk containing the span.

Writes data/tune_qa.json. Deterministic sample (seed 0).
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from bot.llm import chat
from build.index import split_sections

N_SECTIONS = 60
BATCH = 5
OUT = Path("data/tune_qa.json")


def norm_ws(s: str) -> str:
    return " ".join(s.split())


def sample_sections(pages: list[dict]) -> list[dict]:
    cands = []
    for page in pages:
        title = page.get("title", "")
        for heading, body in split_sections(page["text"]):
            if 200 <= len(body) <= 1600:
                cands.append({
                    "url": page["url"], "title": title,
                    "heading": heading, "body": body,
                })
    random.seed(0)
    random.shuffle(cands)
    # spread across pages: at most 2 sections per page
    picked, per_page = [], {}
    for c in cands:
        if per_page.get(c["url"], 0) >= 2:
            continue
        picked.append(c)
        per_page[c["url"]] = per_page.get(c["url"], 0) + 1
        if len(picked) >= N_SECTIONS:
            break
    return picked


PROMPT = """You generate retrieval-evaluation questions from website sections.

For EACH numbered passage below write:
- "q": one factual question a visitor might ask, answerable from that
  passage alone; do not reuse distinctive wording of the answer in the
  question;
- "a": a SHORT span copied VERBATIM from the passage (max 10 words,
  exact characters) that answers the question.

Return ONLY a JSON array with one object per passage, in order:
[{"q": "...", "a": "..."}]

"""


def gen_batch(passages: list[dict]) -> list[dict]:
    prompt = PROMPT + "\n\n".join(
        f"Passage {i+1} (section \"{p['heading'] or p['title']}\" of page \"{p['title']}\"):\n{p['body']}"
        for i, p in enumerate(passages)
    )
    reply = chat([{"role": "user", "content": prompt}], max_tokens=1500)
    m = re.search(r"\[.*\]", reply, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


if __name__ == "__main__":
    pages = json.loads(Path("data/pages.json").read_text(encoding="utf-8"))
    picked = sample_sections(pages)
    print(f"sampled {len(picked)} sections")

    qa, rejected = [], 0
    for i in range(0, len(picked), BATCH):
        batch = picked[i:i + BATCH]
        try:
            entries = gen_batch(batch)
        except Exception as exc:
            print(f"batch {i}: chat failed: {exc}")
            continue
        for p, e in zip(batch, entries):
            ans = norm_ws(str(e.get("a", "")))
            q = norm_ws(str(e.get("q", "")))
            if not q or not ans or ans.lower() not in norm_ws(p["body"]).lower():
                rejected += 1
                continue
            qa.append({"question": q, "answer": ans, "url": p["url"],
                       "heading": p["heading"]})
        print(f"batch {i}: kept {len(qa)} total, rejected {rejected}", flush=True)

    OUT.write_text(json.dumps(qa, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(qa)} QA pairs to {OUT}")
