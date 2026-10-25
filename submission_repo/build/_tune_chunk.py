"""Sweep (size, overlap) and score answer-span recall on the dev set.

For each config: chunk every page with build.index.chunk, embed chunks
and questions with the real embedder, rank by cosine similarity in numpy
(exact top-k, no Chroma native add -> no segfault on this box), and check
whether the gold answer text shows up in the retrieved top-k context.

Uses numpy instead of chromadb because chromadb 1.5.9 access-violates on
collection.add() on this Windows/Python 3.12 build. Ranking on cosine of
the same embeddings is what Chroma does internally, so the recall numbers
carry over; only the approximate HNSW layer is skipped.
"""
import hashlib
import json
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from bot.llm import embed
from build.index import chunk

K = 5  # matches bot/answer.py CONFIG["k"]

# Disk cache: sha256(chunk text) -> embedding. The gateway quota is
# capped, and configs share most chunk texts (overlap only changes the
# ~14% of sections that fall through to the character split), so caching
# makes sweeps after the first nearly free.
CACHE_PATH = Path("data/embed_cache.pkl")


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return pickle.loads(CACHE_PATH.read_bytes())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_bytes(pickle.dumps(cache, protocol=4))


def embed_cached(texts: list[str]) -> np.ndarray:
    """embed() with a disk cache keyed by text hash."""
    cache = _load_cache()
    keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    # Dedupe within the batch too: repeated boilerplate chunks across
    # pages would otherwise each cost an embedding call.
    unique = {}
    for i, k in enumerate(keys):
        if k not in cache:
            unique.setdefault(k, texts[i])
    if unique:
        vecs = embed(list(unique.values()))
        for k, v in zip(unique.keys(), vecs):
            cache[k] = v
        _save_cache(cache)
    print(f"    embed: {len(unique)} new, {len(texts)-len(unique)} hit/dup",
          flush=True)
    return np.asarray([cache[k] for k in keys], dtype=np.float32)


def norm(s: str) -> str:
    s = s.lower().replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def answer_in(answer: str, context: str) -> float:
    """1.0 if the whole answer phrase appears in context; else token overlap."""
    na, nc = norm(answer), norm(context)
    if not na:
        return 0.0
    if na in nc:
        return 1.0
    toks = [t for t in na.split() if len(t) > 1 or t.isdigit()]
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in nc.split()) / len(toks)


def _matrix(texts: list[str]) -> np.ndarray:
    e = embed_cached(texts)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return e / n


def score_config(pages, questions, size, overlap):
    texts = []
    for page in pages:
        texts.extend(chunk(page["text"], size=size, overlap=overlap,
                           title=page.get("title", "")))
    E = _matrix(texts)
    Q = _matrix([q["question"] for q in questions])
    sims = Q @ E.T                       # (nq, nchunks)
    top = np.argsort(-sims, axis=1)[:, :K]

    total = 0.0
    per_q = []
    for qi, q in enumerate(questions):
        ctx = " \n ".join(texts[ci] for ci in top[qi])
        sc = answer_in(q["answer"], ctx)
        total += sc
        per_q.append((q.get("level", "S"), round(sc, 2)))
    return total / len(questions), len(texts), per_q


def load_questions() -> list[dict]:
    """Dev set plus synthetic QA (if generated) - more questions give a
    finer-grained recall signal than the 9 dev questions alone."""
    dev = json.loads(Path("dev_set.json").read_text(encoding="utf-8"))
    synth = []
    p = Path("data/tune_qa.json")
    if p.exists():
        synth = json.loads(p.read_text(encoding="utf-8"))
    print(f"questions: {len(dev)} dev + {len(synth)} synthetic", flush=True)
    return dev + synth


if __name__ == "__main__":
    # Preflight: one cheap embed so an exhausted gateway quota fails
    # immediately instead of halfway through a sweep.
    try:
        embed(["quota probe"])
    except Exception as exc:
        sys.exit(f"embedding gateway unavailable: {str(exc)[:200]}")

    pages = json.loads(Path("data/pages.json").read_text(encoding="utf-8"))
    questions = load_questions()

    if len(sys.argv) > 1:
        # "size:overlap,size:overlap,..." to re-run a focused grid
        grid = [tuple(int(x) for x in c.split(":")) for c in sys.argv[1].split(",")]
    else:
        grid = [
            (400, 50), (600, 75), (800, 100), (1000, 125),
            (1200, 150), (1600, 200),
        ]
    results = []
    for size, overlap in grid:
        t = time.time()
        recall, nchunks, per_q = score_config(pages, questions, size, overlap)
        results.append((recall, size, overlap, nchunks))
        print(f"size={size:5} overlap={overlap:4}  recall={recall:.3f}  "
              f"chunks={nchunks:5}  ({time.time()-t:.1f}s)", flush=True)

    print("\n=== ranked by recall ===")
    for recall, size, overlap, nchunks in sorted(results, reverse=True):
        print(f"  recall={recall:.3f}  size={size:5} overlap={overlap:4} chunks={nchunks}")
    best = max(results)[1:3]
    print(f"\nBEST: CHUNK_SIZE, OVERLAP = {best[0]}, {best[1]}")
