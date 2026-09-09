"""RAG Agent with Citation Grounding.

Retrieves from a local corpus, answers only from what it retrieved, tags every
sentence with the chunk it came from, and drops sentences that cite nothing.
When retrieval is weak it says so instead of inventing an answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .llm import complete
from .logging_setup import log
from .store import Store

CONFIDENCE_FLOOR = 0.05
CHUNK_WORDS = 40
CHUNK_OVERLAP = 10
STORE_PATH = "rag_store.json"
DEMO = "How many times does the agent retry before giving up?"

_CITE = re.compile(r"\[(c\d+)\]")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    score: float = 0.0


@dataclass
class Answer:
    text: str
    citations: list[Chunk] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    low_confidence: bool = False
    used_web_fallback: bool = False


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding window over words. Overlap keeps sentences from being cut in half."""
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    return [" ".join(words[i : i + size]) for i in range(0, len(words), step) if words[i : i + size]]


def index(paths: list[str] | None = None, store_path: str = STORE_PATH) -> Store:
    """Chunk and embed every file in `paths` (defaults to ./data/*)."""
    if paths is None:
        paths = sorted(str(p) for p in Path("data").glob("*.md"))

    store = Store(path=store_path)
    store.records.clear()
    n = 0
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        for piece in chunk_text(text):
            n += 1
            store.add(f"c{n}", piece, source=Path(path).name)
    store.save()
    log.info("indexed", extra={"files": len(paths), "chunks": n})
    return store


def retrieve(question: str, k: int = 4, store_path: str = STORE_PATH) -> list[Chunk]:
    store = Store(path=store_path)
    if not len(store):
        store = index(store_path=store_path)
    return [
        Chunk(id=rec.id, text=rec.text, source=rec.meta.get("source", "?"), score=score)
        for rec, score in store.search(question, k=k)
    ]


def web_fallback(question: str) -> list[Chunk]:
    """Placeholder for a real search API.

    Anything returned here is labelled `web:` so a reader can tell grounded
    answers from external ones. Wire in Tavily/Brave/SerpAPI and keep the label.
    """
    log.info("web_fallback", extra={"q": question[:80]})
    return []


PROMPT = (
    "Answer using ONLY the numbered context below. Put the chunk id in square "
    "brackets at the end of every sentence, like [c3]. If the context does not "
    "answer the question, reply exactly: INSUFFICIENT_CONTEXT.\n\n"
    "Context:\n{context}\n\nQuestion: {q}"
)


def enforce_citations(text: str, valid: set[str]) -> tuple[str, list[str]]:
    """Keep only sentences carrying a citation that actually exists."""
    kept, dropped = [], []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not sentence:
            continue
        ids = set(_CITE.findall(sentence))
        if ids and ids <= valid:
            kept.append(sentence)
        else:
            dropped.append(sentence)
    return " ".join(kept), dropped


def ask(question: str, store_path: str = STORE_PATH) -> Answer:
    chunks = retrieve(question, store_path=store_path)
    best = max((c.score for c in chunks), default=0.0)
    low = best < CONFIDENCE_FLOOR
    used_web = False

    if low:
        log.warning("low_confidence", extra={"best": round(best, 3)})
        extra = web_fallback(question)
        if extra:
            chunks, used_web = extra, True
        else:
            return Answer(
                text="I don't have that in my sources.", citations=[], low_confidence=True
            )

    context = "\n".join(f"[{c.id}] {c.text}" for c in chunks)
    raw = complete([{"role": "user", "content": PROMPT.format(context=context, q=question)}])

    if raw.strip() == "INSUFFICIENT_CONTEXT":
        return Answer(text="I don't have that in my sources.", low_confidence=True)

    text, dropped = enforce_citations(raw, {c.id for c in chunks})
    if dropped:
        log.warning("uncited_sentences_dropped", extra={"n": len(dropped)})

    cited = {cid for cid in _CITE.findall(text)}
    return Answer(
        text=text or "I don't have that in my sources.",
        citations=[c for c in chunks if c.id in cited],
        dropped=dropped,
        low_confidence=low,
        used_web_fallback=used_web,
    )


def run(prompt: str) -> str:
    a = ask(prompt)
    srcs = ", ".join(f"{c.id} ({c.source})" for c in a.citations) or "none"
    return f"{a.text}\n\nSources: {srcs}"
