"""A tiny vector store with zero dependencies.

Embeddings are hashed bag-of-words (stable across processes, unlike hash()).
Good enough to demonstrate retrieval; swap in real embeddings later by
replacing embed() - nothing else has to change.
"""
from __future__ import annotations

import json
import math
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

DIM = 512
_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "it", "on",
    "for", "with", "that", "this", "be", "as", "by", "at", "from", "was", "were",
}


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def embed(text: str) -> list[float]:
    """Hashed bag-of-words, L2-normalised."""
    vec = [0.0] * DIM
    for tok in tokens(text):
        vec[zlib.crc32(tok.encode()) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Record:
    id: str
    text: str
    meta: dict = field(default_factory=dict)
    vec: list[float] = field(default_factory=list)


@dataclass
class Store:
    """Persistent, filterable, in-memory-with-JSON-backing vector store."""

    path: str | None = None
    records: dict[str, Record] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path and Path(self.path).exists():
            self.load()

    def add(self, id: str, text: str, **meta) -> Record:
        rec = Record(id=id, text=text, meta=meta, vec=embed(text))
        self.records[id] = rec
        if self.path:
            self.save()
        return rec

    def search(self, query: str, k: int = 5, **filters) -> list[tuple[Record, float]]:
        qv = embed(query)
        hits = []
        for rec in self.records.values():
            if any(rec.meta.get(key) != val for key, val in filters.items()):
                continue
            hits.append((rec, cosine(qv, rec.vec)))
        hits.sort(key=lambda pair: pair[1], reverse=True)
        return hits[:k]

    # --- persistence -------------------------------------------------
    def save(self) -> None:
        if not self.path:
            return
        payload = [
            {"id": r.id, "text": r.text, "meta": r.meta} for r in self.records.values()
        ]
        Path(self.path).write_text(json.dumps(payload), encoding="utf-8")

    def load(self) -> None:
        if not self.path:
            return
        for row in json.loads(Path(self.path).read_text(encoding="utf-8")):
            self.records[row["id"]] = Record(
                id=row["id"], text=row["text"], meta=row["meta"], vec=embed(row["text"])
            )

    def __len__(self) -> int:
        return len(self.records)
