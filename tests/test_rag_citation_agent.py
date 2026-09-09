import pytest

from src import agent
from src.agent import Store, ask, chunk_text, enforce_citations, index, retrieve

CORPUS = """The agent retries up to three times before giving up.
Each validation failure is logged with the raw output and the attempt number.

Routing picks a small model for simple tasks and a frontier model for hard ones.
Budgets are enforced before the call is made, not discovered afterwards."""


@pytest.fixture()
def corpus(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "notes.md").write_text(CORPUS, encoding="utf-8")
    return [str(d / "notes.md")], str(tmp_path / "s.json")


def test_chunking_overlaps():
    chunks = chunk_text(" ".join(str(i) for i in range(200)), size=40, overlap=10)
    assert len(chunks) > 1
    assert set(chunks[0].split()) & set(chunks[1].split())


def test_index_then_retrieve_scores(corpus):
    paths, sp = corpus
    index(paths, store_path=sp)
    hits = retrieve("how many retries before giving up", store_path=sp)
    assert hits and hits[0].score > 0
    assert "retries" in hits[0].text


def test_scores_are_ordered(corpus):
    paths, sp = corpus
    index(paths, store_path=sp)
    hits = retrieve("budget enforced before the call", store_path=sp)
    assert hits == sorted(hits, key=lambda c: c.score, reverse=True)


def test_enforce_citations_drops_uncited():
    text = "Retries are capped [c1]. This bit is invented."
    kept, dropped = enforce_citations(text, {"c1"})
    assert kept == "Retries are capped [c1]."
    assert dropped == ["This bit is invented."]


def test_enforce_citations_drops_invalid_id():
    kept, dropped = enforce_citations("Made up source [c99].", {"c1"})
    assert kept == ""
    assert len(dropped) == 1


def test_low_confidence_refuses(corpus, monkeypatch):
    paths, sp = corpus
    index(paths, store_path=sp)
    monkeypatch.setattr(agent, "CONFIDENCE_FLOOR", 0.99)
    a = ask("something entirely unrelated to the corpus", store_path=sp)
    assert a.low_confidence
    assert "don't have that" in a.text


def test_insufficient_context_is_honoured(corpus, monkeypatch):
    paths, sp = corpus
    index(paths, store_path=sp)
    monkeypatch.setattr(agent, "complete", lambda m, **k: "INSUFFICIENT_CONTEXT")
    assert ask("retries", store_path=sp).low_confidence


def test_answer_cites_real_chunks(corpus, monkeypatch):
    paths, sp = corpus
    index(paths, store_path=sp)
    monkeypatch.setattr(agent, "complete", lambda m, **k: "Retries are capped at three [c1].")
    a = ask("how many retries", store_path=sp)
    assert a.citations and all(c.id in a.text for c in a.citations)
