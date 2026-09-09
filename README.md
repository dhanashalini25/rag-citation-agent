# 02 - RAG Agent with Citation Grounding

> Answers that cite their sources and admit when they can't.

**What it demonstrates:** Grounding answers in retrieved sources instead of model memory

**Status:** working implementation with passing tests. Built as a learning project to understand the pattern, not as a production service.

---

## Run it right now

No API key needed - every project ships with `MODEL=fake`, a deterministic
offline responder, so you can see the whole flow work before spending anything.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
python -m src.main
pytest -q
```

To use a real model, edit `.env`:

```
MODEL=gpt-4o-mini            # + OPENAI_API_KEY
MODEL=claude-3-5-haiku-latest  # + ANTHROPIC_API_KEY
MODEL=ollama/llama3.1        # free, runs locally
```

## How it works

Documents in `data/` are split into overlapping word windows and embedded into a small local vector store (`src/store.py` - hashed bag-of-words, no downloads, no torch). Retrieval returns chunks with similarity scores attached.

The prompt requires a chunk id in square brackets at the end of every sentence. `enforce_citations` then splits the answer into sentences and **drops any sentence whose citation is missing or points at a chunk that wasn't retrieved** - so a fabricated sentence never survives into the output.

If the best similarity is below `CONFIDENCE_FLOOR`, the agent doesn't answer at all: it either calls the web fallback or says it doesn't have the information.

## What "done" means here

- Every sentence in the answer carries a chunk id that actually exists
- Sentences without a valid citation are dropped and counted
- Below the similarity floor the agent refuses instead of guessing
- `INSUFFICIENT_CONTEXT` from the model is honoured, not paraphrased away
- Chunk size and overlap are configuration, not magic numbers in the loop
- Retrieval scores are returned so the caller can see how confident it was

Every one of those lines has a test behind it in `tests/` - `pytest -q` is the
proof, not the README.

## Layout

```
src/llm.py             provider-agnostic completion, plus offline fake mode
src/fake.py            the canned responses that make MODEL=fake work
src/logging_setup.py   structured JSON logging
src/agent.py           the pattern itself
src/main.py            CLI entrypoint
tests/                 8 tests, all passing
```

## Next steps

- Swap `embed()` in `src/store.py` for real embeddings and compare retrieval quality
- Build a 20-question eval set with expected sources; measure citation precision and recall
- Wire a real search API into `web_fallback` and keep the `web:` source label

## Reference

https://js.langchain.com/docs/how_to/qa_citations

---

Part of a 12-project agentic AI series - [github.com/dhanashalini25](https://github.com/dhanashalini25?tab=repositories)
