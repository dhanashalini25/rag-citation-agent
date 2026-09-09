"""Canned reply for MODEL=fake: cites the two best chunks, plus one uncited
sentence so you can see enforce_citations() drop it."""
from __future__ import annotations

import re


def respond(messages: list[dict]) -> str:
    content = messages[-1]["content"]
    ids = re.findall(r"\[(c\d+)\]", content)
    if not ids:
        return "INSUFFICIENT_CONTEXT"
    first = ids[0]
    second = ids[1] if len(ids) > 1 else ids[0]
    return (
        f"The agent retries up to three times before giving up [{first}]. "
        f"Each failure is logged with the raw output and the attempt number [{second}]. "
        "I think that is probably the best approach overall."
    )
