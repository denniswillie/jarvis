from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "you",
    "your",
}


def visual_interrupt_dedupe_bucket(message: str, *, max_tokens: int = 8) -> str:
    tokens = [token for token in _TOKEN_RE.findall(message.lower()) if token not in _STOP_WORDS]
    if not tokens:
        return "empty"
    return ":".join(tokens[:max_tokens])
