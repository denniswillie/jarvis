"""Deterministic Live factual content; authority identities stay local.

The complete ASCII-escaped returned string is what the existing conservative
byte counter must measure. This is not a model tokenizer or a token estimate.
"""
from __future__ import annotations

import json


def encode_live_context_content(text: str, *, mode: str = "snapshot") -> str:
    # Cursor, dependencies, roots, scope and command/ACK identities remain in
    # the immutable local update and guards, not redundant model-facing text.
    return json.dumps({"kind": "factual_context", "text": text, "mode": mode},
                      ensure_ascii=True, separators=(",", ":"))
