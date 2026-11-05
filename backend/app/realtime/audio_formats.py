from __future__ import annotations

from typing import Any


def normalize_realtime_audio_format(format_name: str, *, rate: int) -> dict[str, Any]:
    normalized = format_name.strip().lower()
    if normalized in {"pcm16", "pcm", "audio/pcm"}:
        return {"type": "audio/pcm", "rate": rate}
    if normalized in {"pcmu", "mulaw", "mu-law", "g711_ulaw", "audio/pcmu"}:
        return {"type": "audio/pcmu"}
    if normalized in {"pcma", "alaw", "a-law", "g711_alaw", "audio/pcma"}:
        return {"type": "audio/pcma"}
    raise ValueError(f"Unsupported realtime audio format: {format_name}")
