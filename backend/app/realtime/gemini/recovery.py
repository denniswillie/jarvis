"""Bounded, process-local recovery state for one pinned Gemini conversation."""
from __future__ import annotations

from collections import deque
import json
import math
from typing import Any

from app.realtime.gemini.client import GeminiTransportError


MAX_RESUME_ATTEMPTS = 2
MAX_DEFERRED_EVENTS = 128
MAX_DEFERRED_BYTES = 8 * 1024 * 1024


class GeminiRecovery:
    def __init__(self) -> None:
        self.handle: str | None = None
        self.deadline: float | None = None
        self.reason: str | None = None
        self.attempts = 0
        self.resumptions = 0
        self._deferred: deque[tuple] = deque()
        self._deferred_bytes = 0

    @property
    def pending(self) -> bool:
        return self.deadline is not None

    @property
    def has_deferred(self) -> bool:
        return bool(self._deferred)

    def observe(self, update: dict, *, checkpoint_safe: bool) -> None:
        if update.get("resumable") is not True:
            self.handle = None
            return
        handle = update.get("newHandle")
        if not isinstance(handle, str) or not 0 < len(handle) <= 8192:
            raise GeminiTransportError("gemini_invalid_resume_handle")
        # Never treat a checkpoint from before a pending input, transcript, tool,
        # or response as proof that those newer changes can be restored.
        if checkpoint_safe:
            self.handle = handle

    def request(self, payload: dict, *, now: float, max_wait: float) -> float:
        raw = payload.get("timeLeft")
        try:
            if not isinstance(raw, str) or not raw.endswith("s"):
                raise ValueError
            remaining = float(raw[:-1])
            if not math.isfinite(remaining) or remaining <= 0:
                raise ValueError
        except (ValueError, TypeError):
            raise GeminiTransportError("gemini_invalid_goaway") from None
        deadline = now + min(remaining, max_wait)
        self.deadline = min(self.deadline, deadline) if self.deadline is not None else deadline
        self.reason = "goaway"
        return self.deadline

    def defer(self, payload: tuple) -> None:
        size = len(json.dumps(payload[0], ensure_ascii=True).encode())
        if len(self._deferred) >= MAX_DEFERRED_EVENTS or self._deferred_bytes + size > MAX_DEFERRED_BYTES:
            raise GeminiTransportError("gemini_recovery_input_limit")
        self._deferred.append(payload)
        self._deferred_bytes += size

    def release(self) -> tuple[tuple, ...]:
        deferred = tuple(self._deferred)
        self.clear()
        return deferred

    def clear(self) -> None:
        self.handle = None
        self.deadline = None
        self.reason = None
        self._deferred.clear()
        self._deferred_bytes = 0

    def diagnostics(self) -> dict[str, Any]:
        return {"pending": self.pending, "checkpoint_available": self.handle is not None,
                "deferred_events": len(self._deferred), "deferred_bytes": self._deferred_bytes,
                "attempts": self.attempts, "resumptions": self.resumptions}
