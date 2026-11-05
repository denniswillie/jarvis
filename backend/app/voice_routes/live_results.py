"""Delegation reply identities, separate from notification/playback evidence."""
from dataclasses import dataclass
from datetime import datetime
import json
from uuid import UUID

from app.voice_routes.contracts import RouteIdentity


@dataclass(frozen=True)
class LiveDelegationResult:
    route: RouteIdentity
    delegation_id: str
    # Direct service replies have a canonical payload but no background task.
    task_id: UUID | None
    task_revision: int
    result_payload_id: UUID
    text: str
    expires_at: datetime
    roots: tuple[UUID, ...]
    # Default keeps existing read-only callers/wire content unchanged. General
    # replies still require the caller's actual canonical result/source guard.
    kind: str = "readonly_delegation_result"
    speak: bool = True
    unsolicited: bool = False

    def chunks(self, count_tokens):
        """Lossless bounded framing, not truncation of an already-bound answer.

        Earlier chunks are quiet context; only the final chunk is commentary.
        Every frame is guarded by the same complete original result identity.
        """
        if self.kind not in {"readonly_delegation_result", "delegation_result"}:
            raise ValueError("live_result_kind_unavailable")
        if not self.text or len(self.text.encode("utf-8")) > 12000:
            raise ValueError("live_result_size_unavailable")
        # A generic delegation may have several independently sourced updates
        # in flight. Numbered pieces must identify their exact original result.
        frame_identity = ({"result_id": str(self.result_payload_id)}
                          if self.kind == "delegation_result" else {})
        pieces, offset = [], 0
        while offset < len(self.text):
            low, high = 0, len(self.text) - offset
            while low < high:
                middle = (low + high + 1) // 2
                encoded = json.dumps({**frame_identity, "kind": self.kind, "part": len(pieces) + 1,
                    "complete": False, "text": self.text[offset:offset + middle]},
                    ensure_ascii=True, separators=(",", ":"))
                if count_tokens(encoded) <= 500:
                    low = middle
                else:
                    high = middle - 1
            if low == 0 or len(pieces) >= 32:
                raise ValueError("live_result_size_unavailable")
            pieces.append(self.text[offset:offset + low])
            offset += low
        return tuple(json.dumps({**frame_identity, "kind": self.kind, "part": i + 1,
            "complete": i == len(pieces) - 1, "text": piece}, ensure_ascii=True, separators=(",", ":"))
            for i, piece in enumerate(pieces))


@dataclass(frozen=True)
class LiveDelegationSubmission:
    delegation_id: str
    status: str
    provider_received: bool = False
    reason: str | None = None
    # Deliberately no notification attempt, output ID, playback or hearing claim.
