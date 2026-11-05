"""Per-message speech delivery policy, independent of provider transport and tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


OutputPhase = Literal["commentary", "final_answer"]
DeliveryDisposition = Literal["pending", "suppressed", "forwarded"]


@dataclass(slots=True)
class RealtimeOutputItemState:
    item_id: str | None
    output_index: int | None = None
    phase: OutputPhase | None = None
    delivery_disposition: DeliveryDisposition = "pending"
    phase_anomaly: bool = False
    transcript: str = ""
    audio_done: bool = False
    transcript_done: bool = False
    persisted: bool = False
    generated_media_events: int = 0
    suppressed_media_events: int = 0
    forwarded_media_events: int = 0
    forwarded_audio_chunks: int = 0

    def observe_phase(self, value: Any) -> str | None:
        """Keep the first known phase and never retract already-forwarded output."""
        if value is None:
            return None
        if value not in ("commentary", "final_answer"):
            return "unknown_phase"
        if self.phase is not None and value != self.phase:
            self.phase_anomaly = True
            return "conflicting_phase"
        self.phase = value
        if value == "commentary":
            if self.delivery_disposition == "forwarded":
                self.phase_anomaly = True
                return "late_commentary_phase"
            self.delivery_disposition = "suppressed"
        return None

    @property
    def allows_output(self) -> bool:
        # Unknown/missing phase uses the legacy streaming path. Once forwarded,
        # a late phase must not cut an utterance in half.
        return self.delivery_disposition != "suppressed"

    def record_generated(self) -> bool:
        self.generated_media_events += 1
        if not self.allows_output:
            self.suppressed_media_events += 1
        return self.allows_output

    def record_forwarded(self, *, audio_chunk: bool = False) -> None:
        self.delivery_disposition = "forwarded"
        self.forwarded_media_events += 1
        self.forwarded_audio_chunks += int(audio_chunk)

    @property
    def has_forwarded_answer(self) -> bool:
        return (
            self.delivery_disposition == "forwarded"
            and self.phase != "commentary"
            and (self.forwarded_audio_chunks > 0 or bool(self.transcript.strip()))
        )

    def as_metadata(self) -> dict[str, Any]:
        """Transport evidence only: forwarding is not proof of playback/hearing."""
        return {
            "item_id": self.item_id,
            "output_index": self.output_index,
            "phase": self.phase,
            "delivery_disposition": self.delivery_disposition,
            "phase_anomaly": self.phase_anomaly,
            "generated_media_events": self.generated_media_events,
            "suppressed_media_events": self.suppressed_media_events,
            "forwarded_media_events": self.forwarded_media_events,
        }
