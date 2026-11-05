from __future__ import annotations

import unittest
from datetime import UTC, datetime
from uuid import uuid4

from app.experience.models import (
    ExperienceEvent,
    UnsafeToolPayloadError,
    VisualFrameRetentionReason,
    normalize_durable_tool_payload,
)


class ExperienceModelTests(unittest.TestCase):
    def test_audio_only_open_event_requires_no_visual_state(self) -> None:
        event = _event(modalities=("audio",), event_kind="interaction")
        self.assertEqual(event.modalities, ("audio",))
        self.assertEqual(event.status, "open")

    def test_closed_event_requires_atomic_readiness_markers(self) -> None:
        now = datetime.now(UTC)
        with self.assertRaisesRegex(ValueError, "atomic readiness"):
            _event(status="closed", ended_at=now)

    def test_tool_payload_rejects_credentials_raw_bytes_and_signed_urls(self) -> None:
        unsafe_payloads = (
            {"access_token": "secret"},
            {"nested": {"raw_bytes": b"image"}},
            {"url": "https://example.test/file?token=secret"},
        )
        for payload in unsafe_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(UnsafeToolPayloadError):
                    normalize_durable_tool_payload(payload, label="tool result")

    def test_tool_payload_preserves_safe_json(self) -> None:
        payload = {"status": "success", "items": [{"title": "Example"}]}
        self.assertEqual(
            normalize_durable_tool_payload(payload, label="tool result"),
            payload,
        )

    def test_finite_retention_reason_requires_expiry(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite expiry"):
            VisualFrameRetentionReason(
                id=uuid4(),
                owner_id="owner",
                visual_frame_id=uuid4(),
                reason_type="event_candidate",
                reason_ref_id=uuid4(),
                expires_at=None,
                created_at=datetime.now(UTC),
            )


def _event(**overrides: object) -> ExperienceEvent:
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "id": uuid4(),
        "owner_id": "owner",
        "session_id": uuid4(),
        "status": "open",
        "event_kind": "visual",
        "modalities": ("visual",),
        "started_at": now,
        "ended_at": None,
        "title": "",
        "summary": "",
        "scene": None,
        "activity": None,
        "state": {},
        "boundary_reason": None,
        "importance": 0.5,
        "confidence": 0.8,
        "keyframes_finalized_at": None,
        "consolidation_ready_at": None,
        "evidence_reduced_at": None,
        "retention_expires_at": None,
        "consolidation_result": None,
        "consolidation_provider": None,
        "consolidation_model": None,
        "consolidated_at": None,
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    return ExperienceEvent(**values)  # type: ignore[arg-type]
