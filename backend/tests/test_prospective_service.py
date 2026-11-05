from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from app.memory.prospective.service import ProspectiveMemoryService
from app.perception.contracts import (
    FastPerceptionFrame,
    FastPerceptionRequest,
    FastProspectiveMatch,
)


class _Repository:
    def __init__(self) -> None:
        self.claimed_subject_ids = []
        self.occurrence_keys = set()
        self.dedupe_windows = []

    async def try_claim_subject(self, **kwargs):
        self.claimed_subject_ids.append(kwargs["subject_id"])
        return uuid4()

    async def reserve_delivery_event(self, **kwargs):
        self.dedupe_windows.append(kwargs["dedupe_window_seconds"])
        occurrence_key = kwargs["occurrence_key"]
        if occurrence_key in self.occurrence_keys:
            return None
        self.occurrence_keys.add(occurrence_key)
        return SimpleNamespace(id=uuid4())

    async def release_subject_claim(self, **kwargs):
        return None

    async def resolve_delivery_event(self, **kwargs):
        return None

    async def mark_event_dispatched(self, **kwargs):
        return None


class _SessionManager:
    async def speak_prospective_interrupt(self, **kwargs):
        return None


class ProspectiveMemoryServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_highest_confidence_candidate_is_selected_before_limit(self) -> None:
        repository = _Repository()
        service = ProspectiveMemoryService(
            repository=repository,
            realtime_session_manager=_SessionManager(),
            settings=_settings(max_candidates=2),
        )
        request = _request()
        candidates = [
            _candidate(request, confidence=0.86),
            _candidate(request, confidence=0.87),
            _candidate(request, confidence=0.99),
        ]

        await service.process_matches(request=request, matches=candidates)

        self.assertEqual(repository.claimed_subject_ids, [candidates[2].subject_id])
        self.assertEqual(repository.dedupe_windows, [10])


def _settings(*, max_candidates: int):
    return SimpleNamespace(
        PROSPECTIVE_MAX_CANDIDATES_PER_ANALYSIS=max_candidates,
        VISUAL_INTERRUPT_CONFIDENCE_THRESHOLD=0.85,
        PROSPECTIVE_MAX_EVIDENCE_AGE_SECONDS=10,
        PROSPECTIVE_TRIGGER_CLAIM_LEASE_SECONDS=120,
        VISUAL_INTERRUPT_DEDUPE_SECONDS=10,
    )


def _request() -> FastPerceptionRequest:
    captured_at = datetime.now(UTC)
    frame = FastPerceptionFrame(
        frame_id=uuid4(),
        frame_sequence=1,
        captured_at=captured_at,
        source="glasses",
        content_type="image/jpeg",
        image_bytes=b"jpeg",
    )
    return FastPerceptionRequest(
        owner_id="owner",
        session_id=uuid4(),
        window_id=uuid4(),
        frames=(frame,),
        previous_working_state={},
        existing_entity_ids=frozenset(),
        recent_audio_transcripts=(),
        behavior_instructions="",
        active_reminders=(),
        active_focus=None,
        recent_interrupts=(),
        requested_at=captured_at,
    )


def _candidate(
    request: FastPerceptionRequest,
    *,
    confidence: float,
) -> FastProspectiveMatch:
    return FastProspectiveMatch(
        subject_type="visual_focus",
        subject_id=uuid4(),
        confidence=confidence,
        observed_condition="The target is visible.",
        supporting_frame_ids=[request.primary_frame.frame_id],
        suggested_message="The target is visible.",
    )
