"""Canonical visual projection never invents stream identity or freshness."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceError
from app.knowledge.observations import OBSERVATION_NAMESPACE
from app.live_state.canonical import context_dependency
from app.live_state.models import GuardedPayloadBinding, SessionScope
from app.live_state.perception_adapter import PerceptionLiveAdapter, observation_from_record
from app.perception.models import SessionWorkingMemoryState
from app.runtime.events import DomainEventBus
from app.tasks.context_events import ContextRecord


NOW = datetime(2026, 10, 30, 18, tzinfo=UTC)


def fixture():
    scope = SessionScope(owner_id="visual-fixture", session_id=uuid4(), area_id=uuid4())
    observation_id, frame_id, stream_id = uuid4(), uuid4(), uuid4()
    raw = dict(id=str(observation_id), owner_id=scope.owner_id, session_id=str(scope.session_id),
        source="glasses", started_at=NOW.isoformat(), ended_at=NOW.isoformat(),
        primary_visual_frame_id=str(frame_id), summary="A display is visible.", scene="room",
        activity="reading", entities=[], visible_text=[dict(text="14", context="display", confidence=0.8)],
        state_changes=[], memory_tags=[], keyframe_candidates=[], interrupt_candidate=None,
        starts_new_event=False, boundary_reason=None, confidence=0.8, analyzer_provider="fixture",
        analyzer_model="fixture", analyzer_contract_version=1, compacted_at=None, created_at=NOW.isoformat())
    image = dict(frame_id=str(frame_id), session_id=str(scope.session_id), visual_session_id=str(stream_id),
        captured_at=NOW.isoformat(), source="historical", content_type="image/jpeg", width=100, height=100)
    record = ContextRecord(id=uuid4(), owner_id=scope.owner_id, session_id=scope.session_id,
        sequence=1, origin_sequence=1, source_namespace=OBSERVATION_NAMESPACE,
        source_event_id=str(observation_id), revision=1, supersedes_event_id=None,
        kind="observation", trust="derived", finality="final", source_time=NOW, recorded_at=NOW,
        payload_id=uuid4(), availability="available", content_digest="c" * 64, is_current=True,
        content={"origin": "perception_observations", "record": raw, "images": [image],
                 "frame_links": [{"frame_id": str(frame_id), "ordinal": 0, "role": "primary"}]})
    state = SessionWorkingMemoryState(session_id=scope.session_id, owner_id=scope.owner_id,
        revision=1, last_observation_id=observation_id, last_observation_ended_at=NOW,
        state={"summary": raw["summary"]}, expires_at=NOW + timedelta(seconds=30),
        created_at=NOW, updated_at=NOW)
    return scope, state, record, stream_id


class ObservationDecodeTests(unittest.TestCase):
    def test_original_capture_time_and_stream_survive(self):
        scope, state, record, stream = fixture()
        observation, frames = observation_from_record(record)
        self.assertEqual(observation.id, state.last_observation_id)
        self.assertEqual(observation.visible_text[0].text, "14")
        self.assertEqual(frames[0].stream_epoch, stream)
        self.assertEqual(frames[0].captured_at, NOW)

    def test_missing_stream_is_not_replaced_with_random_identity(self):
        _, _, record, _ = fixture()
        record.content["images"][0].pop("visual_session_id")
        with self.assertRaises(SourceError):
            observation_from_record(record)

    def test_later_frame_cannot_support_older_analysis(self):
        _, _, record, _ = fixture()
        record.content["images"][0]["captured_at"] = (NOW + timedelta(seconds=1)).isoformat()
        with self.assertRaises(SourceError):
            observation_from_record(record)

    def test_cross_session_original_is_rejected(self):
        _, _, record, _ = fixture()
        record.content["record"]["session_id"] = str(uuid4())
        with self.assertRaises(SourceError):
            observation_from_record(record)

    def test_noncurrent_canonical_original_is_rejected(self):
        _, _, record, _ = fixture()
        with self.assertRaises(SourceError):
            observation_from_record(record.model_copy(update={"is_current": False}))


class PerceptionAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope, self.state, self.record, _ = fixture()
        self.events = []
        def submit(event):
            self.events.append(event)
            return SimpleNamespace(queued=True)
        service = SimpleNamespace(snapshot=lambda: SimpleNamespace(scope=self.scope),
            submit=submit, register_producer=lambda *args: None)
        self.working = SimpleNamespace(get_session_working_memory=AsyncMock(return_value=self.state))
        self.adapter = PerceptionLiveAdapter(service=service, working_memory=self.working,
            sources=SimpleNamespace(pool=None), bus=DomainEventBus(), clock=lambda: NOW)
        self.adapter.producer.start()
        self.binding = GuardedPayloadBinding(scope=self.scope, payload_root_ids=(uuid4(),),
            dependencies=(context_dependency(self.record),))
        self.adapter.binder.bind_record = AsyncMock(return_value=self.binding)
        self.adapter.binder.available = AsyncMock(return_value=True)

    async def test_publishes_only_after_binding_exact_rendered_observation(self):
        await self.adapter._apply((self.state, self.record))
        entry = self.events[-1].entry
        self.assertIn("14", entry.summary)
        self.assertEqual(entry.guard_binding, self.binding)
        self.assertEqual(self.adapter.binder.bind_record.call_args.args[1], entry.summary)
        self.assertEqual(entry.observed_at, NOW)
        self.assertEqual(entry.fresh_until, NOW + timedelta(seconds=10))

    async def test_replacement_during_binding_drops_old_publication(self):
        self.working.get_session_working_memory.return_value = replace(self.state, revision=2)
        await self.adapter._apply((self.state, self.record))
        self.assertEqual(self.events, [])

    async def test_unchanged_revision_rechecks_guard_without_copying_again(self):
        await self.adapter._apply((self.state, self.record))
        count = len(self.events)
        await self.adapter._apply((self.state, None))
        self.assertEqual(len(self.events), count)
        self.adapter.binder.bind_record.assert_awaited_once()
        self.adapter.binder.available.assert_awaited_once_with(self.binding)

    async def test_source_withdrawal_clears_projection(self):
        await self.adapter._apply((self.state, self.record))
        self.adapter.binder.available.return_value = False
        await self.adapter._apply((self.state, None))
        self.assertEqual(self.events[-1].entry_id, "working_observation")
        self.assertIsNone(self.adapter._last_entry)

    async def test_absent_working_memory_is_not_scene_resolution(self):
        await self.adapter._apply((None, None))
        self.assertEqual(self.events[-1].reason, "source_withdrawn")
        self.assertEqual(self.adapter.last_invalidation_reason, "visual_evidence_unavailable")
        self.assertFalse(hasattr(self.events[-1], "resolved"))

    async def test_expiry_and_close_use_typed_retractions_with_separate_diagnostics(self):
        await self.adapter.invalidate("visual_evidence_expired")
        self.assertEqual(self.events[-1].reason, "expired")
        self.assertEqual(self.adapter.last_invalidation_reason, "visual_evidence_expired")
        await self.adapter.close()
        self.assertEqual(self.events[-1].reason, "cancelled")
        self.assertEqual(self.adapter.last_invalidation_reason, "perception_projection_closed")
