from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

from unittest.mock import AsyncMock

from fastapi import WebSocketDisconnect

from app.realtime.socket_sender import WebSocketEventSender
from app.perception.contracts import FastPerceptionOutput
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.perception.session_analysis_coordinator import SessionAnalysisCoordinator
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEventBus, DomainEventName
from app.visual_memory.models import VisualFrameRecord
from app.visual_memory.pipeline import VisualAnalysisPipeline


NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


class _VisualRepository:
    def __init__(self) -> None:
        self.pending: list[UUID] = []
        self.skipped: list[UUID] = []
        self.failed: list[UUID] = []
        self.applied: list[dict[str, object]] = []

    async def mark_analysis_pending(self, *, owner_id, frame_id) -> None:
        self.pending.append(frame_id)

    async def mark_analysis_skipped(self, *, owner_id, frame_id, reason) -> None:
        self.skipped.append(frame_id)

    async def mark_analysis_failed(self, *, owner_id, frame_id) -> None:
        self.failed.append(frame_id)

    async def apply_analysis_result(self, **values):
        self.applied.append(values)
        return None

    async def list_recent_visual_interrupts(self, **kwargs):
        return []

    async def list_user_preferences(self, **kwargs):
        return []

    async def list_active_visual_reminders(self, **kwargs):
        return []


class _SessionRepository:
    def __init__(self, session_id: UUID) -> None:
        self._session = SimpleNamespace(
            id=session_id,
            status="active",
            behavior_instructions="",
            metadata={"active_focus": "making tea"},
        )

    async def get_session(self, owner_id, session_id):
        return self._session if session_id == self._session.id else None

    async def get_recent_messages(self, owner_id, session_id, *, limit):
        return [
            SimpleNamespace(
                role="user",
                modality="audio",
                content="I am making tea.",
                created_at=NOW,
            ),
            SimpleNamespace(
                role="user",
                modality="audio",
                content="This future transcript must not leak backward.",
                created_at=NOW + timedelta(minutes=1),
            ),
        ]


class _BehaviorRepository:
    async def get(self, *, owner_id):
        return SimpleNamespace(behavior_instructions="Keep visual updates concise.")


class _PerceptionRepository:
    def __init__(self) -> None:
        self.inserted: list[tuple[object, tuple[object, ...]]] = []

    async def get_session_working_memory(self, *, owner_id, session_id):
        raise AssertionError("pipeline context must use WorkingMemoryService")

    async def insert_observation(self, *, observation, frame_links):
        self.inserted.append((observation, tuple(frame_links)))
        return observation


class _WorkingMemoryService:
    def __init__(self, perception_repository: _PerceptionRepository) -> None:
        self._perception_repository = perception_repository
        self.applied: list[object] = []

    async def get_session_working_memory(self, *, owner_id, session_id):
        return SimpleNamespace(
            state={
                "scene": "kitchen",
                "entities": [{"entity_id": "mug-1", "name": "blue mug"}],
            }
        )

    async def apply_observation(self, observation):
        assert self._perception_repository.inserted
        assert self._perception_repository.inserted[-1][0].id == observation.id
        self.applied.append(observation)


class _FastAnalyzer:
    provider = "test_provider"
    model = "test_temporal_model"

    def __init__(self, *, fail_calls: int = 0, static: bool = False) -> None:
        self.fail_calls = fail_calls
        self.static = static
        self.requests: list[object] = []

    async def analyze(self, request):
        self.requests.append(request)
        if self.fail_calls:
            self.fail_calls -= 1
            raise RuntimeError("expected analyzer failure")
        first = request.frames[0]
        primary = request.primary_frame
        return FastPerceptionOutput.model_validate(
            {
                "summary": (
                    "A blue mug remains on the counter."
                    if self.static
                    else "The user lifts a blue mug from the counter."
                ),
                "scene": {"label": "kitchen", "confidence": 0.96},
                "activity": (
                    {"label": "lifting a mug", "confidence": 0.93}
                    if not self.static
                    else None
                ),
                "entities": [
                    {
                        "existing_entity_id": "mug-1",
                        "new_local_key": None,
                        "type": "object",
                        "name": "blue mug",
                        "state": "stationary" if self.static else "in hand",
                        "attributes": {"color": "blue"},
                        "last_visible_frame_id": str(primary.frame_id),
                        "confidence": 0.97,
                    }
                ],
                "visible_text": [
                    {
                        "text": "TEA",
                        "context": "mug lettering",
                        "frame_id": str(primary.frame_id),
                        "confidence": 0.9,
                    }
                ],
                "state_changes": (
                    []
                    if self.static
                    else [
                        {
                            "subject_local_key": "mug-1",
                            "change": "moved from counter to hand",
                            "confidence": 0.91,
                        }
                    ]
                ),
                "memory_tags": ["mug", "tea"],
                "keyframe_candidates": [
                    {
                        "frame_id": str(first.frame_id),
                        "reason": "clear initial state",
                        "score": 0.86,
                    },
                    {
                        "frame_id": str(primary.frame_id),
                        "reason": "clear final state",
                        "score": 0.9,
                    },
                ],
                "starts_new_event": False,
                "boundary_reason": None,
                "interrupt_candidate": (
                    None
                    if self.static
                    else {
                        "message": "The mug is very hot.",
                        "reason": "visible steam and unsafe grip",
                        "confidence": 0.95,
                    }
                ),
                "confidence": 0.94,
            }
        )


class _InterruptService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def maybe_interrupt(self, **values) -> None:
        self.calls.append(values)


class FastPerceptionPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_outstanding_analyses_persist_after_diagnostic_departure(self) -> None:
        harness = _Harness(static=True)
        first = harness.frame(sequence=1, captured_at=NOW)
        second = harness.frame(sequence=2, captured_at=NOW + timedelta(seconds=1))
        entered, release = asyncio.Event(), asyncio.Event()
        original_analyze = harness.fast_analyzer.analyze

        async def delayed_analyze(request):
            if request.primary_frame.frame_id == first.id:
                entered.set()
                await release.wait()
            return await original_analyze(request)

        harness.fast_analyzer.analyze = delayed_analyze
        socket = SimpleNamespace(send_json=AsyncMock(side_effect=WebSocketDisconnect()))
        sender = WebSocketEventSender(socket)
        try:
            with self.assertNoLogs("app.visual_memory.pipeline", level="INFO"):
                harness.put(first, b"first")
                await harness.pipeline.enqueue(frame=first, image_bytes=b"first", send_diagnostic=sender.send_event)
                await asyncio.wait_for(entered.wait(), 1)
                harness.put(second, b"second")
                await harness.pipeline.enqueue(frame=second, image_bytes=b"second", send_diagnostic=sender.send_event)
                release.set()
                await asyncio.wait_for(harness.pipeline.drain(), 2)
            self.assertEqual(len(harness.perception_repository.inserted), 2)
            self.assertEqual(len(harness.working_memory_service.applied), 2)
            self.assertEqual(len(harness.visual_repository.applied), 2)
            self.assertEqual(harness.visual_repository.failed, [])
            self.assertTrue(sender.closed)
            self.assertEqual(socket.send_json.await_count, 1)
        finally:
            release.set()
            await harness.pipeline.close()

    async def test_fast_pipeline_persists_temporal_structure(self) -> None:
        harness = _Harness()
        first = harness.frame(sequence=1, captured_at=NOW)
        primary = harness.frame(sequence=2, captured_at=NOW + timedelta(seconds=1))
        harness.put(first, b"first")
        harness.put(primary, b"primary")
        completed_events: list[object] = []

        async def record_event(event) -> None:
            completed_events.append(event)

        harness.event_bus.subscribe(
            DomainEventName.PERCEPTION_OBSERVATION_COMPLETED,
            record_event,
        )
        diagnostics: list[dict[str, object]] = []

        await harness.pipeline.enqueue(
            frame=primary,
            image_bytes=b"primary",
            send_diagnostic=diagnostics.append,
        )
        await harness.pipeline.drain()

        self.assertEqual(harness.visual_repository.failed, [])
        self.assertEqual(len(harness.perception_repository.inserted), 1)
        observation, links = harness.perception_repository.inserted[0]
        self.assertEqual(observation.primary_visual_frame_id, primary.id)
        self.assertEqual(observation.source, "glasses")
        self.assertEqual(observation.scene_confidence, 0.96)
        self.assertEqual(observation.activity_confidence, 0.93)
        self.assertEqual(harness.working_memory_service.applied, [observation])
        self.assertEqual(observation.summary, "The user lifts a blue mug from the counter.")
        self.assertEqual(observation.entities[0].attributes["entity_id"], "mug-1")
        self.assertEqual([link.role for link in links], ["key_candidate", "primary"])
        self.assertEqual(harness.visual_repository.applied[0]["frame_id"], primary.id)
        self.assertEqual(
            harness.visual_repository.applied[0]["frame_description"],
            observation.summary,
        )
        self.assertEqual(harness.visual_repository.applied[0]["memory_tags"], ["mug", "tea"])
        self.assertEqual(harness.interrupt_service.calls[0]["message"], "The mug is very hot.")
        self.assertEqual(
            harness.interrupt_service.calls[0]["source_received_at"],
            primary.server_received_at,
        )
        self.assertEqual(len(completed_events), 1)
        self.assertEqual(
            completed_events[0].payload.observation_id,
            observation.id,
        )
        self.assertEqual(diagnostics[-1]["type"], "visual.frame.analyzed")
        self.assertEqual(
            diagnostics[-1]["perception_observation_id"],
            str(observation.id),
        )

        request = harness.fast_analyzer.requests[0]
        self.assertEqual(request.previous_working_state["scene"], "kitchen")
        self.assertEqual(request.existing_entity_ids, frozenset({"mug-1"}))
        self.assertEqual(request.active_focus, "making tea")
        self.assertEqual(
            [item.content for item in request.recent_audio_transcripts],
            ["I am making tea."],
        )

    async def test_static_window_persists_no_false_state_change(self) -> None:
        harness = _Harness(static=True)
        first = harness.frame(sequence=1, captured_at=NOW)
        primary = harness.frame(sequence=2, captured_at=NOW + timedelta(seconds=1))
        harness.put(first, b"same-scene-one")
        harness.put(primary, b"same-scene-two")

        await harness.pipeline.enqueue(frame=primary, image_bytes=b"same-scene-two")
        await harness.pipeline.drain()

        observation, _ = harness.perception_repository.inserted[0]
        self.assertEqual(observation.state_changes, ())
        self.assertIsNone(observation.activity)
        self.assertEqual(harness.interrupt_service.calls, [])

    async def test_analyzer_failure_does_not_block_a_later_window(self) -> None:
        harness = _Harness(fail_calls=1)
        first = harness.frame(sequence=1, captured_at=NOW)
        second = harness.frame(sequence=2, captured_at=NOW + timedelta(seconds=1))
        harness.put(first, b"first")

        await harness.pipeline.enqueue(frame=first, image_bytes=b"first")
        await harness.pipeline.drain()

        harness.put(second, b"second")
        await harness.pipeline.enqueue(frame=second, image_bytes=b"second")
        await harness.pipeline.drain()

        self.assertEqual(harness.visual_repository.failed, [first.id])
        self.assertEqual(harness.visual_repository.applied[0]["frame_id"], second.id)
        self.assertEqual(len(harness.perception_repository.inserted), 1)

    async def test_phone_debug_source_is_preserved_in_observation(self) -> None:
        harness = _Harness(source="iphone_camera", static=True)
        frame = harness.frame(sequence=1, captured_at=NOW)
        harness.put(frame, b"phone")

        await harness.pipeline.enqueue(frame=frame, image_bytes=b"phone")
        await harness.pipeline.drain()

        observation, _ = harness.perception_repository.inserted[0]
        self.assertEqual(observation.source, "iphone_camera")


class SessionAnalysisCoordinatorConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_different_sessions_run_concurrently(self) -> None:
        both_started = asyncio.Event()
        release = asyncio.Event()
        active: set[str] = set()
        maximum_active = 0

        async def process(value: str) -> None:
            nonlocal maximum_active
            active.add(value)
            maximum_active = max(maximum_active, len(active))
            if len(active) == 2:
                both_started.set()
            await release.wait()
            active.remove(value)

        coordinator = SessionAnalysisCoordinator[str](process)
        await coordinator.submit(uuid4(), "session-one")
        await coordinator.submit(uuid4(), "session-two")
        await asyncio.wait_for(both_started.wait(), timeout=1)
        release.set()
        await coordinator.drain()

        self.assertEqual(maximum_active, 2)


class _Harness:
    def __init__(
        self,
        *,
        fail_calls: int = 0,
        static: bool = False,
        source: str = "glasses",
    ) -> None:
        self.owner_id = "owner"
        self.session_id = uuid4()
        self.visual_session_id = uuid4()
        self.source = source
        self.visual_repository = _VisualRepository()
        self.perception_repository = _PerceptionRepository()
        self.working_memory_service = _WorkingMemoryService(self.perception_repository)
        self.fast_analyzer = _FastAnalyzer(fail_calls=fail_calls, static=static)
        self.interrupt_service = _InterruptService()
        self.event_bus = DomainEventBus()
        self.frame_window = RecentVisualFrameWindow(
            max_frames_per_session=4,
            ttl_seconds=10,
        )
        settings = SimpleNamespace(
            JARVIS_REALTIME_RECENT_MESSAGES_LIMIT=8,
            VISUAL_FAST_PERCEPTION_TIMEOUT_SECONDS=5,
            VISUAL_FAST_PERCEPTION_MAX_WINDOW_FRAMES=4,
            VISUAL_SENSORY_WINDOW_TTL_SECONDS=10,
            VISUAL_INTERRUPT_CONFIDENCE_THRESHOLD=0.85,
        )
        self.pipeline = VisualAnalysisPipeline(
            settings=settings,
            repository=self.visual_repository,
            session_repository=_SessionRepository(self.session_id),
            behavior_repository=_BehaviorRepository(),
            interrupt_service=self.interrupt_service,
            model_limiter=PriorityAsyncLimiter(4),
            fast_analyzer=self.fast_analyzer,
            perception_repository=self.perception_repository,
            sensory_frame_window=self.frame_window,
            domain_event_bus=self.event_bus,
            working_memory_service=self.working_memory_service,
        )

    def frame(self, *, sequence: int, captured_at: datetime) -> VisualFrameRecord:
        return VisualFrameRecord(
            id=uuid4(),
            owner_id=self.owner_id,
            session_id=self.session_id,
            audio_session_id="audio-session",
            visual_session_id=self.visual_session_id,
            frame_sequence=sequence,
            source=self.source,
            captured_at=captured_at,
            server_received_at=captured_at,
            content_type="image/jpeg",
            width=768,
            height=432,
            storage_bucket="visual-frames",
            storage_path=f"frames/{sequence}.jpg",
            raw_image_status="available",
            raw_image_expires_at=captured_at + timedelta(minutes=5),
            analyzer_status="pending",
            frame_description=None,
            memory_tags=[],
            created_at=captured_at,
            updated_at=captured_at,
        )

    def put(self, frame: VisualFrameRecord, image_bytes: bytes) -> None:
        self.frame_window.put(
            SensoryVisualFrame(
                frame_id=frame.id,
                owner_id=frame.owner_id,
                session_id=frame.session_id,
                audio_session_id=frame.audio_session_id,
                visual_session_id=frame.visual_session_id,
                frame_sequence=frame.frame_sequence,
                source=frame.source,
                captured_at=frame.captured_at,
                server_received_at=frame.server_received_at,
                content_type=frame.content_type,
                width=frame.width,
                height=frame.height,
                storage_bucket=frame.storage_bucket,
                storage_path=frame.storage_path,
                raw_image_expires_at=frame.raw_image_expires_at,
                image_bytes=image_bytes,
            )
        )
