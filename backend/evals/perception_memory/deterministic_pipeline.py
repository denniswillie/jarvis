from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid5, NAMESPACE_URL

from app.memory.retrieval.candidates import CandidateCollection
from app.memory.retrieval.models import RetrievalDocument, RetrievalSourceHit
from app.memory.retrieval.query import MemoryRetrievalQuery, NormalizedMemoryRetrievalQuery
from app.memory.retrieval.service import HierarchicalMemoryRetrievalService
from app.sessions.models import JarvisSession
from app.perception.contracts import FastPerceptionOutput
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.perception.current_image import CurrentImageService
from app.perception.tools import CurrentImageToolModule
from app.realtime.tool_images import ToolImageRequest, resolve_and_deliver_tool_image
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry
from app.runtime.concurrency import PriorityAsyncLimiter
from app.realtime.visual_context import VisualConnectionKey, VisualContextState, VisualDeliveryPolicy
from app.realtime.visual_delivery import RealtimeVisualCoordinator
from app.visual_memory.interrupts import VisualInterruptService
from app.visual_memory.models import (
    RecentVisualInterrupt,
    VisualFrameRecord,
    VisualInterruptRecord,
)
from app.visual_memory.pipeline import VisualAnalysisPipeline
from app.visual_memory.validation import validate_visual_frame_event

from .models import (
    EvalScenario,
    FixtureManifest,
    MemoryLevel,
    ProviderUsage,
    ScenarioEventType,
    ScenarioObservation,
)


OWNER_ID = "perception-memory-eval"


@dataclass(frozen=True)
class DeterministicPipelineOutcome:
    observation: ScenarioObservation
    diagnostic_events: list[dict[str, object]]


class _InMemoryVisualRepository:
    def __init__(self) -> None:
        self.records: dict[UUID, VisualFrameRecord] = {}
        self.fixture_ids_by_frame_id: dict[UUID, str] = {}
        self.interrupts: list[VisualInterruptRecord] = []

    async def mark_analysis_pending(self, *, owner_id: str, frame_id: UUID) -> None:
        self.records[frame_id] = replace(self.records[frame_id], analyzer_status="pending")

    async def mark_analysis_skipped(
        self,
        *,
        owner_id: str,
        frame_id: UUID,
        reason: str,
    ) -> None:
        self.records[frame_id] = replace(self.records[frame_id], analyzer_status="skipped")

    async def mark_analysis_failed(self, *, owner_id: str, frame_id: UUID) -> None:
        self.records[frame_id] = replace(self.records[frame_id], analyzer_status="failed")

    async def apply_analysis_result(
        self,
        *,
        owner_id: str,
        frame_id: UUID,
        frame_description: str,
        memory_tags: list[str],
    ) -> VisualFrameRecord:
        record = replace(
            self.records[frame_id],
            analyzer_status="completed",
            frame_description=frame_description,
            memory_tags=memory_tags,
            updated_at=datetime.now(UTC),
        )
        self.records[frame_id] = record
        return record

    async def list_recent_visual_interrupts(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        limit: int,
    ) -> list[RecentVisualInterrupt]:
        return [
            RecentVisualInterrupt(
                id=item.id,
                visual_frame_id=item.visual_frame_id,
                message=item.message,
                reason=item.reason,
                confidence=item.confidence,
                dedupe_bucket=item.dedupe_bucket,
                created_at=item.created_at,
            )
            for item in self.interrupts[-limit:]
        ]

    async def list_user_preferences(self, *, owner_id: str, limit: int) -> list[object]:
        return []

    async def list_active_visual_reminders(self, *, owner_id: str, limit: int) -> list[object]:
        return []

    async def recent_visual_interrupt_exists(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        dedupe_bucket: str,
        window_seconds: int,
    ) -> bool:
        return any(
            interrupt.owner_id == owner_id
            and interrupt.session_id == session_id
            and interrupt.dedupe_bucket == dedupe_bucket
            for interrupt in self.interrupts
        )

    async def create_visual_interrupt(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        visual_frame_id: UUID,
        message: str,
        reason: str,
        confidence: float,
        dedupe_bucket: str,
    ) -> VisualInterruptRecord:
        record = VisualInterruptRecord(
            id=uuid5(NAMESPACE_URL, f"interrupt:{session_id}:{len(self.interrupts)}"),
            owner_id=owner_id,
            session_id=session_id,
            visual_frame_id=visual_frame_id,
            message=message,
            reason=reason,
            confidence=confidence,
            dedupe_bucket=dedupe_bucket,
            spoken_response_id=None,
            created_at=datetime.now(UTC),
        )
        self.interrupts.append(record)
        return record

    def expire_raw_images(self) -> None:
        self.records = {
            frame_id: replace(record, raw_image_status="expired")
            for frame_id, record in self.records.items()
        }

    def completed_descriptions(self) -> list[tuple[str, str]]:
        records = sorted(self.records.values(), key=lambda record: record.frame_sequence)
        return [
            (self.fixture_ids_by_frame_id[record.id], record.frame_description)
            for record in records
            if record.frame_description
        ]

class _SessionRepository:
    def __init__(self, session: JarvisSession) -> None:
        self._session = session

    async def get_session(self, owner_id: str, session_id: UUID) -> JarvisSession | None:
        return self._session if session_id == self._session.id else None

    async def get_recent_messages(
        self,
        owner_id: str,
        session_id: UUID,
        *,
        limit: int,
    ) -> list[object]:
        return []

    def close(self, *, at: datetime) -> None:
        self._session = self._session.model_copy(
            update={
                "status": "closed",
                "closed_at": at,
                "updated_at": at,
                "last_activity_at": at,
            }
        )

    def reopen(self, *, at: datetime) -> None:
        self._session = self._session.model_copy(
            update={
                "status": "active",
                "closed_at": None,
                "updated_at": at,
                "last_activity_at": at,
            }
        )


class _BehaviorRepository:
    async def get(self, *, owner_id: str) -> None:
        return None


class _DeterministicFastAnalyzer:
    provider = "deterministic"
    model = "temporal-fixture"

    def __init__(self, results: dict[UUID, FastPerceptionOutput]) -> None:
        self._results = results

    async def analyze(self, request: object) -> FastPerceptionOutput:
        await asyncio.sleep(0)
        return self._results[request.primary_frame.frame_id]


class _InMemoryPerceptionRepository:
    def __init__(self) -> None:
        self.observations: list[object] = []

    async def insert_observation(self, *, observation: object, frame_links: object) -> object:
        self.observations.append(observation)
        return observation


class _InMemoryWorkingMemoryService:
    def __init__(self) -> None:
        self.state: dict[str, object] = {}

    async def get_session_working_memory(
        self,
        *,
        owner_id: str,
        session_id: UUID,
    ) -> object:
        return SimpleNamespace(state=self.state)

    async def apply_observation(self, observation: object) -> None:
        self.state = {
            "scene": getattr(observation, "scene", None),
            "activity": getattr(observation, "activity", None),
        }


class _RealtimeSessionManager:
    def __init__(self) -> None:
        self.spoken_messages: list[str] = []

    async def speak_visual_interrupt(self, **values: object) -> bool:
        self.spoken_messages.append(str(values["message"]))
        return True


class _InMemoryRetrievalAuditRepository:
    def __init__(self) -> None:
        self.runs: list[dict[str, object]] = []

    async def record_retrieval_run(self, **values: object) -> UUID:
        self.runs.append(dict(values))
        return uuid5(NAMESPACE_URL, f"retrieval-run:{len(self.runs)}")


class _DeterministicMemoryCandidateCollector:
    def __init__(self, documents: list[RetrievalDocument]) -> None:
        self._documents = documents

    async def collect(
        self,
        *,
        owner_id: str,
        query: NormalizedMemoryRetrievalQuery,
        limit: int,
    ) -> CandidateCollection:
        documents = [
            document
            for document in self._documents
            if document.owner_id == owner_id
            and (query.session_id is None or document.session_id == query.session_id)
            and query.start_time <= document.occurred_at < query.end_time
            and (
                not query.memory_types
                or document.record_type in query.memory_types
            )
        ][:limit]
        return CandidateCollection(
            rankings={
                "time": [
                    RetrievalSourceHit(
                        document=document,
                        source="time",
                        raw_score=1.0,
                    )
                    for document in documents
                ]
            },
            provider_spaces=("deterministic:durable_experience_event",),
        )


def _durable_experience_event(
    *,
    scenario: EvalScenario,
    session_id: UUID,
    repository: _InMemoryVisualRepository,
) -> tuple[RetrievalDocument, tuple[str, ...]] | None:
    descriptions = repository.completed_descriptions()
    if not descriptions:
        return None
    fixture_ids = tuple(fixture_id for fixture_id, _ in descriptions)
    summary = " ".join(description for _, description in descriptions)
    occurred_at = min(record.captured_at for record in repository.records.values())
    return (
        RetrievalDocument(
            id=uuid5(NAMESPACE_URL, f"experience-event:{scenario.id}"),
            owner_id=OWNER_ID,
            session_id=session_id,
            record_type="experience_event",
            occurred_at=occurred_at,
            valid_from=occurred_at,
            valid_to=None,
            title=scenario.title,
            summary=summary,
            content=summary,
            importance=0.7,
            confidence=0.95,
            record_status="historical",
            is_explicit=False,
            evidence_count=len(fixture_ids),
            keyframe_count=len(fixture_ids),
            evidence_types=("perception_observation", "frame"),
        ),
        fixture_ids,
    )


async def evaluate_deterministic_scenario(
    *,
    scenario: EvalScenario,
    manifest: FixtureManifest,
    manifest_path: Path,
    simulate_legacy_gaps: bool = True,
) -> DeterministicPipelineOutcome:
    entries = {entry.id: entry for entry in manifest.entries}
    session_id = uuid5(NAMESPACE_URL, f"scenario:{scenario.id}")
    visual_session_id = uuid5(NAMESPACE_URL, f"visual:{scenario.id}")
    now = scenario.capture_timestamps[0] if scenario.capture_timestamps else datetime.now(UTC)
    session = JarvisSession(
        id=session_id,
        title=scenario.title,
        status="active",
        created_at=now,
        updated_at=now,
        closed_at=None,
        last_activity_at=now,
        behavior_instructions="",
        metadata={"eval_scenario_id": scenario.id},
    )
    repository = _InMemoryVisualRepository()
    realtime_manager = _RealtimeSessionManager()
    settings = SimpleNamespace(
        VISUAL_FRAME_MAX_BYTES=2_000_000,
        VISUAL_FRAME_MAX_WIDTH=1920,
        VISUAL_FRAME_MAX_HEIGHT=1920,
        VISUAL_INTERRUPT_CONFIDENCE_THRESHOLD=0.85,
        JARVIS_REALTIME_RECENT_MESSAGES_LIMIT=8,
        VISUAL_FAST_PERCEPTION_TIMEOUT_SECONDS=5,
        VISUAL_FAST_PERCEPTION_MAX_WINDOW_FRAMES=8,
        VISUAL_SENSORY_WINDOW_TTL_SECONDS=30,
    )
    analyzer_results: dict[UUID, FastPerceptionOutput] = {}
    records: list[tuple[VisualFrameRecord, bytes]] = []
    for index, (fixture_id, captured_at) in enumerate(
        zip(scenario.input_frame_fixture_ids, scenario.capture_timestamps, strict=True),
        start=1,
    ):
        entry = entries[fixture_id]
        frame_bytes = (manifest_path.parent / entry.path).read_bytes()
        if hashlib.sha256(frame_bytes).hexdigest() != entry.sha256:
            raise ValueError(f"fixture hash mismatch: {fixture_id}")
        validated = validate_visual_frame_event(
            {
                "type": "visual.frame.capture",
                "source": "glasses",
                "content_type": "image/jpeg",
                "frame_sequence": index,
                "captured_at": captured_at.isoformat(),
                "image_base64": base64.b64encode(frame_bytes).decode("ascii"),
                "width": entry.width,
                "height": entry.height,
            },
            settings=settings,
            allowed_sources={"glasses"},
        )
        frame_id = uuid5(NAMESPACE_URL, f"frame:{scenario.id}:{index}")
        record = VisualFrameRecord(
            id=frame_id,
            owner_id=OWNER_ID,
            session_id=session_id,
            audio_session_id=f"audio-{scenario.id}",
            visual_session_id=visual_session_id,
            frame_sequence=validated.frame_sequence,
            source=validated.source,
            captured_at=validated.captured_at,
            server_received_at=validated.captured_at,
            content_type=validated.content_type,
            width=validated.width,
            height=validated.height,
            storage_bucket="eval-fixtures",
            storage_path=entry.path,
            raw_image_status="available",
            raw_image_expires_at=validated.captured_at + timedelta(minutes=5),
            analyzer_status="pending",
            frame_description=None,
            memory_tags=[],
            created_at=validated.captured_at,
            updated_at=validated.captured_at,
        )
        description = (
            scenario.analyzer_descriptions[index - 1]
            if scenario.analyzer_descriptions
            else entry.description
        )
        interrupt = None
        if scenario.proactive_speech_expected:
            message = (
                scenario.deterministic_observation.interrupts[0]
                if scenario.deterministic_observation.interrupts
                else scenario.deterministic_observation.answer
            )
            interrupt = {
                "message": message,
                "confidence": 0.98,
                "reason": f"Deterministic trigger for {scenario.id}",
            }
        analyzer_results[frame_id] = FastPerceptionOutput.model_validate(
            {
                "summary": description,
                "scene": None,
                "activity": None,
                "entities": [],
                "visible_text": [],
                "state_changes": [],
                "memory_tags": [fixture_id],
                "keyframe_candidates": [
                    {
                        "frame_id": str(frame_id),
                        "reason": "deterministic fixture evidence",
                        "score": 0.9,
                    }
                ],
                "starts_new_event": index == 1,
                "boundary_reason": "scenario start" if index == 1 else None,
                "interrupt_candidate": interrupt,
                "prospective_matches": [],
                "confidence": 0.95,
            }
        )
        repository.records[frame_id] = record
        repository.fixture_ids_by_frame_id[frame_id] = fixture_id
        records.append((record, frame_bytes))

    session_repository = _SessionRepository(session)
    frame_window = RecentVisualFrameWindow(
        max_frames_per_session=8,
        ttl_seconds=30,
    )
    perception_repository = _InMemoryPerceptionRepository()
    working_memory_service = _InMemoryWorkingMemoryService()
    pipeline = VisualAnalysisPipeline(
        settings=settings,
        repository=repository,
        session_repository=session_repository,
        behavior_repository=_BehaviorRepository(),
        interrupt_service=VisualInterruptService(
            repository=repository,
            realtime_session_manager=realtime_manager,
            dedupe_seconds=10,
        ),
        model_limiter=PriorityAsyncLimiter(4),
        fast_analyzer=_DeterministicFastAnalyzer(analyzer_results),
        perception_repository=perception_repository,
        sensory_frame_window=frame_window,
        working_memory_service=working_memory_service,
    )
    diagnostic_events: list[dict[str, object]] = []
    analysis_started = time.perf_counter()

    async def send_diagnostic(payload: dict[str, object]) -> None:
        diagnostic_events.append(payload)

    peak_backlog = 0
    for record, frame_bytes in records:
        frame_window.put(
            SensoryVisualFrame(
                frame_id=record.id,
                owner_id=record.owner_id,
                session_id=record.session_id,
                audio_session_id=record.audio_session_id,
                visual_session_id=record.visual_session_id,
                frame_sequence=record.frame_sequence,
                source=record.source,
                captured_at=record.captured_at,
                server_received_at=record.server_received_at,
                content_type=record.content_type,
                width=record.width,
                height=record.height,
                storage_bucket=record.storage_bucket,
                storage_path=record.storage_path,
                raw_image_expires_at=record.raw_image_expires_at,
                image_bytes=frame_bytes,
            )
        )
        await pipeline.enqueue(
            frame=record,
            image_bytes=frame_bytes,
            send_diagnostic=send_diagnostic,
        )
        peak_backlog = max(peak_backlog, await pipeline.pending_count(session_id))
        await pipeline.drain()
    analysis_lag_ms = (
        (time.perf_counter() - analysis_started) * 1000 / len(records)
        if records
        else None
    )

    session_restarts = 0
    raw_expired = False
    for event in scenario.events:
        if event.type == ScenarioEventType.EXPIRE_RAW_IMAGES:
            repository.expire_raw_images()
            raw_expired = True
        elif event.type == ScenarioEventType.CLOSE_SESSION:
            session_repository.close(at=now + timedelta(seconds=event.at_seconds))
        elif event.type == ScenarioEventType.OPEN_SESSION:
            session_repository.reopen(at=now + timedelta(seconds=event.at_seconds))
            session_restarts += 1

    search_candidate_count = 0
    search_returned_count = 0
    grounded_fixture_ids = [fixture_id for fixture_id, _ in repository.completed_descriptions()]
    tool_calls = list(scenario.deterministic_observation.tool_calls)
    preparation_latency_ms = None
    if not simulate_legacy_gaps and scenario.id.startswith("shared_view_"):
        replay = await _replay_shared_view_cutoff(
            scenario, frame_window.window(OWNER_ID, session_id), session_id,
        )
        cutoff = replay["cutoff"]
        preparation_latency_ms = replay["preparation_latency_ms"]
        tool_calls = ["get_current_camera_view"] if replay["requested_image_calls"] else []
        diagnostic_events.append({
            "type": "shared_view.response_cutoff",
            **replay,
            "provider_mode": "deterministic_acknowledgements",
            "answer_is_fixture": True,
            "model_reasoning_exercised": False,
        })
        frame_ids = {
            item["frame"]["frame_id"] for item in cutoff["acknowledged_items"]
        } if cutoff["synchronized"] and cutoff["availability"]["status"] == "fresh" else set()
        # A broad analyzer caption cannot stand in for current attached pixels.
        if replay["requested_image_calls"]:
            frame_ids = set(replay["requested_image_frame_ids"])
        grounded_fixture_ids = [
            fixture_id for fixture_id, (record, _) in zip(
                scenario.input_frame_fixture_ids, records, strict=True,
            ) if str(record.id) in frame_ids
        ]
    if "search_memory" in scenario.required_tool_calls and records:
        first_capture = min(scenario.capture_timestamps)
        last_offset = max(
            [utterance.at_seconds for utterance in scenario.user_utterances]
            + [event.at_seconds for event in scenario.events]
            + [1],
        )
        durable_event = _durable_experience_event(
            scenario=scenario,
            session_id=session_id,
            repository=repository,
        )
        if durable_event is not None:
            document, fixture_ids = durable_event
            audit_repository = _InMemoryRetrievalAuditRepository()
            retrieval_service = HierarchicalMemoryRetrievalService(
                memory_repository=audit_repository,  # type: ignore[arg-type]
                perception_repository=object(),  # type: ignore[arg-type]
                candidate_collector=_DeterministicMemoryCandidateCollector([document]),  # type: ignore[arg-type]
            )
            search_end = first_capture + timedelta(seconds=last_offset + 1)
            search_response = await retrieval_service.search(
                owner_id=OWNER_ID,
                request=MemoryRetrievalQuery(
                    query=(
                        scenario.user_utterances[-1].text
                        if scenario.user_utterances
                        else scenario.title
                    ),
                    start_time=first_capture - timedelta(seconds=1),
                    end_time=search_end,
                    session_id=session_id,
                ),
                now=search_end,
            )
            search_candidate_count = search_response.candidate_count
            search_returned_count = len(search_response.results)
            returned_ids = {result.id for result in search_response.results}
            grounded_fixture_ids = (
                list(fixture_ids) if document.id in returned_ids else []
            )

    answer = scenario.deterministic_observation.answer
    selected_memory_level: MemoryLevel | None = scenario.expected_memory_level
    if simulate_legacy_gaps and scenario.pixel_detail_intentionally_omitted:
        answer = "The available visual description does not include that exact detail."
    if (
        simulate_legacy_gaps
        and session_restarts
        and scenario.expected_memory_level == MemoryLevel.SESSION
    ):
        answer = "I do not have that fact in the new session."
        selected_memory_level = MemoryLevel.NONE

    duration_seconds = max(
        [utterance.at_seconds for utterance in scenario.user_utterances]
        + [event.at_seconds for event in scenario.events]
        + [float(len(records))],
    )
    provider_usage = [
        ProviderUsage(provider="gemini", model="deterministic-mock")
        for _ in records
    ]
    if scenario.user_utterances or scenario.proactive_speech_expected:
        provider_usage.append(
            ProviderUsage(provider="openai_realtime", model="deterministic-mock")
        )
    observation = scenario.deterministic_observation.model_copy(
        update={
            "answer": answer,
            # Fixture answers have no measured time-to-first-useful-speech.
            "response_latency_ms": (
                None if preparation_latency_ms is not None
                else scenario.deterministic_observation.response_latency_ms
            ),
            "response_preparation_latency_ms": preparation_latency_ms,
            "tool_calls": tool_calls,
            "interrupts": realtime_manager.spoken_messages,
            "grounded_fixture_ids": list(dict.fromkeys(grounded_fixture_ids)),
            "frame_descriptions": [description for _, description in repository.completed_descriptions()],
            "frame_to_description_lag_ms": analysis_lag_ms,
            "current_visual_backlog_peak": peak_backlog,
            "search_candidate_count": search_candidate_count,
            "search_returned_description_count": search_returned_count,
            "selected_memory_level": selected_memory_level,
            "session_duration_seconds": duration_seconds,
            "session_restart_count": session_restarts,
            "raw_image_expired": raw_expired,
            "provider": "deterministic_pipeline",
            "provider_usage": provider_usage,
        }
    )
    return DeterministicPipelineOutcome(
        observation=observation,
        diagnostic_events=diagnostic_events,
    )


async def _replay_shared_view_cutoff(
    scenario: EvalScenario, frames: tuple[SensoryVisualFrame, ...], session_id: UUID,
) -> dict[str, object]:
    """Exercise the real bounded coordinator, not model reasoning, with fixture ACKs."""
    origin = time.monotonic()
    offset = 0.0
    first_capture = scenario.capture_timestamps[0] if scenario.capture_timestamps else datetime.now(UTC)

    def monotonic() -> float:
        return offset + time.monotonic() - origin

    def now() -> datetime:
        return first_capture + timedelta(seconds=monotonic())

    window = RecentVisualFrameWindow(max_frames_per_session=8, ttl_seconds=30, now=now, monotonic=monotonic)
    needs_image_tool = "get_current_camera_view" in scenario.required_tool_calls
    state = VisualContextState(
        connection=VisualConnectionKey(OWNER_ID, session_id, uuid5(NAMESPACE_URL, scenario.id)),
        # The requested-image trial starts without passive input. Other trials
        # still exercise automatic image delivery and an ordinary response cutoff.
        policy=VisualDeliveryPolicy(enabled=not needs_image_tool, min_interval_seconds=0.001),
    )

    class FixtureProvider:
        async def send_event(self, event):
            event = event() if callable(event) else event
            if event is None:
                return False
            if event["type"] == "conversation.item.create":
                coordinator.handle_provider_event({
                    "type": "conversation.item.done", "item": {"id": event["item"]["id"]},
                })
            else:
                coordinator.handle_provider_event({
                    "type": "conversation.item.deleted", "item_id": event["item_id"],
                })
            return True

        async def send_visual_event(self, factory, *, timeout_seconds):
            return await self.send_event(factory)

    coordinator = RealtimeVisualCoordinator(
        state=state, frame_window=window, provider=FixtureProvider(),
        max_image_bytes=2_000_000, now=now, monotonic=monotonic,
    )
    try:
        lease = None
        for frame in frames:
            offset = (frame.captured_at - first_capture).total_seconds()
            if lease is None or lease.visual_session_id != frame.visual_session_id:
                lease = window.begin_stream(OWNER_ID, session_id, frame.visual_session_id)
            window.put(frame, stream_lease=lease)
            # Drain bounded local operations, without generating speech or paying
            # for a provider. Real-model usefulness must be scored from live traces.
            for _ in range(8):
                coordinator._tick()
                if coordinator._write_task is not None:
                    await coordinator._write_task
                if state.latest_acknowledged is not None and state.latest_acknowledged.frame_id == frame.frame_id:
                    break
                await asyncio.sleep(0.001)
        offset = max(
            offset, max((turn.at_seconds for turn in scenario.user_utterances), default=offset),
        )
        preparation_started = time.perf_counter()
        snapshot = await coordinator.prepare_response(
            turn_id=scenario.id, generation_id=uuid5(NAMESPACE_URL, "response:" + scenario.id),
        )
        preparation_latency_ms = (time.perf_counter() - preparation_started) * 1000
        initial_cutoff = snapshot.to_metadata()
        requested_frames: list[str] = []
        requested_calls = 0
        image_status = None
        if needs_image_tool:
            coordinator.cancel_preparation(snapshot)
            service = CurrentImageService(
                frame_window=window, max_age_seconds=state.policy.max_context_age_seconds,
                max_image_bytes=2_000_000,
                max_width=max((frame.width for frame in frames), default=1),
                max_height=max((frame.height for frame in frames), default=1),
                now=now, monotonic=monotonic,
            )
            registry = create_tool_registry(
                modules=(CurrentImageToolModule(service),),
                context=ToolBuildContext(settings=None, owner_id=OWNER_ID, session_id=session_id),
            )
            call_id = "image:" + scenario.id
            selection = await JarvisToolExecutor(registry).execute("get_current_camera_view", {}, provider_call_id=call_id)
            requested_calls += 1
            tool = registry.get("get_current_camera_view")
            assert tool is not None and tool.image_resolver is not None
            result = await resolve_and_deliver_tool_image(
                selection, resolver=tool.image_resolver, coordinator=coordinator,
                request=ToolImageRequest(
                    connection_epoch=state.connection.connection_epoch, provider_call_id=call_id,
                    source_response_id=scenario.id, turn_generation=0, tool_chain_id=scenario.id,
                ),
                is_request_active=lambda: True, monotonic=monotonic,
            )
            image_status = result.get("image_status")
            if image_status == "attached":
                requested_frames = [result["image"]["frame_id"]]
            snapshot = await coordinator.prepare_response(
                turn_id="0", generation_id=uuid5(NAMESPACE_URL, "image-response:" + scenario.id),
                requested_call_ids=(call_id,) if image_status == "attached" else (),
            )
        return {
            "cutoff": snapshot.to_metadata(),
            "initial_cutoff": initial_cutoff,
            "preparation_latency_ms": preparation_latency_ms,
            "requested_image_calls": requested_calls,
            "requested_image_frame_ids": requested_frames,
            "image_delivery_status": image_status,
            "secondary_vision_calls": 0,
        }
    finally:
        await coordinator.close()
