"""Real perception/search/audio services for a private controlled camera replay.

Only capture timing, input pixels and the non-playing phone are fixtures. Frames,
observations, working memory and search outcomes use real owner-scoped storage.
Remote object storage is deliberately absent; no original image is overwritten.
"""
from __future__ import annotations

import asyncio
import base64
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import inspect
import time
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import asyncpg

from app.assistant.context_providers import CurrentTimeContextProvider, RecentMessagesContextProvider, SessionContextProvider
from app.assistant.memory_context import WorkingMemoryContextProvider
from app.assistant.repository import BehaviorProfileRepository
from app.experience.repository import ExperienceRepository, ToolExecutionRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.memory.prospective.context import ProspectiveContextBuilder, ProspectiveContextProvider
from app.memory.prospective.repository import ProspectiveRepository
from app.memory.prospective.service import ProspectiveMemoryService
from app.memory.prospective.tools import ProspectiveToolModule
from app.memory.repository import MemoryRepository
from app.model_providers.gemini_fast_visual import GeminiFastVisualModel
from app.perception.current_image import CurrentImageService
from app.perception.fast_analyzer import FastPerceptionAnalyzer
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.perception.repository import PerceptionRepository
from app.perception.references import PerceptionReferences, REFERENCE_CONTRACT_VERSION
from app.perception.services import WorkingMemoryService
from app.perception.tools import CurrentImageToolModule
from app.perception.working_memory import WorkingMemoryPolicy
from app.realtime.gemini.client import GeminiLiveConnection
from app.realtime.gemini.session import GeminiAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.realtime.session_manager import RealtimeSessionManager
from app.reminders.repository import ReminderRepository
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEvent, DomainEventBus, DomainEventName, VisualFrameAcceptedPayload
from app.sessions.repository import SessionRepository
from app.visual_memory.gemini_client import GeminiVisualAnalyzerClient, gemini_visual_config_from_settings
from app.visual_memory.interrupts import VisualInterruptService
from app.visual_memory.pipeline import VisualAnalysisPipeline
from app.visual_memory.repository import VisualFrameRepository
from app.visual_memory.validation import validate_visual_frame_event
from evals.glasses_reliability.gpt_turn_probe import ObservedConnection as GPTConnection
from evals.glasses_reliability.quality_probe import image_digests, native_event_summary, observe_visual_write
from evals.glasses_reliability.reference_probe import CaptureModel, inspect_result


class NativeTurnJournal:
    """Observe successful requests and terminal events, including generic alerts."""
    def __init__(self):
        self.submitted = 0
        self.completed = 0
        self.terminal_ids = set()

    def request(self):
        self.submitted += 1

    def terminal(self, response_id=None):
        if response_id is not None:
            if response_id in self.terminal_ids:
                return
            self.terminal_ids.add(response_id)
        self.completed += 1

    @property
    def pending(self):
        return self.submitted > self.completed

    def diagnostics(self):
        return {"submitted": self.submitted, "completed": self.completed,
                "balanced": self.submitted == self.completed}


class ObservedAnalyzer:
    """Observe accepted native outputs without changing prompts, parsing or claims."""
    def __init__(self, analyzer, capture_model=None):
        self.delegate = analyzer
        self.capture_model = capture_model
        self.provider, self.model = analyzer.provider, analyzer.model
        self.calls = []

    async def analyze(self, request):
        row = {"window_id": str(request.window_id), "primary_frame_id": str(request.primary_frame.frame_id),
               "started_at": time.monotonic(),
               "frames": [{"id": str(frame.frame_id), "sequence": frame.frame_sequence,
                           "captured_at": frame.captured_at.isoformat(),
                           "sha256": sha256(frame.image_bytes).hexdigest()} for frame in request.frames]}
        self.calls.append(row)
        if self.capture_model is not None:
            self.capture_model.raw = None
            self.capture_model.failed = False
        try:
            references = PerceptionReferences(request)
            row["reference_contract_version"] = REFERENCE_CONTRACT_VERSION
            row["supplied_subjects"] = {"focus": dict(references.focus), "reminders": dict(references.reminders)}
            output = await self.delegate.analyze(request)
            row["output"] = output.model_dump(mode="json")
            row["valid_output_at"] = time.monotonic()
            return output
        except Exception as error:
            row["error_type"] = type(error).__name__
            raise
        finally:
            row["finished_at"] = time.monotonic()
            if self.capture_model is not None and self.capture_model.raw is not None:
                # Private artifacts retain the exact result of this call, including
                # rejected optional candidates. This performs no second inference.
                row["raw_model_output"] = self.capture_model.raw
                try:
                    row["reference_audit"] = inspect_result(self.capture_model.raw, request)
                except Exception as error:
                    row["reference_audit_error"] = type(error).__name__


class ObservedPerceptionRepository(PerceptionRepository):
    def __init__(self, pool):
        super().__init__(pool)
        self.observations = []

    async def insert_observation(self, *, observation, frame_links):
        result = await super().insert_observation(observation=observation, frame_links=frame_links)
        self.observations.append({"id": str(observation.id), "primary_frame_id": str(observation.primary_visual_frame_id),
                                  "persisted_at": time.monotonic()})
        return result


@asynccontextmanager
async def native_search_runtime(*, fixtures, owner, session_id, provider, settings, phone, target):
    async with AsyncExitStack() as stack:
        sessions = SessionRepository(fixtures.pool)
        frames = VisualFrameRepository(fixtures.pool)
        perception = ObservedPerceptionRepository(fixtures.pool)
        prospective = ProspectiveRepository(fixtures.pool)
        manager, bus = RealtimeSessionManager(), DomainEventBus()
        bus.subscribe(DomainEventName.WORKING_MEMORY_UPDATED, manager.handle_working_memory_updated)
        bus.subscribe(DomainEventName.OWNER_WORKING_MEMORY_UPDATED, manager.handle_owner_working_memory_updated)
        bus.subscribe(DomainEventName.VISUAL_FRAME_ACCEPTED, manager.handle_visual_frame_accepted)
        working = WorkingMemoryService(repository=perception, domain_event_bus=bus, policy=WorkingMemoryPolicy(
            ttl_seconds=settings.WORKING_MEMORY_TTL_SECONDS,
            entity_ttl_seconds=settings.WORKING_MEMORY_ENTITY_TTL_SECONDS,
            visible_text_ttl_seconds=settings.WORKING_MEMORY_VISIBLE_TEXT_TTL_SECONDS,
            max_entities=settings.WORKING_MEMORY_MAX_ENTITIES,
            max_recent_changes=settings.WORKING_MEMORY_MAX_RECENT_CHANGES,
        ))
        reminders = ReminderRepository(fixtures.pool)
        context_builder = ProspectiveContextBuilder(
            focus_reader=prospective, reminder_reader=reminders, goal_reader=MemoryRepository(fixtures.pool),
            owner_working_memory_reader=working,
            max_conditional_reminders=settings.PROSPECTIVE_MAX_CONDITIONAL_REMINDERS,
            max_goals=settings.PROSPECTIVE_MAX_GOALS, max_commitments=settings.PROSPECTIVE_MAX_COMMITMENTS,
        )
        service = ProspectiveMemoryService(repository=prospective, realtime_session_manager=manager, settings=settings)
        focus = await service.set_visual_focus(owner_id=owner, target=target["name"], description=target["description"],
                                                purpose=target["purpose"], mode="find_once", expires_in_seconds=600)
        experience = ExperienceEventService(
            repository=ExperienceRepository(fixtures.pool),
            segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(
                settings.PERCEPTION_EVENT_IDLE_CLOSE_SECONDS, settings.PERCEPTION_EVENT_MAX_SECONDS)),
            working_memory_service=working, domain_event_bus=bus,
        )
        stack.push_async_callback(experience.close)
        window = RecentVisualFrameWindow(max_frames_per_session=settings.VISUAL_SENSORY_WINDOW_MAX_FRAMES,
                                         ttl_seconds=settings.VISUAL_SENSORY_WINDOW_TTL_SECONDS)
        lease = window.begin_stream(owner, session_id, uuid4())
        stack.callback(window.end_stream, lease)
        current_image = CurrentImageService(frame_window=window,
            max_age_seconds=settings.VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS,
            max_image_bytes=settings.VISUAL_FRAME_MAX_BYTES,
            max_width=settings.VISUAL_FRAME_MAX_WIDTH, max_height=settings.VISUAL_FRAME_MAX_HEIGHT)
        written_images, native_events, native_usage = [], [], []
        native_turns = NativeTurnJournal()

        class ObservedGPT(GPTConnection):
            async def send_visual_event(self, payload_factory, *, timeout_seconds):
                return await observe_visual_write(super().send_visual_event, payload_factory,
                                                   timeout_seconds=timeout_seconds, written_images=written_images)

            async def send_event(self, event):
                digests = []
                event_types = []

                async def observed():
                    payload = event() if callable(event) else event
                    if inspect.isawaitable(payload):
                        payload = await payload
                    if payload is not None:
                        digests.extend(image_digests(payload))
                        event_types.append(payload.get("type"))
                    return payload

                sent = await super().send_event(observed)
                if sent:
                    written_images.extend({"sha256": digest, "at": time.monotonic()} for digest in digests)
                    if "response.create" in event_types:
                        native_turns.request()
                return sent

            async def receive_event(self):
                event = await super().receive_event()
                if event.get("type") in {"response.created", "response.done"}:
                    response = event.get("response", {})
                    native_events.append({"at": time.monotonic(), "type": event["type"],
                                          "response_id": response.get("id"), "status": response.get("status"),
                                          "metadata": response.get("metadata")})
                    if response.get("usage"):
                        native_usage.append(response["usage"])
                    if event["type"] == "response.done":
                        native_turns.terminal(response.get("id"))
                return event

        class ObservedGemini(GeminiLiveConnection):
            async def send_event(self, event):
                digests = image_digests(event)
                result = await super().send_event(event)
                written_images.extend({"sha256": digest, "at": time.monotonic()} for digest in digests)
                if event.get("clientContent", {}).get("turnComplete"):
                    native_turns.request()
                return result

            async def receive_event(self):
                event = await super().receive_event()
                native_events.append({"at": time.monotonic(), **native_event_summary(event)})
                if event.get("usageMetadata"):
                    native_usage.append(event["usageMetadata"])
                if event.get("serverContent", {}).get("turnComplete"):
                    native_turns.terminal()
                return event

        context = RealtimeSessionContext(audio_session_id=str(uuid4()), owner_id=owner,
            jarvis_session_id=session_id, client_kind="phone_audio", session=await sessions.get_session(owner, session_id))
        extra = {"connection_factory": ObservedGemini} if provider == "gemini" else {}
        cls = GeminiAudioSession if provider == "gemini" else RealtimeAudioSession
        with patch("app.realtime.session.OpenAIRealtimeConnection", ObservedGPT):
            session = cls(websocket=phone, context=context, settings=settings, repository=sessions,
                visual_repository=frames, sensory_frame_window=window, reminder_repository=reminders,
                tool_modules=(ProspectiveToolModule(service), CurrentImageToolModule(current_image)),
                context_providers=(CurrentTimeContextProvider(), SessionContextProvider(), RecentMessagesContextProvider(12),
                                   WorkingMemoryContextProvider(working), ProspectiveContextProvider(context_builder)),
                tool_execution_repository=ToolExecutionRepository(fixtures.pool), experience_event_service=experience, **extra)
        running = asyncio.create_task(session.run())

        async def shutdown_audio():
            try:
                async with asyncio.timeout(25):
                    await session.close("search_latency_probe_complete")
            finally:
                if not running.done():
                    running.cancel()
                await asyncio.gather(running, return_exceptions=True)

        stack.push_async_callback(shutdown_audio)
        await manager.register(audio_session_id=context.audio_session_id, owner_id=owner,
                               jarvis_session_id=session_id, session=session)
        stack.push_async_callback(manager.unregister, audio_session_id=context.audio_session_id,
                                  owner_id=owner, jarvis_session_id=session_id)
        config = gemini_visual_config_from_settings(settings)
        if config is None:
            raise ValueError("The native fast visual analyzer must be configured")
        client = GeminiVisualAnalyzerClient(config)
        stack.push_async_callback(client.close)
        capture_model = CaptureModel(GeminiFastVisualModel(client))
        analyzer = ObservedAnalyzer(FastPerceptionAnalyzer(capture_model), capture_model)
        pipeline = VisualAnalysisPipeline(settings=settings, repository=frames, session_repository=sessions,
            behavior_repository=BehaviorProfileRepository(fixtures.pool),
            interrupt_service=VisualInterruptService(repository=frames, realtime_session_manager=manager,
                                                     dedupe_seconds=settings.VISUAL_INTERRUPT_DEDUPE_SECONDS),
            model_limiter=PriorityAsyncLimiter(settings.MODEL_PROVIDER_MAX_CONCURRENCY or 4),
            fast_analyzer=analyzer, perception_repository=perception, sensory_frame_window=window,
            domain_event_bus=bus, working_memory_service=working, experience_event_service=experience,
            prospective_context_builder=context_builder, prospective_service=service)
        stack.push_async_callback(pipeline.close)
        async with asyncio.timeout(35):
            while not any(e["type"] == "session.created" for e in phone.events):
                if running.done() or phone.closed:
                    raise RuntimeError("Audio session ended before readiness")
                await asyncio.sleep(.01)
        yield SimpleNamespace(owner=owner, session_id=session_id, settings=settings, frames=frames,
            perception=perception, analyzer=analyzer, prospective=prospective, service=service, focus=focus,
            phone=phone, window=window, lease=lease, bus=bus, pipeline=pipeline, session=session,
            running=running, captured=[], diagnostics=[], written_images=written_images,
            native_events=native_events, native_usage=native_usage, native_turns=native_turns)


async def submit_frame(runtime, image, *, eligible_target, phase, scheduled_offset):
    """Inject through production validation, durable frame creation and pipeline."""
    now, captured_clock = datetime.now(UTC), time.monotonic()
    sequence = len(runtime.captured) + 1
    validated = validate_visual_frame_event({
        "type": "visual.frame.capture", "source": "glasses", "frame_sequence": sequence,
        "captured_at": now.isoformat(), "content_type": "image/jpeg",
        "image_base64": base64.b64encode(image).decode(),
    }, settings=runtime.settings, allowed_sources={"glasses"})
    frame = await runtime.frames.create_pending_frame(owner_id=runtime.owner, session_id=runtime.session_id,
        audio_session_id=runtime.session.context.audio_session_id, visual_session_id=runtime.lease.visual_session_id,
        frame=validated, server_received_at=now, storage_bucket="local-evaluation-no-objects",
        storage_path=f"{runtime.owner}/{sequence}.jpg", raw_image_expires_at=now + timedelta(minutes=5))
    # No storage upload is fabricated. The durable row keeps raw_image_status=failed;
    # immediate perception uses the actual owned in-memory pixels, as its API allows.
    admitted = runtime.window.put(SensoryVisualFrame(frame_id=frame.id, owner_id=runtime.owner,
        session_id=runtime.session_id, audio_session_id=frame.audio_session_id,
        visual_session_id=frame.visual_session_id, frame_sequence=sequence, source="glasses",
        captured_at=now, server_received_at=now, content_type="image/jpeg", width=frame.width,
        height=frame.height, storage_bucket=frame.storage_bucket, storage_path=frame.storage_path,
        raw_image_expires_at=frame.raw_image_expires_at, image_bytes=image), stream_lease=runtime.lease)
    if not admitted:
        raise RuntimeError("Fixture frame did not belong to its stream")
    runtime.captured.append({"id": str(frame.id), "sequence": sequence, "captured_at": now.isoformat(),
        "at": captured_clock, "eligible_target": eligible_target, "phase": phase,
        "scheduled_offset": scheduled_offset, "sha256": sha256(image).hexdigest()})
    await runtime.bus.publish(DomainEvent(name=DomainEventName.VISUAL_FRAME_ACCEPTED,
        payload=VisualFrameAcceptedPayload(runtime.owner, runtime.session_id, frame.id, sequence)))

    async def diagnostic(event):
        runtime.diagnostics.append({**event, "at": time.monotonic()})

    await runtime.pipeline.enqueue(frame=frame, image_bytes=image, send_diagnostic=diagnostic)


async def read_durable_state(fixtures, *, owner, session_id, focus_id):
    if owner not in fixtures.owners:
        raise ValueError("Only manifested fixture owners may be read")
    connection = await asyncpg.connect(fixtures.dsn, timeout=15, command_timeout=20)
    try:
        result = {}
        for name, sql in {
            "frames": "select id, owner_id, session_id, frame_sequence, analyzer_status, raw_image_status from visual_frames where owner_id=$1 and session_id=$2 order by frame_sequence",
            "observations": "select id, primary_visual_frame_id, summary from perception_observations where owner_id=$1 and session_id=$2 order by created_at,id",
            "deliveries": "select id, owner_id, session_id, visual_focus_id, event_type, provider_response_id, metadata, evidence from jarvis_prospective_delivery_events where owner_id=$1 and session_id=$2 order by created_at,id",
            "generic_alerts": "select id, visual_frame_id, spoken_response_id, message from visual_interrupts where owner_id=$1 and session_id=$2 order by created_at,id",
            "executions": "select tool_name, status, mutates_state from jarvis_tool_executions where owner_id=$1 and session_id=$2 order by started_at,id",
            "working_memory": "select revision, state from jarvis_session_working_memory where owner_id=$1 and session_id=$2",
        }.items():
            result[name] = [dict(row) for row in await connection.fetch(sql, owner, session_id)]
        focus = await connection.fetchrow("select id, mode, completed_at, cleared_at from jarvis_visual_focuses where owner_id=$1 and id=$2", owner, focus_id)
        result["focus"] = dict(focus) if focus else None
        return result
    finally:
        await connection.close()
