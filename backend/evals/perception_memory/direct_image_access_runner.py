"""Opt-in live acceptance trials through Jarvis's real WebSocket routes.

Expected answers remain in the driver/report, never in session context. Each
attempt has its own owner, database rows, storage paths, sockets and report.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import secrets
import socket
import statistics
import time
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import asyncpg
import uvicorn
import websockets
from fastapi import FastAPI
from openai import AsyncOpenAI

from app.db import create_pool
from app.experience.repository import ExperienceRepository
from app.files.storage import JarvisFileStorageClient
from app.main import _create_assistant_context_providers, _create_tool_modules
from app.memory.repository import MemoryRepository
from app.memory.retrieval.candidates import RetrievalCandidateCollector
from app.memory.retrieval.evidence_repository import HistoricalEvidenceRepository
from app.memory.retrieval.image_evidence import HistoricalImageService
from app.memory.retrieval.service import HierarchicalMemoryRetrievalService
from app.memory.service import MemoryApplicationService
from app.perception.current_image import CurrentImageService
from app.perception.frame_window import RecentVisualFrameWindow
from app.perception.repository import PerceptionRepository
from app.perception.services import WorkingMemoryService
from app.perception.working_memory import WorkingMemoryPolicy
from app.realtime.router import router as audio_router
from app.realtime.session_manager import RealtimeSessionManager
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEventBus, DomainEventName
from app.runtime.services import JarvisRuntimeServices
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.storage.supabase_config import get_supabase_storage_config
from app.tools.invocation import current_tool_invocation
from app.tools.module import FunctionToolModule
from app.visual_memory.latest_frame_buffer import LatestVisualFrameBuffer
from app.visual_memory.router import router as visual_router
from app.visual_memory.storage import VisualFrameStorageClient
from .direct_image_scenes import render_scene
from .fixture_cleanup import cleanup_fixture_owners


SCENARIOS_PATH = Path(__file__).with_name("direct_image_access_scenarios.json")


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[dict]:
    document = json.loads(path.read_text())
    if document["contract_version"] != "direct-image-access-v1":
        raise ValueError("Unsupported direct-image scenario contract.")
    cases = document["scenarios"]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Scenario IDs must be unique.")
    return cases


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


class Trace:
    def __init__(self):
        self.origin = time.monotonic()
        self.entries: list[dict] = []
        self.changed = asyncio.Event()

    def add(self, direction: str, kind: str, **values) -> dict:
        row = {"direction": direction, "type": kind, "at": time.monotonic() - self.origin, **values}
        self.entries.append(row)
        self.changed.set()
        return row

    async def until(self, predicate, *, timeout=45):
        async with asyncio.timeout(timeout):
            while not predicate():
                self.changed.clear()
                if predicate():
                    break
                try:
                    await asyncio.wait_for(self.changed.wait(), timeout=0.1)
                except TimeoutError:
                    pass


class FixtureManifest:
    def __init__(self, directory: Path, owner: str):
        self.directory = directory
        self.owner = owner
        self.rows: list[dict] = []
        self.objects: set[tuple[str, str]] = set()
        self.save()

    def row(self, table: str, row_id):
        self.rows.append({"table": table, "id": str(row_id)})
        self.save()

    def object(self, bucket: str, path: str):
        if not path.startswith(f"owners/{self.owner}/"):
            raise ValueError("Refusing an object outside this fixture owner.")
        self.objects.add((bucket, path))
        self.save()

    def save(self):
        write_json(self.directory / "fixture_manifest.json", {
            "owner_id": self.owner, "explicit_rows": self.rows,
            "objects": [{"bucket": bucket, "path": path} for bucket, path in sorted(self.objects)],
            "row_cleanup_scope": "All public rows with this exact unique owner; dependent rows follow their foreign keys.",
        })


class TrackedVisualStorage(VisualFrameStorageClient):
    def __init__(self, config, manifest, trace):
        super().__init__(config)
        self.manifest, self.trace = manifest, trace
        self.hold = False
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    def build_storage_path(self, **kwargs):
        path = super().build_storage_path(**kwargs)
        self.manifest.object(self.bucket, path)
        return path

    async def download_object(self, **kwargs):
        identity = hashlib.sha256(kwargs["path"].encode()).hexdigest()
        self.trace.add("storage", "download.started", object_hash=identity)
        try:
            if self.hold:
                self.hold = False
                self.entered.set()
                await self.release.wait()
            return await super().download_object(**kwargs)
        finally:
            self.trace.add("storage", "download.finished", object_hash=identity)


class TrackedFileStorage(JarvisFileStorageClient):
    def __init__(self, config, manifest):
        super().__init__(config)
        self.manifest = manifest

    def build_file_path(self, **kwargs):
        path = super().build_file_path(**kwargs)
        self.manifest.object(self.bucket, path)
        return path

    async def upload_bytes(self, **kwargs):
        self.manifest.object(self.bucket, kwargs["path"])
        return await super().upload_bytes(**kwargs)


class TracedProviderSocket:
    def __init__(self, underlying, trial):
        self.underlying, self.trial = underlying, trial

    async def send(self, raw):
        event = json.loads(raw)
        trial = self.trial
        item = event.get("item", {})
        values = {}
        if event["type"] == "response.create":
            if trial.force_queue:
                tool = trial.force_queue.pop(0)
                event.setdefault("response", {})["tool_choice"] = {"type": "function", "name": tool}
                values["forced_tool"] = tool
            snapshot = trial.manager.session._visual_context.active_snapshot
            values["camera_stream_active"] = trial.manager.session._visual_prompt_metadata(snapshot)["camera_stream_active"]
            values["instructions_sha256"] = hashlib.sha256(event["response"]["instructions"].encode()).hexdigest()
            values["requested_item_ids"] = [pin.item_id for pin in snapshot.requested_images] if snapshot else []
            values["cutoff"] = snapshot.to_metadata() if snapshot else None
        images = [part for part in item.get("content", []) if part.get("type") == "input_image"]
        if images:
            pixels = base64.b64decode(images[0]["image_url"].split(",", 1)[1])
            values.update(item_id=item["id"], image_sha256=hashlib.sha256(pixels).hexdigest())
            state_item = next((entry for entry in trial.manager.session._visual_context.items if entry.item_id == item["id"]), None)
            if state_item and state_item.image_reference is not None:
                values["image"] = state_item.image_reference.to_metadata()
                values["turn_generation"] = state_item.turn_generation
            if trial.case.get("reject_image") and not trial.rejection_used:
                trial.rejection_used = True
                images[0]["image_url"] = "data:image/jpeg;base64,AA=="
                values["injected_invalid_wire_image"] = True
        if item.get("type") == "function_call_output":
            result = json.loads(item["output"])
            values.update(
                call_id=item["call_id"], image_status=result.get("image_status"),
                provider_item_id=result.get("provider_item_id"), status=result.get("status"),
                error_code=result.get("error", {}).get("code"),
            )
            if isinstance(result.get("image"), dict):
                values["image"] = result["image"]
        trial.trace.add("send", event["type"], **values)
        await self.underlying.send(json.dumps(event))

    async def recv(self):
        raw = await self.underlying.recv()
        event = json.loads(raw)
        kind = event.get("type", "")
        if kind in {
            "session.created", "session.updated", "conversation.item.created", "conversation.item.done",
            "response.created", "response.done", "response.function_call_arguments.done",
            "conversation.item.input_audio_transcription.completed", "error",
        }:
            values = {
                "item_id": event.get("item", {}).get("id", event.get("item_id")),
                "response_id": event.get("response", {}).get("id", event.get("response_id")),
                "tool_name": event.get("name"), "call_id": event.get("call_id"),
                "response_status": event.get("response", {}).get("status"),
                "error_code": event.get("error", {}).get("code"),
            }
            if kind == "response.function_call_arguments.done":
                try:
                    arguments = json.loads(event.get("arguments", "{}"))
                    values["record_arguments"] = {key: arguments[key] for key in ("record_id", "record_type", "frame_id") if key in arguments}
                except (ValueError, TypeError):
                    values["invalid_arguments"] = True
            self.trial.trace.add("receive", kind, **values)
        return raw

    async def close(self):
        await self.underlying.close()


class TrialManager(RealtimeSessionManager):
    def __init__(self, trial):
        super().__init__()
        self.trial = trial
        self.session = None

    async def register(self, **kwargs):
        accepted = await super().register(**kwargs)
        if accepted:
            self.session = kwargs["session"]
            original_connect = self.session.openai.connect

            async def connect():
                await original_connect()
                self.session.openai._websocket = TracedProviderSocket(self.session.openai._websocket, self.trial)

            self.session.openai.connect = connect
        return accepted


def instrument_module(module, trace):
    def handler_for(tool):
        async def handler(arguments):
            call_id = current_tool_invocation().provider_call_id
            trace.add("domain", "selection.started", tool_name=tool.name, call_id=call_id)
            try:
                result = await tool.handler(arguments)
                trace.add("domain", "selection.finished", tool_name=tool.name, call_id=call_id, status=result.get("status"), image=result.get("image"))
                return result
            except BaseException as error:
                trace.add("domain", "selection.failed", tool_name=tool.name, call_id=call_id, error_type=type(error).__name__)
                raise
        return handler

    def resolver_for(tool):
        async def resolve(selection):
            trace.add("domain", "resolution.started", tool_name=tool.name, image=selection.get("image"))
            try:
                return await tool.image_resolver(selection)
            finally:
                trace.add("domain", "resolution.finished", tool_name=tool.name)
        return resolve

    def build(context):
        return tuple(replace(
            tool, handler=handler_for(tool),
            image_resolver=resolver_for(tool) if tool.image_resolver is not None else None,
        ) for tool in module.build_tools(context))
    return FunctionToolModule(module.name, build)


class LiveTrial:
    def __init__(self, *, settings, case, strategy, directory, questions, passive=False):
        self.case, self.strategy, self.directory, self.questions = case, strategy, directory, questions
        self.owner = "direct-image-eval-" + uuid4().hex
        self.settings = settings.model_copy(update={
            "JARVIS_OWNER_ID": self.owner, "JARVIS_STATIC_BEARER_TOKEN": secrets.token_urlsafe(24),
            "GEMINI_API_KEY": None, "VISUAL_FAST_PERCEPTION_ENABLED": False,
            "VISUAL_REALTIME_ENABLED": bool(passive or case.get("passive", False)),
            "SUPABASE_DB_POOL_MIN_SIZE": 1, "SUPABASE_DB_POOL_MAX_SIZE": 2,
            "JARVIS_AUDIO_FORMAT": "pcm16", "JARVIS_AUDIO_SAMPLE_RATE": 24000,
        })
        self.trace = Trace()
        self.manifest = FixtureManifest(directory, self.owner)
        self.manager = TrialManager(self)
        self.force_queue = []
        self.rejection_used = False
        self.sockets, self.tasks, self.turns = [], [], []
        self.image_hashes = set()
        self.target_hash = None
        self.pool = self.runtime = self.server = self.server_task = None
        self.visual_storage = self.file_storage = None
        self.audio = self.current_visual = None
        self.patches = ExitStack()
        self.sequence = 0

    async def setup(self):
        self.pool = await create_pool(self.settings)
        config = get_supabase_storage_config(self.settings)
        self.visual_storage = TrackedVisualStorage(config, self.manifest, self.trace)
        self.file_storage = TrackedFileStorage(config, self.manifest)
        window = RecentVisualFrameWindow(
            max_frames_per_session=self.settings.VISUAL_SENSORY_WINDOW_MAX_FRAMES,
            ttl_seconds=self.settings.VISUAL_SENSORY_WINDOW_TTL_SECONDS,
        )
        current = CurrentImageService(
            frame_window=window, max_age_seconds=self.settings.VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS,
            max_image_bytes=self.settings.VISUAL_FRAME_MAX_BYTES,
            max_width=self.settings.VISUAL_FRAME_MAX_WIDTH, max_height=self.settings.VISUAL_FRAME_MAX_HEIGHT,
        )
        historical = HistoricalImageService(
            repository=HistoricalEvidenceRepository(self.pool), image_loader=self.visual_storage,
            max_image_bytes=self.settings.VISUAL_FRAME_MAX_BYTES,
            max_width=self.settings.VISUAL_FRAME_MAX_WIDTH, max_height=self.settings.VISUAL_FRAME_MAX_HEIGHT,
            request_timeout_seconds=self.settings.VISUAL_IMAGE_RETRIEVAL_TIMEOUT_SECONDS,
            max_frames=self.settings.MEMORY_REINSPECTION_MAX_KEYFRAMES,
        )
        memory_repository = MemoryRepository(self.pool)
        perception_repository = PerceptionRepository(self.pool)
        bus = DomainEventBus()
        working = WorkingMemoryService(
            repository=perception_repository,
            policy=WorkingMemoryPolicy(
                ttl_seconds=self.settings.WORKING_MEMORY_TTL_SECONDS,
                entity_ttl_seconds=self.settings.WORKING_MEMORY_ENTITY_TTL_SECONDS,
                visible_text_ttl_seconds=self.settings.WORKING_MEMORY_VISIBLE_TEXT_TTL_SECONDS,
                max_entities=self.settings.WORKING_MEMORY_MAX_ENTITIES,
                max_recent_changes=self.settings.WORKING_MEMORY_MAX_RECENT_CHANGES,
            ), domain_event_bus=bus,
        )
        retrieval = HierarchicalMemoryRetrievalService(
            memory_repository=memory_repository, perception_repository=perception_repository,
            candidate_collector=RetrievalCandidateCollector(
                memory_repository=memory_repository, perception_repository=perception_repository, embedding_model=None,
            ), image_service=historical,
        )
        memory = MemoryApplicationService(
            repository=memory_repository, retrieval_service=retrieval,
            session_repository=SessionRepository(self.pool), frame_window=window,
            working_memory_service=working, embedding_job_repository=None,
            current_frame_max_age_seconds=self.settings.VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS,
        )
        self.runtime = JarvisRuntimeServices(
            settings=self.settings, realtime_session_manager=self.manager,
            sensory_frame_window=window, latest_visual_frame_buffer=LatestVisualFrameBuffer(
                ttl_seconds=self.settings.VISUAL_LATEST_FRAME_TTL_SECONDS, frame_window=window,
            ), domain_event_bus=bus, model_limiter=PriorityAsyncLimiter(1),
            db_pool=self.pool, visual_frame_storage=self.visual_storage, jarvis_file_storage=self.file_storage,
            current_image_service=current, historical_image_service=historical,
            working_memory_service=working, memory_retrieval_service=retrieval, memory_service=memory,
        )
        self.runtime.tool_modules = tuple(instrument_module(module, self.trace) for module in _create_tool_modules(self.runtime))
        self.runtime.assistant_context_providers = _create_assistant_context_providers(self.runtime)

        async def accepted(event):
            session = self.manager.session
            if session is not None and event.payload.session_id == session.context.jarvis_session_id:
                session.notify_visual_frame_available()
        bus.subscribe(DomainEventName.VISUAL_FRAME_ACCEPTED, accepted)
        app = FastAPI()
        app.state.runtime_services = self.runtime
        app.include_router(audio_router)
        app.include_router(visual_router)
        for module in ("app.realtime.router", "app.visual_memory.router"):
            self.patches.enter_context(patch(module + ".get_settings", return_value=self.settings))
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        self.base = f"ws://127.0.0.1:{listener.getsockname()[1]}"
        self.server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="warning", access_log=False))
        self.server_task = asyncio.create_task(self.server.serve(sockets=[listener]))
        await self.trace.until(lambda: self.server.started or self.server_task.done(), timeout=10)
        if self.server_task.done():
            await self.server_task
            raise RuntimeError("Evaluation server stopped before accepting connections.")

    async def session_row(self):
        async with self.pool.acquire() as connection:
            identity = await connection.fetchval(
                "insert into jarvis_sessions (owner_id, title) values ($1, 'Disposable direct-image trial') returning id",
                self.owner,
            )
        self.manifest.row("jarvis_sessions", identity)
        return identity

    async def connect_socket(self, path):
        client = await websockets.connect(
            self.base + path,
            additional_headers={"Authorization": "Bearer " + self.settings.JARVIS_STATIC_BEARER_TOKEN},
            max_size=None,
        )
        self.sockets.append(client)
        return client

    async def visual_stream(self, session_id):
        client = await self.connect_socket(f"/realtime/visual/ws?session_id={session_id}&visual_session_id={uuid4()}")
        event = json.loads(await client.recv())
        if event["type"] != "visual.session.created":
            raise RuntimeError("Visual route did not open the fixture stream.")
        self.trace.add("ingress", event["type"], session_id=str(session_id))
        return client

    async def capture(self, client, pixels, sequence):
        digest = hashlib.sha256(pixels).hexdigest()
        self.image_hashes.add(digest)
        await client.send(json.dumps({
            "type": "visual.frame.capture", "frame_sequence": sequence, "source": "glasses",
            "captured_at": datetime.now(UTC).isoformat(), "content_type": "image/jpeg",
            "image_base64": base64.b64encode(pixels).decode(), "width": 640, "height": 480,
        }))
        while True:
            event = json.loads(await client.recv())
            if event["type"] == "visual.frame.accepted":
                self.manifest.row("visual_frames", event["frame_id"])
                self.trace.add("ingress", event["type"], frame_id=event["frame_id"], image_sha256=digest)
                return UUID(event["frame_id"])
            if event["type"] in {"error", "visual.frame.dropped"}:
                raise RuntimeError("Frame ingestion failed: " + event.get("code", "unknown"))

    async def audio_receiver(self):
        async for raw in self.audio:
            event = json.loads(raw)
            kind = event["type"]
            values = {key: event[key] for key in ("response_id", "item_id", "audio_session_id", "text", "role", "code") if key in event}
            if kind == "audio.output.delta":
                values["audio_bytes"] = len(base64.b64decode(event.get("audio_base64", "")))
            self.trace.add("client", kind, **values)

    async def connect_audio(self, session_id):
        self.audio = await self.connect_socket(f"/realtime/audio/ws?session_id={session_id}&client_kind=rayban_audio")
        self.tasks.append(asyncio.create_task(self.audio_receiver()))
        await self.trace.until(lambda: any(
            row["direction"] == "client" and row["type"] in {"session.created", "error"} for row in self.trace.entries
        ), timeout=20)
        if not any(row["direction"] == "client" and row["type"] == "session.created" for row in self.trace.entries):
            raise RuntimeError("Audio route setup failed.")

    async def start_question(self, question, *, force=()):
        self.force_queue = list(force) if self.strategy == "forced" else []
        marker = self.trace.add("driver", "question.started", question=question)
        pcm = self.questions[question]
        for offset in range(0, len(pcm), 4800):
            await self.audio.send(json.dumps({
                "type": "audio.input.append", "audio_base64": base64.b64encode(pcm[offset:offset + 4800]).decode(),
            }))
        end = self.trace.add("driver", "utterance.ended")
        await self.audio.send(json.dumps({"type": "audio.input.commit"}))
        await self.audio.send(json.dumps({"type": "response.create"}))
        return marker, end

    def completed(self, marker):
        rows = [row for row in self.trace.entries if row["at"] >= marker["at"] and row["direction"] == "client"]
        session = self.manager.session
        if session is None or session._normal_turn_pending or session._tool_followup is not None or session._image_tool_tasks:
            return False
        successful_ids = {
            row.get("response_id") for row in self.trace.entries
            if row["at"] >= marker["at"] and row["direction"] == "receive"
            and row["type"] == "response.done" and row.get("response_status") == "completed"
        }
        done_ids = {row.get("response_id") for row in rows if row["type"] == "response.done"} & successful_ids
        answers = [row for row in rows if row["type"] == "transcript.done" and row.get("response_id") in done_ids and row.get("text")]
        return bool(
            answers and any(row["type"] == "user.transcript.done" for row in rows)
            and any(row["type"] == "audio.output.done" and row.get("response_id") == answers[-1]["response_id"] for row in rows)
        )

    def failed_response(self, marker):
        session = self.manager.session
        if session is not None and (session._normal_turn_pending or session._tool_followup is not None):
            # A bounded application failure notice may still be in flight.
            return None
        rows = [row for row in self.trace.entries if row["at"] >= marker["at"]]
        started = [
            row.get("response_id") for row in rows
            if row["direction"] == "receive" and row["type"] == "response.created"
        ]
        settled = {
            row.get("response_id") for row in rows
            if row["direction"] == "client" and row["type"] == "response.done"
        }
        return next((row for row in reversed(rows)
                     if row["direction"] == "receive" and row["type"] == "response.done"
                     and started and row.get("response_id") == started[-1]
                     and row.get("response_id") in settled
                     and row.get("response_status") in {"failed", "cancelled", "incomplete"}), None)

    async def finish_question(self, marker, end):
        await self.trace.until(lambda: self.failed_response(marker) is not None or self.completed(marker))
        rows = [row for row in self.trace.entries if row["at"] >= marker["at"] and row["direction"] == "client"]
        failure = self.failed_response(marker)
        if failure is not None:
            result = {
                "question": marker["question"],
                "transcription": [row.get("text") for row in rows if row["type"] == "user.transcript.done"],
                "answer": None, "response_id": failure["response_id"],
                "response_status": failure["response_status"],
                "partial_transcripts": [row.get("text") for row in rows
                                        if row["type"] == "transcript.done"
                                        and row.get("response_id") == failure["response_id"]],
                "first_final_response_audio_ms": None, "answer_complete_ms": None,
                "terminal_response_ms": 1000 * (failure["at"] - end["at"]),
                "first_substantive_audio_ms": None, "listening_review_required": False,
            }
            self.turns.append(result)
            return result
        done_ids = {row.get("response_id") for row in rows if row["type"] == "response.done"}
        answer = [row for row in rows if row["type"] == "transcript.done" and row.get("response_id") in done_ids and row.get("text")][-1]
        final_id = answer["response_id"]
        audio = [row for row in rows if row["type"] == "audio.output.delta" and row.get("response_id") == final_id and row.get("audio_bytes", 0)]
        completed = [row for row in rows if row["type"] == "response.done" and row.get("response_id") == final_id][-1]
        result = {
            "question": marker["question"], "transcription": [row.get("text") for row in rows if row["type"] == "user.transcript.done"],
            "answer": answer["text"], "response_id": final_id, "response_status": "completed",
            "provider_failures": [
                {"response_id": row.get("response_id"), "status": row["response_status"]}
                for row in self.trace.entries if row["at"] >= marker["at"]
                and row["direction"] == "receive" and row["type"] == "response.done"
                and row.get("response_status") == "failed"
            ],
            "first_final_response_audio_ms": 1000 * (audio[0]["at"] - end["at"]) if audio else None,
            "answer_complete_ms": 1000 * (completed["at"] - end["at"]),
            "first_substantive_audio_ms": None, "listening_review_required": True,
        }
        self.turns.append(result)
        return result

    async def ask(self, question, *, force=()):
        marker, end = await self.start_question(question, force=force)
        return await self.finish_question(marker, end)

    async def seed_historical_memory(self, session_id, frame_id):
        now = datetime.now(UTC)
        experience = ExperienceRepository(self.pool)
        event = await experience.get_or_create_open_event(
            owner_id=self.owner, session_id=session_id, event_kind="visual",
            started_at=now, modalities=("visual",),
        )
        self.manifest.row("experience_events", event.id)
        await experience.select_keyframe(
            owner_id=self.owner, session_id=session_id, event_id=event.id,
            visual_frame_id=frame_id, rank=1, reason="Historical evaluation evidence",
            quality_score=1.0, selected_by="deterministic",
        )
        await experience.close_event(
            owner_id=self.owner, session_id=session_id, event_id=event.id,
            ended_at=now, boundary_reason="direct_image_evaluation",
            keyframes_finalized_at=now, consolidation_ready_at=now,
        )
        repository = MemoryRepository(self.pool)
        memory = await repository.promote_event_to_episode(
            owner_id=self.owner, event_id=event.id,
            title="Earlier board observation", content="An earlier board showed a short printed code.",
            importance=0.7, confidence=0.8, source="direct_image_evaluation",
            promotion_reason="Retain historical board evidence for an isolated evaluation",
        )
        self.manifest.row("memory_items", memory.id)
        await repository.attach_evidence(
            owner_id=self.owner, memory_item_id=memory.id, evidence_type="frame",
            evidence_ref_id=frame_id, evidence_time=now, confidence=1.0,
        )

    async def exercise(self):
        if self.case.get("historical_scene"):
            previous_session = await self.session_row()
            previous_stream = await self.visual_stream(previous_session)
            historical_pixels = render_scene(self.case["historical_scene"])
            self.target_hash = hashlib.sha256(historical_pixels).hexdigest()
            frame_id = await self.capture(previous_stream, historical_pixels, 1)
            await previous_stream.close()
            await self.seed_historical_memory(previous_session, frame_id)
            if self.case.get("expired"):
                async with self.pool.acquire() as connection:
                    await connection.execute(
                        "update visual_frames set raw_image_status = 'expiring', raw_image_expires_at = now() - interval '1 second' where id = $1 and owner_id = $2",
                        frame_id, self.owner,
                    )
        session_id = await self.session_row()
        await self.connect_audio(session_id)
        if self.case.get("scene"):
            self.current_visual = await self.visual_stream(session_id)
            if self.case.get("initial_scene"):
                await self.capture(self.current_visual, render_scene(self.case["initial_scene"]), 1)
                self.sequence = 1
                await self.ask(self.case["setup_question"], force=("get_current_camera_view",))
            self.sequence += 1
            pixels = render_scene(self.case["scene"])
            await self.capture(self.current_visual, pixels, self.sequence)
            if self.target_hash is None:
                self.target_hash = hashlib.sha256(pixels).hexdigest()
            if self.case.get("end_stream"):
                await self.current_visual.close()
                await self.trace.until(lambda: self.runtime.sensory_frame_window.stream_lease(self.owner, session_id) is None)
            elif self.settings.VISUAL_REALTIME_ENABLED:
                digest = hashlib.sha256(pixels).hexdigest()
                await self.trace.until(lambda: any(
                    sent.get("image_sha256") == digest and any(
                        ack["type"] == "conversation.item.done" and ack["direction"] == "receive"
                        and ack.get("item_id") == sent.get("item_id") for ack in self.trace.entries
                    ) for sent in self.trace.entries
                ))
        if self.case.get("barge_in"):
            self.visual_storage.hold = True
            await self.start_question(self.case["question"], force=self.case.get("force", ()))
            await self.trace.until(self.visual_storage.entered.is_set)
            generation = self.manager.session._turn_generation
            self.trace.add("driver", "barge_in", cancelled_generation=generation)
            marker, end = await self.start_question(self.case["replacement_question"])
            await self.trace.until(lambda: self.manager.session._turn_generation > generation)
            self.visual_storage.release.set()
            await self.finish_question(marker, end)
        else:
            await self.ask(self.case["question"], force=self.case.get("force", ()))
        if self.case.get("followup_question"):
            if self.case["id"] == "L10":
                self.sequence += 1
                await self.capture(self.current_visual, render_scene(self.case["scene"]), self.sequence)
            await self.ask(
                self.case["followup_question"],
                force=("get_current_camera_view",) if self.case["id"] == "L10" else (),
            )

    async def cleanup(self):
        errors = []
        if self.visual_storage is not None:
            self.visual_storage.release.set()
        for client in reversed(self.sockets):
            try:
                await client.close()
            except Exception as error:
                errors.append(type(error).__name__)
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.server is not None:
            self.server.should_exit = True
        if self.server_task is not None:
            try:
                await asyncio.wait_for(self.server_task, timeout=10)
            except Exception as error:
                errors.append(type(error).__name__)
        self.patches.close()
        storage_results = []
        try:
            if self.pool is not None:
                for bucket, path in sorted(self.manifest.objects):
                    try:
                        async with self.pool.acquire() as connection:
                            exists = await connection.fetchval(
                                "select exists(select 1 from storage.objects where bucket_id = $1 and name = $2)",
                                bucket, path,
                            )
                        if exists:
                            await self.visual_storage.delete_objects(bucket=bucket, paths=[path])
                        async with self.pool.acquire() as connection:
                            remains = await connection.fetchval(
                                "select exists(select 1 from storage.objects where bucket_id = $1 and name = $2)",
                                bucket, path,
                            )
                        storage_results.append({"bucket": bucket, "path": path, "remaining": bool(remains)})
                    except Exception as error:
                        errors.append(type(error).__name__)
                        storage_results.append({"bucket": bucket, "path": path, "remaining": "unknown"})
            database = await cleanup_fixture_owners(self.settings.SUPABASE_DB_URL, {self.owner})
        except Exception as error:
            errors.append(type(error).__name__)
            database = {"remaining_rows": "unknown"}
        finally:
            if self.runtime is not None:
                await self.runtime.close()
            else:
                if self.visual_storage is not None:
                    await self.visual_storage.close()
                if self.file_storage is not None:
                    await self.file_storage.close()
                if self.pool is not None:
                    await self.pool.close()
        return {
            "database": database, "objects": storage_results, "errors": errors,
            "passed": not errors and database["remaining_rows"] == 0 and all(item["remaining"] is False for item in storage_results),
        }

    async def run(self):
        failure = None
        try:
            async with asyncio.timeout(100):
                await self.setup()
                await self.exercise()
        except Exception as error:
            failure = {"category": "driver_or_infrastructure_failure", "type": type(error).__name__}
        finally:
            cleanup = await self.cleanup()
        gates = transport_gates(self.trace.entries, self.image_hashes)
        gates["all_turns_completed"] = bool(self.turns) and all(
            turn["response_status"] == "completed" and not turn.get("provider_failures") for turn in self.turns
        )
        negative = any(self.case.get(key) for key in ("expired", "end_stream", "reject_image", "barge_in"))
        if self.target_hash is not None and not negative and self.case["id"] != "L12":
            gates["expected_image_acknowledged"] = any(
                image.get("image_sha256") == self.target_hash and any(
                    ack["direction"] == "receive" and ack["type"] == "conversation.item.done"
                    and ack.get("item_id") == image.get("item_id") for ack in self.trace.entries
                ) for image in self.trace.entries
            )
        called = [row["tool_name"] for row in self.trace.entries if row["type"] == "selection.started"]
        forbidden = sorted(set(called) & set(self.case.get("forbidden_tools", ())))
        result = {
            "contract_version": "direct-image-access-v1", "case_id": self.case["id"], "strategy": self.strategy,
            "model": self.settings.OPENAI_REALTIME_MODEL, "passive_delivery": self.settings.VISUAL_REALTIME_ENABLED,
            "background_analysis": False, "embedding_search": False,
            "input_audio": "Synthetic speech, 24 kHz PCM; streamed without human pacing.",
            "scope": "Real local ASGI WebSocket routes, Supabase DB/storage, and OpenAI Realtime; no physical glasses.",
            "image_hashes": sorted(self.image_hashes), "expected": self.case["expected"], "rubric": self.case["rubric"],
            "turns": self.turns, "tool_calls": called, "forbidden_tool_calls": forbidden,
            "recovery": followup_recovery(self.trace.entries, self.turns, self.case["followup_question"])
            if self.case.get("followup_question") else None,
            "transport_gates": gates, "cleanup": cleanup, "failure": failure,
            "transport_passed": failure is None and bool(self.turns) and all(gates.values()),
            "quality": {"status": "requires_manual_review", "accuracy_passed": None},
            "trace": self.trace.entries,
        }
        write_json(self.directory / "result.json", result)
        return result


def followup_recovery(trace: list[dict], turns: list[dict], question: str) -> dict:
    return {
        "followup_attempted": any(row["type"] == "question.started" and row.get("question") == question for row in trace),
        "followup_completed": any(turn["question"] == question and turn["response_status"] == "completed" for turn in turns),
        "same_session": True,
        "driver_repaired_provider_context": False,
    }


def transport_gates(trace: list[dict], known_hashes: set[str]) -> dict[str, bool]:
    images = {row["item_id"]: row for row in trace if row["direction"] == "send" and "image_sha256" in row}
    outputs = [row for row in trace if row["direction"] == "send" and row.get("call_id")]
    attached = [row for row in outputs if row.get("image_status") == "attached"]
    ordered = True
    for output in attached:
        image = images.get(output.get("provider_item_id"))
        ack = next((row for row in trace if row["direction"] == "receive" and row["type"] == "conversation.item.done" and row.get("item_id") == output.get("provider_item_id")), None)
        continuation = next((row for row in trace if row["direction"] == "send" and row["type"] == "response.create" and row["at"] > output["at"] and output.get("provider_item_id") in row.get("requested_item_ids", [])), None)
        ordered = ordered and bool(image and ack and continuation and image["at"] < ack["at"] < output["at"] < continuation["at"])
    barge = next((row for row in trace if row["type"] == "barge_in"), None)
    return {
        "ack_before_success_before_continuation": ordered,
        "one_output_per_call": len({row["call_id"] for row in outputs}) == len(outputs),
        "pixels_match_ingested_fixture": all(row["image_sha256"] in known_hashes for row in images.values()),
        "no_cancelled_generation_image_after_barge_in": not barge or not any(
            row["at"] > barge["at"] and row.get("turn_generation") == barge["cancelled_generation"] for row in images.values()
        ),
    }


def summarize(results: list[dict]) -> dict:
    latencies = [turn["first_final_response_audio_ms"] for result in results for turn in result["turns"] if turn["first_final_response_audio_ms"] is not None]
    return {
        "attempts": len(results), "transport_passed": sum(result["transport_passed"] for result in results),
        "cleanup_passed": sum(result["cleanup"]["passed"] for result in results),
        "forbidden_tool_cases": [result["case_id"] for result in results if result["forbidden_tool_calls"]],
        "accuracy_status": "Manual review required; transport success does not prove correct interpretation.",
        "first_final_response_audio_ms": {
            "median": statistics.median(latencies), "min": min(latencies), "max": max(latencies),
        } if latencies else None,
        "physical_glasses_testing": "outstanding",
    }


async def run_suite(args, cases):
    settings = JarvisSettings().model_copy(update={"OPENAI_REALTIME_MODEL": args.model})
    if not settings.OPENAI_API_KEY or not settings.SUPABASE_DB_URL:
        raise ValueError("Configured OpenAI and Supabase database credentials are required.")
    get_supabase_storage_config(settings)
    run_directory = args.output / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    run_directory.mkdir(parents=True, exist_ok=False)
    print(json.dumps({
        "run_directory": str(run_directory), "model": args.model, "speech_model": args.speech_model,
        "case_ids": [case["id"] for case in cases], "repeats": args.repeats, "strategy": args.strategy,
    }), flush=True)
    results, questions = [], {}
    async with AsyncOpenAI(api_key=settings.OPENAI_API_KEY) as client:
        for case in cases:
            for key in ("question", "setup_question", "followup_question", "replacement_question"):
                question = case.get(key)
                if question and question not in questions:
                    speech = await client.audio.speech.create(
                        model=args.speech_model, voice="alloy", input=question, response_format="pcm",
                    )
                    questions[question] = speech.content
            for repeat in range(args.repeats):
                directory = run_directory / f"{case['id']}-{repeat + 1:02d}"
                directory.mkdir(exist_ok=False)
                trial = LiveTrial(
                    settings=settings, case=case, strategy=args.strategy,
                    directory=directory, questions=questions, passive=args.passive,
                )
                result = await trial.run()
                results.append(result)
                print(json.dumps({
                    "case_id": case["id"], "repeat": repeat + 1,
                    "transport_passed": result["transport_passed"], "cleanup_passed": result["cleanup"]["passed"],
                    "failure": result["failure"], "result": str(directory / "result.json"),
                }), flush=True)
                write_json(run_directory / "summary.json", summarize(results))
                if result["failure"] is not None or not result["cleanup"]["passed"]:
                    return 1
    summary = summarize(results)
    print(json.dumps(summary), flush=True)
    return 0 if all(result["transport_passed"] and result["cleanup"]["passed"] and not result["forbidden_tool_calls"] for result in results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--allow-database-fixtures", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--speech-model", default="tts-1")
    parser.add_argument("--mode", choices=("current", "historical", "lifecycle", "all"), default="all")
    parser.add_argument("--strategy", choices=("autonomous", "forced"), default="autonomous")
    parser.add_argument("--cases", help="Comma-separated case IDs.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--passive", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [case for case in load_scenarios() if args.mode == "all" or case["mode"] == args.mode]
    if args.cases:
        selected = set(args.cases.split(","))
        cases = [case for case in cases if case["id"] in selected]
        if {case["id"] for case in cases} != selected:
            parser.error("Unknown case ID or a case outside the selected mode.")
    if args.list:
        print(json.dumps([{"id": case["id"], "mode": case["mode"], "question": case["question"]} for case in cases], indent=2))
        return
    if not args.live or not args.allow_database_fixtures or not args.model or args.output is None:
        parser.error("Live trials require --live --allow-database-fixtures --model and --output.")
    if args.repeats < 1 or not cases:
        parser.error("Select at least one case and a positive repeat count.")
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(run_suite(args, cases)))


if __name__ == "__main__":
    main()
