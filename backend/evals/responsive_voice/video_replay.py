"""Bounded replay inputs for the existing perception pipeline.

This module does not create a model, database, notification authority or media
decoder. Assets are permission-cleared, extracted frames. An application port
owns canonical context admission, camera setup and isolated resource cleanup.
Execution completion is deliberately not visual-quality acceptance.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
from typing import Awaitable, Callable, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PermissionRecord(StrictRecord):
    evidence: str = Field(min_length=1, max_length=2000)
    local_evaluation: bool = False
    hosted_inference: bool = False
    retain_derived_results: bool = False


class FrameAsset(StrictRecord):
    offset_ms: int = Field(ge=0, strict=True)
    relative_path: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    content_type: Literal["image/jpeg", "image/png"] = "image/jpeg"
    width: int = Field(gt=0, le=8192, strict=True)
    height: int = Field(gt=0, le=8192, strict=True)


class ContextEvent(StrictRecord):
    event_id: str = Field(min_length=1, max_length=100)
    offset_ms: int = Field(ge=0, strict=True)
    kind: Literal["utterance", "goal", "cancel_goal", "prior_context"]
    text: str = Field(min_length=1, max_length=4000)


class ReplayInputs(StrictRecord):
    duration_ms: int = Field(gt=0, le=180000, strict=True)
    initial_context: tuple[str, ...] = Field(default=(), max_length=8)
    frames: tuple[FrameAsset, ...] = Field(min_length=1, max_length=360)
    events: tuple[ContextEvent, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def chronological(self):
        offsets = [f.offset_ms for f in self.frames]
        if offsets != sorted(set(offsets)):
            raise ValueError("frame offsets must be unique and chronological")
        if any(offset > self.duration_ms for offset in offsets):
            raise ValueError("frame outside replay duration")
        if any(e.offset_ms > self.duration_ms for e in self.events):
            raise ValueError("context event outside replay duration")
        if len({e.event_id for e in self.events}) != len(self.events):
            raise ValueError("context event identities must be unique")
        if any(not text.strip() or len(text) > 4000 for text in self.initial_context):
            raise ValueError("initial context must contain bounded nonblank statements")
        return self


class ReplayCase(StrictRecord):
    case_id: str = Field(min_length=1, max_length=100)
    permission: PermissionRecord
    inputs: ReplayInputs
    # These fields belong to the evaluator; they are never passed to the port.
    expected_events: tuple[str, ...] = Field(default=(), max_length=64)
    forbidden_claims: tuple[str, ...] = Field(default=(), max_length=64)


@dataclass(frozen=True)
class ReplayFrame:
    frame_id: UUID
    sequence: int
    captured_at: datetime
    offset_ms: int
    image_bytes: bytes
    content_type: str
    width: int
    height: int


class ReplayClock:
    def __init__(self, origin: datetime, *, realtime: bool, monotonic, sleep):
        if origin.tzinfo is None or origin.utcoffset() is None:
            raise ValueError("replay origin must be timezone-aware")
        self.origin, self.realtime = origin, realtime
        self._monotonic, self._sleep = monotonic, sleep
        self._start, self._offset_ms = monotonic(), 0

    def now(self):
        elapsed = ((self._monotonic() - self._start) * 1000
                   if self.realtime else self._offset_ms)
        return self.origin + timedelta(milliseconds=elapsed)

    async def advance(self, offset_ms):
        if offset_ms < self._offset_ms:
            raise ValueError("replay clock cannot move backwards")
        if self.realtime:
            delay = max(0, self._start + offset_ms / 1000 - self._monotonic())
            await self._sleep(delay)
        else:
            await self._sleep(0)
        self._offset_ms = offset_ms

    def scheduled_at(self, offset_ms):
        return self.origin + timedelta(milliseconds=offset_ms)


class ReplayCallBudget:
    def __init__(self, maximum: int):
        if type(maximum) is not int or not 1 <= maximum <= 200:
            raise ValueError("model-call budget must be between 1 and 200")
        self.maximum, self.attempted = maximum, 0

    def consume(self):
        # Same event-loop owner, no suspension between check and increment.
        if self.attempted >= self.maximum:
            raise RuntimeError("replay_model_call_budget_exhausted")
        self.attempted += 1


class BudgetedTemporalModel:
    """Wrap the existing borrowed model, including targeted predicate calls."""
    def __init__(self, model, budget: ReplayCallBudget):
        self._model, self.budget = model, budget

    @property
    def provider(self):
        return self._model.provider

    @property
    def model(self):
        return self._model.model

    async def generate_temporal_json(self, *, prompt, images):
        self.budget.consume()
        return await self._model.generate_temporal_json(prompt=prompt, images=images)


class ReplayPort(Protocol):
    execution_kind: Literal["synthetic", "local_model", "hosted_model"]

    async def open(self, context: tuple[str, ...], clock: ReplayClock,
                   budget: ReplayCallBudget) -> None: ...
    async def context(self, event: ContextEvent, occurred_at: datetime) -> None: ...
    async def enqueue(self, frame: ReplayFrame) -> None: ...
    async def drain(self) -> None: ...
    async def outcomes(self) -> dict: ...
    async def close(self) -> None: ...


def read_frame_asset(root: Path, asset: FrameAsset, *, max_bytes=4 * 1024 * 1024):
    root = root.resolve(strict=True)
    path = (root / asset.relative_path).resolve(strict=True)
    if Path(asset.relative_path).is_absolute() or not path.is_relative_to(root):
        raise ValueError("replay_asset_outside_root")
    with path.open("rb") as source:
        content = source.read(max_bytes + 1)
    if not content or len(content) > max_bytes:
        raise ValueError("replay_frame_size_invalid")
    if hashlib.sha256(content).hexdigest() != asset.sha256:
        raise ValueError("replay_asset_checksum_mismatch")
    return content


async def run_replay(case: ReplayCase, *, asset_root: Path, port: ReplayPort,
                     origin: datetime, realtime=True, max_model_calls=40,
                     timeout_seconds=240, cleanup_seconds=10,
                     monotonic=None, sleep=asyncio.sleep):
    """Feed capture inputs without awaiting analysis after each frame.

    No dataset label, asset path, complete manifest or future frame is exposed to
    the application port. End-of-run draining is bounded, separately from cadence.
    Permission records are declarations supplied by curation, not legal analysis.
    """
    if port.execution_kind not in {"synthetic", "local_model", "hosted_model"}:
        raise ValueError("unknown replay execution kind")
    rights = case.permission
    if not rights.local_evaluation or not rights.retain_derived_results:
        raise ValueError("replay_permission_unresolved")
    if port.execution_kind == "hosted_model" and not rights.hosted_inference:
        raise ValueError("replay_hosted_permission_unresolved")
    if not 0 < timeout_seconds <= 600 or not 0 < cleanup_seconds <= 60:
        raise ValueError("replay deadlines must be positive and bounded")
    budget = ReplayCallBudget(max_model_calls)
    clock = ReplayClock(origin, realtime=realtime,
        monotonic=monotonic or asyncio.get_running_loop().time, sleep=sleep)
    report = {"case_id": case.case_id, "execution_kind": port.execution_kind,
        "realtime": realtime, "execution_status": "running", "acceptance_status": "not_scored",
        "submitted_frames": [], "context_events": [], "outcomes": {},
        "model_calls_attempted": 0, "port_closed": False,
        "fixture_cleanup_audited": False, "first_useful_audio_ms": None}
    opened = False
    try:
        async with asyncio.timeout(timeout_seconds):
            opened = True  # A partial open still owns teardown.
            await port.open(case.inputs.initial_context, clock, budget)
            # Events at a frame's timestamp are delivered before that frame.
            timeline = [(event.offset_ms, 0, i, event)
                        for i, event in enumerate(case.inputs.events)]
            timeline += [(frame.offset_ms, 1, i, frame)
                         for i, frame in enumerate(case.inputs.frames)]
            for offset, kind, index, item in sorted(timeline, key=lambda row: row[:3]):
                await clock.advance(offset)
                if kind == 0:
                    await port.context(item, clock.scheduled_at(offset))
                    report["context_events"].append(item.event_id)
                    continue
                data = read_frame_asset(asset_root, item)
                frame = ReplayFrame(uuid4(), index + 1, clock.scheduled_at(offset), offset,
                    data, item.content_type, item.width, item.height)
                await port.enqueue(frame)
                report["submitted_frames"].append({"id": str(frame.frame_id), "sequence": frame.sequence,
                    "offset_ms": offset, "captured_at": frame.captured_at.isoformat(),
                    "enqueue_lateness_ms": max(0, (clock.now() - frame.captured_at).total_seconds() * 1000)})
            await clock.advance(case.inputs.duration_ms)
            await port.drain()
            report["outcomes"] = await port.outcomes()
            report["execution_status"] = "completed"
    except Exception as error:
        # Avoid exception text: it can contain private paths or model responses.
        report["execution_status"] = "failed"
        report["error_type"] = type(error).__name__
    finally:
        report["model_calls_attempted"] = budget.attempted
        if opened:
            try:
                async with asyncio.timeout(cleanup_seconds):
                    await port.close()
                report["port_closed"] = True
            except Exception as error:
                report["execution_status"] = "failed"
                report["cleanup_error_type"] = type(error).__name__
    return report


class PipelineReplayPort:
    """Thin port over an owned production VisualAnalysisPipeline.

    Fixture composition supplies canonical context/camera admission and frame-row
    creation. It must install the budgeted borrowed model before submitting frames.
    This port never resolves a check or sends a notification itself.
    """
    def __init__(self, pipeline, *, execution_kind, begin: Callable[..., Awaitable],
                 apply_context: Callable[..., Awaitable], make_frame: Callable[..., Awaitable],
                 collect: Callable[..., Awaitable], cleanup: Callable[..., Awaitable]):
        from app.visual_memory.pipeline import VisualAnalysisPipeline
        if not isinstance(pipeline, VisualAnalysisPipeline):
            raise TypeError("replay requires the actual visual pipeline")
        self.pipeline, self.execution_kind = pipeline, execution_kind
        self._begin, self._context, self._frame = begin, apply_context, make_frame
        self._collect, self._cleanup = collect, cleanup
        self._closed = False

    async def open(self, context, clock, budget):
        await self._begin(context, clock, budget)

    async def context(self, event, occurred_at):
        await self._context(event, occurred_at)

    async def enqueue(self, frame):
        canonical = await self._frame(frame)
        if (canonical.id != frame.frame_id or canonical.frame_sequence != frame.sequence
                or canonical.captured_at != frame.captured_at):
            raise ValueError("replay_capture_identity_changed")
        await self.pipeline.enqueue(frame=canonical, image_bytes=frame.image_bytes,
                                    send_diagnostic=self._diagnostic)

    async def _diagnostic(self, payload):
        # Application observations/diagnostics are collected by the fixture's
        # existing event/recording adapters, not used to alter replay inputs.
        return None

    async def drain(self):
        await self.pipeline.drain()

    async def outcomes(self):
        return await self._collect()

    async def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            await self.pipeline.close()
        finally:
            await self._cleanup()
