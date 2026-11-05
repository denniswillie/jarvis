"""Synthetic replay boundary tests; no model/video-quality or frontend claim."""
import asyncio
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from pydantic import ValidationError

from evals.responsive_voice.video_replay import (
    BudgetedTemporalModel, ContextEvent, FrameAsset, PermissionRecord, PipelineReplayPort,
    ReplayCallBudget, ReplayCase, ReplayInputs, read_frame_asset, run_replay,
)


class Port:
    execution_kind = "synthetic"

    def __init__(self):
        self.trace, self.frames = [], []
        self.opened, self.closed, self.drained = False, False, False

    async def open(self, context, clock, budget):
        self.opened = True
        self.context_received, self.clock, self.budget = context, clock, budget

    async def context(self, event, occurred_at):
        self.trace.append(("context", event.offset_ms))

    async def enqueue(self, frame):
        assert not self.drained
        self.trace.append(("frame", frame.offset_ms))
        self.frames.append(frame)

    async def drain(self):
        self.drained = True

    async def outcomes(self):
        return {"visible_findings": [], "notification_intents": []}

    async def close(self):
        self.closed = True


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = b"synthetic bytes, not an actual photograph"
        (self.root / "opaque-frame.bin").write_bytes(self.data)
        self.frame = FrameAsset(offset_ms=0, relative_path="opaque-frame.bin",
            sha256=hashlib.sha256(self.data).hexdigest(), width=432, height=768)
        self.case = ReplayCase(case_id="evaluator-only-case-name",
            permission=PermissionRecord(evidence="synthetic fixture", local_evaluation=True,
                                        retain_derived_results=True),
            inputs=ReplayInputs(duration_ms=1000, initial_context=("Look for my keys.",),
                frames=(self.frame, self.frame.model_copy(update={"offset_ms": 1000})),
                events=(ContextEvent(event_id="goal-v2", offset_ms=1000, kind="goal", text="Look for the bag."),)),
            expected_events=("SECRET_EXPECTATION",), forbidden_claims=("SECRET_ANNOTATION",))
        self.origin = datetime(2026, 10, 30, tzinfo=timezone.utc)

    async def run_case(self, port=None, **kwargs):
        return await run_replay(self.case, asset_root=self.root, port=port or Port(),
                                origin=self.origin, realtime=False, **kwargs)

    async def test_chronology_context_before_tied_frame_no_per_frame_drain(self):
        port = Port()
        result = await self.run_case(port)
        self.assertEqual(port.trace, [("frame", 0), ("context", 1000), ("frame", 1000)])
        self.assertTrue(port.drained and port.closed)
        self.assertEqual(result["execution_status"], "completed")
        self.assertEqual(result["acceptance_status"], "not_scored")
        self.assertFalse(result["fixture_cleanup_audited"])
        self.assertIsNone(result["first_useful_audio_ms"])

    async def test_evaluator_names_labels_and_asset_paths_are_not_forwarded(self):
        port = Port()
        await self.run_case(port)
        self.assertEqual(port.context_received, ("Look for my keys.",))
        for frame in port.frames:
            self.assertFalse(hasattr(frame, "relative_path"))
            self.assertFalse(hasattr(frame, "expected_events"))
            self.assertNotIn("SECRET", repr(frame))

    async def test_future_frame_not_available_while_first_frame_is_enqueued(self):
        port = Port()
        original = port.enqueue
        async def capture(frame):
            self.assertEqual(len(port.frames), frame.sequence - 1)
            self.assertGreaterEqual(port.clock.now(), frame.captured_at)
            await original(frame)
        port.enqueue = capture
        await self.run_case(port)
        self.assertEqual([f.sequence for f in port.frames], [1, 2])

    async def test_hosted_permission_checked_before_open_or_asset_read(self):
        port = Port()
        port.execution_kind = "hosted_model"
        with self.assertRaisesRegex(ValueError, "hosted_permission"):
            await self.run_case(port)
        self.assertFalse(port.opened)

    async def test_checksum_failure_closes_partial_run_without_submit(self):
        (self.root / "opaque-frame.bin").write_bytes(b"changed")
        port = Port()
        result = await self.run_case(port)
        self.assertEqual(result["execution_status"], "failed")
        self.assertEqual(port.frames, [])
        self.assertTrue(port.closed)

    def test_path_escape_is_rejected(self):
        with TemporaryDirectory() as outside:
            path = Path(outside) / "private.bin"
            path.write_bytes(self.data)
            (self.root / "escape").symlink_to(path)
            with self.assertRaisesRegex(ValueError, "outside_root"):
                read_frame_asset(self.root, self.frame.model_copy(update={"relative_path": "escape"}))

    def test_frame_size_bound_and_timeline_validation(self):
        with self.assertRaisesRegex(ValueError, "size_invalid"):
            read_frame_asset(self.root, self.frame, max_bytes=3)
        with self.assertRaises(ValidationError):
            ReplayInputs(duration_ms=100, frames=(self.frame, self.frame))
        with self.assertRaises(ValidationError):
            ReplayInputs(duration_ms=100, frames=(self.frame.model_copy(update={"offset_ms": 101}),))

    async def test_partial_open_failure_still_closes(self):
        port = Port()
        port.open = AsyncMock(side_effect=RuntimeError("private-path-not-for-report"))
        result = await self.run_case(port)
        self.assertTrue(port.closed)
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertNotIn("private-path", repr(result))

    async def test_model_budget_prevents_next_actual_call(self):
        raw = SimpleNamespace(provider="fixture", model="fixture-model",
                              generate_temporal_json=AsyncMock(return_value="{}"))
        budget = ReplayCallBudget(1)
        wrapper = BudgetedTemporalModel(raw, budget)
        await wrapper.generate_temporal_json(prompt="fixture", images=())
        with self.assertRaisesRegex(RuntimeError, "budget_exhausted"):
            await wrapper.generate_temporal_json(prompt="fixture", images=())
        self.assertEqual(raw.generate_temporal_json.await_count, 1)
        self.assertEqual(budget.attempted, 1)

    async def test_timeout_closes_owned_port(self):
        port = Port()
        port.drain = AsyncMock(side_effect=lambda: None)
        async def blocked():
            await asyncio.Event().wait()
        port.drain = blocked
        result = await self.run_case(port, timeout_seconds=.02)
        self.assertEqual(result["error_type"], "TimeoutError")
        self.assertTrue(port.closed)

    async def test_realtime_schedule_keeps_original_capture_times_when_late(self):
        port = Port()
        elapsed = [0.0]
        async def sleep(delay):
            elapsed[0] += delay
            await asyncio.sleep(0)
        original = port.enqueue
        async def slow_first(frame):
            await original(frame)
            if frame.sequence == 1:
                elapsed[0] += 1.5
        port.enqueue = slow_first
        result = await run_replay(self.case, asset_root=self.root, port=port,
            origin=self.origin, realtime=True, monotonic=lambda:elapsed[0], sleep=sleep)
        self.assertEqual(result["submitted_frames"][1]["offset_ms"], 1000)
        self.assertEqual(result["submitted_frames"][1]["enqueue_lateness_ms"], 500)
        self.assertEqual((port.frames[1].captured_at-self.origin).total_seconds(), 1)

    def test_pipeline_port_refuses_fake_downstream_pipeline(self):
        with self.assertRaises(TypeError):
            PipelineReplayPort(SimpleNamespace(), execution_kind="synthetic",
                begin=AsyncMock(), apply_context=AsyncMock(), make_frame=AsyncMock(),
                collect=AsyncMock(), cleanup=AsyncMock())


if __name__ == "__main__":
    unittest.main()
