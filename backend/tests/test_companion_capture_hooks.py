from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.live_state.companion import CompanionState
from app.perception.companion_runtime import initialize_companion_perception
from app.realtime.live_session_capture import LiveCaptionCapture
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextRecord
from app.visual_memory.services import VisualMemoryServices
from test_companion_state import Memory, Reminders
from test_multimodal_perception import response


class Frames:
    def __init__(self):
        self.status = {}

    async def mark_analysis_pending(self, *, frame_id, **kwargs):
        self.status[frame_id] = "pending"

    async def mark_analysis_skipped(self, *, frame_id, **kwargs):
        self.status[frame_id] = "skipped"

    async def mark_analysis_failed(self, *, frame_id, **kwargs):
        self.status[frame_id] = "failed"

    async def apply_analysis_result(self, *, frame_id, **kwargs):
        self.status[frame_id] = "completed"
        return NS(id=frame_id)


class CanonicalRecords:
    def __init__(self):
        self.records = []

    async def append(self, owner, event):
        revision = 1 + sum(r.source_event_id == event.source_event_id for r in self.records)
        record = ContextRecord(id=uuid4(), owner_id=owner, session_id=event.session_id,
            sequence=len(self.records) + 1, origin_sequence=1, source_namespace=event.source_namespace,
            source_event_id=event.source_event_id, revision=revision, supersedes_event_id=None,
            kind=event.kind, trust=event.trust, finality=event.finality, source_time=event.source_time,
            recorded_at=datetime.now(UTC), payload_id=uuid4(), content_digest=None,
            content=event.payload, is_current=True,
            availability="not_finalized" if event.finality == "provisional" else "available")
        self.records.append(record)
        return record


class CompanionCaptureHooksTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = datetime.now(UTC)
        self.state = CompanionState(owner_id="fixture", working_memory=Memory(), reminders=Reminders())
        await self.state.initialize()
        self.frames = Frames()
        self.legacy = NS(enqueue=AsyncMock(), close=AsyncMock())
        self.model = NS(generate_temporal_json=AsyncMock(return_value=response()))
        self.visual = VisualMemoryServices(self.frames, self.legacy, None, None)
        self.runtime = initialize_companion_perception(state=self.state, visual_services=self.visual,
            model=self.model, on_reminder=AsyncMock(), caption_settle_seconds=0.03)
        self.canonical = CanonicalRecords()
        self.capture = SessionContextCapture(self.canonical, owner_id="fixture", session_id=uuid4(),
            provider="openai", audio_session_id="capture-fixture",
            committed_record_observer=self.runtime.observe_committed_record)

    async def asyncTearDown(self):
        await self.capture.close()
        await self.visual.close()

    def frame(self, sequence=0):
        return NS(id=uuid4(), owner_id="fixture", session_id=uuid4(), frame_sequence=sequence,
                  captured_at=self.now - timedelta(seconds=30 - sequence), source="glasses",
                  content_type="image/jpeg", raw_image_status="available")

    def caption(self, text, *, finality="provisional", source="caption"):
        self.capture._put(source, kind="user_transcript", trust="user_input", finality=finality,
            payload={"voice_route": "live", "caption_id": source, "speaker": "user", "text": text})

    async def test_selected_route_completes_frame_without_calling_legacy(self):
        frame = self.frame()
        await self.visual.enqueue(frame=frame, image_bytes=b"jpeg")
        await self.runtime.drain()
        self.assertEqual(self.frames.status[frame.id], "completed")
        self.legacy.enqueue.assert_not_awaited()
        self.model.generate_temporal_json.assert_awaited_once()

    async def test_unselected_route_preserves_legacy_dispatch(self):
        visual = VisualMemoryServices(self.frames, self.legacy, None, None)
        frame = self.frame()
        await visual.enqueue(frame=frame, image_bytes=b"jpeg")
        self.legacy.enqueue.assert_awaited_once_with(frame=frame, image_bytes=b"jpeg", send_diagnostic=None)

    async def test_coalesced_frames_get_final_bookkeeping(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate(**kwargs):
            if not entered.is_set():
                entered.set()
                await release.wait()
            return response()

        self.model.generate_temporal_json.side_effect = generate
        frames = [self.frame(i) for i in range(9)]
        await self.visual.enqueue(frame=frames[0], image_bytes=b"jpeg")
        await asyncio.wait_for(entered.wait(), 1)
        for frame in frames[1:]:
            await self.visual.enqueue(frame=frame, image_bytes=b"jpeg")
        release.set()
        await self.runtime.drain()
        self.assertEqual([self.frames.status[f.id] for f in frames],
                         ["completed", *(["skipped"] * 4), *(["completed"] * 4)])
        self.legacy.enqueue.assert_not_awaited()

    async def test_failed_model_marks_frame_failed_without_storage_rollback(self):
        self.model.generate_temporal_json.side_effect = RuntimeError("synthetic")
        frame = self.frame()
        await self.visual.enqueue(frame=frame, image_bytes=b"jpeg")
        await self.runtime.drain()
        self.assertEqual(self.frames.status[frame.id], "failed")

    async def test_close_marks_active_and_pending_frames_skipped(self):
        entered = asyncio.Event()

        async def generate(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        self.model.generate_temporal_json.side_effect = generate
        first, second = self.frame(), self.frame(1)
        await self.visual.enqueue(frame=first, image_bytes=b"jpeg")
        await asyncio.wait_for(entered.wait(), 1)
        await self.visual.enqueue(frame=second, image_bytes=b"jpeg")
        await self.visual.close()
        self.assertEqual(self.frames.status[first.id], "skipped")
        self.assertEqual(self.frames.status[second.id], "skipped")

    async def test_whole_committed_caption_settles_once_without_memory_service(self):
        self.caption("I")
        await self.capture.flush()
        self.caption("I need to focus")
        await self.capture.flush()
        self.caption("I need to focus now.")
        await self.capture.flush()
        await self.runtime.drain()
        self.model.generate_temporal_json.assert_awaited_once()
        prompt = self.model.generate_temporal_json.call_args.kwargs["prompt"]
        self.assertIn('"text":"I need to focus now."', prompt)
        self.assertNotIn('"text":"I"', prompt)
        self.assertIn("application_settled", prompt)
        self.assertIsNone(self.runtime.memory)
        # Receiving an identical committed revision again is not another event.
        self.runtime.observe_committed_record(self.canonical.records[-1])
        await self.runtime.drain()
        self.assertEqual(self.model.generate_temporal_json.await_count, 1)

    async def test_real_live_caption_wrapper_reaches_post_commit_hook(self):
        context = NS(owner_id="fixture", jarvis_session_id=self.capture.session_id,
                     audio_session_id="capture-fixture")
        live = LiveCaptionCapture(self.capture, context)
        identity = uuid4()
        event = NS(final=False, output=None, transcript_id=identity, revision=1, text="Hello from Live",
                   role="user", observed_at=self.now, route=NS(model_dump=lambda **kwargs: {}))
        fragment = NS(event_id="provider-event", start_ms=10, end_ms=800, delta=event.text)
        part = NS(transcript_id=identity, revision=1, fragment=fragment)
        self.assertTrue(live.transcript(event, (part,)))
        await self.capture.flush()
        await self.runtime.drain()
        self.assertIn("Hello from Live", self.model.generate_temporal_json.call_args.kwargs["prompt"])
        await live.close()

    async def test_second_utterance_preserves_caption_start_and_original_revision_time(self):
        start = self.now - timedelta(seconds=30)
        end = start + timedelta(seconds=15)
        for observation_backed in (False, True):
            with self.subTest(observation_backed=observation_backed):
                context = NS(owner_id="fixture", jarvis_session_id=self.capture.session_id,
                             audio_session_id="capture-fixture")
                live = LiveCaptionCapture(self.capture, context, observation_backed=observation_backed)
                identity = uuid4()
                parts, text, prompts = [], "", []
                try:
                    for revision, (at, delta, offset) in enumerate((
                            (start, "Remind me to take my umbrella when I leave.", 800),
                            (end, " I am leaving now.", 15600)), start=1):
                        text += delta
                        fragment = NS(event_id=f"{identity}-{revision}", start_ms=offset,
                                      end_ms=offset + 1000, delta=delta, speaker="user")
                        parts.append(NS(transcript_id=identity, revision=revision, fragment=fragment))
                        event = NS(final=False, output=None, transcript_id=identity, revision=revision,
                                   text=text, role="user", observed_at=at,
                                   route=NS(model_dump=lambda **kwargs: {}))
                        self.assertTrue(live.transcript(event, tuple(parts)))
                        if observation_backed:
                            await live.observations(tuple(parts))
                        self.assertTrue(await self.capture.flush())
                        await self.runtime.drain()
                        prompt = self.model.generate_temporal_json.call_args.kwargs["prompt"]
                        prompts.append(json.loads(prompt.split("Inputs follow:\n", 1)[1])["dialogue"][0])
                    records = [r for r in self.canonical.records if r.kind == "user_transcript"
                               and r.content["caption_id"] == str(identity)]
                    self.assertEqual(len(records), 2)
                    self.assertEqual([r.source_time for r in records], [start, start])
                    self.assertEqual(records[0].source_event_id, records[1].source_event_id)
                    self.assertEqual(records[1].content["observed_at"], end.isoformat())
                    self.assertEqual([p["start_ms"] for p in records[1].content["fragments"]], [800, 15600])
                    self.assertEqual(prompts[0]["caption_id"], prompts[1]["caption_id"])
                    for prompt, expected_end in zip(prompts, (start, end)):
                        self.assertEqual(datetime.fromisoformat(prompt["observed_from"]), start)
                        self.assertEqual(datetime.fromisoformat(prompt["observed_until"]), expected_end)
                        self.assertEqual(prompt["time_basis"], "application_ingress_approximate")
                    self.assertLess(end, records[1].recorded_at)
                finally:
                    await live.close()

    async def test_finalized_record_does_not_wait_for_settle_timeout(self):
        self.runtime.caption_settle_seconds = 30
        self.caption("Finished sentence", finality="final")
        await self.capture.flush()
        await asyncio.wait_for(self.runtime.drain(), 1)
        self.model.generate_temporal_json.assert_awaited_once()

    async def test_forget_during_settle_discards_pending_caption(self):
        self.caption("A pending old claim")
        await self.capture.flush()
        await self.state.invalidate_memory(NS(owner_id="fixture", document_ids=("doc",)))
        await self.runtime.drain()
        self.model.generate_temporal_json.assert_not_awaited()

    async def test_raw_fragments_are_not_analyzer_dialogue(self):
        self.capture._put("fragment", kind="observation", trust="derived",
                          payload={"voice_route": "live", "delta": "one", "speaker": "user"})
        await self.capture.flush()
        await self.runtime.drain()
        self.model.generate_temporal_json.assert_not_awaited()

    async def test_observer_error_does_not_reclassify_successful_capture(self):
        def broken(record):
            raise RuntimeError("synthetic observer failure")

        self.capture.committed_record_observer = broken
        self.caption("Capture survives")
        self.assertTrue(await self.capture.flush())
        self.assertIsNone(self.capture.issue)
        self.assertEqual(self.capture.observer_issue, "committed_record_observer_failed")
        self.assertEqual(len(self.canonical.records), 1)

    async def test_initializer_is_idempotent_and_does_not_start_a_model(self):
        same = initialize_companion_perception(state=self.state, visual_services=self.visual,
                                               model=self.model, on_reminder=AsyncMock())
        self.assertIs(same, self.runtime)
        self.model.generate_temporal_json.assert_not_awaited()
