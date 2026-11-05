from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import httpx

from app.live_state.companion import CompanionState
from app.model_providers.gemini_fast_visual import GeminiFastVisualModel
from app.model_providers.protocols import VisualModelImage
from app.perception.companion_runtime import CompanionPerceptionRuntime
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.shared_memory.client import BankScope, NativeMemory, source_document_id
from app.shared_memory.ingestion import CanonicalCaption
from app.shared_memory.intake import LiveDialogueIntake
from app.visual_memory.gemini_client import GeminiVisualAnalyzerClient, GeminiVisualAnalyzerConfig
from test_companion_state import Memory, Reminders
from test_multimodal_perception import NOW, fact, response, view_fact
from test_shared_memory_intake import Source


class GeminiTextRequestTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []

        async def handler(request):
            self.requests.append(json.loads(request.content))
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})

        self.client = GeminiVisualAnalyzerClient(GeminiVisualAnalyzerConfig("synthetic", "fixture"))
        await self.client._client.aclose()
        self.client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def asyncTearDown(self):
        await self.client.close()

    async def test_real_temporal_adapter_sends_text_without_image_placeholder(self):
        result = await GeminiFastVisualModel(self.client).generate_temporal_json(prompt="Settled dialogue", images=())
        self.assertEqual(result, "{}")
        self.assertEqual(self.requests[0]["contents"][0]["parts"], [{"text": "Settled dialogue"}])

    async def test_genuinely_empty_request_and_empty_attachment_rejected(self):
        for prompt, images in ((" \n", ()), ("", (VisualModelImage("frame", b"", "image/jpeg"),))):
            with self.assertRaises(ValueError):
                await self.client.generate_multiframe_json(prompt=prompt, images=images)
        self.assertEqual(self.requests, [])

    async def test_image_only_path_still_preserves_original_bytes(self):
        await self.client.generate_multiframe_json(prompt="", images=(VisualModelImage("frame:0", b"abc", "image/jpeg"),))
        self.assertEqual(self.requests[0]["contents"][0]["parts"][-1]["inlineData"]["data"], "YWJj")


class CompanionPerceptionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = NOW + timedelta(seconds=10)
        self.state = CompanionState(owner_id="fixture", working_memory=Memory(), reminders=Reminders(),
                                     clock=lambda: self.now)
        await self.state.initialize()
        self.model = NS(generate_temporal_json=AsyncMock(return_value=response()))
        self.notices = AsyncMock()
        self.posts = []

        async def handler(request):
            if request.method == "GET":
                return httpx.Response(200, json={"status": "completed"})
            if request.method == "DELETE":
                return httpx.Response(204)
            self.posts.append(json.loads(request.content))
            return httpx.Response(200, json={"success": True, "operation_id": "fixture-operation"})

        self.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.memory = NativeMemory(scope=BankScope("fixture", "fixture-bank"), base_url="http://fixture",
                                   api_key="synthetic", http_client=self.http)
        self.temp = tempfile.TemporaryDirectory()
        source = Source()
        source.owner_id = "fixture"
        self.intake = LiveDialogueIntake(self.memory, source, Path(self.temp.name) / "cursor.sqlite",
                                        invalidate=AsyncMock(), poll_seconds=0.001)
        self.runtime = CompanionPerceptionRuntime(state=self.state, model=self.model, memory=self.memory,
                                                   intake=self.intake, on_reminder=self.notices, clock=lambda: self.now)
        self.session = str(uuid4())
        self.visual_session = uuid4()
        self.frames = RecentVisualFrameWindow(max_frames_per_session=16, ttl_seconds=30,
            monotonic=lambda: 0, now=lambda: self.now)
        self.lease = self.frames.begin_stream("fixture", UUID(self.session), self.visual_session)

    async def asyncTearDown(self):
        await self.runtime.close()
        self.intake.abort()
        self.temp.cleanup()
        await self.http.aclose()

    def frame(self, second=0):
        return NS(id=uuid4(), owner_id="fixture", session_id=self.session, frame_sequence=second,
                  captured_at=NOW + timedelta(seconds=second), source="glasses", content_type="image/jpeg",
                  raw_image_status="available")

    def caption(self, revision=1):
        return CanonicalCaption(1, self.session, "canonical-live", "caption-key", revision,
            "user_transcript", NOW, NOW, "not_finalized", {"voice_route": "live", "speaker": "user",
            "caption_id": "native-caption-id", "caption_revision": revision,
            "text": "I need to focus now.", "fragments": []})

    def buffered_frame(self, second=10, *, image_bytes=b"image"):
        frame = self.frame(second)
        self.frames.put(SensoryVisualFrame(frame_id=frame.id, owner_id=frame.owner_id,
            session_id=UUID(self.session), audio_session_id=None,
            visual_session_id=self.visual_session, frame_sequence=frame.frame_sequence,
            source="glasses", captured_at=frame.captured_at, server_received_at=self.now,
            content_type="image/jpeg", width=432, height=768, storage_bucket="fixture",
            storage_path=str(frame.id), raw_image_expires_at=self.now + timedelta(seconds=30),
            image_bytes=image_bytes), stream_lease=self.lease)
        return frame

    async def fresh_view(self):
        return await self.runtime.read_current_view(frame_window=self.frames,
            owner_id="fixture", session_id=UUID(self.session))

    async def test_fresh_view_waits_for_newest_buffered_frame_not_older_inflight_result(self):
        entered = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]
        calls = []

        async def generate(**kwargs):
            index = len(calls)
            calls.append(kwargs)
            entered[index].set()
            await release[index].wait()
            return response(current_view=view_fact("Old desk" if index == 0 else "Standing fan"))

        self.model.generate_temporal_json.side_effect = generate
        old = self.buffered_frame(9)
        self.runtime.submit_frame(old, b"old")
        await asyncio.wait_for(entered[0].wait(), 1)
        newest = self.buffered_frame(10)
        self.runtime.submit_frame(newest, b"new")
        reader = asyncio.create_task(self.fresh_view())
        await asyncio.sleep(0)
        self.now += timedelta(seconds=2)
        release[0].set()
        await asyncio.wait_for(entered[1].wait(), 1)
        self.assertFalse(reader.done())
        release[1].set()
        value = await asyncio.wait_for(reader, 1)
        self.assertEqual(value["target_frame_id"], str(newest.id))
        self.assertEqual([item["description"] for item in value["facts"]], ["Standing fan"])
        self.assertEqual(value["facts"][0]["evidence_at"], newest.captured_at.isoformat())
        self.assertEqual(self.model.generate_temporal_json.await_count, 2)

    async def test_fresh_view_keeps_requested_frame_when_ring_evicts_it_during_analysis(self):
        self.frames = self.frames.__class__(max_frames_per_session=4, ttl_seconds=10,
                                           now=lambda: self.now)
        self.lease = self.frames.begin_stream("fixture", UUID(self.session), self.visual_session)
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate(**kwargs):
            entered.set()
            await release.wait()
            return response(current_view=view_fact("Household wipes"))

        self.model.generate_temporal_json.side_effect = generate
        target = self.buffered_frame(10)
        self.runtime.submit_frame(target, b"image")
        await asyncio.wait_for(entered.wait(), 1)
        reader = asyncio.create_task(self.fresh_view())
        await asyncio.sleep(0)
        for second in range(11, 15):
            self.now = NOW + timedelta(seconds=second)
            self.buffered_frame(second)
        self.assertNotIn(target.id, [frame.frame_id for frame in
                                    self.frames.window("fixture", UUID(self.session))])
        release.set()
        value = await asyncio.wait_for(reader, 1)
        self.assertEqual(value["status"], "available")
        self.assertEqual(value["target_frame_id"], str(target.id))
        self.assertEqual(value["facts"][0]["description"], "Household wipes")
        self.assertEqual(value["facts"][0]["evidence_at"], target.captured_at.isoformat())
        self.model.generate_temporal_json.assert_awaited_once()

    async def test_fresh_view_completed_current_fact_needs_no_extra_work_or_memory(self):
        frame = self.buffered_frame()
        self.model.generate_temporal_json.return_value = response(current_view=view_fact())
        self.runtime.submit_frame(frame, b"image")
        await self.runtime.drain()
        with patch.object(self.runtime.analyzer, "wait_for_visual_result", new_callable=AsyncMock) as wait:
            value = await self.fresh_view()
            wait.assert_not_awaited()
        self.assertEqual(value["status"], "available")
        self.assertEqual(value["facts"][0]["description"], "Standing fan")
        self.assertEqual(value["text"], "Current camera observations.")
        self.assertEqual(value["facts"][0]["evidence_at"], frame.captured_at.isoformat())
        self.assertEqual(self.posts, [])
        self.model.generate_temporal_json.assert_awaited_once()

    async def test_fresh_view_deadline_does_not_cancel_shared_analysis(self):
        entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def generate(**kwargs):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return response(current_view=view_fact())

        self.model.generate_temporal_json.side_effect = generate
        self.runtime.submit_frame(self.buffered_frame(), b"image")
        await asyncio.wait_for(entered.wait(), 1)
        timeout = asyncio.timeout
        with patch("app.perception.companion_runtime.asyncio.timeout",
                   side_effect=lambda seconds: timeout(0.02 if seconds == 6 else seconds)):
            value = await self.fresh_view()
        self.assertEqual((value["status"], value["reason"]), ("freshness_unavailable", "deadline"))
        self.assertIn("A fresh current view could not be obtained", value["text"])
        self.assertNotIn("facts", value)
        self.assertFalse(cancelled.is_set())
        self.assertFalse(self.runtime.analyzer._task.done())
        release.set()
        await self.runtime.drain()
        self.assertEqual((await self.fresh_view())["status"], "available")
        self.model.generate_temporal_json.assert_awaited_once()

    async def test_fresh_view_requires_recent_nonempty_frame_and_current_stream(self):
        for name, second, image in (("none", None, b"image"), ("stale", 7, b"image"),
                                    ("future", 11, b"image"), ("empty", 10, b"")):
            with self.subTest(name=name):
                self.lease = self.frames.begin_stream("fixture", UUID(self.session), self.visual_session)
                if second is not None:
                    self.buffered_frame(second, image_bytes=image)
                value = await self.fresh_view()
                self.assertEqual(value["reason"], "no_recent_frame")
        self.buffered_frame()
        self.frames.end_stream(self.lease)
        self.assertEqual((await self.fresh_view())["reason"], "no_recent_frame")
        self.model.generate_temporal_json.assert_not_awaited()
        with self.assertRaises(ValueError):
            await self.runtime.read_current_view(frame_window=self.frames,
                owner_id="other", session_id=UUID(self.session))

    async def test_fresh_view_rejects_cached_old_generation_then_accepts_current_generation(self):
        frame = self.buffered_frame()
        self.model.generate_temporal_json.return_value = response(current_view=view_fact())
        self.runtime.submit_frame(frame, b"image")
        await self.runtime.drain()
        await self.state.invalidate_memory(NS(owner_id="fixture", document_ids=("forgotten",)))
        reader = asyncio.create_task(self.fresh_view())
        await asyncio.sleep(0)
        self.assertFalse(reader.done())
        self.runtime.submit_frame(frame, b"image")
        value = await asyncio.wait_for(reader, 1)
        self.assertEqual(value["status"], "available")
        self.assertEqual(self.runtime.analyzer.latest_visual_result.human_generation, 1)

    async def test_fresh_view_generation_change_during_wait_is_unavailable(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate(**kwargs):
            entered.set()
            await release.wait()
            return response(current_view=view_fact())

        self.model.generate_temporal_json.side_effect = generate
        self.runtime.submit_frame(self.buffered_frame(), b"image")
        await asyncio.wait_for(entered.wait(), 1)
        reader = asyncio.create_task(self.fresh_view())
        await asyncio.sleep(0)
        await self.state.invalidate_memory(NS(owner_id="fixture", document_ids=("forgotten",)))
        release.set()
        value = await asyncio.wait_for(reader, 1)
        self.assertEqual(value["reason"], "source_changed")
        self.assertNotIn("facts", value)

    async def test_fresh_view_reconnect_with_same_stream_id_invalidates_wait(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate(**kwargs):
            entered.set()
            await release.wait()
            return response(current_view=view_fact())

        self.model.generate_temporal_json.side_effect = generate
        self.runtime.submit_frame(self.buffered_frame(), b"image")
        await asyncio.wait_for(entered.wait(), 1)
        reader = asyncio.create_task(self.fresh_view())
        await asyncio.sleep(0)
        self.frames.begin_stream("fixture", UUID(self.session), self.visual_session)
        release.set()
        value = await asyncio.wait_for(reader, 1)
        self.assertEqual(value["reason"], "source_changed")
        self.assertNotIn("facts", value)

    async def test_fresh_view_fan_excludes_older_laptop_inference_dialogue_and_mixed_sources(self):
        old, current = self.buffered_frame(9), self.buffered_frame(10)
        mixed = {**fact("Desk and fan", "frame:1"), "sources": ["frame:0", "frame:1"]}
        self.model.generate_temporal_json.return_value = response({
            "setting": fact("Desk", "frame:0"),
            "activities": [{**fact("Working at laptop", "frame:0"), "status": "active"}],
            "relevant_details": [fact("Standing fan", "frame:1"),
                                 fact("Fan may be running", "frame:1", "inferred"), mixed],
            "stated_intentions": [fact("Wants to focus", "dialogue:0", "user_report")],
            "uncertainties": [fact("Fan power unreadable", "frame:1")]}, memory={
                "kind": "transition", "description": "Spoken fan claim",
                "sources": ["frame:1", "dialogue:0"]}, current_view=view_fact(source="frame:1"))
        self.runtime.submit_frame(old, b"old")
        self.runtime.submit_frame(current, b"new")
        self.runtime.submit_settled_caption(self.caption(), owner_id="fixture")
        await self.runtime.drain()
        value = await self.fresh_view()
        self.assertEqual([item["description"] for item in value["facts"]],
                         ["Standing fan"])
        for item in value["facts"]:
            self.assertEqual(item["sources"], [f"frame:{current.id}"])
            self.assertEqual(item["evidence_at"], current.captured_at.isoformat())

    async def test_unchanged_scene_with_reobserved_source_has_actual_fresh_answer(self):
        self.model.generate_temporal_json.return_value = response(
            {"setting": fact("Standing fan")}, current_view=view_fact())
        self.runtime.submit_frame(self.buffered_frame(9), b"old")
        await self.runtime.drain()
        current = self.buffered_frame(10)
        self.runtime.submit_frame(current, b"new")
        await self.runtime.drain()
        value = await self.fresh_view()
        self.assertEqual(value["facts"][0]["description"], "Standing fan")
        self.assertEqual(value["facts"][0]["sources"], [f"frame:{current.id}"])
        self.assertEqual(value["facts"][0]["evidence_at"], current.captured_at.isoformat())

    async def test_unchanged_null_or_carried_human_state_still_has_fresh_view_without_memory(self):
        self.model.generate_temporal_json.return_value = response({"setting": fact("Standing fan")})
        old = self.buffered_frame(9)
        self.runtime.submit_frame(old, b"old")
        await self.runtime.drain()
        carried = self.state.read()["human"]["content"]
        current = self.buffered_frame(10)
        for content in (None, carried):
            with self.subTest(output="null" if content is None else "carried"):
                self.model.generate_temporal_json.return_value = response(content, current_view=view_fact())
                self.runtime.submit_frame(current, b"new")
                await self.runtime.drain()
                value = await self.fresh_view()
                self.assertEqual(value["status"], "available")
                self.assertEqual(value["facts"][0]["description"], "Standing fan")
                self.assertEqual(value["facts"][0]["sources"], [f"frame:{current.id}"])
                self.assertEqual(value["facts"][0]["evidence_at"], current.captured_at.isoformat())
                self.assertEqual(self.state.read()["human"]["content"]["setting"]["sources"],
                                 [f"frame:{old.id}"])
                self.assertEqual(self.posts, [])

    async def test_missing_current_view_does_not_substitute_generic_human_state(self):
        self.model.generate_temporal_json.return_value = response({"setting": fact("Working at desk")})
        self.runtime.submit_frame(self.buffered_frame(), b"image")
        await self.runtime.drain()
        timeout = asyncio.timeout
        with patch("app.perception.companion_runtime.asyncio.timeout",
                   side_effect=lambda seconds: timeout(0.02 if seconds == 6 else seconds)):
            value = await self.fresh_view()
        self.assertEqual((value["status"], value["reason"]), ("freshness_unavailable", "deadline"))
        self.assertNotIn("facts", value)

    async def test_dialogue_uses_same_canonical_document_id_and_does_not_retain_again(self):
        self.model.generate_temporal_json.return_value = response({"stated_intentions": [
            fact("Needs to focus", "dialogue:0", "user_report")]})
        caption = self.caption()
        self.runtime.submit_settled_caption(caption, owner_id="fixture")
        await self.runtime.drain()
        human = self.state.read()["human"]
        self.assertEqual(human["source_document_ids"], [caption.document_id])
        self.assertEqual(human["content"]["stated_intentions"][0]["evidence_at"], NOW.isoformat().replace("+00:00", "Z"))
        self.assertEqual(self.posts, [])
        context = await self.runtime.read_context()
        self.assertEqual(context.human.revision, human["revision"])
        self.assertEqual(context.human.observed_through, NOW)

    async def test_visual_delta_has_stable_real_document_not_raw_frame_id(self):
        frame = self.frame()
        self.model.generate_temporal_json.return_value = response({"setting": fact()}, memory={
            "kind": "placement", "description": "Keys placed on counter", "sources": ["frame:0"]})
        self.runtime.submit_frame(frame, b"jpeg")
        await self.runtime.drain()
        expected = source_document_id("observation", self.session, f"multimodal-frame:{frame.id}")
        self.assertEqual(self.posts[0]["items"][0]["document_id"], expected)
        self.assertEqual(self.posts[0]["items"][0]["update_mode"], "replace")
        self.assertTrue(self.posts[0]["async"])
        self.assertEqual(self.state.read()["human"]["source_document_ids"], [expected])
        self.runtime.submit_frame(frame, b"jpeg")
        await self.runtime.drain()
        self.assertEqual(self.posts[1]["items"][0]["document_id"], expected)

    async def test_later_caption_dates_departure_without_refreshing_prior_intention(self):
        first = self.caption()
        first = replace(first, recorded_at=NOW + timedelta(seconds=2), content={
            **first.content, "observed_at": NOW.isoformat(),
            "text": "Remind me to take my umbrella when I leave."})
        self.model.generate_temporal_json.return_value = response({"stated_intentions": [
            fact("Take umbrella when leaving", "dialogue:0", "user_report")]})
        self.runtime.submit_settled_caption(first, owner_id="fixture")
        await self.runtime.drain()
        prior = self.state.read()["human"]["content"]["stated_intentions"][0]
        reminder = await self.state.manage_reminder(action="create", title="Umbrella", condition="Leaving")
        self.state.reminders.rows[reminder["id"]]["updated_at"] = self.now.isoformat()
        await self.state.refresh_reminders()
        end = NOW + timedelta(seconds=15)
        self.now = NOW + timedelta(seconds=40)
        second = replace(first, sequence=2, revision=2, recorded_at=NOW + timedelta(seconds=35),
            content={**first.content, "caption_revision": 2, "observed_at": end.isoformat(),
                     "text": first.content["text"] + " I am leaving now."})
        self.model.generate_temporal_json.return_value = response({
            "stated_intentions": [prior],
            "activities": [{**fact("Leaving", "dialogue:0", "user_report"), "status": "active"}]},
            reminders=[{"reminder_id": "reminder:0", "description": "Leaving now", "sources": ["dialogue:0"]}])
        self.runtime.submit_settled_caption(second, owner_id="fixture")
        await self.runtime.drain()
        human = self.state.read()["human"]
        self.assertEqual(human["content"]["stated_intentions"][0], prior)
        self.assertEqual(human["content"]["activities"][0]["evidence_at"], end.isoformat().replace("+00:00", "Z"))
        context = await self.runtime.read_context()
        self.assertEqual(context.human.observed_through, end)
        self.assertEqual(context.human.generated_at, self.now)
        self.assertEqual(context.human.stated_intentions[0].evidence_at, NOW)
        match, documents, _ = self.notices.call_args.args
        self.assertEqual(match.reminder_id, reminder["id"])
        self.assertEqual(match.evidence_at, end)
        self.assertEqual(documents, (first.document_id,))
        self.assertEqual(second.document_id, first.document_id)
        self.assertEqual(self.posts, [])

    async def test_caption_without_revision_time_does_not_use_later_commit_time(self):
        caption = replace(self.caption(), recorded_at=NOW + timedelta(seconds=8))
        self.model.generate_temporal_json.return_value = response({"stated_intentions": [
            fact("Needs to focus", "dialogue:0", "user_report")]})
        self.runtime.submit_settled_caption(caption, owner_id="fixture")
        await self.runtime.drain()
        context = await self.runtime.read_context()
        self.assertEqual(context.human.observed_through, NOW)
        self.assertEqual(context.human.stated_intentions[0].evidence_at, NOW)

    async def test_static_frame_updates_state_without_inventing_memory_document(self):
        self.model.generate_temporal_json.return_value = response({"setting": fact()})
        self.runtime.submit_frame(self.frame(), b"jpeg")
        await self.runtime.drain()
        self.assertEqual(self.posts, [])
        self.assertEqual(self.state.read()["human"]["source_document_ids"], [])

    async def test_reminder_callback_gets_actual_updated_at_and_no_implicit_completion(self):
        row = await self.state.manage_reminder(action="create", title="Keys", condition="Keys visible")
        self.state.reminders.rows[row["id"]]["updated_at"] = NOW.isoformat()
        await self.state.refresh_reminders()
        self.model.generate_temporal_json.return_value = response(reminders=[{
            "reminder_id": "reminder:0", "description": "Keys visible", "sources": ["frame:0"]}])
        self.runtime.submit_frame(self.frame(), b"jpeg")
        await self.runtime.drain()
        match, documents, generation = self.notices.call_args.args
        self.assertEqual(match.reminder_revision, NOW.isoformat())
        self.assertEqual(match.reminder_id, row["id"])
        self.assertEqual(generation, 0)
        self.assertEqual(documents, ())
        self.assertEqual(self.state.reminders.rows[row["id"]]["status"], "active")

    async def test_forget_during_analysis_suppresses_all_old_outputs(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def generate(**kwargs):
            entered.set()
            await release.wait()
            return response({"setting": fact()}, memory={"kind": "placement", "description": "Keys", "sources": ["frame:0"]})

        self.model.generate_temporal_json.side_effect = generate
        self.runtime.submit_frame(self.frame(), b"jpeg")
        await asyncio.wait_for(entered.wait(), 1)
        await self.state.invalidate_memory(NS(owner_id="fixture", document_ids=("forgotten",)))
        release.set()
        await self.runtime.drain()
        self.assertIsNone(self.state.read()["human"])
        self.assertEqual(self.posts, [])
        self.notices.assert_not_awaited()
        self.assertEqual(self.runtime.last_publication_issue, "discarded_pre_forget_result")

    async def test_memory_submission_does_not_block_next_perception_pass(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def retain(*args, **kwargs):
            entered.set()
            await release.wait()
            return {"operation_id": "fixture", "status": "accepted", "submitted": True}

        self.intake.retain_observation = AsyncMock(side_effect=retain)
        self.model.generate_temporal_json.return_value = response(memory={
            "kind": "placement", "description": "Keys", "sources": ["frame:0"]})
        self.runtime.submit_frame(self.frame(), b"jpeg")
        await self.runtime.drain(include_memory=False)
        await asyncio.wait_for(entered.wait(), 1)
        self.model.generate_temporal_json.return_value = response()
        self.runtime.submit_frame(self.frame(1), b"jpeg")
        await asyncio.wait_for(self.runtime.drain(include_memory=False), 1)
        self.assertEqual(self.model.generate_temporal_json.await_count, 2)
        release.set()
        await self.runtime.drain()

    async def test_memory_failure_does_not_undo_valid_human_state(self):
        self.intake.retain_observation = AsyncMock(side_effect=RuntimeError("synthetic"))
        self.model.generate_temporal_json.return_value = response({"setting": fact()}, memory={
            "kind": "placement", "description": "Keys", "sources": ["frame:0"]})
        self.runtime.submit_frame(self.frame(), b"jpeg")
        await self.runtime.drain()
        self.assertIsNotNone(self.state.read()["human"])
        self.assertEqual(self.runtime.last_memory_error, "RuntimeError")
        self.assertEqual(self.runtime.memory_submissions, 0)

    async def test_missing_intake_never_falls_back_to_direct_native_write(self):
        runtime = CompanionPerceptionRuntime(state=self.state, model=self.model, memory=self.memory,
                                            on_reminder=self.notices, clock=lambda: self.now)
        try:
            self.model.generate_temporal_json.return_value = response(memory={
                "kind": "placement", "description": "Keys", "sources": ["frame:0"]})
            runtime.submit_frame(self.frame(), b"jpeg")
            await runtime.drain()
            self.assertEqual(self.posts, [])
            self.assertEqual(runtime.last_memory_error, "serialized_memory_writer_not_configured")
        finally:
            await runtime.close()

    async def test_adapter_cannot_resurrect_a_tombstoned_visual_document(self):
        frame = self.frame()
        document_id = source_document_id("observation", self.session, f"multimodal-frame:{frame.id}")
        await self.intake.forget([document_id])
        self.model.generate_temporal_json.return_value = response(memory={
            "kind": "placement", "description": "Keys", "sources": ["frame:0"]})
        self.runtime.submit_frame(frame, b"jpeg")
        await self.runtime.drain()
        self.assertEqual(self.runtime.memory_submissions, 0)
        self.assertEqual(self.runtime.last_memory_status, "suppressed_forgotten")
        self.assertFalse(any(post and "items" in post for post in self.posts))

    async def test_timed_background_reminders_do_not_enter_situational_context(self):
        row = await self.state.manage_reminder(action="create", title="Later", condition="Leaving")
        self.state.reminders.rows[row["id"]].update(updated_at=NOW.isoformat(), remind_at=NOW.isoformat())
        await self.state.refresh_reminders()
        self.assertEqual((await self.runtime.read_context()).reminders, ())

    async def test_foreign_inputs_and_closed_runtime_rejected(self):
        frame = self.frame()
        frame.owner_id = "other"
        with self.assertRaises(ValueError):
            self.runtime.submit_frame(frame, b"jpeg")
        with self.assertRaises(ValueError):
            self.runtime.submit_settled_caption(self.caption(), owner_id="other")
        await self.runtime.close()
        with self.assertRaises(ValueError):
            self.runtime.submit_frame(self.frame(), b"jpeg")
