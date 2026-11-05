from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from pydantic import ValidationError

from app.live_state.human import HumanStateContent, stamp_human_state
from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest, FastPerceptionTranscript
from app.perception.multimodal import (
    EventBatch, MultimodalEventAnalyzer, MultimodalRequest, PerceptionContext, SettledDialogue,
    SituationalReminder, build_multimodal_prompt, coalesce_events, parse_multimodal_output,
)


NOW = datetime(2026, 11, 4, 12, tzinfo=UTC)


def frame(second=0):
    return FastPerceptionFrame(uuid4(), second, NOW + timedelta(seconds=second), "glasses", "image/jpeg", b"image")


def dialogue(second=0, **changes):
    return SettledDialogue(**{
        "caption_id": "caption-one", "revision": 1, "speaker": "user", "text": "I need to focus now.",
        "observed_from": NOW + timedelta(seconds=second), "observed_until": NOW + timedelta(seconds=second),
        "finality": "application_settled", "time_basis": "application_ingress_approximate", **changes})


def fact(description="At the kitchen counter", source="frame:0", basis="observed", second=0):
    return {"description": description, "sources": [source], "basis": basis,
            "evidence_at": (NOW + timedelta(seconds=second)).isoformat()}


def view_fact(description="Standing fan", source="frame:0"):
    return {"description": description, "sources": [source]}


def response(human=None, reminders=None, memory=None, current_view=None):
    return json.dumps({"updated_human_state": human, "triggered_reminders": reminders or [],
                       "memory_worthy_observation": memory, "current_view": current_view})


def request(frames=None, speech=(), previous=None, reminders=(), second=10):
    return MultimodalRequest(EventBatch("owner", (frame(),) if frames is None else frames, speech),
                             PerceptionContext(previous, reminders), NOW + timedelta(seconds=second))


def parse(raw, req):
    return parse_multimodal_output(raw, req, analyzed_at=req.current_time + timedelta(seconds=1))


class MultimodalContractTests(unittest.TestCase):
    def test_current_view_uses_newest_capture_not_list_order_or_generation_time(self):
        newest = frame(5)
        req = request(frames=(newest, frame(1)))
        result = parse(response(current_view=view_fact()), req)
        self.assertEqual(req.current_view_source(), "frame:0")
        self.assertEqual(result.current_view.sources, (f"frame:{newest.frame_id}",))
        self.assertEqual(result.current_view.evidence_at, newest.captured_at)
        self.assertEqual(result.current_view.basis, "observed")
        self.assertIsNone(result.updated_human_state)
        self.assertIsNone(result.memory_worthy_observation)

    def test_current_view_rejects_old_mixed_dialogue_and_invented_sources(self):
        req = request(frames=(frame(5), frame(1)), speech=(dialogue(9),))
        for sources in (["frame:1"], ["dialogue:0"], ["frame:99"],
                        ["frame:0", "frame:1"], ["frame:0", "frame:0"]):
            with self.subTest(sources=sources):
                result = parse(response(current_view={"description": "Fan", "sources": sources}), req)
                self.assertIsNone(result.current_view)
                self.assertIn("current_view:invalid_or_noncurrent_frame", result.rejected)

    def test_current_view_cannot_supply_timestamp_or_describe_dialogue_only_pass(self):
        forged = {**view_fact(), "evidence_at": NOW.isoformat()}
        self.assertIsNone(parse(response(current_view=forged), request()).current_view)
        req = request(frames=(), speech=(dialogue(),))
        self.assertIsNone(parse(response(current_view=view_fact(source="dialogue:0")), req).current_view)
        self.assertEqual(parse(response(), req).rejected, ())

    def test_result_preserves_generation_captured_before_analysis(self):
        req = replace(request(), context=PerceptionContext(human_generation=7))
        result = parse(response({"setting": fact()}), req)
        self.assertEqual(result.human_generation, 7)
        self.assertEqual(result.expected_human_revision, 0)

    def test_legacy_request_accepts_dialogue_but_not_empty_input(self):
        fields = dict(owner_id="owner", session_id=uuid4(), window_id=uuid4(), frames=(),
                      previous_working_state={}, existing_entity_ids=frozenset(),
                      behavior_instructions="", active_reminders=(), active_focus=None,
                      recent_interrupts=(), requested_at=NOW)
        with self.assertRaises(ValueError):
            FastPerceptionRequest(**fields, recent_audio_transcripts=())
        accepted = FastPerceptionRequest(**fields, recent_audio_transcripts=(FastPerceptionTranscript("user", "Focus", NOW),))
        with self.assertRaisesRegex(ValueError, "dialogue-only"):
            _ = accepted.primary_frame

    def test_settled_dialogue_only_intent_and_advisory_availability(self):
        req = request(frames=(), speech=(dialogue(),))
        result = parse(response({"stated_intentions": [fact("Needs to focus", "dialogue:0", "user_report")],
                                 "availability": "do_not_disturb"}), req)
        self.assertEqual(result.updated_human_state.revision, 1)
        self.assertEqual(result.updated_human_state.availability, "do_not_disturb")
        self.assertNotIn("quiet_mode", result.updated_human_state.model_dump())
        prompt = build_multimodal_prompt(req)
        self.assertIn('"frames":[]', prompt)
        self.assertIn("application_settled", prompt)
        self.assertIn("application_ingress_approximate", prompt)

    def test_server_stamps_generation_and_observation_time(self):
        req = request()
        result = parse(response({"setting": fact(second=500)}), req)
        human = result.updated_human_state
        self.assertEqual(human.setting.evidence_at, NOW)
        self.assertEqual(human.generated_at, req.current_time + timedelta(seconds=1))
        self.assertEqual(human.setting.sources, (f"frame:{req.batch.frames[0].frame_id}",))

    def test_model_cannot_assign_server_revision(self):
        result = parse(response({"revision": 90, "setting": fact()}), request())
        self.assertIsNone(result.updated_human_state)
        self.assertIn("human:invalid_or_stale_snapshot", result.rejected)

    def test_invalid_reminder_and_frame_references_do_not_discard_human(self):
        req = request(reminders=(SituationalReminder(id="r", revision=2, condition="Leaving"),))
        matches = [{"reminder_id": "invented", "description": "Leaving", "sources": ["frame:0"]},
                   {"reminder_id": "reminder:0", "description": "Leaving", "sources": ["frame:99"]}]
        result = parse(response({"setting": fact()}, matches,
                                {"kind": "placement", "description": "Keys", "sources": ["frame:99"]}), req)
        self.assertIsNotNone(result.updated_human_state)
        self.assertEqual(result.triggered_reminders, ())
        self.assertIsNone(result.memory_worthy_observation)
        self.assertEqual(len(result.rejected), 3)

    def test_reminder_uses_captured_revision_not_a_guessed_current_one(self):
        req = request(reminders=(SituationalReminder(id="actual-r", revision=7, condition="Keys visible"),))
        match = {"reminder_id": "reminder:0", "description": "Keys visible", "sources": ["frame:0"]}
        result = parse(response(reminders=[match, match]), req)
        self.assertEqual(len(result.triggered_reminders), 1)
        self.assertEqual(result.triggered_reminders[0].reminder_id, "actual-r")
        self.assertEqual(result.triggered_reminders[0].reminder_revision, 7)

    def test_assistant_claim_is_not_user_intent_or_camera_observation(self):
        req = request(frames=(), speech=(dialogue(speaker="assistant"),))
        result = parse(response({"stated_intentions": [fact("Will cook", "dialogue:0", "user_report")],
                                 "setting": fact("In kitchen", "dialogue:0", "observed")}), req)
        self.assertEqual(result.updated_human_state.stated_intentions, ())
        self.assertIsNone(result.updated_human_state.setting)
        self.assertEqual(len(result.rejected), 2)

    def test_prior_inference_cannot_be_promoted_to_observed_without_current_source(self):
        initial = parse(response({"setting": fact(basis="inferred")}), request()).updated_human_state
        promoted = initial.content().model_dump(mode="json")
        promoted["setting"]["basis"] = "observed"
        req = request(frames=(frame(2),), previous=initial)
        result = parse(response(promoted), req)
        self.assertIsNone(result.updated_human_state)
        self.assertIn("human.setting:invalid_or_stale_evidence", result.rejected)

    def test_late_evidence_can_be_memory_but_cannot_replace_current_state(self):
        initial = parse(response({"setting": fact("Office")}), request(frames=(frame(5),))).updated_human_state
        result = parse(response({"setting": fact("Kitchen")}, memory={"kind": "placement",
                              "description": "Keys were placed on counter", "sources": ["frame:0"]}),
                       request(previous=initial))
        self.assertIsNone(result.updated_human_state)
        self.assertEqual(result.memory_worthy_observation.evidence_at, NOW)

    def test_mixed_old_frame_new_dialogue_preserves_fresher_setting(self):
        initial = parse(response({"setting": fact("Office")}), request(frames=(frame(5),))).updated_human_state
        req = request(frames=(frame(),), speech=(dialogue(8),), previous=initial)
        result = parse(response({"setting": fact("Kitchen"),
                                 "stated_intentions": [fact("Focus", "dialogue:0", "user_report")]}), req)
        self.assertEqual(result.updated_human_state.setting.description, "Office")
        self.assertEqual(result.updated_human_state.stated_intentions[0].evidence_at, NOW + timedelta(seconds=8))

    def test_cooking_and_departure_can_coexist_without_resolving_stove(self):
        req = request(speech=(dialogue(text="Answering the door then returning"),))
        human = {"activities": [{**fact("Cooking"), "status": "interrupted"},
                                 {**fact("Answering door", "dialogue:0", "user_report"), "status": "active"}],
                 "uncertainties": [fact("Stove shutdown unconfirmed", basis="inferred")]}
        result = parse(response(human), req)
        self.assertEqual(len(result.updated_human_state.activities), 2)
        self.assertEqual(result.triggered_reminders, ())

    def test_snapshot_refresh_does_not_refresh_old_fact_evidence(self):
        previous = parse(response({"setting": fact()}), request()).updated_human_state
        human = previous.content().model_dump(mode="json")
        human["availability"] = "focused"
        req = request(frames=(), speech=(dialogue(600),), previous=previous, second=601)
        result = parse(response(human), req)
        context = result.updated_human_state.context(req.current_time)
        self.assertTrue(context["setting"]["stale"])
        self.assertEqual(result.updated_human_state.setting.evidence_at, NOW)

    def test_static_null_and_dialogue_memory_do_not_duplicate_ingestion(self):
        self.assertIsNone(parse(response(), request()).memory_worthy_observation)
        result = parse(response(memory={"kind": "transition", "description": "User focus request",
                                        "sources": ["dialogue:0"]}), request(frames=(), speech=(dialogue(),)))
        self.assertIsNone(result.memory_worthy_observation)

    def test_invalid_human_reference_preserves_other_valid_sections(self):
        req = request(speech=(dialogue(),))
        result = parse(response({"setting": fact(source="frame:500"), "stated_intentions": [
            fact("Focus", "dialogue:0", "user_report")]}), req)
        self.assertIsNone(result.updated_human_state.setting)
        self.assertEqual(len(result.updated_human_state.stated_intentions), 1)

    def test_no_empty_or_future_or_provisional_input(self):
        with self.assertRaises(ValueError):
            EventBatch("owner")
        with self.assertRaises(ValueError):
            request(frames=(frame(50),))
        with self.assertRaises(ValidationError):
            dialogue(finality="provisional")

    def test_coalescer_bounds_whole_inputs_and_keeps_latest_caption(self):
        pending = EventBatch("owner", (frame(),), (dialogue(),))
        for second in range(1, 25):
            pending = coalesce_events(pending, EventBatch("owner", (frame(second),), (
                dialogue(second, caption_id=f"c{second}", text="x" * 1000),)))
        self.assertLessEqual(len(pending.frames), 4)
        self.assertLessEqual(sum(len(d.text) for d in pending.dialogue), 12000)
        self.assertGreater(pending.omitted_frames, 0)
        self.assertGreater(pending.omitted_dialogue, 0)
        old = EventBatch("owner", dialogue=(dialogue(),))
        newer = EventBatch("owner", dialogue=(dialogue(revision=2, text="Whole corrected caption"),))
        combined = coalesce_events(newer, old)
        self.assertEqual(combined.dialogue[0].text, "Whole corrected caption")

    def test_coalescer_does_not_join_owners_or_conflicting_revisions(self):
        batch = EventBatch("owner", dialogue=(dialogue(),))
        with self.assertRaises(ValueError):
            coalesce_events(batch, EventBatch("other", dialogue=(dialogue(),)))
        with self.assertRaises(ValueError):
            coalesce_events(batch, EventBatch("owner", dialogue=(dialogue(text="Different"),)))


class SingleflightTests(unittest.IsolatedAsyncioTestCase):
    async def test_visual_waiter_sees_published_result_and_text_only_pass_does_not_replace_it(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def publish(result):
            entered.set()
            await release.wait()

        model = type("Model", (), {"generate_temporal_json": AsyncMock(side_effect=[
            response({"setting": fact("Standing fan")}), response()])})()
        analyzer = MultimodalEventAnalyzer(owner_id="owner", model=model,
            read_context=AsyncMock(return_value=PerceptionContext()), publish=publish, clock=lambda: NOW)
        self.addAsyncCleanup(analyzer.close)
        reader = asyncio.create_task(analyzer.wait_for_visual_result(None))
        analyzer.submit(EventBatch("owner", (frame(),)))
        await asyncio.wait_for(entered.wait(), 1)
        self.assertIsNone(analyzer.latest_visual_result)
        self.assertFalse(reader.done())
        release.set()
        result = await asyncio.wait_for(reader, 1)
        self.assertEqual(result.updated_human_state.setting.description, "Standing fan")
        analyzer.submit(EventBatch("owner", dialogue=(dialogue(),)))
        await analyzer.drain()
        self.assertIs(analyzer.latest_visual_result, result)
        self.assertIs(await analyzer.wait_for_visual_result(None), result)
        self.assertEqual(model.generate_temporal_json.await_count, 2)
        await analyzer.close()
        self.assertIsNone(analyzer.latest_visual_result)

    async def test_close_wakes_visual_waiter_without_starting_model_work(self):
        model = type("Model", (), {"generate_temporal_json": AsyncMock()})()
        analyzer = MultimodalEventAnalyzer(owner_id="owner", model=model,
            read_context=AsyncMock(return_value=PerceptionContext()), publish=AsyncMock(), clock=lambda: NOW)
        reader = asyncio.create_task(analyzer.wait_for_visual_result(None))
        await asyncio.sleep(0)
        await analyzer.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await asyncio.wait_for(reader, 1)
        model.generate_temporal_json.assert_not_awaited()

    async def test_one_call_at_a_time_pending_inputs_coalesce_and_context_is_fresh(self):
        entered, release = asyncio.Event(), asyncio.Event()
        prompts, results = [], []
        active = 0
        maximum = 0

        async def generate(**kwargs):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            prompts.append(kwargs)
            if len(prompts) == 1:
                entered.set()
                await release.wait()
            active -= 1
            return response()

        model = type("Model", (), {})()
        model.generate_temporal_json = generate
        context = AsyncMock(return_value=PerceptionContext())
        publish = AsyncMock(side_effect=results.append)
        analyzer = MultimodalEventAnalyzer(owner_id="owner", model=model, read_context=context,
                                           publish=publish, clock=lambda: NOW + timedelta(seconds=100))
        analyzer.submit(EventBatch("owner", (frame(),)))
        await asyncio.wait_for(entered.wait(), 1)
        for second in range(1, 20):
            analyzer.submit(EventBatch("owner", (frame(second),), (dialogue(second, revision=second),)))
        release.set()
        await analyzer.drain()
        self.assertEqual(maximum, 1)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(context.await_count, 2)
        self.assertEqual(len(prompts[-1]["images"]), 4)
        self.assertGreater(results[-1].omitted_frames, 0)
        await analyzer.close()

    async def test_text_only_has_zero_image_attachments_and_one_model_call(self):
        model = type("Model", (), {"generate_temporal_json": AsyncMock(return_value=response())})()
        publish = AsyncMock()
        analyzer = MultimodalEventAnalyzer(owner_id="owner", model=model,
            read_context=AsyncMock(return_value=PerceptionContext()), publish=publish, clock=lambda: NOW)
        analyzer.submit(EventBatch("owner", dialogue=(dialogue(),)))
        await analyzer.drain()
        self.assertEqual(model.generate_temporal_json.await_count, 1)
        self.assertEqual(model.generate_temporal_json.call_args.kwargs["images"], ())
        publish.assert_awaited_once()
        await analyzer.close()

    async def test_failure_is_visible_without_automatic_retries_and_close_rejects_input(self):
        model = type("Model", (), {"generate_temporal_json": AsyncMock(side_effect=RuntimeError("provider failed"))})()
        publish = AsyncMock()
        analyzer = MultimodalEventAnalyzer(owner_id="owner", model=model,
            read_context=AsyncMock(return_value=PerceptionContext()), publish=publish, clock=lambda: NOW)
        analyzer.submit(EventBatch("owner", (frame(),)))
        await analyzer.drain()
        self.assertEqual(analyzer.last_error, "RuntimeError")
        self.assertEqual(model.generate_temporal_json.await_count, 1)
        publish.assert_not_awaited()
        await analyzer.close()
        with self.assertRaises(ValueError):
            analyzer.submit(EventBatch("owner", (frame(),)))
