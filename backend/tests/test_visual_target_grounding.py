"""Real registered handlers/owners/briefs with controlled storage, not SQL/native proof."""
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.knowledge.observation_inputs import ObservationInputSnapshot, request_document
from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest
from app.perception.live_brief import seal_live_brief_input
from app.perception.prompts import build_fast_perception_prompt
from app.perception.references import PerceptionReferences
from app.tasks.models import digest
import test_visual_memory_observation_tools as existing


REQUEST = "Please remember where I put the pepper core."


def arguments(**changes):
    return {"title": "Requested visual memory", "question": "Where is the pepper grinder?",
            "object_association": "pepper grinder", **changes}


def span(text, literal):
    start = text.index(literal)
    return {"start": start, "end": start + len(literal)}


class VisualTargetGroundingTests(unittest.IsolatedAsyncioTestCase):
    invoke = existing.VisualMemoryAdmissionTests.invoke
    command = existing.VisualMemoryAdmissionTests.command
    capture_result = existing.VisualMemoryAdmissionTests.capture_result

    async def asyncSetUp(self):
        await existing.VisualMemoryAdmissionTests.asyncSetUp(self)
        self.store.events[self.source.request_event_id]["content"] = {"text": REQUEST}

    async def test_wrong_model_target_never_becomes_the_authoritative_question(self):
        result = await self.invoke("track_visual_memory", arguments())
        self.assertEqual(result["tracking"]["status"], "accepted")
        self.assertEqual(result["check"]["question"], REQUEST)
        grounding = result["check"]["request_grounding"]
        self.assertEqual(grounding["status"], "unresolved")
        self.assertEqual(grounding["reason"], "model_target_not_supported")
        self.assertTrue(grounding["clarification_needed"])
        self.assertFalse(grounding["intent_verified"])
        self.assertFalse(grounding["referent_resolved"])
        self.assertEqual(grounding["wording_sha256"], sha256(REQUEST.encode()).hexdigest())
        self.assertEqual(grounding["request_ref"]["source_id"], str(self.source.request_event_id))
        self.assertEqual(result["model_interpretation"]["object_association"], "pepper grinder")
        row, = self.store.occurrences.values()
        self.assertEqual(row["check_payload"]["visual_request"]["source_id"], str(self.source.request_event_id))
        self.assertNotIn("request_grounding", row["check_payload"])
        self.assertNotIn("user_wording", row["check_payload"])
        self.assertNotIn(REQUEST, json.dumps(row["check_payload"]))
        self.assertIsNone(result["check"]["resolution"])
        self.assertFalse(self.store.reminders)
        await self.capture_result("track_visual_memory", result)
        self.assertEqual(self.capture.roots, (self.store.events[self.source.request_event_id]["payload_id"],))
        self.assertTrue(any(edge[2] == self.source.request_event_id for edge in self.store.dependencies))

    async def test_literal_target_accepts_without_camera_or_extra_model_call(self):
        result = await self.invoke("track_visual_memory", arguments(object_association="pepper core"))
        grounding = result["check"]["request_grounding"]
        self.assertEqual(grounding["status"], "literal_supported")
        self.assertEqual(grounding["target_spans"], [span(REQUEST, "pepper core")])
        self.assertFalse(grounding["clarification_needed"])
        self.assertEqual(result["check"]["question"], REQUEST)
        self.assertFalse(any("visual_frames" in query for query in self.store.queries))
        self.assertFalse(result["location_fact_recorded_by_this_call"])

    async def test_legacy_ambiguous_interpretation_keeps_tracking_without_inventing_a_target(self):
        self.store.events[self.source.request_event_id]["content"] = {"text": existing.RAW_REQUEST}
        result = await self.invoke("track_visual_memory", existing.arguments())
        self.assertEqual(result["tracking"]["status"], "accepted")
        self.assertEqual(result["check"]["question"], existing.RAW_REQUEST)
        self.assertEqual(result["check"]["request_grounding"]["status"], "unresolved")
        self.assertEqual(len(self.store.occurrences), 1)

    async def test_deictic_multiobject_and_missing_targets_remain_unresolved(self):
        cases = [
            ("Remember where I put that.", "that", (), "deictic_reference_unresolved"),
            ("Remember where I put it.", "it", (), "deictic_reference_unresolved"),
            ("Track the keys and wallet.", None,
             (span("Track the keys and wallet.", "keys"), span("Track the keys and wallet.", "wallet")),
             "multiple_targets_unresolved"),
            ("Keep observing what I requested.", None, (), "no_target_span"),
        ]
        for text, association, spans, reason in cases:
            with self.subTest(reason=reason, text=text):
                source = self.store.event(self.scope.owner_id, self.scope.session_id, content={"text": text})
                result = await self.invoke("track_visual_memory", arguments(
                    object_association=association, target_spans=spans), source=source)
                grounding = result["check"]["request_grounding"]
                self.assertEqual(result["tracking"]["status"], "accepted")
                self.assertEqual(result["check"]["question"], text)
                self.assertEqual(grounding["reason"], reason)
                self.assertEqual(grounding["status"], "unresolved")
                self.assertFalse(grounding["intent_verified"])
                self.assertFalse(grounding["referent_resolved"])

    async def test_literal_span_preserves_negation_not_proof_of_requested_target(self):
        text = "Not the grinder; track the core instead."
        self.store.events[self.source.request_event_id]["content"] = {"text": text}
        result = await self.invoke("track_visual_memory", arguments(object_association=None,
            target_spans=[span(text, "grinder")]))
        grounding = result["check"]["request_grounding"]
        self.assertEqual(grounding["status"], "literal_supported")
        self.assertEqual(result["check"]["question"], text)
        self.assertFalse(grounding["intent_verified"])
        self.assertFalse(grounding["referent_resolved"])
        self.assertIsNone(result["check"]["resolution"])

    async def test_unicode_offsets_and_case_matching_preserve_exact_source_wording(self):
        text = "Remember the caf\u00e9 key."
        self.store.events[self.source.request_event_id]["content"] = {"text": text}
        result = await self.invoke("track_visual_memory", arguments(object_association="CAF\u00c9 KEY",
            target_spans=[span(text, "caf\u00e9 key")]))
        grounding = result["check"]["request_grounding"]
        self.assertEqual(grounding["status"], "literal_supported")
        self.assertEqual(grounding["user_wording"], text)
        self.assertEqual(grounding["wording_sha256"], sha256(text.encode()).hexdigest())

    async def test_invalid_partial_overlapping_and_repeated_spans_do_not_bind(self):
        cases = [
            ([{"start": 999, "end": 1000}], "invalid_target_span"),
            ([span(REQUEST, "pepper"), span(REQUEST, "pepper core")], "invalid_target_span"),
            ([span(REQUEST, "ore")], "invalid_target_span"),
        ]
        for spans, reason in cases:
            with self.subTest(spans=spans):
                source = self.store.event(self.scope.owner_id, self.scope.session_id, content={"text": REQUEST})
                result = await self.invoke("track_visual_memory", arguments(
                    object_association=None, target_spans=spans), source=source)
                self.assertEqual(result["check"]["request_grounding"]["reason"], reason)
                self.assertEqual(result["check"]["question"], REQUEST)
        repeated = "Track the key, not the other key."
        source = self.store.event(self.scope.owner_id, self.scope.session_id, content={"text": repeated})
        result = await self.invoke("track_visual_memory", arguments(object_association="key"), source=source)
        self.assertEqual(result["check"]["request_grounding"]["reason"], "ambiguous_literal_match")

    async def test_provisional_missing_deleted_and_overbudget_wording_cannot_admit(self):
        cases = ({"finality": "provisional"}, {"finality": "unavailable"},
            {"content": {"text": None}}, {"content": {"text": "   "}},
            {"content": {"text": "x" * 1001}}, {"availability": "deleted", "content": None},
            {"owner_id": "another-owner"}, {"session_id": uuid4()})
        for changes in cases:
            with self.subTest(changes=changes):
                source = self.store.event(self.scope.owner_id, self.scope.session_id, **changes)
                result = await self.invoke("track_visual_memory", arguments(), source=source)
                self.assertEqual(result["status"], "failed")
                self.assertIn(result["error"]["code"], {"request_evidence_unavailable", "request_wording_unavailable"})
        self.assertFalse(self.store.occurrences)

    async def test_same_request_dedupes_and_changed_arguments_conflict(self):
        first = await self.invoke("track_visual_memory", arguments())
        retry = await self.invoke("track_visual_memory", arguments(),
                                  source=replace(self.source, provider_call_id=str(uuid4())))
        self.assertTrue(retry["replayed"])
        self.assertEqual(first["check"], retry["check"])
        conflict = await self.invoke("track_visual_memory", arguments(object_association="pepper core"))
        self.assertEqual(conflict["error"]["code"], "request_conflict")
        self.assertEqual(len(self.store.occurrences), 1)

    async def test_correction_rebinds_wording_and_snooze_keeps_that_origin(self):
        first = await self.invoke("track_visual_memory", arguments())
        text = "Track the token instead."
        correction = self.store.event(self.scope.owner_id, self.scope.session_id, content={"text": text})
        changed = await self.invoke("change_episode_check", self.command(first, "revise",
            check={"question": "Watch the token", "object_association": "token"}), source=correction)
        self.assertEqual(changed["check"]["question"], text)
        self.assertEqual(changed["check"]["revision"], first["check"]["revision"] + 1)
        self.assertEqual(changed["check"]["request_grounding"]["request_ref"]["source_id"],
                         str(correction.request_event_id))
        later = self.store.event(self.scope.owner_id, self.scope.session_id, content={"text": "Snooze that check."})
        snoozed = await self.invoke("change_episode_check", self.command(changed, "snooze",
            until=(self.now + timedelta(minutes=1)).isoformat()), source=later)
        self.assertEqual(snoozed["check"]["question"], text)
        read = await self.owner.read(self.scope)
        self.assertIn(correction.request_event_id, {ref.source_id for ref in read.snapshots[0].dependencies})

    async def test_withdrawal_removes_quote_and_does_not_prevent_authorized_cancel(self):
        result = await self.invoke("track_visual_memory", arguments())
        self.store.events[self.source.request_event_id].update(availability="deleted", content=None)
        read = await self.owner.read(self.scope)
        self.assertEqual(read.snapshots, ())
        with self.assertRaises(Exception):
            await self.capture_result("track_visual_memory", result)
        cancel = self.store.event(self.scope.owner_id, self.scope.session_id, content={"text": "Cancel tracking."})
        cancelled = await self.invoke("change_episode_check", self.command(result, "cancel"), source=cancel)
        self.assertEqual(cancelled["check"]["lifecycle"], "dismissed")
        self.assertIsNone(cancelled["check"]["resolution"])
        self.assertIsNone(cancelled["check"]["request_grounding"])
        self.assertNotIn(REQUEST, json.dumps(cancelled))
        self.assertNotIn("pepper grinder", json.dumps(cancelled))

    async def test_source_revision_replacement_fences_original_projection(self):
        result = await self.invoke("track_visual_memory", arguments())
        self.store.events[self.source.request_event_id]["current"] = False
        read = await self.owner.read(self.scope)
        self.assertEqual(read.snapshots, ())
        self.assertTrue(read.gaps)
        with self.assertRaises(Exception):
            await self.capture_result("track_visual_memory", result)

    async def test_tool_cannot_supply_trusted_grounding_or_request_reference(self):
        for extra in ({"request_grounding": {}}, {"visual_request": {}},
                      {"target_spans": [{"start": True, "end": 2}]},
                      {"target_spans": [{"start": 2, "end": 1}]}):
            with self.subTest(extra=extra):
                result = await self.invoke("track_visual_memory", arguments(**extra))
                self.assertEqual(result["error"]["code"], "invalid_tool_input")
        self.assertFalse(self.store.occurrences)


class GroundedVisualCompositionTests(unittest.IsolatedAsyncioTestCase):
    invoke = existing.VisualMemoryCompositionTests.invoke
    command = existing.VisualMemoryCompositionTests.command
    live = existing.VisualMemoryCompositionTests.live
    until = existing.VisualMemoryCompositionTests.until
    asyncTearDown = existing.VisualMemoryCompositionTests.asyncTearDown

    async def asyncSetUp(self):
        await existing.VisualMemoryCompositionTests.asyncSetUp(self)
        self.db.events[self.source.request_event_id]["content"] = {"text": REQUEST}

    async def test_real_brief_prompt_and_guard_use_source_wording_then_withdraw(self):
        result = await self.invoke("track_visual_memory", arguments())
        live = self.live(perception=True)
        await live.start()
        await self.until(lambda: live.perception_brief_bridge._binding is not None
            and live.perception_brief_bridge.composition is not None
            and any(q.question == REQUEST for q in live.perception_brief_bridge.composition.brief.questions))
        brief = live.perception_brief_bridge.composition.brief
        binding = live.perception_brief_bridge._binding
        question, = brief.questions
        self.assertEqual(str(question.check_ref.source_id), result["check"]["id"])
        self.assertEqual(question.request_grounding.status, "unresolved")
        self.assertNotIn("grinder", brief.model_dump_json())
        async with live.perception_runtime.binder.guard(brief, binding):
            pass
        request = FastPerceptionRequest(owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            window_id=uuid4(), frames=(FastPerceptionFrame(uuid4(), 1, self.now, "glasses", "image/jpeg", b"offline"),),
            previous_working_state={}, existing_entity_ids=frozenset(), recent_audio_transcripts=(),
            behavior_instructions="Observe only.", active_reminders=(), active_focus=None,
            recent_interrupts=(), requested_at=self.now)
        request = live.perception_brief_bridge.attach(request)
        request = seal_live_brief_input(request, ObservationInputSnapshot(self.scope.owner_id,
            self.scope.session_id, uuid4(), digest(request_document(request))))
        with patch("app.perception.live_brief._utcnow", return_value=self.now):
            prompt = build_fast_perception_prompt(request, references=PerceptionReferences(request))
        context = json.loads(prompt.split("Context JSON:\n```json\n", 1)[1].split("\n```", 1)[0])
        assigned, = context["perception_assignment"]["questions"]
        self.assertEqual(assigned["question"], REQUEST)
        self.assertEqual(assigned["request_grounding"]["user_wording"], REQUEST)
        self.assertNotIn("grinder", json.dumps(context["perception_assignment"]))
        self.db.events[self.source.request_event_id].update(availability="deleted", content=None)
        with self.assertRaises(SourceError):
            async with live.perception_runtime.binder.guard(brief, binding):
                self.fail("Withdrawn quote was usable")
        await self.until(lambda: live.perception_brief_bridge.composition is not None
            and not live.perception_brief_bridge.composition.brief.questions)
        self.assertNotIn(REQUEST, live.perception_brief_bridge.composition.brief.model_dump_json())
        self.assertFalse(self.db.reminders)
