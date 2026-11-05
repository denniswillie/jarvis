"""Offline real tool/owner -> source-bound check -> perception composition.

Existing controlled storage replaces PostgreSQL; no visual answer is fabricated
and no provider runs. This qualifies admission/context, not model recognition.
"""
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.knowledge.observation_inputs import ObservationInputSnapshot, request_document
from app.live_state.models import SessionScope
from app.live_state.perception_runtime import CanonicalPerceptionBinder
from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest
from app.perception.live_brief import seal_live_brief_input
from app.perception.prompts import build_fast_perception_prompt
from app.perception.references import PerceptionReferences
from app.reminders.duties import ResponsibilityRepository
from app.reminders.episode_checks import EpisodeCheckOwner
from app.reminders.episode_tools import EpisodeCheckToolModule
from app.runtime.events import DomainEventBus
from app.runtime.feature_config import MemoryAuthority
from app.tasks.models import digest
from app.tools.module import ToolBuildContext
from app.tools.provenance import SourceBoundToolResult
import test_episode_check_tools as tool_fixture
import test_live_state_episode_context as composition_fixture


RAW_REQUEST = "Please observe where I put the synthetic cap, or possibly a lid; keep that uncertainty."
QUESTION = "Where is the user's cap or lid visible? Report its position and uncertainty, not an inferred release."


def arguments():
    return {"title": "Requested visual observation", "question": QUESTION,
            "object_association": "cap or lid; identity uncertain"}


class VisualMemoryAdmissionTests(unittest.IsolatedAsyncioTestCase):
    # Reuse helpers, not the TestCase superclass (no duplicate test collection).
    invoke = tool_fixture.EpisodeToolTests.invoke
    command = tool_fixture.EpisodeToolTests.command
    capture_result = tool_fixture.EpisodeToolTests.capture_result

    async def asyncSetUp(self):
        await tool_fixture.EpisodeToolTests.asyncSetUp(self)
        self.store.events[self.source.request_event_id]["content"] = {"text": RAW_REQUEST}

    async def test_request_required_and_assistant_interpretation_cannot_admit(self):
        result = await self.tools["track_visual_memory"].handler(arguments())
        self.assertEqual(result["error"]["code"], "request_evidence_required")
        assistant = self.store.event(self.scope.owner_id, self.scope.session_id,
            kind="assistant_output", trust="assistant_output", content={"text": "It must be another object."})
        result = await self.invoke("track_visual_memory", arguments(), source=assistant)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.store.rules)

    async def test_retry_keeps_one_identity_raw_source_and_no_reminder_or_location_fact(self):
        first = await self.invoke("track_visual_memory", arguments())
        retry_source = replace(self.source, provider_call_id=str(uuid4()))
        replay = await self.invoke("track_visual_memory", arguments(), source=retry_source)
        self.assertEqual(first["tracking"]["status"], "accepted")
        self.assertTrue(replay["replayed"])
        self.assertEqual(first["check"]["id"], replay["check"]["id"])
        self.assertEqual(len(self.store.rules), 1)
        self.assertEqual(len(self.store.occurrences), 1)
        self.assertTrue(first["observation_only"])
        self.assertFalse(first["location_fact_recorded_by_this_call"])
        self.assertFalse(first["physical_action_verified"])
        self.assertIsNone(first["check"]["resolution"])
        self.assertEqual(first["check"]["evidence_refs"], [])
        row = next(iter(self.store.occurrences.values()))
        self.assertIsNone(row["check_payload"].get("visual_resolution"))
        self.assertEqual(row["check_payload"]["admission_request"]["source_id"], str(self.source.request_event_id))
        self.assertEqual(self.store.events[self.source.request_event_id]["content"], {"text": RAW_REQUEST})
        rule = next(iter(self.store.rules.values()))
        self.assertEqual(rule["spec"]["kind"], "episode_check")
        self.assertIsNone(rule["next_due_at"])
        self.assertEqual(await self.owner.repository.materialize(self.scope.owner_id, now=self.now + timedelta(days=1)), 0)
        self.assertFalse(self.store.reminders)
        captured = await self.capture_result("track_visual_memory", first)
        self.assertIsInstance(captured, SourceBoundToolResult)
        self.assertEqual(set(self.capture.roots), {self.store.events[self.source.request_event_id]["payload_id"]})

    async def test_no_location_resolution_notification_or_source_override_arguments(self):
        forbidden = ({"location": "invented answer"}, {"visual_resolution": {}},
                     {"notify": True}, {"source_event_id": str(uuid4())},
                     {"request_slot": True}, {"question": ""})
        for extra in forbidden:
            with self.subTest(extra=tuple(extra)):
                result = await self.invoke("track_visual_memory", arguments() | extra)
                self.assertEqual(result["error"]["code"], "invalid_tool_input")
        self.assertFalse(self.store.rules)

    async def test_changed_same_request_slot_conflicts_without_second_admission(self):
        await self.invoke("track_visual_memory", arguments())
        result = await self.invoke("track_visual_memory", arguments() | {"question": "Inspect a different object."})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(self.store.occurrences), 1)

    async def test_unavailable_and_foreign_request_fail_without_acceptance(self):
        foreign = self.store.event("other-owner", self.scope.session_id, content={"text": RAW_REQUEST})
        result = await self.invoke("track_visual_memory", arguments(), source=foreign)
        self.assertEqual(result["status"], "failed")
        self.owner._unavailable()
        result = await self.invoke("track_visual_memory", arguments())
        self.assertEqual(result["error"]["code"], "episode_tracking_unavailable")
        self.assertNotIn("observation_only", result)
        self.assertFalse(self.store.rules)


class VisualMemoryCompositionTests(unittest.IsolatedAsyncioTestCase):
    invoke = tool_fixture.EpisodeToolTests.invoke
    command = tool_fixture.EpisodeToolTests.command
    live = composition_fixture.EpisodeContextIntegrationTests.live
    until = composition_fixture.EpisodeContextIntegrationTests.until
    asyncTearDown = composition_fixture.EpisodeContextIntegrationTests.asyncTearDown

    async def asyncSetUp(self):
        self.db = composition_fixture.IntegrationStore()
        self.now = composition_fixture.NOW + timedelta(seconds=1)
        self.scope = SessionScope(owner_id="visual-memory-offline", session_id=uuid4())
        self.source = self.db.event(self.scope.owner_id, self.scope.session_id,
            at=self.now, content={"text": RAW_REQUEST})
        self.owner = EpisodeCheckOwner(ResponsibilityRepository(self.db), enabled=True, clock=lambda: self.now)
        await self.owner.initialize()
        self.settings = SimpleNamespace(RESPONSIVE_VOICE_ENABLED=True)
        self.services = SimpleNamespace(memory_authority=MemoryAuthority.LEGACY, _closed=False,
            background_task_context_service=SimpleNamespace(repository=SimpleNamespace(pool=self.db,
                snapshot=AsyncMock(return_value={"recent": [], "pinned": []}))),
            background_task_service=None, db_pool=self.db, domain_event_bus=DomainEventBus(),
            settings=self.settings, responsibility_issue=None,
            attention_service=SimpleNamespace(_closed=False, responsibilities=SimpleNamespace(episode_checks=self.owner)))
        self.tools = {tool.name: tool for tool in EpisodeCheckToolModule(self.owner).build_tools(
            ToolBuildContext(settings=self.settings, owner_id=self.scope.owner_id, session_id=self.scope.session_id))}
        self.lives = []

    async def admitted_brief(self):
        admitted = await self.invoke("track_visual_memory", arguments())
        self.assertEqual(admitted["tracking"]["status"], "accepted")
        live = self.live(perception=True)
        await live.start()
        await self.until(lambda: live.perception_brief_bridge._binding is not None
            and any(q.question == RAW_REQUEST for q in live.perception_brief_bridge.composition.brief.questions),
            detail=lambda: (live.episode_context.worker.last_error, live.perception_runtime.worker.last_error))
        return admitted, live, live.perception_brief_bridge.composition.brief, live.perception_brief_bridge._binding

    async def test_bare_transcript_does_not_admit_without_model_tool_selection(self):
        live = self.live(perception=True)
        await live.start()
        await self.until(lambda: live.service.snapshot().readiness == "ready"
            and live.perception_brief_bridge.composition is not None)
        self.assertEqual(live.perception_brief_bridge.composition.brief.questions, ())
        self.assertFalse(self.db.rules)

    async def test_accepted_goal_reaches_source_bound_brief_and_real_prompt_without_visual_claim(self):
        admitted, live, brief, binding = await self.admitted_brief()
        question, = brief.questions
        self.assertEqual(str(question.check_ref.source_id), admitted["check"]["id"])
        self.assertTrue(any(edge[1] == binding.payload_root_ids[0] and edge[2] == self.source.request_event_id
                            for edge in self.db.edges))
        async with live.perception_runtime.binder.guard(brief, binding):
            pass
        request = FastPerceptionRequest(owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            window_id=uuid4(), frames=(FastPerceptionFrame(uuid4(), 1, self.now, "glasses", "image/jpeg", b"offline"),),
            previous_working_state={}, existing_entity_ids=frozenset(), recent_audio_transcripts=(),
            behavior_instructions="Observe only.", active_reminders=(), active_focus=None,
            recent_interrupts=(), requested_at=self.now)
        request = live.perception_brief_bridge.attach(request)
        self.assertIsNotNone(request.perception_brief)
        request = seal_live_brief_input(request, ObservationInputSnapshot(self.scope.owner_id,
            self.scope.session_id, uuid4(), digest(request_document(request))))
        with patch("app.perception.live_brief._utcnow", return_value=self.now):
            prompt = build_fast_perception_prompt(request, references=PerceptionReferences(request))
        context = json.loads(prompt.split("Context JSON:\n```json\n", 1)[1].split("\n```", 1)[0])
        assigned = context["perception_assignment"]["questions"][0]
        self.assertEqual(assigned["question"], RAW_REQUEST)
        self.assertEqual(assigned["request_grounding"]["user_wording"], RAW_REQUEST)
        self.assertEqual(assigned["request_grounding"]["status"], "unresolved")
        self.assertEqual(admitted["model_interpretation"]["question"], QUESTION)
        self.assertIsNone(context["active_focus"])
        self.assertEqual(context["active_reminders"], [])
        self.assertFalse(admitted["location_fact_recorded_by_this_call"])
        self.assertIsNone(admitted["check"]["resolution"])
        self.assertFalse(any(entry.kind == "observation" for entry in live.service.snapshot().entries))
        self.assertFalse(self.db.reminders)

    async def test_replaced_question_fences_old_assignment_and_excludes_old_goal(self):
        admitted, live, brief, binding = await self.admitted_brief()
        replacement = "Where is the newly specified synthetic token visible? Retain uncertainty."
        replacement_request = "Track the token instead, not the earlier object."
        source = self.db.event(self.scope.owner_id, self.scope.session_id, at=self.now,
            content={"text": replacement_request})
        changed = await self.invoke("change_episode_check", self.command(admitted, "revise",
            check={"question": replacement, "object_association": "synthetic token"}), source=source)
        self.assertEqual(changed["status"], "success")
        with self.assertRaises(SourceError):
            async with live.perception_runtime.binder.guard(brief, binding):
                self.fail("Replaced assignment accepted")
        await self.until(lambda: live.perception_brief_bridge.composition is not None
            and [q.question for q in live.perception_brief_bridge.composition.brief.questions] == [replacement_request])
        self.assertEqual(changed["model_interpretation"]["question"], replacement)
        self.assertEqual(len(self.db.occurrences), 1)
        self.assertFalse(self.db.reminders)

    async def test_cancel_fences_inflight_assignment_and_removes_question_without_completion(self):
        admitted, live, brief, binding = await self.admitted_brief()
        source = self.db.event(self.scope.owner_id, self.scope.session_id, at=self.now,
            content={"text": "Stop tracking that observation."})
        changed = await self.invoke("change_episode_check", self.command(admitted, "cancel"), source=source)
        self.assertEqual(changed["check"]["lifecycle"], "dismissed")
        self.assertIsNone(changed["check"]["resolution"])
        with self.assertRaises(SourceError):
            async with live.perception_runtime.binder.guard(brief, binding):
                self.fail("Cancelled assignment accepted")
        await self.until(lambda: live.perception_brief_bridge.composition is not None
            and live.perception_brief_bridge.composition.brief.questions == ())

    async def test_unrelated_revision_and_user_event_preserve_original_assignment_authority(self):
        _, live, brief, binding = await self.admitted_brief()
        self.db.event(self.scope.owner_id, self.scope.session_id, at=self.now,
            content={"text": "An unrelated conversational turn."})
        snapshot = live.service.snapshot()
        newer = snapshot.model_copy(update={"cursor": snapshot.cursor.model_copy(
            update={"revision": snapshot.cursor.revision + 7})})
        await live.perception_runtime._apply((newer, (), None))
        current = live.perception_brief_bridge.composition.brief
        self.assertEqual(current.id, brief.id)
        self.assertEqual(current.questions, brief.questions)
        async with live.perception_runtime.binder.guard(brief, binding):
            pass

    async def test_original_request_withdrawal_still_fences_the_assignment(self):
        _, live, brief, binding = await self.admitted_brief()
        self.db.events[self.source.request_event_id].update(availability="deleted", content=None)
        with self.assertRaises(SourceError):
            async with live.perception_runtime.binder.guard(brief, binding):
                self.fail("Removed request accepted")


if __name__ == "__main__":
    unittest.main()
