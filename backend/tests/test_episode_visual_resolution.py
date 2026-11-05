"""Offline owner/model-interface/receipt tests; NOT model or appliance qualification."""
import asyncio
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.live_state.perception import CheckResolution
from app.perception.live_brief import seal_live_brief_input
from app.knowledge.observation_inputs import request_document
from app.reminders.episode_checks import EpisodeCheckOwner
from app.reminders.episode_models import EpisodeCheckDraft, VisualResolutionPolicy
from app.reminders.visual_resolution import DECISION_NAMESPACE, EpisodeVisualResolver
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEventBus, DomainEventName
from app.tasks.models import digest
import tests.test_perception_evidence_consumer as fixtures


class ResolutionStore(fixtures.EvidenceStore):
    async def fetchrow(self, query, *args):
        q = " ".join(query.lower().split())
        if q.startswith("select e.*,p.availability") and args[1] in self.context_rows:
            self.sql(query)
            record = self.record(args[1])
            if (record.owner_id != args[0] or record.revision != args[2] or not record.is_current
                    or not self.available(args[0], record.payload_id)):
                return None
            return record.model_dump()
        return await super().fetchrow(query, *args)


class EpisodeVisualResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.policy = VisualResolutionPolicy(authorization="explicit_user_request",
            condition="The status display reads READY and the arm is inside the marked zone.")
        await self.arrange(self.policy)

    async def arrange(self, policy):
        self.f = fixtures.PerceptionEvidenceConsumerTests(
            "test_actual_owner_guard_and_ingress_preserve_exact_provenance_without_settlement")
        original_admit = EpisodeCheckOwner.admit
        async def configure(owner, scope, source, request):
            request = request.model_copy(update={"check":request.check.model_copy(update={"visual_resolution":policy})})
            return await original_admit(owner, scope, source, request)
        with patch.object(fixtures, "EvidenceStore", ResolutionStore), patch.object(EpisodeCheckOwner, "admit", configure):
            await self.f.asyncSetUp()
        f = self.f
        self.connection_id = uuid4()
        self.active_connection = self.connection_id
        f.request = replace(f.request, camera_visual_session_id=f.stream, camera_connection_id=self.connection_id,
                            perception_brief_input_digest=None)
        document = request_document(f.request)
        f.capture = replace(f.capture, request_digest=digest(document))
        f.db.copies[f.input_root]["content"] = {
            "format":"jarvis.visual_input.v1", "request_digest":f.capture.request_digest, "request":document}
        f.request = seal_live_brief_input(f.request, f.capture)
        self.camera = lambda request, stream:(request.camera_connection_id == self.active_connection
            and request.camera_visual_session_id == stream == f.stream)
        f.consumer.camera_is_current = self.camera
        f.finding = f.finding.model_copy(update={"literal_finding":"The display reads READY; the arm is inside the marked zone.",
                                                "confidence":.98})
        f.output = f.output.model_copy(update={"perception_findings":[f.finding]})
        self.evidence = await f.consume()
        self.assertEqual(self.evidence.status, "recorded", self.evidence.reason)
        self.answer = dict(verdict="satisfied", basis="condition_observed",
            observed_condition="The status is READY and the arm is visibly inside the outlined zone.",
            confidence=.99, uncertainties=[], supporting_frame_ids=["frame:0", "frame:1"])
        self.during_model = None
        self.limiter = PriorityAsyncLimiter(1)
        async def generate(**kwargs):
            self.assertFalse(f.db.is_in_transaction(), "Provider must never run under database locks")
            self.assertEqual(self.limiter.active_count, 1)
            self.assertEqual(len(kwargs["images"]), 2)
            if self.during_model is not None:
                await self.during_model()
            return json.dumps(self.answer)
        self.model = SimpleNamespace(provider="offline", model="predicate-fixture",
                                     generate_temporal_json=AsyncMock(side_effect=generate))
        self.bus, self.events = DomainEventBus(), []
        async def changed(event):
            self.assertFalse(f.db.is_in_transaction(), "Projection wake must follow commit")
            self.events.append(event)
        self.bus.subscribe(DomainEventName.MEMORY_ITEM_CHANGED, changed)
        self.resolver = EpisodeVisualResolver(owner=f.owner, canonical=f.canonical,
            model=self.model, limiter=self.limiter, bus=self.bus, enabled=True)

    async def asyncTearDown(self):
        await self.resolver.close()

    async def evaluate(self, **changes):
        f = self.f
        return await self.resolver.evaluate(**(dict(request=f.request, evidence=self.evidence,
            question_id=f.question.id, effect_guard=f.guard, camera_is_current=self.camera) | changes))

    async def reset(self, policy=None):
        await self.resolver.close()
        await self.arrange(self.policy if policy is None else policy)

    def occurrence(self):
        return self.f.db.occurrences[self.f.check.check.id]

    def assert_unresolved(self):
        row = self.occurrence()
        self.assertNotEqual(row["check_lifecycle"], "resolved")
        self.assertIsNone(row["check_resolution"])
        self.assertNotEqual(row["report_state"], "user_reported_done")
        self.assertEqual(self.events, [])

    async def test_real_predicate_call_and_owner_commit_preserve_user_report_semantics(self):
        previous_requests = dict(self.f.db.receipts)
        result = await self.evaluate()
        self.assertEqual(result.status, "resolved", result.reason)
        self.assertEqual(result.owner_result.check.lifecycle.value, "resolved")
        self.assertEqual(result.owner_result.check.revision, self.f.check.check.revision + 1)
        row = self.occurrence()
        self.assertEqual(row["status"], "closed")
        self.assertEqual(row["report_state"], "unknown")
        self.assertIsNone(row["report_event_id"])
        self.assertIsNone(row["report_revision"])
        self.assertIsNone(row["reported_at"])
        self.assertEqual(self.f.db.receipts, previous_requests, "No manufactured user operation receipt")
        resolution = CheckResolution.model_validate(row["check_resolution"])
        self.assertEqual(resolution.basis, "visual_observation")
        self.assertEqual(tuple(ref.captured_at for ref in resolution.source_refs if ref.frame_id is not None),
                         tuple(frame.captured_at for frame in self.f.request.frames))
        self.assertEqual(result.decision_receipt.source_namespace, DECISION_NAMESPACE)
        self.assertEqual(result.decision_receipt.trust, "derived")
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0].payload.memory_item_id, self.f.check.check.id)
        self.model.generate_temporal_json.assert_awaited_once()

    async def test_repeated_concurrent_candidates_resolve_once_and_replay_actual_receipt(self):
        first, second = await asyncio.gather(self.evaluate(), self.evaluate())
        self.assertEqual((first.status, second.status), ("resolved", "replayed"), (first.reason, second.reason))
        self.assertEqual(first.decision_receipt.id, second.decision_receipt.id)
        self.model.generate_temporal_json.assert_awaited_once()
        self.assertEqual(len(self.events), 1)
        self.assertEqual(len(self.f.db.context_rows), 2)  # analyzer evidence + owner decision

    async def test_postgres_json_text_resolution_replays_without_reinterpreting_or_duplicate_effect(self):
        first = await self.evaluate()
        self.assertEqual(first.status, "resolved", first.reason)
        # asyncpg can return JSON/JSONB as text, as handled by existing owner reads.
        self.occurrence()["check_resolution"] = json.dumps(self.occurrence()["check_resolution"])
        again = await self.evaluate()
        self.assertEqual(again.status, "replayed", again.reason)
        self.assertEqual(first.decision_receipt.id, again.decision_receipt.id)
        self.model.generate_temporal_json.assert_awaited_once()

    async def test_visibility_alone_uncertainty_low_confidence_and_negative_are_not_resolution(self):
        for changes in ({"basis":"visibility_only"}, {"uncertainties":["One marking cannot be read."]},
                {"confidence":.8}, {"verdict":"uncertain"}, {"verdict":"not_satisfied"},
                {"supporting_frame_ids":[]}):
            with self.subTest(changes=changes):
                await self.reset()
                self.answer.update(changes)
                result = await self.evaluate()
                self.assertEqual(result.status, "observed_without_resolution", result.reason)
                self.assert_unresolved()
                self.assertEqual(result.decision_receipt.content["decision"]["uncertainties"], self.answer["uncertainties"])
                repeat = await self.evaluate()
                self.assertEqual(repeat.status, "observed_without_resolution", repeat.reason)
                self.model.generate_temporal_json.assert_awaited_once()

    async def test_unclear_or_not_visible_findings_do_not_admit_an_evaluator_call(self):
        for visibility in ("unclear", "not_visible"):
            with self.subTest(visibility=visibility):
                await self.reset()
                finding = self.f.finding.model_copy(update={"visibility":visibility})
                evidence = self.evidence.evidence.model_copy(update={"findings":(finding,)})
                # Arrange the synthetic canonical evidence before evaluation.
                content = dict(self.evidence.record.content, evidence=evidence.model_dump(mode="json"))
                self.f.db.copies[self.evidence.record.payload_id]["content"] = content
                record = self.f.db.record(self.evidence.record.id)
                self.evidence = replace(self.evidence, evidence=evidence, record=record)
                self.assertEqual((await self.evaluate()).status, "not_eligible")
                self.model.generate_temporal_json.assert_not_awaited()
                self.assert_unresolved()

    async def test_missing_explicit_policy_preserves_user_report_only_and_request_fingerprints(self):
        await self.resolver.close()
        await self.arrange(None)
        result = await self.evaluate()
        self.assertEqual(result.status, "not_eligible", result.reason)
        self.model.generate_temporal_json.assert_not_awaited()
        self.assertNotIn("visual_resolution", EpisodeCheckDraft(question="A normal check?").model_dump(mode="json"))
        self.assertNotIn("visual_resolution", self.occurrence()["check_payload"])
        self.assert_unresolved()

    async def test_natural_language_policy_and_actual_images_reach_existing_model_interface(self):
        result = await self.evaluate()
        self.assertEqual(result.status, "resolved", result.reason)
        call = self.model.generate_temporal_json.await_args
        self.assertIn(self.policy.condition, call.kwargs["prompt"])
        self.assertIn("visibility alone", call.kwargs["prompt"])
        self.assertEqual(tuple(image.image_bytes for image in call.kwargs["images"]),
                         tuple(frame.image_bytes for frame in self.f.request.frames))

    async def test_owner_cancel_snooze_rebind_and_policy_change_during_model_block_mutation(self):
        for operation, changes in (("cancel", {}), ("snooze", {"until":fixtures.NOW + timedelta(minutes=2)}),
                ("revise", {"check":EpisodeCheckDraft(question="A different episode?", visual_resolution=self.policy),
                            "rebind_episode":True}),
                ("revise", {"check":EpisodeCheckDraft(question="Reconfigured?", visual_resolution=
                    self.policy.model_copy(update={"condition":"A different natural-language condition."}))})):
            with self.subTest(operation=operation, changes=tuple(changes)):
                await self.reset()
                async def mutate():
                    await self.f.change_check(operation, **changes)
                self.during_model = mutate
                result = await self.evaluate()
                self.assertEqual(result.status, "not_applied", result.reason)
                self.assert_unresolved()
                self.assertEqual(len(self.f.db.context_rows), 1)

    async def test_source_withdrawal_supersession_camera_reconnect_and_runtime_close_block_late_answer(self):
        for change in ("source", "supersede", "camera", "runtime"):
            with self.subTest(change=change):
                await self.reset()
                async def mutate():
                    if change == "source":
                        self.f.db.copies[self.evidence.record.payload_id]["content"] = None
                    elif change == "supersede":
                        self.f.db.events[self.f.admission.request_event_id]["current"] = False
                    elif change == "camera":
                        self.active_connection = uuid4()
                    else:
                        await self.resolver.close()
                self.during_model = mutate
                result = await self.evaluate()
                self.assertEqual(result.status, "not_applied", result.reason)
                self.assert_unresolved()

    async def test_evidence_age_is_rechecked_after_model_without_refreshing_capture_time(self):
        await self.reset(self.policy.model_copy(update={"max_evidence_age_seconds":3}))
        async def advance():
            self.f.now += timedelta(seconds=1)
        self.during_model = advance
        result = await self.evaluate()
        self.assertEqual(result.reason, "visual_resolution_evidence_expired_or_prior_episode")
        self.assert_unresolved()

    async def test_untouched_or_unrelated_context_does_not_invalidate_episode(self):
        async def unrelated():
            self.f.db.event(self.f.scope.owner_id, self.f.scope.session_id, at=self.f.now,
                            content={"text":"An unrelated task result."})
        self.during_model = unrelated
        result = await self.evaluate()
        self.assertEqual(result.status, "resolved", result.reason)

    async def test_hallucinated_frames_extra_commands_and_trusted_sensor_claims_are_rejected(self):
        for changes in ({"supporting_frame_ids":["frame:99"]}, {"supporting_frame_ids":["frame:0", "frame:0"]},
                        {"resolve":True}, {"basis":"trusted_source"}):
            with self.subTest(changes=changes):
                await self.reset()
                self.answer.update(changes)
                self.assertEqual((await self.evaluate()).status, "not_applied")
                self.assert_unresolved()
                self.assertEqual(len(self.f.db.context_rows), 1)

    async def test_changed_request_pixels_and_wrong_scope_never_reach_model(self):
        changed = replace(self.f.request, frames=(replace(self.f.request.frames[0], image_bytes=b"other"),
                                                  self.f.request.frames[1]))
        self.assertEqual((await self.evaluate(request=changed)).status, "not_applied")
        self.model.generate_temporal_json.assert_not_awaited()
        self.assert_unresolved()

    async def test_failed_commit_rolls_back_decision_owner_and_projection_event(self):
        async def fail_commit():
            self.f.db.fail_commit = True
        self.during_model = fail_commit
        result = await self.evaluate()
        self.assertEqual(result.status, "unavailable", result.reason)
        self.assert_unresolved()
        self.assertEqual(len(self.f.db.context_rows), 1)

    async def test_background_offer_is_bounded_and_cancellation_leaves_no_success_receipt(self):
        entered = asyncio.Event()
        async def wait():
            entered.set()
            await asyncio.Event().wait()
        self.during_model = wait
        args = dict(request=self.f.request, evidence=self.evidence,
                    effect_guard=self.f.guard, camera_is_current=self.camera)
        self.assertTrue(self.resolver.offer(**args))
        await asyncio.wait_for(entered.wait(), 1)
        self.assertFalse(self.resolver.offer(**args))
        await self.resolver.close()
        self.assert_unresolved()
        self.assertEqual(len(self.f.db.context_rows), 1)

    async def test_disabled_runtime_does_not_schedule_or_call_model(self):
        self.resolver.enabled = False
        self.assertEqual((await self.evaluate()).status, "disabled")
        self.assertFalse(self.resolver.offer(request=self.f.request, evidence=self.evidence,
            effect_guard=self.f.guard, camera_is_current=self.camera))
        self.model.generate_temporal_json.assert_not_awaited()
