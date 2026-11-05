"""Minimal startup assignments are independent of background memory readiness."""
import asyncio
from datetime import UTC, datetime, timedelta
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.live_state.canonical import CanonicalContextBinder
from app.live_state.models import (
    GuardedPayloadBinding, LiveCapabilities, LiveSnapshot, SessionScope,
    SourceDependency, StateCursor,
)
from app.live_state.perception import (
    CheckLifecycle, EvidenceReference, ObservationalQuestion, PerceptionActivity,
    PriorEvidence, SituationalCheck,
)
from app.live_state.perception_runtime import CanonicalPerceptionBinder
from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest
from app.perception.live_brief import LivePerceptionBriefBridge
from test_session_perception_runtime import _CanonicalIO


class MinimalStartupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime.now(UTC)
        self.scope = SessionScope(owner_id="minimal-startup", session_id=uuid4(), area_id=uuid4())
        self.epoch = uuid4()
        self.snapshot = LiveSnapshot(scope=self.scope, cursor=StateCursor(epoch=self.epoch, revision=1),
            as_of=self.now, readiness="warming",
            capabilities=LiveCapabilities(durable_responsibilities="available"))
        self.bridge = self.make_bridge(allow_warming_minimal=True)
        self.bind = AsyncMock(side_effect=self.binding)
        self.episode = SourceDependency(source_namespace="episode", source_id=uuid4(), revision=2)
        self.focus = SourceDependency(source_namespace="visual_focus", source_id=uuid4(), revision=3)
        self.source = SourceDependency(source_namespace="context_event", source_id=uuid4(), revision=1)
        self.check = SituationalCheck(id=self.focus.source_id, revision=self.focus.revision,
            scope=self.scope, episode_or_object_ref=self.episode, origin="explicit_request",
            canonical_owner_ref=self.focus, question="Are the controls readable?",
            lifecycle=CheckLifecycle.TRACKING, relevance_now=True)

    def make_bridge(self, **options):
        return LivePerceptionBriefBridge(scope=self.scope, scope_epoch=self.epoch,
            enabled=options.pop("enabled", True), clock=lambda: self.now, **options)

    def binding(self, brief):
        return GuardedPayloadBinding(scope=brief.scope, payload_root_ids=(uuid4(),),
                                     dependencies=brief.dependencies)

    def request(self):
        return FastPerceptionRequest(owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            window_id=uuid4(), frames=(FastPerceptionFrame(uuid4(), 1, self.now,
                "glasses", "image/jpeg", b"offline"),),
            previous_working_state={}, existing_entity_ids=frozenset(), recent_audio_transcripts=(),
            behavior_instructions="", active_reminders=(), active_focus=None,
            recent_interrupts=(), requested_at=self.now)

    async def test_default_bridge_still_waits_for_ready(self):
        bridge = self.make_bridge()
        for readiness in ("warming", "degraded"):
            with self.subTest(readiness=readiness):
                self.assertIsNone(await bridge.refresh(
                    self.snapshot.model_copy(update={"readiness": readiness}), bind=self.bind))
                self.assertEqual(bridge.last_omission, "live_state_not_ready")
        self.bind.assert_not_awaited()

    async def test_warming_strips_all_unready_enrichment_without_authorizing_it(self):
        predicate = Mock(side_effect=AssertionError("Unready enrichment was authorized"))
        result = await self.bridge.refresh(self.snapshot, bind=self.bind,
            dependency_eligible=predicate, checks=(self.check,),
            explicit_questions=(ObservationalQuestion(id=uuid4(), question="Read the label.",
                                                       priority="explicit_inspection"),),
            activity=PerceptionActivity(label="cooking", confidence=0.9), intent_ref=self.source,
            episode_or_object_ref=self.episode, context_dependencies=(self.source,),
            prior_evidence=(PriorEvidence(reference=EvidenceReference(source=self.source,
                captured_at=self.now - timedelta(seconds=1), episode_or_object_ref=self.episode),
                finding="A control appeared on.", confidence=0.5, uncertainty="Label unclear."),))
        brief = result.brief
        self.assertEqual(brief.dependencies, ())
        self.assertEqual(brief.questions, ())
        self.assertEqual(brief.prior_evidence, ())
        self.assertIsNone(brief.activity)
        self.assertIsNone(brief.intent_ref)
        self.assertIsNone(brief.episode_or_object_ref)
        self.assertIn("context is still loading", brief.scene_change_cues[0])
        self.assertEqual(self.snapshot.readiness, "warming")
        predicate.assert_not_called()
        self.bind.assert_awaited_once_with(brief)
        self.assertIs(self.bridge.attach(self.request()).perception_brief, brief)
        self.assertEqual(self.check.lifecycle, CheckLifecycle.TRACKING)

    async def test_minimal_brief_uses_real_canonical_binder_without_memory_reads(self):
        io = _CanonicalIO(self.scope)
        canonical = CanonicalContextBinder(io.sources, self.scope, clock=lambda: self.now)
        io.patch_io(canonical)
        binder = CanonicalPerceptionBinder(canonical)
        result = await self.bridge.refresh(self.snapshot, bind=binder.bind)
        request = self.bridge.attach(self.request())
        self.assertEqual(request.perception_brief, result.brief)
        self.assertEqual(request.perception_brief_binding.dependencies, ())
        canonical.sources._read_many.assert_not_awaited()
        canonical.tasks.payload.assert_awaited_once()
        content = canonical.tasks.payload.call_args.args[2]
        self.assertEqual(content["brief"], result.brief.model_dump(mode="json"))
        self.assertEqual(canonical.tasks.payload.call_args.kwargs["source_payload_ids"], ())

    async def test_ready_replaces_loading_caveat_even_without_new_enrichment(self):
        first = await self.bridge.refresh(self.snapshot, bind=self.bind)
        ready = self.snapshot.model_copy(update={"readiness": "ready"})
        second = await self.bridge.refresh(ready, bind=self.bind)
        self.assertFalse(second.reused)
        self.assertEqual(first.brief.id, second.brief.id)
        self.assertGreater(second.brief.version, first.brief.version)
        self.assertNotIn("still loading", " ".join(second.brief.scene_change_cues))
        self.assertEqual(second.brief.dependencies, ())

    async def test_ready_restores_enrichment_but_revocation_still_blocks_attachment(self):
        await self.bridge.refresh(self.snapshot, bind=self.bind)
        allowed = {self.focus, self.episode}
        ready = self.snapshot.model_copy(update={"readiness": "ready"})
        result = await self.bridge.refresh(ready, bind=self.bind, checks=(self.check,),
            dependency_eligible=lambda source: source in allowed)
        self.assertEqual(result.brief.questions[0].check_ref, self.focus)
        self.assertIsNotNone(self.bridge.attach(self.request()).perception_brief)
        allowed.remove(self.focus)
        self.assertIsNone(self.bridge.attach(self.request()).perception_brief)
        self.assertEqual(self.bridge.last_omission, "assignment_dependency_changed")

    async def test_degraded_clears_minimal_binding_instead_of_masking_failure(self):
        await self.bridge.refresh(self.snapshot, bind=self.bind)
        self.bind.reset_mock()
        self.assertIsNone(await self.bridge.refresh(
            self.snapshot.model_copy(update={"readiness": "degraded"}), bind=self.bind))
        self.assertIsNone(self.bridge.attach(self.request()).perception_brief)
        self.bind.assert_not_awaited()

    async def test_disabled_bridge_remains_inert_and_frame_attachment_has_no_io(self):
        disabled = self.make_bridge(enabled=False, allow_warming_minimal=True)
        self.assertIsNone(await disabled.refresh(self.snapshot, bind=self.bind))
        self.bind.assert_not_awaited()
        request = self.request()
        self.assertIs(disabled.attach(request), request)
        await self.bridge.refresh(self.snapshot, bind=self.bind)
        with patch("app.perception.live_brief.compose_perception_brief", side_effect=AssertionError("frame composition")):
            self.assertIsNotNone(self.bridge.attach(request).perception_brief)
            self.assertIsNotNone(self.bridge.attach(request).perception_brief)
        self.bind.assert_awaited_once()

    async def test_late_warming_bind_cannot_replace_ready_or_closed_epoch(self):
        for close in (False, True):
            with self.subTest(close=close):
                bridge = self.make_bridge(allow_warming_minimal=True)
                entered, release = asyncio.Event(), asyncio.Event()

                async def slow(brief):
                    entered.set()
                    await release.wait()
                    return self.binding(brief)

                pending = asyncio.create_task(bridge.refresh(self.snapshot, bind=slow))
                try:
                    await asyncio.wait_for(entered.wait(), timeout=1)
                    ready = await bridge.refresh(self.snapshot.model_copy(update={"readiness": "ready"}),
                                                 bind=self.bind)
                    if close:
                        bridge.close()
                    release.set()
                    await pending
                    attached = bridge.attach(self.request())
                    if close:
                        self.assertIsNone(attached.perception_brief)
                    else:
                        self.assertEqual(attached.perception_brief, ready.brief)
                finally:
                    release.set()
                    await asyncio.gather(pending, return_exceptions=True)
