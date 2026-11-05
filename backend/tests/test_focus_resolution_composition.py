"""Post-recording focus hook and real owner adapter with controlled storage/model.

The focus fixture seeds canonical recorded evidence; these tests do not replace
the separate actual camera-pipeline/binder or PostgreSQL qualification.
"""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.perception.evidence_consumer import PerceptionEvidenceConsumer
from app.perception.evidence_session import SessionPerceptionEvidence, create_session_perception_evidence
from app.perception.focus_resolution import FocusVisualResolver
from app.perception.frame_window import RecentVisualFrameWindow
from app.runtime.concurrency import PriorityAsyncLimiter
import test_focus_typed_evidence as focus_fixtures
import test_visual_resolution_composition as pipeline_fixtures


class FocusResolutionCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = focus_fixtures.FocusTypedEvidenceTests()
        await self.f.asyncSetUp()
        await self.f.assignment()
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()
        self.uncertain = False
        self.limiter = PriorityAsyncLimiter(1)
        async def generate(**kwargs):
            self.assertFalse(self.f.store.is_in_transaction())
            self.assertEqual(self.limiter.active_count, 1)
            self.assertTrue(kwargs["images"])
            self.entered.set()
            await self.release.wait()
            return json.dumps(dict(verdict="uncertain" if self.uncertain else "satisfied",
                basis="insufficient_evidence" if self.uncertain else "condition_observed",
                observed_condition="The requested target's distinguishing visible features match.",
                confidence=.99, uncertainties=["unclear match"] if self.uncertain else [],
                supporting_frame_ids=["frame:0"]))
        self.model = SimpleNamespace(generate_temporal_json=AsyncMock(side_effect=generate), close=AsyncMock())
        self.resolver = FocusVisualResolver(canonical=self.f.canonical, prospective=self.f.service,
            model=self.model, limiter=self.limiter, enabled=True)

    async def asyncTearDown(self):
        self.release.set()
        await self.resolver.close()
        self.f.doCleanups()

    def offer(self):
        return self.resolver.offer(request=self.f.request, evidence=self.f.evidence,
            effect_guard=self.f.guard, camera_is_current=lambda *args: self.f.camera)

    async def finish(self):
        await asyncio.wait_for(self.resolver._task, 2)
        return self.resolver.last_result

    async def test_existing_model_owner_and_attention_route_are_used_without_physical_completion(self):
        self.assertTrue(self.offer())
        result = await self.finish()
        self.assertEqual(result.status, "owner_processed", result.reason)
        row = self.f.store.rows[self.f.focus.id]
        self.assertIsNotNone(row["check_resolution"])
        self.assertIsNone(row["completed_at"])
        self.assertEqual(len(self.f.store.intents), 1)
        self.f.manager.speak_prospective_interrupt.assert_not_awaited()
        self.model.generate_temporal_json.assert_awaited_once()
        self.model.close.assert_not_awaited()

    async def test_slow_model_is_nonblocking_and_busy_work_has_no_backlog(self):
        self.release.clear()
        self.assertTrue(self.offer())
        await asyncio.wait_for(self.entered.wait(), 1)
        self.assertFalse(self.offer())
        self.assertFalse(self.resolver._task.done())
        self.assertFalse(self.f.store.intents)
        self.release.set()
        self.assertEqual((await self.finish()).status, "owner_processed")
        self.assertEqual(self.model.generate_temporal_json.await_count, 1)

    async def test_late_camera_change_prevents_owner_effect(self):
        self.release.clear()
        self.offer()
        await self.entered.wait()
        self.f.camera = False
        self.release.set()
        self.assertEqual((await self.finish()).status, "not_applied")
        self.assertFalse(self.f.store.intents)
        self.assertIsNone(self.f.store.rows[self.f.focus.id]["check_resolution"])

    async def test_source_withdrawal_during_limiter_wait_prevents_model_call(self):
        await self.limiter.acquire()
        try:
            self.offer()
            async with asyncio.timeout(1):
                while not self.limiter.waiting_count:
                    await asyncio.sleep(0.001)
            self.f.store.roots.remove(self.f.evidence.record.payload_id)
        finally:
            await self.limiter.release()
        self.assertEqual((await self.finish()).status, "not_applied")
        self.model.generate_temporal_json.assert_not_awaited()
        self.assertFalse(self.f.store.intents)

    async def test_model_budget_is_bounded_across_successive_observations(self):
        self.uncertain = True
        self.resolver.calls_per_minute = 1
        self.offer()
        self.assertEqual((await self.finish()).status, "observed_without_resolution")
        self.offer()
        result = await self.finish()
        self.assertEqual(result.reason, "focus_predicate_budget")
        self.assertEqual(self.model.generate_temporal_json.await_count, 1)
        self.assertFalse(self.f.store.intents)

    async def test_actual_session_hook_only_schedules_after_recording(self):
        entry = SessionPerceptionEvidence(canonical=self.f.canonical,
            publication=SimpleNamespace(owner_id=self.f.scope.owner_id),
            window=RecentVisualFrameWindow(max_frames_per_session=8, ttl_seconds=30),
            focus_resolver=self.resolver)
        entry.activate()
        entry.camera.is_current = lambda *args: self.f.camera
        self.release.clear()
        try:
            with patch.object(PerceptionEvidenceConsumer, "consume", AsyncMock(return_value=self.f.evidence)) as record:
                result = await entry.consumer.consume(request=self.f.request, output=None,
                    observation=None, analysis_input=None, effect_guard=self.f.guard)
            record.assert_awaited_once()
            self.assertIs(result, self.f.evidence)
            self.assertTrue(entry.last_focus_resolution_scheduled)
            await self.entered.wait()
            self.assertFalse(self.f.store.intents)
            self.release.set()
            self.assertEqual((await self.finish()).status, "owner_processed")
        finally:
            self.release.set()
            await entry.aclose()

    async def test_repeated_retirement_does_not_cancel_owned_cleanup_again(self):
        cleanup_entered = asyncio.Event()
        self.release.clear()
        async def generate(**kwargs):
            self.entered.set()
            try:
                await asyncio.Future()
            finally:
                cleanup_entered.set()
                await self.release.wait()
        self.model.generate_temporal_json.side_effect = generate
        self.offer()
        await self.entered.wait()
        self.resolver.retire()
        await cleanup_entered.wait()
        self.resolver.retire()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertFalse(self.resolver._task.done(), "Repeated retirement cancelled model cleanup")
        self.release.set()
        await self.resolver.close()
        self.assertFalse(self.f.store.intents)
        self.model.close.assert_not_awaited()


class FocusFactoryWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_pipeline_resources_are_borrowed_and_no_focus_work_is_invented(self):
        rig = pipeline_fixtures.VisualResolutionCompositionTests()
        await rig.asyncSetUp()
        entry = None
        try:
            service = SimpleNamespace(process_matches=AsyncMock())
            rig.rig.pipeline._prospective_service = service
            entry = create_session_perception_evidence(rig.live, rig.rig.pipeline, clock=lambda: rig.rig.f.now)
            self.assertIs(entry.focus_resolver.model, rig.model)
            self.assertIs(entry.focus_resolver.limiter, rig.rig.pipeline._model_limiter)
            self.assertIs(entry.focus_resolver.prospective, service)
            self.assertIs(entry.focus_resolver.canonical, rig.live.binder)
            self.assertFalse(entry.focus_resolver.offer(request=rig.rig.f.request,
                evidence=SimpleNamespace(status="not_recorded"), effect_guard=None, camera_is_current=None))
            rig.model.generate_temporal_json.assert_not_awaited()
        finally:
            if entry is not None:
                await entry.aclose()
            await rig.asyncTearDown()
