"""Actual pipeline/camera/evidence/owner composition with controlled model and SQL.

This exercises shared resource borrowing and the nonblocking runtime hook, not
real model recognition or physical safety. No external provider is contacted.
"""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.knowledge.models import SourceError
from app.perception.evidence_session import create_session_perception_evidence
from app.perception.fast_analyzer import FastPerceptionAnalyzer
from app.reminders.episode_checks import EpisodeCheckOwner
from app.reminders.episode_models import VisualResolutionPolicy
from app.runtime.events import DomainEventBus, DomainEventName
import tests.test_perception_evidence_consumer as evidence_fixtures
from tests.test_episode_visual_resolution import ResolutionStore
from tests.test_perception_evidence_sessions import PerceptionEvidenceSessionPipelineTests


class VisualResolutionCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.rig = PerceptionEvidenceSessionPipelineTests()
        policy = VisualResolutionPolicy(authorization="explicit_user_request",
            condition="The status display reads READY and the arm is inside the marked zone.")
        original = EpisodeCheckOwner.admit
        async def configure(owner, scope, source, request):
            request = request.model_copy(update={"check": request.check.model_copy(
                update={"visual_resolution": policy})})
            return await original(owner, scope, source, request)
        with patch.object(evidence_fixtures, "EvidenceStore", ResolutionStore), \
             patch.object(EpisodeCheckOwner, "admit", configure):
            await self.rig.asyncSetUp()
        f = self.rig.f
        f.finding = f.finding.model_copy(update={"literal_finding":
            "The display reads READY and the arm is inside the marked zone.", "confidence": .98})
        f.output = f.output.model_copy(update={"perception_findings": [f.finding]})
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.release.set()
        async def generate(**kwargs):
            self.assertFalse(f.db.is_in_transaction())
            self.assertEqual(self.rig.pipeline._model_limiter.active_count, 1)
            self.assertEqual(len(kwargs["images"]), 2)
            self.entered.set()
            await self.release.wait()
            return json.dumps(dict(verdict="satisfied", basis="condition_observed",
                observed_condition="The status reads READY and the arm is visibly inside the zone.",
                confidence=.99, uncertainties=[], supporting_frame_ids=["frame:0", "frame:1"]))
        self.model = SimpleNamespace(provider="offline", model="predicate-fixture",
            generate_temporal_json=AsyncMock(side_effect=generate), close=AsyncMock())
        previous = self.rig.pipeline._fast_analyzer
        analyzer = FastPerceptionAnalyzer(self.model)
        # The scene result is a fixed provider fixture; the actual predicate
        # model interface is separately exercised by the owner resolver.
        analyzer.analyze = previous.analyze
        self.rig.pipeline._fast_analyzer = analyzer
        self.bus, self.events = DomainEventBus(), []
        async def changed(event):
            self.assertFalse(f.db.is_in_transaction())
            self.events.append(event)
        self.bus.subscribe(DomainEventName.MEMORY_ITEM_CHANGED, changed)
        self.rig.pipeline._domain_event_bus = self.bus
        self.live = SimpleNamespace(binder=f.canonical, episode_context=f.canonical.episode_context)
        self.entry = create_session_perception_evidence(self.live, self.rig.pipeline, clock=lambda: f.now)
        self.rig.pipeline.register_perception_evidence_session(self.entry)

    async def asyncTearDown(self):
        await self.rig.pipeline.close()
        await self.entry.aclose()

    async def finish(self):
        await asyncio.wait_for(self.entry.resolver._task, 2)
        return self.entry.resolver.last_results

    def occurrence(self):
        return self.rig.f.db.occurrences[self.rig.f.check.check.id]

    async def test_real_camera_pipeline_records_then_resolves_with_existing_owners(self):
        await self.rig.run_analysis()
        result, = await self.finish()
        self.assertEqual(result.status, "resolved", result.reason)
        self.assertEqual(self.rig.evidence_diagnostic()["status"], "recorded")
        self.assertEqual(self.occurrence()["check_lifecycle"], "resolved")
        self.assertEqual(self.occurrence()["report_state"], "unknown")
        self.assertIsNone(self.occurrence()["report_event_id"])
        self.assertIs(self.entry.resolver.owner, self.rig.f.owner)
        self.assertIs(self.entry.resolver.canonical, self.rig.f.canonical)
        self.assertIs(self.entry.resolver.model, self.model)
        self.assertIs(self.entry.resolver.limiter, self.rig.pipeline._model_limiter)
        self.assertIs(self.entry.resolver.bus, self.bus)
        self.assertEqual(len(self.events), 1)
        self.model.generate_temporal_json.assert_awaited_once()
        self.rig.pipeline._interrupt_service.maybe_interrupt.assert_not_awaited()

    async def test_slow_predicate_does_not_block_pipeline_or_recording(self):
        self.release.clear()
        await asyncio.wait_for(self.rig.run_analysis(), 1)
        await asyncio.wait_for(self.entered.wait(), 1)
        self.assertTrue(self.entry.last_resolution_scheduled)
        self.assertEqual(self.rig.evidence_diagnostic()["status"], "recorded")
        self.assertFalse(self.entry.resolver._task.done())
        self.assertNotEqual(self.occurrence()["check_lifecycle"], "resolved")
        self.assertEqual(self.events, [])
        self.release.set()
        result, = await self.finish()
        self.assertEqual(result.status, "resolved", result.reason)

    async def test_same_id_camera_reconnect_blocks_late_owner_mutation(self):
        self.release.clear()
        await self.rig.run_analysis()
        await asyncio.wait_for(self.entered.wait(), 1)
        f = self.rig.f
        replacement = self.rig.window.begin_stream(f.scope.owner_id, f.scope.session_id, f.stream)
        self.assertNotEqual(replacement.connection_id, self.rig.lease.connection_id)
        self.release.set()
        result, = await self.finish()
        self.assertEqual(result.status, "not_applied")
        self.assertNotEqual(self.occurrence()["check_lifecycle"], "resolved")
        self.assertEqual(self.events, [])

    async def test_sync_unregister_retires_and_async_close_joins_pending_predicate(self):
        self.release.clear()
        await self.rig.run_analysis()
        await asyncio.wait_for(self.entered.wait(), 1)
        self.rig.pipeline.unregister_perception_evidence_session(self.entry)
        self.assertFalse(self.entry.camera.is_current(self.rig.analyzed[0], self.rig.f.stream))
        await asyncio.wait_for(self.entry.aclose(), 1)
        self.assertTrue(self.entry.resolver._task.done())
        self.assertNotEqual(self.occurrence()["check_lifecycle"], "resolved")
        self.model.close.assert_not_awaited()

    async def test_pipeline_shutdown_joins_predicate_but_keeps_borrowed_model_open(self):
        self.release.clear()
        await self.rig.run_analysis()
        await asyncio.wait_for(self.entered.wait(), 1)
        await asyncio.wait_for(self.rig.pipeline.close(), 1)
        self.assertTrue(self.entry.resolver._task.done())
        self.assertNotEqual(self.occurrence()["check_lifecycle"], "resolved")
        self.model.close.assert_not_awaited()

    async def test_missing_canonical_owner_never_creates_resolution_authority(self):
        def unavailable():
            raise SourceError("owner_unavailable", "No initialized owner")
        self.live.episode_context = SimpleNamespace(require_owner=unavailable)
        entry = create_session_perception_evidence(self.live, self.rig.pipeline, clock=lambda: self.rig.f.now)
        self.assertIsNotNone(entry)
        self.assertIsNone(entry.resolver)
        await entry.aclose()
        self.model.generate_temporal_json.assert_not_awaited()

    async def test_existing_pipeline_without_temporal_model_keeps_recording_only(self):
        self.rig.pipeline._fast_analyzer = SimpleNamespace(analyze=AsyncMock())
        entry = create_session_perception_evidence(self.live, self.rig.pipeline, clock=lambda: self.rig.f.now)
        self.assertIsNotNone(entry)
        self.assertIsNone(entry.resolver)
        await entry.aclose()


if __name__ == "__main__":
    unittest.main()
