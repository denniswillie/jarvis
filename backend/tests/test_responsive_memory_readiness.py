"""Controlled bootstrap regressions; not native voice or Hindsight acceptance.

Actual route/control, canonical state, minimal briefs and the frame pipeline run
against the joined fixture's bounded SQL/model/wire endpoints. Barriers control
when work resumes; no readiness flag, selected memory or final answer is seeded.
"""
import asyncio
from datetime import UTC, datetime
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.live_state.models import SessionScope
from app.live_state.producers import CanonicalStateProducer
from app.live_state.service import LiveStateService
import responsive_joined_enrichment_fixtures as fixtures


class MemoryReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_idle_wait_rechecks_work_enqueued_after_its_notification(self):
        clock = lambda: datetime.now(UTC)
        service = LiveStateService(SessionScope(owner_id="idle-race", session_id=uuid4()), clock=clock)
        await service.start()
        self.addAsyncCleanup(service.close)
        producer = CanonicalStateProducer(service, producer_id="canonical-test", clock=clock)
        producer.start()
        self.assertTrue(producer.retract("first", "superseded").queued)
        original_wait = service._idle.wait
        requeued = False

        async def stale_notification():
            nonlocal requeued
            await original_wait()
            if not requeued:
                requeued = True
                self.assertTrue(producer.retract("second", "superseded").queued)
                self.assertFalse(service._idle.is_set())

        with patch.object(service._idle, "wait", stale_notification):
            await asyncio.wait_for(service.wait_idle(), 1)
        self.assertTrue(requeued)
        cursor = next(item for item in service.snapshot().producer_cursors
                      if item.producer_id == producer.producer_id)
        self.assertEqual(cursor.aggregate.revision, producer._sequence)
        self.assertEqual(producer._sequence, 2)
        # Exercise the real trusted readiness API only after both reductions.
        service.mark_ready()
        self.assertEqual(service.snapshot().readiness, "ready")
        self.assertFalse(service.needs_resync)

    async def test_monitor_survives_one_two_three_real_requeues_at_idle_boundary(self):
        original_factory = fixtures.create_session_live_state
        for count in (1, 2, 3):
            with self.subTest(requeues=count):
                f = fixtures.JoinedEnrichmentFixture(self)
                injected = []

                def factory(*args, **kwargs):
                    live = original_factory(*args, **kwargs)
                    original_wait = live.service._idle.wait

                    async def boundary():
                        await original_wait()
                        if asyncio.current_task() is live._monitor and len(injected) < count:
                            receipt = live.producer.retract("bootstrap-probe", "superseded")
                            self.assertTrue(receipt.queued)
                            injected.append(receipt)
                            self.assertFalse(live.service._idle.is_set())

                    live.service._idle.wait = boundary
                    return live

                try:
                    with patch.object(fixtures, "create_session_live_state", factory):
                        await f.setup()
                    self.assertEqual(len(injected), count)
                    self.assertEqual(f.live.service.snapshot().readiness, "ready")
                    self.assertFalse(f.live._monitor.done())
                    self.assertFalse(f.live.requires_reconstruction)
                    self.assertFalse(f.attachment._subscription._needs_resync)
                    self.assertIsNone(f.attachment.failure)
                    self.assertTrue(f.route.ready)
                    self.assertEqual(len(f.calls), 1)
                    self.assertFalse(f.release_a.is_set())
                finally:
                    await f.close()

    async def test_warming_fallback_handles_voice_and_frame_before_selected_history(self):
        f = fixtures.JoinedEnrichmentFixture(self)
        canonical_entered, canonical_release = asyncio.Event(), asyncio.Event()
        original_factory = fixtures.create_session_live_state

        def factory(*args, **kwargs):
            live = original_factory(*args, **kwargs)
            original_read = live.worker.read

            async def held_read():
                canonical_entered.set()
                await canonical_release.wait()
                return await original_read()

            live.worker.read = held_read
            return live

        try:
            with patch.object(fixtures, "create_session_live_state", factory):
                await f.setup(wait_for_retrieval=False)
            self.assertTrue(canonical_entered.is_set())
            self.assertFalse(canonical_release.is_set())
            self.assertEqual(f.live.service.snapshot().readiness, "warming")
            self.assertFalse(f.live._healthy)
            self.assertFalse(f.live._monitor.done())
            self.assertEqual(f.calls, [])
            self.assertTrue(f.route.ready)
            self.assertIsNotNone(f.live.perception_brief_bridge._binding)
            self.assertIsNone(f.route.request.initial_context.guard_binding)

            await f.ask("warming-fallback")
            await f.until(lambda: bool(f.session._active_responses))
            await f.session._handle_client_event({"type": "response.cancel"})
            await f.until(lambda: not f.session._active_responses)
            frame = await f.frame(1)
            self.assertFalse(f.lane.harness.visual_repository.failed)
            self.assertEqual(len(f.db.observations), 1)
            self.assertIn(frame.id, f.db.frames)
            self.assertEqual(f.lane.requests[-1].camera_connection_id, f.lease.connection_id)
            self.assertIsNotNone(f.lane.requests[-1].perception_brief_input_digest)
            self.assertFalse(canonical_release.is_set())
            self.assertEqual(f.calls, [])
            self.assertTrue(f.route.ready)

            canonical_release.set()
            await asyncio.wait_for(f.entered_a.wait(), 2)
            self.assertEqual(f.live.service.snapshot().readiness, "ready")
            self.assertFalse(f.release_a.is_set())
            self.assertEqual(len(f.calls), 1)
            self.assertTrue(f.route.ready)
            self.assertFalse(f.live.requires_reconstruction)
            self.assertIsNone(f.attachment.failure)
        finally:
            canonical_release.set()
            await f.close()
