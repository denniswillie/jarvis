"""Offline refresh interleavings; no SQL, provider or native causality claim."""
import asyncio
from datetime import timedelta
import unittest
from uuid import uuid4

from app.knowledge.models import SourceError
from app.live_state.models import GuardedPayloadBinding
from app.live_state.perception import CheckLifecycle, PerceptionActivity
import test_perception_live_brief as brief_tests
import test_session_perception_runtime as runtime_tests


class BriefRefreshLifecycleTests(unittest.IsolatedAsyncioTestCase):
    refresh = brief_tests.LiveBriefTests.refresh
    request = brief_tests.LiveBriefTests.request

    def setUp(self):
        brief_tests.LiveBriefTests.setUp(self)
        self.addCleanup(self.bridge.close)

    async def held_refresh(self, *, failure=None, **changes):
        entered, release = asyncio.Event(), asyncio.Event()

        async def bind(brief):
            entered.set()
            await release.wait()
            if failure is not None:
                raise failure
            return GuardedPayloadBinding(scope=brief.scope, payload_root_ids=(uuid4(),),
                                         dependencies=brief.dependencies)

        task = asyncio.create_task(self.refresh(bind=bind, **changes))

        async def join():
            release.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)

        self.addAsyncCleanup(join)
        await asyncio.wait_for(entered.wait(), timeout=1)
        return task, release

    async def test_one_two_three_unchanged_revalidations_keep_exact_assignment(self):
        first = await self.refresh()
        for repetition in (1, 2, 3):
            with self.subTest(repetition=repetition):
                old_binding = self.bridge._binding
                self.now += timedelta(seconds=1)
                self.snapshot = self.snapshot.model_copy(update={"cursor":
                    self.snapshot.cursor.model_copy(update={"revision":self.snapshot.cursor.revision + 100})})
                task, release = await self.held_refresh()
                attached = self.bridge.attach(self.request())
                self.assertIs(attached.perception_brief, first.brief)
                self.assertIs(attached.perception_brief_binding, old_binding)
                self.assertEqual(attached.perception_brief.created_at, first.brief.created_at)
                self.assertEqual(attached.perception_brief.expires_at, first.brief.expires_at)
                self.assertEqual(attached.perception_brief.questions[0].check_ref.revision, self.check.revision)
                release.set()
                composition = await asyncio.wait_for(task, timeout=1)
                self.assertTrue(composition.reused)
        self.model.generate_temporal_json.assert_not_awaited()

    async def test_failed_revalidation_withdraws_without_resolving_unfinished_check(self):
        await self.refresh()
        task, release = await self.held_refresh(failure=SourceError("knowledge_source_unavailable", "controlled withdrawal"))
        self.assertIsNotNone(self.bridge.attach(self.request()).perception_brief)
        release.set()
        with self.assertRaises(SourceError):
            await asyncio.wait_for(task, timeout=1)
        self.assertIsNone(self.bridge.attach(self.request()).perception_brief)
        self.assertEqual(self.check.lifecycle, CheckLifecycle.TRACKING)
        self.assertIsNone(self.check.resolution)

    async def test_cancelled_revalidation_is_joined_and_withdrawn(self):
        await self.refresh()
        task, _ = await self.held_refresh()
        self.assertIsNotNone(self.bridge.attach(self.request()).perception_brief)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(task.done())
        self.assertIsNone(self.bridge.attach(self.request()).perception_brief)
        self.assertIsNone(self.check.resolution)

    async def test_revocation_during_unchanged_revalidation_is_never_attachable(self):
        await self.refresh()
        task, release = await self.held_refresh()
        self.allowed.remove(self.focus)
        self.assertIsNone(self.bridge.attach(self.request()).perception_brief)
        self.assertEqual(self.bridge.last_omission, "assignment_dependency_changed")
        release.set()
        await asyncio.wait_for(task, timeout=1)
        self.assertIsNone(self.bridge._binding)
        self.assertEqual(self.bridge.last_omission, "assignment_expired_or_dependency_changed")

    async def test_goal_revision_replacement_does_not_keep_previous_binding(self):
        first = await self.refresh()
        self.focus = self.focus.model_copy(update={"revision":self.focus.revision + 1})
        self.check = self.check.model_copy(update={"revision":self.focus.revision, "canonical_owner_ref":self.focus})
        self.allowed = {self.episode, self.focus}
        task, release = await self.held_refresh()
        self.assertIsNone(self.bridge.attach(self.request()).perception_brief)
        release.set()
        newest = await asyncio.wait_for(task, timeout=1)
        self.assertGreater(newest.brief.version, first.brief.version)
        self.assertEqual(self.bridge.attach(self.request()).perception_brief.questions[0].check_ref.revision,
                         self.check.revision)

    async def test_expiry_renews_assignment_not_old_observational_truth_or_completion(self):
        first = await self.refresh()
        self.now = first.brief.expires_at
        self.assertIsNone(self.bridge.attach(self.request()).perception_brief)
        task, release = await self.held_refresh(prior_evidence=())
        self.assertIsNone(self.bridge.attach(self.request()).perception_brief)
        release.set()
        renewed = await asyncio.wait_for(task, timeout=1)
        self.assertFalse(renewed.reused)
        self.assertGreater(renewed.brief.version, first.brief.version)
        self.assertEqual(renewed.brief.created_at, self.now)
        self.assertEqual(renewed.brief.expires_at - renewed.brief.created_at, timedelta(seconds=30))
        self.assertEqual(renewed.brief.prior_evidence, ())
        self.assertEqual(renewed.brief.questions[0].check_ref.source_id, self.check.id)
        self.assertEqual(self.check.lifecycle, CheckLifecycle.TRACKING)
        self.assertIsNone(self.check.resolution)

    async def test_activity_change_recomposes_but_does_not_complete_unresolved_goal(self):
        first = await self.refresh()
        self.now += timedelta(seconds=1)
        activity = PerceptionActivity(label="moving to another workspace", confidence=0.6,
                                      uncertainty="Current activity is an interpretation.")
        task, release = await self.held_refresh(activity=activity)
        self.assertIsNone(self.bridge.attach(self.request()).perception_brief)
        release.set()
        changed = await asyncio.wait_for(task, timeout=1)
        self.assertGreater(changed.brief.version, first.brief.version)
        self.assertEqual(changed.brief.activity, activity)
        self.assertEqual(changed.brief.questions[0].check_ref.source_id, self.check.id)
        self.assertEqual(self.check.lifecycle, CheckLifecycle.TRACKING)
        self.assertIsNone(self.check.resolution)


class PeriodicBriefRefreshTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = runtime_tests.SessionPerceptionRuntimeTests.asyncSetUp
    asyncTearDown = runtime_tests.SessionPerceptionRuntimeTests.asyncTearDown
    until = runtime_tests.SessionPerceptionRuntimeTests.until
    ready = runtime_tests.SessionPerceptionRuntimeTests.ready
    request = runtime_tests.SessionPerceptionRuntimeTests.request

    async def test_actual_periodic_runtime_recomposes_expiry_without_completion_event(self):
        await self.ready()
        bridge = self.live.perception_brief_bridge
        await self.until(lambda: self.live.service.snapshot().readiness == "ready"
                         and bridge._binding is not None
                         and "still loading" not in " ".join(bridge.composition.brief.scene_change_cues))
        first = bridge.composition.brief
        before = self.live.perception_runtime.worker.refresh_count
        self.now = first.expires_at + timedelta(microseconds=1)
        self.assertIsNone(bridge.attach(self.request()).perception_brief)
        await self.until(lambda: self.live.perception_runtime.worker.refresh_count > before
                         and bridge._binding is not None and bridge.composition.brief.expires_at > self.now)
        renewed = bridge.attach(self.request()).perception_brief
        self.assertIsNotNone(renewed)
        self.assertGreater(renewed.version, first.version)
        self.assertEqual(renewed.expires_at - renewed.created_at, timedelta(seconds=30))
        self.live.sources.current_snapshot.assert_not_awaited()
        self.services.working_memory_service.get_session_working_memory.assert_not_awaited()
