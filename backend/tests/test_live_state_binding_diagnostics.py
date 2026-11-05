"""Observe real publisher/listener boundaries with offline, controlled binding.

No SQL/provider timing, cancellation optimization or source-authority inference.
"""
import asyncio
from datetime import timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

from app.knowledge.models import SourceError
from app.realtime.live_state_bridge import _BINDING_SAMPLE_LIMIT
import test_live_state_voice_attachment as fixtures


class BindingDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        fixtures.ProjectionAttachmentTests.setUp(self)
        self.live.close = AsyncMock()
        self.diagnostic = self.enterContext(patch(
            "app.realtime.live_state_bridge._BINDING_LOGGER.warning"))

    def records(self, stage=None):
        rows = [json.loads(call.args[1]) for call in self.diagnostic.call_args_list]
        return [row for row in rows if stage is None or row["stage"] == stage]

    async def join(self, task):
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def held(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def bind(selection):
            entered.set()
            await release.wait()
            return fixtures.BINDING

        self.live.bind.side_effect = bind
        task = asyncio.create_task(self.bridge.publish_latest())
        self.addAsyncCleanup(self.join, task)
        await asyncio.wait_for(entered.wait(), 1)
        return task, release

    async def listener(self):
        queue = asyncio.Queue()
        self.bridge._subscription = SimpleNamespace(receive=queue.get, close=lambda: None)
        task = asyncio.create_task(self.bridge._listen())
        self.addAsyncCleanup(self.join, task)

        async def notify():
            self.bridge._wake.clear()
            queue.put_nowait(SimpleNamespace(resync_required=False))
            await asyncio.wait_for(self.bridge._wake.wait(), 1)
        return task, notify

    def source_revision(self, revision, *, cursor):
        source = fixtures.SOURCE.model_copy(update={"revision": revision})
        binding = fixtures.BINDING.model_copy(update={"dependencies": (source,)})
        entry = self.current.entries[0].model_copy(update={"sources": (source,), "guard_binding": binding})
        return self.current.model_copy(update={"entries": (entry,),
            "cursor": self.current.cursor.model_copy(update={"revision": cursor})})

    async def test_cursor_only_notice_does_not_mark_obsolete_or_add_io(self):
        task, release = await self.held()
        _, notify = await self.listener()
        for revision in (2, 3, 4):
            self.current = fixtures.snapshot(revision)
            await notify()
        self.assertEqual(self.records("first_observed_obsolete"), [])
        self.live.bind.assert_awaited_once()
        self.route.offer_context.assert_not_called()
        release.set()
        await task
        self.route.offer_context.assert_called_once()
        self.assertIsNone(self.bridge._binding_observation)
        self.assertEqual(self.records("binding_exit")[0]["binding_outcome"], "returned")

    async def test_exact_source_change_observed_once_before_completion(self):
        task, release = await self.held()
        _, notify = await self.listener()
        self.current = self.source_revision(2, cursor=2)
        await notify()
        changed = self.records("first_observed_obsolete")
        self.assertEqual(len(changed), 1)
        first = changed[0]
        self.assertEqual(first["observation_boundary"], "notice")
        self.assertTrue(first["key_changes"]["dependencies"])
        self.assertFalse(first["key_changes"]["text"])
        self.current = self.source_revision(3, cursor=3)
        await notify()
        self.assertEqual(len(self.records("first_observed_obsolete")), 1)
        self.assertFalse(task.done())
        self.live.bind.assert_awaited_once()
        release.set()
        await task
        finish = self.records("binding_exit")[0]
        self.assertEqual(finish["first_obsolete_cursor_revision"], 2)
        self.assertEqual(finish["first_obsolete_monotonic_s"], first["first_obsolete_monotonic_s"])
        self.assertLessEqual(self.records("binding_entry")[0]["monotonic_s"], first["first_obsolete_monotonic_s"])
        self.assertLessEqual(first["first_obsolete_monotonic_s"], finish["monotonic_s"])
        self.assertEqual(finish["binding_outcome"], "returned")
        self.route.offer_context.assert_not_called()
        self.assertEqual(self.bridge.last_omission, "selection_changed_during_binding")
        self.assertIsNone(self.bridge._binding_observation)

    async def test_completion_fallback_does_not_invent_notice_time(self):
        async def bind(selection):
            self.current = fixtures.snapshot(2, text="different private source")
            return fixtures.BINDING
        self.live.bind.side_effect = bind
        await self.bridge.publish_latest()
        changed = self.records("first_observed_obsolete")[0]
        self.assertEqual(changed["observation_boundary"], "binding_completion")
        self.assertLessEqual(changed["first_obsolete_monotonic_s"], self.records("binding_exit")[0]["monotonic_s"])
        self.route.offer_context.assert_not_called()

    async def test_transient_change_is_not_reclassified_as_final_discard(self):
        task, release = await self.held()
        _, notify = await self.listener()
        self.current = fixtures.snapshot(2, text="temporary private source")
        await notify()
        self.current = fixtures.snapshot(3)
        await notify()
        release.set()
        await task
        self.assertEqual(len(self.records("first_observed_obsolete")), 1)
        self.assertEqual(self.records("binding_exit")[0]["binding_outcome"], "returned")
        self.route.offer_context.assert_called_once()
        self.assertIsNone(self.bridge.last_omission)

    async def test_exact_committed_binding_reuse_is_not_a_provider_ack(self):
        await self.bridge.publish_latest()
        self.now += timedelta(seconds=9)
        self.current = fixtures.snapshot(2, expiry=fixtures.NOW + timedelta(seconds=20))
        await self.bridge.publish_latest()
        self.assertEqual([row["binding_mode"] for row in self.records("binding_entry")],
                         ["new_binding", "exact_committed_binding_reuse"])
        self.live.bind.assert_awaited_once()
        self.assertEqual(self.route.offer_context.call_count, 2)
        offer = self.records("offer_return")[-1]
        self.assertEqual(offer["binding_sequence"], 2)
        self.assertEqual(offer["selection_remaining_freshness_ms"], 11000)
        self.assertEqual(offer["update_remaining_freshness_ms"], 11000)
        self.assertTrue(offer["diagnostic_only"])
        self.assertNotIn("provider_acknowledged", json.dumps(self.records()))

    async def test_remaining_freshness_exposes_expiry_without_extending_it(self):
        async def bind(selection):
            self.now += timedelta(seconds=11)
            return fixtures.BINDING
        self.live.bind.side_effect = bind
        with patch("app.realtime.live_state_bridge._PREPARATION_LOGGER.warning") as preparation:
            await self.bridge.publish_latest()
        self.assertEqual(self.records("binding_entry")[0]["selection_remaining_freshness_ms"], 10000)
        self.assertEqual(self.records("binding_exit")[0]["selection_remaining_freshness_ms"], -1000)
        self.route.offer_context.assert_not_called()
        observations = [json.loads(call.args[1]) for call in preparation.call_args_list]
        expired = [row for row in observations if row.get("code") == "preparation_expired"]
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0]["stage"], "preparation_exit")
        self.assertEqual(expired[0]["outcome"], "omitted")
        self.assertTrue(expired[0]["binding_returned"])
        self.assertIsNone(self.bridge.last_omission)

    async def test_scope_epoch_or_route_change_is_not_reported_as_same_scope_source_change(self):
        for boundary in ("scope", "epoch", "route"):
            with self.subTest(boundary=boundary):
                original, request = self.current, self.route.request
                task, release = await self.held()
                _, notify = await self.listener()
                self.current = fixtures.snapshot(2, text="different private source")
                if boundary == "scope":
                    self.current = self.current.model_copy(update={"scope": fixtures.SCOPE.model_copy(
                        update={"owner_id": "foreign-private-owner"})})
                elif boundary == "epoch":
                    self.current = self.current.model_copy(update={"cursor": self.current.cursor.model_copy(
                        update={"epoch": UUID(int=99)})})
                else:
                    self.route.request = request.model_copy(update={"route": request.route.model_copy(
                        update={"route_generation": 2})})
                await notify()
                self.assertEqual(self.records("first_observed_obsolete"), [])
                self.current, self.route.request = original, request
                release.set()
                await task
                # Make the next subcase a new binding, without changing policy.
                self.bridge._last_update = self.bridge._last_key = None

    async def test_source_error_and_unexpected_error_keep_original_semantics(self):
        source_error = SourceError("live_state_context_revoked", "private error sentinel")
        self.live.bind.side_effect = source_error
        await self.bridge.publish_latest()
        self.assertEqual(self.bridge.last_omission, source_error.code)
        self.assertEqual(self.records("binding_exit")[-1]["binding_outcome"], "failed")
        original = RuntimeError("private unexpected sentinel")
        self.live.bind.side_effect = original
        with self.assertRaises(RuntimeError) as caught:
            await self.bridge.publish_latest()
        self.assertIs(caught.exception, original)
        self.assertIsNone(self.bridge._binding_observation)
        self.route.offer_context.assert_not_called()
        self.assertNotIn("private", json.dumps(self.records()))

    async def test_close_preserves_cancellation_and_clears_observation(self):
        task, _ = await self.held()
        self.bridge._tasks = (task,)
        await self.bridge.close("offline diagnostic cleanup")
        self.assertTrue(task.cancelled())
        self.assertEqual(self.records("binding_exit")[-1]["binding_outcome"], "cancelled")
        self.assertIsNone(self.bridge._binding_observation)
        self.assertIsNone(self.bridge._prepared_context)
        self.assertFalse(self.bridge._preparing)
        self.route.offer_context.assert_not_called()
        self.live.close.assert_awaited_once()

    async def test_logging_failure_does_not_change_binding_or_offer(self):
        self.diagnostic.side_effect = RuntimeError("private logging failure")
        await self.bridge.publish_latest()
        self.live.bind.assert_awaited_once()
        self.route.offer_context.assert_called_once()
        self.assertIsNone(self.bridge._binding_observation)
        self.assertIsNone(self.bridge.last_omission)

    async def test_diagnostic_snapshot_failure_is_isolated(self):
        # Initial business selection, diagnostic-only completion observation,
        # then the unchanged real post-binding selection.
        self.live.service.snapshot = Mock(side_effect=[self.current,
            RuntimeError("private diagnostic snapshot failure"), self.current])
        await self.bridge.publish_latest()
        self.live.bind.assert_awaited_once()
        self.route.offer_context.assert_called_once()
        self.assertEqual(self.records("first_observed_obsolete"), [])
        self.assertIsNone(self.bridge._binding_observation)

    async def test_sampling_cap_does_not_cap_business_work_or_retain_sources(self):
        for revision in range(1, _BINDING_SAMPLE_LIMIT + 5):
            self.current = fixtures.snapshot(revision, text=f"private payload {revision}")
            await self.bridge.publish_latest()
        self.assertEqual(self.live.bind.await_count, _BINDING_SAMPLE_LIMIT + 4)
        self.assertEqual(self.route.offer_context.call_count, _BINDING_SAMPLE_LIMIT + 4)
        self.assertEqual(len(self.records("binding_entry")), _BINDING_SAMPLE_LIMIT)
        self.assertEqual(len(self.records("binding_exit")), _BINDING_SAMPLE_LIMIT)
        self.assertEqual(len(self.records("offer_return")), _BINDING_SAMPLE_LIMIT)
        self.assertEqual(len(self.records("truncated")), 1)
        self.assertEqual(self.bridge._binding_samples, _BINDING_SAMPLE_LIMIT + 1)
        self.assertIsNone(self.bridge._binding_observation)
        encoded = json.dumps(self.records())
        for secret in ("private payload", str(fixtures.ROOT), str(fixtures.SOURCE.source_id),
                       fixtures.SCOPE.owner_id, "dependencies", "parent_payload_root_ids"):
            self.assertNotIn(secret, encoded)

    async def test_no_binding_has_no_synthetic_binding_record(self):
        self.current = self.current.model_copy(update={"entries": ()})
        await self.bridge.publish_latest()
        self.live.bind.assert_not_awaited()
        self.route.offer_context.assert_called_once()
        self.assertEqual(self.records(), [])
