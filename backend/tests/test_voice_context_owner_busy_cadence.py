"""Lock contention scheduling must recover without relaxing write retry limits."""
from dataclasses import replace
from datetime import timedelta
import unittest
from unittest.mock import patch

from app.voice_routes.bridge import ContextBridgeStatus
import test_live_state_voice_attachment as fixtures


class OwnerBusyCadenceTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.ProjectionAttachmentTests.setUp

    async def held(self, reason="context_owner_busy_awaiting_snapshot", *, retry=True):
        await self.bridge.publish_latest()
        self.update = self.route.offer_context.call_args.args[0]
        self.route.context_publication_status = ContextBridgeStatus(
            identity=self.update.route, running=True, fenced=False, reason=reason,
            newest_cursor=self.update.cursor, last_receipt=None,
            retry_update_id=self.update.update_id if retry else None)
        self.route.offer_context.reset_mock()

    async def test_busy_rechecks_are_rate_limited_and_outlive_write_retry_budget(self):
        await self.held()
        with patch("app.realtime.live_state_bridge.monotonic", return_value=100.0) as clock:
            await self.bridge.publish_latest()
            for tick in (100.1, 100.9):
                clock.return_value = tick
                await self.bridge.publish_latest()
            self.route.offer_context.assert_not_called()
            for tick in (101.0, 102.0, 103.0, 104.0):
                clock.return_value = tick
                await self.bridge.publish_latest()
                await self.bridge.publish_latest()
            self.assertEqual(self.route.offer_context.call_count, 4)
            self.assertEqual(self.bridge._unwritten_retry_count, 0)
            self.assertTrue(all(call.args == (self.update,)
                                for call in self.route.offer_context.call_args_list))

    async def test_other_proven_unwritten_failures_keep_existing_retry_limit(self):
        await self.held("context_publication_deferred")
        for _ in range(6):
            await self.bridge.publish_latest()
        self.assertEqual(self.route.offer_context.call_count, 2)
        self.assertEqual(self.bridge.last_omission, "safe_context_retry_exhausted")

    async def test_busy_without_exact_no_write_proof_does_not_reoffer(self):
        await self.held(retry=False)
        with patch("app.realtime.live_state_bridge.monotonic", return_value=1000.0):
            for _ in range(6):
                await self.bridge.publish_latest()
        self.route.offer_context.assert_not_called()

    async def test_awaiting_snapshot_prepares_newer_unchanged_content(self):
        await self.held(retry=False)
        self.route.context_refresh_pending = True
        self.current = fixtures.snapshot(2)
        await self.bridge.publish_latest()
        self.route.offer_context.assert_called_once()
        offered = self.route.offer_context.call_args.args[0]
        self.assertEqual(offered.cursor, self.current.cursor)
        self.assertEqual(offered.text, self.update.text)
        self.assertNotEqual(offered.update_id, self.update.update_id)

    async def test_empty_slot_does_not_invent_a_new_state_revision(self):
        await self.held(retry=False)
        self.route.context_refresh_pending = True
        await self.bridge.publish_latest()
        self.route.offer_context.assert_not_called()

    async def test_fresh_snapshot_cannot_reopen_a_fenced_publication(self):
        await self.held(retry=False)
        self.route.context_refresh_pending = True
        self.current = fixtures.snapshot(2)
        self.route.context_publication_status = replace(
            self.route.context_publication_status, fenced=True)
        await self.bridge.publish_latest()
        self.route.offer_context.assert_not_called()

    async def test_expired_selection_is_not_reoffered(self):
        await self.held()
        self.now += timedelta(seconds=11)
        with patch("app.realtime.live_state_bridge.monotonic", return_value=1000.0):
            await self.bridge.publish_latest()
        self.route.offer_context.assert_not_called()

    async def test_expired_update_yields_to_newer_same_content_selection(self):
        await self.held()
        self.now += timedelta(seconds=11)
        self.current = fixtures.snapshot(2, expiry=fixtures.NOW + timedelta(seconds=20))
        with patch("app.realtime.live_state_bridge.monotonic", return_value=1000.0):
            await self.bridge.publish_latest()
        self.route.offer_context.assert_called_once()
        offered = self.route.offer_context.call_args.args[0]
        self.assertNotEqual(offered.update_id, self.update.update_id)
        self.assertEqual(offered.cursor, self.current.cursor)
        self.assertGreater(offered.expires_at, self.now)
        self.assertEqual(offered.guard_binding, self.update.guard_binding)

    async def test_fenced_publication_does_not_reoffer(self):
        await self.held()
        self.route.context_publication_status = replace(
            self.route.context_publication_status, fenced=True)
        await self.bridge.publish_latest()
        self.route.offer_context.assert_not_called()


if __name__ == "__main__":
    unittest.main()
