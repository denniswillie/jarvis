"""Regression for an obsolete preflight result, without provider or database I/O."""

import asyncio
import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from app.live_state.models import SessionScope, StateCursor
from app.voice_routes import bridge as bridge_module
from app.voice_routes.bridge import VoiceContextBridge
from app.voice_routes.contracts import ContextReceipt, RouteIdentity, VoiceContextUpdate


class VoiceContextSupersessionRaceTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, *, supersede):
        now = datetime(2026, 10, 31, tzinfo=timezone.utc)
        identity = RouteIdentity(
            scope=SessionScope(owner_id="context-race-fixture", session_id=uuid4(), area_id=None),
            route_id=uuid4(), route_generation=1,
        )
        epoch = uuid4()

        def update(revision):
            return VoiceContextUpdate(
                route=identity, update_id=uuid4(),
                cursor=StateCursor(epoch=epoch, revision=revision), mode="snapshot",
                text=f"Fixture revision {revision}", dependencies=(), created_at=now,
            )

        entered = asyncio.Event()
        release = asyncio.Event()
        published = []
        failures = []

        async def validity(value):
            if value.cursor.revision == 1:
                entered.set()
                await release.wait()
                return False
            return True

        class Route:
            async def publish_context(self, value):
                published.append(value.cursor.revision)
                return ContextReceipt(
                    route=identity, update_id=value.update_id, cursor=value.cursor,
                    observed_at=now, status="written",
                )

        route = Route()
        route.identity = identity
        bridge = VoiceContextBridge(
            route, update(0), validity=validity, publish_timeout=1,
            clock=lambda: now, on_failure=failures.append,
        )
        await bridge.start()
        try:
            bridge.offer(update(1))
            await asyncio.wait_for(entered.wait(), timeout=1)
            if supersede:
                bridge.offer(update(2))
            release.set()
            async with asyncio.timeout(1):
                while not bridge.status.fenced and not published:
                    await asyncio.sleep(0)
            return bridge.status, published, failures
        finally:
            release.set()
            await bridge.close()

    async def test_superseded_invalid_preflight_does_not_fence_newer_context(self):
        status, published, failures = await self.exercise(supersede=True)
        self.assertFalse(status.fenced, status.reason)
        self.assertEqual(published, [2])
        self.assertEqual(failures, [])

    async def test_current_invalid_preflight_still_fails_closed(self):
        status, published, failures = await self.exercise(supersede=False)
        self.assertTrue(status.fenced)
        self.assertEqual(status.reason, "context_validity_rejected")
        self.assertEqual(published, [])
        self.assertEqual(failures, ["context_validity_rejected"])


class VoiceContextDeadlineDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = datetime(2026, 10, 31, tzinfo=timezone.utc)
        self.identity = RouteIdentity(
            scope=SessionScope(owner_id="context-deadline-fixture", session_id=uuid4(), area_id=None),
            route_id=uuid4(), route_generation=1)
        self.epoch = uuid4()
        self.published, self.failures = [], []
        logger = patch.object(bridge_module._PUBLICATION_LOGGER, "warning")
        self.log = logger.start()
        self.addCleanup(logger.stop)

    def update(self, revision):
        return VoiceContextUpdate(route=self.identity, update_id=uuid4(),
            cursor=StateCursor(epoch=self.epoch, revision=revision), mode="snapshot",
            text="PRIVATE_CONTEXT_DO_NOT_LOG", dependencies=(), created_at=self.now)

    async def begin(self, validity, publish=None):
        async def publish_default(value):
            self.published.append(value.cursor.revision)
            return ContextReceipt(route=self.identity, update_id=value.update_id,
                cursor=value.cursor, observed_at=self.now, status="written")
        route = SimpleNamespace(identity=self.identity, publish_context=publish or publish_default)
        bridge = VoiceContextBridge(route, self.update(0), validity=validity, publish_timeout=0.03,
            clock=lambda: self.now, on_failure=self.failures.append)
        self.addAsyncCleanup(bridge.close)
        await bridge.start()
        return bridge

    async def until(self, predicate):
        async with asyncio.timeout(1):
            while not predicate():
                await asyncio.sleep(0)

    def rows(self, stage):
        return [row for call in self.log.call_args_list
                if (row := json.loads(call.args[1]))["stage"] == stage]

    async def test_superseded_validity_timeout_joins_before_latest_snapshot(self):
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        active, high_water = 0, 0

        async def validity(value):
            nonlocal active, high_water
            active += 1
            high_water = max(active, high_water)
            try:
                if value.cursor.revision == 1:
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        cancelled.set()
                        await release.wait()
                        raise
                return True
            finally:
                active -= 1

        bridge = await self.begin(validity)
        first = self.update(1)
        try:
            bridge.offer(first)
            await asyncio.wait_for(entered.wait(), 1)
            bridge.offer(self.update(2))
            await asyncio.wait_for(cancelled.wait(), 1)
            self.assertEqual(self.published, [])
            self.assertEqual(len(bridge._jobs), 1)
            bridge.offer(self.update(3))
            release.set()
            await self.until(lambda: bridge.status.last_receipt is not None)
            self.assertEqual(self.published, [3])
            self.assertEqual(high_water, 1)
            self.assertEqual(active, 0)
            self.assertFalse(bridge.status.fenced)
            self.assertEqual(self.failures, [])
            deadline = self.rows("publication_deadline")[0]
            self.assertEqual(deadline["phase"], "validity")
            self.assertEqual(deadline["outcome"], "superseded_preflight")
            self.assertEqual(deadline["update_id"], str(first.update_id))
            self.assertFalse(deadline["adapter_entered"])
            self.assertEqual(len(self.rows("cancellation_join_return")), 1)
            self.assertEqual(len(self.rows("publication_skipped")), 1)
            self.assertEqual([row["cursor_revision"] for row in self.rows("adapter_entry")], [3])
        finally:
            release.set()

    async def test_current_validity_timeout_still_fails_closed(self):
        async def validity(_):
            await asyncio.Event().wait()

        bridge = await self.begin(validity)
        bridge.offer(self.update(1))
        await self.until(lambda: bridge.status.fenced)
        self.assertEqual(self.failures, ["context_publication_deadline"])
        self.assertEqual(self.published, [])
        self.assertIsNone(bridge.status.last_receipt)
        row = self.rows("publication_deadline")[0]
        self.assertEqual((row["phase"], row["outcome"]), ("validity", "failed_closed"))
        self.assertFalse(row["adapter_entered"])
        self.assertFalse(row["superseded"])

    async def test_superseded_adapter_timeout_still_has_unknown_effects_and_fences(self):
        entered = asyncio.Event()

        async def validity(_):
            return True

        async def publish(_):
            entered.set()
            await asyncio.Event().wait()

        bridge = await self.begin(validity, publish)
        bridge.offer(self.update(1))
        await asyncio.wait_for(entered.wait(), 1)
        bridge.offer(self.update(2))
        await self.until(lambda: bridge.status.fenced)
        self.assertEqual(self.failures, ["context_publication_deadline"])
        self.assertIsNone(bridge.status.last_receipt)
        row = self.rows("publication_deadline")[0]
        self.assertEqual((row["phase"], row["outcome"]), ("adapter", "failed_closed"))
        self.assertTrue(row["adapter_entered"])
        self.assertTrue(row["superseded"])
        self.assertEqual(self.rows("publication_skipped"), [])
        self.assertEqual([row["cursor_revision"] for row in self.rows("adapter_entry")], [1])

    async def test_uncooperative_preflight_stays_owned_and_blocks_replacement(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def validity(_):
            entered.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
            return True

        bridge = await self.begin(validity)
        try:
            bridge.offer(self.update(1))
            await asyncio.wait_for(entered.wait(), 1)
            bridge.offer(self.update(2))
            await self.until(lambda: bridge.status.fenced)
            self.assertEqual(self.failures, ["context_publication_deadline"])
            self.assertEqual(len(bridge._jobs), 1)
            self.assertEqual(self.published, [])
            self.assertEqual(self.rows("adapter_entry"), [])
            self.assertEqual(self.rows("cancellation_join_deadline")[0]["outcome"], "cleanup_unconfirmed")
            closing = asyncio.create_task(bridge.close())
            await asyncio.sleep(0)
            self.assertFalse(closing.done())
            release.set()
            await asyncio.wait_for(closing, 1)
            self.assertEqual(len(bridge._jobs), 0)
        finally:
            release.set()

    async def test_static_diagnostics_do_not_log_context_or_exception_contents(self):
        async def validity(_):
            raise RuntimeError("PRIVATE_ERROR Bearer sk-secret SDP")

        bridge = await self.begin(validity)
        value = self.update(1)
        bridge.offer(value)
        await self.until(lambda: bridge.status.fenced)
        self.assertEqual(self.failures, ["context_publication_failed"])
        self.assertEqual(self.rows("validity_exit")[0]["outcome"], "failed")
        for row in self.rows("validity_entry") + self.rows("validity_exit"):
            self.assertEqual(row["route_id"], str(self.identity.route_id))
            self.assertEqual(row["update_id"], str(value.update_id))
            self.assertEqual(row["deadline_ms"], 30)
            self.assertGreaterEqual(row["elapsed_ms"], row["phase_elapsed_ms"])
            self.assertGreaterEqual(row["phase_elapsed_ms"], 0)
        rendered = str(self.log.call_args_list)
        for private in (value.text, "PRIVATE_ERROR", "Bearer", "sk-secret", "SDP", self.identity.scope.owner_id):
            self.assertNotIn(private, rendered)

    async def test_diagnostic_bound_and_logger_failure_do_not_block_publication(self):
        async def validity(_):
            return True

        bridge = await self.begin(validity)
        value = self.update(1)
        now = asyncio.get_running_loop().time()
        progress = bridge_module._PublicationProgress(started_at=now, phase_started_at=now)
        for _ in range(bridge_module._PUBLICATION_LOG_LIMIT + 20):
            bridge._diagnostic(value, progress, "validity_entry", "started")
        self.assertEqual(len(self.log.call_args_list), bridge_module._PUBLICATION_LOG_LIMIT + 1)
        self.assertEqual(len(self.rows("truncated")), 1)
        self.assertTrue(self.rows("truncated")[0]["absence_after_limit_is_inconclusive"])
        bridge.offer(value)
        await self.until(lambda: bridge.status.last_receipt is not None)
        self.assertEqual(self.published, [1])
        self.assertFalse(bridge.status.fenced)
        self.log.side_effect = RuntimeError("PRIVATE_LOG_FAILURE")
        other = await self.begin(validity)
        other.offer(self.update(1))
        await self.until(lambda: other.status.last_receipt is not None)
        self.assertFalse(other.status.fenced)


if __name__ == "__main__":
    unittest.main()
