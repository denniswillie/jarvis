"""Adapter-entry expiry boundaries, offline; composed native flow is separate."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.knowledge.models import SourceError
from app.voice_routes.coordinator import CoordinationPhase
import test_voice_response_context_boundary as fixtures
from test_realtime_voice_route import NOW, make_context


class AdapterExpiryRecoveryTests(unittest.IsolatedAsyncioTestCase):
    # Reuse helpers, not a TestCase subclass: no duplicate collected tests.
    voice = fixtures.VoiceResponseContextBoundaryTests.voice
    route = fixtures.VoiceResponseContextBoundaryTests.route
    until = fixtures.VoiceResponseContextBoundaryTests.until

    async def setup_route(self, *, failure="expiry", candidate_expired=True, enabled=True):
        self.now = NOW
        (*_, self.runtime, self.control, self.adapter, self.owner) = await self.route()
        self.bridge = self.owner._bridge
        self.bridge._clock = self.adapter._clock = lambda: self.now
        self.bridge._expiry_refresh_enabled = enabled
        original = self.runtime._atomic_guard
        self.guard_entries = []

        @asynccontextmanager
        async def guard(update):
            async with original(update) as connection:
                self.guard_entries.append(update.cursor.revision)
                if update.cursor.revision == 1:
                    # Preflight already succeeded. Both old context and the
                    # short-lived candidate age during adapter guard entry.
                    self.now = NOW + timedelta(minutes=6)
                    if failure == "expiry":
                        raise SourceError("voice_context_expired", "private context is not diagnostic")
                    if failure == "withdrawal":
                        raise SourceError("source_withdrawn", "private source is not diagnostic")
                    if failure == "spoofed":
                        error = RuntimeError("private exception is not diagnostic")
                        error.code = "voice_context_expired"
                        raise error
                    if failure == "timeout":
                        raise TimeoutError("private timeout is not diagnostic")
                yield connection

        self.runtime._atomic_guard = guard
        self.candidate = make_context(revision=1,
            expires_at=NOW + timedelta(minutes=1 if candidate_expired else 10))
        logger = patch("app.voice_routes.bridge._PUBLICATION_LOGGER.warning")
        self.log = logger.start()
        self.addCleanup(logger.stop)

    def rows(self, stage):
        return [row for call in self.log.call_args_list
                if (row := json.loads(call.args[1]))["stage"] == stage]

    def mutations(self):
        return [row["type"] for row in self.control.writes]

    async def offer_and_settle(self):
        self.owner.offer_context(self.candidate)
        await self.until(lambda: self.bridge.status.fenced or
                         self.bridge.status.reason == "context_expired_awaiting_snapshot")

    async def test_typed_adapter_expiry_holds_then_fresh_snapshot_recovers(self):
        await self.setup_route()
        await self.offer_and_settle()
        self.assertEqual(self.owner.status.phase, CoordinationPhase.ACTIVE)
        self.assertTrue(self.runtime.context_publication_ready)
        self.assertFalse(self.runtime.ready)
        self.assertTrue(self.runtime.context_refresh_pending)
        self.assertEqual(self.mutations(), ["conversation.item.create", "conversation.item.delete"])
        self.assertEqual(self.guard_entries, [1])
        row = self.rows("adapter_return")[-1]
        self.assertEqual(row["outcome"], "returned")
        self.assertTrue(row["adapter_entered"])
        self.assertEqual(row["write_state"], "write_possible")
        self.assertTrue(row["owned_work_joined"])
        self.assertEqual(row["first_possible_operation"], "context_delete")
        self.assertTrue(row["deleted_context_acknowledged"])
        self.assertFalse(row["replacement_create_possible"])
        self.assertEqual(self.rows("context_refresh_wait")[-1]["outcome"], "expired_after_acknowledged_delete")
        self.assertIsNone(self.bridge.status.retry_update_id)
        self.assertFalse(await self.runtime.wait_for_context_ready(0.01))
        self.assertEqual(self.owner.status.phase, CoordinationPhase.ACTIVE)
        self.assertTrue(all(job.done() for job in self.bridge._jobs))
        fresh = make_context(revision=2, expires_at=self.now + timedelta(seconds=10))
        self.owner.offer_context(fresh)
        self.assertTrue(await self.runtime.wait_for_context_ready(1))
        self.assertTrue(self.runtime.ready)
        self.assertEqual(self.adapter._context, fresh)
        self.assertEqual(self.mutations(), ["conversation.item.create", "conversation.item.delete",
                                            "conversation.item.create"])
        self.assertEqual(self.rows("context_refresh_return")[-1]["outcome"], "fresh_snapshot_written")
        self.assertNotIn("private", str(self.log.call_args_list))

    async def test_expiry_after_guard_returns_rejected_receipt_without_replacement(self):
        await self.setup_route(failure=None)
        await self.offer_and_settle()
        self.assertEqual(self.owner.status.phase, CoordinationPhase.ACTIVE)
        row = self.rows("adapter_return")[-1]
        self.assertEqual(row["outcome"], "returned")
        self.assertEqual(row["write_state"], "write_possible")
        self.assertTrue(row["deleted_context_acknowledged"])
        self.assertFalse(row["replacement_create_possible"])
        self.assertEqual(self.mutations(), ["conversation.item.create", "conversation.item.delete"])
        self.assertTrue(self.runtime.context_refresh_pending)
        self.assertIsNone(self.bridge.status.retry_update_id)

    async def test_nonexpiry_errors_are_not_reclassified_when_candidate_is_old(self):
        # Each case has expired old context, so retaining the old item cannot
        # conceal an incorrect recovery of the candidate rejection.
        for failure in ("withdrawal", "spoofed", "timeout"):
            with self.subTest(failure=failure):
                await self.setup_route(failure=failure)
                await self.offer_and_settle()
                self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
                self.assertEqual(self.rows("context_refresh_wait"), [])
                self.assertFalse(self.runtime.ready)
                self.assertEqual(self.mutations(), ["conversation.item.create", "conversation.item.delete"])
                await self.owner.close()

    async def test_expiry_code_without_actual_expiry_remains_failure(self):
        await self.setup_route(candidate_expired=False)
        await self.offer_and_settle()
        self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
        self.assertEqual(self.rows("context_refresh_wait"), [])
        self.assertEqual(self.mutations(), ["conversation.item.create", "conversation.item.delete"])

    async def test_nonoptin_route_does_not_gain_expiry_recovery(self):
        await self.setup_route(enabled=False)
        await self.offer_and_settle()
        self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
        self.assertEqual(self.rows("context_refresh_wait"), [])

    async def test_untracked_adapter_cannot_claim_no_write(self):
        await self.setup_route()
        self.bridge._route = SimpleNamespace(identity=self.adapter.identity,
            publish_context=self.adapter.publish_context,
            can_retain_context=self.adapter.can_retain_context)
        await self.offer_and_settle()
        self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
        self.assertEqual(self.rows("adapter_return")[-1]["write_state"], "unknown")
        self.assertEqual(self.rows("context_refresh_wait"), [])

    async def test_unjoined_tracked_work_cannot_claim_no_write(self):
        await self.setup_route()

        async def unjoined(update, *, write_evidence):
            write_evidence.begin_tracking()
            try:
                return await self.adapter.publish_context(update)
            finally:
                write_evidence.close()
                # Deliberately no joined receipt from this malformed adapter.

        self.bridge._route = SimpleNamespace(identity=self.adapter.identity,
            publish_context_tracked=unjoined,
            can_retain_context=self.adapter.can_retain_context)
        await self.offer_and_settle()
        self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
        row = self.rows("adapter_return")[-1]
        self.assertEqual(row["write_state"], "prewrite")
        self.assertFalse(row["owned_work_joined"])
        self.assertEqual(self.rows("context_refresh_wait"), [])

    async def test_expiry_after_possible_delete_is_unknown_and_never_retried(self):
        await self.setup_route()
        self.adapter._bindings = replace(self.adapter._bindings, context_replacement_guard=None)
        await self.offer_and_settle()
        self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
        row = self.rows("adapter_return")[-1]
        self.assertEqual(row["write_state"], "write_possible")
        self.assertEqual(row["first_possible_operation"], "context_delete")
        self.assertEqual(self.rows("context_refresh_wait"), [])
        self.assertEqual(self.mutations(), ["conversation.item.create", "conversation.item.delete"])
        self.assertIsNone(self.bridge.status.retry_update_id)


if __name__ == "__main__":
    unittest.main()
