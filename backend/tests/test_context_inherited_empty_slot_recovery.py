"""Native 22 -> 28 ordering through real runtime/adapter/bridge, offline wire only.

Exact controlled provider item ACKs establish mutation evidence, not native
playback, source authorization, or a successful real-world conversation.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.voice_routes.bridge import OfferDisposition
from app.voice_routes.contracts import ContextWriteEvidence
from app.voice_routes.coordinator import CoordinationPhase
import test_voice_response_context_boundary as fixtures
from test_realtime_voice_route import NOW, make_context, make_identity


class InheritedEmptySlotRecoveryTests(unittest.IsolatedAsyncioTestCase):
    voice = fixtures.VoiceResponseContextBoundaryTests.voice
    route = fixtures.VoiceResponseContextBoundaryTests.route
    until = fixtures.VoiceResponseContextBoundaryTests.until

    async def setup_route(self):
        self.now = NOW
        (*_, self.runtime, self.control, self.adapter, self.owner) = await self.route()
        self.bridge = self.owner._bridge
        self.bridge._clock = self.adapter._clock = lambda: self.now
        self.ownership_current = True
        self.adapter._bindings = replace(self.adapter._bindings,
            context_reconciliation_current=lambda: self.ownership_current)
        self.initial_context = self.adapter._context
        self.old_item = self.adapter._context_item
        self.failures = {22: "expiry", 28: "unavailable"}
        self.guards = {}
        self.evidence = {}
        original_guard = self.adapter._bindings.source_guard
        original_publish = self.adapter.publish_context_tracked

        @asynccontextmanager
        async def guard(update):
            async with original_guard(update) as connection:
                revision = update.cursor.revision
                self.guards[revision] = self.guards.get(revision, 0) + 1
                if self.guards[revision] == 1:
                    # This binding is the actual provider-create source guard,
                    # not a guessed ordinal among internal authority checks.
                    self.assertIsNone(self.adapter._context_item)
                    failure = self.failures.get(revision)
                    if failure == "expiry":
                        self.now = update.expires_at + timedelta(seconds=1)
                        raise SourceError("voice_context_expired", "private expired context")
                    if failure == "unavailable":
                        raise SourceError("live_state_runtime_unavailable", "private refresh_not_ready")
                    if failure == "timeout":
                        raise TimeoutError("private guard timeout")
                    if failure:
                        raise SourceError(failure, "private source guard rejection")
                yield connection

        async def publish(update, *, write_evidence):
            self.evidence[update.cursor.revision] = write_evidence
            return await original_publish(update, write_evidence=write_evidence)

        self.adapter._bindings = replace(self.adapter._bindings, source_guard=guard)
        self.adapter.publish_context_tracked = publish
        capture = patch("app.voice_routes.bridge._PUBLICATION_LOGGER.warning")
        self.log = capture.start()
        self.addCleanup(capture.stop)

    def candidate(self, revision):
        return make_context(revision=revision, expires_at=self.now + timedelta(seconds=10))

    def rows(self):
        return [json.loads(call.args[1]) for call in self.log.call_args_list]

    def mutation_types(self):
        return [event["type"] for event in self.control.writes]

    async def seed_empty_slot(self):
        update = self.candidate(22)
        self.owner.offer_context(update)
        await self.until(lambda: self.bridge.status.fenced or
                         self.bridge.status.reason == "context_expired_awaiting_snapshot")
        self.assertFalse(self.bridge.status.fenced)
        self.assertEqual(self.guards[22], 1)
        proof = self.evidence[22]
        self.assertTrue(proof.confirmed_deleted_without_create)
        self.assertEqual(proof._deleted_item_id, self.old_item)
        self.assertEqual(proof.first_operation, "context_delete")
        self.assertFalse(proof.proven_unwritten)
        self.assertEqual(self.adapter._empty_context_candidate,
                         (update, self.adapter._source_epoch, proof))
        self.assertEqual(self.mutation_types(), ["conversation.item.create", "conversation.item.delete"])
        self.assertIsNone(self.adapter._context_item)
        return update

    async def offer_and_settle(self, update):
        self.owner.offer_context(update)
        await self.until(lambda: self.bridge.status.fenced or
                         (update.cursor.revision in self.evidence
                          and self.evidence[update.cursor.revision]._closed
                          and all(job.done() for job in self.bridge._jobs)))
        # Let the bridge consume the terminal publication result.
        await asyncio.sleep(0)

    async def assert_held(self):
        self.assertFalse(self.bridge.status.fenced)
        self.assertEqual(self.owner.status.phase, CoordinationPhase.ACTIVE)
        self.assertFalse(self.runtime.ready)
        self.assertTrue(self.runtime.context_refresh_pending)
        self.assertIsNone(self.bridge.status.retry_update_id)
        self.assertIsNone(self.adapter._context_item)
        self.assertFalse(self.adapter.can_retain_context(self.initial_context))
        self.assertEqual(self.mutation_types(), ["conversation.item.create", "conversation.item.delete"])
        self.assertFalse(await self.runtime.wait_for_context_ready(.01))
        self.assertTrue(all(job.done() for job in self.bridge._jobs))

    async def test_revision_22_delete_expiry_then_28_refresh_not_ready_then_fresh_29(self):
        await self.setup_route()
        original = await self.seed_empty_slot()
        update = self.candidate(28)
        await self.offer_and_settle(update)
        await self.assert_held()
        self.assertEqual(self.guards[28], 1)
        self.assertEqual(self.bridge.status.reason, "context_source_unavailable_awaiting_snapshot")
        self.assertTrue(self.evidence[28].proven_unwritten)
        self.assertFalse(self.evidence[28].deleted_context_acknowledged)
        self.assertEqual(self.adapter._empty_context_candidate[0], original)
        row = [row for row in self.rows() if row["stage"] == "context_refresh_wait"][-1]
        self.assertEqual(row["outcome"], "unavailable_in_acknowledged_empty_context")
        self.assertEqual(row["write_state"], "unwritten")
        self.assertIsNone(row["first_possible_operation"])
        self.assertFalse(row["deleted_context_acknowledged"])
        self.assertFalse(row["replacement_create_possible"])
        self.assertTrue(row["owned_work_joined"])
        fresh = self.candidate(29)
        self.owner.offer_context(fresh)
        self.assertTrue(await self.runtime.wait_for_context_ready(1))
        self.assertEqual(self.adapter._context, fresh)
        self.assertIsNone(self.adapter._empty_context_candidate)
        self.assertEqual(self.mutation_types(), ["conversation.item.create", "conversation.item.delete",
                                                "conversation.item.create"])
        self.assertNotIn("private", str(self.log.call_args_list))

    async def test_one_two_three_independent_unavailable_candidates_never_busy_retry(self):
        await self.setup_route()
        await self.seed_empty_slot()
        for count, revision in enumerate((28, 30, 32), 1):
            with self.subTest(successive_rejections=count):
                self.failures[revision] = "unavailable"
                update = self.candidate(revision)
                await self.offer_and_settle(update)
                await self.assert_held()
                calls = dict(self.guards)
                self.assertEqual(self.bridge.offer(update).disposition, OfferDisposition.DUPLICATE)
                await asyncio.sleep(.01)
                self.assertEqual(self.guards, calls)
                self.assertTrue(self.evidence[revision].proven_unwritten)
        await self.owner.close()
        self.assertIsNone(self.adapter._empty_context_candidate)
        self.assertTrue(all(job.done() for job in self.bridge._jobs))

    async def test_expired_fresh_candidate_cannot_restore_stale_brief_or_ready(self):
        await self.setup_route()
        original = await self.seed_empty_slot()
        self.failures[28] = "expiry"
        update = self.candidate(28)
        await self.offer_and_settle(update)
        await self.assert_held()
        self.assertGreaterEqual(self.now, update.expires_at)
        self.assertEqual(self.bridge.status.reason, "context_expired_awaiting_snapshot")
        # The old snapshot TTL is not a TTL on the historical delete ACK.
        # Neither the original nor this expired snapshot can authorize output.
        self.assertEqual(self.adapter._empty_context_candidate[0], original)
        self.assertTrue(self.evidence[28].proven_unwritten)
        fresh = self.candidate(29)
        self.owner.offer_context(fresh)
        self.assertTrue(await self.runtime.wait_for_context_ready(1))
        self.assertIsNone(self.adapter._empty_context_candidate)

    async def test_withdrawal_correction_permission_and_unknown_guard_failure_still_fence(self):
        for failure in ("source_withdrawn", "source_revision_changed", "source_permission_denied",
                        "voice_context_expired", "timeout"):
            with self.subTest(failure=failure):
                await self.setup_route()
                await self.seed_empty_slot()
                self.failures[28] = failure
                await self.offer_and_settle(self.candidate(28))
                await self.until(lambda: self.owner.status.phase == CoordinationPhase.DEGRADED)
                self.assertTrue(self.bridge.status.fenced)
                self.assertFalse(self.runtime.ready)
                self.assertIsNone(self.adapter._empty_context_candidate)
                self.assertEqual(self.mutation_types(), ["conversation.item.create", "conversation.item.delete"])

    async def test_unjoined_new_attempt_cannot_inherit_old_delete_ack(self):
        await self.setup_route()
        await self.seed_empty_slot()
        with patch.object(ContextWriteEvidence, "mark_joined", return_value=None):
            await self.offer_and_settle(self.candidate(28))
            await self.until(lambda: self.bridge.status.fenced)
        self.assertFalse(self.evidence[28].proven_unwritten)
        self.assertIsNone(self.adapter._empty_context_candidate)
        self.assertFalse(self.runtime.ready)
        self.assertTrue(all(job.done() for job in self.bridge._jobs))

    async def test_possible_create_with_lost_return_or_missing_ack_never_replayed(self):
        for acknowledge in (True, False):
            with self.subTest(acknowledge=acknowledge):
                await self.setup_route()
                await self.seed_empty_slot()
                self.failures[28] = None
                original_write = self.control._write
                if not acknowledge:
                    self.control.item_event = None

                async def lost_return(event):
                    await original_write(event)
                    raise SourceError("live_state_runtime_unavailable", "private lost return")

                self.control._write = lost_return
                await self.offer_and_settle(self.candidate(28))
                await self.until(lambda: self.bridge.status.fenced)
                self.assertTrue(self.evidence[28].replacement_create_possible)
                self.assertEqual(self.evidence[28].state, "write_possible")
                self.assertIsNone(self.adapter._empty_context_candidate)
                self.assertFalse(self.runtime.ready)
                self.assertIsNone(self.bridge.status.retry_update_id)
                self.assertEqual(self.mutation_types(), ["conversation.item.create", "conversation.item.delete",
                                                        "conversation.item.create"])
                self.control._write = original_write

    async def test_lost_local_ownership_rejects_and_clears_prior_proof(self):
        for mode in ("false", "raises", "missing"):
            with self.subTest(ownership=mode):
                await self.setup_route()
                await self.seed_empty_slot()
                if mode == "false":
                    self.ownership_current = False
                else:
                    def broken():
                        raise RuntimeError("private ownership failure")
                    self.adapter._bindings = replace(self.adapter._bindings,
                        context_reconciliation_current=broken if mode == "raises" else None)
                await self.offer_and_settle(self.candidate(28))
                await self.until(lambda: self.bridge.status.fenced)
                self.assertIsNone(self.adapter._empty_context_candidate)
                self.assertFalse(self.runtime.ready)

    async def test_proof_is_bound_to_route_generation_source_epoch_and_cursor_epoch(self):
        for mode in ("route_generation", "source_epoch", "cursor_epoch"):
            with self.subTest(boundary=mode):
                await self.setup_route()
                await self.seed_empty_slot()
                update = self.candidate(28)
                evidence = ContextWriteEvidence()
                evidence.begin_tracking()
                evidence.close()
                evidence.mark_joined()
                if mode == "route_generation":
                    update = update.model_copy(update={"route": make_identity(2)})
                elif mode == "source_epoch":
                    self.adapter._source_epoch += 1
                else:
                    update = update.model_copy(update={"cursor": update.cursor.model_copy(update={"epoch": uuid4()})})
                self.assertFalse(self.adapter.can_await_fresh_context_after_delete(update, write_evidence=evidence))
                self.assertIsNone(self.adapter._empty_context_candidate)
                self.assertEqual(self.mutation_types(), ["conversation.item.create", "conversation.item.delete"])

    async def test_disabled_recovery_does_not_inherit_empty_slot(self):
        await self.setup_route()
        await self.seed_empty_slot()
        self.bridge._expiry_refresh_enabled = False
        await self.offer_and_settle(self.candidate(28))
        await self.until(lambda: self.bridge.status.fenced)
        self.assertFalse(self.runtime.ready)
        self.assertIsNone(self.bridge.status.retry_update_id)
