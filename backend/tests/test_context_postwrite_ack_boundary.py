"""Post-write ACK boundaries through the existing real session composition.

Reuse the real publisher, binder/source guards, adapter/bridge and serialized
writer. Only the existing controlled provider socket's ACK timing/shape changes.
No native, provider, SQL, audio/hearing or historical ACK acceptance is claimed.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
import json
import unittest

from app.knowledge.models import SourceError
from app.voice_routes.coordinator import CoordinationPhase
import test_voice_context_response_pipeline as pipeline


class ContextPostwriteAckBoundaryTests(unittest.IsolatedAsyncioTestCase):
    # Alias helpers, not the TestCase: do not collect its existing tests twice.
    asyncSetUp = pipeline.VoiceContextResponsePipelineTests.asyncSetUp
    cleanup = pipeline.VoiceContextResponsePipelineTests.cleanup
    until = pipeline.VoiceContextResponsePipelineTests.until
    rows = pipeline.VoiceContextResponsePipelineTests.rows
    observation = pipeline.VoiceContextResponsePipelineTests.observation
    expire_inside_adapter_guard = pipeline.VoiceContextResponsePipelineTests.expire_inside_adapter_guard
    mutations = pipeline.VoiceContextResponsePipelineTests.mutations
    creates = pipeline.VoiceContextResponsePipelineTests.creates
    question_and_accept = pipeline.VoiceContextResponsePipelineTests.question_and_accept

    async def begin_candidate(self, *, ack="conversation.item.done", override=None,
                              expire=True, hold_send=False, change_epoch=False,
                              lose_runtime_authority=False):
        self.storage.on_lock = None
        adapter = self.runtime.adapter
        bridge = self.runtime.coordinator._bridge
        # Existing timeouts only; no alternative ACK deadline or state machine.
        adapter._io_timeout = .15
        bridge._timeout = .5
        self.old_context = adapter._context
        self.old_id = adapter._context_item
        self.old_item = adapter._items[self.old_id]
        self.before = len(self.mutations())
        self.boundary_seen = asyncio.Event()
        self.boundary_send_returned = asyncio.Event()
        self.release_send = asyncio.Event()
        self.boundary_item = None
        original_inject = self.provider.inject
        original_send = self.provider.send
        self.original_inject = original_inject

        async def inject(kind, **fields):
            raw = fields.get("item")
            if (kind == "conversation.item.done" and self.boundary_item is None
                    and isinstance(raw, dict) and raw.get("id") != self.old_id):
                self.boundary_item = adapter._items[raw["id"]]
                self.boundary_fields = dict(fields)
                self.candidate = bridge._latest
                self.assertEqual(self.old_item.state, "deleted")
                self.assertNotIn(self.old_id, adapter._items)
                self.assertEqual(self.boundary_item.state, "create_pending")
                if expire:
                    self.assertIsNotNone(self.candidate.expires_at)
                    self.now = self.candidate.expires_at + timedelta(milliseconds=1)
                self.boundary_seen.set()
                if hold_send:
                    await self.release_send.wait()
                if change_epoch:
                    # Existing authority token seam, also used by the existing
                    # confirmed-delete fixture. No fabricated recovery outcome.
                    adapter._source_epoch += 1
                if ack is not None:
                    item = {**raw, **(override or {})}
                    await original_inject(ack, **{**fields, "item": item})
                    if (ack in {"conversation.item.done", "conversation.item.created"}
                            and not override):
                        # Force this ordering: actual receiver consumes exact
                        # final ACK before the controlled socket send returns.
                        await self.until(lambda: self.boundary_item.state == "created")
                if lose_runtime_authority:
                    # The final ACK was consumed, but selected-runtime
                    # ownership disappears before the socket send returns.
                    self.runtime._closing = True
                return
            await original_inject(kind, **fields)

        async def send(raw):
            result = await original_send(raw)
            event = json.loads(raw)
            if (event.get("type") == "conversation.item.create"
                    and self.boundary_item is not None
                    and event["item"]["id"] == self.boundary_item.item_id):
                self.boundary_send_returned.set()
            return result

        self.provider.inject, self.provider.send = inject, send
        self.addCleanup(setattr, self.provider, "inject", original_inject)
        self.addCleanup(setattr, self.provider, "send", original_send)
        await self.observation("postwrite-expiry-candidate", seconds=2)
        await self.until(self.boundary_seen.is_set)

    async def settle_candidate(self):
        bridge = self.runtime.coordinator._bridge
        await self.until(lambda: all(job.done() for job in bridge._jobs) and
                         (bridge.status.fenced
                          or bridge.status.reason == "context_expired_awaiting_snapshot"
                          or (self.runtime.ready and bridge.status.last_receipt is not None
                              and bridge.status.last_receipt.update_id == self.candidate.update_id)))
        returned = [row for row in self.rows(self.publication, "adapter_return")
                    if row.get("update_id") == str(self.candidate.update_id)]
        self.assertEqual(len(returned), 1)
        self.assertEqual(returned[0]["write_state"], "write_possible")
        self.assertTrue(returned[0]["deleted_context_acknowledged"])
        self.assertTrue(returned[0]["replacement_create_possible"])
        self.assertTrue(returned[0]["owned_work_joined"])
        self.assertEqual([e["type"] for e in self.mutations()[self.before:]],
                         ["conversation.item.delete", "conversation.item.create"])
        self.assertEqual(self.mutations()[self.before]["item_id"], self.old_id)
        return returned[0]

    async def exact_final_ack(self, kind="conversation.item.done"):
        await self.original_inject(kind, **self.boundary_fields)
        await self.until(lambda: self.boundary_item.state == "created")

    async def assert_fenced_without_replay(self):
        bridge = self.runtime.coordinator._bridge
        await self.until(lambda: bridge.status.fenced)
        self.assertFalse(self.runtime.ready)
        self.assertFalse(self.runtime.context_refresh_pending)
        self.assertIsNone(bridge.status.retry_update_id)
        self.assertFalse(await self.runtime.wait_for_context_ready(.01))
        before = list(self.mutations())
        # Real offer entrypoints must not replay the old or possibly-created
        # candidate, even when a genuinely newer observation subsequently exists.
        self.runtime.coordinator.offer_context(self.old_context)
        self.runtime.coordinator.offer_context(self.candidate)
        await self.observation("new-snapshot-cannot-unfence-lost-authority")
        await asyncio.sleep(.12)  # One existing publisher tick, not a latency SLA.
        self.assertEqual(self.mutations(), before)
        self.assertFalse(self.creates())
        self.assertFalse(self.runtime.ready)

    async def recover_only_with_fresh_guarded_snapshot(self):
        bridge = self.runtime.coordinator._bridge
        adapter = self.runtime.adapter
        await self.settle_candidate()
        self.assertEqual(self.boundary_item.state, "created")
        self.assertEqual(adapter._context_item, self.boundary_item.item_id)
        self.assertFalse(adapter.can_retain_context(self.candidate))
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertTrue(self.runtime.context_refresh_pending)
        self.assertTrue(self.runtime.context_publication_ready)
        self.assertFalse(self.runtime.ready)
        self.assertIsNone(bridge.status.retry_update_id)
        self.assertFalse(await self.runtime.wait_for_context_ready(.01))
        self.assertFalse(self.creates())
        self.runtime.coordinator.offer_context(self.old_context)
        self.runtime.coordinator.offer_context(self.candidate)
        ask = asyncio.create_task(self.question_and_accept())
        try:
            pin = await asyncio.wait_for(ask, 3)
            self.assertFalse(self.runtime.ready)
            self.assertIsNone(adapter._context_item)
            self.assertIn("Current Live State briefing is temporarily unavailable",
                          self.creates()[0]["response"]["instructions"])
            mutations = self.mutations()[self.before:]
            self.assertEqual([e["type"] for e in mutations],
                             ["conversation.item.delete", "conversation.item.create",
                              "conversation.item.delete"])
            self.assertEqual(mutations[2]["item_id"], self.boundary_item.item_id)
            self.assertEqual(len(self.creates()), 1)
            await self.provider.inject("response.output_audio_transcript.done",
                response_id="answer", item_id="answer-item", transcript="Controlled offline answer.")
            await self.provider.inject("response.done", response={"id": "answer",
                "status": "completed", "output": [{"id": "answer-item", "type": "message",
                "phase": "final_answer", "content": [{"type": "output_audio"}]}]})
            await self.until(lambda: any(e["type"] == "response.done" for e in self.app.events))
            await self.provider.inject("output_audio_buffer.stopped", response_id="answer")
            await self.until(lambda: "answer" not in self.session._direct_output_pending)
            await self.observation("fresh-guarded-after-postwrite-ack")
            await self.until(lambda: self.runtime.ready and
                "fresh-guarded-after-postwrite-ack" in adapter._context.text)
            self.assertNotEqual(adapter._context.update_id, self.candidate.update_id)
            self.assertNotIn("postwrite-expiry-candidate", adapter._context.text)
            self.assertEqual([e["type"] for e in self.mutations()[self.before:]],
                             ["conversation.item.delete", "conversation.item.create",
                              "conversation.item.delete", "conversation.item.create"])
            assistant = [call.kwargs for call in self.repository.create_message.await_args_list
                         if call.kwargs["role"] == "assistant"]
            self.assertEqual(len(assistant), 1)
            self.assertEqual(assistant[0]["source_payload_id"], pin.payload_id)
        finally:
            ask.cancel()
            await asyncio.gather(ask, return_exceptions=True)

    async def test_exact_done_after_expiry_resolves_mutation_not_speech(self):
        await self.begin_candidate()
        await self.recover_only_with_fresh_guarded_snapshot()

    async def test_legacy_created_after_expiry_has_same_strong_match_contract(self):
        await self.begin_candidate(ack="conversation.item.created")
        await self.recover_only_with_fresh_guarded_snapshot()

    async def test_exact_final_ack_after_send_return_uses_bounded_reconciliation(self):
        await self.begin_candidate(ack=None)
        await self.boundary_send_returned.wait()
        # Socket send has returned, while the existing real adapter/bridge owns
        # the publication. The same exact final ACK arrives through its receiver.
        await self.until(lambda: self.runtime.adapter.context_reconciliation_pending)
        self.assertTrue(self.runtime.context_refresh_pending)
        self.assertFalse(self.runtime.ready)
        self.assertFalse(self.creates())
        await self.exact_final_ack()
        await self.recover_only_with_fresh_guarded_snapshot()

    async def test_fresh_exact_ack_keeps_ordinary_ready_publication(self):
        await self.begin_candidate(expire=False)
        await self.settle_candidate()
        self.assertTrue(self.runtime.ready)
        self.assertFalse(self.runtime.context_refresh_pending)
        self.assertTrue(self.runtime.adapter.can_retain_context(self.candidate))
        self.assertFalse(self.creates())

    async def reject_unconfirmed_ack(self, *, ack=None, override=None):
        await self.begin_candidate(ack=ack, override=override)
        await self.settle_candidate()
        self.assertEqual(self.boundary_item.state, "create_pending")
        await self.assert_fenced_without_replay()

    async def test_no_ack_never_retries_possible_create(self):
        await self.reject_unconfirmed_ack()

    async def test_foreign_item_final_ack_is_not_our_create_ack(self):
        await self.reject_unconfirmed_ack(ack="conversation.item.done",
                                          override={"id": "foreign-owned-item"})

    async def test_wrong_role_final_ack_is_not_valid_publication(self):
        await self.reject_unconfirmed_ack(ack="conversation.item.done",
                                          override={"role": "assistant"})

    async def test_wrong_item_type_final_ack_is_not_valid_publication(self):
        await self.reject_unconfirmed_ack(ack="conversation.item.done",
                                          override={"type": "function_call"})

    async def test_added_only_is_not_a_final_ack_even_for_exact_item(self):
        await self.reject_unconfirmed_ack(ack="conversation.item.added")

    async def test_late_exact_final_ack_cannot_unfence_expired_deadline(self):
        await self.begin_candidate(ack=None)
        await self.settle_candidate()
        await self.until(lambda: self.runtime.coordinator._bridge.status.fenced)
        await self.exact_final_ack()
        await self.assert_fenced_without_replay()

    async def test_duplicate_exact_final_ack_does_not_replay_mutation(self):
        await self.begin_candidate()
        await self.settle_candidate()
        before = list(self.mutations())
        await self.exact_final_ack()
        self.assertEqual(self.mutations(), before)
        await self.recover_only_with_fresh_guarded_snapshot()

    async def test_fresh_source_withdrawal_blocks_delete_of_acknowledged_stale_item(self):
        await self.begin_candidate()
        await self.settle_candidate()
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertTrue(self.runtime.context_refresh_pending)
        rejected_fresh_guard = asyncio.Event()

        async def reject_fresh_roots(roots):
            if self.expiry_target is None:
                return
            source_id, _ = self.expiry_target
            matching = [body for identity, body in self.storage.payloads
                        if identity in roots and body.get("format") == "jarvis.live_state.voice.v1"
                        and any(item["source_id"] == str(source_id)
                                for item in body["dependencies"])]
            if matching:
                # Existing controlled canonical storage predicate, not a mocked
                # adapter receipt or an authority bypass. Target only the newly
                # bound fresh source, not prior conversation evidence.
                self.storage.rejected.update(roots)
                rejected_fresh_guard.set()

        self.storage.on_lock = reject_fresh_roots
        # Reuse the fixture's source-ID target capture; the replacement storage
        # hook above withdraws that source and never changes the clock.
        await self.observation("fresh-but-withdrawn-before-delete", expire_in_adapter=True)
        await self.until(rejected_fresh_guard.is_set)
        await self.until(lambda: self.runtime.coordinator._bridge.status.fenced)
        self.assertFalse(self.runtime.ready)
        self.assertFalse(self.creates())
        self.assertEqual(self.boundary_item.state, "created")
        self.assertEqual([e["type"] for e in self.mutations()[self.before:]],
                         ["conversation.item.delete", "conversation.item.create"])

    async def test_scope_denial_on_fresh_precheck_keeps_stale_item_untouched(self):
        await self.begin_candidate()
        await self.settle_candidate()
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertTrue(self.runtime.context_refresh_pending)
        scope_checks = []
        original_guard = self.live.guard

        @asynccontextmanager
        async def scope_denied(*args, **kwargs):
            # Controlled denial at the existing scope/current-source guard
            # boundary, NOT a claim of SQL owner/session predicates executing.
            scope_checks.append(True)
            raise SourceError("live_state_scope_changed", "controlled scope denial")
            yield  # Preserve the existing async-context-manager interface.

        self.live.guard = scope_denied
        self.addCleanup(setattr, self.live, "guard", original_guard)
        await self.observation("fresh-but-no-longer-scope-authorized")
        await self.until(lambda: bool(scope_checks))
        await self.until(lambda: self.runtime.coordinator._bridge.status.fenced)
        self.assertFalse(self.runtime.ready)
        self.assertFalse(self.creates())
        self.assertEqual(self.boundary_item.state, "created")
        self.assertEqual([e["type"] for e in self.mutations()[self.before:]],
                         ["conversation.item.delete", "conversation.item.create"])

    async def test_source_epoch_change_prevents_exact_ack_recovery(self):
        await self.begin_candidate(change_epoch=True)
        await self.settle_candidate()
        self.assertEqual(self.boundary_item.state, "created")
        await self.assert_fenced_without_replay()

    async def test_selected_runtime_ownership_lost_after_exact_ack_before_send_return(self):
        try:
            await self.begin_candidate(lose_runtime_authority=True)
            await self.settle_candidate()
            self.assertEqual(self.boundary_item.state, "created")
        finally:
            self.runtime._closing = False
        await self.assert_fenced_without_replay()

    async def test_selected_runtime_ownership_lost_during_bounded_ack_wait(self):
        await self.begin_candidate(ack=None)
        await self.boundary_send_returned.wait()
        await self.until(lambda: self.runtime.adapter.context_reconciliation_pending)
        self.assertFalse(self.runtime.ready)
        self.runtime._closing = True
        try:
            # Queue a strong ACK, not added-only. Loss of route ownership must
            # remain decisive whether the receiver still consumes it or fences it.
            await self.original_inject("conversation.item.done", **self.boundary_fields)
            await self.settle_candidate()
        finally:
            self.runtime._closing = False
        await self.assert_fenced_without_replay()

    async def test_cancelled_possible_create_does_not_recover_on_late_exact_ack(self):
        await self.begin_candidate(ack=None, hold_send=True)
        bridge = self.runtime.coordinator._bridge
        jobs = tuple(bridge._jobs)
        self.assertTrue(jobs)
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        self.assertFalse(self.boundary_send_returned.is_set())
        await self.until(lambda: bridge.status.fenced)
        await self.exact_final_ack()
        await self.assert_fenced_without_replay()
