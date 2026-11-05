"""Real session/route/binder composition with controlled storage and transport.

No SQL, provider, model, media, latency saving, or hearing qualification. Reuse
fixture helpers explicitly rather than inheriting/collecting their test cases.
"""
import asyncio
from dataclasses import is_dataclass, replace
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.realtime.live_state_bridge import select_voice_context
from app.voice_routes.coordinator import CoordinationPhase
import test_voice_context_response_pipeline as pipeline


def copied(value, **changes):
    return replace(value, **changes) if is_dataclass(value) else value.model_copy(update=changes)


class BackgroundPreparationTests(unittest.IsolatedAsyncioTestCase):
    cleanup = pipeline.VoiceContextResponsePipelineTests.cleanup
    until = pipeline.VoiceContextResponsePipelineTests.until
    rows = pipeline.VoiceContextResponsePipelineTests.rows
    observation = pipeline.VoiceContextResponsePipelineTests.observation
    expire_inside_adapter_guard = pipeline.VoiceContextResponsePipelineTests.expire_inside_adapter_guard
    mutations = pipeline.VoiceContextResponsePipelineTests.mutations
    creates = pipeline.VoiceContextResponsePipelineTests.creates
    question_and_accept = pipeline.VoiceContextResponsePipelineTests.question_and_accept

    async def asyncSetUp(self):
        await pipeline.VoiceContextResponsePipelineTests.asyncSetUp(self)
        await self.observation("stable-authorized-answer")
        await self.until(lambda: "stable-authorized-answer" in self.runtime.adapter._context.text)
        await self.question_and_accept()
        self.assertFalse(self.runtime.context_publication_idle)
        self.assertTrue(self.runtime.context_preparation_ready)
        self.real_bind = self.live.bind
        self.bind_calls = []
        self.bind_active = self.bind_max = 0
        self.before_bind = None

        async def counted(selection):
            self.bind_calls.append(selection)
            self.bind_active += 1
            self.bind_max = max(self.bind_max, self.bind_active)
            try:
                if self.before_bind is not None:
                    await self.before_bind(selection)
                return await self.real_bind(selection)
            finally:
                self.bind_active -= 1

        self.live.bind = counted
        self.baseline = list(self.provider.writes)

    async def prepared(self, text):
        await self.until(lambda: self.attachment._prepared_context is not None
            and text in self.attachment._prepared_context.update.text
            and self.bind_active == 0 and self.pool.entered == self.pool.exited
            and self.storage.entered == self.storage.exited)
        self.assertFalse(self.runtime.context_publication_idle)
        self.assertEqual(self.provider.writes, self.baseline)
        return self.attachment._prepared_context

    async def finish_answer(self):
        await self.provider.inject("response.output_audio_transcript.done", response_id="answer",
            item_id="answer-item", transcript="A controlled as-of answer.")
        await self.provider.inject("response.done", response={"id": "answer", "status": "completed",
            "output": [{"id": "answer-item", "type": "message", "phase": "final_answer",
                        "content": [{"type": "output_audio"}]}]})
        await self.until(lambda: any(event["type"] == "response.done" for event in self.app.events))
        self.assertFalse(self.runtime.context_publication_idle)
        await self.provider.inject("output_audio_buffer.stopped", response_id="answer")

    def context_texts(self):
        return [part.get("text", "") for event in self.mutations()
                if event["type"] == "conversation.item.create"
                for part in event["item"].get("content", [])]

    async def test_prepare_during_output_release_transaction_then_publish_once_despite_global_revision(self):
        await self.observation("semantic-unchanged-candidate")
        candidate = await self.prepared("semantic-unchanged-candidate")
        count = len(self.bind_calls)
        before = len(self.mutations())

        def unrelated_revision(snapshot, **kwargs):
            selection = select_voice_context(snapshot, **kwargs)
            return copied(selection, cursor=copied(selection.cursor,
                revision=selection.cursor.revision + 17))

        with patch("app.realtime.live_state_bridge.select_voice_context", unrelated_revision):
            for _ in range(3):
                await self.attachment.publish_latest()
            self.assertIs(self.attachment._prepared_context, candidate)
            self.assertEqual(len(self.bind_calls), count)
            self.assertEqual(self.provider.writes, self.baseline)
            await self.finish_answer()
            await self.until(lambda: self.runtime.adapter._context.update_id == candidate.update.update_id)
            self.assertEqual(len(self.bind_calls), count)
            self.assertEqual(len(self.mutations()), before + 2)
            self.assertIsNone(self.attachment._prepared_context)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)

    async def test_semantic_updates_during_one_binding_keep_only_newest_candidate(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def hold_first(selection):
            if "superseded-candidate" in selection.text:
                entered.set()
                await release.wait()

        self.before_bind = hold_first
        await self.observation("superseded-candidate")
        await asyncio.wait_for(entered.wait(), 3)
        await self.observation("intermediate-candidate")
        await self.observation("newest-candidate")
        await self.attachment.publish_latest()
        self.assertEqual(self.bind_active, 1)
        release.set()
        candidate = await self.prepared("newest-candidate")
        self.assertEqual(self.bind_max, 1)
        self.assertEqual(len(self.bind_calls), 2)
        await self.finish_answer()
        await self.until(lambda: self.runtime.adapter._context.update_id == candidate.update.update_id)
        self.assertFalse(any("superseded-candidate" in text or "intermediate-candidate" in text
                             for text in self.context_texts()))

    async def test_correction_drops_unexposed_candidate_and_publishes_new_binding(self):
        old = await self.observation("uncorrected-private-candidate")
        candidate = await self.prepared("uncorrected-private-candidate")
        self.storage.rejected.add(old.payload_id)
        await self.observation("corrected-current-candidate")
        corrected = await self.prepared("corrected-current-candidate")
        self.assertNotEqual(candidate.update.update_id, corrected.update.update_id)
        await self.finish_answer()
        await self.until(lambda: self.runtime.adapter._context.update_id == corrected.update.update_id)
        self.assertFalse(any("uncorrected-private-candidate" in text for text in self.context_texts()))

    async def test_canonical_deletion_after_preparation_still_fails_preflight_before_write(self):
        source = await self.observation("deleted-private-candidate")
        candidate = await self.prepared("deleted-private-candidate")
        before = len(self.mutations())
        previous = self.runtime.adapter._context
        history = set(self.session._source_context._sources)
        pin = self.session._active_responses["answer"].source_snapshot
        self.assertTrue(history)
        self.assertTrue(set(pin.sources).issubset(history))
        self.assertTrue(set(candidate.update.guard_binding.payload_root_ids).isdisjoint(history))
        # Prove this is an unexposed original, including transitive dependencies
        # of the actual provider history, not merely a different copied root.
        ancestors = set(history)
        for _ in range(len(self.storage.dependencies) + 1):
            for group in self.storage.dependencies:
                ancestors.update(parent for _, parent, child in group if child in ancestors)
        self.assertNotIn(source.payload_id, ancestors)
        with patch("app.realtime.live_state_bridge._VALIDITY_LOGGER.warning") as diagnostic:
            self.storage.rejected.add(source.payload_id)
            await self.finish_answer()
            await self.until(lambda: any(
                row["update_id"] == str(candidate.update.update_id)
                and row["code"] == "live_state_context_revoked"
                for row in self.rows(diagnostic, "validity_rejected"))
                and not self.runtime.coordinator._bridge._jobs
                and self.attachment._prepared_context is None)
        self.assertEqual(len(self.mutations()), before)
        self.assertIsNone(self.attachment._prepared_context)
        self.assertIs(self.runtime.adapter._context, previous)
        self.assertEqual(self.session._source_context._sources, history)
        async with self.session._source_context.context_guard():
            pass
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertFalse(self.session._context_revoked)
        self.assertFalse(self.session._closed.is_set())
        self.assertFalse(any("deleted-private-candidate" in text for text in self.context_texts()))

    async def test_previously_exposed_root_withdrawal_still_fences_accepted_output(self):
        pin = self.session._active_responses["answer"].source_snapshot
        self.assertTrue(pin.sources)
        exposed = pin.sources[0]
        self.assertIn(exposed, self.session._source_context._sources)
        await self.observation("unexposed-addition-before-real-withdrawal")
        candidate = await self.prepared("unexposed-addition-before-real-withdrawal")
        self.assertNotIn(exposed, candidate.update.guard_binding.payload_root_ids)
        before = len(self.mutations())
        self.storage.rejected.add(exposed)
        await self.provider.inject("response.output_audio_transcript.done", response_id="answer",
            item_id="answer-item", transcript="Must not be forwarded after original withdrawal.")
        await self.until(lambda: self.session._context_revoked)
        await self.until(lambda: self.session._closed.is_set())
        self.assertFalse(self.runtime.ready)
        self.assertFalse(any(event["type"] == "transcript.done" for event in self.app.events))
        self.assertFalse(any(call.kwargs["role"] == "assistant"
                             for call in self.repository.create_message.await_args_list))
        self.assertEqual(self.session._close_reason, "live_context_rebuild_required")
        self.assertEqual(len(self.mutations()), before)
        self.assertEqual(len(self.creates()), 1)

    async def test_canonical_owner_change_after_preparation_keeps_full_guard(self):
        await self.observation("wrong-owner-private-candidate")
        await self.prepared("wrong-owner-private-candidate")
        before = len(self.mutations())
        old_scope = self.binder.scope
        try:
            self.binder.scope = old_scope.model_copy(update={"owner_id": "other-offline-owner"})
            await self.finish_answer()
            await self.until(lambda: self.runtime.coordinator.status.phase == CoordinationPhase.DEGRADED
                             or self.session._closed.is_set())
            self.assertEqual(len(self.mutations()), before)
            self.assertIsNone(self.attachment._prepared_context)
        finally:
            self.binder.scope = old_scope

    async def test_route_generation_change_discards_prepared_identity_without_offer(self):
        await self.observation("old-route-private-candidate")
        await self.prepared("old-route-private-candidate")
        request = self.runtime.request
        try:
            self.runtime.request = request.model_copy(update={"route": request.route.model_copy(
                update={"route_generation": request.route.route_generation + 1})})
            await self.attachment.publish_latest()
            self.assertIsNone(self.attachment._prepared_context)
            self.assertEqual(self.attachment.last_omission, "route_identity_mismatch")
            self.assertEqual(self.provider.writes, self.baseline)
        finally:
            self.runtime.request = request

    async def test_expiry_drops_old_candidate_without_extending_its_freshness(self):
        await self.observation("short-lived-private-candidate", seconds=2)
        candidate = await self.prepared("short-lived-private-candidate")
        self.now = candidate.update.expires_at + timedelta(milliseconds=1)
        await self.attachment.publish_latest()
        self.assertIsNot(self.attachment._prepared_context, candidate)
        await self.finish_answer()
        await self.until(lambda: not self.runtime._answer_owns_context())
        await self.attachment.publish_latest()
        self.assertFalse(any("short-lived-private-candidate" in text for text in self.context_texts()))

    async def test_cancel_inside_binding_joins_transaction_and_never_publishes(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()
        target_roots = set()

        async def arm_before_canonical_binding(selection):
            if "cancelled-private-binding" in selection.text:
                target_roots.update(selection.parent_payload_root_ids)

        async def held_source_lock(roots):
            if target_roots.intersection(roots):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

        # Both hooks are installed before publication. The existing counted
        # binder invokes before_bind before entering the canonical transaction.
        self.before_bind = arm_before_canonical_binding
        self.storage.on_lock = held_source_lock
        await self.observation("cancelled-private-binding")
        await asyncio.wait_for(entered.wait(), 3)
        self.assertEqual(self.bind_active, 1)
        self.assertGreater(self.storage.entered, self.storage.exited)
        self.assertGreater(self.pool.entered, self.pool.exited)
        self.assertEqual(self.provider.writes, self.baseline)
        before = len(self.mutations())
        await asyncio.wait_for(self.attachment.close("offline-binding-cancel"), 5)
        self.assertTrue(cancelled.is_set())
        self.assertIsNone(self.attachment._prepared_context)
        self.assertEqual(self.bind_active, 0)
        self.assertEqual(self.pool.entered, self.pool.exited)
        self.assertEqual(self.storage.entered, self.storage.exited)
        self.assertEqual(len(self.mutations()), before)
        self.assertTrue(all(task.done() for task in self.attachment._tasks))

    async def test_close_during_idle_wait_drops_prepared_candidate(self):
        await self.observation("closed-private-candidate")
        await self.prepared("closed-private-candidate")
        before = len(self.mutations())
        await asyncio.wait_for(self.attachment.close("offline-prepared-close"), 5)
        self.assertIsNone(self.attachment._prepared_context)
        self.assertEqual(len(self.mutations()), before)
        self.assertEqual(self.pool.entered, self.pool.exited)
        self.assertTrue(all(task.done() for task in self.attachment._tasks))

    async def test_routes_without_explicit_preparation_seam_retain_idle_only_behavior(self):
        route = self.attachment.route
        self.attachment.route = SimpleNamespace(request=route.request, ready=True,
            context_publication_ready=True, context_publication_idle=False)
        try:
            await self.attachment.publish_latest()
            self.assertIsNone(self.attachment._prepared_context)
            self.assertEqual(self.bind_calls, [])
            self.assertEqual(self.provider.writes, self.baseline)
        finally:
            self.attachment.route = route
