"""Real context/session/writer composition with controlled lock results only."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.knowledge.models import CanonicalOwnerBusy
from app.knowledge.sources import SourceRepository
from app.voice_routes.coordinator import CoordinationPhase
import test_voice_context_response_pipeline as fixtures
import test_voice_context_delete_authority as authority_fixtures


class OwnerBusyContextTests(unittest.IsolatedAsyncioTestCase):
    cleanup = fixtures.VoiceContextResponsePipelineTests.cleanup
    until = fixtures.VoiceContextResponsePipelineTests.until
    rows = fixtures.VoiceContextResponsePipelineTests.rows
    observation = fixtures.VoiceContextResponsePipelineTests.observation
    mutations = fixtures.VoiceContextResponsePipelineTests.mutations
    creates = fixtures.VoiceContextResponsePipelineTests.creates
    question_and_accept = fixtures.VoiceContextResponsePipelineTests.question_and_accept
    expire_inside_adapter_guard = fixtures.VoiceContextResponsePipelineTests.expire_inside_adapter_guard

    async def asyncSetUp(self):
        await fixtures.VoiceContextResponsePipelineTests.asyncSetUp(self)
        self.storage.on_lock = None
        self.bridge = self.runtime.coordinator._bridge
        await self.until(lambda: self.runtime.ready and not self.bridge._jobs and self.bridge._pending is None)
        sources = self.binder.sources
        sources.transaction = lambda owner, **options: SourceRepository.transaction(sources, owner, **options)
        sources.read_guard_transaction = lambda owner, **options: SourceRepository.read_guard_transaction(sources, owner, **options)
        self.busy = False
        self.after_delete = False
        self.lock_kind = "memory"
        self.try_calls = 0
        original = self.storage.fetchval

        async def fetchval(query, *args):
            if "pg_try_advisory_xact_lock" in query:
                self.try_calls += 1
                if (self.busy and f":jarvis-{self.lock_kind}-owner:" in query
                        and (not self.after_delete or self.runtime.adapter._context_item is None)):
                    return False
            return await original(query, *args)

        self.storage.fetchval = fetchval

    async def held(self):
        await self.until(lambda: self.bridge.status.fenced or (
            self.bridge.status.reason == "context_owner_busy_awaiting_snapshot" and not self.bridge._jobs))
        self.assertFalse(self.bridge.status.fenced)
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertFalse(self.runtime.ready)
        self.assertFalse(await self.runtime.wait_for_context_ready(0.01))

    async def recovered(self, summary="fresh-after-busy"):
        self.busy = False
        await self.observation(summary)
        await self.until(lambda: self.runtime.ready and not self.bridge._jobs
                         and summary in self.runtime.adapter._context.text)

    async def test_memory_contention_before_delete_keeps_remote_item_but_pauses_speech(self):
        old_item = self.runtime.adapter._context_item
        before = len(self.mutations())
        self.busy = True
        await self.observation("blocked-before-delete")
        await self.held()
        self.assertEqual(self.runtime.adapter._context_item, old_item)
        self.assertEqual(len(self.mutations()), before)
        await self.recovered()
        self.assertEqual([row["type"] for row in self.mutations()[before:]],
                         ["conversation.item.delete", "conversation.item.create"])
        await self.question_and_accept()
        self.assertEqual(len(self.creates()), 1)

    async def test_unchanged_context_recovers_after_more_than_write_retry_budget(self):
        before = len(self.mutations())
        self.busy = True
        with patch("app.realtime.live_state_bridge.monotonic", return_value=100.0) as clock:
            await self.observation("unchanged-through-long-lock")
            await self.held()
            cursor = self.service.snapshot().cursor
            await self.attachment.publish_latest()
            for tick in (101.0, 102.0, 103.0, 104.0):
                calls = self.try_calls
                clock.return_value = tick
                await self.attachment.publish_latest()
                await self.until(lambda: self.try_calls > calls and not self.bridge._jobs)
                self.assertFalse(self.runtime.ready)
                self.assertEqual(len(self.mutations()), before)
            self.busy = False
            clock.return_value = 105.0
            await self.attachment.publish_latest()
            await self.until(lambda: self.runtime.ready and not self.bridge._jobs)
        self.assertEqual(self.service.snapshot().cursor, cursor)
        self.assertIn("unchanged-through-long-lock", self.runtime.adapter._context.text)
        self.assertEqual([row["type"] for row in self.mutations()[before:]],
                         ["conversation.item.delete", "conversation.item.create"])
        await self.question_and_accept()
        self.assertEqual(len(self.creates()), 1)

    async def test_task_contention_after_delete_needs_new_guarded_context_without_redelete(self):
        self.busy = self.after_delete = True
        self.lock_kind = "task"
        before = len(self.mutations())
        await self.observation("blocked-after-delete")
        await self.held()
        self.assertIsNone(self.runtime.adapter._context_item)
        self.assertIsNone(self.bridge.status.retry_update_id)
        self.assertEqual([row["type"] for row in self.mutations()[before:]], ["conversation.item.delete"])
        await self.recovered()
        self.assertEqual([row["type"] for row in self.mutations()[before:]],
                         ["conversation.item.delete", "conversation.item.create"])

    async def test_confirmed_empty_slot_recovers_from_newer_identical_selection(self):
        self.busy = self.after_delete = True
        self.lock_kind = "task"
        before = len(self.mutations())
        await self.observation("same-selection-after-confirmed-delete")
        await self.held()
        previous = self.attachment._last_update
        self.assertIsNone(self.runtime.adapter._context_item)
        self.assertIsNone(self.bridge.status.retry_update_id)
        self.busy = False
        self.service.mark_ready()
        await self.until(lambda: self.runtime.ready and not self.bridge._jobs)
        current = self.runtime.adapter._context
        self.assertEqual(current.text, previous.text)
        self.assertEqual(current.dependencies, previous.dependencies)
        self.assertGreater(current.cursor.revision, previous.cursor.revision)
        self.assertNotEqual(current.update_id, previous.update_id)
        self.assertEqual([row["type"] for row in self.mutations()[before:]],
                         ["conversation.item.delete", "conversation.item.create"])

    async def test_identical_fresh_selection_still_rejects_withdrawn_roots(self):
        roots = set(self.session._source_context._sources)
        self.busy = self.after_delete = True
        self.lock_kind = "task"
        before = len(self.mutations())
        await self.observation("same-selection-cannot-erase-withdrawal")
        await self.held()
        self.storage.rejected.update(roots)
        self.busy = False
        self.service.mark_ready()
        await self.until(lambda: self.bridge.status.fenced or self.session._context_revoked)
        self.assertFalse(self.runtime.ready)
        self.assertFalse(any(row["type"] == "conversation.item.create"
                             for row in self.mutations()[before:]))

    async def test_contention_in_predelete_authorization_preserves_unwritten_proof(self):
        before = len(self.mutations())
        authorize = self.runtime._authorize_context_replacement

        async def busy(candidate):
            raise CanonicalOwnerBusy()

        self.runtime._authorize_context_replacement = busy
        await self.observation("busy-between-preflight-and-delete")
        await self.held()
        self.assertEqual(len(self.mutations()), before)
        self.runtime._authorize_context_replacement = authorize
        await self.recovered()
        self.assertEqual([row["type"] for row in self.mutations()[before:]],
                         ["conversation.item.delete", "conversation.item.create"])

    async def test_source_withdrawal_during_busy_hold_does_not_become_authorized(self):
        before = len(self.mutations())
        previous = set(self.session._source_context._sources)
        self.assertTrue(previous)
        self.busy = True
        await self.observation("busy-before-withdrawal")
        await self.held()
        self.storage.rejected.update(previous)
        self.busy = False
        await self.observation("new-context-cannot-erase-withdrawal")
        await self.until(lambda: self.bridge.status.fenced or self.session._context_revoked)
        self.assertFalse(self.runtime.ready)
        self.assertFalse(any(row["type"] == "conversation.item.create" for row in self.mutations()[before:]))

    async def test_possible_create_cannot_be_reclassified_as_owner_busy(self):
        before = len(self.mutations())
        send = self.provider.send

        async def uncertain(raw):
            event = json.loads(raw)
            if event["type"] == "conversation.item.create":
                self.provider.writes.append(event)
                raise CanonicalOwnerBusy()
            await send(raw)

        self.provider.send = uncertain
        await self.observation("create-already-possible")
        await self.until(lambda: self.bridge.status.fenced)
        self.assertFalse(self.runtime.ready)
        self.assertEqual([row["type"] for row in self.mutations()[before:]],
                         ["conversation.item.delete", "conversation.item.create"])
        self.assertNotEqual(self.bridge.status.reason, "context_owner_busy_awaiting_snapshot")


class StartupOwnerPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_context_uses_start_authorizer_not_nonblocking_replacement(self):
        runtime, update = authority_fixtures.PredeleteEligibilityTests.runtime(self)
        runtime.request.initial_context = update
        runtime._authorize_context_replacement = AsyncMock(side_effect=AssertionError("Not a replacement"))
        async with runtime._context_replacement_guard(update) as admitted:
            self.assertIs(admitted, True)
        runtime._authorize_start.assert_awaited_once_with(runtime.request)
        runtime._authorize_context_replacement.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
