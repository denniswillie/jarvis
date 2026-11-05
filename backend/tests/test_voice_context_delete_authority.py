"""Real session/writer/bridge composition with controlled storage and socket only.

No PostgreSQL, provider, native timing, playback or hearing qualification. Helpers
are reused without inheriting a TestCase, avoiding duplicate test collection.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.knowledge.sources import SourceRepository
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.voice_routes.contracts import ContextWriteEvidence
from app.voice_routes.coordinator import CoordinationPhase
import test_voice_context_response_pipeline as fixtures
from test_realtime_voice_route import make_context


class ContextDeleteAuthorityTests(unittest.IsolatedAsyncioTestCase):
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
        await self.until(lambda: self.runtime.ready and not self.bridge._jobs
                         and self.bridge._pending is None)
        # Exercise the REAL memory-owner transaction/diagnostic seam, with only
        # its connection and pool controlled. The canonical binder is unchanged.
        sources = self.binder.sources
        sources.transaction = lambda owner, **options: SourceRepository.transaction(sources, owner, **options)
        self.bridge._timeout = 0.2  # Controlled test bound, not a production change.

    async def settled(self):
        await self.until(lambda: self.bridge.status.fenced or (
            self.bridge.awaiting_fresh_context and not self.bridge._jobs
            and not self.bridge.publication_in_flight))

    async def test_scope_only_predelete_keeps_two_full_guards_and_no_lock_across_ack(self):
        original_guard, original_scope = self.live.guard, self.live.scope_guard
        full, scopes, delete_states = [], [], []
        previous_roots = set(self.session._source_context._sources)

        @asynccontextmanager
        async def guard(binding, **options):
            full.append(binding)
            async with original_guard(binding, **options) as connection:
                yield connection

        @asynccontextmanager
        async def scope(**options):
            scopes.append(True)
            async with original_scope(**options) as connection:
                yield connection

        self.live.guard, self.live.scope_guard = guard, scope
        original_inject = self.provider.inject

        async def inject(kind, **fields):
            if kind == "conversation.item.deleted":
                delete_states.append(self.storage.is_in_transaction())
            await original_inject(kind, **fields)

        self.provider.inject = inject
        await self.observation("scope-count-candidate")
        await self.until(lambda: self.runtime.ready and not self.bridge._jobs
                         and "scope-count-candidate" in self.runtime.adapter._context.text)
        self.assertEqual(len(full), 2)  # Full preflight and actual create, not pre-delete.
        self.assertEqual(len(scopes), 1)
        self.assertEqual(delete_states, [False])
        self.assertEqual(full[0], full[1])
        new_roots = set(self.runtime.adapter._context.guard_binding.payload_root_ids)
        self.assertTrue(previous_roots)
        self.assertIn(tuple(sorted(previous_roots | new_roots)), self.storage.locks)

    async def test_real_bridge_preflight_rejects_new_source_before_any_delete(self):
        before = len(self.mutations())
        old_item = self.runtime.adapter._context_item
        old_roots = set(self.session._source_context._sources)
        old_source_ids = {str(key[0]) for key in self.records}

        async def reject_candidate(roots):
            # Binding first materializes the new copy. Reject that copy at the
            # actual canonical preflight lock, not the previous authorized one.
            for identity, body in self.storage.payloads:
                if (identity in roots and body.get("format") == "jarvis.live_state.voice.v1"
                        and any(row["source_id"] not in old_source_ids
                                for row in body["dependencies"])):
                    self.storage.rejected.add(identity)

        self.storage.on_lock = reject_candidate
        await self.observation("preflight-rejected-candidate")
        await self.until(lambda: self.bridge.status.fenced or
                         self.bridge.status.reason == "context_publication_deferred_unwritten")
        self.assertEqual(self.bridge.status.reason, "context_publication_deferred_unwritten")
        self.assertEqual(len(self.mutations()), before)
        self.assertEqual(self.runtime.adapter._context_item, old_item)
        self.assertEqual(self.session._source_context._sources, old_roots)
        update_id = str(self.bridge._latest.update_id)
        self.assertTrue(any(row.get("update_id") == update_id and row["outcome"] == "invalid"
                            for row in self.rows(self.publication, "validity_exit")))
        self.assertFalse(any(row.get("update_id") == update_id
                             for row in self.rows(self.publication, "adapter_entry")))

    async def test_outer_deadline_in_memory_owner_lock_holds_empty_then_fresh_recovers(self):
        old_item = self.runtime.adapter._context_item
        before = len(self.mutations())
        entered, released = asyncio.Event(), asyncio.Event()
        fetchval = self.storage.fetchval

        async def blocked(query, *args):
            if "jarvis-memory-owner:" in query and self.runtime.adapter._context_item is None:
                entered.set()
                await released.wait()
            return await fetchval(query, *args)

        self.storage.fetchval = blocked
        with patch("app.realtime.telemetry.LOGGER.info") as trace:
            await self.observation("deadline-candidate")
            await asyncio.wait_for(entered.wait(), 1)
            await self.settled()
        diagnostics = [json.loads(call.args[1]) for call in trace.call_args_list
                       if len(call.args) > 1 and isinstance(call.args[1], str)]
        waits = [row for row in diagnostics if row.get("event") == "provider_guard_wait_cancelled"]
        self.assertEqual(len(waits), 1)
        self.assertEqual(waits[0]["phase"], "memory_owner_lock")
        self.assertIs(waits[0]["write_possible"], False)
        self.assertIs(waits[0]["send_returned"], False)
        returned = self.rows(self.publication, "adapter_return")[-1]
        self.assertTrue(returned["owned_work_joined"])
        self.assertTrue(returned["deleted_context_acknowledged"])
        self.assertFalse(returned["replacement_create_possible"])
        self.assertEqual(self.bridge.status.reason, "context_empty_awaiting_snapshot")
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.ACTIVE)
        self.assertFalse(self.runtime.ready)
        self.assertFalse(await self.runtime.wait_for_context_ready(0.01))
        self.assertEqual([row["type"] for row in self.mutations()[before:]], ["conversation.item.delete"])
        self.assertIsNone(self.bridge.status.retry_update_id)
        stale = self.bridge._latest
        self.assertEqual(self.runtime.offer_context(stale).disposition.value, "duplicate")
        await self.provider.inject("conversation.item.deleted", item_id=old_item)
        await self.provider.inject("conversation.item.deleted", item_id="foreign-old-item")
        self.assertEqual(len(self.mutations()), before + 1)
        released.set()
        await self.observation("fresh-authorized-after-deadline")
        await self.until(lambda: self.runtime.ready and not self.bridge._jobs
                         and "fresh-authorized-after-deadline" in self.runtime.adapter._context.text)
        self.assertEqual([row["type"] for row in self.mutations()[before:]],
                         ["conversation.item.delete", "conversation.item.create"])
        self.assertNotIn("deadline-candidate", self.runtime.adapter._context.text)
        await self.question_and_accept()
        self.assertEqual(len(self.creates()), 1)

    async def test_missing_and_foreign_delete_ack_remain_unknown(self):
        before = len(self.mutations())
        send = self.provider.send

        async def missing_ack(raw):
            event = json.loads(raw)
            if event["type"] == "conversation.item.delete":
                self.provider.writes.append(event)
                await self.provider.inject("conversation.item.deleted", item_id="foreign-item")
                return
            await send(raw)

        self.provider.send = missing_ack
        await self.observation("missing-delete-ack")
        await self.until(lambda: self.bridge.status.fenced)
        self.assertEqual(self.runtime.coordinator.status.phase, CoordinationPhase.DEGRADED)
        self.assertEqual([row["type"] for row in self.mutations()[before:]], ["conversation.item.delete"])
        self.assertEqual(self.rows(self.publication, "context_refresh_wait"), [])
        self.assertIsNone(self.bridge.status.retry_update_id)

    async def test_possible_replacement_create_is_not_known_empty(self):
        before = len(self.mutations())
        send = self.provider.send

        async def uncertain_create(raw):
            event = json.loads(raw)
            if event["type"] == "conversation.item.create":
                self.provider.writes.append(event)
                await asyncio.Event().wait()
            else:
                await send(raw)

        self.provider.send = uncertain_create
        await self.observation("possible-create")
        await self.until(lambda: self.bridge.status.fenced)
        row = self.rows(self.publication, "adapter_return")[-1]
        self.assertTrue(row["deleted_context_acknowledged"])
        self.assertTrue(row["replacement_create_possible"])
        self.assertEqual(self.rows(self.publication, "context_refresh_wait"), [])
        self.assertEqual([row["type"] for row in self.mutations()[before:]],
                         ["conversation.item.delete", "conversation.item.create"])

    async def test_scope_change_in_final_guard_still_fences_after_delete(self):
        original = self.binder._session

        async def selected_scope(connection):
            await original(connection)
            if self.runtime.adapter._context_item is None:
                raise SourceError("live_state_scope_changed", "controlled scope withdrawal")

        self.binder._session = selected_scope
        before = len(self.mutations())
        await self.observation("scope-withdrawal")
        await self.until(lambda: self.bridge.status.fenced)
        self.assertEqual(self.rows(self.publication, "context_refresh_wait"), [])
        self.assertEqual([row["type"] for row in self.mutations()[before:]], ["conversation.item.delete"])
        self.assertFalse(self.runtime.ready)

    async def test_previously_exposed_source_withdrawal_still_fences_final_create(self):
        previous = set(self.session._source_context._sources)
        self.assertTrue(previous)
        original = self.provider.inject

        async def withdraw(kind, **fields):
            if kind == "conversation.item.deleted":
                self.storage.rejected.update(previous)
            await original(kind, **fields)

        self.provider.inject = withdraw
        before = len(self.mutations())
        await self.observation("new-content-does-not-erase-old-authority")
        await self.until(lambda: self.bridge.status.fenced or self.session._context_revoked)
        self.assertEqual([row["type"] for row in self.mutations()[before:]], ["conversation.item.delete"])
        self.assertEqual(self.rows(self.publication, "context_refresh_wait"), [])
        self.assertFalse(self.runtime.ready)

    async def test_unjoined_cancelled_work_never_enters_empty_hold(self):
        release, entered = asyncio.Event(), asyncio.Event()
        fetchval = self.storage.fetchval

        async def uncooperative(query, *args):
            if "jarvis-memory-owner:" in query and self.runtime.adapter._context_item is None:
                entered.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        pass
            return await fetchval(query, *args)

        self.storage.fetchval = uncooperative
        try:
            await self.observation("unjoined-create-guard")
            await asyncio.wait_for(entered.wait(), 1)
            await self.until(lambda: self.bridge.status.fenced)
            self.assertEqual(self.rows(self.publication, "context_refresh_wait"), [])
            self.assertFalse(self.runtime.ready)
        finally:
            release.set()
        await self.until(lambda: not self.runtime.adapter._data_tasks and not self.bridge._jobs)


class PredeleteEligibilityTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self):
        runtime = object.__new__(RealtimeVoiceRouteRuntime)
        update = make_context(revision=1)
        runtime.request = SimpleNamespace(route=update.route)
        runtime._closing = runtime._closed = False
        runtime.session = SimpleNamespace(_closed=asyncio.Event(), _context_revoked=False,
            _pending_response=None, _active_responses={}, _direct_output_pending=set(), _native_playback={})
        runtime._context_replacement_active = False
        runtime._authorize_start = AsyncMock(return_value=True)
        return runtime, update

    async def test_local_fences_are_rechecked_after_authorization(self):
        for flag in ("closing", "closed", "session_closed", "revoked", "answer", "route_generation"):
            with self.subTest(flag=flag):
                runtime, update = self.runtime()

                async def authorize(request):
                    self.assertIs(request, runtime.request)
                    if flag == "closing":
                        runtime._closing = True
                    elif flag == "closed":
                        runtime._closed = True
                    elif flag == "session_closed":
                        runtime.session._closed.set()
                    elif flag == "revoked":
                        runtime.session._context_revoked = True
                    elif flag == "answer":
                        runtime.session._active_responses["answer"] = object()
                    else:
                        runtime.request = SimpleNamespace(route=update.route.model_copy(update={
                            "route_generation": update.route.route_generation + 1}))
                    return True

                runtime._authorize_start = authorize
                async with runtime._context_replacement_guard(update) as admitted:
                    self.assertIs(admitted, False)
                self.assertFalse(runtime._context_replacement_active)
                self.assertTrue(runtime._context_replacement_settled.is_set())

    async def test_wrong_generation_never_calls_scope_authorizer(self):
        runtime, update = self.runtime()
        wrong = update.model_copy(update={"route": update.route.model_copy(update={"route_generation": 9})})
        async with runtime._context_replacement_guard(wrong) as admitted:
            self.assertIs(admitted, False)
        runtime._authorize_start.assert_not_awaited()

    async def test_authority_failure_propagates_without_candidate_validation(self):
        runtime, update = self.runtime()
        runtime._authorize_start.side_effect = SourceError("live_state_scope_changed", "private")
        runtime._source_guard = AsyncMock()
        with self.assertRaises(SourceError):
            async with runtime._context_replacement_guard(update):
                self.fail("Scope failure cannot authorize deletion")
        runtime._source_guard.assert_not_called()
        self.assertFalse(runtime._context_replacement_active)


if __name__ == "__main__":
    unittest.main()
