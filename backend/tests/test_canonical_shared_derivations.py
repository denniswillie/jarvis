"""Shared derivation opt-in; generic effects retain exclusive ownership."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.live_state.canonical import CanonicalContextBinder, memory_dependency
from app.live_state.models import LiveSnapshot, SessionScope, StateCursor
from app.live_state.perception_context import compose_perception_brief
from app.live_state.perception_runtime import CanonicalPerceptionBinder, perception_brief_document
from app.live_state.projection import VoiceSelection
from test_live_state_canonical_adapters import NOW, memory, record
from test_session_perception_runtime import _CanonicalIO


class SharedDerivationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.scope = SessionScope(owner_id="shared-derivation-fixture", session_id=uuid4(), area_id=uuid4())
        self.io = _CanonicalIO(self.scope)
        self.binder = CanonicalContextBinder(self.io.sources, self.scope, clock=lambda: NOW)
        self.io.patch_io(self.binder)
        self.snapshot = LiveSnapshot(scope=self.scope, cursor=StateCursor(epoch=uuid4(), revision=1),
                                    as_of=NOW, readiness="ready")

    def original_record(self):
        value = record(self.scope)
        self.io.records[value.id] = value
        self.io.roots.add(value.payload_id)
        return value

    def original_memory(self):
        value = memory(self.scope)
        self.io.memories[value.source_id] = value
        self.io.roots.add(value.payload_id)
        return value

    def assert_shared(self, count=1):
        self.binder.tasks.lock_owner.assert_not_awaited()
        self.assertEqual(self.binder.tasks.lock_owner_shared.await_count, count)

    async def test_record_and_combined_copies_use_shared_ownership(self):
        binding = await self.binder.bind_record(self.original_record(), "Synthetic exact record")
        self.assert_shared()
        selection = VoiceSelection(self.scope, self.snapshot.cursor, "Synthetic combined context",
            binding.dependencies, binding.payload_root_ids, ("record",), (), None)
        combined = await self.binder(selection)
        self.assert_shared(2)
        self.assertEqual(combined.dependencies, binding.dependencies)
        self.assertNotEqual(combined.payload_root_ids, binding.payload_root_ids)

    async def test_memory_provenance_does_not_upgrade_shared_task_lock(self):
        original = self.original_memory()
        binding = await self.binder.bind_memory(original, "Synthetic selected memory")
        self.assert_shared(2)  # Outer ownership and provenance's compatible lock.
        self.assertEqual(binding.dependencies, (memory_dependency(original),))
        self.assertTrue(self.io.connection.executemany.await_count)

    async def test_memory_brief_derivation_is_shared_but_effect_guard_is_exclusive(self):
        original = self.original_memory()
        brief = compose_perception_brief(self.snapshot, now=NOW,
            scope_epoch=self.snapshot.cursor.epoch, context_dependencies=(memory_dependency(original),),
            dependency_eligible=lambda source: True).brief
        perception = CanonicalPerceptionBinder(self.binder)
        binding = await perception.bind(brief)
        self.assert_shared(2)
        previous = self.io.fetchval

        async def fetchval(query, *args):
            if query.startswith("select content from jarvis_context_payloads"):
                return {"format": "jarvis.perception_brief.v1", "brief": perception_brief_document(brief)}
            return await previous(query, *args)

        self.io.connection.fetchval.side_effect = fetchval
        async with perception.guard(brief, binding):
            pass
        self.binder.tasks.lock_owner.assert_awaited_once()
        self.assertEqual(self.binder.tasks.lock_owner_shared.await_count, 2)

    async def test_generic_provenance_callback_remains_exclusive(self):
        materialize = AsyncMock(return_value=("value", uuid4()))
        result = await self.binder.provenance.bind(self.scope.owner_id, self.scope.area_id, [], materialize)
        self.assertEqual(result, "value")
        self.binder.tasks.lock_owner.assert_awaited_once()
        self.binder.tasks.lock_owner_shared.assert_not_awaited()

    async def test_legacy_source_adapter_keeps_exclusive_derivation(self):
        del self.io.sources.derivation_transaction
        await self.binder.bind_record(self.original_record(), "Legacy adapter copy")
        self.binder.tasks.lock_owner.assert_awaited_once()
        self.binder.tasks.lock_owner_shared.assert_not_awaited()

    async def test_invalid_transaction_modes_are_rejected_before_storage(self):
        for options in ({"derivation": 1}, {"derivation": True, "read_only_guard": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                async with self.binder._transaction(**options):
                    self.fail("Invalid mode entered storage")
        self.assertEqual(self.io.transactions, 0)
        with self.assertRaises(ValueError):
            await self.binder.provenance.bind_in_transaction(self.io.connection,
                self.scope.owner_id, self.scope.area_id, [], AsyncMock(), shared_task_owner=1)
