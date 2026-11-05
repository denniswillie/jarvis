"""Effect connection ownership and fail-closed mutations, using real repositories."""
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from app.live_state.models import SessionScope, SourceDependency
from app.memory.prospective.repository import ProspectiveRepository
from app.perception.applicability import PerceptionEffectGuard, PerceptionEffectUnavailable, effect_connection
from app.perception.repository import PerceptionRepository
from app.visual_memory.repository import VisualFrameRepository


class EffectGuardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="owner", session_id=uuid4())
        self.live = True
        self.original = uuid4()
        self.connection = SimpleNamespace(is_in_transaction=lambda: True,
            fetchrow=AsyncMock(return_value=None), fetchval=AsyncMock(return_value=False), execute=AsyncMock())
        self.events = []

        @asynccontextmanager
        async def transaction():
            self.events.append("transaction")
            yield

        self.connection.transaction = transaction

        @asynccontextmanager
        async def canonical(brief, binding, *, effect_payload_ids):
            self.events.append(("lock_original", effect_payload_ids))
            try:
                yield self.connection
            except BaseException:
                self.events.append("rollback")
                raise
            else:
                self.events.append("commit")

        self.guard = PerceptionEffectGuard(SimpleNamespace(dependencies=()), object(), canonical,
            lambda: self.live, self.original)
        self.pool = SimpleNamespace(acquire=Mock(side_effect=AssertionError("Must borrow the guarded connection")))

    async def test_same_connection_and_original_locks_survive_through_effect(self):
        async with effect_connection(self.pool, self.guard) as connection:
            self.assertIs(connection, self.connection)
            self.assertEqual(self.events, [("lock_original", (self.original,))])
            self.events.append("effect")
        self.assertEqual(self.events[-2:], ["effect", "commit"])
        self.pool.acquire.assert_not_called()

    async def test_close_during_effect_rolls_back_instead_of_committing(self):
        with self.assertRaises(PerceptionEffectUnavailable):
            async with self.guard():
                self.live = False
        self.assertEqual(self.events[-1], "rollback")

    async def test_missing_canonical_guard_never_grants_effect_permission(self):
        guard = PerceptionEffectGuard(self.guard.brief, object(), None, lambda: True, self.original)
        with self.assertRaises(PerceptionEffectUnavailable):
            async with guard():
                self.fail("Missing owner proof admitted a write")

    async def test_focus_fence_uses_existing_owner_generation_not_a_new_revision(self):
        source = SourceDependency(source_namespace="visual_focus_subject", source_id=uuid4(), revision=8, generation=8)
        guard = PerceptionEffectGuard(SimpleNamespace(dependencies=(source,)), object(), None,
            lambda: True, self.original)
        self.assertEqual(guard.focus_fence, {"subject_id": str(source.source_id), "generation": 8})
        self.assertIsNone(self.guard.focus_fence)

    async def test_working_memory_rejects_before_cas_and_without_another_pool_lease(self):
        self.live = False
        repository = PerceptionRepository(self.pool)
        with self.assertRaises(PerceptionEffectUnavailable):
            await repository.compare_and_swap_session_working_memory(owner_id="owner",
                session_id=self.scope.session_id, expected_revision=0, observation_id=uuid4(),
                state={}, expires_at=datetime.now(timezone.utc) + timedelta(seconds=10), effect_guard=self.guard)
        self.connection.fetchrow.assert_not_awaited()
        self.pool.acquire.assert_not_called()

    async def test_visual_alert_rejects_before_insert_or_attention_enqueue(self):
        self.live = False
        repository = VisualFrameRepository(self.pool)
        attention = SimpleNamespace(enqueue_visual=AsyncMock())
        with self.assertRaises(PerceptionEffectUnavailable):
            await repository.create_visual_interrupt(owner_id="owner", session_id=self.scope.session_id,
                visual_frame_id=uuid4(), message="fixture", reason="fixture", confidence=0.9,
                dedupe_bucket="fixture", effect_guard=self.guard, attention=attention)
        self.connection.fetchrow.assert_not_awaited()
        attention.enqueue_visual.assert_not_awaited()
        self.pool.acquire.assert_not_called()

    async def test_prospective_reservation_borrows_guarded_transaction(self):
        repository = ProspectiveRepository(self.pool)
        result = await repository.reserve_delivery_event(owner_id="owner", session_id=self.scope.session_id,
            subject_type="visual_focus", subject_id=uuid4(), claim_token=uuid4(), occurrence_key="fixture",
            message="fixture", evidence={}, dedupe_window_seconds=10, effect_guard=self.guard)
        self.assertIsNone(result)  # The fake canonical claim is not owned.
        self.connection.fetchval.assert_awaited_once()
        self.assertEqual(self.events[0], ("lock_original", (self.original,)))
        self.assertEqual(self.events[-1], "commit")
        self.pool.acquire.assert_not_called()

    async def test_prospective_reservation_rejects_before_claim_or_intent(self):
        self.live = False
        repository = ProspectiveRepository(self.pool)
        with self.assertRaises(PerceptionEffectUnavailable):
            await repository.reserve_delivery_event(owner_id="owner", session_id=self.scope.session_id,
                subject_type="visual_focus", subject_id=uuid4(), claim_token=uuid4(), occurrence_key="fixture",
                message="fixture", evidence={}, dedupe_window_seconds=10, effect_guard=self.guard)
        self.connection.fetchval.assert_not_awaited()
        self.connection.fetchrow.assert_not_awaited()
        self.pool.acquire.assert_not_called()
