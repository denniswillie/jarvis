from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from app.memory.embeddings import PreparedMemoryEmbedding
from app.memory.embeddings.job_handlers import EmbedMemoryItemJobHandler
from app.memory.entities.models import MemoryEntity
from app.memory.jobs.handlers import MemoryJobValidationError
from app.memory.models import MemoryItem
from app.memory.semantic.contracts import semantic_reconciliation_dedupe_key
from app.memory.semantic.job_handler import (
    PreparedSemanticReconciliation,
    ReconcileSemanticMemoryJobHandler,
)


class SemanticJobAtomicityTests(unittest.IsolatedAsyncioTestCase):
    async def test_created_state_enqueues_all_embeddings_on_commit_connection(self) -> None:
        owner_id = "owner"
        session_id = uuid4()
        event_id = uuid4()
        episode_id = uuid4()
        now = datetime.now(UTC)
        entity = MemoryEntity(
            id=uuid4(),
            owner_id=owner_id,
            entity_type="object",
            canonical_name="Silver keys",
            normalized_name="silver keys",
            aliases=(),
            description="",
            confidence=0.95,
            external_namespace=None,
            external_id=None,
            metadata={},
            created_at=now,
            updated_at=now,
        )
        fact = _fact(session_id=session_id, event_id=event_id, now=now)
        context = SimpleNamespace(
            owner_id=owner_id,
            session_id=session_id,
            event_id=event_id,
            episode_memory_id=episode_id,
        )
        repository = AsyncMock()
        repository.apply_plan.return_value = SimpleNamespace(
            created_entities=(entity,),
            created_facts=(fact,),
            updated_facts=(),
            created_relation_count=0,
            already_reconciled=False,
        )
        jobs = AsyncMock()
        handler = ReconcileSemanticMemoryJobHandler(
            repository=repository,
            reconciler=AsyncMock(),
            job_repository=jobs,
            embedding_model=SimpleNamespace(provider="gemini", model="embedding"),
        )
        connection = AsyncMock()
        job = SimpleNamespace(
            owner_id=owner_id,
            session_id=session_id,
            payload={"event_id": str(event_id)},
        )

        await handler.commit(
            job,
            PreparedSemanticReconciliation(
                context=context,
                plan=SimpleNamespace(),
                entity_embeddings={},
            ),
            connection,
        )

        self.assertEqual(jobs.enqueue.await_count, 2)
        self.assertEqual(
            {call.kwargs["job_type"] for call in jobs.enqueue.await_args_list},
            {"embed_memory_entity", "embed_memory_item"},
        )
        self.assertTrue(
            all(
                call.kwargs["connection"] is connection
                for call in jobs.enqueue.await_args_list
            )
        )
        self.assertTrue(
            all(
                call.kwargs["session_id"] == session_id
                for call in jobs.enqueue.await_args_list
            )
        )


class SemanticFactEmbeddingAttributionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.origin_session_id = uuid4()
        self.fact = _fact(
            session_id=self.origin_session_id,
            event_id=uuid4(),
            now=datetime.now(UTC),
        )

    async def test_reinforcement_keeps_original_session_and_dedupe_key(self) -> None:
        initial = await self._enqueue_fact(self.fact, self.origin_session_id)
        reinforced = await self._enqueue_fact(self.fact, uuid4())

        self.assertEqual(reinforced["session_id"], self.origin_session_id)
        self.assertEqual(reinforced["dedupe_key"], initial["dedupe_key"])
        self.assertEqual(reinforced["payload"], initial["payload"])

    async def test_changed_content_has_new_key_but_same_origin_session(self) -> None:
        initial = await self._enqueue_fact(self.fact, self.origin_session_id)
        changed = await self._enqueue_fact(replace(self.fact, confidence=0.99), uuid4())

        self.assertEqual(changed["session_id"], self.origin_session_id)
        self.assertNotEqual(changed["dedupe_key"], initial["dedupe_key"])

    async def test_owner_level_fact_keeps_null_session(self) -> None:
        enqueued = await self._enqueue_fact(replace(self.fact, session_id=None), uuid4())

        self.assertIsNone(enqueued["session_id"])

    async def _enqueue_fact(self, fact: MemoryItem, session_id: UUID) -> dict:
        context = SimpleNamespace(
            owner_id=fact.owner_id,
            session_id=session_id,
            event_id=uuid4(),
            episode_memory_id=uuid4(),
        )
        repository = AsyncMock()
        repository.apply_plan.return_value = SimpleNamespace(
            created_entities=(),
            created_facts=(),
            updated_facts=(fact,),
            created_relation_count=0,
            already_reconciled=False,
        )
        jobs = AsyncMock()
        connection = AsyncMock()
        handler = ReconcileSemanticMemoryJobHandler(
            repository=repository,
            reconciler=AsyncMock(),
            job_repository=jobs,
            embedding_model=SimpleNamespace(provider="gemini", model="embedding"),
        )
        await handler.commit(
            SimpleNamespace(
                owner_id=fact.owner_id,
                session_id=session_id,
                payload={"event_id": str(context.event_id)},
            ),
            PreparedSemanticReconciliation(
                context=context,
                plan=SimpleNamespace(),
                entity_embeddings={},
            ),
            connection,
        )

        jobs.enqueue.assert_awaited_once()
        enqueued = jobs.enqueue.await_args.kwargs
        self.assertIs(enqueued["connection"], connection)
        self.assertEqual(enqueued["owner_id"], fact.owner_id)
        self.assertEqual(enqueued["job_type"], "embed_memory_item")
        self.assertEqual(enqueued["payload"], {"memory_item_id": str(fact.id)})
        return enqueued


class MemoryEmbeddingFollowupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.memory_id = uuid4()
        self.repository = AsyncMock()
        self.jobs = AsyncMock()
        self.connection = AsyncMock()
        self.handler = EmbedMemoryItemJobHandler(
            repository=self.repository,
            service=AsyncMock(),
            job_repository=self.jobs,
        )
        self.prepared = PreparedMemoryEmbedding(
            memory_item_id=self.memory_id,
            owner_id="owner",
            existing=None,
            write=None,
        )

    async def test_followup_uses_event_session_not_embedding_job_session(self) -> None:
        event_id = uuid4()
        event_session_id = uuid4()
        self.connection.fetchrow.return_value = {
            "id": event_id,
            "session_id": event_session_id,
        }
        for job_session_id in (uuid4(), None):
            with self.subTest(job_session_id=job_session_id):
                self.jobs.reset_mock()
                self.connection.reset_mock()
                result = await self.handler.commit(
                    self._job(job_session_id), self.prepared, self.connection
                )

                self.jobs.enqueue.assert_awaited_once()
                enqueued = self.jobs.enqueue.await_args.kwargs
                self.assertEqual(enqueued["owner_id"], "owner")
                self.assertEqual(enqueued["session_id"], event_session_id)
                self.assertEqual(enqueued["job_type"], "reconcile_semantic_memory")
                self.assertEqual(
                    enqueued["dedupe_key"], semantic_reconciliation_dedupe_key(event_id)
                )
                self.assertEqual(enqueued["payload"], {"event_id": str(event_id)})
                self.assertIs(enqueued["connection"], self.connection)
                self.assertEqual(result["semantic_event_id"], str(event_id))
                self.assertEqual(
                    self.connection.fetchrow.await_args.args[1:],
                    ("owner", self.memory_id),
                )

    async def test_no_episode_followup_does_not_require_a_session(self) -> None:
        self.connection.fetchrow.return_value = None

        result = await self.handler.commit(self._job(None), self.prepared, self.connection)

        self.jobs.enqueue.assert_not_awaited()
        self.assertEqual(
            result,
            {
                "memory_item_id": str(self.memory_id),
                "embedding_written": False,
                "semantic_event_id": None,
            },
        )

    async def test_prepared_owner_and_memory_identity_are_still_enforced(self) -> None:
        for prepared in (
            replace(self.prepared, owner_id="other-owner"),
            replace(self.prepared, memory_item_id=uuid4()),
        ):
            with self.subTest(prepared=prepared):
                with self.assertRaises(MemoryJobValidationError):
                    await self.handler.commit(self._job(None), prepared, self.connection)

        self.repository.store_memory_embedding.assert_not_awaited()
        self.connection.fetchrow.assert_not_awaited()
        self.jobs.enqueue.assert_not_awaited()

    def _job(self, session_id: UUID | None) -> SimpleNamespace:
        return SimpleNamespace(
            owner_id="owner",
            session_id=session_id,
            payload={"memory_item_id": str(self.memory_id)},
        )


def _fact(*, session_id: UUID | None, event_id: UUID, now: datetime) -> MemoryItem:
    return MemoryItem(
        id=uuid4(),
        owner_id="owner",
        session_id=session_id,
        memory_type="semantic_fact",
        status="active",
        title="Silver keys located on the counter",
        content="Silver keys located on the counter",
        summary="Silver keys located on the counter",
        source_experience_event_id=event_id,
        occurred_at=now,
        valid_from=now,
        valid_to=None,
        importance=0.9,
        confidence=0.95,
        source="user_statement",
        is_explicit=True,
        superseded_by_memory_id=None,
        metadata={},
        created_at=now,
        updated_at=now,
    )


if __name__ == "__main__":
    unittest.main()
