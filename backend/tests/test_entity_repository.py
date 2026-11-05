from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from app.memory.entities.models import MemoryEntityRelation
from app.memory.entities.repository import EntityRepository
from app.memory.models import EMBEDDING_DIMENSIONS, content_hash


class _AsyncContext:
    def __init__(self, value: object) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(self, *_: object) -> None:
        return None


class _Connection:
    def __init__(
        self,
        *,
        fetch_results: list[list[dict[str, Any]]] | None = None,
        fetchrow_results: list[dict[str, Any] | None] | None = None,
        fetchval_results: list[object] | None = None,
    ) -> None:
        self.fetch_results = list(fetch_results or [])
        self.fetchrow_results = list(fetchrow_results or [])
        self.fetchval_results = list(fetchval_results or [])
        self.calls: list[tuple[str, str, tuple[object, ...]]] = []
        self.transaction_count = 0

    def transaction(self) -> _AsyncContext:
        self.transaction_count += 1
        return _AsyncContext(self)

    async def fetch(self, query: str, *args: object) -> list[dict[str, Any]]:
        self.calls.append(("fetch", query, args))
        return self.fetch_results.pop(0)

    async def fetchrow(self, query: str, *args: object) -> dict[str, Any] | None:
        self.calls.append(("fetchrow", query, args))
        return self.fetchrow_results.pop(0)

    async def fetchval(self, query: str, *args: object) -> object:
        self.calls.append(("fetchval", query, args))
        return self.fetchval_results.pop(0)


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self) -> _AsyncContext:
        return _AsyncContext(self.connection)


class EntityRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_name_entities_remain_distinct_candidates(self) -> None:
        first = _entity_row(canonical_name="Alex Smith")
        second = _entity_row(canonical_name="Alex Smith")
        connection = _Connection(fetch_results=[[first, second]])
        repository = EntityRepository(_Pool(connection))  # type: ignore[arg-type]

        results = await repository.find_by_name(owner_id="owner", name=" ALEX   SMITH ")

        self.assertEqual([item.id for item in results], [first["id"], second["id"]])
        self.assertEqual(connection.calls[0][2][1], "alex smith")
        self.assertIn("aliases @>", connection.calls[0][1])

    async def test_external_identity_is_idempotent_but_intent_guarded(self) -> None:
        entity = _entity_row(
            external_namespace="contacts",
            external_id="person-1",
        )
        connection = _Connection(fetchrow_results=[None, entity])
        repository = EntityRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.create_entity(
            owner_id="owner",
            entity_type="person",
            canonical_name="Dennis",
            confidence=0.9,
            external_namespace="contacts",
            external_id="person-1",
        )

        self.assertEqual(result.id, entity["id"])
        self.assertIn("on conflict", connection.calls[0][1].lower())

    async def test_entity_without_external_identity_has_no_name_conflict_clause(self) -> None:
        entity = _entity_row()
        connection = _Connection(fetchrow_results=[entity])
        repository = EntityRepository(_Pool(connection))  # type: ignore[arg-type]

        await repository.create_entity(
            owner_id="owner",
            entity_type="person",
            canonical_name="Dennis",
            confidence=0.9,
        )

        self.assertNotIn("on conflict", connection.calls[0][1].lower())

    async def test_mention_validates_entity_and_memory_in_one_transaction(self) -> None:
        entity = _entity_row()
        mention = _mention_row(entity["id"])
        connection = _Connection(
            fetchrow_results=[entity, mention],
            fetchval_results=[True],
        )
        repository = EntityRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.attach_mention(
            owner_id="owner",
            entity_id=entity["id"],
            memory_item_id=mention["memory_item_id"],
            role="subject",
            confidence=0.8,
        )

        self.assertEqual(result.id, mention["id"])
        self.assertEqual(connection.transaction_count, 1)
        self.assertIn("memory_items", connection.calls[1][1])

    async def test_entity_embedding_is_provider_namespaced_and_hash_guarded(self) -> None:
        entity = _entity_row()
        vector = tuple(0.01 for _ in range(EMBEDDING_DIMENSIONS))
        embedding = _embedding_row(entity["id"], vector)
        connection = _Connection(fetchrow_results=[entity, embedding])
        repository = EntityRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.upsert_embedding(
            owner_id="owner",
            entity_id=entity["id"],
            embedding_kind="identity",
            provider="openai",
            model="text-embedding-3-small",
            embedded_content="Dennis",
            embedding=vector,
        )

        self.assertEqual(result.content_hash, content_hash("Dennis"))
        self.assertIn("embedding_kind, provider, model", connection.calls[1][1])
        self.assertIn("content_hash", connection.calls[1][1])

    async def test_relation_requires_exactly_one_object_representation(self) -> None:
        repository = EntityRepository(_Pool(_Connection()))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            await repository.create_relation(
                owner_id="owner",
                subject_entity_id=uuid4(),
                predicate="located_at",
                confidence=0.9,
                valid_from=datetime.now(UTC),
                supporting_memory_id=uuid4(),
            )

    async def test_relation_validates_owner_scoped_subject_object_and_support(self) -> None:
        subject = _entity_row()
        object_entity = _entity_row(entity_type="place", canonical_name="Kitchen")
        relation = _relation_row(subject["id"], object_entity_id=object_entity["id"])
        connection = _Connection(
            fetchrow_results=[subject, object_entity, relation],
            fetchval_results=[True],
        )
        repository = EntityRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.create_relation(
            owner_id="owner",
            subject_entity_id=subject["id"],
            predicate="located_at",
            object_entity_id=object_entity["id"],
            confidence=0.9,
            valid_from=relation["valid_from"],
            supporting_memory_id=relation["supporting_memory_id"],
        )

        self.assertEqual(result.id, relation["id"])
        self.assertEqual(connection.transaction_count, 1)
        self.assertIn("on conflict do nothing", connection.calls[-1][1].lower())

    async def test_superseding_relation_preserves_closed_history(self) -> None:
        original = _relation_row(uuid4())
        replacement = _relation_row(
            original["subject_entity_id"],
            object_value="jacket pocket",
            valid_from=original["valid_from"] + timedelta(minutes=10),
        )
        superseded = dict(original)
        superseded.update(
            status="superseded",
            valid_to=replacement["valid_from"],
            superseded_by_relation_id=replacement["id"],
        )
        connection = _Connection(fetchrow_results=[original, replacement, superseded])
        repository = EntityRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.supersede_relation(
            owner_id="owner",
            relation_id=original["id"],
            replacement_relation_id=replacement["id"],
        )

        self.assertEqual(result.status, "superseded")
        self.assertEqual(result.valid_to, replacement["valid_from"])
        self.assertEqual(result.superseded_by_relation_id, replacement["id"])

    async def test_unrelated_relation_cannot_be_used_as_replacement(self) -> None:
        original = _relation_row(uuid4())
        replacement = _relation_row(
            uuid4(),
            object_value="jacket pocket",
            valid_from=original["valid_from"] + timedelta(minutes=10),
        )
        connection = _Connection(fetchrow_results=[original, replacement])
        repository = EntityRepository(_Pool(connection))  # type: ignore[arg-type]

        with self.assertRaisesRegex(
            Exception,
            "share subject and predicate",
        ):
            await repository.supersede_relation(
                owner_id="owner",
                relation_id=original["id"],
                replacement_relation_id=replacement["id"],
            )

    def test_relation_model_rejects_active_relation_with_end_time(self) -> None:
        row = _relation_row(uuid4())
        with self.assertRaisesRegex(ValueError, "cannot be closed"):
            MemoryEntityRelation(
                **{
                    **row,
                    "valid_to": row["valid_from"] + timedelta(seconds=1),
                }
            )


def _entity_row(
    *,
    entity_type: str = "person",
    canonical_name: str = "Dennis",
    external_namespace: str | None = None,
    external_id: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "id": uuid4(),
        "owner_id": "owner",
        "entity_type": entity_type,
        "canonical_name": canonical_name,
        "normalized_name": " ".join(canonical_name.casefold().split()),
        "aliases": [],
        "description": "",
        "confidence": 0.9,
        "external_namespace": external_namespace,
        "external_id": external_id,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }


def _mention_row(entity_id: object) -> dict[str, Any]:
    return {
        "id": uuid4(),
        "owner_id": "owner",
        "entity_id": entity_id,
        "memory_item_id": uuid4(),
        "role": "subject",
        "confidence": 0.8,
        "created_at": datetime.now(UTC),
    }


def _embedding_row(entity_id: object, vector: tuple[float, ...]) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "entity_id": entity_id,
        "embedding_kind": "identity",
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": EMBEDDING_DIMENSIONS,
        "embedded_content": "Dennis",
        "content_hash": content_hash("Dennis"),
        "embedding": vector,
        "embedded_at": now,
        "updated_at": now,
    }


def _relation_row(
    subject_entity_id: object,
    *,
    object_entity_id: object | None = None,
    object_value: Any | None = "kitchen counter",
    valid_from: datetime | None = None,
) -> dict[str, Any]:
    now = valid_from or datetime.now(UTC)
    if object_entity_id is not None:
        object_value = None
    return {
        "id": uuid4(),
        "owner_id": "owner",
        "subject_entity_id": subject_entity_id,
        "predicate": "located_at",
        "object_entity_id": object_entity_id,
        "object_value": object_value,
        "status": "active",
        "confidence": 0.9,
        "valid_from": now,
        "valid_to": None,
        "supporting_memory_id": uuid4(),
        "superseded_by_relation_id": None,
        "metadata": {},
        "created_at": now,
        "updated_at": now,
    }
