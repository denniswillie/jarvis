from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

from app.memory.entities.models import MemoryEntity
from app.memory.semantic.contracts import SemanticEntityIdentity
from app.memory.semantic.entity_resolver import SemanticEntityResolver


class SemanticEntityResolverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.repository = AsyncMock()
        self.repository.find_external_identity.return_value = None
        self.repository.find_exact_entities.return_value = ()
        self.repository.find_alias_entities.return_value = ()
        self.repository.find_embedding_entities.return_value = ()
        self.connection = AsyncMock()
        self.resolver = SemanticEntityResolver(
            self.repository,
            embedding_provider="gemini",
            embedding_model="gemini-embedding-2",
        )

    async def test_stable_external_identity_has_highest_priority(self) -> None:
        person = _entity("Dennis", "person")
        self.repository.find_external_identity.return_value = person
        identity = SemanticEntityIdentity(
            name="Dennis",
            entity_type="person",
            confidence=0.9,
            external_namespace="contacts",
            external_id="dennis",
        )

        resolved = await self._resolve(identity)

        self.assertEqual(resolved.entity.id, person.id)
        self.assertEqual(resolved.resolution, "external_identity")
        self.repository.find_exact_entities.assert_not_awaited()

    async def test_exact_nonperson_name_resolves_without_embeddings(self) -> None:
        keys = _entity("Silver keys", "object")
        self.repository.find_exact_entities.return_value = (keys,)

        resolved = await self._resolve(_identity("Silver keys", "object"))

        self.assertEqual(resolved.entity.id, keys.id)
        self.assertEqual(resolved.resolution, "exact_name")
        self.repository.find_alias_entities.assert_not_awaited()

    async def test_alias_resolves_before_semantic_embedding(self) -> None:
        keys = _entity("House keys", "object")
        self.repository.find_alias_entities.return_value = (keys,)

        resolved = await self._resolve(_identity("Silver keys", "object"))

        self.assertEqual(resolved.resolution, "alias")
        self.repository.find_embedding_entities.assert_not_awaited()

    async def test_same_name_people_are_not_merged(self) -> None:
        existing = _entity("Alex Smith", "person")
        created = _entity("Alex Smith", "person")
        self.repository.find_exact_entities.return_value = (existing,)
        self.repository.insert_entity.return_value = created

        resolved = await self._resolve(_identity("Alex Smith", "person"))

        self.assertTrue(resolved.created)
        self.assertEqual(resolved.entity.id, created.id)

    async def test_ambiguous_exact_candidates_create_a_new_entity(self) -> None:
        self.repository.find_exact_entities.return_value = (
            _entity("Keys", "object"),
            _entity("Keys", "object"),
        )
        created = _entity("Keys", "object")
        self.repository.insert_entity.return_value = created

        resolved = await self._resolve(_identity("Keys", "object"))

        self.assertTrue(resolved.created)
        self.repository.find_alias_entities.assert_not_awaited()

    async def test_single_embedding_candidate_can_resolve_nonperson(self) -> None:
        keys = _entity("House keys", "object")
        self.repository.find_embedding_entities.return_value = (keys,)
        identity = _identity("Silver keys", "object")

        resolved = await self.resolver.resolve(
            owner_id="owner",
            identity=identity,
            connection=self.connection,
            cache={},
            embeddings={identity.identity_key: (0.1,)},
        )

        self.assertEqual(resolved.resolution, "embedding")

    async def _resolve(self, identity: SemanticEntityIdentity):
        return await self.resolver.resolve(
            owner_id="owner",
            identity=identity,
            connection=self.connection,
            cache={},
        )


def _identity(name: str, entity_type: str) -> SemanticEntityIdentity:
    return SemanticEntityIdentity(name=name, entity_type=entity_type, confidence=0.9)


def _entity(name: str, entity_type: str) -> MemoryEntity:
    now = datetime.now(UTC)
    return MemoryEntity(
        id=uuid4(),
        owner_id="owner",
        entity_type=entity_type,
        canonical_name=name,
        normalized_name=" ".join(name.casefold().split()),
        aliases=(),
        description="",
        confidence=0.9,
        external_namespace=None,
        external_id=None,
        metadata={},
        created_at=now,
        updated_at=now,
    )
