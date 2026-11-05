from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import asyncpg

from app.memory.entities.models import MemoryEntity
from app.memory.semantic.contracts import SemanticEntityIdentity


class SemanticEntityStore(Protocol):
    async def lock_entity_identity(
        self,
        *,
        owner_id: str,
        identity: SemanticEntityIdentity,
        connection: asyncpg.Connection,
    ) -> None: ...

    async def find_external_identity(
        self,
        *,
        owner_id: str,
        namespace: str,
        external_id: str,
        connection: asyncpg.Connection,
    ) -> MemoryEntity | None: ...

    async def find_exact_entities(
        self,
        *,
        owner_id: str,
        identity: SemanticEntityIdentity,
        connection: asyncpg.Connection,
    ) -> tuple[MemoryEntity, ...]: ...

    async def find_alias_entities(
        self,
        *,
        owner_id: str,
        identity: SemanticEntityIdentity,
        connection: asyncpg.Connection,
    ) -> tuple[MemoryEntity, ...]: ...

    async def find_embedding_entities(
        self,
        *,
        owner_id: str,
        identity: SemanticEntityIdentity,
        embedding: Sequence[float],
        provider: str,
        model: str,
        minimum_similarity: float,
        connection: asyncpg.Connection,
    ) -> tuple[MemoryEntity, ...]: ...

    async def insert_entity(
        self,
        *,
        owner_id: str,
        identity: SemanticEntityIdentity,
        connection: asyncpg.Connection,
    ) -> MemoryEntity: ...


@dataclass(frozen=True, slots=True)
class ResolvedSemanticEntity:
    entity: MemoryEntity
    created: bool
    resolution: str


class SemanticEntityResolver:
    """Resolve code-owned identities without merging people solely by name."""

    def __init__(
        self,
        repository: SemanticEntityStore,
        *,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        minimum_embedding_similarity: float = 0.92,
    ) -> None:
        if not 0 <= minimum_embedding_similarity <= 1:
            raise ValueError("entity similarity must be between zero and one")
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._embedding_model = embedding_model
        self._minimum_embedding_similarity = minimum_embedding_similarity

    async def resolve(
        self,
        *,
        owner_id: str,
        identity: SemanticEntityIdentity,
        connection: asyncpg.Connection,
        cache: dict[str, ResolvedSemanticEntity],
        embeddings: Mapping[str, Sequence[float]] | None = None,
    ) -> ResolvedSemanticEntity:
        cache_key = f"{owner_id}:{identity.identity_key}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        await self._repository.lock_entity_identity(
            owner_id=owner_id,
            identity=identity,
            connection=connection,
        )
        if identity.external_namespace is not None:
            existing = await self._repository.find_external_identity(
                owner_id=owner_id,
                namespace=identity.external_namespace,
                external_id=identity.external_id or "",
                connection=connection,
            )
            if existing is not None:
                if existing.entity_type != identity.entity_type:
                    raise ValueError("external entity identity has an incompatible type")
                return self._remember(cache, cache_key, existing, "external_identity")

        exact = await self._repository.find_exact_entities(
            owner_id=owner_id,
            identity=identity,
            connection=connection,
        )
        if len(exact) == 1 and identity.entity_type != "person":
            return self._remember(cache, cache_key, exact[0], "exact_name")
        if len(exact) > 1:
            return await self._create(owner_id, identity, connection, cache, cache_key)

        aliases = await self._repository.find_alias_entities(
            owner_id=owner_id,
            identity=identity,
            connection=connection,
        )
        if len(aliases) == 1:
            return self._remember(cache, cache_key, aliases[0], "alias")
        if len(aliases) > 1:
            return await self._create(owner_id, identity, connection, cache, cache_key)

        vector = embeddings.get(identity.identity_key) if embeddings is not None else None
        if (
            vector is not None
            and identity.entity_type != "person"
            and self._embedding_provider is not None
            and self._embedding_model is not None
        ):
            matches = await self._repository.find_embedding_entities(
                owner_id=owner_id,
                identity=identity,
                embedding=vector,
                provider=self._embedding_provider,
                model=self._embedding_model,
                minimum_similarity=self._minimum_embedding_similarity,
                connection=connection,
            )
            if len(matches) == 1:
                return self._remember(cache, cache_key, matches[0], "embedding")

        return await self._create(owner_id, identity, connection, cache, cache_key)

    async def _create(
        self,
        owner_id: str,
        identity: SemanticEntityIdentity,
        connection: asyncpg.Connection,
        cache: dict[str, ResolvedSemanticEntity],
        cache_key: str,
    ) -> ResolvedSemanticEntity:
        entity = await self._repository.insert_entity(
            owner_id=owner_id,
            identity=identity,
            connection=connection,
        )
        resolved = ResolvedSemanticEntity(entity=entity, created=True, resolution="created")
        cache[cache_key] = resolved
        return resolved

    @staticmethod
    def _remember(
        cache: dict[str, ResolvedSemanticEntity],
        key: str,
        entity: MemoryEntity,
        resolution: str,
    ) -> ResolvedSemanticEntity:
        resolved = ResolvedSemanticEntity(entity=entity, created=False, resolution=resolution)
        cache[key] = resolved
        return resolved
