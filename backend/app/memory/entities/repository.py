from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast
from uuid import UUID

import asyncpg

from app.memory.entities.models import (
    EntityRelationStatus,
    MemoryEntity,
    MemoryEntityEmbedding,
    MemoryEntityMention,
    MemoryEntityRelation,
    normalize_aliases,
    normalize_entity_name,
)
from app.memory.jobs.models import normalize_job_json_object
from app.memory.models import EMBEDDING_DIMENSIONS, content_hash, normalize_embedding


class EntityRepositoryError(Exception):
    pass


class EntityMissingError(EntityRepositoryError):
    pass


class EntityConflictError(EntityRepositoryError):
    pass


class EntityScopeError(EntityRepositoryError):
    pass


class EntityRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_entity(
        self,
        *,
        owner_id: str,
        entity_type: str,
        canonical_name: str,
        aliases: Sequence[str] = (),
        description: str = "",
        confidence: float,
        external_namespace: str | None = None,
        external_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryEntity:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_type = _require_text(entity_type, "entity_type")
        normalized_canonical_name = _require_text(canonical_name, "canonical_name")
        normalized_name = normalize_entity_name(normalized_canonical_name)
        normalized_aliases = normalize_aliases(list(aliases))
        _require_score(confidence, "confidence")
        namespace, identity = _normalize_external_identity(
            external_namespace,
            external_id,
        )
        safe_metadata = _normalize_metadata(metadata)
        conflict_clause = (
            "on conflict (owner_id, external_namespace, external_id) "
            "where external_namespace is not null and external_id is not null do nothing"
            if namespace is not None
            else ""
        )

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                insert into memory_entities (
                    owner_id,
                    entity_type,
                    canonical_name,
                    normalized_name,
                    aliases,
                    description,
                    confidence,
                    external_namespace,
                    external_id,
                    metadata
                )
                values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
                {conflict_clause}
                returning *
                """,
                normalized_owner_id,
                normalized_type,
                normalized_canonical_name,
                normalized_name,
                list(normalized_aliases),
                description.strip(),
                confidence,
                namespace,
                identity,
                _json_dumps(safe_metadata),
            )
            if row is None and namespace is not None:
                row = await conn.fetchrow(
                    """
                    select * from memory_entities
                    where owner_id = $1
                      and external_namespace = $2
                      and external_id = $3
                    """,
                    normalized_owner_id,
                    namespace,
                    identity,
                )
        if row is None:
            raise EntityConflictError("entity insert lost without an existing identity")
        entity = _entity_from_row(row)
        if namespace is not None:
            _require_matching_entity_intent(
                entity,
                entity_type=normalized_type,
                canonical_name=normalized_canonical_name,
                aliases=normalized_aliases,
                description=description.strip(),
                confidence=confidence,
                metadata=safe_metadata,
            )
        return entity

    async def find_by_name(
        self,
        *,
        owner_id: str,
        name: str,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[MemoryEntity]:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_name = normalize_entity_name(name)
        normalized_type = _require_optional_text(entity_type, "entity_type")
        _require_limit(limit, maximum=100)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select * from memory_entities
                where owner_id = $1
                  and (normalized_name = $2 or aliases @> array[$2]::text[])
                  and ($3::text is null or entity_type = $3)
                order by confidence desc, created_at asc, id asc
                limit $4
                """,
                normalized_owner_id,
                normalized_name,
                normalized_type,
                limit,
            )
        return [_entity_from_row(row) for row in rows]

    async def attach_mention(
        self,
        *,
        owner_id: str,
        entity_id: UUID,
        memory_item_id: UUID,
        role: str,
        confidence: float,
    ) -> MemoryEntityMention:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_role = _require_text(role, "role")
        _require_score(confidence, "confidence")
        async with self._pool.acquire() as conn:  # noqa: SIM117
            async with conn.transaction():
                await _load_owned_entity(conn, normalized_owner_id, entity_id)
                await _load_owned_memory(conn, normalized_owner_id, memory_item_id)
                row = await conn.fetchrow(
                    """
                    insert into memory_entity_mentions (
                        owner_id,
                        entity_id,
                        memory_item_id,
                        role,
                        confidence
                    )
                    values ($1, $2, $3, $4, $5)
                    on conflict (entity_id, memory_item_id, role) do nothing
                    returning *
                    """,
                    normalized_owner_id,
                    entity_id,
                    memory_item_id,
                    normalized_role,
                    confidence,
                )
                if row is None:
                    row = await conn.fetchrow(
                        """
                        select * from memory_entity_mentions
                        where entity_id = $1 and memory_item_id = $2 and role = $3
                        """,
                        entity_id,
                        memory_item_id,
                        normalized_role,
                    )
        if row is None:
            raise EntityConflictError("entity mention insert lost without an existing row")
        mention = _mention_from_row(row)
        if mention.owner_id != normalized_owner_id or mention.confidence != confidence:
            raise EntityConflictError(
                "entity mention identity was reused with different immutable data"
            )
        return mention

    async def upsert_embedding(
        self,
        *,
        owner_id: str,
        entity_id: UUID,
        embedding_kind: str,
        provider: str,
        model: str,
        embedded_content: str,
        embedding: Sequence[float],
    ) -> MemoryEntityEmbedding:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_kind = _require_text(embedding_kind, "embedding_kind")
        normalized_provider = _require_text(provider, "provider")
        normalized_model = _require_text(model, "model")
        normalized_content = _require_text(embedded_content, "embedded_content")
        normalized_embedding = normalize_embedding(embedding)
        digest = content_hash(normalized_content)
        async with self._pool.acquire() as conn:  # noqa: SIM117
            async with conn.transaction():
                await _load_owned_entity(conn, normalized_owner_id, entity_id)
                row = await conn.fetchrow(
                    """
                    insert into memory_entity_embeddings (
                        entity_id,
                        embedding_kind,
                        provider,
                        model,
                        dimensions,
                        embedded_content,
                        content_hash,
                        embedding,
                        embedded_at
                    )
                    values ($1, $2, $3, $4, $5, $6, $7, $8::vector, now())
                    on conflict (entity_id, embedding_kind, provider, model)
                    do update
                    set dimensions = excluded.dimensions,
                        embedded_content = excluded.embedded_content,
                        content_hash = excluded.content_hash,
                        embedding = excluded.embedding,
                        embedded_at = now(),
                        updated_at = now()
                    where memory_entity_embeddings.content_hash
                        is distinct from excluded.content_hash
                    returning *
                    """,
                    entity_id,
                    normalized_kind,
                    normalized_provider,
                    normalized_model,
                    EMBEDDING_DIMENSIONS,
                    normalized_content,
                    digest,
                    _vector_literal(normalized_embedding),
                )
                if row is None:
                    row = await conn.fetchrow(
                        """
                        select * from memory_entity_embeddings
                        where entity_id = $1
                          and embedding_kind = $2
                          and provider = $3
                          and model = $4
                        """,
                        entity_id,
                        normalized_kind,
                        normalized_provider,
                        normalized_model,
                    )
        if row is None:
            raise EntityConflictError("entity embedding upsert lost without an existing row")
        return _embedding_from_row(row)

    async def create_relation(
        self,
        *,
        owner_id: str,
        subject_entity_id: UUID,
        predicate: str,
        object_entity_id: UUID | None = None,
        object_value: Any | None = None,
        confidence: float,
        valid_from: datetime,
        supporting_memory_id: UUID,
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryEntityRelation:
        if (object_entity_id is None) == (object_value is None):
            raise ValueError("relation requires exactly one entity or value object")
        if subject_entity_id == object_entity_id:
            raise ValueError("entity relation cannot point to itself")
        normalized_owner_id = _require_text(owner_id, "owner_id")
        normalized_predicate = _require_text(predicate, "predicate")
        _require_score(confidence, "confidence")
        safe_object_value = (
            _normalize_json_value(object_value, label="relation object value")
            if object_value is not None
            else None
        )
        safe_metadata = _normalize_metadata(metadata)
        object_json = (
            _json_value_dumps(safe_object_value) if safe_object_value is not None else None
        )
        async with self._pool.acquire() as conn:  # noqa: SIM117
            async with conn.transaction():
                await _load_owned_entity(conn, normalized_owner_id, subject_entity_id)
                if object_entity_id is not None:
                    await _load_owned_entity(conn, normalized_owner_id, object_entity_id)
                await _load_owned_memory(conn, normalized_owner_id, supporting_memory_id)
                row = await conn.fetchrow(
                    """
                    insert into memory_entity_relations (
                        owner_id,
                        subject_entity_id,
                        predicate,
                        object_entity_id,
                        object_value,
                        confidence,
                        valid_from,
                        supporting_memory_id,
                        metadata
                    )
                    values ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9::jsonb)
                    on conflict do nothing
                    returning *
                    """,
                    normalized_owner_id,
                    subject_entity_id,
                    normalized_predicate,
                    object_entity_id,
                    object_json,
                    confidence,
                    valid_from,
                    supporting_memory_id,
                    _json_dumps(safe_metadata),
                )
                if row is None:
                    row = await conn.fetchrow(
                        """
                        select * from memory_entity_relations
                        where owner_id = $1
                          and subject_entity_id = $2
                          and predicate = $3
                          and status = 'active'
                          and (
                                ($4::uuid is not null and object_entity_id = $4)
                                or ($4::uuid is null and object_value = $5::jsonb)
                              )
                        """,
                        normalized_owner_id,
                        subject_entity_id,
                        normalized_predicate,
                        object_entity_id,
                        object_json,
                    )
        if row is None:
            raise EntityConflictError("entity relation insert lost without an existing row")
        relation = _relation_from_row(row)
        if (
            relation.confidence != confidence
            or relation.valid_from != valid_from
            or relation.supporting_memory_id != supporting_memory_id
            or dict(relation.metadata) != safe_metadata
        ):
            raise EntityConflictError(
                "active entity relation was reused with different durable data"
            )
        return relation

    async def supersede_relation(
        self,
        *,
        owner_id: str,
        relation_id: UUID,
        replacement_relation_id: UUID,
    ) -> MemoryEntityRelation:
        if relation_id == replacement_relation_id:
            raise ValueError("entity relation cannot supersede itself")
        normalized_owner_id = _require_text(owner_id, "owner_id")
        async with self._pool.acquire() as conn:  # noqa: SIM117
            async with conn.transaction():
                relation = await _load_owned_relation(
                    conn,
                    normalized_owner_id,
                    relation_id,
                    lock=True,
                )
                replacement = await _load_owned_relation(
                    conn,
                    normalized_owner_id,
                    replacement_relation_id,
                    lock=True,
                )
                if relation.status == "superseded":
                    if relation.superseded_by_relation_id == replacement_relation_id:
                        return relation
                    raise EntityConflictError("entity relation has a different replacement")
                if relation.status != "active":
                    raise EntityConflictError("only active entity relations can be superseded")
                if replacement.status != "active":
                    raise EntityConflictError("replacement entity relation must be active")
                if (
                    replacement.subject_entity_id != relation.subject_entity_id
                    or replacement.predicate != relation.predicate
                ):
                    raise EntityConflictError(
                        "replacement entity relation must share subject and predicate"
                    )
                if replacement.valid_from < relation.valid_from:
                    raise ValueError("replacement cannot begin before the original relation")
                row = await conn.fetchrow(
                    """
                    update memory_entity_relations
                    set status = 'superseded',
                        valid_to = $3,
                        superseded_by_relation_id = $4,
                        updated_at = now()
                    where owner_id = $1 and id = $2 and status = 'active'
                    returning *
                    """,
                    normalized_owner_id,
                    relation_id,
                    replacement.valid_from,
                    replacement_relation_id,
                )
        if row is None:
            raise EntityConflictError("entity relation changed while being superseded")
        return _relation_from_row(row)

    async def retract_relation(
        self,
        *,
        owner_id: str,
        relation_id: UUID,
        valid_to: datetime,
    ) -> MemoryEntityRelation:
        normalized_owner_id = _require_text(owner_id, "owner_id")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                update memory_entity_relations
                set status = 'retracted', valid_to = $3, updated_at = now()
                where owner_id = $1
                  and id = $2
                  and status = 'active'
                  and valid_from <= $3
                returning *
                """,
                normalized_owner_id,
                relation_id,
                valid_to,
            )
            if row is not None:
                return _relation_from_row(row)
            existing_row = await conn.fetchrow(
                "select * from memory_entity_relations where owner_id = $1 and id = $2",
                normalized_owner_id,
                relation_id,
            )
        if existing_row is None:
            raise EntityMissingError()
        existing = _relation_from_row(existing_row)
        if existing.status == "retracted" and existing.valid_to == valid_to:
            return existing
        if existing.status != "active":
            raise EntityConflictError("entity relation is already terminal")
        raise ValueError("entity relation cannot end before it begins")


async def _load_owned_entity(
    conn: asyncpg.Connection,
    owner_id: str,
    entity_id: UUID,
) -> MemoryEntity:
    row = await conn.fetchrow(
        "select * from memory_entities where owner_id = $1 and id = $2",
        owner_id,
        entity_id,
    )
    if row is None:
        raise EntityScopeError("entity is outside the owner scope")
    return _entity_from_row(row)


async def _load_owned_memory(
    conn: asyncpg.Connection,
    owner_id: str,
    memory_id: UUID,
) -> None:
    exists = await conn.fetchval(
        "select exists(select 1 from memory_items where owner_id = $1 and id = $2)",
        owner_id,
        memory_id,
    )
    if not exists:
        raise EntityScopeError("supporting memory is outside the owner scope")


async def _load_owned_relation(
    conn: asyncpg.Connection,
    owner_id: str,
    relation_id: UUID,
    *,
    lock: bool,
) -> MemoryEntityRelation:
    lock_clause = " for update" if lock else ""
    row = await conn.fetchrow(
        f"select * from memory_entity_relations where owner_id = $1 and id = $2{lock_clause}",
        owner_id,
        relation_id,
    )
    if row is None:
        raise EntityScopeError("entity relation is outside the owner scope")
    return _relation_from_row(row)


def _require_matching_entity_intent(
    entity: MemoryEntity,
    *,
    entity_type: str,
    canonical_name: str,
    aliases: tuple[str, ...],
    description: str,
    confidence: float,
    metadata: Mapping[str, Any],
) -> None:
    if (
        entity.entity_type != entity_type
        or entity.canonical_name != canonical_name
        or entity.aliases != aliases
        or entity.description != description
        or entity.confidence != confidence
        or dict(entity.metadata) != dict(metadata)
    ):
        raise EntityConflictError(
            "external entity identity was reused with different durable data"
        )


def _entity_from_row(row: Mapping[str, Any]) -> MemoryEntity:
    return MemoryEntity(
        id=row["id"],
        owner_id=row["owner_id"],
        entity_type=row["entity_type"],
        canonical_name=row["canonical_name"],
        normalized_name=row["normalized_name"],
        aliases=tuple(row["aliases"]),
        description=row["description"],
        confidence=float(row["confidence"]),
        external_namespace=row["external_namespace"],
        external_id=row["external_id"],
        metadata=_json_object(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _mention_from_row(row: Mapping[str, Any]) -> MemoryEntityMention:
    return MemoryEntityMention(
        id=row["id"],
        owner_id=row["owner_id"],
        entity_id=row["entity_id"],
        memory_item_id=row["memory_item_id"],
        role=row["role"],
        confidence=float(row["confidence"]),
        created_at=row["created_at"],
    )


def _embedding_from_row(row: Mapping[str, Any]) -> MemoryEntityEmbedding:
    return MemoryEntityEmbedding(
        entity_id=row["entity_id"],
        embedding_kind=row["embedding_kind"],
        provider=row["provider"],
        model=row["model"],
        dimensions=row["dimensions"],
        embedded_content=row["embedded_content"],
        content_hash=row["content_hash"],
        embedding=_parse_vector(row["embedding"]),
        embedded_at=row["embedded_at"],
        updated_at=row["updated_at"],
    )


def _relation_from_row(row: Mapping[str, Any]) -> MemoryEntityRelation:
    return MemoryEntityRelation(
        id=row["id"],
        owner_id=row["owner_id"],
        subject_entity_id=row["subject_entity_id"],
        predicate=row["predicate"],
        object_entity_id=row["object_entity_id"],
        object_value=_json_value(row["object_value"]),
        status=cast(EntityRelationStatus, row["status"]),
        confidence=float(row["confidence"]),
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        supporting_memory_id=row["supporting_memory_id"],
        superseded_by_relation_id=row["superseded_by_relation_id"],
        metadata=_json_object(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _normalize_external_identity(
    namespace: str | None,
    external_id: str | None,
) -> tuple[str | None, str | None]:
    if namespace is None and external_id is None:
        return None, None
    if namespace is None or external_id is None:
        raise ValueError("external entity identity requires both namespace and id")
    return (
        _require_text(namespace, "external_namespace"),
        _require_text(external_id, "external_id"),
    )


def _normalize_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return normalize_job_json_object(metadata or {}, label="entity metadata")


def _normalize_json_value(value: Any, *, label: str) -> Any:
    if value is None:
        raise ValueError(f"{label} cannot be null")
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-compatible") from exc
    if len(serialized.encode("utf-8")) > 32768:
        raise ValueError(f"{label} exceeds 32768 bytes")
    return json.loads(serialized)


def _json_object(value: Any) -> dict[str, Any]:
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, Mapping):
        raise TypeError("database JSON value must be an object")
    return dict(decoded)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _json_value_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in values) + "]"


def _parse_vector(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.startswith("[") or not stripped.endswith("]"):
            raise ValueError("database embedding has an invalid vector representation")
        values = tuple(float(item) for item in stripped[1:-1].split(","))
        return normalize_embedding(values)
    if isinstance(value, Sequence):
        return normalize_embedding(value)
    raise TypeError("database embedding must be a vector string or numeric sequence")


def _require_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized


def _require_optional_text(value: str | None, label: str) -> str | None:
    return None if value is None else _require_text(value, label)


def _require_score(value: float, label: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{label} must be between 0 and 1")


def _require_limit(limit: int, *, maximum: int) -> None:
    if isinstance(limit, bool) or not 1 <= limit <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
