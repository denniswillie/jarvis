from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

import asyncpg

from app.memory.models import ForgetMemoryResult, MemoryType


class MemoryForgettingRepository(Protocol):
    async def forget_derived_memory(
        self,
        *,
        owner_id: str,
        memory_id: UUID,
        reason: str,
    ) -> ForgetMemoryResult: ...


@dataclass(frozen=True, slots=True)
class MemoryForgettingService:
    repository: MemoryForgettingRepository

    async def forget(
        self,
        *,
        owner_id: str,
        memory_id: UUID,
        reason: str | None = None,
    ) -> ForgetMemoryResult:
        normalized_owner_id = owner_id.strip()
        if not normalized_owner_id:
            raise ValueError("owner_id cannot be blank")
        normalized_reason = (reason or "User requested that Jarvis forget this memory.").strip()
        if not normalized_reason:
            raise ValueError("reason cannot be blank")
        if len(normalized_reason) > 1_000:
            raise ValueError("reason must be at most 1000 characters")
        return await self.repository.forget_derived_memory(
            owner_id=normalized_owner_id,
            memory_id=memory_id,
            reason=normalized_reason,
        )


def memory_identity_fingerprint(
    *,
    memory_type: MemoryType,
    content: str,
    metadata: Mapping[str, Any] | None = None,
    source_experience_event_id: UUID | None = None,
) -> str:
    safe_metadata = dict(metadata or {})
    persisted = safe_metadata.get("memory_identity_fingerprint")
    if isinstance(persisted, str) and _is_fingerprint(persisted):
        return persisted
    if memory_type == "episode" and source_experience_event_id is not None:
        return _identity_digest("episode", str(source_experience_event_id))
    semantic_identity = safe_metadata.get("semantic_identity_key")
    if memory_type == "semantic_fact" and isinstance(semantic_identity, str):
        return _identity_digest("semantic_fact", semantic_identity)
    range_key = safe_metadata.get("range_key")
    if memory_type == "reflection" and isinstance(range_key, str):
        return _identity_digest("reflection", range_key)
    procedure_key = safe_metadata.get("procedure_key")
    if memory_type == "procedure" and isinstance(procedure_key, str):
        return procedure_identity_fingerprint(procedure_key)
    return _identity_digest(memory_type, _normalize_text(content))


def semantic_fact_identity_fingerprint(
    *,
    subject_identity_key: str,
    predicate: str,
    value: str,
) -> str:
    return _identity_digest(
        "semantic_fact",
        _normalize_text(subject_identity_key),
        _normalize_text(predicate),
        _normalize_text(value),
    )


def procedure_identity_fingerprint(procedure_key: str) -> str:
    return _identity_digest("procedure", _normalize_text(procedure_key))


def reflection_identity_fingerprint(range_key: str) -> str:
    return _identity_digest("reflection", range_key.strip())


async def suppress_automatic_memory_recreation(
    connection: asyncpg.Connection,
    *,
    owner_id: str,
    memory_type: MemoryType,
    identity_fingerprint: str,
) -> UUID | None:
    row = await connection.fetchrow(
        """
        update memory_forgetting_tombstones
        set suppressed_recreation_count = suppressed_recreation_count + 1,
            last_suppressed_at = now(),
            updated_at = now()
        where owner_id = $1
          and memory_type = $2
          and identity_fingerprint = $3
          and overridden_at is null
        returning source_memory_id
        """,
        owner_id,
        memory_type,
        identity_fingerprint,
    )
    return row["source_memory_id"] if row is not None else None


async def record_explicit_memory_override(
    connection: asyncpg.Connection,
    *,
    owner_id: str,
    memory_type: MemoryType,
    identity_fingerprint: str,
    override_memory_id: UUID,
    source_memory_id: UUID | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[UUID, ...]:
    rows = await connection.fetch(
        """
        update memory_forgetting_tombstones
        set overridden_at = now(),
            override_memory_id = $4,
            override_metadata = $6::jsonb,
            updated_at = now()
        where owner_id = $1
          and overridden_at is null
          and (
              (memory_type = $2 and identity_fingerprint = $3)
              or ($5::uuid is not null and source_memory_id = $5)
          )
        returning id
        """,
        owner_id,
        memory_type,
        identity_fingerprint,
        override_memory_id,
        source_memory_id,
        json.dumps(
            dict(metadata or {}),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )
    return tuple(row["id"] for row in rows)


def _identity_digest(*parts: str) -> str:
    payload = json.dumps(
        list(parts),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _is_fingerprint(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
