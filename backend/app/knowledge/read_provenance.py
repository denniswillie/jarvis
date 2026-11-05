"""Bind selected-memory reads to existing removable payloads and source revisions."""
from __future__ import annotations

import json
from collections import OrderedDict

from app.knowledge.models import SourceError, MemorySourceReference
from app.runtime.feature_config import TaskLimits
from app.tasks.repository import TaskRepository
from app.tasks.models import digest


class SourceReadProvenance:
    reference_limit = 32

    def __init__(self, sources):
        self.sources = sources
        self.tasks = TaskRepository(sources.pool, TaskLimits())
        self._context_cache = OrderedDict()

    async def bind(self, owner_id, area_id, metadata, materialize):
        async with self.sources.transaction(owner_id) as connection:
            return await self.bind_in_transaction(connection, owner_id, area_id, metadata, materialize)

    async def bind_in_transaction(self, connection, owner_id, area_id, metadata, materialize,
                                  *, shared_task_owner=False):
        """Compose effects/receipts, or explicitly audited append-only derivations.

        Shared callers must already hold shared memory ownership, must create
        only derived payloads/provenance, and must not upgrade owner locks.
        Generic callbacks may mutate canonical state and remain exclusive.
        """
        if type(shared_task_owner) is not bool:
            raise ValueError("Shared task ownership must be explicit")
        if not connection.is_in_transaction():
            raise RuntimeError("Source-bound copies require a transaction")
        if len(metadata) > self.reference_limit:
            raise SourceError("knowledge_read_evidence_limit", "Too many source references.")
        references = {}
        for item in metadata:
            reference = MemorySourceReference.from_metadata(item)
            key = (reference.source_id, reference.revision)
            if key in references and references[key] != reference:
                raise SourceError("knowledge_read_evidence_conflict", "Inconsistent source references.")
            references[key] = reference
        # Match publication's source-owner, task-owner, then payload order.
        lock = self.tasks.lock_owner_shared if shared_task_owner else self.tasks.lock_owner
        await lock(connection, owner_id)
        await self.sources._area(connection, owner_id, area_id)
        resolved = await self._resolve_many(connection, owner_id, area_id, references.values())
        payload_ids = tuple(dict.fromkeys(source.payload_id for source in resolved))
        if payload_ids and not await connection.fetchval(
                "select lock_jarvis_copy_sources($1,$2::uuid[])", owner_id, list(payload_ids)):
            raise SourceError("knowledge_read_evidence_unavailable", "Original evidence is unavailable.")
        if references:
            await connection.fetch("""select id from jarvis_knowledge_sources
                where owner_id=$1 and area_id=$2 and id=any($3::uuid[]) order by id for share""",
                owner_id, area_id, [reference.source_id for reference in references.values()])
        # Revalidate after row locks; no provider request runs in this transaction.
        resolved = await self._resolve_many(connection, owner_id, area_id, references.values())
        value, dependent = await materialize(connection, payload_ids)
        if resolved:
            await connection.executemany("""insert into jarvis_context_memory_sources
                (owner_id,dependent_payload_id,area_id,source_id,source_revision,
                 source_generation,source_payload_id,content_digest,source_usage)
                values($1,$2,$3,$4,$5,$6,$7,$8,$9)
                on conflict(owner_id,dependent_payload_id,source_id,source_revision) do nothing""",
                [(owner_id, dependent, area_id, source.source_id, source.revision,
                  source.generation, source.payload_id, source.content_digest, reference.source_usage)
                 for source, reference in zip(resolved, references.values(), strict=True)])
        return value

    async def _resolve_many(self, connection, owner_id, area_id, references):
        references = tuple(references)
        sources = await self.sources._read_many(connection, owner_id, area_id,
            [(reference.source_id, reference.revision if reference.source_usage == "historical" else None)
             for reference in references])
        for source, reference in zip(sources, references, strict=True):
            if MemorySourceReference.from_metadata(source.original_evidence()) != reference:
                raise SourceError("knowledge_read_evidence_changed", "The selected source revision changed.")
        return sources

    async def bind_context(self, owner_id, area_id, payload):
        if not payload["sources"]:
            return ()
        references = sorted((MemorySourceReference.from_metadata(item).model_dump(mode="json")
                             for item in payload["sources"]), key=lambda item:(item["source_id"], item["revision"]))
        descriptor = {"kind":"source_index_dependencies", "area_id":str(area_id), "sources":references}
        key = digest({"owner_id":owner_id, **descriptor})
        async def materialize(connection, originals):
            cached = self._context_cache.get(key)
            if cached is not None and await connection.fetchval(
                    "select lock_jarvis_copy_source($1,$2)", owner_id, cached) == "available":
                return (cached,), cached
            identity = await connection.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available',$2::jsonb) returning id""",
                owner_id, json.dumps(descriptor, ensure_ascii=True, allow_nan=False))
            if originals:
                await connection.executemany("""insert into jarvis_context_payload_dependencies
                    (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
                    [(owner_id, source, identity) for source in originals])
            return (identity,), identity
        result = await self.bind(owner_id, area_id, payload["sources"], materialize)
        # Advisory identity cache only. Every hit revalidates originals and the
        # owned removable root under the same transaction as a fresh binding.
        self._context_cache[key] = result[0]
        self._context_cache.move_to_end(key)
        while len(self._context_cache) > 32:
            self._context_cache.popitem(last=False)
        return result
