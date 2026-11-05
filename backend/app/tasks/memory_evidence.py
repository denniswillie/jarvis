"""Task dependencies on selected-memory originals, not another memory store."""
from __future__ import annotations

from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.models import SourceError, MemorySourceReference
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError




class TaskMemoryCitation(MemorySourceReference):
    """Keep the established task-result schema name while sharing its contract."""


class TaskMemoryEvidence:
    """Bind exact originals before exposing them; final citations only verify reads."""
    def __init__(self, context, sources, *, area_id, access_check=None):
        if sources.pool is not context.tasks.pool:
            raise ValueError("Task evidence and selected sources must share one database")
        self.context, self.sources, self.area_id = context, sources, area_id
        self.access_check = access_check

    async def _resolve(self, connection, citation):
        source = await self.sources._read(
            connection, self.context.task.owner_id, self.area_id, citation.source_id,
            citation.revision if citation.source_usage == "historical" else None,
        )
        expected = TaskMemoryCitation.from_metadata(source.original_evidence())
        if expected != citation:
            raise SourceError("task_memory_evidence_changed", "The original memory revision changed.")
        return source

    async def _access(self, citations, *, bind):
        async with self.sources.transaction(self.context.task.owner_id) as connection:
            return await self._access_in_transaction(connection, citations, bind=bind)

    async def _access_in_transaction(self, connection, citations, *, bind):
        if not connection.is_in_transaction():
            raise RuntimeError("Task memory dependencies require a transaction")
        owner, tasks, claim = self.context.task.owner_id, self.context.tasks, self.context.claim
        if self.access_check is not None:
            self.access_check(owner)
        # Same source-before-task order as canonical publication.
        await tasks.lock_owner(connection, owner)
        task, manifest, _ = await ContextRepository(tasks)._archive_scope(connection, claim, None)
        resolved = [await self._resolve(connection, citation) for citation in citations]
        if resolved:
            await connection.fetch(
                "select id from jarvis_context_payloads where owner_id=$1 and id=any($2::uuid[]) "
                "order by id for share", owner, [source.payload_id for source in resolved],
            )
        # Re-read after the payload lock; direct payload removal must not race binding.
        for citation in citations:
            source = await self._resolve(connection, citation)
            if bind:
                await connection.execute(
                    """insert into jarvis_task_memory_sources(owner_id,manifest_id,area_id,source_id,
                        source_revision,source_generation,payload_id,content_digest,source_usage)
                        values($1,$2,$3,$4,$5,$6,$7,$8,$9)
                        on conflict(owner_id,manifest_id,source_id,source_revision) do nothing""",
                    owner, manifest, self.area_id, source.source_id, source.revision,
                    source.generation, source.payload_id, source.content_digest, citation.source_usage,
                )
            bound = await connection.fetchrow(
                """select source_generation,payload_id,content_digest,source_usage
                    from jarvis_task_memory_sources where owner_id=$1 and manifest_id=$2
                    and source_id=$3 and source_revision=$4 and area_id=$5""",
                owner, manifest, source.source_id, source.revision, self.area_id,
            )
            if (bound is None or bound["source_generation"] != source.generation
                    or bound["payload_id"] != source.payload_id
                    or bound["content_digest"] != citation.content_digest
                    or bound["source_usage"] != citation.source_usage):
                raise TaskError("task_memory_evidence_not_read", "This task has not read that exact memory evidence.")
        if self.access_check is not None:
            self.access_check(owner)
        return [citation.model_dump(mode="json") for citation in citations]

    async def bind_in_transaction(self, connection, metadata):
        citations = [TaskMemoryCitation.from_metadata(source) for source in metadata]
        return await self._access_in_transaction(connection, citations, bind=True)

    async def bind(self, metadata):
        citations = [TaskMemoryCitation.from_metadata(source) for source in metadata]
        return await self._access(citations, bind=True)

    async def verify(self, citation):
        await self._access([citation], bind=False)


class TaskSourceMemoryReadBackend(SourceMemoryReadBackend):
    """Existing shared handlers with durable task dependencies, no legacy fallback."""
    def __init__(self, *args, task_evidence, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_evidence = task_evidence
        if (task_evidence.sources is not self.sources or task_evidence.area_id != self.area_id
                or task_evidence.context.task.owner_id != self.owner_id):
            raise ValueError("Task memory evidence must match the read backend scope")

    async def search(self, **arguments):
        result = await super().search(**arguments)
        citations = await self.task_evidence.bind(result["results"])
        return result | {"results": [source | {"task_evidence": citation}
                         for source, citation in zip(result["results"], citations)]}

    async def versions(self, **arguments):
        result = await super().versions(**arguments)
        citations = await self.task_evidence.bind(result["versions"])
        return result | {"versions": [source | {"task_evidence": citation}
                         for source, citation in zip(result["versions"], citations)]}

    async def explain(self, **arguments):
        result = await super().explain(**arguments)
        citations = await self.task_evidence.bind([result["source"]])
        return result | {"task_evidence": citations[0]}
