"""Freeze a worker's actual read dependencies without depending on its growing manifest."""
from app.knowledge.models import SourceError
from app.knowledge.read_provenance import SourceReadProvenance
from app.tasks.context_repository import ContextRepository


async def freeze_artifact_evidence(connection, context, sources, area_id):
    tasks, owner = context.tasks, context.task.owner_id
    await tasks.lock_owner(connection, owner)
    task, manifest_id, _ = await ContextRepository(tasks)._archive_scope(connection, context.claim, None)
    if task.knowledge_area_id != area_id:
        raise SourceError("artifact_area_mismatch", "Artifact evidence must use the task's selected knowledge area.")
    events = await connection.fetch("""select event_id,event_revision,payload_id,content_digest,source_usage
        from jarvis_task_context_sources where owner_id=$1 and manifest_id=$2 order by event_id""",
        owner, manifest_id)
    memories = await connection.fetch("""select area_id,source_id,source_revision,source_generation,
        payload_id,content_digest,source_usage from jarvis_task_memory_sources
        where owner_id=$1 and manifest_id=$2 order by source_id,source_revision""", owner, manifest_id)
    if len(events) + len(memories) > 2048 or any(row["area_id"] != area_id for row in memories):
        raise SourceError("artifact_evidence_scope", "Artifact evidence exceeds its authorized source scope.")
    descriptor = {"format": "jarvis.task_artifact_evidence.v1", "task_id": str(task.id),
        "task_revision": task.revision, "manifest_id": str(manifest_id),
        "context_sources": [{key: str(value) if key in {"event_id", "payload_id"} else value
                             for key, value in dict(row).items()} for row in events],
        "memory_sources": [{key: str(value) if key in {"area_id", "source_id", "payload_id"} else value
                            for key, value in dict(row).items()} for row in memories],
        "meaning": "Exact inputs read before this artifact; later reads are not retroactively included."}
    payload = await tasks.payload(connection, owner, descriptor)
    parents = [row["payload_id"] for row in events]
    if parents:
        if not await connection.fetchval("select lock_jarvis_copy_sources($1,$2::uuid[])", owner, parents):
            raise SourceError("artifact_evidence_unavailable", "A source read by this worker is unavailable.")
        await connection.executemany("""insert into jarvis_context_payload_dependencies
            (owner_id,dependent_payload_id,source_payload_id) values($1,$2,$3) on conflict do nothing""",
            [(owner, payload, parent) for parent in parents])
        # Historical snapshots retain deletion dependencies without claiming to
        # be the current revision. Current reads also inherit correction fences.
        current = [row for row in events if row["source_usage"] == "current"]
        if current:
            await connection.executemany("""insert into jarvis_context_event_dependencies
                (owner_id,dependent_payload_id,event_id,event_revision,source_payload_id,source_usage)
                values($1,$2,$3,$4,$5,'current')""",
                [(owner, payload, row["event_id"], row["event_revision"], row["payload_id"]) for row in current])
    provenance = SourceReadProvenance(sources)
    async def materialize(conn, originals):
        if originals:
            await conn.executemany("""insert into jarvis_context_payload_dependencies
                (owner_id,dependent_payload_id,source_payload_id) values($1,$2,$3) on conflict do nothing""",
                [(owner, payload, original) for original in originals])
        return payload, payload
    for offset in range(0, len(memories), provenance.reference_limit):
        metadata = [{"source_id": str(row["source_id"]), "revision": row["source_revision"],
            "generation": row["source_generation"], "content_digest": row["content_digest"],
            "is_current": row["source_usage"] == "current"}
            for row in memories[offset:offset + provenance.reference_limit]]
        await provenance.bind_in_transaction(connection, owner, area_id, metadata, materialize)
    return payload
