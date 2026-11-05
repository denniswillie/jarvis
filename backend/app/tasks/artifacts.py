"""Current artifact links from accepted task receipts, never model-authored URLs."""
from uuid import UUID

from app.files.repository import _file_from_record
from app.knowledge.models import MemorySourceReference, SourceError
from app.knowledge.product_files import read_file_original
from app.knowledge.sources import SourceRepository
from app.tasks.models import json_object


async def result_artifacts(connection, tasks, task, output, *, include_targets=False):
    if not output or not output.expert_available:
        return []
    keys = []
    for reference in output.content.completed_operation_refs:
        try:
            identity = UUID(hex=reference.removeprefix("op_"))
        except (ValueError, AttributeError):
            continue
        if reference == "op_" + identity.hex:
            keys.append("tool-grant:" + identity.hex)
    if not keys:
        return []
    rows = await connection.fetch("""select a.id,a.grant_id,p.content from jarvis_task_actions a
        join jarvis_context_payloads p on p.owner_id=a.owner_id and p.id=a.receipt_payload_id
        join jarvis_operation_grants g on g.owner_id=a.owner_id and g.id=a.grant_id
        where a.owner_id=$1 and a.task_id=$2 and a.operation_key=any($3::text[])
            and a.status='accepted' and p.availability='available' and g.capability in ('files.create_text','files.publish_snapshot')
        order by a.created_at,a.id""", task.owner_id, task.id, keys)
    result, seen = [], set()
    for row in rows:
        try:
            receipt = json_object(row["content"])
            if receipt.get("tool") not in {"create_text_file", "publish_generated_file"} or receipt["result"].get("status") != "success":
                continue
            saved = receipt["result"]["file"]
            file_id = UUID(saved["id"])
            record = await connection.fetchrow(
                "select * from jarvis_files where owner_id=$1 and id=$2", task.owner_id, file_id)
            if record is None or file_id in seen:
                continue
            file = _file_from_record(record)
            lineage = file.metadata.get("task_artifact", {})
            if lineage.get("task_id") != str(task.id) or lineage.get("action_id") != str(row["id"]):
                continue
            area_id = task.knowledge_area_id or UUID(lineage["area_id"])
            source, _ = await read_file_original(connection, SourceRepository(tasks.pool), task.owner_id,
                                                 area_id, file, lock=False)
            if MemorySourceReference.from_metadata(source.original_evidence()) != MemorySourceReference.from_metadata(
                    saved["memory_source"]):
                continue
            item = {"file_id": str(file_id), "filename": file.filename,
                    "content_type": file.content_type, "size_bytes": file.size_bytes}
            if include_targets:
                # Internal callers may bind exact targets to the same original
                # evidence. The existing inbox wire shape remains unchanged.
                item.update(version_id=source.document_id,
                            operation_ref="op_" + row["grant_id"].hex,
                            source_payload_id=str(source.payload_id))
            result.append(item)
            seen.add(file_id)
        except (ValueError, TypeError, KeyError, SourceError):
            # A stale/corrupt optional link does not manufacture current access or
            # hide an otherwise valid written answer. Database errors still fail.
            continue
    return result
