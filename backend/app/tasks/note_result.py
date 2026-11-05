"""Receipt-derived note outcomes in the existing task/attention result ledger.

Worker prose is never evidence of a save. Runtime-authored status is distinct
from expert-authored content. All canonical reads accept the caller's transaction.
"""
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.knowledge.models import SourceError
from app.knowledge.product_notes import NOTE_NAMESPACE
from app.knowledge.sources import SourceRepository, VERSION_SQL
from app.notes.workspace_account_note import account_note_id
from app.tasks.models import TaskError, digest, json_object


NOTE_PUBLICATION_CAPABILITY = "notes.publish_account_edit"
FORMAT = "jarvis.note.task_result.v1"


class NoteResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["jarvis.note.task_result.v1"] = FORMAT
    owner_id: str
    task_id: UUID
    task_revision: int = Field(strict=True, ge=1)
    attempt_id: UUID
    operation_id: UUID
    request_payload_id: UUID
    artifact_id: UUID
    area_id: UUID
    effect_committed: bool = Field(strict=True)
    outcome: Literal["completed", "failed", "clarification_required", "paused", "outcome_unknown"]
    publication_operation_id: UUID | None = None
    version_id: str | None = None
    source_id: UUID | None = None
    source_revision: int | None = None
    source_payload_id: UUID | None = None
    action_receipt_payload_id: UUID | None = None
    authored_by: Literal["note_service"] = "note_service"
    # The enclosing trusted result store owns this attribution, not a model.
    authorship: dict[Literal["kind"], Literal["runtime"]] = {"kind": "runtime"}


class NoteNotification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["jarvis.note.notification.v1"] = "jarvis.note.notification.v1"
    source_namespace: Literal["task_result"] = "task_result"
    authored_by: Literal["note_service"] = "note_service"
    task_id: UUID
    operation_id: UUID
    publication_operation_id: UUID | None
    artifact_id: UUID
    version_id: str | None
    effect_committed: bool
    is_current: bool | None
    message: str


def notification(result, *, is_current=None):
    if result.effect_committed:
        message = ("Your note edit was saved." if is_current else
                   "Your earlier note edit was saved. The note has since been updated.")
    else:
        message = {
            "failed": "The note edit finished without a confirmed save.",
            "clarification_required": "The note edit needs more information; nothing has been saved.",
            "paused": "The note edit is paused; nothing has been saved.",
            "outcome_unknown": "The note edit's saving status is not confirmed.",
        }[result.outcome]
    return NoteNotification(task_id=result.task_id, operation_id=result.operation_id,
        publication_operation_id=result.publication_operation_id, artifact_id=result.artifact_id,
        version_id=result.version_id, effect_committed=result.effect_committed, is_current=is_current,
        message=message)


async def commit_proof(connection, owner_id, task_id, operation_id):
    """Exact immutable domain/action/source agreement; no row locks or writes.

This can run in Attention's read-only snapshot. Its saved payload and action
receipt are roots of the result dependency graph, so withdrawal revokes copies.
"""
    row = await connection.fetchrow("""select n.*,a.receipt_payload_id,a.task_revision,
            g.request_event_id,g.request_revision,p.content as action_receipt,
            p.availability as action_availability,s.content as scope,s.availability as scope_availability,
            mapping.id as source_id,mapping.namespace,mapping.source_key
        from jarvis_note_operations n
        join jarvis_task_actions a on a.owner_id=n.owner_id and a.id=n.operation_id
        join jarvis_operation_grants g on g.owner_id=a.owner_id and g.id=a.grant_id
        join jarvis_task_grants binding on binding.owner_id=a.owner_id and binding.task_id=a.task_id and binding.grant_id=g.id
        join jarvis_context_payloads p on p.owner_id=a.owner_id and p.id=a.receipt_payload_id
        join jarvis_context_payloads s on s.owner_id=g.owner_id and s.id=g.scope_payload_id
        join jarvis_knowledge_sources mapping on mapping.owner_id=n.owner_id and mapping.area_id=n.area_id
            and mapping.namespace=$5 and mapping.source_key=n.artifact_id::text
        where n.owner_id=$1 and a.task_id=$2 and n.operation_id=$3 and g.capability=$4""",
        owner_id, task_id, operation_id, NOTE_PUBLICATION_CAPABILITY, NOTE_NAMESPACE)
    if row is None or row["action_availability"] != "available" or row["scope_availability"] != "available":
        raise TaskError("note_result_unavailable", "The saved note receipt is unavailable.")
    envelope, scope = json_object(row["action_receipt"]), json_object(row["scope"])
    result = envelope.get("result", {})
    artifact = result.get("artifact", {})
    if (row["operation_kind"] != "patch" or row["artifact_id"] != account_note_id(owner_id)
            or row["parameters_digest"] != scope.get("candidate_digest")
            or scope.get("task_id") != str(task_id) or scope.get("artifact_id") != str(row["artifact_id"])
            or envelope.get("tool") != "edit_account_note" or envelope.get("parameters_digest") != digest(scope)
            or result.get("effect_committed") is not True or result.get("outcome") != "completed"
            or result.get("operation_id") != str(operation_id)
            or artifact.get("artifact_id") != str(row["artifact_id"])
            or artifact.get("version_id") != row["result_version_id"]):
        raise TaskError("note_result_receipt_mismatch", "The note effect and its action receipt disagree.")
    source_row = await connection.fetchrow(VERSION_SQL + " and v.document_id=$3",
        owner_id, row["area_id"], row["result_version_id"])
    source = SourceRepository._version(source_row, historical=True)
    if source.source_id != row["source_id"]:
        raise TaskError("note_result_receipt_mismatch", "The saved version belongs to another note source.")
    bound = await connection.fetchval("""select exists(select 1 from jarvis_context_payload_dependencies
        where owner_id=$1 and source_payload_id=$2 and dependent_payload_id=$3)""",
        owner_id, source.payload_id, row["receipt_payload_id"])
    if not bound:
        raise TaskError("note_result_receipt_mismatch", "The receipt lacks its saved-source dependency.")
    return row, source


async def note_result_source(connection, row, task, payload):
    """Validate after Attention's normal task/manifest/availability fences."""
    from app.attention.notifications import NotificationOutput
    try:
        result = NoteResult.model_validate(payload)
        if (result.owner_id != task.owner_id or result.task_id != task.id or result.task_revision != task.revision
                or result.area_id != task.knowledge_area_id or result.artifact_id != account_note_id(task.owner_id)
                or result.authorship != {"kind": "runtime"}):
            raise ValueError("Note result binding changed")
        roots = [result.request_payload_id]
        current = None
        if result.effect_committed:
            if result.outcome != "completed" or result.publication_operation_id is None:
                raise ValueError("Missing committed operation")
            saved, source = await commit_proof(connection, task.owner_id, task.id, result.publication_operation_id)
            if (source.document_id, source.source_id, source.revision, source.payload_id, saved["receipt_payload_id"]) != (
                    result.version_id, result.source_id, result.source_revision, result.source_payload_id, result.action_receipt_payload_id):
                raise ValueError("Saved source changed")
            roots.extend((source.payload_id, saved["receipt_payload_id"]))
            current = source.is_current
        elif result.outcome == "completed" or any(getattr(result, key) is not None for key in (
                "publication_operation_id", "version_id", "source_id", "source_revision", "source_payload_id", "action_receipt_payload_id")):
            raise ValueError("Noncommitted result claims a saved source")
        linked = await connection.fetchval("""select count(distinct d.source_payload_id)
            from jarvis_context_payload_dependencies d join jarvis_context_payloads p
                on p.owner_id=d.owner_id and p.id=d.source_payload_id
            where d.owner_id=$1 and d.dependent_payload_id=$2 and d.source_payload_id=any($3::uuid[])
                and p.availability='available'""", task.owner_id, row["source_payload_id"], roots)
        if linked != len(set(roots)):
            raise ValueError("Missing result source dependencies")
        return NotificationOutput(notification(result, is_current=current)), None
    except (ValidationError, ValueError, TypeError, KeyError, TaskError, SourceError):
        return None, "note_result_unavailable"
