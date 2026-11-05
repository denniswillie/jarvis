"""Reconcile submitted task files without restoring an agent or repeating an upload."""
from __future__ import annotations

from uuid import UUID

from pydantic import Field

from app.evidence.models import FrozenRecord
from app.files.generated import ArtifactSnapshot
from app.files.tools import _file_payload, text_file_parameters
from app.files.transfer import FileTransferPolicy, MP4_OUTPUT_POLICY
from app.knowledge.models import SourceError
from app.tasks.actions import TaskActionRepository
from app.tasks.models import TaskError, TaskStatus, digest, json_object
from app.tasks.repository import ACTIVE_ATTEMPTS
from app.tools.inputs import failed, success


class TaskFileLineage(FrozenRecord):
    task_id: UUID
    area_id: UUID
    task_revision: int = Field(ge=1)
    attempt_id: UUID
    action_id: UUID


class TaskFileRecovery:
    """Use KnowledgeRuntime's existing poller and FilePublicationService effects.

    Only already-recorded task publications are eligible. No model, VM, new
    grant, upload or task restart is part of reconciliation.
    """
    def __init__(self, sources, tasks, storage, *, owner_id, area_id, access_check=None,
                 media_policy=None, media_inspector=None):
        if sources.pool is not tasks.pool:
            raise ValueError("Task file recovery requires the same source/action database")
        if media_policy is not None or media_inspector is not None:
            if (not isinstance(media_policy, FileTransferPolicy) or media_policy.policy_id != MP4_OUTPUT_POLICY
                    or not callable(media_inspector)):
                raise ValueError("Media recovery requires its trusted output policy and inspector")
        self.sources, self.tasks, self.storage = sources, tasks, storage
        self.owner_id, self.area_id, self.access_check = owner_id, area_id, access_check
        self.media_policy, self.media_inspector = media_policy, media_inspector
        self.actions = TaskActionRepository(tasks)

    def _authorize(self, owner_id):
        if owner_id != self.owner_id:
            raise SourceError("file_owner_mismatch", "File recovery is outside its owner scope.", 403)
        if self.access_check is not None:
            self.access_check(owner_id)

    def _files(self, area_id):
        from app.knowledge.files import SourceFileRepository
        return SourceFileRepository(self.sources, self.storage, owner_id=self.owner_id,
            area_id=area_id, access_check=self.access_check, task_recovery=self,
            media_policy=self.media_policy, media_inspector=self.media_inspector)

    async def _scope(self, connection, action_id, area_id):
        await self.tasks.lock_owner(connection, self.owner_id)
        action = await connection.fetchrow(
            "select * from jarvis_task_actions where owner_id=$1 and id=$2 for update",
            self.owner_id, action_id)
        if action is None or action["submission_token"] is None:
            raise TaskError("file_recovery_action_missing", "No original submitted action exists.")
        task = await self.tasks.load(connection, self.owner_id, action["task_id"], lock=True)
        origins = await connection.fetch("""select a.id,a.task_revision,a.status,
                a.lease_until>clock_timestamp() as live
            from jarvis_task_reservations r join jarvis_task_attempts a
                on a.owner_id=r.owner_id and a.id=r.attempt_id and a.task_id=r.task_id
            where r.owner_id=$1 and r.task_id=$2 and r.operation_key=$3
                and r.status in ('submitted','unknown','settled')""",
            self.owner_id, task.id, "action:" + str(action_id))
        if len(origins) != 1 or origins[0]["task_revision"] != action["task_revision"]:
            raise TaskError("file_recovery_origin_missing", "The original submission attempt is not unambiguous.")
        origin = origins[0]
        lineage = TaskFileLineage(task_id=task.id, area_id=area_id, task_revision=action["task_revision"],
                                 attempt_id=origin["id"], action_id=action_id)
        live = (task.current_attempt_id == origin["id"] and task.revision == action["task_revision"]
                and task.status == TaskStatus.RUNNING and origin["live"] and origin["status"] in ACTIVE_ATTEMPTS)
        return task, action, lineage, live

    async def _disposition(self, connection, row, action_id):
        if row["delete_requested"]:
            # A retained deletion request is sufficient cleanup authority even
            # after the task/grant or its original execution evidence expires.
            action = await connection.fetchrow(
                "select * from jarvis_task_actions where owner_id=$1 and id=$2 for update",
                self.owner_id, action_id)
            return "close", action
        task, action, lineage, live = await self._scope(connection, action_id, row["area_id"])
        if action["status"] not in {"submitting", "outcome_unknown"}:
            return "settled", action
        if (row["delete_requested"] or row["availability"] != "available"
                or task.revision != action["task_revision"]
                or task.status in {TaskStatus.CANCELLED, TaskStatus.CANCEL_REQUESTED}
                or await self.tasks.context_problem(connection, task)
                or await connection.fetchval("select lock_jarvis_copy_source($1,$2)",
                    self.owner_id, row["metadata_payload_id"]) != "available"):
            return "close", action
        if live:
            return "awaiting_task", action
        return ("close" if row["phase"] == "prepared" else "reconcile"), action

    async def authorize_adoption(self, connection, area_id, file, spec, content, metadata_payload_id):
        self._authorize(self.owner_id)
        lineage = TaskFileLineage.model_validate(spec.metadata.get("task_artifact"))
        task, action, expected, live = await self._scope(connection, lineage.action_id, area_id)
        row = await connection.fetchrow("""select f.*,p.availability from jarvis_file_publications f
            join jarvis_context_payloads p on p.owner_id=f.owner_id and p.id=f.metadata_payload_id
            where f.owner_id=$1 and f.area_id=$2 and f.file_id=$3 and f.metadata_payload_id=$4""",
            self.owner_id, area_id, file.id, metadata_payload_id)
        if row is None or (await self._disposition(connection, row, action["id"]))[0] != "reconcile":
            raise TaskError("file_recovery_not_admitted", "The original task publication is not recoverable now.")
        grant = await self.tasks.validate_grant(connection, self.owner_id, action["grant_id"])
        if not await connection.fetchval("""select exists(select 1 from jarvis_task_grants
            where owner_id=$1 and task_id=$2 and grant_id=$3)""", self.owner_id, task.id, grant["id"]):
            raise TaskError("file_recovery_grant_missing", "The submitted grant is not attached to this task.")
        scope = json_object(grant["scope"])
        if (lineage != expected or spec.session_id != task.session_id or spec.source != "assistant"
                or spec.source_visual_frame_id is not None or action["parameters_digest"] != digest(scope)
                or action["operation_key"] != "tool-grant:" + grant["id"].hex):
            raise TaskError("file_recovery_lineage_mismatch", "Stored file lineage differs from the original submission.")
        metadata = {"task_artifact": expected.model_dump(mode="json")}
        snapshot = None
        if grant["capability"] == "files.publish_snapshot":
            snapshot = ArtifactSnapshot.model_validate(scope)
            parent = await self.tasks.validate_grant(connection, self.owner_id, snapshot.parent_grant_id)
            attached = await connection.fetchval("""select exists(select 1 from jarvis_task_grants
                where owner_id=$1 and task_id=$2 and grant_id=$3)""", self.owner_id, task.id, parent["id"])
            if (not attached or parent["capability"] != "files.publish_generated"
                    or snapshot.task_id != task.id or task.knowledge_area_id != area_id
                    or (grant["request_event_id"], grant["request_revision"]) !=
                       (parent["request_event_id"], parent["request_revision"])
                    or not snapshot.matches(content) or spec.filename != snapshot.request.filename
                    or spec.description != snapshot.request.description
                    or spec.content_type != snapshot.inspection.content_type
                    or row["operation_key"] != "task-generated-file:" + str(action["id"])):
                raise TaskError("file_recovery_snapshot_mismatch", "The stored object differs from its authorized snapshot.")
            metadata["generated_artifact"] = {
                "operation_key": snapshot.request.operation_key, "format": snapshot.request.format,
                "validation": snapshot.inspection.summary(), "evidence_payload_id": str(snapshot.evidence_payload_id)}
        elif grant["capability"] == "files.create_text":
            if (set(scope) - {"filename", "content", "description"}
                    or text_file_parameters(scope) != (spec.filename, content.decode("utf-8"), spec.description)
                    or spec.content_type != "text/plain; charset=utf-8"
                    or row["operation_key"] != "task-text-file:" + str(action["id"])):
                raise TaskError("file_recovery_text_mismatch", "Stored text differs from its original exact grant.")
        else:
            raise TaskError("file_recovery_capability_mismatch", "This is not an authorized task file operation.")
        if spec.metadata != metadata:
            raise TaskError("file_recovery_lineage_mismatch", "Stored file metadata differs from its original submission.")
        return action, snapshot

    async def record_receipt(self, connection, action, file, source, *, tool_name):
        # Recovery observes storage, not an agent read. A later worker binds its
        # own source citation when replaying this exact accepted receipt.
        metadata = source.original_evidence()
        metadata.pop("content")
        file = file.model_copy(update={"memory_source": metadata, "task_evidence": None})
        await self.actions.record_receipt_in_transaction(connection, self.owner_id, action["task_id"],
            action["id"], action["submission_token"], "accepted", {"tool": tool_name,
                "parameters_digest": action["parameters_digest"], "result": success({"file": _file_payload(file)})})
        receipt = await connection.fetchval(
            "select receipt_payload_id from jarvis_task_actions where owner_id=$1 and id=$2",
            self.owner_id, action["id"])
        await connection.execute("""insert into jarvis_context_payload_dependencies
            (owner_id,dependent_payload_id,source_payload_id) values($1,$2,$3) on conflict do nothing""",
            self.owner_id, receipt, source.payload_id)

    async def _close(self, files, row, action):
        result = await files.publications.delete(owner_id=self.owner_id, operation_id=row["id"])
        if result.phase == "deleted":
            async with self.tasks.transaction(self.owner_id) as connection:
                current = await connection.fetchrow(
                    "select * from jarvis_task_actions where owner_id=$1 and id=$2 for update",
                    self.owner_id, action["id"])
                if current["status"] in {"submitting", "outcome_unknown"}:
                    tool = "publish_generated_file" if row["operation_key"].startswith("task-generated-file:") else "create_text_file"
                    await self.actions.record_receipt_in_transaction(connection, self.owner_id, current["task_id"],
                        current["id"], current["submission_token"], "rejected", {"tool": tool,
                            "parameters_digest": current["parameters_digest"],
                            "result": failed("task_file_publication_closed",
                                "The unadopted file was retired and its tracked object cleanup confirmed. It is not saved.")})
        return result

    async def reconcile_pending(self, *, owner_id, limit=16):
        self._authorize(owner_id)
        if type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("Task file recovery limit must be between 1 and 32")
        async with self.sources.pool.acquire() as connection:
            rows = await connection.fetch("""select f.id,f.area_id,a.id as action_id from jarvis_file_publications f
                join jarvis_task_actions a on a.owner_id=f.owner_id and
                    f.operation_key in ('task-generated-file:' || a.id::text,'task-text-file:' || a.id::text)
                join jarvis_context_payloads p on p.owner_id=f.owner_id and p.id=f.metadata_payload_id
                where f.owner_id=$1 and (
                    (f.phase not in ('available','deleted') and
                        (a.status in ('submitting','outcome_unknown') or f.delete_requested)
                        and (f.phase<>'quarantined' or f.delete_requested or p.availability<>'available'))
                    or (f.phase='deleted' and a.status in ('submitting','outcome_unknown')))
                order by f.updated_at,f.id limit $2""",
                owner_id, limit)
        outcomes = []
        for selected in rows:
            try:
                files = self._files(selected["area_id"])
                async with self.sources.transaction(owner_id) as connection:
                    row = await files.publications._row(connection, owner_id, selected["id"])
                    disposition, action = await self._disposition(connection, row, selected["action_id"])
                if disposition == "reconcile":
                    result = await files.publications.reconcile(owner_id=owner_id, operation_id=row["id"])
                elif disposition == "close":
                    result = await self._close(files, row, action)
                else:
                    outcomes.append({"operation_id": str(row["id"]), "phase": disposition})
                    continue
                outcomes.append({"operation_id": str(row["id"]), "phase": result.phase})
            except Exception as error:
                outcomes.append({"operation_id": str(selected["id"]), "phase": "unresolved",
                                 "error_type": type(error).__name__, "code": getattr(error, "code", None)})
        if rows:
            async with self.sources.pool.acquire() as connection:
                await connection.execute("""update jarvis_file_publications set updated_at=clock_timestamp()
                    where owner_id=$1 and id=any($2::uuid[]) and phase not in ('available','deleted')""",
                    owner_id, [row["id"] for row in rows])
        return outcomes
