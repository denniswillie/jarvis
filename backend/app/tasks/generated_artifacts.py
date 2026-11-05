"""Trusted generated-byte publication using bounded grants and existing action receipts.

This is the producer adapter, not an enabled model tool or a grant issuer for
user requests. Runtime admission must explicitly attach files.publish_generated.
"""
from __future__ import annotations

import asyncio

from app.files.generated import ArtifactPolicy, ArtifactRequest, ArtifactSnapshot, inspect_artifact, inspect_file_artifact
from app.files.local_file import VerifiedLocalFile
from app.files.tools import _file_payload
from app.files.transfer import FileManifest, FileTransferPolicy
from app.tasks.artifact_evidence import freeze_artifact_evidence
from app.tasks.models import ClaimLost, TaskError, digest, json_object
from app.tasks.tool_bridge import GrantedToolAction
from app.tools.inputs import success
from app.tools.registry import ToolDefinition

PUBLISH_CAPABILITY = "files.publish_generated"
SNAPSHOT_CAPABILITY = "files.publish_snapshot"
PUBLISH_TOOL = "publish_generated_file"


class GeneratedArtifactPublisher:
    def __init__(self, bridge, files, parent_grant_id):
        if files.task_evidence is None or files.task_evidence.context is not bridge.context:
            raise ValueError("Generated artifacts require the same worker's source-backed file adapter")
        self.bridge, self.context, self.files = bridge, bridge.context, files
        self.parent_grant_id = parent_grant_id

    async def _parent(self, connection):
        tasks, ctx = self.context.tasks, self.context
        await tasks.lock_owner(connection, ctx.task.owner_id)
        task, _ = await tasks.lock_claim(connection, ctx.claim)
        if await tasks.context_problem(connection, task):
            raise ClaimLost()
        attached = await connection.fetchval("""select exists(select 1 from jarvis_task_grants
            where owner_id=$1 and task_id=$2 and grant_id=$3)""",
            task.owner_id, task.id, self.parent_grant_id)
        grant = await tasks.validate_grant(connection, task.owner_id, self.parent_grant_id)
        if not attached or grant["capability"] != PUBLISH_CAPABILITY or task.session_id is None:
            raise TaskError("artifact_publication_not_granted", "This task has no generated-file publication scope.", 403)
        return grant, ArtifactPolicy.model_validate(json_object(grant["scope"]))

    def _key(self, request):
        return "artifact-snapshot:" + self.context.task.id.hex + ":" + digest({"operation_key": request.operation_key})

    @staticmethod
    def _action(grant):
        return GrantedToolAction(grant["id"], SNAPSHOT_CAPABILITY, PUBLISH_TOOL,
            json_object(grant["scope"]), grant["request_event_id"], grant["request_revision"])

    async def _existing(self, connection, request):
        owner, task_id = self.context.task.owner_id, self.context.task.id
        row = await connection.fetchrow("""select g.id from jarvis_operation_grants g
            join jarvis_task_grants t on t.owner_id=g.owner_id and t.grant_id=g.id
            where g.owner_id=$1 and t.task_id=$2 and g.operation_key=$3""", owner, task_id, self._key(request))
        if row is None:
            return None
        grant = await self.context.tasks.validate_grant(connection, owner, row["id"])
        snapshot = ArtifactSnapshot.model_validate(json_object(grant["scope"]))
        if (grant["capability"] != SNAPSHOT_CAPABILITY or snapshot.parent_grant_id != self.parent_grant_id
                or snapshot.task_id != task_id or snapshot.request != request):
            raise TaskError("artifact_operation_conflict", "This artifact key already names different output.")
        return self._action(grant), snapshot

    def _media_configuration(self, request, content, policy):
        transfer = getattr(self.files, "media_policy", None)
        inspector = getattr(self.files, "media_inspector", None)
        if (not isinstance(content, VerifiedLocalFile) or not isinstance(transfer, FileTransferPolicy)
                or transfer.policy_id != "mp4_output_v1" or not policy.supports_media
                or not callable(inspector)
                or not request.filename.lower().endswith(".mp4")
                or not 0 < content.size_bytes <= min(policy.limit_for("mp4"), transfer.max_bytes)):
            raise ValueError("Fresh media needs an exact file and matching trusted output policy")
        content._check()
        return policy.for_format("mp4", deployment_limit_bytes=transfer.max_bytes), inspector

    async def prepare(self, request, content=None):
        """Reuse an accepted snapshot without needing its now-destroyed producer VM."""
        request = ArtifactRequest.model_validate(request)
        owner, tasks = self.context.task.owner_id, self.context.tasks
        async with self.files.sources.transaction(owner) as connection:
            _, policy = await self._parent(connection)
            prior = await self._existing(connection, request)
        if prior is not None:
            if content is not None and not prior[1].matches(content):
                raise TaskError("artifact_operation_conflict", "This artifact key already names different bytes.")
            return prior
        if content is None:
            raise TaskError("artifact_original_required", "New publication requires exact generated bytes.")
        try:
            if request.format == "mp4":
                transfer, media_inspector = self._media_configuration(request, content, policy)
                measured = await media_inspector(content, transfer)
                inspection = inspect_file_artifact(request, content, policy, measured)
            else:
                inspection = await asyncio.to_thread(inspect_artifact, request, content, policy)
        except ValueError:
            raise TaskError("artifact_publication_invalid", "The original exceeds its file allowance or declared delivery-type policy; no upload was submitted.", 422) from None
        async with self.files.sources.transaction(owner) as connection:
            parent, policy = await self._parent(connection)
            prior = await self._existing(connection, request)
            if prior is not None:
                if not prior[1].matches(content):
                    raise TaskError("artifact_operation_conflict", "This artifact key already names different bytes.")
                return prior
            if request.format == "mp4":
                try:
                    # Inspection did not hold a transaction. Recheck the live
                    # parent and effective repository policy before freezing.
                    current_transfer, current_inspector = self._media_configuration(request, content, policy)
                    if current_inspector is not media_inspector:
                        raise ValueError("The trusted media inspector changed")
                    measured.validate_file(FileManifest.from_file(content), current_transfer)
                    inspection = inspect_file_artifact(request, content, policy, measured)
                except ValueError:
                    raise TaskError("artifact_publication_invalid",
                        "The exact media no longer meets its trusted publication policy; no upload was submitted.", 422) from None
            rows = await connection.fetch("""select p.content,p.availability from jarvis_task_grants t
                join jarvis_operation_grants g on g.owner_id=t.owner_id and g.id=t.grant_id
                join jarvis_context_payloads p on p.owner_id=g.owner_id and p.id=g.scope_payload_id
                where t.owner_id=$1 and t.task_id=$2 and g.capability=$3""",
                owner, self.context.task.id, SNAPSHOT_CAPABILITY)
            if any(row["availability"] != "available" for row in rows):
                raise TaskError("artifact_allowance_unavailable", "Removed artifact history cannot restore publication allowance.")
            snapshots = [ArtifactSnapshot.model_validate(json_object(row["content"])) for row in rows]
            used = [item for item in snapshots if item.parent_grant_id == self.parent_grant_id]
            if inspection.format not in policy.formats:
                raise TaskError("artifact_allowance_exhausted", "This output format is not granted.")
            configured = getattr(self.files, "media_policy", None)
            concrete = policy.for_format(inspection.format, deployment_limit_bytes=(
                configured.deployment_limit_bytes if isinstance(configured, FileTransferPolicy) else None))
            if (len(used) >= policy.max_files
                    or inspection.size_bytes > concrete.max_bytes
                    or sum(item.inspection.size_bytes for item in used) + inspection.size_bytes > policy.max_total_bytes):
                raise TaskError("artifact_allowance_exhausted", "Generated files exceed their publication allowance.")
            evidence = await freeze_artifact_evidence(connection, self.context, self.files.sources, self.files.area_id)
            snapshot = ArtifactSnapshot(request=request, inspection=inspection, parent_grant_id=parent["id"],
                task_id=self.context.task.id, evidence_payload_id=evidence,
                transfer_limit_bytes=(concrete.max_bytes if concrete.is_media or concrete.max_bytes != 8 * 1024 * 1024
                                      else None))
            scope = await tasks.payload(connection, owner, snapshot.model_dump(mode="json"),
                                        source_payload_ids=(parent["scope_payload_id"], evidence))
            identity = await connection.fetchval("""insert into jarvis_operation_grants
                (owner_id,capability,operation_key,request_event_id,request_revision,
                 scope_revision,scope_payload_id,expires_at)
                values($1,$2,$3,$4,$5,1,$6,$7) returning id""",
                owner, SNAPSHOT_CAPABILITY, self._key(request), parent["request_event_id"],
                parent["request_revision"], scope, parent["expires_at"])
            await connection.execute("""insert into jarvis_task_grants(owner_id,task_id,grant_id)
                values($1,$2,$3)""", owner, self.context.task.id, identity)
            grant = await tasks.validate_grant(connection, owner, identity)
            return self._action(grant), snapshot

    async def publish(self, request, content=None, *, provider_call_id):
        if (not isinstance(provider_call_id, str) or not 1 <= len(provider_call_id) <= 200
                or not provider_call_id.strip() or "\x00" in provider_call_id):
            raise ValueError("A bounded native tool-call identity is required")
        action, snapshot = await self.prepare(request, content)
        self.bridge.actions[action.reference] = action

        async def adopt(arguments):
            file = await self.files.publish_generated_snapshot(
                owner_id=self.context.task.owner_id, snapshot=snapshot, content=content)
            return success({"file": _file_payload(file)})

        tool = ToolDefinition(PUBLISH_TOOL, "Publish the exact authorized generated-file snapshot.",
                              ArtifactSnapshot.model_json_schema(), adopt, mutates_state=True)
        # The existing action table owns unknown outcomes. A recovery read must
        # reconcile the same object under its original invocation, never upload.
        async with self.context.tasks.transaction(self.context.task.owner_id) as connection:
            record = await connection.fetchrow("""select * from jarvis_task_actions
                where owner_id=$1 and task_id=$2 and operation_key=$3""",
                self.context.task.owner_id, self.context.task.id, "tool-grant:" + action.grant_id.hex)
        if record is not None and record["status"] in {"submitting", "outcome_unknown"}:
            await self.bridge._handler(tool, action.parameters, provider_call_id, action=action,
                action_id=record["id"], operation_key=record["operation_key"])
            record = await self.context.actions.get(self.context.task.owner_id, self.context.task.id, record["id"])
            return await self.bridge._replay(action, record)
        if content is None and (record is None or record["status"] in {"prepared", "authorized"}):
            raise TaskError("artifact_original_required", "An unsubmitted snapshot still requires its exact original bytes.")
        return await self.bridge.execute_granted(action, tool, provider_call_id=provider_call_id)
