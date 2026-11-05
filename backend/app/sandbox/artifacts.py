"""Native workspace output publication through the shared generated-file producer."""
import asyncio
from uuid import uuid4

from pydantic import Field, ValidationError, field_validator

from app.evidence.models import FrozenRecord
from app.files.generated import ArtifactRequest, ArtifactSnapshot
from app.files.local_file import VerifiedLocalFile
from app.files.transfer import FileTransferPolicy, MP4_OUTPUT_POLICY
from app.sandbox.file_transfer import workspace_file
from app.tasks.generated_artifacts import GeneratedArtifactPublisher, PUBLISH_CAPABILITY, PUBLISH_TOOL
from app.tasks.models import ClaimLost, TaskError, json_object
from app.tools.inputs import failed, success
from app.tools.registry import ToolDefinition


class PublishFile(ArtifactRequest):
    path: str = Field(min_length=1, max_length=512, description="Existing workspace-relative regular file.")

    @field_validator("path")
    @classmethod
    def relative_file(cls, value):
        return workspace_file(value)


class ArtifactLookup(FrozenRecord):
    operation_key: str | None = Field(min_length=1, max_length=100)


class SandboxArtifactTools:
    def __init__(self, bridge, broker, files, grant_id, policy):
        self.bridge, self.context, self.broker = bridge, bridge.context, broker
        self.publisher = GeneratedArtifactPublisher(bridge, files, grant_id)
        self.policy = policy

    @classmethod
    async def attach(cls, bridge, broker, files, configuration):
        context = bridge.context
        async with context.tasks.transaction(context.task.owner_id) as connection:
            task, _ = await context.tasks.lock_claim(connection, context.claim)
            rows = await connection.fetch("""select g.grant_id from jarvis_task_grants g
                join jarvis_operation_grants p on p.owner_id=g.owner_id and p.id=g.grant_id
                where g.owner_id=$1 and g.task_id=$2 and p.capability=$3""",
                task.owner_id, task.id, PUBLISH_CAPABILITY)
        if not rows:
            return None
        if configuration is None or files is None:
            raise TaskError("artifact_worker_unavailable", "This worker has no configured source-backed file publisher.", 503)
        if len(rows) != 1:
            raise TaskError("artifact_grant_ambiguous", "Use one generated-file allowance per task.")
        result = cls(bridge, broker, files, rows[0]["grant_id"], configuration.policy)
        async with files.sources.transaction(task.owner_id) as connection:
            _, granted = await result.publisher._parent(connection)
            if not configuration.policy.accepts(granted):
                raise TaskError("artifact_policy_exceeds_worker", "This output allowance exceeds the worker's configured bounds.")
            result.policy = granted
            children = await connection.fetch("""select g.id from jarvis_task_grants t
                join jarvis_operation_grants g on g.owner_id=t.owner_id and g.id=t.grant_id
                where t.owner_id=$1 and t.task_id=$2 and g.capability='files.publish_snapshot'""",
                task.owner_id, task.id)
            for child in children:
                grant = await context.tasks.validate_grant(connection, task.owner_id, child["id"])
                snapshot = ArtifactSnapshot.model_validate(json_object(grant["scope"]))
                if snapshot.parent_grant_id != result.publisher.parent_grant_id or snapshot.task_id != task.id:
                    raise TaskError("artifact_snapshot_scope", "A generated-file receipt has inconsistent task authority.")
                action = result.publisher._action(grant)
                bridge.actions[action.reference] = action
        return result

    async def _direct(self, arguments):
        raise TaskError("artifact_invocation_required", "Invoke publication through the task's scoped sandbox bridge.", 403)

    def definitions(self):
        return (
            ToolDefinition("sandbox_publish_file",
                "Publish an existing workspace file as a durable downloadable artifact. Permission is separate "
                "from sandbox computation. Choose one stable operation_key per logical output; retries reuse the "
                "same exact filename/format/description and never authorize changed bytes. Supported formats and "
                "allowance are in authorized_operations. Inspect the requested contents with sandbox commands before "
                "publishing and disclose checks you could not complete. A success receipt proves the file was "
                "saved, not that its contents are correct, valid, recalculated or malware-free. "
                "Include its task_operation.operation_ref in completed_operation_refs. A workspace path alone "
                "is not a downloadable file. On uncertainty use sandbox_artifact_status with the SAME key; do "
                "not rerun computation or choose another key merely because an acknowledgement was lost.",
                PublishFile.model_json_schema(), self._direct, mutates_state=True),
            ToolDefinition("sandbox_artifact_status",
                "Inspect or reconcile an existing generated-file operation without uploading again or needing "
                "the original VM. Use the original operation_key; null lists the task's keys and states. "
                "A not_submitted snapshot still requires original bytes; unknown is not failure.",
                ArtifactLookup.model_json_schema(), self._direct),
        )

    async def ready_allocation(self):
        ctx = self.context
        async with ctx.tasks.transaction(ctx.task.owner_id) as connection:
            await ctx.tasks.lock_claim(connection, ctx.claim)
            identity = await connection.fetchval("""select id from jarvis_sandbox_allocations
                where owner_id=$1 and task_id=$2 and attempt_id=$3 and grant_id=$4 and phase='ready'""",
                ctx.task.owner_id, ctx.task.id, ctx.claim.attempt_id, self.broker.grant_id)
        if identity is None:
            raise TaskError("artifact_workspace_unavailable", "No existing ready workspace contains this output; no replacement VM was created.")
        return await self.broker.allocations.current(ctx.claim, identity)

    async def publish(self, arguments, provider_call_id):
        value = PublishFile.model_validate(arguments)
        request = ArtifactRequest.model_validate(value.model_dump(exclude={"path"}))
        try:
            return await self.publisher.publish(request, provider_call_id=provider_call_id)
        except TaskError as error:
            if error.code != "artifact_original_required":
                raise
        allocation = await self.ready_allocation()
        content = None
        try:
            if request.format == "mp4":
                policy = getattr(self.publisher.files, "media_policy", None)
                if (not self.policy.supports_media
                        or not isinstance(policy, FileTransferPolicy) or policy.policy_id != MP4_OUTPUT_POLICY
                        or not callable(getattr(self.publisher.files, "media_inspector", None))):
                    raise TaskError("artifact_media_unavailable", "This worker has no trusted media output configuration.", 503)
                try:
                    effective = self.policy.for_format("mp4", deployment_limit_bytes=policy.max_bytes)
                    content = await self.broker.provider.read_workspace_file_snapshot(
                        self.broker.allocations.handle(allocation), value.path, policy=effective,
                        max_bytes=effective.max_bytes)
                except BaseException:
                    # Allocation identity already exists durably. An uncertain
                    # remote stage never authorizes a replacement VM or upload.
                    await asyncio.shield(self.broker.allocations.request_cleanup(
                        self.context.task.owner_id, allocation["id"], reason="artifact_transfer_incomplete"))
                    raise
            else:
                content = await self.broker.provider.read_workspace_file(
                    self.broker.allocations.handle(allocation), value.path, max_bytes=self.policy.limit_for(request.format))
            try:
                await self.broker.allocations.current(self.context.claim, allocation["id"])
                await self.bridge._check()
            except (TaskError, ClaimLost):
                await self.broker.allocations.request_cleanup(self.context.task.owner_id, allocation["id"],
                                                             reason="artifact_source_revoked")
                raise
            return await self.publisher.publish(request, content, provider_call_id=provider_call_id)
        finally:
            if isinstance(content, VerifiedLocalFile):
                content.close()

    async def status(self, arguments, provider_call_id):
        lookup = ArtifactLookup.model_validate(arguments)
        ctx = self.context
        async with self.publisher.files.sources.transaction(ctx.task.owner_id) as connection:
            await self.publisher._parent(connection)
            rows = await connection.fetch("""select g.id,p.content,a.status as action_status
                from jarvis_task_grants t
                join jarvis_operation_grants g on g.owner_id=t.owner_id and g.id=t.grant_id
                join jarvis_context_payloads p on p.owner_id=g.owner_id and p.id=g.scope_payload_id
                left join jarvis_task_actions a on a.owner_id=g.owner_id and a.task_id=t.task_id and a.grant_id=g.id
                where t.owner_id=$1 and t.task_id=$2 and g.capability='files.publish_snapshot'
                    and p.availability='available' order by g.id limit 33""", ctx.task.owner_id, ctx.task.id)
        if len(rows) > 32:
            raise TaskError("artifact_history_limit", "Artifact history exceeds its configured bound.")
        snapshots = [(row, ArtifactSnapshot.model_validate(json_object(row["content"]))) for row in rows]
        selected = [(row, item) for row, item in snapshots if item.parent_grant_id == self.publisher.parent_grant_id]
        if lookup.operation_key is None:
            return success({"artifact_operations": [{"operation_key": item.request.operation_key,
                "filename": item.request.filename, "format": item.request.format,
                "state": row["action_status"] or "not_submitted", "operation_ref": "op_" + row["id"].hex}
                for row, item in selected]})
        match = next(((row, item) for row, item in selected if item.request.operation_key == lookup.operation_key), None)
        if match is None:
            return failed("artifact_operation_not_found", "No such artifact operation exists; no VM or upload was started.")
        row, snapshot = match
        if row["action_status"] in {None, "prepared", "authorized"}:
            return success({"artifact_operation": {"operation_key": lookup.operation_key, "state": "not_submitted",
                "meaning": "Exact original bytes are still required; this is not a saved file."}})
        return await self.publisher.publish(snapshot.request, provider_call_id=provider_call_id)

    async def invoke(self, name, arguments, *, provider_call_id):
        operation = self.publish if name == "sandbox_publish_file" else self.status
        reservation = await self.context.reserve("artifact-tool:" + uuid4().hex, tool_steps=1)
        await self.context.budgets.submit(self.context.claim, reservation["id"])
        try:
            try:
                parsed = (PublishFile if name == "sandbox_publish_file" else ArtifactLookup).model_validate(arguments)
            except ValidationError:
                return failed("invalid_tool_input", "Use this artifact tool's declared fields; no new file operation was submitted.")
            # A later validation error can follow an upload. Let the outer
            # bridge report uncertain internal state, not "nothing submitted".
            return await operation(parsed.model_dump(mode="json"), provider_call_id)
        finally:
            await asyncio.shield(self.context.settle(reservation["id"], input_tokens=0, output_tokens=0, tool_steps=1))
