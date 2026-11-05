"""Managed renderer tools over existing command, source and artifact records.

No request supplies execution paths, decode evidence, limits or authority. A
command's fixed launcher path is only a locator for an accepted saved recipe,
never permission to reuse historical workspace bytes.
"""
import asyncio
import re
import shlex
from uuid import uuid4

from pydantic import Field, ValidationError, field_validator

from app.files.local_file import VerifiedLocalFile
from app.files.transfer import FileManifest, FileTransferPolicy, MP4Inspection, MP4_OUTPUT_POLICY
from app.plugins.media import ManagedMediaJob
from app.sandbox.file_transfer import INPUT_ROOT
from app.sandbox.inputs import SavedFileInput, _MediaInput
from app.sandbox.media_decode import MediaDecodeReceipt
from app.sandbox.models import CommandRequest
from app.tasks.generated_artifacts import GeneratedArtifactPublisher
from app.tasks.models import TaskError, digest, json_object
from app.tools.inputs import failed, success
from app.tools.registry import ToolDefinition
from app.workspace.media_publication import MediaPublicationStore, publication_requests
from app.workspace.media_recipe import (
    FrozenRecord, MediaEditRecipeV1, MediaRecipeError, SourceInspection,
    TemplateInventoryEntry, canonical_recipe, validate_recipe,
)
from app.workspace.media_render import AuthorizedRenderInput, LAUNCHER, WORKSPACE, prepare_render


class MediaOperation(FrozenRecord):
    operation_key: str = Field(strict=True, min_length=1, max_length=100)

    @field_validator("operation_key")
    @classmethod
    def command_key(cls, value):
        return CommandRequest(operation_key=value, command=LAUNCHER).operation_key


class RenderMedia(MediaOperation):
    recipe: MediaEditRecipeV1


def render_tool_schema():
    """Creative fields only; deployment identity comes from the admitted job.

    Keep the full canonical recipe model for persistence and old explicit-pin
    callers. The model-facing tool never asks an agent to invent a build hash.
    """
    schema = RenderMedia.model_json_schema()
    recipe = schema["$defs"]["MediaEditRecipeV1"]
    del recipe["properties"]["template"]
    recipe["required"].remove("template")
    schema["$defs"].pop("TemplatePin")
    return schema


class ManagedMediaTools:
    def __init__(self, bridge, broker, inputs, artifacts, job, files):
        context = bridge.context
        if (not isinstance(job, ManagedMediaJob) or job.binding.owner_id != context.task.owner_id
                or not isinstance(artifacts.publisher, GeneratedArtifactPublisher)
                or artifacts.publisher.context is not context or artifacts.publisher.files is not files
                or inputs.files is not files or inputs.bridge.context is not context
                or broker.context is not context):
            raise ValueError("Media tools require the same task's trusted job, input reader and publisher")
        self.bridge, self.context, self.broker = bridge, context, broker
        self.inputs, self.artifacts, self.job, self.files = inputs, artifacts, job, files
        self.store = MediaPublicationStore(context.tasks, files)

    async def _direct(self, arguments):
        raise TaskError("media_invocation_required", "Use the current worker's scoped media bridge.", 403)

    def definitions(self):
        return (
            ToolDefinition("media_render",
                "Render one exact retained video using the admitted renderer and MediaEditRecipeV1. "
                "Keep the same operation_key on uncertainty. Use exact source/version/hash and predecessor "
                "references; never substitute current versions. Accepted recipe, submitted command, rendered "
                "bytes and published file are different outcomes. The runtime supplies the task's fixed "
                "renderer template; provide the edit fields only. No paths, inspections or limits are accepted.",
                render_tool_schema(), self._direct, mutates_state=True),
            ToolDefinition("media_status",
                "Reconcile the original media operation without allocation, staging or rendering. "
                "Unknown is not failure. Poll this same key only within the task's existing budget.",
                MediaOperation.model_json_schema(), self._direct),
            ToolDefinition("media_publish",
                "Publish the exact successful render through the separate artifact grant. Reconcile an "
                "existing publication first. Returns canonical accepted output evidence, not phone display "
                "or content-quality proof. Include returned task_operation.operation_ref in completed_operation_refs.",
                MediaOperation.model_json_schema(), self._direct, mutates_state=True),
        )

    async def _command(self, operation_key):
        """Read across attempts, so losing an attempt cannot authorize a rerender."""
        ctx = self.context
        async with ctx.tasks.transaction(ctx.task.owner_id) as connection:
            await ctx.tasks.lock_claim(connection, ctx.claim)
            rows = await connection.fetch("""select c.*,p.content as parameters,p.availability
                from jarvis_sandbox_commands c
                join jarvis_context_payloads p on p.owner_id=c.owner_id and p.id=c.parameters_payload_id
                where c.owner_id=$1 and c.task_id=$2 and c.operation_digest=$3
                order by c.created_at,c.id limit 2""",
                ctx.task.owner_id, ctx.task.id, digest({"key": operation_key}))
        if len(rows) > 1:
            raise TaskError("media_operation_ambiguous", "This key names multiple historical commands; no new work was started.")
        if rows and rows[0]["availability"] != "available":
            raise TaskError("media_plan_unavailable", "The original command evidence was withdrawn.")
        return dict(rows[0]) if rows else None

    @staticmethod
    def _plan_locator(row):
        try:
            request = CommandRequest.model_validate(json_object(row["parameters"]))
            argv = shlex.split(request.command)
            match = re.fullmatch(re.escape(INPUT_ROOT) + r"/render-([0-9a-f]{64})/input.json", argv[1])
            if len(argv) != 5 or argv[0] != LAUNCHER or match is None:
                raise ValueError("not a render command")
            return request, match.group(1)
        except (IndexError, ValueError, TypeError):
            raise TaskError("media_plan_unavailable", "The operation has no exact saved-render locator.") from None

    async def _recover(self, operation_key, provider_call_id):
        row = await self._command(operation_key)
        if row is None:
            return None, None, None
        request, plan_digest = self._plan_locator(row)
        ctx = self.context
        saved = await self.store.load_plan(ctx.task.owner_id, ctx.task.id, plan_digest)
        if saved is None:
            # Existing publication owns uncertain upload reconciliation. A
            # prepared snapshot without original bytes is NOT a saved recipe.
            receipt = await self.artifacts.status({"operation_key": "media-recipe:" + plan_digest}, provider_call_id)
            saved = await self.store.load_plan(ctx.task.owner_id, ctx.task.id, plan_digest)
            if saved is None:
                return None, row, success({"media_operation": {
                    "operation_key": operation_key, "plan_sha256": plan_digest,
                    "state": "recipe_unresolved", "recipe_publication": receipt,
                    "meaning": "No accepted provenance is available; no staging or rendering was repeated."}})
        plan = saved.plan
        if (plan.owner_id != ctx.task.owner_id or plan.task_id != ctx.task.id
                or plan.operation_key != operation_key or plan.binding != self.job.bundle.renderer
                or row["request_digest"] != digest(plan.command().model_dump(mode="json"))):
            raise TaskError("media_operation_conflict", "The saved operation differs from this task's pinned renderer.")
        plan.require_command(request)
        await self._sources(plan)
        return plan, row, None

    async def _sources(self, plan):
        await self._require_task_source(plan.recipe.recipe)
        await self.store._verify_reference(self.context.task.owner_id, plan.recipe.recipe.sources[0].file)
        previous = plan.recipe.recipe.previous_recipe
        if previous is not None:
            resolved = await self.store.previous_recipe(self.context.task.owner_id, previous)
            if resolved.edit_id != plan.recipe.recipe.edit_id:
                raise TaskError("media_predecessor_mismatch", "The saved predecessor belongs to a different edit.")
        await self.inputs.revalidate_staged()
        await self.bridge._check()

    async def _require_task_source(self, recipe):
        """Recheck durable admission before staging or recovering saved work."""
        from app.tasks.media_editing import require_media_recipe_input

        ctx = self.context
        async with ctx.tasks.transaction(ctx.task.owner_id) as connection:
            task, _ = await ctx.tasks.lock_claim(connection, ctx.claim)
            return await require_media_recipe_input(ctx.tasks, connection, task, recipe)

    async def _output(self, plan):
        output = await self.store.load_accepted_output(self.context.task.owner_id, self.context.task.id, plan.digest)
        await self.bridge._check()
        return output

    @staticmethod
    def _rendered(command):
        return (command.get("state") == "completed" and command.get("started") is True
                and type(command.get("exit_code")) is int and command["exit_code"] == 0
                and command.get("timed_out") is False and command.get("runner_error") is None)

    def _result(self, plan, *, command=None, output=None, publication=None):
        state = "accepted"
        if output is not None:
            state = "published"
        elif command is not None:
            phase = command.get("state")
            state = ("rendered" if self._rendered(command) else
                     "completed_unsuccessfully" if phase == "completed" else
                     "submitted" if phase in {"submitting", "running"} else
                     "accepted" if phase == "prepared" else "unknown")
        value = {"operation_key": plan.operation_key, "plan_sha256": plan.digest,
                 "state": state, "recipe_publication": "accepted",
                 "meaning": "Rendered is a successful command receipt, not published or verified content quality."}
        if command is not None:
            value["command"] = command
        if output is not None:
            value["output"] = output.model_dump(mode="json")
        if publication is not None:
            value["publication"] = publication
        result = success({"media_operation": value})
        if publication is not None and "task_operation" in publication:
            result["task_operation"] = publication["task_operation"]
        return result

    async def _observe(self, plan, row):
        if row["attempt_id"] != self.context.claim.attempt_id:
            raise TaskError("media_original_attempt_unavailable", "Only the original attempt may observe unfinished work; no replacement was allocated.")
        await self._sources(plan)
        result = await self.broker.observe(row["id"])
        await self._sources(plan)
        return result["sandbox_command"]

    async def _decode(self, handle, staged, measured, reference):
        decode = getattr(self.broker.provider, "decode_media_render_input", None)
        if not callable(decode):
            raise TaskError("media_decode_unavailable", "This provider has no trusted complete-decode capability.", 503)
        raw = await decode(handle, input_path=staged["path"], content_sha256=reference.content_sha256,
                           size_bytes=measured.size_bytes, binding=self.job.bundle.renderer)
        try:
            if not isinstance(raw, MediaDecodeReceipt):
                raise ValueError("the provider's typed decode receipt is required")
            # Revalidate fields even if a trusted dependency constructed its
            # model without validation. Metadata inspection never supplies flags.
            receipt = MediaDecodeReceipt.model_validate({
                name: getattr(raw, name) for name in MediaDecodeReceipt.model_fields})
            if (receipt.sandbox_id != handle.sandbox_id or receipt.allocation_id != handle.scope.allocation_id
                    or receipt.input_path != staged["path"] or receipt.content_sha256 != reference.content_sha256
                    or receipt.size_bytes != measured.size_bytes
                    or receipt.software_sha256 != self.job.bundle.renderer.software_sha256
                    or receipt.duration_ms != measured.duration_ms
                    or receipt.audio_streams != int(measured.audio_codec is not None)
                    or receipt.video_decoded is not True
                    or receipt.audio_decoded is not (receipt.audio_streams == 1)
                    or (receipt.audio_streams == 0 and receipt.audio_duration_us != 0)
                    or (receipt.audio_streams == 1 and receipt.audio_duration_us <= 0)):
                raise ValueError("decode evidence differs from the exact staged original")
        except (ValidationError, ValueError, TypeError, AttributeError):
            raise TaskError("media_decode_invalid", "No complete decode receipt matches the exact source and pinned renderer.") from None
        return SourceInspection(file=reference, duration_ms=measured.duration_ms, size_bytes=measured.size_bytes,
            video_decodable=receipt.video_decoded, audio_streams=receipt.audio_streams,
            audio_decodable=receipt.audio_decoded)

    @staticmethod
    def _recipe_unsubmitted(pending):
        receipt = pending["media_operation"]["recipe_publication"]
        return ((receipt.get("status") == "failed"
                 and receipt.get("error", {}).get("code") == "artifact_operation_not_found")
                or (receipt.get("status") == "success"
                    and receipt.get("artifact_operation", {}).get("state") == "not_submitted"))

    async def _prepared_allocation(self, original):
        """Read the original identities only; never reserve or prepare a VM."""
        ctx = self.context
        if original["attempt_id"] != ctx.claim.attempt_id:
            raise TaskError("media_original_attempt_unavailable", "The prepared command belongs to another attempt.")
        current = await self.broker.commands.current(ctx.claim, original["id"])
        if (original["phase"] != "prepared" or original.get("submission_token") is not None
                or current["phase"] != "prepared" or current.get("submission_token") is not None):
            raise TaskError("media_reconstruction_unavailable", "Only the still-unsubmitted original can reconstruct its recipe.")
        if (any(current[key] != original[key] for key in
                ("id", "attempt_id", "allocation_id", "request_digest"))
                or json_object(current["parameters"]) != json_object(original["parameters"])):
            raise TaskError("media_operation_conflict", "The original command identity changed during recovery.")
        request, _ = self._plan_locator(current)
        if digest(request.model_dump(mode="json")) != current["request_digest"]:
            raise TaskError("media_operation_conflict", "The original command fingerprint is inconsistent.")
        allocation = await self.broker.allocations.current(ctx.claim, original["allocation_id"])
        if allocation["id"] != original["allocation_id"] or allocation["phase"] != "ready":
            raise TaskError("media_original_allocation_unavailable", "The original allocation is not ready; no replacement was created.")
        return allocation

    async def _fresh_plan(self, request, *, original=None):
        recipe, renderer, ctx = request.recipe, self.job.bundle.renderer, self.context
        if recipe.template != renderer.template:
            raise TaskError("media_template_unavailable", "Use this task's exact admitted renderer template.")
        await self._require_task_source(recipe)
        inspect_input = getattr(getattr(self.files, "media_inspector", None), "inspect_retained_input", None)
        if not callable(inspect_input):
            raise TaskError("media_inspection_unavailable", "No trusted retained-input metadata inspector is configured.", 503)
        if not callable(getattr(self.broker.provider, "decode_media_render_input", None)):
            raise TaskError("media_decode_unavailable", "Complete decode evidence is required before rendering.", 503)
        if original is not None:
            await self._prepared_allocation(original)
        async with self.files.sources.transaction(ctx.task.owner_id) as connection:
            _, policy = await self.artifacts.publisher._parent(connection)
            if not {"json", "mp4"}.issubset(policy.formats):
                raise TaskError("media_publication_not_granted", "Both exact recipe and MP4 publication must be separately granted.", 403)
        previous = (None if recipe.previous_recipe is None else
                    await self.store.previous_recipe(ctx.task.owner_id, recipe.previous_recipe))
        reference = recipe.sources[0].file
        await self.store._verify_reference(ctx.task.owner_id, reference)
        measured = None

        async def read_original(selection):
            nonlocal measured
            content = None
            try:
                content, metadata, revalidate = await self.inputs.saved_file(
                    selection, max_bytes=renderer.policy.max_source_bytes)
                if (not isinstance(content, _MediaInput) or not isinstance(content.local_file, VerifiedLocalFile)
                        or content.local_file.content_sha256 != reference.content_sha256
                        or content.local_file.size_bytes > renderer.policy.max_source_bytes):
                    raise TaskError("media_source_mismatch", "The admitted original is not the exact bounded retained video.")
                measured = await inspect_input(content.local_file, content.policy)
                if not isinstance(measured, MP4Inspection):
                    raise TaskError("media_inspection_invalid", "A trusted exact-byte metadata inspection is required.")
                measured.validate_file(FileManifest.from_file(content.local_file), content.policy)
                if measured.duration_ms > renderer.policy.max_source_duration_ms:
                    raise TaskError("media_source_limit", "The exact original exceeds the pinned renderer duration.")

                async def exact():
                    await revalidate()
                    await self.store._verify_reference(ctx.task.owner_id, reference)
                    await self.bridge._check()

                await exact()
                if original is not None:
                    await self._prepared_allocation(original)
                return content, metadata, exact
            except BaseException:
                if isinstance(content, _MediaInput):
                    content.local_file.close()
                raise

        stage_options = {} if original is None else {"allocation_id": original["allocation_id"]}
        staged = (await self.inputs.stage(read_original, SavedFileInput(file_id=reference.file_id),
                                         **stage_options))["sandbox_input"]
        allocation = (await self.artifacts.ready_allocation() if original is None
                      else await self._prepared_allocation(original))
        try:
            handle = self.broker.allocations.handle(allocation)
            self.job.bundle.require_image(handle)
            if (staged.get("sha256") != reference.content_sha256 or staged.get("size_bytes") != measured.size_bytes
                    or staged.get("read_only") is not True
                    or re.fullmatch(re.escape(INPUT_ROOT) + r"/[0-9a-f]{64}", staged.get("path", "")) is None):
                raise TaskError("media_staging_invalid", "Staging did not return the exact protected original.")
            inspection = await self._decode(handle, staged, measured, reference)
            checked = validate_recipe(recipe, inspections=[inspection], previous=previous, policy=renderer.policy,
                inventory=[TemplateInventoryEntry(template=renderer.template, output_profile=recipe.output_profile, ready=True)])
            plan = prepare_render(owner_id=ctx.task.owner_id, task_id=ctx.task.id, operation_key=request.operation_key,
                recipe=checked, source=AuthorizedRenderInput(ctx.task.owner_id, ctx.task.id, handle.sandbox_id,
                                                            staged["path"], inspection), binding=renderer, previous=previous)
            await self._sources(plan)
        except BaseException:
            await asyncio.shield(self.broker.allocations.request_cleanup(ctx.task.owner_id, allocation["id"],
                                                                        reason="media_input_unverified"))
            raise
        if original is not None:
            persisted, plan_digest = self._plan_locator(original)
            if (plan.digest != plan_digest or plan.command() != persisted
                    or digest(plan.command().model_dump(mode="json")) != original["request_digest"]):
                raise TaskError("media_operation_conflict", "Reconstruction differs from the original plan or command; no recipe was published.")
            await self._prepared_allocation(original)
            return plan, original
        # Existing command uniqueness pins operation -> exact plan before any
        # recipe publication. Only execute_media may materialize/submit it.
        row = await self.broker.commands.prepare(ctx.claim, allocation["id"], plan.command())
        return plan, row

    async def render(self, request, provider_call_id):
        plan, row, pending = await self._recover(request.operation_key, provider_call_id)
        original = None
        if pending is not None:
            if row["phase"] != "prepared" or not self._recipe_unsubmitted(pending):
                return pending
            original = row
        if plan is not None and plan.recipe.canonical_bytes != canonical_recipe(request.recipe):
            raise TaskError("media_operation_conflict", "The original operation key cannot name a changed recipe.")
        if plan is None:
            plan, row = await self._fresh_plan(request, original=original)
            receipt = await self.store.publish_recipe(plan, self.artifacts.publisher, provider_call_id=provider_call_id)
            saved = await self.store.load_plan(self.context.task.owner_id, self.context.task.id, plan.digest)
            if saved is None:
                return success({"media_operation": {"operation_key": plan.operation_key,
                    "plan_sha256": plan.digest, "state": "recipe_unresolved", "recipe_publication": receipt}})
            if saved.plan != plan:
                raise TaskError("media_operation_conflict", "The accepted recipe differs from the prepared command.")
        output = await self._output(plan)
        if output is not None:
            return self._result(plan, output=output)
        if row["phase"] != "prepared":
            return self._result(plan, command=await self._observe(plan, row))
        if row["attempt_id"] != self.context.claim.attempt_id:
            raise TaskError("media_original_attempt_unavailable", "The prepared original belongs to another attempt.")
        await self._sources(plan)
        result = await self.broker.execute_media(plan)
        await self._sources(plan)
        return self._result(plan, command=result["sandbox_command"])

    async def status(self, request, provider_call_id):
        plan, row, pending = await self._recover(request.operation_key, provider_call_id)
        if pending is not None:
            return pending
        if plan is None:
            return failed("media_operation_not_found", "No original render exists; no allocation or staging was started.")
        output = await self._output(plan)
        return self._result(plan, output=output, command=None if output is not None else await self._observe(plan, row))

    async def publish(self, request, provider_call_id):
        plan, row, pending = await self._recover(request.operation_key, provider_call_id)
        if pending is not None:
            return pending
        if plan is None:
            return failed("media_operation_not_found", "Publish only an existing exact render operation.")
        # Reconcile the original upload before consulting workspace lifetime.
        try:
            receipt = await self.store.publish_output(plan, self.artifacts.publisher, provider_call_id=provider_call_id)
        except TaskError as error:
            if error.code != "artifact_original_required":
                raise
        else:
            output = await self._output(plan)
            return self._result(plan, output=output, publication=receipt)
        command = await self._observe(plan, row)
        if not self._rendered(command):
            return self._result(plan, command=command)
        allocation = await self.artifacts.ready_allocation()
        content = None
        try:
            handle = self.broker.allocations.handle(allocation)
            plan.require_allocation(handle)
            if allocation["id"] != row["allocation_id"]:
                raise TaskError("media_original_allocation_required", "The output must remain in its original command allocation.")
            policy = getattr(self.files, "media_policy", None)
            if not isinstance(policy, FileTransferPolicy) or policy.policy_id != MP4_OUTPUT_POLICY:
                raise TaskError("media_publication_unavailable", "No trusted MP4 output transfer policy is configured.", 503)
            effective = self.artifacts.policy.for_format("mp4", deployment_limit_bytes=policy.max_bytes)
            try:
                content = await self.broker.provider.read_workspace_file_snapshot(handle,
                    plan.output_path.removeprefix(WORKSPACE + "/"), policy=effective,
                    max_bytes=min(effective.max_bytes, plan.binding.max_output_bytes))
                if not isinstance(content, VerifiedLocalFile):
                    raise TaskError("media_output_unverified", "Output readback did not return exact owned bytes.")
                content._check()
                await self.broker.allocations.current(self.context.claim, allocation["id"])
                await self._sources(plan)
            except BaseException:
                await asyncio.shield(self.broker.allocations.request_cleanup(self.context.task.owner_id,
                    allocation["id"], reason="media_output_unverified"))
                raise
            receipt = await self.store.publish_output(plan, self.artifacts.publisher, content,
                                                     provider_call_id=provider_call_id)
            return self._result(plan, command=command, output=await self._output(plan), publication=receipt)
        finally:
            if isinstance(content, VerifiedLocalFile):
                content.close()

    async def invoke(self, name, arguments, *, provider_call_id):
        models = {"media_render": RenderMedia, "media_status": MediaOperation, "media_publish": MediaOperation}
        if name not in models:
            raise TaskError("media_tool_not_allowed", "Only this managed renderer's declared tools are available.", 403)
        if (not isinstance(provider_call_id, str) or not 1 <= len(provider_call_id) <= 200
                or not provider_call_id.strip() or "\x00" in provider_call_id):
            raise ValueError("A bounded actual provider call identity is required")
        reservation = await self.context.reserve("media-tool:" + uuid4().hex, tool_steps=1)
        await self.context.budgets.submit(self.context.claim, reservation["id"])
        try:
            try:
                if (name == "media_render" and isinstance(arguments, dict)
                        and isinstance(arguments.get("recipe"), dict)
                        and "template" not in arguments["recipe"]):
                    # Copy rather than modify model/provider-call evidence.
                    # Explicit legacy pins are left intact for canonical
                    # equality/replay checks; mismatches are never repaired.
                    arguments = arguments | {"recipe": arguments["recipe"] | {
                        "template": self.job.bundle.renderer.template.model_dump(mode="json")}}
                request = models[name].model_validate(arguments)
            except ValidationError:
                return failed("invalid_tool_input", "Use only the declared exact recipe or operation key fields.")
            await self.bridge._check()
            operation = {"media_render": self.render, "media_status": self.status, "media_publish": self.publish}[name]
            try:
                return await operation(request, provider_call_id)
            except MediaRecipeError as error:
                raise TaskError(error.code, error.message) from None
        finally:
            await asyncio.shield(self.context.settle(reservation["id"], input_tokens=0, output_tokens=0, tool_steps=1))
