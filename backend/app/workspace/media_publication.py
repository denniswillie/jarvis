"""Media recipes and accepted outputs over existing file/task/source records.

No new table, grant issuer, action ledger, renderer or presentation authority.
Core supplies the existing scoped file repository and generated-file publisher.
Reads use accepted receipts plus current source access; they never publish again.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from uuid import UUID

from app.files.generated import ArtifactRequest, ArtifactSnapshot
from app.files.local_file import VerifiedLocalFile
from app.files.workspace_access import file_version
from app.knowledge.models import MemorySourceReference
from app.knowledge.product_files import read_file_original
from app.tasks.models import digest, json_object
from app.workspace.media_recipe import (
    MediaRecipeError, PreviousRecipe, RetainedRef, SourceInspection, TemplateInventoryEntry,
    parse_recipe, validate_recipe,
)
from app.workspace.media_render import (
    AcceptedMediaOutput, AuthorizedRenderInput, MediaRenderPlan, RendererBinding, prepare_render,
)


MAX_PLAN_BYTES = 131072


def unavailable(message="The exact media publication is unavailable or no longer matches its sources."):
    raise MediaRecipeError("media_publication_unavailable", message)


def _identity(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("An exact render-plan digest is required")
    return value


def publication_requests(plan: MediaRenderPlan):
    """Two stable identities within the existing generated-file allowance."""
    if (not isinstance(plan, MediaRenderPlan) or len(plan.provenance_bytes) > MAX_PLAN_BYTES
            or hashlib.sha256(plan.provenance_bytes).hexdigest() != plan.digest):
        unavailable("The prepared plan does not match its immutable provenance.")
    stem = "edit-" + plan.recipe.recipe.edit_id.hex + "-" + plan.digest[:12]
    return (
        ArtifactRequest(operation_key="media-recipe:" + plan.digest, filename=stem + ".json", format="json",
                        description="Exact edit instructions and rendering provenance."),
        ArtifactRequest(operation_key="media-output:" + plan.digest, filename=stem + ".mp4", format="mp4",
                        description="Rendered video for saved edit instructions " + plan.digest + "."),
    )


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate stored plan field")
        value[key] = item
    return value


def restore_plan(raw: bytes, *, owner_id: str, task_id: UUID, plan_digest: str) -> MediaRenderPlan:
    """Reconstruct accepted historical data, never execution/source permission.

    The loader establishes accepted publication and current source access first.
    A previous sandbox identity/path is historical; Core must admit/stage a new
    edit afresh. Full ancestry is owned by existing source dependencies, not a
    recursive second history store in this adapter.
    """
    _identity(plan_digest)
    if (type(raw) is not bytes or not 0 < len(raw) <= MAX_PLAN_BYTES
            or hashlib.sha256(raw).hexdigest() != plan_digest):
        unavailable("Stored plan bytes differ from their exact retained digest.")
    try:
        data = json.loads(raw, object_pairs_hook=_unique)
        if data["owner_id"] != owner_id or UUID(data["task_id"]) != task_id:
            unavailable("The saved plan belongs to a different owner or task.")
        recipe = parse_recipe(data["recipe"])
        binding = RendererBinding.model_validate(data["renderer"])
        inspection = SourceInspection.model_validate(data["inspection"])
        previous = (None if recipe.previous_recipe is None else
                    PreviousRecipe(file=recipe.previous_recipe, edit_id=recipe.edit_id))
        checked = validate_recipe(recipe, inspections=[inspection], policy=binding.policy, previous=previous,
            inventory=[TemplateInventoryEntry(template=binding.template, output_profile=recipe.output_profile, ready=True)])
        plan = prepare_render(owner_id=owner_id, task_id=task_id, operation_key=data["operation_key"],
            recipe=checked, source=AuthorizedRenderInput(owner_id, task_id, data["sandbox_id"], data["input_path"], inspection),
            binding=binding, previous=previous)
        if plan.digest != plan_digest or plan.provenance_bytes != raw:
            unavailable("The stored recipe, layout or provenance is inconsistent.")
        return plan
    except (KeyError, TypeError, ValueError):
        unavailable("The saved media plan is invalid or inconsistent.")


@dataclass(frozen=True)
class SavedMediaPlan:
    plan: MediaRenderPlan
    file: RetainedRef
    action_id: UUID


@dataclass(frozen=True)
class _AcceptedFile:
    file: object
    snapshot: ArtifactSnapshot
    action_id: UUID


class MediaPublicationStore:
    def __init__(self, tasks, files):
        if tasks.pool is not files.sources.pool:
            raise ValueError("Media publication must use the existing task/source database")
        self.tasks, self.files = tasks, files

    async def _accepted_file(self, owner_id, task_id, operation_key, format):
        self.files._authorize(owner_id)
        if not isinstance(task_id, UUID):
            raise ValueError("An exact task UUID is required")
        # Parent grant expiry does not erase historical acceptance. Removed
        # scope/receipt/source payloads do deny access. No write or claim renewal.
        async with self.tasks.pool.acquire() as connection:
            rows = await connection.fetch("""select a.id,a.grant_id,a.operation_key,a.parameters_digest,
                a.parameters_payload_id,g.scope_payload_id,g.operation_key as grant_operation_key,
                p.content as receipt,s.content as scope,t.knowledge_area_id
                from jarvis_task_actions a
                join jarvis_background_tasks t on t.owner_id=a.owner_id and t.id=a.task_id
                join jarvis_task_grants tg on tg.owner_id=a.owner_id and tg.task_id=a.task_id and tg.grant_id=a.grant_id
                join jarvis_operation_grants g on g.owner_id=a.owner_id and g.id=a.grant_id
                join jarvis_context_payloads p on p.owner_id=a.owner_id and p.id=a.receipt_payload_id
                join jarvis_context_payloads s on s.owner_id=g.owner_id and s.id=g.scope_payload_id
                where a.owner_id=$1 and a.task_id=$2 and a.status='accepted'
                  and g.capability='files.publish_snapshot' and p.availability='available' and s.availability='available'
                  and s.content->'request'->>'operation_key'=$3 limit 2""", owner_id, task_id, operation_key)
            if not rows:
                return None
            if len(rows) != 1:
                unavailable("More than one accepted action claims the same media publication.")
            row = rows[0]
            try:
                scope = json_object(row["scope"])
                snapshot = ArtifactSnapshot.model_validate(scope)
                receipt = json_object(row["receipt"])
                saved = receipt["result"]["file"]
                saved_id = UUID(saved["id"])
                saved_source = MemorySourceReference.from_metadata(saved["memory_source"])
                saved_version = saved["memory_source"]["document_id"]
            except (KeyError, TypeError, ValueError):
                unavailable("The accepted publication has invalid retained metadata.")
            if (snapshot.task_id != task_id or snapshot.request.operation_key != operation_key
                    or snapshot.request.format != format or snapshot.inspection.format != format
                    or row["parameters_payload_id"] != row["scope_payload_id"]
                    or row["parameters_digest"] != digest(scope) or receipt.get("parameters_digest") != digest(scope)
                    or row["operation_key"] != "tool-grant:" + row["grant_id"].hex
                    or row["grant_operation_key"] != "artifact-snapshot:" + task_id.hex + ":" + digest({"operation_key": operation_key})
                    or receipt.get("tool") != "publish_generated_file"
                    or receipt.get("result", {}).get("status") != "success"
                    or row["knowledge_area_id"] is None):
                unavailable("The accepted action and exact generated-file grant disagree.")
            parent = await connection.fetchval("""select exists(select 1 from jarvis_task_grants tg
                join jarvis_operation_grants g on g.owner_id=tg.owner_id and g.id=tg.grant_id
                where tg.owner_id=$1 and tg.task_id=$2 and g.id=$3 and g.capability='files.publish_generated')""",
                owner_id, task_id, snapshot.parent_grant_id)
            if not parent:
                unavailable("The accepted output is missing its task's publication grant.")
        file = await self.files.get_file(owner_id=owner_id, file_id=saved_id)
        if file is None:
            unavailable()
        lineage = file.metadata.get("task_artifact", {})
        expected_metadata = {"operation_key": snapshot.request.operation_key, "format": format,
            "validation": snapshot.inspection.summary(), "evidence_payload_id": str(snapshot.evidence_payload_id)}
        if (lineage.get("task_id") != str(task_id) or lineage.get("action_id") != str(row["id"])
                or lineage.get("area_id") != str(row["knowledge_area_id"])
                or file.metadata.get("generated_artifact") != expected_metadata
                or file.filename != snapshot.request.filename or file.description != snapshot.request.description
                or file.content_type != snapshot.inspection.content_type or file.size_bytes != snapshot.inspection.size_bytes
                or file_version(file) != saved_version
                or MemorySourceReference.from_metadata(file.memory_source) != saved_source):
            unavailable("The current retained file differs from its accepted publication.")
        return _AcceptedFile(file, snapshot, row["id"])

    async def _verify_reference(self, owner_id, reference):
        reference = RetainedRef.model_validate(reference)
        file = await self.files.get_file(owner_id=owner_id, file_id=reference.file_id)
        if file is None or file_version(file) != reference.version_id:
            unavailable("The exact original source version is unavailable.")
        async with self.files.sources.pool.acquire() as connection:
            area = await self.files._area_for(connection, file.id)
            original, _ = await read_file_original(connection, self.files.sources, owner_id, area, file, lock=False)
            if (original.content.get("content_sha256") != reference.content_sha256
                    or MemorySourceReference.from_metadata(original.original_evidence()) != MemorySourceReference.from_metadata(file.memory_source)):
                unavailable("The source bytes or source generation changed.")
        return file

    async def load_plan(self, owner_id, task_id, plan_digest):
        _identity(plan_digest)
        accepted = await self._accepted_file(owner_id, task_id, "media-recipe:" + plan_digest, "json")
        if accepted is None:
            return None
        if accepted.snapshot.inspection.size_bytes > MAX_PLAN_BYTES or accepted.snapshot.inspection.content_sha256 != plan_digest:
            unavailable("The saved recipe does not have the requested plan digest.")
        raw = await self.files.read_content(owner_id=owner_id, file_id=accepted.file.id)
        plan = restore_plan(raw, owner_id=owner_id, task_id=task_id, plan_digest=plan_digest)
        if accepted.snapshot.request != publication_requests(plan)[0]:
            unavailable("The saved recipe's publication identity changed.")
        # Bind current file/version again after the byte read. Never use a newer
        # source version as a convenient replacement for the requested history.
        reference = RetainedRef(file_id=accepted.file.id, version_id=file_version(accepted.file), content_sha256=plan_digest)
        await self._verify_reference(owner_id, reference)
        return SavedMediaPlan(plan, reference, accepted.action_id)

    async def previous_recipe(self, owner_id, reference):
        file = await self._verify_reference(owner_id, reference)
        try:
            task_id = UUID(file.metadata["task_artifact"]["task_id"])
        except (KeyError, TypeError, ValueError):
            unavailable("The predecessor is not a source-backed accepted media recipe.")
        reference = RetainedRef.model_validate(reference)
        saved = await self.load_plan(owner_id, task_id, reference.content_sha256)
        if saved is None or saved.file != reference:
            unavailable("The requested predecessor has no matching accepted publication.")
        return PreviousRecipe(file=reference, edit_id=saved.plan.recipe.recipe.edit_id)

    async def load_accepted_output(self, owner_id, task_id, plan_digest):
        """Callable seam for present_accepted_output; read-only, no publication."""
        saved = await self.load_plan(owner_id, task_id, plan_digest)
        if saved is None:
            return None
        accepted = await self._accepted_file(owner_id, task_id, "media-output:" + plan_digest, "mp4")
        if accepted is None:
            return None
        if (accepted.snapshot.request != publication_requests(saved.plan)[1]
                or accepted.file.size_bytes > saved.plan.binding.max_output_bytes):
            unavailable("The accepted video does not match its saved edit plan.")
        reference = RetainedRef(file_id=accepted.file.id, version_id=file_version(accepted.file),
                               content_sha256=accepted.snapshot.inspection.content_sha256)
        await self._verify_reference(owner_id, reference)
        await self._verify_reference(owner_id, saved.file)
        return AcceptedMediaOutput(owner_id=owner_id, task_id=task_id, action_id=accepted.action_id,
            render_plan_sha256=plan_digest, file=reference, content_type="video/mp4", size_bytes=accepted.file.size_bytes)

    def _publisher(self, plan, publisher):
        if (publisher.files is not self.files or publisher.context.tasks is not self.tasks
                or publisher.context.task.owner_id != plan.owner_id or publisher.context.task.id != plan.task_id):
            unavailable("Publication must use this task's current existing worker context.")

    async def publish_recipe(self, plan, publisher, *, provider_call_id):
        self._publisher(plan, publisher)
        await self._verify_reference(plan.owner_id, plan.recipe.recipe.sources[0].file)
        predecessor = plan.recipe.recipe.previous_recipe
        if predecessor is not None:
            previous = await self.previous_recipe(plan.owner_id, predecessor)
            if previous.edit_id != plan.recipe.recipe.edit_id:
                unavailable("The predecessor belongs to a different edit.")
        request, _ = publication_requests(plan)
        # Existing publisher reconciles the same action and bytes after unknown
        # outcomes; this adapter adds no retry or upload loop.
        return await publisher.publish(request, plan.provenance_bytes, provider_call_id=provider_call_id)

    async def publish_output(self, plan, publisher, content=None, *, provider_call_id):
        self._publisher(plan, publisher)
        saved = await self.load_plan(plan.owner_id, plan.task_id, plan.digest)
        if saved is None or saved.plan.provenance_bytes != plan.provenance_bytes:
            unavailable("Save the exact recipe before publishing its rendered output.")
        await self._verify_reference(plan.owner_id, plan.recipe.recipe.sources[0].file)
        # Reading the recipe through this worker's scoped SourceFileRepository
        # binds it into existing task evidence before the publisher freezes output
        # dependencies. A withdrawn recipe therefore cannot silently stay valid.
        if content is not None and (not isinstance(content, VerifiedLocalFile)
                                    or content.size_bytes > plan.binding.max_output_bytes):
            unavailable("Fresh output requires a bounded verified file from the fixed render path.")
        _, request = publication_requests(plan)
        return await publisher.publish(request, content, provider_call_id=provider_call_id)
