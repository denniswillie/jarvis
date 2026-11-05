"""Small adapter between exact media recipes and Core's managed task machinery.

No admission, provider allocation, installation, shell preparation, publication or
durable ledger lives here. Core supplies current source/staging authority and a
qualified binding, materializes the returned inputs, and submits the fixed
command through its existing broker. Unknown commands retain their original key.
These Python inputs are trusted composition objects, never model/HTTP authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import shlex
from typing import Literal
from uuid import UUID

from pydantic import Field

from app.files.transfer import CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS
from app.sandbox.file_transfer import INPUT_ROOT
from app.sandbox.models import CommandRequest
from app.workspace.media_recipe import (
    Digest, FrozenRecord, MediaRecipeError, MediaRecipePolicy, PreviousRecipe,
    RetainedRef, SourceInspection, TemplateInventoryEntry, TemplatePin,
    ValidatedMediaRecipe, validate_recipe,
)
from app.workspace.models import WorkspaceContext


LAUNCHER = "/opt/jarvis-render/render-clip"
WORKSPACE = "/home/jarvis/workspace"


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _reject(code, message):
    raise MediaRecipeError(code, message)


class RendererBinding(FrozenRecord):
    """Core-owned qualified mapping; constructing this does not qualify an image."""

    template: TemplatePin
    physical_template_id: str = Field(strict=True, pattern=r"^[A-Za-z0-9_-]{1,200}$")
    build_id: str = Field(strict=True, pattern=r"^[A-Za-z0-9_-]{1,200}$")
    software_sha256: Digest
    build_receipt_sha256: Digest
    policy: MediaRecipePolicy
    max_output_bytes: int = Field(strict=True, gt=0, le=64 * 1024 * 1024)


@dataclass(frozen=True)
class AuthorizedRenderInput:
    """Result of Core's current source checks and existing input staging.

    The same task/attempt must revalidate source access before execution and
    publication. A matching path or cached inspection alone grants no access.
    """

    owner_id: str
    task_id: UUID
    sandbox_id: str
    path: str
    inspection: SourceInspection


@dataclass(frozen=True)
class MediaRenderPlan:
    owner_id: str
    task_id: UUID
    operation_key: str
    sandbox_id: str
    binding: RendererBinding
    recipe: ValidatedMediaRecipe
    input_path: str
    envelope_bytes: bytes
    provenance_bytes: bytes
    digest: str

    @property
    def directory(self):
        return f"{WORKSPACE}/media/{self.task_id.hex}/{self.digest}"

    @property
    def input_directory(self):
        return f"{INPUT_ROOT}/render-{self.digest}"

    @property
    def output_path(self):
        return self.directory + "/render/output.mp4"

    def staging(self):
        """Declarative preparation for Core's protected installer, not commands.

        Core creates parent/inputs directories and the exact envelope, then
        copies the already protected source with size/hash checks. It must leave
        render/ absent for the sealed launcher's exclusive creation. Reconcile
        the existing broker operation before touching a prior attempt directory.
        """
        source = self.recipe.recipe.sources[0].file
        return {"directory": self.directory, "input_directory": self.input_directory,
                "input_file": self.input_directory + "/input.json",
                "staging_directory": self.input_directory + "/staging",
                "source_from": self.input_path,
                "source_to": self.input_directory + "/staging/inputs/" + source.content_sha256 + ".mp4",
                "content_sha256": source.content_sha256,
                "size_bytes": json.loads(self.provenance_bytes)["inspection"]["size_bytes"]}

    def command(self):
        return CommandRequest(operation_key=self.operation_key, cwd=".", command=shlex.join((
            LAUNCHER, self.input_directory + "/input.json", self.input_directory + "/staging",
            self.directory + "/render", self.output_path)))

    def require_command(self, request):
        if request != self.command():
            _reject("media_render_command_mismatch", "Use the exact pinned launcher request and original operation key.")

    def require_allocation(self, handle):
        if handle.sandbox_id != self.sandbox_id or handle.template_id != self.binding.physical_template_id:
            _reject("media_render_allocation_mismatch", "The render must use its staged source allocation and qualified image.")


def prepare_render(*, owner_id: str, task_id: UUID, operation_key: str,
                   recipe: ValidatedMediaRecipe, source: AuthorizedRenderInput,
                   binding: RendererBinding, previous: PreviousRecipe | None = None) -> MediaRenderPlan:
    """Build immutable input/provenance and one fixed launcher request, without I/O.

    Preserve provenance_bytes with the task's existing accepted publication
    metadata. They include the canonical recipe and exact predecessor/source;
    they are not a second record of job state or a success receipt.
    """
    if (not isinstance(owner_id, str) or not owner_id.strip() or not isinstance(task_id, UUID)
            or not isinstance(source, AuthorizedRenderInput) or source.owner_id != owner_id
            or source.task_id != task_id or not isinstance(source.sandbox_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", source.sandbox_id)):
        _reject("media_render_scope_mismatch", "Use the current task's exact authorized staged input.")
    if not isinstance(binding, RendererBinding) or not isinstance(recipe, ValidatedMediaRecipe):
        _reject("media_render_context_invalid", "An explicit qualified renderer and validated recipe are required.")
    binding = RendererBinding.model_validate(binding)
    # Never widen the deployed template's independently enforced limits.
    policy = MediaRecipePolicy.model_validate(binding.policy)
    if (policy.max_source_bytes > CAPTURE_MAX_BYTES or policy.max_source_duration_ms > CAPTURE_MAX_DURATION_MS
            or policy.max_output_frames > 720):
        _reject("media_render_template_limit", "The supplied policy exceeds the fixed template's limits.")
    inspection = SourceInspection.model_validate(source.inspection)
    checked = validate_recipe(recipe.recipe, inspections=[inspection], policy=policy, previous=previous,
        inventory=[TemplateInventoryEntry(template=binding.template, output_profile="portrait_720p24_v1", ready=True)])
    if checked != recipe:
        _reject("media_render_recipe_changed", "The validation result differs from its canonical recipe.")
    if recipe.recipe.audio == "preserve" and inspection.audio_streams != 1:
        _reject("media_render_audio_unsupported", "This template requires one inspected mixed audio stream.")
    if not isinstance(source.path, str) or not re.fullmatch(re.escape(INPUT_ROOT) + r"/[0-9a-f]{64}", source.path):
        _reject("media_render_input_path_invalid", "Use the protected input path returned by existing staging.")
    # CommandRequest supplies the existing stable-key contract, including bounds.
    key = CommandRequest(operation_key=operation_key, command=LAUNCHER).operation_key
    selected = recipe.recipe.sources[0]
    envelope = _bytes({"recipe": checked.template_props(), "sources": [{
        "sourceId": selected.source_id, "assetPath": "inputs/" + selected.file.content_sha256 + ".mp4",
        "contentSha256": selected.file.content_sha256, "durationMs": inspection.duration_ms,
        "audioStreams": inspection.audio_streams}]})
    if len(envelope) > 64 * 1024:
        _reject("media_render_envelope_too_large", "The template input exceeds its fixed 64 KiB limit.")
    provenance = _bytes({"format": "jarvis.media-render-plan.v1", "layout": "protected-inputs-v1", "owner_id": owner_id,
        "task_id": str(task_id), "operation_key": key, "sandbox_id": source.sandbox_id,
        "input_path": source.path, "inspection": inspection.model_dump(mode="json"),
        "renderer": binding.model_dump(mode="json"), "recipe": json.loads(checked.canonical_bytes),
        "recipe_sha256": checked.digest, "envelope_sha256": hashlib.sha256(envelope).hexdigest()})
    return MediaRenderPlan(owner_id, task_id, key, source.sandbox_id, binding, checked, source.path,
                           envelope, provenance, hashlib.sha256(provenance).hexdigest())


class AcceptedMediaOutput(FrozenRecord):
    """Core loader output from an accepted action plus currently accessible file.

    The loader must query existing task/action/source authorities and check the
    stored render-plan digest; never construct this from model output, a local
    render path, command exit status or an unverified publication response.
    """

    owner_id: str = Field(strict=True, min_length=1)
    task_id: UUID
    action_id: UUID
    render_plan_sha256: Digest
    file: RetainedRef
    content_type: Literal["video/mp4"]
    size_bytes: int = Field(strict=True, gt=0)


async def present_accepted_output(plan: MediaRenderPlan, *, ctx: WorkspaceContext,
                                  action: Literal["open", "download"] | None,
                                  load_accepted_output, presentations):
    """Optional explicit presentation; existing owner rechecks the exact file.

    load_accepted_output(owner_id, task_id, plan_digest) reads trusted durable
    acceptance. Return the existing PresentationService outcome unchanged: an
    offer is not evidence of display, saving, playback, hearing or publication.
    The foreground command owner supplies an admitted context, never a worker
    impersonating foreground authority. None performs no reads or effects.
    """
    if action is None:
        return None
    if action not in {"open", "download"} or ctx.owner_id != plan.owner_id or ctx.operation_id is None:
        _reject("media_presentation_not_admitted", "An explicitly requested, admitted presentation is required.")
    output = await load_accepted_output(plan.owner_id, plan.task_id, plan.digest)
    if not isinstance(output, AcceptedMediaOutput):
        _reject("media_output_not_accepted", "The exact render has no accepted, accessible output within its allowance.")
    try:
        output = AcceptedMediaOutput.model_validate(output)
    except ValueError:
        _reject("media_output_not_accepted", "The accepted output record is invalid.")
    if (output.owner_id != plan.owner_id
            or output.task_id != plan.task_id or output.render_plan_sha256 != plan.digest
            or output.size_bytes > plan.binding.max_output_bytes):
        _reject("media_output_not_accepted", "The exact render has no accepted, accessible output within its allowance.")
    return await presentations.present(ctx, action=action, artifact_kind="file",
        artifact_id=output.file.file_id, version_id=output.file.version_id)
