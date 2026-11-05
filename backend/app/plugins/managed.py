"""Explicit managed jobs on the existing task/action/sandbox records.

Composition supplies real, owner-scoped inventory/readiness observations and an
immutable executable mapping. There are deliberately no installed defaults,
probes, installers, provider calls, or additional durable stores here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
import shlex
from typing import ClassVar
from uuid import UUID

from pydantic import Field, field_validator

from app.errors import JarvisHTTPError
from app.plugins.authority import ManagedPluginBinding, SubmittedManagedJob
from app.plugins.models import FrozenRecord, InventoryObservation, ReadinessObservation, Digest, BundleVersion
from app.plugins.media import (
    MEDIA_CAPABILITY, MEDIA_FORMAT, MEDIA_GRANT_PREFIX, MEDIA_PLUGIN, MEDIA_ROUTE,
    ManagedMediaJob, MediaRendererBundle, MediaRendererEvidence,
)
from app.sandbox.models import CommandRequest
from app.tasks.models import ClaimLost, TaskError, digest, json_object


ROUTE = "files_data"
CAPABILITY = "plugins.files_data"
GRANT_PREFIX = "managed-files-data:"
FORMAT = "jarvis.managed-files-data.v1"


class FilesDataBundle(FrozenRecord):
    """A qualified Python/stdlib bundle in a specific existing sandbox image.

    The digest identifies the real qualified bundle, not a hash manufactured
    from this declaration. The executable is used, not merely shown in a prompt.
    """

    configuration_id: UUID
    bundle_version: BundleVersion
    bundle_sha256: Digest
    sandbox_template_id: str = Field(min_length=1, max_length=200, strict=True)
    executable: str = Field(min_length=1, max_length=1000, strict=True)

    @field_validator("executable")
    @classmethod
    def absolute_executable(cls, value):
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or str(path) != value or "\x00" in value:
            raise ValueError("Use the exact absolute qualified bundle executable")
        return value

    @field_validator("sandbox_template_id")
    @classmethod
    def exact_template(cls, value):
        if value != value.strip() or "\x00" in value:
            raise ValueError("Use an exact image identity")
        return value

    def command(self, arguments):
        request = FilesDataCommand.model_validate(arguments)
        return CommandRequest(operation_key=request.operation_key, cwd=request.cwd,
            command=shlex.join((self.executable, "-I", "-S", "-c", request.code)))

    def require_command(self, request):
        try:
            words = shlex.split(request.command)
        except ValueError:
            words = []
        # Require our canonical quoting, too: shell syntax cannot be smuggled
        # behind an otherwise matching executable prefix.
        if (len(words) != 5 or words[:4] != [self.executable, "-I", "-S", "-c"]
                or request.command != shlex.join(words)):
            raise TaskError("managed_command_required", "This job must use its pinned Files & data executable.", 403)

    def require_image(self, handle):
        if handle.template_id != self.sandbox_template_id:
            raise TaskError("managed_bundle_image_mismatch", "The actual sandbox image is not the qualified bundle.", 503)


class FilesDataCommand(FrozenRecord):
    operation_key: str = Field(min_length=1, max_length=100)
    code: str = Field(min_length=1, max_length=12000)
    cwd: str = Field(default=".", min_length=1, max_length=1000)


@dataclass(frozen=True)
class FilesDataEvidence:
    """Trusted observer output, never model or HTTP request authority."""

    owner_id: str
    sandbox_template_id: str
    executable: str
    inventory: InventoryObservation
    readiness: ReadinessObservation


@dataclass(frozen=True)
class ManagedFilesDataJob:
    binding: ManagedPluginBinding
    bundle: FilesDataBundle
    route: ClassVar[str] = ROUTE
    grant_prefix: ClassVar[str] = GRANT_PREFIX

    def __post_init__(self):
        if (not isinstance(self.binding, ManagedPluginBinding) or self.binding.plugin_id != ROUTE
                or not isinstance(self.bundle, FilesDataBundle)
                or self.binding.configuration_id != self.bundle.configuration_id):
            raise ValueError("A Files & data binding and the exact executable mapping are required")

    def payload(self):
        binding = asdict(self.binding)
        binding["configuration_id"] = str(self.binding.configuration_id)
        return {"format": FORMAT, "binding": binding, "bundle": self.bundle.model_dump(mode="json")}

    @classmethod
    def from_payload(cls, value, owner_id):
        try:
            if not isinstance(value, dict) or set(value) != {"format", "binding", "bundle"} or value["format"] != FORMAT:
                raise ValueError("Invalid managed scope")
            binding = dict(value["binding"])
            binding["configuration_id"] = UUID(binding["configuration_id"])
            job = cls(ManagedPluginBinding(**binding), FilesDataBundle.model_validate(value["bundle"]))
            if job.binding.owner_id != owner_id:
                raise ValueError("Wrong owner")
            return job
        except (ValueError, TypeError, KeyError, AttributeError):
            raise TaskError("managed_scope_unavailable", "The original managed scope is unavailable.", 403) from None


def managed_capability(route):
    if route == ROUTE:
        return CAPABILITY
    if route == MEDIA_ROUTE:
        return MEDIA_CAPABILITY
    raise TaskError("managed_task_scope_invalid", "Select an explicit supported managed route.", 403)


def managed_grant_prefix(route):
    managed_capability(route)
    return GRANT_PREFIX if route == ROUTE else MEDIA_GRANT_PREFIX


def decode_managed_job(value, owner_id, *, route=None):
    kind = ManagedMediaJob if isinstance(value, dict) and value.get("format") == MEDIA_FORMAT else ManagedFilesDataJob
    job = kind.from_payload(value, owner_id)
    if route is not None and job.route != route:
        raise TaskError("managed_route_mismatch", "The selected job does not match the admitted route.", 403)
    return job


def require_request(request, route, command_admission=None):
    managed_capability(route)
    if command_admission is not None:
        from app.live_state.admission import require_command, require_command_create
        require_command(command_admission, getattr(command_admission, "owner_id", None), request.request_id, "create")
        if route != MEDIA_ROUTE or command_admission.tool_name != "start_media_task":
            raise TaskError("managed_task_scope_invalid", "Only the exact media command may enter this managed route.", 403)
        require_command_create(command_admission, request)
    if (request.backend_kind != "frontier"
            or request.execution_environment != "offline_sandbox"
            or request.edit_memory or request.grant_ids
            or (route == MEDIA_ROUTE and not request.publish_artifacts)):
        raise TaskError("managed_task_scope_invalid",
            "Managed work requires its explicit offline route and trusted request authority; media also requires publication.", 403)


async def load_managed_task(connection, task):
    """Read the original binding and an independent marker on its attached grant.

    Removed descriptions or a missing binding cannot downgrade a managed task
    to general terminal work. The marker is on existing grants, not new schema.
    """
    grants = await connection.fetch("""select g.id,g.operation_key from jarvis_task_grants tg
        join jarvis_operation_grants g on g.owner_id=tg.owner_id and g.id=tg.grant_id
        where tg.owner_id=$1 and tg.task_id=$2 and
            (g.operation_key like 'managed-files-data:%' or g.operation_key like 'managed-video-editor:%')""",
        task.owner_id, task.id)
    row = await connection.fetchrow("""select availability,content from jarvis_context_payloads
        where owner_id=$1 and id=$2""", task.owner_id, task.description_payload_id)
    description = json_object(row["content"]) if row and row["availability"] == "available" else {}
    payload = description.get("managed_plugin")
    if not grants and payload is None:
        return None
    if len(grants) != 1 or payload is None:
        raise TaskError("managed_binding_required", "Managed work cannot lose its original binding or grant.", 403)
    job = decode_managed_job(payload, task.owner_id)
    if grants[0]["operation_key"] != job.grant_prefix + str(task.id):
        raise TaskError("managed_binding_required", "The managed grant must match the retained job variant.", 403)
    return job, grants[0]["id"]


class ManagedPluginRuntime:
    def __init__(self, authority, *, bundles, observe, clock=None):
        """observe(connection, owner_id, bundle) must read real local evidence.

        It runs on the caller's owner-locked transaction: no provider/network I/O.
        Missing, expired or mismatching evidence denies new work. Core owns the
        deployment mapping/observer and worker composition, with no defaults here.
        """
        values = tuple(bundles)
        if (not values or any(not isinstance(item, (FilesDataBundle, MediaRendererBundle)) for item in values)
                or len({item.configuration_id for item in values}) != len(values) or not callable(observe)):
            raise ValueError("Explicit unique executable mappings and a real observation reader are required")
        self.authority = authority
        self.bundles = {item.configuration_id: item for item in values}
        self.observe = observe
        self.clock = clock or (lambda: datetime.now(UTC))

    @property
    def capabilities(self):
        return tuple(capability for kind, capability in
            ((FilesDataBundle, CAPABILITY), (MediaRendererBundle, MEDIA_CAPABILITY))
            if any(isinstance(bundle, kind) for bundle in self.bundles.values()))

    async def _ready(self, connection, owner_id, bundle):
        if self.bundles.get(bundle.configuration_id) != bundle:
            raise TaskError("managed_bundle_changed", "The selected executable mapping is no longer available.", 503)
        evidence = await self.observe(connection, owner_id, bundle)
        now = self.clock()
        media = isinstance(bundle, MediaRendererBundle)
        expected_type = MediaRendererEvidence if media else FilesDataEvidence
        if not isinstance(evidence, expected_type) or evidence.owner_id != owner_id:
            raise TaskError("managed_bundle_not_ready", "Actual owner-scoped bundle readiness is unavailable.", 503)
        inventory, readiness = evidence.inventory, evidence.readiness
        mapping_matches = (evidence.renderer == bundle.renderer if media else
            evidence.sandbox_template_id == bundle.sandbox_template_id and evidence.executable == bundle.executable)
        if (not isinstance(inventory, InventoryObservation) or not isinstance(readiness, ReadinessObservation)
                or not mapping_matches
                or any(item.plugin_id != (MEDIA_PLUGIN if media else ROUTE) or item.configuration_id != bundle.configuration_id
                       or not item.current_at(now) or item.bundle_sha256 != bundle.bundle_sha256
                       for item in (inventory, readiness))
                or inventory.state != "installed" or inventory.bundle_version != bundle.bundle_version
                or readiness.state != "ready"):
            raise TaskError("managed_bundle_not_ready", "Installed bundle and current readiness must match the exact mapping.", 503)

    async def admit(self, connection, owner_id, *, route=ROUTE):
        managed_capability(route)
        try:
            binding = await self.authority.select_new_work(owner_id=owner_id, category=route, connection=connection)
        except JarvisHTTPError as error:
            raise TaskError(error.code, "Managed selection was not admitted.", 403) from None
        bundle = self.bundles.get(binding.configuration_id)
        expected_type = MediaRendererBundle if route == MEDIA_ROUTE else FilesDataBundle
        if not isinstance(bundle, expected_type):
            raise TaskError("managed_bundle_unmapped", "No executable bundle maps this configuration.", 503)
        await self._ready(connection, owner_id, bundle)
        job_type = ManagedMediaJob if route == MEDIA_ROUTE else ManagedFilesDataJob
        return job_type(binding, bundle).payload()

    async def require_new_work(self, connection, task, job):
        try:
            await self.authority.require_new_work(connection, owner_id=task.owner_id, binding=job.binding)
        except JarvisHTTPError as error:
            raise TaskError(error.code, "The managed job was not submitted.", 403) from None
        await self._ready(connection, task.owner_id, job.bundle)

    async def ensure_submitted(self, context, grant_id):
        """Use the existing action token as the whole managed-job submission pin.

        Accepted here means a durable local execution submission, not that any
        provider command ran or that the requested result was produced.
        """
        from app.tasks.actions import TaskActionRepository

        async with context.tasks.transaction(context.task.owner_id) as connection:
            task, _ = await context.tasks.lock_claim(connection, context.claim)
            if await context.tasks.context_problem(connection, task):
                raise ClaimLost()
            managed = await load_managed_task(connection, task)
            if managed is None or managed[1] != grant_id:
                raise TaskError("managed_binding_required", "The exact managed grant is required.", 403)
            job, _ = managed
            grant = await context.tasks.validate_grant(connection, task.owner_id, grant_id)
        actions = TaskActionRepository(context.tasks)
        record = await actions.prepare(context.claim, job.grant_prefix + str(task.id), grant_id,
                                       "sandbox.execute", json_object(grant["scope"]))
        if record["status"] in {"prepared", "authorized"}:
            await actions.authorize(context.claim, record["id"])
            try:
                await actions.begin_submission(context.claim, record["id"], managed_plugins=self)
            except TaskError as error:
                if error.code != "task_action_already_submitted":
                    raise
        # Re-read under the owner lock, rather than trusting an in-memory token.
        async with context.tasks.transaction(task.owner_id) as connection:
            current, _ = await context.tasks.lock_claim(connection, context.claim)
            managed = await load_managed_task(connection, current)
            if managed != (job, grant_id):
                raise TaskError("managed_scope_changed", "The original managed job changed.", 403)
            await self.submitted_job(connection, context.tasks, current)
        return job

    async def submitted_job(self, connection, tasks, task, *, expected=None):
        """Read a durable exact-job pin; never select, prepare or submit work.

        The caller holds the task-owner lock. Ordinary task/source/grant validity
        still applies, but today's plugin enablement, mapping and observation
        expiry cannot revoke an already-submitted job's original output scope.
        """
        if not connection.is_in_transaction():
            raise RuntimeError("Managed continuation requires the owner-locked transaction")
        if await tasks.context_problem(connection, task):
            raise ClaimLost()
        managed = await load_managed_task(connection, task)
        if managed is None:
            raise TaskError("managed_binding_required", "This task has no original managed binding.", 403)
        job, grant_id = managed
        grant = await tasks.validate_grant(connection, task.owner_id, grant_id)
        action = await connection.fetchrow("""select a.*,p.content as receipt,p.availability as receipt_availability
            from jarvis_task_actions a left join jarvis_context_payloads p
            on p.owner_id=a.owner_id and p.id=a.receipt_payload_id
            where a.owner_id=$1 and a.task_id=$2 and a.operation_key=$3 for update of a""",
            task.owner_id, task.id, job.grant_prefix + str(task.id))
        receipt = json_object(action["receipt"]) if action and action["receipt_availability"] == "available" else {}
        if (not action or action["status"] != "accepted" or not isinstance(action["submission_token"], UUID)
                or action["task_id"] != task.id or action["grant_id"] != grant_id
                or grant["capability"] != "sandbox.execute"
                or action["parameters_digest"] != digest(json_object(grant["scope"]))
                or receipt != submission_receipt(task, job, action["submission_token"])):
            raise TaskError("managed_submission_unconfirmed", "Reconcile the original managed submission before continuing.", 409)
        pin = SubmittedManagedJob(job.binding, action["id"], action["submission_token"])
        if expected is not None and (not isinstance(expected, SubmittedManagedJob) or expected != pin):
            raise TaskError("managed_submission_mismatch", "The supplied pin is not this task's exact submitted job.", 403)
        return pin

    async def require_publication(self, connection, tasks, task, grant_id, parameters, *, expected=None):
        """The pin permits continuation, not artifact authority or new work.

        Retain the publisher's existing parent/child grant, exact snapshot and
        operation identity checks. No grant is issued or inferred from the pin.
        """
        from app.files.generated import ArtifactSnapshot

        pin = await self.submitted_job(connection, tasks, task, expected=expected)
        attached = await connection.fetchval("""select exists(select 1 from jarvis_task_grants
            where owner_id=$1 and task_id=$2 and grant_id=$3)""", task.owner_id, task.id, grant_id)
        if not attached:
            raise TaskError("task_tool_grant_not_attached", "Publication requires this task's separate exact grant.", 403)
        grant = await tasks.validate_grant(connection, task.owner_id, grant_id)
        scope = json_object(grant["scope"])
        try:
            snapshot = ArtifactSnapshot.model_validate(scope)
        except (ValueError, TypeError):
            raise TaskError("managed_publication_scope", "Only an exact granted artifact snapshot may continue this job.", 403) from None
        operation_key = "artifact-snapshot:" + task.id.hex + ":" + digest({"operation_key": snapshot.request.operation_key})
        if (grant["capability"] != "files.publish_snapshot" or snapshot.task_id != task.id
                or task.session_id is None or grant["operation_key"] != operation_key or digest(parameters) != digest(scope)):
            raise TaskError("managed_publication_scope", "The artifact grant does not match this task and exact operation.", 403)
        parent_attached = await connection.fetchval("""select exists(select 1 from jarvis_task_grants
            where owner_id=$1 and task_id=$2 and grant_id=$3)""", task.owner_id, task.id, snapshot.parent_grant_id)
        if not parent_attached:
            raise TaskError("artifact_publication_not_granted", "The original publication allowance is not attached.", 403)
        parent = await tasks.validate_grant(connection, task.owner_id, snapshot.parent_grant_id)
        if (parent["capability"] != "files.publish_generated"
                or grant["request_event_id"] != parent["request_event_id"]
                or grant["request_revision"] != parent["request_revision"]):
            raise TaskError("artifact_publication_not_granted", "The exact parent publication authority is unavailable.", 403)
        return pin


def submission_receipt(task, job, token):
    return {"format": ("jarvis.managed-media.submission.v1" if isinstance(job, ManagedMediaJob)
                       else "jarvis.managed-files-data.submission.v1"), "task_id": str(task.id),
        "description_payload_id": str(task.description_payload_id), "scope": job.payload(),
        "submission_token": str(token), "state": "submitted", "effect": "not_yet_observed"}


# Preserve existing imports and retained Files & data records. Both variants use
# the same authority/action implementation rather than a second executor.
ManagedFilesDataRuntime = ManagedPluginRuntime
