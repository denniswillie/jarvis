"""Narrow recorded-video command authority on existing task/source records."""
from uuid import UUID
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.files.generated import ArtifactPolicy, ArtifactSnapshot
from app.files.transfer import CAPTURE_MAX_BYTES
from app.sandbox.models import SandboxPolicy
from app.tasks.models import TaskError, TaskVoiceSource, digest, json_object
from app.workspace.models import StoredText
from app.workspace.media_recipe import RetainedRef


MEDIA_TOOL = "start_media_task"
MEDIA_CAPABILITIES = frozenset({"sandbox.execute", "files.publish_generated", "files.publish_snapshot"})


class SavedRecordingReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    file_id: UUID
    source_id: UUID
    version_id: StoredText = Field(min_length=1, max_length=256)
    receipt_event_id: UUID



class AcceptedOutputTarget(BaseModel):
    """Exact fields copied from verified_media.outputs / AcceptedMediaOutput."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: UUID
    action_id: UUID
    render_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file: RetainedRef

    @classmethod
    def from_output(cls, output):
        return cls(task_id=output.task_id, action_id=output.action_id,
                   render_plan_sha256=output.render_plan_sha256, file=output.file)


class MediaInstruction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    instruction: StoredText = Field(min_length=1, max_length=8000)
    recording_id: UUID | None = None
    saved: SavedRecordingReference | None = None
    recording_profile: Literal["retained_recording.v1"] | None = None
    accepted_output: AcceptedOutputTarget | None = None

    @model_validator(mode="after")
    def one_target(self):
        if self.accepted_output is not None and any(value is not None for value in
                (self.recording_id, self.saved, self.recording_profile)):
            raise ValueError("Choose the exact accepted output or recording target, never both")
        return self

    @field_validator("instruction")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("A nonblank English edit instruction is required")
        return value


class MediaInput(SavedRecordingReference):
    """Resolved by trusted metadata/provenance readers, never a tool argument."""
    owner_id: StoredText
    session_id: UUID
    recording_id: UUID
    area_id: UUID
    source_payload_id: UUID
    receipt_payload_id: UUID
    admission_payload_id: UUID
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(strict=True, ge=1, le=CAPTURE_MAX_BYTES)

    @property
    def dependencies(self):
        return (self.source_payload_id, self.receipt_payload_id, self.admission_payload_id)



class AcceptedMediaProof(BaseModel):
    """Existing immutable row identities, not a second publication receipt."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    action_id: UUID
    grant_id: UUID
    scope_payload_id: UUID
    receipt_payload_id: UUID
    parent_scope_payload_id: UUID
    source_payload_id: UUID
    metadata_payload_id: UUID
    parameters_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def dependencies(self):
        return (self.scope_payload_id, self.receipt_payload_id, self.parent_scope_payload_id,
                self.source_payload_id, self.metadata_payload_id)


class RetainedMediaInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    profile: Literal["retained_recording.v1"] = "retained_recording.v1"
    original: MediaInput
    capture_session_id: UUID
    command_session_id: UUID

    @model_validator(mode="after")
    def capture_identity(self):
        if self.capture_session_id != self.original.session_id:
            raise ValueError("Capture provenance cannot be rebound to a new conversation")
        return self

    @property
    def owner_id(self):
        return self.original.owner_id

    @property
    def session_id(self):
        return self.command_session_id

    @property
    def area_id(self):
        return self.original.area_id

    @property
    def file_id(self):
        return self.original.file_id

    @property
    def version_id(self):
        return self.original.version_id

    @property
    def content_sha256(self):
        return self.original.content_sha256

    @property
    def dependencies(self):
        return self.original.dependencies


class CorrectedMediaInput(RetainedMediaInput):
    profile: Literal["accepted_output.v1"] = "accepted_output.v1"
    target: AcceptedOutputTarget
    previous_recipe: RetainedRef
    edit_id: UUID
    recipe_action_id: UUID
    recipe_proof: AcceptedMediaProof
    output_proof: AcceptedMediaProof

    @model_validator(mode="after")
    def exact_predecessor(self):
        if (self.previous_recipe.content_sha256 != self.target.render_plan_sha256
                or self.recipe_proof.action_id != self.recipe_action_id
                or self.output_proof.action_id != self.target.action_id
                or self.recipe_action_id == self.target.action_id):
            raise ValueError("The predecessor must be this accepted output's exact saved plan")
        return self

    @property
    def dependencies(self):
        return tuple(dict.fromkeys((*self.original.dependencies,
            *self.recipe_proof.dependencies, *self.output_proof.dependencies)))


def parse_media_input(value):
    if isinstance(value, (MediaInput, RetainedMediaInput)):
        value = value.model_dump(mode="json")
    profile = value.get("profile") if isinstance(value, dict) else None
    kind = {"retained_recording.v1": RetainedMediaInput, "accepted_output.v1": CorrectedMediaInput}.get(profile, MediaInput)
    return kind.model_validate(value)


def media_error():
    return TaskError("media_command_scope_unavailable", "The exact recorded-video authority is unavailable or changed.", 403)


def require_media_command_create(command, request, *, source=None, source_event_id=None,
                                 mode=None, grant_factory=None):
    args = MediaInstruction.model_validate(command.normalized_arguments)
    if (command.tool_name != MEDIA_TOOL or command.capability != "tasks.create"
            or (args.accepted_output is None and (args.recording_id is None or args.saved is None))
            or args.model_dump(mode="json", exclude_none=True) != command.normalized_arguments
            or request.session_id != command.session_id or request.request_id != command.operation_id
            or request.requested_outcome != args.instruction or mode not in {None, "background"}
            or request.backend_kind != "frontier" or request.execution_environment != "offline_sandbox"
            or not request.publish_artifacts or request.edit_memory or request.grant_ids
            or source is not None or source_event_id is not None
            or (grant_factory is not None and not callable(grant_factory))):
        raise media_error()


def require_media_input(command, value):
    pin = parse_media_input(value)
    args = MediaInstruction.model_validate(command.normalized_arguments)
    if pin.owner_id != command.owner_id or pin.session_id != command.session_id:
        raise media_error()
    if args.accepted_output is not None:
        if not isinstance(pin, CorrectedMediaInput) or pin.target != args.accepted_output:
            raise media_error()
        return pin
    if isinstance(pin, CorrectedMediaInput):
        raise media_error()
    retained = args.recording_profile == "retained_recording.v1"
    if retained != isinstance(pin, RetainedMediaInput):
        raise media_error()
    original = pin.original if retained else pin
    if (original.recording_id != args.recording_id or args.saved is None
            or any(getattr(original, key) != getattr(args.saved, key) for key in SavedRecordingReference.model_fields)):
        raise media_error()
    return pin


# No current-session selection and no network/media read. This is also the
# durable recheck after disconnect. Exact source and publication rows remain
# authoritative; a remembered hash or model-provided reference is not enough.
MEDIA_SOURCE_SQL = """select r.*,v.payload_id as source_payload_id,
        p.content->>'content_sha256' as original_sha256,
        f.content_sha256,f.size_bytes
    from jarvis_recordings r
    join jarvis_knowledge_areas a on a.owner_id=r.owner_id and a.id=r.area_id
    join jarvis_knowledge_source_versions v on v.owner_id=r.owner_id and v.document_id=r.version_id
        and v.source_id=r.source_id and v.area_id=r.area_id
    join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
    join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
    join jarvis_context_payloads c on c.owner_id=r.owner_id and c.id=r.completion_payload_id
    join jarvis_context_events ce on ce.owner_id=r.owner_id and ce.id=r.completion_event_id and ce.payload_id=c.id
    join jarvis_context_payloads ap on ap.owner_id=r.owner_id and ap.id=r.admission_payload_id
    join jarvis_context_events ae on ae.owner_id=r.owner_id and ae.id=r.admission_event_id and ae.payload_id=ap.id
    join jarvis_file_publications f on f.owner_id=r.owner_id and f.id=r.publication_id
        and f.file_id=r.file_id and f.area_id=r.area_id
    join jarvis_context_payloads fp on fp.owner_id=f.owner_id and fp.id=f.metadata_payload_id
    where r.owner_id=$1 and r.session_id=$2 and r.recording_id=$3 and not r.revoked
        and r.capture_closed and a.status='active' and s.status='active' and v.revision=s.current_revision
        and p.availability='available' and c.availability='available' and ap.availability='available'
        and fp.availability='available' and f.phase='available' and f.upload_settled and not f.delete_requested
        and f.operation_key='recording:' || r.recording_id::text and f.transfer_policy='mp4_capture_v1'
        and p.content->>'recording_id'=r.recording_id::text
        and not exists(select 1 from jarvis_context_events newer where newer.owner_id=ae.owner_id
            and newer.source_namespace=ae.source_namespace and newer.source_event_id=ae.source_event_id
            and newer.revision>ae.revision)"""


async def validate_media_input(connection, owner_id, session_id, value, *, lock=True):
    pin = parse_media_input(value)
    if isinstance(pin, RetainedMediaInput):
        from app.workspace.media_correction import validate_retained_media_input
        return await validate_retained_media_input(connection, owner_id, session_id, pin, lock=lock)
    if pin.owner_id != owner_id or pin.session_id != session_id:
        raise media_error()
    # Canonical context snapshots are READ ONLY. Preserve every source check
    # there without row locks; admission and execution keep locking by default.
    lock_sql = " for share of r,a,s,v,p,c,ap,f,fp,ce,ae" if lock else ""
    row = await connection.fetchrow(MEDIA_SOURCE_SQL + lock_sql, owner_id, session_id, pin.recording_id)
    if row is None:
        raise media_error()
    aliases = {"receipt_event_id": "completion_event_id", "receipt_payload_id": "completion_payload_id"}
    if (any(row[aliases.get(key, key)] != getattr(pin, key) for key in MediaInput.model_fields)
            or row["original_sha256"] != pin.content_sha256):
        raise media_error()
    return pin


async def media_description(connection, task, *, lock=True):
    row = await connection.fetchrow("select content,availability from jarvis_context_payloads "
        "where owner_id=$1 and id=$2" + (" for share" if lock else ""), task.owner_id, task.description_payload_id)
    if row is None or row["availability"] != "available":
        raise media_error()
    body = json_object(row["content"])
    pin = parse_media_input(body["media_input"])
    raw = body["instruction_request"]
    instruction = TaskVoiceSource(UUID(raw["event_id"]), raw["revision"], UUID(raw["session_id"]))
    if (task.backend_kind != "frontier" or task.knowledge_area_id != pin.area_id
            or task.session_id != pin.session_id or instruction.session_id != task.session_id):
        raise media_error()
    from app.plugins.managed import decode_managed_job
    decode_managed_job(body["managed_plugin"], task.owner_id, route="video_editing")
    return body, pin, instruction


async def validate_media_command_target(connection, task, *, lock=True):
    """Family identity only, like the note owner; execution checks expiry too.

    Do not call task_command_source/validate_grant here: that would recurse.
    """
    try:
        _, pin, instruction = await media_description(connection, task, lock=lock)
        rows = await connection.fetch("""select g.*,p.content as scope from jarvis_task_grants tg
            join jarvis_operation_grants g on g.owner_id=tg.owner_id and g.id=tg.grant_id
            join jarvis_context_payloads p on p.owner_id=g.owner_id and p.id=g.scope_payload_id
            where tg.owner_id=$1 and tg.task_id=$2""" + (" for share of g,p" if lock else ""), task.owner_id, task.id)
        sandbox, parent, children = None, None, []
        for grant in rows:
            if (grant["capability"] not in MEDIA_CAPABILITIES
                    or (grant["request_event_id"], grant["request_revision"]) != (instruction.event_id, instruction.revision)):
                raise media_error()
            scope = json_object(grant["scope"])
            if grant["capability"] == "sandbox.execute":
                policy = SandboxPolicy.model_validate(scope)
                if (sandbox is not None or policy.template != "base" or policy.networked
                        or grant["operation_key"] != "managed-video-editor:" + str(task.id)):
                    raise media_error()
                sandbox = grant
            elif grant["capability"] == "files.publish_generated":
                policy = ArtifactPolicy.model_validate(scope)
                if (parent is not None or policy.formats != ("json", "mp4") or policy.max_files != 2
                        or grant["operation_key"] != "artifact-task:" + str(task.id)):
                    raise media_error()
                parent = grant
            else:
                children.append((grant, ArtifactSnapshot.model_validate(scope)))
        if sandbox is None or parent is None:
            raise media_error()
        allowance = ArtifactPolicy.model_validate(json_object(parent["scope"]))
        for grant, snapshot in children:
            if (snapshot.task_id != task.id or snapshot.parent_grant_id != parent["id"]
                    or snapshot.request.format != snapshot.inspection.format
                    or snapshot.inspection.size_bytes > allowance.limit_for(snapshot.request.format)
                    or grant["expires_at"] > parent["expires_at"]
                    or grant["operation_key"] != "artifact-snapshot:" + task.id.hex + ":" + digest({"operation_key": snapshot.request.operation_key})):
                raise media_error()
        return pin
    except (KeyError, TypeError, ValueError, AttributeError):
        raise media_error() from None


async def media_command_grant_source_available(tasks, connection, owner_id, grant, *, lock=True):
    if grant["capability"] not in MEDIA_CAPABILITIES:
        return False
    from app.live_state.admission import task_command_source
    try:
        rows = await connection.fetch("select task_id from jarvis_task_grants where owner_id=$1 and grant_id=$2",
                                      owner_id, grant["id"])
        if len(rows) != 1:
            return False
        task = await tasks.load(connection, owner_id, rows[0]["task_id"], lock="share" if lock else False)
        _, pin, instruction = await media_description(connection, task, lock=lock)
        if (instruction.event_id, instruction.revision) != (grant["request_event_id"], grant["request_revision"]):
            return False
        record = await task_command_source(tasks, connection, task, event_id=instruction.event_id,
            event_revision=instruction.revision, current=False, lock=lock)
        if (record is None or record.content["command"]["tool_name"] != MEDIA_TOOL
                or record.content.get("media_input") != pin.model_dump(mode="json")):
            return False
        await validate_media_input(connection, owner_id, task.session_id, pin)
        return True
    except (TaskError, KeyError, TypeError, ValueError, AttributeError):
        return False


async def require_media_recipe_input(tasks, connection, task, recipe):
    """Check a worker recipe against durable admission before staging/recovery.

    The caller owns the existing task-owner/claim transaction. This function
    neither stages nor allocates and must not be called under a provider lock.
    It returns the exact MediaInput for command work. None means only that the
    task has ordinary finalized-user-input authority, whose existing media
    validators still apply; arbitrary derived or missing authority never skips.

    A recording profile admits no predecessor. The versioned accepted-output
    profile admits only the pinned previous recipe/edit_id and original input.
    Capture provenance and the current command session remain independent.
    """
    from app.live_state.admission import task_command_source
    from app.tasks.instruction_source import task_instruction_source
    from app.workspace.media_recipe import MediaEditRecipeV1, RetainedRef

    if not connection.is_in_transaction():
        raise RuntimeError("Media recipe authority requires the caller's task transaction")
    instruction = await task_instruction_source(connection, task)
    record = await task_command_source(tasks, connection, task, event_id=instruction.event_id,
        event_revision=instruction.revision, current=False)
    if record is None:
        if not await tasks._source_available(connection, task.owner_id, instruction.event_id, instruction.revision):
            raise media_error()
        return None
    if record.content["command"]["tool_name"] != MEDIA_TOOL or record.content["operation"] != "create":
        raise media_error()
    # task_command_source has already checked the immutable create receipt,
    # matching description pin, grant family and current exact source rows.
    pin = parse_media_input(record.content["media_input"])
    parsed = MediaEditRecipeV1.model_validate(recipe)
    expected = RetainedRef(file_id=pin.file_id, version_id=pin.version_id, content_sha256=pin.content_sha256)
    if parsed.sources[0].file != expected:
        raise TaskError("media_command_recipe_mismatch", "The recipe must use the exact admitted recording version and bytes.", 403)
    if isinstance(pin, CorrectedMediaInput):
        if parsed.previous_recipe != pin.previous_recipe or parsed.edit_id != pin.edit_id:
            raise TaskError("media_command_predecessor_mismatch", "Use the exact admitted predecessor and edit lineage.", 403)
    elif parsed.previous_recipe is not None:
        raise TaskError("media_command_predecessor_not_admitted", "This recording edit did not admit a predecessor recipe.", 403)
    return pin
