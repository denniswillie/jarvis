"""Server-only, receipt-backed commands. Never generic request authority.

The adapter constructs this carrier after authenticating its session and maps
the normalized command to TaskCreate/TaskMutation. No model-facing request may
construct it. Acceptance uses the existing task request receipt, not a new store.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from uuid import UUID

from app.live_state.command_identity import command_arguments_digest, command_operation_id
from app.tasks.context_events import ContextIngress
from app.tasks.models import TaskCreate, TaskError, TaskMutation, digest


COMMAND_FORMAT = "jarvis.task_command.v1"
COMMAND_CAPABILITIES = frozenset({"tasks.create", "tasks.update", "tasks.cancel", "tasks.resume",
    "workspace.open", "workspace.download",
    "recordings.start", "recordings.stop"})


def _json_copy(value):
    if isinstance(value, Mapping):
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    return value


@dataclass(frozen=True, slots=True, init=False)
class CommandAdmission:
    owner_id: str
    session_id: UUID
    provider_session_key: str
    provider_call_id: str
    tool_name: str
    capability: str
    operation_id: UUID
    arguments_digest: str
    _arguments_json: str = field(repr=False)

    def __init__(self, *, owner_id, session_id, provider_session_key, provider_call_id,
                 tool_name, capability, normalized_arguments):
        operation_id = command_operation_id(owner_id, session_id, provider_session_key, provider_call_id)
        arguments_digest = command_arguments_digest(tool_name, normalized_arguments)
        if not isinstance(capability, str) or capability not in COMMAND_CAPABILITIES:
            raise ValueError("Only explicitly supported foreground command capabilities are allowed")
        arguments_json = json.dumps(_json_copy(normalized_arguments), sort_keys=True,
                                    ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        for name, value in (
            ("owner_id", owner_id), ("session_id", session_id),
            ("provider_session_key", provider_session_key), ("provider_call_id", provider_call_id),
            ("tool_name", tool_name), ("capability", capability), ("operation_id", operation_id),
            ("arguments_digest", arguments_digest), ("_arguments_json", arguments_json),
        ):
            object.__setattr__(self, name, value)

    @property
    def normalized_arguments(self):
        return json.loads(self._arguments_json)

    @property
    def journal_call_key(self):
        return "lean:" + str(self.operation_id)

    def parameters(self):
        return {
            "owner_id": self.owner_id, "session_id": str(self.session_id),
            "provider_session_key": self.provider_session_key, "provider_call_id": self.provider_call_id,
            "operation_id": str(self.operation_id), "tool_name": self.tool_name,
            "capability": self.capability, "arguments_digest": self.arguments_digest,
            "arguments": self.normalized_arguments,
        }


def require_command(command, owner_id, request_id, operation):
    if (type(command) is not CommandAdmission or command.owner_id != owner_id
            or command.operation_id != request_id or command.capability != "tasks." + operation):
        raise TaskError("task_command_scope_invalid", "An exact server-constructed task command is required.", 403)
    return command


def require_command_create(command, request, *, source=None, source_event_id=None, mode=None, grant_factory=None):
    require_command(command, command.owner_id, request.request_id, "create")
    if command.tool_name == "start_media_task":
        from app.tasks.media_editing import require_media_command_create
        require_media_command_create(command, request, source=source, source_event_id=source_event_id,
                                     mode=mode, grant_factory=grant_factory)
        return
    if (source is not None or source_event_id is not None
            or request.session_id != command.session_id or mode not in {None, "consult", "background"}
            or request.backend_kind != "frontier" or request.execution_environment != "none"
            or request.publish_artifacts or request.edit_memory or request.grant_ids):
        raise TaskError("task_command_scope_invalid", "This command cannot grant unrelated task capabilities.", 403)
    if grant_factory is not None:
        raise TaskError("task_command_scope_invalid", "Ordinary command admission permits only grant-free frontier work.", 403)


def command_parameters(command):
    # Resolved target/base/area are deliberately absent: the first receipt pins
    # those. Re-delivery of identical tool arguments must not resolve them again.
    return {"format": COMMAND_FORMAT, "command_admission": command.parameters()}


async def require_command_target(connection, task, *, lock=True):
    if task.backend_kind != "frontier":
        raise TaskError("task_command_scope_invalid", "This command cannot control this task backend.", 403)
    if await connection.fetchval(
        "select exists(select 1 from jarvis_task_grants where owner_id=$1 and task_id=$2)",
        task.owner_id, task.id,
    ):
        if await connection.fetchval("""select exists(select 1 from jarvis_task_grants tg
            join jarvis_operation_grants g on g.owner_id=tg.owner_id and g.id=tg.grant_id
            where tg.owner_id=$1 and tg.task_id=$2 and g.operation_key=$3)""",
            task.owner_id, task.id, "managed-video-editor:" + str(task.id)):
            from app.tasks.media_editing import validate_media_command_target
            await validate_media_command_target(connection, task, lock=lock)
            return
        raise TaskError("task_command_scope_invalid", "This command cannot control unrelated granted work.", 403)


async def append_task_command(tasks, connection, command, operation, request, *, task_id, area_id,
                              base_revision, mode=None, media_input=None):
    from app.tasks.context_repository import ContextRepository

    require_command(command, command.owner_id, request.request_id, operation)
    record = await ContextRepository(tasks).append_in_transaction(connection, command.owner_id, ContextIngress(
        session_id=command.session_id, source_namespace="voice_command",
        source_event_id=str(command.operation_id), receipt_id=command.journal_call_key,
        kind="command_request", trust="derived", finality="final", expected_revision=0,
        payload={
            "format": COMMAND_FORMAT, "command": command.parameters(), "operation": operation,
            "task_id": str(task_id), "base_revision": base_revision,
            "area_id": str(area_id) if area_id is not None else None,
            "request": request.model_dump(mode="json"), "mode": mode,
            "parameters_digest": digest(command_parameters(command)),
            **({"media_input": media_input.model_dump(mode="json")} if media_input is not None else {}),
        },
    ))
    if media_input is not None:
        for payload_id in media_input.dependencies:
            await connection.execute("""insert into jarvis_context_payload_dependencies
                (owner_id,dependent_payload_id,source_payload_id) values($1,$2,$3) on conflict do nothing""",
                command.owner_id, record.payload_id, payload_id)
    return {"id": record.id, "revision": record.revision, "sequence": record.sequence,
            "payload_id": record.payload_id, "session_id": record.session_id}


def decode_task_command(record):
    """Validate representation, not authority; authority also needs its receipt."""
    if record.source_namespace != "voice_command" and record.kind != "command_request":
        return None
    try:
        body = record.content
        if (record.source_namespace != "voice_command" or record.kind != "command_request"
                or record.trust != "derived" or record.finality != "final" or record.revision != 1
                or not record.is_current or record.availability != "available" or not record.content_digest
                or not isinstance(body, dict) or set(body) - {"media_input"} != {
                    "format", "command", "operation", "task_id", "base_revision", "area_id",
                    "request", "mode", "parameters_digest"}
                or body["format"] != COMMAND_FORMAT):
            raise ValueError()
        raw = body["command"]
        command = CommandAdmission(
            owner_id=raw["owner_id"], session_id=UUID(raw["session_id"]),
            provider_session_key=raw["provider_session_key"], provider_call_id=raw["provider_call_id"],
            tool_name=raw["tool_name"], capability=raw["capability"], normalized_arguments=raw["arguments"],
        )
        if (command.parameters() != raw or command.owner_id != record.owner_id
                or command.session_id != record.session_id or str(command.operation_id) != record.source_event_id
                or command.capability != "tasks." + body["operation"]
                or body["parameters_digest"] != digest(command_parameters(command))
                or str(UUID(body["task_id"])) != body["task_id"]
                or (body["area_id"] is not None and str(UUID(body["area_id"])) != body["area_id"])
                or type(body["base_revision"]) is not int or body["base_revision"] < 0):
            raise ValueError()
        if body["operation"] == "create":
            request_payload = dict(body["request"])
            if request_payload.get("edit_note") is False:
                request_payload.pop("edit_note")  # Read-only compatibility for pre-retirement commands.
            request = TaskCreate.model_validate(request_payload)
            require_command_create(command, request, mode=body["mode"])
            if command.tool_name == "start_media_task":
                from app.tasks.media_editing import require_media_input
                pin = require_media_input(command, body["media_input"])
                if str(pin.area_id) != body["area_id"]:
                    raise ValueError()
            elif "media_input" in body:
                raise ValueError()
            if body["base_revision"] != 0 or (
                request.knowledge_area_id is not None and str(request.knowledge_area_id) != body["area_id"]
            ):
                raise ValueError()
        else:
            if "media_input" in body:
                raise ValueError()
            request = TaskMutation.model_validate(body["request"])
            if body["base_revision"] != request.expected_revision or body["mode"] is not None:
                raise ValueError()
        request_payload = request.model_dump(mode="json")
        if body["operation"] == "create" and body["request"].get("edit_note") is False:
            request_payload["edit_note"] = False
        if request.request_id != command.operation_id or request_payload != body["request"]:
            raise ValueError()
        return command, body
    except (KeyError, TypeError, ValueError, AttributeError, TaskError) as error:
        raise TaskError("task_command_source_unavailable", "The immutable command source is unavailable or changed.", 403) from error


async def task_command_source(tasks, connection, task, *, event_id=None, event_revision=None,
                              operation_id=None, current=True, lock=True):
    """Durable validation independent of voice connection lifetime.

    Generic derived records, even with matching prose, cannot pass. The exact
    command must have an existing task request receipt committed with that task.
    Only the scoped note-grant owner reuses this command proof. Ordinary derived
    grants and capability preferences remain outside this admission path.
    """
    from app.tasks.context_repository import RECORD_SQL, context_record

    if lock:
        if not connection.is_in_transaction():
            raise RuntimeError("Command source locking requires a transaction")
        await connection.fetchval("select 1 from jarvis_context_streams where owner_id=$1 for share", task.owner_id)
    if operation_id is not None:
        clause = " where e.owner_id=$1 and e.source_namespace='voice_command' and e.source_event_id=$2 and e.revision=1"
        identity = str(operation_id)
    else:
        clause, identity = " where e.owner_id=$1 and e.id=$2", event_id or task.request_event_id
    row = await connection.fetchrow(RECORD_SQL + clause + (" for share of e,p" if lock else ""), task.owner_id, identity)
    if row is None:
        raise TaskError("task_command_source_unavailable", "The command source is missing.", 403)
    record = context_record(row)
    decoded = decode_task_command(record)
    if decoded is None:
        if operation_id is not None:
            raise TaskError("task_command_source_unavailable", "The command source is missing.", 403)
        return None
    command, body = decoded
    revision = task.request_revision if current else event_revision
    receipt = await connection.fetchrow(
        "select * from jarvis_task_requests where owner_id=$1 and request_id=$2", task.owner_id, command.operation_id)
    if (record.owner_id != task.owner_id or (revision is not None and record.revision != revision)
            or body["task_id"] != str(task.id)
            or body["area_id"] != (str(task.knowledge_area_id) if task.knowledge_area_id is not None else None)
            or (body["operation"] == "create" and command.session_id != task.session_id)
            or not receipt or receipt["task_id"] != task.id or receipt["operation"] != body["operation"]
            or receipt["parameters_digest"] != body["parameters_digest"]
            or receipt["accepted_revision"] != body["base_revision"] + 1
            or receipt["accepted_revision"] > task.revision
            or (current and receipt["accepted_revision"] != task.revision)):
        raise TaskError("task_command_source_unavailable", "The command has no matching immutable task receipt.", 403)
    await require_command_target(connection, task, lock=lock)
    if command.tool_name == "start_media_task" and body["operation"] == "create":
        from app.tasks.media_editing import media_description, validate_media_input
        _, pin, _ = await media_description(connection, task, lock=lock)
        if body.get("media_input") != pin.model_dump(mode="json"):
            raise TaskError("media_command_scope_unavailable", "The pinned media input changed.", 403)
        await validate_media_input(connection, task.owner_id, task.session_id, pin, lock=lock)
    return record
