"""Explicit foreground Files & data admission; composition remains opt-in."""
from dataclasses import dataclass
from typing import ClassVar
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, ValidationError

from app.plugins.models import FrozenRecord
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskError, TaskVoiceSource, json_text
from app.tasks.voice_tools import audit_task_arguments, audit_task_result, task_payload
from app.tools.inputs import failed, success
from app.tools.invocation import CanonicalRequestIdentity, current_tool_invocation
from app.tools.registry import ToolDefinition


class FilesDataTaskArguments(FrozenRecord):
    title: str = Field(min_length=1, max_length=200)
    requested_outcome: str = Field(min_length=1, max_length=32000)
    knowledge_area_id: UUID | None = None
    publish_artifacts: bool = Field(default=False, strict=True)
    request_slot: int = Field(default=1, ge=1, le=20, strict=True)


@dataclass(frozen=True)
class FilesDataTaskToolModule:
    service: object
    name: ClassVar[str] = "managed_files_data_tasks"
    arguments_type: ClassVar[type] = FilesDataTaskArguments
    create_method: ClassVar[str] = "create_files_data"
    identity_namespace: ClassVar[str] = "jarvis.files-data-task.v1"
    route: ClassVar[str] = "files_data"
    tool_name: ClassVar[str] = "start_files_data_task"
    tool_help: ClassVar[str] = (
        "Delegate requested file/table calculations to the selected managed Files & data bundle. "
        "Offline computation only. Publication requires publish_artifacts=true and its separate grant. "
        "After denial do not fall back to a general terminal task. Use existing background-task status/cancel/resume tools.")

    def build_tools(self, context):
        if context.session_id is None or self.service.managed_plugins is None:
            return []

        async def start(arguments):
            try:
                args = self.arguments_type.model_validate(arguments)
                invocation = current_tool_invocation()
                if invocation.task is not None or invocation.request_event_id is None:
                    raise TaskError("task_request_not_finalized", "An exact finalized user request is required.")
                names = invocation.request_identity
                if names is None:
                    record = await ContextRepository(self.service.repository).get(
                        context.owner_id, invocation.request_event_id, session_id=context.session_id)
                    if not record.authorizes_request or record.revision != invocation.request_revision:
                        raise TaskError("task_request_not_finalized", "The original user request is unavailable.")
                    names = CanonicalRequestIdentity.from_record(record)
                identity = uuid5(NAMESPACE_URL, json_text({"namespace": self.identity_namespace,
                    "owner": context.owner_id, "session": str(context.session_id),
                    "source_namespace": names.namespace, "source_event_id": names.event_id, "slot": args.request_slot}))
                task = await getattr(self.service, self.create_method)(context.owner_id, TaskCreate(
                    request_id=identity, title=args.title, requested_outcome=args.requested_outcome,
                    session_id=context.session_id, knowledge_area_id=args.knowledge_area_id,
                    execution_environment="offline_sandbox", publish_artifacts=args.publish_artifacts),
                    source=TaskVoiceSource(invocation.request_event_id, invocation.request_revision, context.session_id),
                    mode="background")
                return success({"accepted": True, "task": task_payload(self.service.repository, task,
                    getattr(self.service, "attention_service", None)), "managed_plugin": self.route,
                    "meaning": "Durably accepted, not submitted, completed, saved or heard. Do not retry as general terminal work."})
            except ValidationError:
                return failed("invalid_tool_input", "Use only the declared managed-task request fields.")
            except TaskError as error:
                return failed(error.code, error.message)

        return [ToolDefinition(self.tool_name, self.tool_help,
            self.arguments_type.model_json_schema(), start, mutates_state=True,
            audit_arguments=audit_task_arguments, audit_result=audit_task_result)]


class MediaTaskArguments(FilesDataTaskArguments):
    publish_artifacts: bool = Field(default=True, strict=True)


class MediaTaskToolModule(FilesDataTaskToolModule):
    name = "managed_media_tasks"
    arguments_type = MediaTaskArguments
    create_method = "create_media"
    identity_namespace = "jarvis.media-task.v1"
    route = "video_editing"
    tool_name = "start_media_task"
    tool_help = (
        "Delegate a requested edit of retained video to the selected managed video editor. "
        "Describe the intended edit and exact available source references; the background worker constructs the recipe. "
        "Uses offline bounded rendering and the separately granted recipe/video publication allowance. "
        "Acceptance is not a completed or saved video. A denial cannot fall back to general terminal work. "
        "Use existing background-task status/cancel/resume tools.")
