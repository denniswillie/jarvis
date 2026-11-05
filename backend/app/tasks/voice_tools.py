"""Fast, provider-neutral voice access to the durable task service.

Models choose what warrants delegation. This boundary owns identity, scope and
receipts, not an agent loop or semantic task classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.attention.models import AttentionError
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes_result import inspect_expert_output
from app.tasks.models import TaskCreate, TaskError, TaskMutation, TaskVoiceSource, digest, json_text
from app.tools.inputs import failed, success
from app.tools.invocation import current_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.registry import ToolDefinition


TASK_TOOL_NAMES = frozenset({
    "start_background_task", "list_background_tasks", "get_background_task",
    "update_background_task", "cancel_background_task", "resume_background_task",
})


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*")
    @classmethod
    def safe_text(cls, value):
        if isinstance(value, str) and "\x00" in value:
            raise ValueError("text must contain no NUL")
        return value


class StartArguments(Arguments):
    title: str = Field(min_length=1, max_length=200)
    requested_outcome: str = Field(min_length=1, max_length=32000, description=(
        "Faithful assignment summary. Copy explicit user restrictions verbatim instead of paraphrasing "
        "them; do not add constraints or substitute a ban on proposals for a ban on performing changes."))
    mode: Literal["consult", "background"] = "background"
    knowledge_area_id: UUID | None = Field(default=None, description=(
        "Omit to retain the knowledge area associated with this exact user request when source-backed memory is selected. "
        "Supply only a known owned area for a deliberately different assignment. Later topic switches do not move accepted tasks."))
    request_slot: int = Field(default=1, ge=1, le=20, strict=True, description=(
        "Ordinal of independent tasks explicitly requested in THIS user message. Use 1 normally. "
        "Reuse the same slot after interruption or a retry; never increment it to retry a task."))
    requested_capabilities: list[Literal["reasoning", "session_context", "sandbox_compute", "sandbox_network",
                                        "sandbox_desktop", "publish_files", "memory_edit"]] = Field(
        default_factory=lambda: ["reasoning", "session_context"], max_length=7,
        description="Add sandbox_compute for offline commands, or sandbox_network for internet-enabled commands "
                    "including installing dependencies; network includes computation, not root or external account actions. "
                    "Add publish_files only for requested durable output; it implies offline computation unless "
                    "sandbox_network is also selected. Add sandbox_desktop for browser/desktop observation and "
                    "interaction in the isolated environment; add sandbox_network too for public websites. "
                    "Desktop includes computation, not access to personal accounts or the user's host desktop. "
                    "Computation alone never authorizes publication. Availability requires a ready capable worker; "
                    "never claim acceptance before the receipt. Add memory_edit only when the user requests "
                    "corrections to retained knowledge. Let the frontier inspect originals, select sources and "
                    "preserve unrelated facts; do not formulate replacement bodies in foreground. "
                    "Read-only review, suggestions and recall do not grant memory editing.")


class ListArguments(Arguments):
    active_only: bool = True
    limit: int = Field(default=10, ge=1, le=20, strict=True)
    offset: int = Field(default=0, ge=0, le=100000, strict=True)


class GetArguments(Arguments):
    task_ref: str = Field(pattern=r"^task_[0-9a-f]{12}$")
    expected_revision: int | None = Field(default=None, ge=1, strict=True)


class MutationArguments(Arguments):
    task_ref: str = Field(pattern=r"^task_[0-9a-f]{12}$")
    expected_revision: int = Field(ge=1, strict=True)
    message: str = Field(default="", max_length=32000, description=(
        "Full revised objective for updates. Retain still-applicable goals and copy explicit restrictions "
        "verbatim; a routing summary must not invent or strengthen the user's instructions."))



class CancelArguments(MutationArguments):
    suppress_result_announcement: bool = Field(default=False, strict=True, description=(
        "Set true only when the user explicitly asks not to announce this task's result. "
        "Completed work stays completed; an exact result announcement may be dismissed separately. "
        "Read both work and delivery outcomes. This does not undo external effects or prove audio was unheard."))


def answer_delivery_status(coordinator=None):
    # Report process-local coordinator availability, not task completion or
    # assumed playback. Status reads never start a delivery or expose its prose.
    if coordinator is not None:
        return coordinator.voice_status()
    return {"route": "direct_tts", "state": "unavailable", "reason": "attention_coordinator_unavailable"}


def task_payload(repository, task, coordinator=None):
    return {
        "task_ref": repository.reference(task.id), "revision": task.revision,
        "title": task.title, "mode": task.mode, "state": task.status.value,
        "reason": task.reason, "pending_state": task.pending_status,
        "session_id": str(task.session_id) if task.session_id else None,
        "knowledge_area_id": str(task.knowledge_area_id) if task.knowledge_area_id else None,
        "updated_at": task.updated_at.isoformat(),
        "expert_owns_substantive_followups": True,
        "delivery_state": "not_established_by_task_status",
        "answer_delivery": answer_delivery_status(coordinator),
    }


def audit_task_arguments(arguments):
    return {"input_digest": digest(arguments), **{
        key: arguments[key] for key in ("task_ref", "expected_revision", "request_slot", "mode", "suppress_result_announcement")
        if key in arguments
    }}


def audit_task_result(result):
    """Keep durable operation identity, not copied titles, questions or expert prose."""
    receipt = {"status": result.get("status", "failed"), "content_omitted": True,
               "meaning": "Historical task-operation receipt, not current task state. Read the task for current execution and delivery availability."}
    if "accepted" in result:
        receipt["accepted"] = result["accepted"]
    task = result.get("task")
    if isinstance(task, dict):
        receipt["task"] = {key: task[key] for key in ("task_ref", "revision", "state", "knowledge_area_id", "delivery_state") if key in task}

    for key in ("work", "delivery"):
        if isinstance(result.get(key), dict):
            # These application-authored receipts contain identities/states, not result prose.
            receipt[key] = {name: result[key][name] for name in (
                "status", "cancellation_requested", "completed_work_cancelled", "external_effects",
                "reason", "durable_suppression", "intent_id", "task_revision", "source_payload_id",
                "generation", "revoked_generation", "replayed_request", "hearing",
                "prior_delivery", "transport_stop",
            ) if name in result[key]}
    error = result.get("error")
    if isinstance(error, dict):
        receipt["error"] = {"code": error.get("code", "task_operation_failed"),
                            "message": "Read current task status or correct the tool request."}
    return receipt


@dataclass(frozen=True)
class BackgroundTaskToolModule:
    service: object
    name: ClassVar[str] = "background_tasks"

    def build_tools(self, context: ToolBuildContext):
        if context.session_id is None:
            return []
        repository = self.service.repository
        sources = ContextRepository(repository)

        async def request_source(operation, *, task_ref=None, slot=1):
            invocation = current_tool_invocation()
            if invocation.task is not None or invocation.request_event_id is None:
                raise TaskError("task_request_not_finalized",
                    "No exact finalized user request is bound to this call. Do not claim acceptance; ask for clarification.", 409)
            names = invocation.request_identity
            if names is None:
                # Compatibility for trusted callers that supply only an exact
                # event/revision. Native voice already resolved these names.
                record = await sources.get(context.owner_id, invocation.request_event_id, session_id=context.session_id)
                if not record.authorizes_request or record.revision != invocation.request_revision:
                    raise TaskError("task_request_not_finalized", "The bound user transcript is unavailable or changed.", 409)
                from app.tools.invocation import CanonicalRequestIdentity
                names = CanonicalRequestIdentity.from_record(record)
            # TaskRepository rechecks current source ownership/finality/version
            # inside the atomic mutation. These names only avoid a duplicate read.
            # Independent of provider call ID and transcript wording. Revised ASR
            # cannot allocate another task. Proven newer ASR uses owner update;
            # incompatible or same-revision competing proposals still conflict.
            identity = uuid5(NAMESPACE_URL, json_text({
                "namespace": "jarvis.voice_task.v1", "owner": context.owner_id,
                "session": str(context.session_id), "source_namespace": names.namespace,
                "source_event_id": names.event_id, "operation": operation,
                "task_ref": task_ref, "slot": slot,
            }))
            return identity, TaskVoiceSource(invocation.request_event_id, invocation.request_revision, context.session_id)

        async def start(args):
            identity, source = await request_source("create", slot=args.request_slot)
            # Admission imports these shared schemas; defer the reverse import.
            from app.attention.notification_ledger.admission import create_or_correct
            task = await create_or_correct(self.service, context.owner_id, TaskCreate(
                request_id=identity, title=args.title, requested_outcome=args.requested_outcome,
                session_id=context.session_id, knowledge_area_id=args.knowledge_area_id,
                execution_environment=(("networked_desktop_sandbox" if "sandbox_network" in args.requested_capabilities
                    else "desktop_sandbox") if "sandbox_desktop" in args.requested_capabilities else
                    "networked_sandbox" if "sandbox_network" in args.requested_capabilities else
                    "offline_sandbox" if set(args.requested_capabilities) & {"sandbox_compute", "publish_files"} else "none"),
                publish_artifacts="publish_files" in args.requested_capabilities,
                edit_memory="memory_edit" in args.requested_capabilities,
            ), source=source, mode=args.mode, slot=args.request_slot, sources=sources)
            return success({
                "accepted": True, "task": task_payload(repository, task, getattr(self.service, "attention_service", None)),
                "effective_budget": task.budget.model_dump(mode="json"),
                "capabilities": ["reasoning", "session_context"]+(
                    ["sandbox_compute"] if set(args.requested_capabilities) & {
                        "sandbox_compute", "sandbox_network", "sandbox_desktop", "publish_files"} else [])+(
                    ["sandbox_network"] if "sandbox_network" in args.requested_capabilities else [])+(
                    ["sandbox_desktop"] if "sandbox_desktop" in args.requested_capabilities else [])+(
                    ["publish_files"] if "publish_files" in args.requested_capabilities else [])+(
                    ["memory_edit"] if "memory_edit" in args.requested_capabilities else []),
                "external_action_grants": [],
                "meaning": "Durably accepted, not necessarily running, complete, or heard. "
                    "Do not create it again after interrupted speech. Images and optional tools "
                    "are supplied only when actually available. If the user requested several independent tasks, "
                    "finish submitting those requests and collect every admission result before speaking. "
                    "Then give one brief joint acknowledgment of the actual accepted and failed requests and end the turn; "
                    "do not announce each receipt separately, abandon a remaining requested task, "
                    "call status repeatedly, or wait here for an expert answer. Consult is asynchronous too. "
                    "Read status on a later user question or an authorized attention event.",
            })

        async def listing(args):
            tasks = await repository.list(context.owner_id, limit=args.limit + 1,
                offset=args.offset, active_only=args.active_only)
            return success({
                "observed_at": datetime.now(UTC).isoformat(),
                "tasks": [task_payload(repository, task, getattr(self.service, "attention_service", None)) for task in tasks[:args.limit]],
                "has_more": len(tasks) > args.limit,
                "next_offset": args.offset + min(len(tasks), args.limit),
                "meaning": "Exact current references; titles and earlier speech are not task identities.",
            })

        async def get(args):
            task_id = await repository.resolve_reference(context.owner_id, args.task_ref)
            task, details = await repository.voice_detail(context.owner_id, task_id)
            if args.expected_revision is not None and args.expected_revision != task.revision:
                raise TaskError("task_revision_conflict", "Read the current task revision before using this status.")
            result = details.pop("result")
            output = inspect_expert_output(result, task=task)
            # Keep canonical expert content for the authorized delivery reader,
            # not a status lookup or an implicit request to speak.
            return success({"task": task_payload(repository, task, getattr(self.service, "attention_service", None)), **details,
                "expert_output_available": output.expert_available,
                "stored_output_kind": output.kind,
                "expert_speech_prepared": output.speech_ready,
                "expert_output_unavailable_reason": output.unavailable_reason,
                "answer_delivery": answer_delivery_status(getattr(self.service, "attention_service", None)),
                "observed_at": datetime.now(UTC).isoformat(),
                "meaning": "This is a read-only status lookup, not a request to deliver or replay an answer. "
                    "It supersedes older queued/running snapshots. Available expert output is stored, "
                    "not supplied here for retelling and not proof of user access or playback. "
                    "A succeeded task with output is finished even when delivery is unavailable. "
                    "Keep routine status acknowledgments to one short sentence about the supported execution state, then yield. "
                    "Unknown hearing alone is not a task or delivery failure; do not routinely recite missing playback receipts. "
                    "State known delivery failures or unavailable routes when they prevent the requested outcome "
                    "or the user asks about delivery. Preserve actual task failures, cancellation, paused/waiting states "
                    "and relevant result unavailability. Do not promise imminent playback from stored output or route availability. "
                    "Do not invent or substitute the expert's answer or claim delivery or hearing. "
                    "Missing output is not proof that work is running or will resume."})

        async def mutation(args, operation):
            task_id = await repository.resolve_reference(context.owner_id, args.task_ref)

            suppress = operation == "cancel" and getattr(args, "suppress_result_announcement", False)
            identity, source = await request_source(
                "cancel_and_suppress_result" if suppress else operation, task_ref=args.task_ref)
            if suppress:
                task, work, delivery = await self.service.cancel_with_result_suppression(
                    context.owner_id, task_id, TaskMutation(request_id=identity,
                        expected_revision=args.expected_revision, message=args.message), source=source)
                return success({
                    "accepted": work["cancellation_requested"] or delivery["durable_suppression"],
                    "task": task_payload(repository, task, getattr(self.service, "attention_service", None)),
                    "work": work, "delivery": delivery,
                    "meaning": (
                        "Work cancellation and result-announcement suppression are separate outcomes. "
                        "Already-completed work and its result are preserved. Claim suppression only when "
                        "delivery.durable_suppression is true; a missing exact intent is not a future mute. "
                        "Previously forwarded or played audio is not retracted and hearing is not inferred. "
                        "No task/action is resumed or replayed."
                    ),
                })
            task = await self.service.mutate(context.owner_id, task_id, operation, TaskMutation(
                request_id=identity, expected_revision=args.expected_revision, message=args.message,
            ), source=source)
            return success({"accepted": True, "task": task_payload(repository, task, getattr(self.service, "attention_service", None)),
                "meaning": "Only this task was changed. Cancellation may still need reconciliation; "
                    "speech interruption does not cancel work and no completed external effect is undone."})

        def definition(name, description, schema, handler, *, mutates=False):
            async def invoke(arguments):
                try:
                    parsed = schema.model_validate(arguments)
                    return await handler(parsed)
                except ValidationError:
                    return failed("invalid_tool_input", "Arguments do not match this tool's schema. Copy exact task references and revisions.")
                except (TaskError, AttentionError) as error:
                    return failed(error.code, error.message)
            return ToolDefinition(name=name, description=description,
                input_schema=schema.model_json_schema(), handler=invoke, mutates_state=mutates,
                audit_arguments=audit_task_arguments, audit_result=audit_task_result)

        tools = [
            definition("start_background_task",
                "Delegate a new complex query or sustained task to the frontier expert. Returns after durable acceptance, "
                "without waiting for reasoning. Supply the full outcome and relevant constraints; original context is "
                "assembled by Jarvis. Use mode consult for a bounded expert answer or background for ongoing work. "
                "Read task status after an interruption rather than restarting accepted work.",
                StartArguments, start, mutates=True),
            definition("list_background_tasks",
                "List exact task references, revisions and actual state, including tasks from earlier sessions. "
                "Set active_only=false to resolve completed work. Does not start, resume or cancel anything.",
                ListArguments, listing),
            definition("get_background_task",
                "Read current task status and expert-output/delivery availability, without fetching expert prose "
                "for retelling or initiating playback. Use the returned revision for updates. Expert answers and "
                "clarifications require authorized delivery by the selected voice route; this tool cannot deliver them. "
                "Keep routine status concise. Report actual delivery unavailability when it prevents the requested outcome "
                "or the user asks about delivery. Unknown hearing alone is not a failure. "
                "Work completion does not establish that its answer was heard.",
                GetArguments, get),
            definition("update_background_task",
                "Route a substantive follow-up or correction to its owning expert task. message is the FULL revised "
                "requested outcome, retaining still-applicable goals and incorporating the new user information. "
                "Old execution/output is fenced. Do not independently advance the same technical step in foreground.",
                MutationArguments, lambda args: mutation(args, "update"), mutates=True),
            definition("cancel_background_task",
                "Cancel only the exact task the user intends to stop. A playback interruption or ambiguous 'stop' "

                "with multiple tasks is not permission to cancel all work. Read current state before claiming it stopped. "
                "When the user also explicitly says not to announce this result, set suppress_result_announcement=true "
                "and report the separate work and delivery receipts. Completed work cannot be retroactively cancelled.",
                CancelArguments, lambda args: mutation(args, "cancel"), mutates=True),
            definition("resume_background_task",
                "Resume an explicitly requested paused or waiting task after reconciliation. To incorporate an answer "
                "to the expert's question or change its objective, use update_background_task instead.",
                MutationArguments, lambda args: mutation(args, "resume"), mutates=True),
        ]
        # Keep accepted work inspectable and controllable when its provider is
        # disabled. Worker health is transient and remains an admission check,
        # not a reason to remove a statically configured capability.
        if not self.service.provider_configured("frontier"):
            tools = [tool for tool in tools if tool.name != "start_background_task"]
        return tools
