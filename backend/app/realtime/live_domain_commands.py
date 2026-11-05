"""API-process tools for admitted Live/Hermes work, not another agent.

The exchange owns admission, call identity and journaling. Domain services own
effects and receipts. Generic file work remains with Hermes; this adapter never
adds a domain-specific agent or changes invocation scope.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from app.assistant.models import BehaviorProfileRevisionConflict, BehaviorProfileUpdate
from app.files.tools import FileToolModule
from app.files.workspace_access import WorkspaceFileAccess
from app.knowledge.files import SourceFileRepository
from app.knowledge.models import SourceError
from app.realtime.live_state_reply import compact_live_state_reply
from app.tasks.live_command_exchange import LiveCommandReceipt
from app.workspace.commands import _file_view, _json_input
from app.workspace.models import StoredText, StrictRecord, WorkspaceContext


FileVersion = Annotated[StoredText, Field(min_length=1, max_length=256)]


class _Empty(StrictRecord):
    pass


class _LiveStateRead(StrictRecord):
    section: Literal["human", "system", "all"] = "all"


class _VoiceMessage(StrictRecord):
    text: Annotated[StoredText, Field(min_length=1, max_length=4000)]


class _FileRead(StrictRecord):
    file_id: UUID
    version_id: FileVersion | None = None


class _FileSearch(StrictRecord):
    query: Annotated[StoredText, Field(max_length=500)] | None = None
    limit: Annotated[int, Field(strict=True, ge=1, le=8)] = 8


class _Present(StrictRecord):
    artifact_kind: Literal["file"]
    artifact_id: UUID
    version_id: FileVersion


class _OperationRead(StrictRecord):
    domain: Literal["presentation"]
    operation_id: UUID


class _SandboxView(StrictRecord):
    task_id: UUID
    allocation_id: UUID | None = None


def _available(summary, **fields):
    return {"status": "success", "outcome": "available", "effect_committed": False,
            "summary": summary, **fields}


def _roots(*identities):
    """Only identities from trusted domain reads, never model-supplied roots."""
    return tuple(dict.fromkeys(UUID(str(value)) for value in identities if value is not None))


def _file_roots(file):
    return _roots((file.memory_source or {}).get("payload_id"))


class LiveDomainCommands:
    def __init__(self, services, application, context):
        self.services, self.application, self.context = services, application, context
        self.preferences = getattr(services, "behavior_profile_service", None)
        self.presentations = getattr(services, "workspace_presentations", None)
        self.sandbox_views = getattr(services, "sandbox_views", None)
        resolver = getattr(self.presentations, "resolver", None)
        self.files = getattr(resolver, "files", None)
        if self.files is None:
            recording = getattr(services, "recording_source_access", None)
            if recording is not None:
                recording.check(context.owner_id)
                self.files = WorkspaceFileAccess(recording.files)
            else:
                # Canonical Files may be configured without Notes/presentation.
                repositories = [module.repository for module in getattr(services, "tool_modules", ())
                    if isinstance(module, FileToolModule)
                    and isinstance(module.repository, SourceFileRepository)
                    and module.repository.owner_id == context.owner_id
                    and module.repository.task_evidence is None]
                if len(repositories) == 1:
                    self.files = WorkspaceFileAccess(repositories[0])
        self._models, self._definitions = {}, {}
        if callable(getattr(application, "read_live_state", None)):
            self._add("read_live_state", _LiveStateRead,
                "Read current state: human for observations, system for status/IDs, all by default. "
                "Preserve evidence times, uncertainty and explicit omissions.")
        self._add("reply_to_user", _VoiceMessage,
            "Send a response or clarification to the user's delegated request through Jarvis's one voice. "
            "Use actual receipts for completion claims. Acceptance is not proof the user heard it.")
        self._add("notify_user", _VoiceMessage,
            "Explicitly request the user's attention for a meaningful notice, warning or alert. "
            "Do not use for routine progress. You choose when a notice matters; quiet and delivery policy still apply.")
        if callable(getattr(self.preferences, "get", None)):
            self._add("read_preferences", _Empty, "Read the user's editable instructions and current revision.")
            if callable(getattr(self.preferences, "update", None)):
                self._add("edit_preferences", BehaviorProfileUpdate,
                    "Replace the user's editable instructions at the exact revision read. Preserve unrelated "
                    "preferences. Saving does not prove any active voice model has consumed the update.")
        if callable(getattr(self.files, "get", None)):
            self._add("read_workspace_file", _FileRead,
                "Read authorized retained file metadata and exact version, not raw bytes or signed URLs.")
        if callable(getattr(self.files, "scoped", None)):
            self._add("search_workspace_files", _FileSearch,
                "Search authorized retained Files metadata. A full page may not include every matching file.")
        self._presentation_kinds = ["file"] if getattr(resolver, "files", None) is not None else []
        if self._presentation_kinds and callable(getattr(self.presentations, "present", None)):
            self._add("present_workspace_resource", _Present,
                "Ask the phone to open an exact retained file version. An accepted request is not proof "
                "that the phone displayed it. This tool does not export or download content.")
            self._definitions["present_workspace_resource"]["parameters"]["properties"]["artifact_kind"]["enum"] = self._presentation_kinds[:]
        self._receipt_owners = {name: service for name, service in
            (("presentation", self.presentations),)
            if callable(getattr(service, "operation_receipt", None))}
        if self._receipt_owners:
            self._add("read_workspace_operation", _OperationRead,
                "Read an exact owning service's operation receipt without repeating the effect. "
                "A missing receipt means the outcome is unknown, not that nothing happened.")
            self._definitions["read_workspace_operation"]["parameters"]["properties"]["domain"]["enum"] = list(self._receipt_owners)
        recording = getattr(application, "recording", None)
        if (callable(getattr(recording, "definition", None))
                and callable(getattr(application, "execute_recording", None))):
            definition = recording.definition()
            self._models[definition.name] = None  # The existing dispatcher validates action-specific inputs.
            self._definitions[definition.name] = {"name": definition.name,
                "description": definition.description, "parameters": deepcopy(definition.input_schema)}
        if (callable(getattr(self.sandbox_views, "present", None))
                and callable(getattr(self.presentations, "present", None))):
            self._add("view_sandbox", _SandboxView,
                "Open a view-only stream of the exact work's already active sandbox on the phone. "
                "Supply an allocation ID when needed. Never creates, resumes or extends a sandbox.")

    def _add(self, name, model, description):
        self._models[name] = model
        self._definitions[name] = {"name": name, "description": description,
                                   "parameters": model.model_json_schema()}

    def catalog(self):
        return deepcopy(list(self._definitions.values()))

    @staticmethod
    def _current(admit_if):
        if not callable(admit_if) or not admit_if():
            raise SourceError("live_command_route_closed", "The admitted Live route is no longer current.", 403)

    async def execute(self, name, arguments, *, invocation, provider_call_id, backend_session_key, admit_if):
        if name not in self._models:
            raise ValueError("Unsupported Live domain command")
        arguments = _json_input(arguments)
        model = self._models[name]
        parsed = model.model_validate(arguments) if model is not None else None
        ctx = WorkspaceContext(self.context.owner_id, self.context.jarvis_session_id, invocation.operation_id)
        if (ctx.operation_id is None or invocation.task.owner_id != ctx.owner_id
                or invocation.task.session_id != ctx.session_id):
            raise SourceError("live_command_scope_mismatch", "The work does not belong to this Live session.", 403)
        self._current(admit_if)
        async with invocation.tasks.transaction(ctx.owner_id) as connection:
            await invocation.authorize(connection)
            self._current(admit_if)
            if name in {"reply_to_user", "notify_user"}:
                await invocation.tasks.emit(connection, invocation.task,
                    "live_voice_reply" if name == "reply_to_user" else "live_voice_notice",
                    attempt_id=invocation.claim.attempt_id,
                    payload={"text": parsed.text, "operation_id": str(invocation.operation_id),
                             "agent_selected": True}, source_payload_ids=(invocation.request_payload_id,))
                return LiveCommandReceipt({"status": "success", "outcome": "accepted",
                    "effect_committed": False, "operation_id": str(invocation.operation_id),
                    "summary": "Voice delivery was requested. Playback and hearing are not confirmed."})
        # Never retain the task-owner transaction around another owner's I/O.
        self._current(admit_if)
        if name == "read_live_state":
            state, roots = await self.application.read_live_state()
            reply = compact_live_state_reply(state, parsed.section)
            return LiveCommandReceipt(_available("Current Live State was explicitly requested.", live_state=reply), roots)
        if name == "read_preferences":
            return LiveCommandReceipt(_available("The current editable instructions are available.",
                preferences=await self.preferences.get(owner_id=ctx.owner_id)))
        if name == "edit_preferences":
            try:
                result = await self.preferences.update(owner_id=ctx.owner_id, update=parsed)
            except BehaviorProfileRevisionConflict as error:
                return LiveCommandReceipt({"status": "success", "outcome": "conflict",
                    "effect_committed": False, "error": error.error_payload(),
                    "summary": "Preferences changed. Read the current revision before preparing another edit."})
            return LiveCommandReceipt({"status": "success", "outcome": "completed",
                "effect_committed": True, "preferences": result,
                "summary": "Preferences were saved. Active model consumption is not confirmed."})
        if name == "read_workspace_file":
            file = await self.files.get(ctx.owner_id, ctx.session_id, parsed.file_id, parsed.version_id)
            return LiveCommandReceipt(_available("Retained file metadata is available.", **_file_view(file)),
                                      _file_roots(file))
        if name == "search_workspace_files":
            repository = await self.files.scoped(ctx.owner_id, ctx.session_id)
            files = await repository.list_files(owner_id=ctx.owner_id, query=parsed.query, limit=parsed.limit)
            return LiveCommandReceipt(_available("Retained file search results are available.",
                items=[_file_view(file) for file in files], page_full=len(files) == parsed.limit),
                _roots(*(root for file in files for root in _file_roots(file))))
        if name == "present_workspace_resource":
            if parsed.artifact_kind not in self._presentation_kinds:
                raise ValueError("This presentation kind is not backed")
            result = await self.presentations.present(ctx, action="open", **parsed.model_dump())
            return LiveCommandReceipt(result, await self._artifact_roots(ctx, result))
        if name == "read_workspace_operation":
            owner = self._receipt_owners.get(parsed.domain)
            if owner is None:
                raise ValueError("This operation domain is not backed")
            result = await owner.operation_receipt(ctx, parsed.operation_id)
            if result is None:
                return LiveCommandReceipt({"status": "success", "outcome": "outcome_unknown",
                    "effect_committed": None, "operation_id": str(parsed.operation_id),
                    "summary": "No authoritative receipt is available; do not repeat the effect blindly."})
            return LiveCommandReceipt(result, await self._artifact_roots(ctx, result))
        if name == "live_recording":
            result = await self.application.execute_recording(arguments, provider_call_id=provider_call_id,
                backend_session_key=backend_session_key, admit_if=admit_if)
            return LiveCommandReceipt(result)
        if name == "view_sandbox":
            result = await self.sandbox_views.present(ctx, self.presentations,
                task_id=parsed.task_id, allocation_id=parsed.allocation_id)
            return LiveCommandReceipt(result)
        raise ValueError("Unsupported Live domain command")

    async def _artifact_roots(self, ctx, result):
        """Optional exact-version evidence must not erase an already saved receipt.

        The receipt remains truthful if its content was withdrawn, or a later
        provenance read fails. No latest-version fallback and no effect replay.
        """
        artifact = result.get("artifact")
        if not isinstance(artifact, dict) or not artifact.get("version_id"):
            return ()
        try:
            if artifact.get("kind") == "file" and self.files is not None:
                file = await self.files.get(ctx.owner_id, ctx.session_id, UUID(artifact["artifact_id"]),
                                            artifact["version_id"])
                return _file_roots(file)
        except Exception:
            # This is only a provenance lookup after an owning-service result.
            # Cancellation (BaseException) still propagates; effects are not retried.
            return ()
        return ()
