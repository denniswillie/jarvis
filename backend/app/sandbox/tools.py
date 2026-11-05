"""Compose bounded computation with, not instead of, exact external-action tools."""
import asyncio
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.sandbox.broker import SandboxCommandBroker
from app.sandbox.inputs import SandboxInputService
from app.sandbox.images import SandboxImageTools
from app.sandbox.artifacts import SandboxArtifactTools
from app.tasks.generated_artifacts import PUBLISH_TOOL
from app.images import ImageUnavailableError
from app.knowledge.models import SourceError
from app.sandbox.models import CommandRequest, SandboxError, SandboxPolicy
from app.tasks.models import ClaimLost, TaskError, json_object, json_text
from app.tools.inputs import failed
from app.tools.registry import ToolDefinition


class CommandLookup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_key: str | None = Field(max_length=100, min_length=1)


class SandboxToolBridge:
    def __init__(self, bridge, provider, grant_id, policy, *, files=None, managed_job=None, managed_plugins=None, continuation_id=None, reusable=False):
        self.base, self.context, self.registry = bridge, bridge.context, bridge.registry
        self.broker = SandboxCommandBroker(self.context, provider, grant_id, managed_plugins=managed_plugins,
            reusable=reusable, continuation_id=continuation_id)
        self.managed_job = managed_job
        self.policy = policy
        self.artifacts = None
        self.media = None
        self.media_names = frozenset()
        self.artifact_names = frozenset()
        self.inputs = SandboxInputService(bridge, self.broker, files=files)
        self.images = SandboxImageTools(bridge, self.broker, self.inputs)
        input_definitions = self.inputs.definitions()
        self.input_names = frozenset(item.name for item in input_definitions)
        network_help = (
            "Outbound internet is enabled for this task. " if policy.internet_access else
            "Outbound access is limited to the domains in the recorded scope. " if policy.allowed_domains else
            "This scope is offline; downloads are unavailable. "
        )
        desktop_help = ("An isolated Xfce desktop is available on DISPLAY=:0. Use installed Linux controls "
            "such as xdotool for window, pointer and keyboard input, and scrot plus sandbox_view_image for "
            "screenshots. Discover available application executables (for example with command -v) instead of "
            "assuming a browser or editor name. If a launch shows no window, inspect its log and the installed "
            "alternatives before declaring the task blocked. Choose interaction steps for the requested task. Launch persistent "
            "GUI applications in the background with their input/output redirected. Observe actual application "
            "state before claiming success; a successful input command alone does not prove the intended effect. "
            "After uncertain input, inspect the UI and existing command status before deciding whether to retry. "
            if policy.template == "desktop" else "No desktop is admitted for this task. ")
        self.definitions = (
            ToolDefinition("sandbox_execute",
                "Run Linux shell commands inside the authorized sandbox. " + network_help + desktop_help +
                "Choose programs/libraries to compute, create, inspect and repair workspace files. Install missing "
                "dependencies when the recorded network scope allows it, using a writable workspace environment "
                "such as .venv; reuse that environment in later commands. No root/sudo or host installation. "
                "Report missing dependencies or failed installs honestly. Choose one stable "
                "operation_key per logical command and reuse identical parameters when checking a retry. "
                "An unknown outcome is NOT failure or permission to execute under a new key. "
                "Workspace files are not durable published artifacts. "
                "Commands may mutate the sandbox only; no local host access, secrets, or unapproved external actions. "
                "Use sandbox_status for running/unknown commands. Report exit code and any truncated output honestly.",
                CommandRequest.model_json_schema(), self._execute, mutates_state=True),
            ToolDefinition("sandbox_status",
                "Observe a previous sandbox command without executing it again. Use its operation_key; "
                "use null to list the latest 20 command keys. This does not cancel, restart or extend work.",
                CommandLookup.model_json_schema(), self._status),
        )
        from app.plugins.media import ManagedMediaJob
        if isinstance(managed_job, ManagedMediaJob):
            # Only the media adapter may form the fixed renderer command. Do
            # not expose Python/general shell or arbitrary output publication.
            self.definitions = (self.definitions[1],)
        elif managed_job is not None:
            from app.plugins.managed import FilesDataCommand
            self.definitions = (ToolDefinition("files_data_execute",
                "Execute Python code using this job's pinned, qualified Files & data bundle. "
                "Use stdlib CSV/JSON/statistics/decimal and workspace files; no downloads or implicit publication. "
                "Reuse the same operation_key and exact code for retries; unknown is not permission to repeat. "
                "Use sandbox_status to observe running commands. Do not fall back to general terminal execution.",
                FilesDataCommand.model_json_schema(), self._execute, mutates_state=True), self.definitions[1])
        self.definitions += input_definitions + (self.images.definition(),)
        self.registry.register_many(self.definitions)
        self.handlers = {definition.name: definition.handler for definition in self.definitions}

    @classmethod
    async def attach(cls, bridge, provider, *, policy_limit=None, files=None, artifact_configuration=None,
                     managed_plugins=None):
        context = bridge.context
        async with context.tasks.transaction(context.task.owner_id) as connection:
            task, _ = await context.tasks.lock_claim(connection, context.claim)
            if await context.tasks.context_problem(connection, task):
                raise ClaimLost()
            grants = await connection.fetch("""select g.grant_id from jarvis_task_grants g
                join jarvis_operation_grants a on a.owner_id=g.owner_id and a.id=g.grant_id
                where g.owner_id=$1 and g.task_id=$2 and a.capability='sandbox.execute'""",
                task.owner_id, task.id)
            if not grants:
                return bridge
            if len(grants) != 1:
                raise TaskError("sandbox_grant_ambiguous", "Use one computation scope per task.")
            grant = await context.tasks.validate_grant(connection, task.owner_id, grants[0]["grant_id"])
            policy = SandboxPolicy.model_validate(json_object(grant["scope"]))
            if policy_limit is not None and not policy.within(policy_limit):
                raise TaskError("sandbox_policy_exceeds_worker", "The granted computation bounds exceed this worker's policy.")
            if policy.networked and policy_limit is None:
                raise TaskError("sandbox_network_disabled", "This worker has no configured network execution policy.", 503)
            if policy.template == "desktop" and policy_limit is None:
                raise TaskError("sandbox_desktop_disabled", "This worker has no configured desktop policy.", 503)
            from app.plugins.managed import ManagedFilesDataRuntime, load_managed_task
            managed = await load_managed_task(connection, task)
            if managed is not None and (not isinstance(managed_plugins, ManagedFilesDataRuntime)
                    or managed[1] != grant["id"] or policy.template != "base" or policy.networked):
                raise TaskError("managed_worker_unavailable", "No qualified managed route serves this scope.", 503)
        from app.sandbox.allocations import SandboxAllocationRepository
        continuation_id = await SandboxAllocationRepository(context.tasks).continuation_for_task(context.claim)
        result = cls(bridge, provider, grant["id"], policy, files=files,
                     managed_job=managed[0] if managed is not None else None, managed_plugins=managed_plugins,
                     reusable=True, continuation_id=continuation_id)
        if managed is not None:
            bridge.bind_managed_plugins(managed_plugins)
        result.artifacts = await SandboxArtifactTools.attach(bridge, result.broker, files, artifact_configuration)
        from app.plugins.media import ManagedMediaJob
        if isinstance(result.managed_job, ManagedMediaJob):
            if result.artifacts is None:
                raise TaskError("managed_media_publication_unavailable", "Media requires its separate exact publication allowance.", 503)
            from app.sandbox.media import ManagedMediaTools
            result.media = ManagedMediaTools(bridge, result.broker, result.inputs, result.artifacts,
                                             result.managed_job, files)
            definitions = result.media.definitions()
            result.definitions += definitions
            result.media_names = frozenset(item.name for item in definitions)
            result.registry.register_many(definitions)
            result.handlers.update({item.name: item.handler for item in definitions})
        elif result.artifacts is not None:
            definitions = result.artifacts.definitions()
            result.definitions += definitions
            result.artifact_names = frozenset(item.name for item in definitions)
            result.registry.register_many(definitions)
            result.handlers.update({item.name: item.handler for item in definitions})
        return result

    def __getattr__(self, name):
        return getattr(self.base, name)

    async def resolve_image(self, name, result):
        if name == "sandbox_view_image":
            return await self.images.resolve(result)
        return await self.base.resolve_image(name, result)

    def catalog(self):
        return [item for item in self.base.catalog() if item["name"] != PUBLISH_TOOL]+[{"name": item.name, "description": item.description,
            "parameters": item.input_schema} for item in self.definitions]

    def authorized_operations(self):
        result = [item for item in self.base.authorized_operations() if item["tool"] != PUBLISH_TOOL]+[{
            "tool": ("media_render" if self.media is not None else
                     "files_data_execute" if self.managed_job is not None else "sandbox_execute"),
            "capability": "sandbox.execute", "scope": self.policy.model_dump(mode="json"),
            "meaning": "Sandbox execution within the recorded environment/network scope, not an external-action grant. "
                       "Do not include command IDs in completed_operation_refs.",
        }]
        if self.media is not None:
            result[-1]["renderer"] = {
                "template": self.media.job.bundle.renderer.template.model_dump(mode="json"),
                "binding": "Runtime-supplied from this task's admitted renderer; not a user/model selection "
                           "or fresh readiness observation. Execution still rechecks the bound authority.",
            }
        if self.artifacts is not None:
            result.append({"tool": "media_publish" if self.media is not None else "sandbox_publish_file", "capability": "files.publish_generated",
                "scope": self.artifacts.policy.model_dump(mode="json"),
                "meaning": "Bounded publication, not permission for unrelated effects or changing an existing saved snapshot. "
                           "Use the returned accepted task_operation reference in completed_operation_refs."})
        return result

    async def _execute(self, arguments):
        try:
            request = (self.managed_job.bundle.command(arguments) if self.managed_job is not None
                       else CommandRequest.model_validate(arguments))
        except ValidationError as error:
            return self.invalid(error)
        return await self.broker.execute(request)

    async def _status(self, arguments):
        try:
            lookup = CommandLookup.model_validate(arguments)
        except ValidationError as error:
            return self.invalid(error)
        reservation = await self.context.reserve("sandbox-status:"+uuid4().hex, tool_steps=1)
        await self.context.budgets.submit(self.context.claim, reservation["id"])
        try:
            return await self.broker.status(lookup.operation_key)
        finally:
            await asyncio.shield(self.context.settle(reservation["id"], input_tokens=0, output_tokens=0, tool_steps=1))

    @staticmethod
    def invalid(error):
        # Show declared field names/error kinds, never copy rejected values.
        issues = [{"field": str(item["loc"][0]) if item["loc"] and item["loc"][0] in
                   {"operation_key", "command", "cwd"} else "unrecognized_field",
                   "issue": item["type"]} for item in error.errors(include_input=False, include_url=False)]
        return failed("invalid_tool_input", "No execution was submitted by this invalid invocation. "
            "Provide operation_key and command; cwd is optional and must be workspace-relative, usually '.'. "
            "Status takes operation_key or null. Correct these fields: "+json_text({"issues":issues}))

    async def invoke(self, name, arguments, *, provider_call_id):
        if name not in self.handlers:
            return await self.base.invoke(name, arguments, provider_call_id=provider_call_id)
        if (not isinstance(provider_call_id, str) or not 1<=len(provider_call_id)<=200
                or not provider_call_id.strip() or "\x00" in provider_call_id):
            raise ValueError("A bounded native tool-call identity is required")
        arguments = json_object(arguments)
        if len(json_text(arguments))>64000:
            return failed("invalid_tool_input", "Tool arguments exceed the task transport bound.")
        await self.base._check()
        try:
            await self.inputs.revalidate_staged()
            if name in self.media_names:
                result = await self.media.invoke(name, arguments, provider_call_id=provider_call_id)
            else:
                result = (await self.artifacts.invoke(name, arguments, provider_call_id=provider_call_id)
                          if name in self.artifact_names else await self.handlers[name](arguments))
        except ClaimLost:
            raise
        except ValidationError:
            return failed("sandbox_internal_validation", "An internal receipt or scope could not be validated. "
                          "This is not proof that the command failed; inspect its existing status.")
        except (TaskError, SandboxError, SourceError, ImageUnavailableError) as error:
            if name == "sandbox_view_image":
                return failed(error.code, "The workspace image could not be inspected. Do not infer its contents "
                              "or claim that it is the current screen.")
            if name in self.artifact_names:
                return failed(error.code, "The file operation could not be confirmed. Inspect sandbox_artifact_status "
                              "with its existing key; do not invent a successful save or repeat under a new key.")
            if name in self.input_names:
                return failed(error.code, "The exact input could not be staged or confirmed. "
                              "Do not substitute a summary, infer missing evidence, or claim publication.")
            return failed(error.code, "The sandbox request could not be confirmed. "
                          "Inspect the existing command; do not infer failure or repeat under a new key.")
        await self.base._check()
        return result
