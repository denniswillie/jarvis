from __future__ import annotations

from app.migrations.runner import read_schema_readiness
from app.runtime.feature_config import Capability, TaskLimits, artifact_worker_capabilities, get_second_brain_settings
from app.files.generated import ArtifactPolicy
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.sandbox.models import SandboxPolicy
from app.tasks.models import TaskCreate, TaskError, TaskMutation, TaskVoiceSource, json_object
from app.tasks.repository import TaskRepository
from app.tasks.context_repository import ContextRepository
from app.tasks.session_context import SessionTaskContextProvider, SessionTaskContextService
from app.tasks.memory_editing import MEMORY_EDIT_CAPABILITY, admit_memory_editing
from app.live_state.admission import require_command, require_command_create


class TaskService:
    def __init__(self, repository: TaskRepository, providers, *, registry=None, sandbox_configuration=None,
                 artifact_configuration=None, knowledge_area_resolver=None, managed_plugins=None):
        self.repository, self.providers, self.registry = repository, providers, registry
        self.sandbox_configuration = sandbox_configuration
        self.artifact_configuration = artifact_configuration
        self.knowledge_area_resolver = knowledge_area_resolver
        self.attention_service = None
        self.managed_plugins = managed_plugins
        from app.tasks.conversation import TaskConversation
        self.conversation = TaskConversation(repository)

    def provider_configured(self, backend_kind: str) -> bool:
        """Static configuration only, not a worker lease or admission guarantee."""
        return self.providers.get(backend_kind) is not None

    def execution_policy(self, environment):
        if self.sandbox_configuration is None:
            raise TaskError("sandbox_not_configured", "Sandbox execution is disabled; no task was accepted.", 503)
        policy = self.sandbox_configuration.policy
        desktop = environment in {"desktop_sandbox", "networked_desktop_sandbox"}
        if desktop and policy.template != "desktop":
            raise TaskError("sandbox_desktop_disabled", "Desktop execution is disabled; no task was accepted.", 503)
        # A deployment ceiling never upgrades the user's recorded environment.
        policy = policy.model_copy(update={"template": "desktop" if desktop else "base"})
        if environment in {"networked_sandbox", "networked_desktop_sandbox"}:
            if not policy.networked:
                raise TaskError("sandbox_network_disabled", "Network-enabled execution is disabled; no task was accepted.", 503)
            return policy
        return policy.model_copy(update={"internet_access": False, "allowed_domains": ()})

    async def create_files_data(self, owner_id: str, request: TaskCreate, *, source=None, mode=None):
        return await self.create(owner_id, request, source=source, mode=mode, managed_route="files_data")

    async def create_media(self, owner_id: str, request: TaskCreate, *, source=None, mode=None,
                           command_admission=None, media_input=None):
        return await self.create(owner_id, request, source=source, mode=mode, managed_route="video_editing",
                                 command_admission=command_admission, media_input=media_input)

    async def create(self, owner_id: str, request: TaskCreate, *, source=None, mode=None, command_admission=None,
                     managed_route=None, media_input=None):
        media_command = getattr(command_admission, "tool_name", None) == "start_media_task"
        if media_command and managed_route != "video_editing":
            raise TaskError("managed_route_required", "The video command requires its explicit managed route.", 403)
        if media_input is not None:
            if not media_command:
                raise TaskError("media_command_required", "A media input pin cannot authorize ordinary tasks.", 403)
            from app.tasks.media_editing import require_media_input
            media_input = require_media_input(command_admission, media_input)
        if managed_route is not None:
            from app.plugins.managed import require_request
            require_request(request, managed_route, command_admission)
        voice = {"source": source, "mode": mode} if source is not None else {}
        if command_admission is not None:
            require_command(command_admission, owner_id, request.request_id, "create")
            require_command_create(command_admission, request, source=source, mode=mode)
            voice = {"command_admission": command_admission, "mode": mode}
        provider = self.providers.get(request.backend_kind)
        if managed_route is not None:
            voice["managed_route"] = managed_route
        if provider is None:
            # A removed configuration must not hide already accepted work.
            replay = await self.repository.replay_create(owner_id, request, **voice)
            if replay is not None:
                return replay
            raise TaskError("task_provider_not_configured", "No execution provider is configured for this task; nothing was accepted.", 503)

        async def admission(connection):
            required = ()
            if managed_route is not None:
                from app.plugins.managed import managed_capability
                if self.registry is None:
                    raise TaskError("managed_worker_unavailable", "Managed work requires an actually ready worker.", 503)
                required += (managed_capability(managed_route),)
                if managed_route == "video_editing" and (self.artifact_configuration is None
                        or not {"json", "mp4"}.issubset(self.artifact_configuration.policy.formats)
                        or self.artifact_configuration.policy.max_files < 2):
                    raise TaskError("managed_media_publication_unavailable",
                        "Media editing requires an explicit recipe and MP4 publication allowance.", 503)
            if request.edit_memory:
                if self.knowledge_area_resolver is None or source is None:
                    raise TaskError("memory_edit_not_configured",
                        "Source-backed memory and a current conversation request are required; no task was accepted.", 503)
                required += (MEMORY_EDIT_CAPABILITY,)
            if request.execution_environment != "none":
                required += self.execution_policy(request.execution_environment).required_capabilities
            if request.publish_artifacts:
                if self.artifact_configuration is None:
                    raise TaskError("artifacts_not_configured", "Generated-file publication is disabled; no task was accepted.", 503)
                required += self.artifact_configuration.required_capabilities
            if self.registry is not None:
                extra = {"required_capabilities": required} if required else {}
                await self.registry.require(owner_id, request.backend_kind, provider.model, connection=connection, **extra)

        async def grant_factory(connection, task_id, event):
            scopes = []
            if request.execution_environment != "none":
                from app.plugins.managed import managed_grant_prefix
                scopes.append(("sandbox.execute", self.sandbox_configuration,
                               managed_grant_prefix(managed_route) if managed_route is not None else "sandbox-task:",
                               self.execution_policy(request.execution_environment)))
            if request.publish_artifacts:
                policy = self.artifact_configuration.policy
                if media_command:
                    policy = policy.model_copy(update={"formats": ("json", "mp4"), "max_files": 2})
                scopes.append(("files.publish_generated", self.artifact_configuration, "artifact-task:", policy))
            identities = []
            for capability, config, prefix, policy in scopes:
                payload = await self.repository.payload(connection, owner_id, policy.model_dump(mode="json"),
                                                         source_payload_ids=(event["payload_id"],))
                identity = await connection.fetchval("""insert into jarvis_operation_grants
                    (owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                    values($1,$2,$3,$4,$5,1,$6,clock_timestamp()+$7*interval '1 second') returning id""",
                    owner_id, capability, prefix+str(task_id), event["id"], event["revision"], payload, float(config.grant_seconds))
                identities.append(identity)
            if request.edit_memory:
                identities.append(await admit_memory_editing(self.repository, connection, owner_id, task_id, event))
            return tuple(identities)

        readiness = {"admission": admission} if self.registry is not None or request.execution_environment!="none" or request.edit_memory else {}
        if request.execution_environment != "none" or request.edit_memory:
            readiness["grant_factory"] = grant_factory
        if request.publish_artifacts:
            readiness["default_knowledge_area_id"] = (
                self.artifact_configuration.area_id if self.artifact_configuration is not None else None)
        if self.knowledge_area_resolver is not None:
            readiness["area_resolver"] = self.knowledge_area_resolver
        if managed_route is not None:
            async def managed_admission(connection):
                if self.managed_plugins is None:
                    raise TaskError("managed_runtime_unavailable", "The managed runtime is not configured.", 503)
                options = {} if managed_route == "files_data" else {"route": managed_route}
                return await self.managed_plugins.admit(connection, owner_id, **options)
            readiness["managed_admission"] = managed_admission
        if media_command:
            readiness["media_input"] = media_input
        return await self.repository.create(owner_id, request, model=provider.model, **voice, **readiness)

    async def mutate(self, owner_id, task_id, operation, request: TaskMutation, *, source=None, command_admission=None):
        voice = {"source": source} if source is not None else {}
        if command_admission is not None:
            require_command(command_admission, owner_id, request.request_id, operation)
            if source is not None:
                raise TaskError("task_command_scope_invalid", "Command and transcript admission cannot be combined.", 403)
            voice = {"command_admission": command_admission}

        async def admission(connection, task):
            provider = self.providers.get(task.backend_kind)
            if provider is None:
                raise TaskError("task_provider_not_configured", "This task's execution provider is unavailable.", 503)
            if provider.model != task.model:
                raise TaskError("task_model_configuration_changed", "This task's admitted model is not configured.", 503)
            scopes = await connection.fetch("""select g.capability,p.content as scope from jarvis_task_grants tg
                join jarvis_operation_grants g on g.owner_id=tg.owner_id and g.id=tg.grant_id
                join jarvis_context_payloads p on p.owner_id=g.owner_id and p.id=g.scope_payload_id
                where tg.owner_id=$1 and tg.task_id=$2
                    and g.capability in ('sandbox.execute','files.publish_generated','memory.correct_sources','notes.edit_account')""", owner_id, task.id)
            required = set()
            for row in scopes:
                required.add(row["capability"])
                if row["capability"] == "sandbox.execute":
                    if self.sandbox_configuration is None:
                        raise TaskError("sandbox_not_configured", "Sandbox execution is disabled; work was not resumed.", 503)
                    try:
                        if row["scope"] is None:
                            raise ValueError("Removed scope")
                        policy = SandboxPolicy.model_validate(json_object(row["scope"]))
                    except ValueError:
                        raise TaskError("sandbox_scope_unavailable", "The original execution scope is unavailable.", 403) from None
                    if not policy.within(self.sandbox_configuration.policy):
                        raise TaskError("sandbox_policy_exceeds_worker", "The original execution scope is not enabled; work was not resumed.", 503)
                    required.update(policy.required_capabilities)
                elif row["capability"] == MEMORY_EDIT_CAPABILITY:
                    if self.knowledge_area_resolver is None:
                        raise TaskError("memory_edit_not_configured", "Source-backed memory editing is disabled; work was not resumed.", 503)
                elif row["capability"] == "notes.edit_account":
                    raise TaskError("note_feature_retired", "Note-taking has been removed; this saved work cannot be resumed.", 410)
                elif self.artifact_configuration is None:
                    raise TaskError("artifacts_not_configured", "Generated-file publication is disabled; work was not resumed.", 503)
                else:
                    try:
                        policy = ArtifactPolicy.model_validate(json_object(row["scope"]))
                    except (TypeError, ValueError):
                        raise TaskError("artifact_scope_unavailable", "The original publication scope is unavailable.", 403) from None
                    if not self.artifact_configuration.policy.accepts(policy):
                        raise TaskError("artifact_policy_exceeds_worker", "The original publication scope is not enabled; work was not resumed.", 503)
                    required.update(artifact_worker_capabilities(policy))
            if self.registry is not None:
                extra = {"required_capabilities": tuple(sorted(required))} if required else {}
                await self.registry.require(owner_id, task.backend_kind, task.model, connection=connection, **extra)

        readiness = {"admission": admission} if operation == "resume" else {}
        return await self.repository.mutate(owner_id, task_id, operation, request, **voice, **readiness)


    async def cancel_with_result_suppression(self, owner_id, task_id, request, *, source):
        """Explicit conversational opt-in; never reinterpret completed work as cancelled."""
        if not isinstance(source, TaskVoiceSource):
            raise TaskError("task_request_evidence_missing", "A finalized user request is required.", 403)
        try:
            task = await self.mutate(owner_id, task_id, "cancel", request, source=source)
        except TaskError as error:
            if error.code != "task_state_conflict":
                raise
            if self.attention_service is None:
                raise TaskError("task_announcement_control_unavailable",
                    "Work was not changed and result-announcement suppression is unavailable.", 503) from None
            task, delivery = await self.attention_service.repository.dismiss_task_result(
                owner_id, task_id, request, source=source)
            work = {"status": "already_completed", "cancellation_requested": False,
                    "completed_work_cancelled": False, "external_effects": "not_undone"}
            if delivery["durable_suppression"]:
                delivery["transport_stop"] = await self.attention_service.stop_dismissed_result(owner_id, delivery)
            return task, work, delivery
        # Existing cancellation invalidates its old task revision and fences its worker.
        # It is not a separate durable dismissal of an intent that does not yet exist.
        return task, {
            "status": task.status.value, "cancellation_requested": True,
            "completed_work_cancelled": False, "external_effects": "not_undone",
        }, {
            "status": "not_established", "durable_suppression": False,
            "reason": "cancellation_receipt_only_no_separate_result_dismissal",
            "hearing": "not_established", "prior_delivery": "not_retracted",
        }

    async def provider_status(self, owner_id):
        return [{
            "backend_kind":kind,"model":provider.model,"configured":True,
            "worker_ready":await self.registry.ready(owner_id,kind,provider.model) if self.registry else False,
            "meaning":"A recent worker readiness lease, not proof of task completion or future availability.",
        } for kind,provider in self.providers.items()]


async def initialize_task_runtime(services):
    """API admission/context only. Dedicated workers own TaskRunner and Hermes."""
    flags = get_second_brain_settings()
    resolution = flags.resolve(Capability.TASKS, database_url=services.settings.SUPABASE_DB_URL)
    services.background_task_issue = resolution.issue or "tasks_disabled"
    if not resolution.enabled or not resolution.configured or services.db_pool is None:
        return
    if not isinstance(resolution.configuration, TaskLimits):
        services.background_task_issue = "task_configuration_invalid"
        return
    try:
        async with services.db_pool.acquire() as connection:
            schema = await connection.fetchval("select current_schema()")
            readiness = await read_schema_readiness(connection, schema=schema)
        if not readiness.ready:
            services.background_task_issue = "task_schema_not_ready"
            return
        providers = {}
        services.background_task_provider_issue = None
        try:
            selected = flags.frontier_selection()
            if selected is not None:
                providers["frontier"] = ConfiguredTaskProvider(selected.model)
        except ValueError:
            services.background_task_provider_issue = "frontier_configuration_invalid"
        repository = TaskRepository(services.db_pool, resolution.configuration)
        context_service = SessionTaskContextService(ContextRepository(repository))
        services.background_task_service = TaskService(
            repository,providers,registry=TaskWorkerRegistry(services.db_pool),
            sandbox_configuration=flags.resolve(Capability.SANDBOX,
                database_url=services.settings.SUPABASE_DB_URL).configuration,
            artifact_configuration=flags.resolve(Capability.ARTIFACTS,
                database_url=services.settings.SUPABASE_DB_URL).configuration,
        )
        services.background_task_providers = providers
        from app.tasks.voice_tools import BackgroundTaskToolModule
        services.tool_modules = (*getattr(services, "tool_modules", ()),
                                 BackgroundTaskToolModule(services.background_task_service))
        services.background_task_context_service = context_service
        services.assistant_context_providers = (*services.assistant_context_providers, SessionTaskContextProvider(context_service))
        services.background_task_issue = None
    except Exception:
        services.background_task_issue = "task_initialization_failed"
