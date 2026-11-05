"""Dedicated frontier worker. No API, voice session or generic agent loop runs here."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import tempfile
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.db import create_pool
from app.files.storage import JarvisFileStorageClient
from app.files.tools import FileToolModule
from app.files.transfer import FileTransferPolicy, MP4_OUTPUT_POLICY
from app.knowledge.files import SourceFileRepository
from app.tasks.memory_evidence import TaskMemoryEvidence, TaskSourceMemoryReadBackend
from app.tasks.memory_writes import MEMORY_WRITE_CAPABILITY, TaskMemoryWriteBackend
from app.tasks.memory_editing import MEMORY_EDIT_CAPABILITY, TaskMemoryEditingBridge
from app.tasks.memory_corrections import (
    MEMORY_CORRECTION_CAPABILITY, TaskMemoryCorrectionBackend, TaskMemoryCorrectionToolModule,
)
from app.knowledge.runtime import KnowledgeRuntime
from app.knowledge.measurements import MeasurementRepository, MeasurementToolModule
from app.memory.tools import MemoryToolModule
from app.migrations.runner import read_schema_readiness
from app.runtime.feature_config import Capability, SecondBrainSettings
from app.settings import JarvisSettings, is_configured_secret
from app.storage.supabase_config import get_supabase_storage_config
from app.visual_memory.storage import VisualFrameStorageClient
from app.tasks.stored_images import StoredTaskImageFactory
from app.tasks.catalog import TaskWorkerRegistry
from app.tasks.hermes import HermesContextFactory, HermesRuntimeConfiguration, HermesTaskProvider, ensure_private_runtime_root
from app.tasks.model_gateway import OpenAIResponsesHTTP
from app.tasks.models import TaskError
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner


LOGGER = logging.getLogger(__name__)

# Only code identities, never exception text, locals or caller-supplied paths.
# The supervisor independently validates this small wire allowlist.
_STARTUP_FRAME_SOURCES = {
    "tasks/worker.py": ("worker", frozenset({"main", "run_command", "worker_settings", "probe_native",
        "worker_health", "__init__", "from_flags", "native", "absolute_path", "stable_host", "run"})),
    "knowledge/runtime.py": ("knowledge_runtime", frozenset({"create", "__init__", "start", "close", "set_file_publications"})),
    "knowledge/hindsight.py": ("knowledge_client", frozenset({"__init__"})),
    "knowledge/outbox.py": ("knowledge_outbox", frozenset({"__init__"})),
    "knowledge/ingestion.py": ("knowledge_ingestion", frozenset({"__init__"})),
    "sandbox/runtime.py": ("sandbox_runtime", frozenset({"load_sandbox_runtime"})),
    "tasks/hermes.py": ("hermes_initialization", frozenset({"__init__", "ensure_private_runtime_root"})),
    "tasks/model_gateway.py": ("model_gateway", frozenset({"__init__"})),
    "storage/supabase_config.py": ("storage_configuration", frozenset({"get_supabase_storage_config"})),
}
_STARTUP_FRAME_PATHS = {
    str(Path(__file__).parent.parent / relative): (stage, Path(relative).name, functions)
    for relative, (stage, functions) in _STARTUP_FRAME_SOURCES.items()
}
_STARTUP_FAILURE_TYPES = frozenset({"TaskError", "ValueError", "ValidationError"})
_STARTUP_FAILURE_CODES = frozenset({
    "task_worker_startup_failed", "task_worker_not_configured", "task_schema_not_ready",
    "task_worker_root_in_use", "task_worker_stopped", "sandbox_cleanup_worker_stopped",
    "artifact_configuration_invalid", "artifact_media_not_ready", "sandbox_configuration_invalid",
    "task_model_not_accessible", "task_memory_not_configured", "artifact_worker_unavailable",
})


def startup_failure_receipt(error):
    """Bounded caught-exception location; not a root-cause or readiness verdict.

    Inspect at most 64 traceback frames and retain only the deepest allowlisted
    one encountered. Do not follow causes/contexts, inspect locals, fetch source
    lines or stringify the exception. An unlisted origin stays explicitly unknown.
    """
    diagnostic = {"version": 1, "stage": "unavailable", "frame": None,
                  "basis": "unavailable", "truncated": False}
    traceback = error.__traceback__
    selected = None
    for _ in range(64):
        if traceback is None:
            break
        code = traceback.tb_frame.f_code
        allowed = _STARTUP_FRAME_PATHS.get(code.co_filename)
        if allowed is not None and code.co_name in allowed[2] and 1 <= traceback.tb_lineno <= 1_000_000:
            diagnostic.update(stage=allowed[0], frame={"source": allowed[1],
                "function": code.co_name, "line": traceback.tb_lineno}, basis="filtered_caller")
            selected = traceback
        terminal = traceback
        traceback = traceback.tb_next
    diagnostic["truncated"] = traceback is not None
    if selected is not None and traceback is None and selected is terminal:
        diagnostic["basis"] = "exception_origin"
    try:
        error_code = getattr(error, "code", "task_worker_startup_failed")
    except Exception:
        error_code = None
    error_type = type(error).__name__
    return {"ready": False,
            "error_type": error_type if error_type in _STARTUP_FAILURE_TYPES else "unrecognized_error_type",
            "code": error_code if type(error_code) is str and error_code in _STARTUP_FAILURE_CODES else "unrecognized_code",
            "diagnostic": diagnostic}


class WorkerRoleConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True,extra="forbid")
    python: Path
    work_root: Path
    host_id: str = Field(min_length=1,max_length=200)
    heartbeat_seconds: float = Field(default=5,ge=1,le=60,allow_inf_nan=False)

    @field_validator("python","work_root")
    @classmethod
    def absolute_path(cls, value):
        if not value.is_absolute():
            raise ValueError("Worker paths must be explicit and absolute")
        return value

    @field_validator("host_id")
    @classmethod
    def stable_host(cls, value):
        if value != value.strip() or "\x00" in value:
            raise ValueError("A stable operator-owned host identity is required")
        return value

    @classmethod
    def from_flags(cls, flags):
        return cls(python=flags.hermes_python,work_root=flags.hermes_work_root,
                   host_id=flags.worker_host_id,heartbeat_seconds=flags.worker_heartbeat_seconds)

    def native(self):
        return HermesRuntimeConfiguration(self.python,self.work_root,host_id=self.host_id)


class WorkerContextFactory:
    """Selected source-memory reads, never an implicit legacy memory fallback."""
    def __init__(self, pool, *, knowledge=None, images_factory=None, file_storage=None, sandbox_provider=None,
                 sandbox_policy=None, artifact_configuration=None, media_policy=None, media_inspector=None,
                 managed_plugins=None):
        self.pool, self.knowledge, self.images_factory = pool, knowledge, images_factory
        self.file_storage = file_storage
        self.sandbox_provider = sandbox_provider
        self.sandbox_policy = sandbox_policy
        self.artifact_configuration = artifact_configuration
        self.media_policy, self.media_inspector = media_policy, media_inspector
        self.managed_plugins = managed_plugins
        if managed_plugins is not None and sandbox_provider is None:
            raise ValueError("Managed computation requires the existing sandbox provider")
        if artifact_configuration is not None and (knowledge is None or file_storage is None or sandbox_provider is None):
            raise ValueError("Artifact tools require source-backed files, storage and an enabled sandbox provider")
        media_selected = (artifact_configuration is not None
                          and artifact_configuration.policy.supports_media)
        if media_selected:
            if (not isinstance(media_policy, FileTransferPolicy) or media_policy.policy_id != MP4_OUTPUT_POLICY
                    or not callable(media_inspector)):
                raise ValueError("A media worker requires its trusted output policy and inspector")
        elif media_policy is not None or media_inspector is not None:
            raise ValueError("Media dependencies require an explicitly selected MP4 artifact allowance")
        if managed_plugins is not None and "plugins.video_editor" in managed_plugins.capabilities:
            if (not media_selected or "json" not in artifact_configuration.policy.formats
                    or not callable(getattr(sandbox_provider, "materialize_media_render", None))
                    or not callable(getattr(sandbox_provider, "decode_media_render_input", None))
                    or not callable(getattr(media_inspector, "inspect_retained_input", None))):
                raise ValueError("Managed rendering requires input inspection, output inspection, decoder, publisher and materialization provider")

    @property
    def capabilities(self):
        from app.tasks.readonly import CAPABILITY as READONLY_CAPABILITY
        return ((READONLY_CAPABILITY,) + ((MEMORY_EDIT_CAPABILITY,) if self.knowledge is not None else ())
                + (self.managed_plugins.capabilities if self.managed_plugins is not None else ()))

    async def conversation_for(self, context):
        # Existing specialized owners retain their completion hooks. New generic
        # Live work starts with an explicit mailbox message before it is claimed.
        from app.tasks.readonly_worker import is_readonly
        from app.tasks.conversation import INPUT, TaskConversation
        if await is_readonly(context):
            return None
        async with self.pool.acquire() as connection:
            present = await connection.fetchval("""select exists(select 1 from jarvis_task_events
                where owner_id=$1 and task_id=$2 and event_type=$3)""",
                context.task.owner_id, context.task.id, INPUT)
        return TaskConversation(context.tasks) if present else None

    async def __call__(self, context):
        from app.tasks.readonly_worker import is_readonly, prepare_readonly
        if await is_readonly(context):
            return await prepare_readonly(context, self)
        modules, readers, capabilities = [], [], {}
        evidence = writer = corrector = files = None
        memory = {"status":"unavailable","meaning":"No selected long-term memory service is configured."}
        if self.knowledge is not None:
            self.knowledge.context_for(context.task.owner_id)
            area = context.task.knowledge_area_id or self.knowledge.configuration.area_id
            evidence = TaskMemoryEvidence(
                context,self.knowledge.sources,area_id=area,access_check=self.knowledge.context_for,
            )
            backend = TaskSourceMemoryReadBackend(
                self.knowledge.sources,self.knowledge.retrieval,owner_id=context.task.owner_id,area_id=area,
                access_check=self.knowledge.context_for,task_evidence=evidence,
            )
            writer = TaskMemoryWriteBackend(evidence)
            modules.append(MemoryToolModule(read_backend=backend,write_backend=writer))
            corrector = TaskMemoryCorrectionBackend(evidence)
            modules.append(TaskMemoryCorrectionToolModule(corrector))
            capabilities[MEMORY_CORRECTION_CAPABILITY] = "correct_memory_source"
            if getattr(self.knowledge, "projects", None) is not None:
                from app.projects.tools import ProjectToolModule
                modules.append(ProjectToolModule(self.knowledge.projects, context.task.owner_id, area,
                    access_check=self.knowledge.context_for, task_evidence=evidence))
                readers.extend(("list_project_roots", "inspect_project", "get_project_result"))
            capabilities[MEMORY_WRITE_CAPABILITY] = "remember"
            modules.append(MeasurementToolModule(MeasurementRepository(
                self.knowledge.sources, context.tasks, owner_id=context.task.owner_id, area_id=area,
                access_check=self.knowledge.context_for, task_evidence=evidence)))
            readers.extend(("list_measurements", "calculate_measurements"))
            readers.extend(("search_memory","explain_memory","list_memory_versions"))
            memory = {"status":"configured","area_id":str(area),
                      "meaning":"Use source-backed tools; configuration is not proof that recall is available."}
        if context.task.session_id is not None:
            if self.knowledge is not None and self.file_storage is not None:
                files = SourceFileRepository(self.knowledge.sources, self.file_storage,
                    owner_id=context.task.owner_id, area_id=area,
                    access_check=self.knowledge.context_for, task_evidence=evidence,
                    media_policy=self.media_policy, media_inspector=self.media_inspector)
                modules.append(FileToolModule(files, self.file_storage))
                readers.extend(("list_files", "get_file_metadata"))
                capabilities["files.create_text"] = "create_text_file"
        async def revalidate_receipt(action, result):
            if writer is not None:
                result = await writer.revalidate_receipt(action, result)
            if corrector is not None:
                result = await corrector.revalidate_receipt(action, result)
            if files is not None:
                result = await files.revalidate_receipt(action, result)
            return result
        async def shared(_):
            return {"long_term_memory":memory,"images":{
                "resolver_configured":self.images_factory is not None,
                "meaning":"Only actually resolved original pixels support visual claims.",
            }}
        async def sandbox_bridge(bridge):
            bridge = await TaskMemoryEditingBridge.attach(bridge, corrector)
            from app.tasks.live_tools import LiveWorkToolBridge
            bridge = await LiveWorkToolBridge.attach(bridge)
            if self.sandbox_provider is None:
                return bridge
            from app.sandbox.tools import SandboxToolBridge
            return await SandboxToolBridge.attach(bridge, self.sandbox_provider, policy_limit=self.sandbox_policy,
                                                 files=files, artifact_configuration=self.artifact_configuration,
                                                 managed_plugins=self.managed_plugins)
        return await HermesContextFactory(
            modules=modules,read_tools=readers,capability_tools=capabilities,
            images_factory=self.images_factory,shared_context_factory=shared,memory_evidence=evidence,
            receipt_revalidator=revalidate_receipt if self.knowledge is not None else None,
            bridge_factory=sandbox_bridge,
        )(context)


class TaskWorkerRole:
    def __init__(self, repository, providers, *, owner_id, configuration, worker_id=None, sandbox_runtime=None,
                 extra_capabilities=(), readiness_refresh=None):
        extra_capabilities = tuple(extra_capabilities) + tuple(
            capability for provider in providers.values()
            for capability in getattr(getattr(provider, "prepare", None), "capabilities", ()))
        self.repository, self.providers = repository, providers
        self.owner_id, self.configuration = owner_id, configuration
        self.sandbox_runtime = sandbox_runtime
        self.readiness_refresh = readiness_refresh
        self.capabilities = tuple(dict.fromkeys((sandbox_runtime.capabilities if sandbox_runtime is not None else ())
                                               + tuple(extra_capabilities)))
        self.worker_id = worker_id or "frontier-"+uuid4().hex
        self.registry = TaskWorkerRegistry(repository.pool)
        self.runner = TaskRunner(repository,providers,worker_id=self.worker_id,
                                 owner_id=owner_id,host_id=configuration.host_id,capabilities=self.capabilities)

    async def run(self, stop):
        root = ensure_private_runtime_root(self.configuration.work_root)
        descriptor = os.open(root/"worker.lock",os.O_CREAT|os.O_RDWR,0o600)
        acquired = False
        runner_task = stopper = None
        cleaner_task = None
        readiness_task = None
        cleanup_stop = asyncio.Event()
        readiness_stop = asyncio.Event()
        marker = root/"worker.json"
        try:
            try:
                fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise TaskError("task_worker_root_in_use","Another worker owns this private runtime root.",503) from error
            acquired = True
            await self.repository.configure_scheduler_limits()
            marker.write_text(json.dumps({
                "format":"jarvis.task_worker.v1","worker_id":self.worker_id,
                "owner_digest":hashlib.sha256(self.owner_id.encode()).hexdigest(),
                "host_id":self.configuration.host_id,"pid":os.getpid(),
            }))
            marker.chmod(0o600)
            if self.readiness_refresh is not None:
                readiness_task = asyncio.create_task(self.readiness_refresh(readiness_stop),
                                                     name="managed-media-readiness")
            runner_task = asyncio.create_task(self.runner.run())
            if self.sandbox_runtime is not None:
                cleaner_task = asyncio.create_task(self.sandbox_runtime.run(cleanup_stop, self.worker_id+"-cleanup"))
            stopper = asyncio.create_task(stop.wait())
            while not stop.is_set():
                if runner_task.done():
                    await runner_task
                    raise TaskError("task_worker_stopped","The task runner stopped unexpectedly.",503)
                if cleaner_task is not None and cleaner_task.done():
                    await cleaner_task
                    raise TaskError("sandbox_cleanup_worker_stopped", "The resource cleanup worker stopped unexpectedly.", 503)
                await self.registry.advertise(
                    worker_id=self.worker_id,owner_id=self.owner_id,host_id=self.configuration.host_id,
                    providers=self.providers,ttl_seconds=self.configuration.heartbeat_seconds*3,
                    capabilities=self.capabilities,
                )
                watched = (runner_task, stopper) + ((cleaner_task,) if cleaner_task is not None else ())
                await asyncio.wait(watched,timeout=self.configuration.heartbeat_seconds,
                                   return_when=asyncio.FIRST_COMPLETED)
        finally:
            if acquired:
                # Stop the sole refresh child before potentially slow runner/SQL
                # cleanup. It owns no VM and must not outlive the worker lock.
                readiness_stop.set()
                if readiness_task is not None:
                    readiness_task.cancel()
                    await asyncio.gather(readiness_task, return_exceptions=True)
                with suppress(Exception):
                    await self.registry.withdraw(self.owner_id,self.worker_id)
                await self.runner.stop()
                if runner_task:
                    runner_task.cancel()
                    await asyncio.gather(runner_task,return_exceptions=True)
                cleanup_stop.set()
                if cleaner_task is not None:
                    cleaner_task.cancel()
                    await asyncio.gather(cleaner_task, return_exceptions=True)
                    try:
                        await self.sandbox_runtime.final_cleanup(self.worker_id+"-shutdown")
                    except Exception as error:
                        LOGGER.warning("Sandbox shutdown cleanup remains pending (%s)", type(error).__name__)
                marker.unlink(missing_ok=True)
            if stopper:
                stopper.cancel()
                await asyncio.gather(stopper,return_exceptions=True)
            os.close(descriptor)


async def probe_native(configuration):
    root = ensure_private_runtime_root(configuration.work_root)
    with tempfile.TemporaryDirectory(prefix="probe-",dir=root) as home:
        child = await asyncio.create_subprocess_exec(
            str(configuration.python),str(Path(__file__).with_name("hermes_child.py")),"--probe",
            cwd=home,env={"PATH":"/usr/bin:/bin","HOME":home,"HERMES_HOME":home,"TMPDIR":home,
                          "PYTHONNOUSERSITE":"1","PYTHONUNBUFFERED":"1"},
            stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL,start_new_session=True,
        )
        try:
            output,_ = await asyncio.wait_for(child.communicate(),20)
            value = json.loads(output)
            if child.returncode != 0 or value.get("ready") is not True:
                raise ValueError("Invalid native readiness receipt")
        except BaseException:
            if child.returncode is None:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid,signal.SIGKILL)
                await child.wait()
            raise


async def worker_health(pool, owner_id, configuration, *, required_capabilities=()):
    try:
        marker = json.loads((configuration.work_root/"worker.json").read_text())
        if (marker.get("format")!="jarvis.task_worker.v1" or marker.get("host_id")!=configuration.host_id
                or marker.get("owner_digest")!=hashlib.sha256(owner_id.encode()).hexdigest()):
            return False
        async with pool.acquire() as connection:
            return bool(await connection.fetchval("""
                select exists(select 1 from jarvis_task_workers where owner_id=$1 and worker_id=$2
                    and host_id=$3 and not draining and expires_at>clock_timestamp()
                    and capabilities @> $4::text[])
            """,owner_id,marker["worker_id"],configuration.host_id,list(required_capabilities)))
    except (OSError,ValueError,KeyError):
        return False


def worker_settings():
    values = {}
    # Only the trusted worker receives these mounted secrets. The Hermes child
    # receives a different, per-attempt loopback credential.
    for environment,setting in (
        ("JARVIS_FRONTIER_DB_URL_FILE","SUPABASE_DB_URL"),
        ("JARVIS_FRONTIER_OPENAI_API_KEY_FILE","OPENAI_API_KEY"),
        ("JARVIS_FRONTIER_STORAGE_KEY_FILE","SUPABASE_SERVICE_ROLE_KEY"),
    ):
        path = os.environ.get(environment)
        if path:
            value = Path(path).read_text().strip()
            if not value:
                raise ValueError("Empty worker secret")
            values[setting] = value
    return JarvisSettings(**values)


async def run_command(command, *, provider_transform=None):
    """Compose the ordinary worker, optionally decorating its provider instance.

    The CLI never supplies a transform. Isolated evaluations can decorate the
    real executor without replacing claim, heartbeat, cancellation or settlement.
    """
    settings, flags = worker_settings(), SecondBrainSettings()
    configuration = WorkerRoleConfiguration.from_flags(flags)
    tasks = flags.resolve(Capability.TASKS,database_url=settings.SUPABASE_DB_URL)
    frontier = flags.resolve(Capability.FRONTIER,database_url=settings.SUPABASE_DB_URL,
                             provider_key=settings.OPENAI_API_KEY)
    if not tasks.configured or not frontier.configured:
        raise TaskError("task_worker_not_configured","Enable and configure tasks and frontier explicitly.",503)
    pool = await create_pool(settings)
    knowledge = storage = file_storage = None
    try:
        async with pool.acquire() as connection:
            schema = await connection.fetchval("select current_schema()")
            if not (await read_schema_readiness(connection,schema=schema)).ready:
                raise TaskError("task_schema_not_ready","Apply reviewed schema migrations first.",503)
        artifact_selection = flags.resolve(Capability.ARTIFACTS, database_url=settings.SUPABASE_DB_URL)
        if artifact_selection.enabled and not artifact_selection.configured:
            raise TaskError("artifact_configuration_invalid", "Selected generated-file publication is unavailable.", 503)
        artifact_configuration = artifact_selection.configuration if artifact_selection.configured else None
        from app.plugins.authority import PluginAuthority
        from app.plugins.deployment import CURATED_DEPLOYMENT_BASELINE_V1
        from app.plugins.media_deployment import load_managed_media
        from app.plugins.repository import PluginPolicyRepository
        from app.tasks.context_repository import ContextRepository
        repository = TaskRepository(pool,tasks.configuration)
        plugin_authority = PluginAuthority(PluginPolicyRepository(pool), ContextRepository(repository),
            repository, deployment_baseline=CURATED_DEPLOYMENT_BASELINE_V1)
        sandbox_selection = flags.resolve(Capability.SANDBOX, database_url=settings.SUPABASE_DB_URL)
        managed_plugins = load_managed_media(plugin_authority, owner_id=settings.JARVIS_OWNER_ID,
            sandbox_configuration=sandbox_selection.configuration if sandbox_selection.configured else None,
            artifact_configuration=artifact_configuration)
        if command=="check":
            sandbox_selection = flags.resolve(Capability.SANDBOX, database_url=settings.SUPABASE_DB_URL)
            if sandbox_selection.enabled and not sandbox_selection.configured:
                raise TaskError("sandbox_configuration_invalid", "Selected sandbox policy is unavailable.", 503)
            ready = await worker_health(pool,settings.JARVIS_OWNER_ID,configuration,
                required_capabilities=(sandbox_selection.configuration.policy.required_capabilities
                                       if sandbox_selection.enabled else ())
                    + (artifact_configuration.required_capabilities if artifact_configuration else ())
                    + (managed_plugins.capabilities if managed_plugins else ()))
            print(json.dumps({"ready":ready}))
            return 0 if ready else 1
        from app.files.media_runtime import prepare_artifact_media
        try:
            media_policy, media_inspector = await prepare_artifact_media(artifact_configuration)
        except ValueError:
            raise TaskError("artifact_media_not_ready", "The selected media inspector is not ready.", 503) from None
        await probe_native(configuration)
        upstream = OpenAIResponsesHTTP(frontier.configuration.api_key.get_secret_value())
        try:
            response = await upstream.client.get("models/"+frontier.configuration.model)
            if response.status_code != 200:
                raise TaskError("task_model_not_accessible","The configured frontier model is not accessible.",503)
        finally:
            await upstream.aclose()
        from app.sandbox.runtime import load_sandbox_runtime
        sandbox_runtime = await load_sandbox_runtime(flags, repository, settings.JARVIS_OWNER_ID,
            database_url=settings.SUPABASE_DB_URL, key_file=os.environ.get("JARVIS_FRONTIER_E2B_API_KEY_FILE"))
        memory = flags.resolve(Capability.KNOWLEDGE,database_url=settings.SUPABASE_DB_URL)
        if memory.enabled:
            if not memory.configured:
                raise TaskError("task_memory_not_configured","Selected memory configuration is unavailable.",503)
            knowledge = await KnowledgeRuntime.create(pool=pool,owner_id=settings.JARVIS_OWNER_ID,
                                                       configuration=memory.configuration,tasks=repository)
            knowledge.start()
        images_factory = None
        if is_configured_secret(settings.SUPABASE_URL) and is_configured_secret(settings.SUPABASE_SERVICE_ROLE_KEY):
            storage = VisualFrameStorageClient(get_supabase_storage_config(settings))
            images_factory = StoredTaskImageFactory(pool,storage,settings)
            if knowledge is not None:
                file_storage = JarvisFileStorageClient(get_supabase_storage_config(settings))
                from app.tasks.file_recovery import TaskFileRecovery
                knowledge.set_file_publications(TaskFileRecovery(
                    knowledge.sources, repository, file_storage, owner_id=settings.JARVIS_OWNER_ID,
                    area_id=knowledge.configuration.area_id, access_check=knowledge.context_for,
                    media_policy=media_policy, media_inspector=media_inspector))
        if artifact_configuration is not None:
            if knowledge is None or file_storage is None or sandbox_runtime is None or not sandbox_runtime.enabled:
                raise TaskError("artifact_worker_unavailable", "Artifact publication requires selected memory, storage and execution.", 503)
        provider = HermesTaskProvider(frontier.configuration,configuration.native(),
                                       prepare=WorkerContextFactory(pool,knowledge=knowledge,
                                           images_factory=images_factory,file_storage=file_storage,
                                           sandbox_provider=sandbox_runtime.provider if sandbox_runtime and sandbox_runtime.enabled else None,
                                           sandbox_policy=sandbox_runtime.configuration.policy if sandbox_runtime else None,
                                           artifact_configuration=artifact_configuration,
                                           media_policy=media_policy, media_inspector=media_inspector,
                                           managed_plugins=managed_plugins))
        if provider_transform is not None:
            provider = provider_transform(provider)
        from app.plugins.media_probe import worker_media_refresh
        role = TaskWorkerRole(repository,{"frontier":provider},owner_id=settings.JARVIS_OWNER_ID,
                              configuration=configuration,sandbox_runtime=sandbox_runtime,
                              extra_capabilities=artifact_configuration.required_capabilities if artifact_configuration else (),
                              readiness_refresh=worker_media_refresh(managed_plugins, sandbox_runtime))
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM,signal.SIGINT):
            loop.add_signal_handler(signum,stop.set)
        try:
            await role.run(stop)
        finally:
            for signum in (signal.SIGTERM,signal.SIGINT):
                loop.remove_signal_handler(signum)
        return 0
    finally:
        if knowledge:
            await knowledge.close()
        if storage:
            await storage.close()
        if file_storage:
            await file_storage.close()
        await pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("run","check"))
    args = parser.parse_args()
    try:
        return asyncio.run(run_command(args.command))
    except Exception as error:
        print(json.dumps(startup_failure_receipt(error)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
