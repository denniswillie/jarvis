import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from pydantic import ValidationError

from app.runtime.feature_config import SecondBrainSettings
from app.tasks.catalog import ConfiguredTaskProvider
from app.tasks.models import TaskCreate, TaskError, TaskMutation
from app.tasks.service import TaskService, initialize_task_runtime
from app.tasks.worker import WorkerContextFactory, WorkerRoleConfiguration, probe_native


class WorkerRoleTests(unittest.IsolatedAsyncioTestCase):
    def service(self):
        repository = AsyncMock()
        repository.replay_create.return_value = None
        repository.replay_mutation.return_value = None
        registry = AsyncMock()
        # Model the repository's atomic callback boundary. Real replay/transaction
        # ordering is covered by WorkerRoleDatabaseTests and voice DB fixtures.
        async def create(*args, admission=None, **kwargs):
            if repository.replay_create.return_value is not None:
                return repository.replay_create.return_value
            if admission is not None:
                await admission("fixture-transaction")
            return object()
        async def mutate(*args, admission=None, **kwargs):
            if repository.replay_mutation.return_value is not None:
                return repository.replay_mutation.return_value
            if admission is not None:
                await admission("fixture-transaction", repository.get.return_value)
            return object()
        repository.create.side_effect = create
        repository.mutate.side_effect = mutate
        return TaskService(repository,{"frontier":ConfiguredTaskProvider("fixture-model")},registry=registry)

    async def test_intake_requires_owner_and_model_scoped_worker_readiness(self):
        service = self.service()
        service.registry.require.side_effect = TaskError("task_worker_not_ready","Unavailable",503)
        request = TaskCreate(request_id=uuid4(),title="Fixture",requested_outcome="Synthetic task")
        with self.assertRaises(TaskError):
            await service.create("owner",request)
        service.registry.require.assert_awaited_once_with("owner","frontier","fixture-model", connection="fixture-transaction")
        service.repository.create.assert_awaited_once()

    async def test_accepted_request_replays_even_after_the_worker_stops(self):
        service = self.service()
        accepted = object()
        service.repository.replay_create.return_value = accepted
        self.assertIs(await service.create("owner",TaskCreate(
            request_id=uuid4(),title="Fixture",requested_outcome="Synthetic task")),accepted)
        service.registry.require.assert_not_awaited()

    async def test_mutation_replay_precedes_resume_readiness(self):
        service = self.service()
        accepted = object()
        service.repository.replay_mutation.return_value = accepted
        self.assertIs(await service.mutate("owner",uuid4(),"resume",
            TaskMutation(request_id=uuid4(),expected_revision=1)),accepted)
        service.registry.require.assert_not_awaited()

    async def test_resume_cannot_silently_change_the_admitted_model(self):
        service = self.service()
        service.repository.get.return_value = SimpleNamespace(backend_kind="frontier",model="old-model")
        with self.assertRaises(TaskError) as caught:
            await service.mutate("owner",uuid4(),"resume",TaskMutation(request_id=uuid4(),expected_revision=1))
        self.assertEqual(caught.exception.code,"task_model_configuration_changed")
        service.repository.mutate.assert_awaited_once()

    async def test_cancel_and_updates_do_not_need_a_live_worker(self):
        for operation in ("cancel","update"):
            service = self.service()
            await service.mutate("owner",uuid4(),operation,
                                 TaskMutation(request_id=uuid4(),expected_revision=1,message="Synthetic"))
            service.registry.require.assert_not_awaited()
            service.repository.mutate.assert_awaited_once()

    def test_api_model_selection_needs_no_worker_paths_or_provider_secret(self):
        flags = SecondBrainSettings(_env_file=None,enabled="true",frontier_enabled="true",
                                    hermes_python="bad relative path",worker_heartbeat_seconds="bad")
        self.assertEqual(flags.frontier_selection().model,flags.frontier_model)
        with self.assertRaises(ValidationError):
            WorkerRoleConfiguration.from_flags(flags)
        self.assertIsNone(SecondBrainSettings(_env_file=None,hermes_python="bad").frontier_selection())

    async def test_enabled_api_registers_only_metadata_not_a_runner(self):
        connection = AsyncMock()
        connection.fetchval.return_value = "fixture"
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        flags = SecondBrainSettings(_env_file=None,enabled="true",tasks_enabled="true",frontier_enabled="true",
                                    hermes_python="missing",worker_host_id="")
        services = SimpleNamespace(settings=SimpleNamespace(SUPABASE_DB_URL="configured"),
                                   db_pool=pool,assistant_context_providers=())
        with patch("app.tasks.service.get_second_brain_settings",return_value=flags), patch(
            "app.tasks.service.read_schema_readiness",AsyncMock(return_value=SimpleNamespace(ready=True)),
        ), patch("asyncio.create_subprocess_exec") as spawn, patch("asyncio.create_task") as start:
            await initialize_task_runtime(services)
        self.assertIsNone(services.background_task_issue)
        self.assertEqual(services.background_task_providers["frontier"].model,flags.frontier_model)
        self.assertFalse(hasattr(services.background_task_providers["frontier"],"execute"))
        self.assertFalse(hasattr(services,"background_task_runner"))
        spawn.assert_not_called()
        start.assert_not_called()

    def test_worker_requires_absolute_paths_stable_identity_and_bounded_heartbeat(self):
        for changes in ({"python":"relative"},{"work_root":"relative"},{"host_id":""},
                        {"host_id":" unstable "},{"heartbeat_seconds":float("inf")}):
            with self.subTest(changes=changes),self.assertRaises(ValidationError):
                WorkerRoleConfiguration(**({"python":"/bin/python","work_root":"/tmp/private","host_id":"host-a"}|changes))

    async def test_no_worker_advertises_retired_note_tools(self):
        from readonly_live_store import legacy_worker_context
        for session_id in (None,uuid4()):
            prepare = AsyncMock(return_value="prepared")
            pool, context = await legacy_worker_context(session_id)
            with patch("app.tasks.worker.HermesContextFactory",return_value=prepare) as factory:
                self.assertEqual(await WorkerContextFactory(pool)(context),"prepared")
            self.assertNotIn("search_notes", factory.call_args.kwargs["read_tools"])
            self.assertNotIn("notes.create", factory.call_args.kwargs["capability_tools"])
