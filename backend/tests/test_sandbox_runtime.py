import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.runtime.feature_config import Capability, SandboxConfiguration, SecondBrainSettings
from app.sandbox.runtime import load_sandbox_runtime
from app.tasks.models import TaskError
from app.tasks.worker import TaskWorkerRole, WorkerRoleConfiguration


class SandboxRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def repository(self, *, outstanding=False):
        connection = AsyncMock()
        connection.fetchval.return_value = outstanding
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        return SimpleNamespace(pool=pool)

    def flags(self, enabled=True, **changes):
        return SecondBrainSettings(_env_file=None, enabled="true", sandbox_enabled=str(enabled).lower(), **changes)

    def test_public_configuration_contains_bounds_not_credentials(self):
        resolution = self.flags().resolve(Capability.SANDBOX, database_url="configured")
        self.assertTrue(resolution.configured)
        self.assertEqual(resolution.configuration.policy.allowed_domains, ())
        self.assertNotIn("api_key", resolution.configuration.model_dump_json())
        self.assertFalse(self.flags(sandbox_lifetime_seconds="0").resolve(
            Capability.SANDBOX, database_url="configured").configured)

    def test_network_configuration_is_explicit_and_invalid_value_fails_closed(self):
        offline = self.flags().resolve(Capability.SANDBOX, database_url="configured")
        enabled = self.flags(sandbox_internet_enabled="true").resolve(Capability.SANDBOX, database_url="configured")
        invalid = self.flags(sandbox_internet_enabled="maybe").resolve(Capability.SANDBOX, database_url="configured")
        self.assertFalse(offline.configuration.policy.internet_access)
        self.assertTrue(enabled.configuration.policy.internet_access)
        self.assertFalse(invalid.configured)

    async def test_network_runtime_advertises_both_required_capabilities(self):
        with TemporaryDirectory() as directory:
            key = Path(directory)/"key"
            key.write_text("synthetic-key")
            provider = SimpleNamespace(discover=AsyncMock(return_value=[]),
                                       configuration=SimpleNamespace(request_timeout_seconds=20))
            runtime = await load_sandbox_runtime(self.flags(sandbox_internet_enabled="true"),
                self.repository(), "owner", database_url="configured", key_file=key,
                provider_factory=lambda _:provider)
            self.assertEqual(runtime.capabilities, ("sandbox.execute", "sandbox.network"))

    async def test_unused_worker_does_not_import_or_construct_sdk(self):
        factory = MagicMock()
        result = await load_sandbox_runtime(self.flags(False), self.repository(), "owner",
                                            database_url="configured", provider_factory=factory)
        self.assertIsNone(result)
        factory.assert_not_called()

    async def test_selected_or_outstanding_resources_require_cleanup_credential(self):
        for enabled, outstanding in ((True, False), (False, True)):
            with self.subTest(enabled=enabled), self.assertRaises(TaskError) as caught:
                await load_sandbox_runtime(self.flags(enabled), self.repository(outstanding=outstanding),
                                           "owner", database_url="configured")
            self.assertEqual(caught.exception.code, "sandbox_cleanup_credential_required")

    async def test_disabled_execution_retains_authenticated_cleanup_without_vm_creation(self):
        with TemporaryDirectory() as directory:
            key = Path(directory)/"key"
            key.write_text("synthetic-key")
            provider = SimpleNamespace(discover=AsyncMock(return_value=[]), create=AsyncMock(),
                                       configuration=SimpleNamespace(request_timeout_seconds=20))
            factory = MagicMock(return_value=provider)
            runtime = await load_sandbox_runtime(self.flags(False), self.repository(outstanding=True), "owner",
                database_url="configured", key_file=key, provider_factory=factory)
            self.assertEqual(runtime.capabilities, ())
            self.assertFalse(runtime.enabled)
            provider.discover.assert_awaited_once()
            provider.create.assert_not_awaited()
            self.assertNotIn("synthetic-key", repr(factory.call_args.args[0]))

    async def test_enabled_runtime_advertises_computation_only(self):
        with TemporaryDirectory() as directory:
            key = Path(directory)/"key"
            key.write_text("synthetic-key")
            provider = SimpleNamespace(discover=AsyncMock(return_value=[]),
                                       configuration=SimpleNamespace(request_timeout_seconds=20))
            runtime = await load_sandbox_runtime(self.flags(), self.repository(), "owner",
                database_url="configured", key_file=key, provider_factory=lambda _:provider)
            self.assertEqual(runtime.capabilities, ("sandbox.execute",))

    async def test_worker_supervises_cleaner_and_withdraws_on_cleaner_failure(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)/"worker"
            runner_stop = asyncio.Event()
            runner = SimpleNamespace(run=AsyncMock(side_effect=runner_stop.wait),
                stop=AsyncMock(side_effect=lambda:runner_stop.set()))
            runtime = SimpleNamespace(capabilities=("sandbox.execute",),
                run=AsyncMock(side_effect=RuntimeError("synthetic cleaner failure")), final_cleanup=AsyncMock())
            repository = SimpleNamespace(pool=object(), configure_scheduler_limits=AsyncMock())
            with patch("app.tasks.worker.TaskRunner", return_value=runner):
                role = TaskWorkerRole(repository, {}, owner_id="owner",
                    configuration=WorkerRoleConfiguration(python="/bin/python", work_root=root, host_id="fixture"),
                    sandbox_runtime=runtime)
            role.registry = SimpleNamespace(advertise=AsyncMock(), withdraw=AsyncMock())
            with self.assertRaises(RuntimeError):
                await role.run(asyncio.Event())
            role.registry.withdraw.assert_awaited_once()
            runtime.final_cleanup.assert_awaited_once()
            runner.stop.assert_awaited()
