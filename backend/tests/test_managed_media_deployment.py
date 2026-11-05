"""Offline production composition: all provider/SQL/native boundaries are doubles."""
from contextlib import ExitStack, asynccontextmanager
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.files.generated import ArtifactPolicy
from app.files.transfer import FileTransferPolicy, MP4_OUTPUT_POLICY
from app.plugins.authority import PluginAuthority
from app.plugins.deployment import CURATED_DEPLOYMENT_BASELINE_V1
from app.plugins.media_deployment import DEPLOYMENT_ENV, load_managed_media
from app.plugins.repository import PluginPolicyRepository
from app.plugins.runtime import initialize_plugin_runtime
from app.runtime.feature_config import ArtifactConfiguration, Capability, SandboxConfiguration, TaskLimits
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from app.tasks import worker


class MediaDeploymentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = datetime.now(UTC)
        self.owner = "offline-composition-owner"
        self.connection = SimpleNamespace(fetchval=AsyncMock(return_value="offline_fixture"))

        @asynccontextmanager
        async def acquire():
            yield self.connection

        self.pool = SimpleNamespace(acquire=Mock(side_effect=acquire), close=AsyncMock())
        self.tasks = TaskRepository(self.pool, TaskLimits())
        self.contexts = ContextRepository(self.tasks)
        self.authority = PluginAuthority(PluginPolicyRepository(self.pool), self.contexts, self.tasks,
            deployment_baseline=CURATED_DEPLOYMENT_BASELINE_V1)
        self.sandbox = SandboxConfiguration()
        self.artifacts = ArtifactConfiguration(area_id=uuid4(), policy=ArtifactPolicy(
            formats=("json", "mp4"), mp4_max_file_bytes=20_000_000))
        configuration_id = next(str(e.configuration_id) for e in
            CURATED_DEPLOYMENT_BASELINE_V1.policy.entries if e.plugin_id == "video_editor")
        build_path = self.root / "build.json"
        self.write(build_path, {"template_id": "offline-image", "build_id": "offline-build",
                               "name": "offline-name", "status": "ready"})
        build_sha = hashlib.sha256(build_path.read_bytes()).hexdigest()
        renderer = {
            "template": {"id": "jarvis.clip.v1", "version": "offline-v1", "bundle_sha256": "a" * 64},
            "physical_template_id": "offline-image", "build_id": "offline-build",
            "software_sha256": "a" * 64, "build_receipt_sha256": build_sha,
            "policy": {"max_source_bytes": 20_000_000, "max_source_duration_ms": 60_000,
                       "max_output_frames": 720, "max_recipe_bytes": 65536},
            "max_output_bytes": 20_000_000,
        }
        self.proof = {
            "format": "jarvis.managed-media.qualification.v1", "renderer": renderer,
            "qualification": {"passed": True},
            "input_decodes": [{"software_sha256": "a" * 64, "video_decoded": True,
                               "audio_streams": 1, "audio_decoded": True}],
            "fixed_command_receipt": {"result": {"started": True, "exit_code": 0,
                "timed_out": False, "runner_error": None}},
            "accepted_output": {"file": {"content_sha256": "c" * 64}, "size_bytes": 100},
            "independent_output_inspection": {"full_video_audio_decode": True,
                "content_sha256": "c" * 64, "size_bytes": 100},
        }
        self.proof_path = self.root / "qualification.json"
        self.write(self.proof_path, self.proof)
        observation = {"plugin_id": "video_editor", "configuration_id": configuration_id,
            "observed_at": self.now.isoformat(), "expires_at": (self.now + timedelta(seconds=300)).isoformat(),
            "bundle_sha256": "a" * 64}
        self.observations = {"schema_version": 1, "owner_id": self.owner, "renderer": renderer,
            "inventory": observation | {"state": "installed", "bundle_version": "offline-v1"},
            "readiness": observation | {"state": "ready"}}
        self.observations_path = self.root / "observations.json"
        self.write(self.observations_path, self.observations)
        self.configuration = {"schema_version": 1, "owner_id": self.owner,
            "build_receipt": {"path": str(build_path), "sha256": build_sha},
            "bundle": {"configuration_id": configuration_id, "bundle_version": "offline-v1",
                "bundle_sha256": "a" * 64, "renderer": renderer},
            "qualification": {"path": str(self.proof_path),
                "sha256": hashlib.sha256(self.proof_path.read_bytes()).hexdigest()},
            "observations_path": str(self.observations_path)}
        self.path = self.root / "deployment.json"
        self.write(self.path, self.configuration)
        self.environment = {DEPLOYMENT_ENV: str(self.path)}

    def write(self, path, data):
        path.write_text(json.dumps(data))
        path.chmod(0o600)

    def load(self, **kwargs):
        return load_managed_media(self.authority, **({"owner_id": self.owner,
            "sandbox_configuration": self.sandbox, "artifact_configuration": self.artifacts,
            "environ": self.environment, "clock": lambda: self.now} | kwargs))

    def services(self):
        task = TaskService(self.tasks, {}, sandbox_configuration=self.sandbox,
                           artifact_configuration=self.artifacts)
        return SimpleNamespace(db_pool=self.pool, settings=SimpleNamespace(JARVIS_OWNER_ID=self.owner),
            background_task_context_service=SimpleNamespace(repository=self.contexts),
            background_task_service=task, knowledge_runtime=None)

    def test_absent_opt_in_performs_no_reads_and_keeps_media_unavailable(self):
        with patch("app.plugins.media_deployment._read_json", side_effect=AssertionError("No reads")):
            self.assertIsNone(self.load(environ={}))
            with patch.dict(os.environ, {}, clear=True):
                services = self.services()
                initialize_plugin_runtime(services)
        self.assertIsNone(services.background_task_service.managed_plugins)
        self.assertIsNotNone(services.plugin_policy_service)
        self.pool.acquire.assert_not_called()

    async def test_api_binds_real_task_service_and_shared_authority(self):
        services = self.services()
        with patch.dict(os.environ, self.environment, clear=True):
            initialize_plugin_runtime(services)
        runtime = services.background_task_service.managed_plugins
        self.assertIs(runtime.authority, services.plugin_authority)
        self.assertIs(runtime.authority, services.plugin_policy_service.authority)
        self.assertFalse(services.plugin_policy_service.changes_enabled)
        self.assertEqual(runtime.capabilities, ("plugins.video_editor",))
        bundle = next(iter(runtime.bundles.values()))
        await runtime._ready(None, self.owner, bundle)
        with self.assertRaises(TaskError):
            await runtime._ready(None, "different-owner", bundle)
        self.pool.acquire.assert_not_called()

    def test_invalid_selected_config_preserves_api_and_sanitizes_error(self):
        for content in ("private-path-secret", "[]", "{}"):
            with self.subTest(content=content):
                self.path.write_text(content)
                services = self.services()
                task = services.background_task_service
                with patch.dict(os.environ, self.environment, clear=True):
                    initialize_plugin_runtime(services)
                self.assertIs(services.background_task_service, task)
                self.assertIsNone(task.managed_plugins)
                self.assertIsNotNone(services.plugin_policy_service)
                with self.assertRaises(TaskError) as caught:
                    self.load()
                self.assertEqual(caught.exception.code, "managed_media_configuration_invalid")
                self.assertNotIn("private-path-secret", str(caught.exception))

    async def test_expiry_deletion_and_replacement_do_not_refresh_cached_readiness(self):
        runtime = self.load()
        bundle = next(iter(runtime.bundles.values()))
        await runtime._ready(None, self.owner, bundle)
        self.now += timedelta(seconds=300)
        with self.assertRaises(TaskError):
            await runtime._ready(None, self.owner, bundle)
        with self.assertRaises(TaskError):
            self.load()
        self.now -= timedelta(seconds=300)
        self.observations["renderer"]["build_id"] = "wrong-build"
        self.write(self.observations_path, self.observations)
        with self.assertRaises(TaskError):
            await runtime._ready(None, self.owner, bundle)
        self.observations_path.unlink()
        with self.assertRaises(TaskError):
            await runtime._ready(None, self.owner, bundle)

    def test_qualification_bytes_and_actual_render_success_are_required(self):
        self.proof["qualification"]["passed"] = False
        self.write(self.proof_path, self.proof)
        with self.assertRaises(TaskError):
            self.load()
        self.configuration["qualification"]["sha256"] = hashlib.sha256(self.proof_path.read_bytes()).hexdigest()
        self.write(self.path, self.configuration)
        with self.assertRaises(TaskError):
            self.load()

    def test_mismatching_lease_or_dependencies_cannot_activate(self):
        for changes in ({"owner_id": "other"}, {"sandbox_configuration": None},
                        {"artifact_configuration": ArtifactConfiguration(area_id=uuid4())}):
            with self.subTest(changes=tuple(changes)), self.assertRaises(TaskError):
                self.load(**changes)
        for field, value in (("bundle_sha256", "d" * 64), ("state", "unknown"),
                ("observed_at", (self.now + timedelta(seconds=1)).isoformat()),
                ("expires_at", (self.now + timedelta(seconds=301)).isoformat())):
            original = self.observations["readiness"][field]
            self.observations["readiness"][field] = value
            self.write(self.observations_path, self.observations)
            with self.subTest(field=field), self.assertRaises(TaskError):
                self.load()
            self.observations["readiness"][field] = original
        self.write(self.observations_path, self.observations)
        self.path.chmod(0o666)
        with self.assertRaises(TaskError):
            self.load()

    async def worker_run(self, *, command="run"):
        settings = SimpleNamespace(SUPABASE_DB_URL="offline", JARVIS_OWNER_ID=self.owner,
            OPENAI_API_KEY="offline", SUPABASE_URL="offline", SUPABASE_SERVICE_ROLE_KEY="offline")
        frontier = SimpleNamespace(api_key=SimpleNamespace(get_secret_value=lambda: "offline"), model="offline")
        choices = {Capability.TASKS: TaskLimits(), Capability.FRONTIER: frontier,
                   Capability.ARTIFACTS: self.artifacts, Capability.SANDBOX: self.sandbox,
                   Capability.KNOWLEDGE: SimpleNamespace()}
        flags = SimpleNamespace(resolve=lambda capability, **_: SimpleNamespace(
            enabled=True, configured=True, configuration=choices[capability]))
        upstream = SimpleNamespace(client=SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(status_code=200))),
                                   aclose=AsyncMock())
        knowledge = SimpleNamespace(start=Mock(), close=AsyncMock(), set_file_publications=Mock(),
            sources=object(), configuration=SimpleNamespace(area_id=self.artifacts.area_id), context_for=Mock())
        provider = SimpleNamespace(materialize_media_render=AsyncMock(), decode_media_render_input=AsyncMock())
        from pydantic import SecretStr
        provider.configuration = SimpleNamespace(api_key=SecretStr("offline-existing-credential"))
        sandbox_runtime = SimpleNamespace(enabled=True, provider=provider, configuration=self.sandbox,
                                        capabilities=self.sandbox.policy.required_capabilities)
        inspector = AsyncMock()
        inspector.inspect_retained_input = AsyncMock()
        factory_result = []

        def hermes(frontier, native, *, prepare):
            factory_result.append(prepare)
            return SimpleNamespace(prepare=prepare)

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, self.environment, clear=True))
            for name, value in (("worker_settings", lambda: settings), ("SecondBrainSettings", lambda: flags),
                    ("create_pool", AsyncMock(return_value=self.pool)),
                    ("read_schema_readiness", AsyncMock(return_value=SimpleNamespace(ready=True))),
                    ("probe_native", AsyncMock()), ("OpenAIResponsesHTTP", lambda _: upstream),
                    ("is_configured_secret", lambda _: True), ("get_supabase_storage_config", lambda _: None),
                    ("StoredTaskImageFactory", Mock()), ("HermesTaskProvider", hermes),
                    ("VisualFrameStorageClient", Mock(return_value=SimpleNamespace(close=AsyncMock()))),
                    ("JarvisFileStorageClient", Mock(return_value=SimpleNamespace(close=AsyncMock())))):
                stack.enter_context(patch.object(worker, name, value))
            stack.enter_context(patch.object(worker.WorkerRoleConfiguration, "from_flags",
                return_value=SimpleNamespace(native=lambda: None, host_id="offline")))
            stack.enter_context(patch.object(worker.KnowledgeRuntime, "create", AsyncMock(return_value=knowledge)))
            stack.enter_context(patch("app.tasks.file_recovery.TaskFileRecovery", Mock()))
            stack.enter_context(patch("app.sandbox.runtime.load_sandbox_runtime", AsyncMock(return_value=sandbox_runtime)))
            stack.enter_context(patch("app.files.media_runtime.prepare_artifact_media", AsyncMock(return_value=(
                FileTransferPolicy(MP4_OUTPUT_POLICY, 20_000_000), inspector))))
            role = stack.enter_context(patch.object(worker, "TaskWorkerRole"))
            role.return_value.run = AsyncMock()
            health = stack.enter_context(patch.object(worker, "worker_health", AsyncMock(return_value=True)))
            result = await worker.run_command(command)
        return result, factory_result, role, health

    async def test_actual_worker_composition_passes_runtime_to_factory(self):
        result, factories, role, _ = await self.worker_run()
        self.assertEqual(result, 0)
        self.assertEqual(len(factories), 1)
        factory = factories[0]
        self.assertIsInstance(factory, worker.WorkerContextFactory)
        self.assertIn("plugins.video_editor", factory.capabilities)
        self.assertIs(factory.managed_plugins.authority.tasks, role.call_args.args[0])
        bundle = next(iter(factory.managed_plugins.bundles.values()))
        await factory.managed_plugins._ready(None, self.owner, bundle)

    async def test_unselected_worker_preserves_ordinary_factory_without_media_plugin(self):
        self.environment = {}
        result, factories, _, _ = await self.worker_run()
        self.assertEqual(result, 0)
        self.assertIsNone(factories[0].managed_plugins)
        self.assertNotIn("plugins.video_editor", factories[0].capabilities)
        self.assertIsNotNone(factories[0].artifact_configuration)

    async def test_only_explicit_worker_refresh_is_bound_to_existing_role(self):
        from app.plugins.media_probe import WorkerMediaReadinessRefresh
        _, _, role, _ = await self.worker_run()
        self.assertIsNone(role.call_args.kwargs["readiness_refresh"])
        self.configuration["readiness_refresh"] = "worker"
        self.write(self.path, self.configuration)
        with patch("app.plugins.media_probe.provider_state", side_effect=AssertionError("No probe during composition")):
            _, factories, role, _ = await self.worker_run()
        refresh = role.call_args.kwargs["readiness_refresh"]
        self.assertIsInstance(refresh, WorkerMediaReadinessRefresh)
        self.assertEqual(refresh.owner_id, self.owner)
        self.assertEqual(refresh.api_key.get_secret_value(), "offline-existing-credential")
        self.assertEqual(factories[0].managed_plugins.observe.deployment.readiness_refresh, "worker")

    async def test_preparation_checks_real_provider_identity_before_emitting_a_lease(self):
        from app.plugins.media_probe import refresh_observations
        self.observations_path.unlink()
        with patch("app.plugins.media_probe.provider_state", AsyncMock(return_value=(
            ("offline-image", "offline-build", "ready"),
            {"templateID": "offline-image", "buildID": "different-build", "buildStatus": "ready"}))):
            with self.assertRaises(ValueError):
                await refresh_observations(self.path, owner_id=self.owner, api_key="offline-secret")
        self.assertFalse(self.observations_path.exists())
        with patch("app.plugins.media_probe.provider_state", AsyncMock(return_value=(
            ("offline-image", "offline-build", "ready"),
            {"templateID": "offline-image", "buildID": "offline-build", "buildStatus": "ready"}))) as probe:
            observation = await refresh_observations(self.path, owner_id=self.owner, api_key="offline-secret")
        probe.assert_awaited_once()
        self.assertEqual(observation.owner_id, self.owner)
        self.assertEqual((observation.readiness.expires_at - observation.readiness.observed_at).total_seconds(), 300)
        self.assertEqual(self.observations_path.stat().st_mode & 0o777, 0o600)
        self.now = datetime.now(UTC)
        self.assertIsNotNone(self.load())

    async def test_failed_probe_preserves_old_bytes_and_does_not_extend_expiry(self):
        from app.plugins.media_probe import refresh_observations
        original = self.observations_path.read_bytes()
        with patch("app.plugins.media_probe.provider_state", AsyncMock(side_effect=TimeoutError())):
            with self.assertRaises(TimeoutError):
                await refresh_observations(self.path, owner_id=self.owner, api_key="offline-secret")
        self.assertEqual(self.observations_path.read_bytes(), original)
        self.now += timedelta(seconds=300)
        with self.assertRaises(TaskError):
            self.load()

    def test_static_check_never_loads_credentials_or_probes(self):
        from app.plugins import media_probe
        self.observations_path.unlink()
        with patch("sys.argv", ["media_probe", "--check", "--deployment", str(self.path), "--owner", self.owner]), \
                patch.object(media_probe, "read_operator_bytes", side_effect=AssertionError("No credentials")), \
                patch.object(media_probe, "provider_state", side_effect=AssertionError("No network")):
            self.assertEqual(media_probe.main(), 0)

    async def test_worker_check_requires_media_capability_and_invalid_config_fails_startup(self):
        result, factories, _, health = await self.worker_run(command="check")
        self.assertEqual(result, 0)
        self.assertEqual(factories, [])
        self.assertIn("plugins.video_editor", health.call_args.kwargs["required_capabilities"])
        self.observations_path.unlink()
        with self.assertRaises(TaskError) as caught:
            await self.worker_run()
        self.assertEqual(caught.exception.code, "managed_media_configuration_invalid")
