"""Recording composition, not capture/cloud/database qualification."""
from contextlib import asynccontextmanager
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.files.media_inspection import MediaInspectionError
from app.files.media_runtime import prepare_media_inspector
from app.files.transfer import CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS, FileTransferPolicy
from app.recordings.runtime import initialize_recording_runtime
from app.recordings.source_access import RecordingSourceAccess
from app.knowledge.models import SourceError
from app.files.router import _repository
from app.runtime.workspace_delivery import initialize_workspace_presentations
from app.runtime.services import JarvisRuntimeServices
from app.errors import JarvisHTTPError
from app.runtime.feature_config import Capability, MediaInspectorConfiguration, MemoryAuthority, SecondBrainSettings


def flags(**extra):
    values = dict(_env_file=None, enabled="true", recordings_enabled="true", knowledge_enabled="true",
        hindsight_api_key="fixture-configured-key", hindsight_url="https://memory.example.com",
        knowledge_area_id=str(uuid4()), recording_storage_max_file_bytes="40000000",
        recording_ffprobe_path="/not-installed/ffprobe", recording_ffprobe_sha256="a" * 64)
    return SecondBrainSettings(**(values | extra))


class RecordingSettingsTests(unittest.TestCase):
    def test_disabled_feature_is_lazy(self):
        result = flags(recordings_enabled="false", recording_storage_max_file_bytes="bad").resolve(Capability.RECORDINGS)
        self.assertFalse(result.enabled)
        self.assertIsNone(result.issue)

    def test_capture_allows_named_maximum_without_expanding_generated_limits(self):
        result = flags(recording_storage_max_file_bytes=str(CAPTURE_MAX_BYTES)).resolve(
            Capability.RECORDINGS, database_url="postgresql://configured")
        self.assertTrue(result.configured)
        self.assertEqual(result.configuration.media.storage_max_file_bytes, CAPTURE_MAX_BYTES)

    def test_recordings_need_owned_area_but_no_memory_credentials_or_notes(self):
        area = uuid4()
        result = flags(knowledge_enabled="false", knowledge_area_id="", recording_area_id=str(area),
            hindsight_api_key="", hindsight_url="").resolve(Capability.RECORDINGS, database_url="postgresql://configured")
        self.assertTrue(result.configured)
        self.assertEqual(result.configuration.area_id, area)

    def test_missing_dependency_or_pin_does_not_claim_configuration(self):
        for extra in ({"knowledge_area_id": ""}, {"recording_ffprobe_sha256": ""},
                      {"recording_storage_max_file_bytes": str(CAPTURE_MAX_BYTES + 1)}):
            self.assertFalse(flags(**extra).resolve(Capability.RECORDINGS,
                                                    database_url="postgresql://configured").configured)


class RecordingRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.connection = SimpleNamespace(fetchval=AsyncMock(side_effect=["fixture", True]))

        @asynccontextmanager
        async def acquire():
            yield self.connection

        pool = SimpleNamespace(acquire=Mock(side_effect=acquire))
        sources, tasks = SimpleNamespace(pool=pool), SimpleNamespace(pool=pool)
        runtime = SimpleNamespace(sources=sources, context_repository=SimpleNamespace(tasks=tasks),
            area_selections=SimpleNamespace(sources=sources), context_for=Mock())
        self.services = SimpleNamespace(settings=SimpleNamespace(SUPABASE_DB_URL="postgresql://configured",
            JARVIS_OWNER_ID="fixture-owner"), memory_authority=MemoryAuthority.HINDSIGHT,
            knowledge_runtime=runtime, db_pool=pool, jarvis_file_storage=object())
        self.ready = patch("app.recordings.runtime.read_schema_readiness", new_callable=AsyncMock,
                           return_value=SimpleNamespace(ready=True)).start()
        self.inspect = patch("app.recordings.runtime.prepare_media_inspector", new_callable=AsyncMock,
                             return_value=AsyncMock()).start()
        self.admit = patch("app.recordings.runtime.RecordingAdmission").start()
        self.ingress = patch("app.recordings.runtime.RecordingIngress").start()
        self.addCleanup(patch.stopall)

    async def test_disabled_does_not_acquire_pool_or_probe(self):
        await initialize_recording_runtime(self.services, flags=flags(recordings_enabled="false"))
        self.services.db_pool.acquire.assert_not_called()
        self.inspect.assert_not_awaited()
        self.assertIsNone(self.services.recording_ingress)

    async def test_selected_runtime_reuses_exact_existing_owners(self):
        await initialize_recording_runtime(self.services, flags=flags())
        runtime = self.services.knowledge_runtime
        self.admit.assert_called_once_with(runtime.sources, runtime.context_repository.tasks,
            runtime.area_selections, owner_id="fixture-owner", access_check=runtime.context_for)
        self.assertIs(self.services.recording_admission, self.admit.return_value)
        self.assertIs(self.services.recording_ingress, self.ingress.return_value)
        self.assertEqual(self.services.recording_policy.policy_id, "mp4_capture_v1")
        self.assertEqual(self.services.recording_policy.max_bytes, 40000000)
        self.assertIsNone(self.services.recording_issue)

    async def test_unavailable_storage_or_selected_memory_does_not_fall_back(self):
        self.services.jarvis_file_storage = None
        await initialize_recording_runtime(self.services, flags=flags())
        self.assertEqual(self.services.recording_issue, "recording_dependencies_unavailable")
        self.services.jarvis_file_storage = object()
        self.services.knowledge_runtime = None
        await initialize_recording_runtime(self.services, flags=flags())
        self.inspect.assert_not_awaited()
        self.admit.assert_not_called()

    async def test_different_pool_is_rejected_before_io(self):
        self.services.knowledge_runtime.context_repository.tasks.pool = object()
        await initialize_recording_runtime(self.services, flags=flags())
        self.assertEqual(self.services.recording_issue, "recording_scope_configuration_invalid")
        self.services.db_pool.acquire.assert_not_called()

    async def test_missing_migration_does_not_advertise_recording(self):
        self.ready.return_value = SimpleNamespace(ready=False)
        await initialize_recording_runtime(self.services, flags=flags())
        self.assertEqual(self.services.recording_issue, "recording_schema_not_ready")
        self.inspect.assert_not_awaited()

    async def test_failed_database_is_explicit_and_does_not_start_capture(self):
        self.connection.fetchval.side_effect = TimeoutError()
        await initialize_recording_runtime(self.services, flags=flags())
        self.assertEqual(self.services.recording_issue, "recording_database_unavailable")
        self.admit.assert_not_called()

    async def test_bad_binary_is_explicit_and_does_not_mount_an_ingress_instance(self):
        self.inspect.side_effect = MediaInspectionError("media_probe_binary_mismatch")
        await initialize_recording_runtime(self.services, flags=flags())
        self.assertEqual(self.services.recording_issue, "recording_inspector_not_ready")
        self.assertIsNone(self.services.recording_ingress)

    async def test_legacy_composition_preserves_memory_and_uses_owned_source_access(self):
        self.services.memory_authority = MemoryAuthority.LEGACY
        self.services.knowledge_runtime = None
        sentinel = object()
        self.services.tool_modules = (sentinel,)
        pool = self.services.db_pool
        sources = SimpleNamespace(pool=pool, _area=AsyncMock())
        @asynccontextmanager
        async def transaction(owner):
            self.assertEqual(owner, "fixture-owner")
            yield self.connection
        sources.transaction = transaction
        with patch("app.recordings.source_access.SourceRepository", return_value=sources):
            await initialize_recording_runtime(self.services, flags=flags(knowledge_enabled="false", hindsight_api_key=""))
        access = self.services.recording_source_access
        self.assertIsNotNone(access)
        self.assertIs(self.services.memory_authority, MemoryAuthority.LEGACY)
        self.assertIsNone(self.services.knowledge_runtime)
        self.assertEqual(self.services.tool_modules, (sentinel,))
        self.admit.assert_called_once_with(sources, access.tasks, access.area_selections,
            owner_id="fixture-owner", access_check=access.check)
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime_services=self.services)))
        self.assertIs(_repository(request, "fixture-owner"), access.files)
        with self.assertRaises(JarvisHTTPError):
            _repository(request, "another-owner")
        initialize_workspace_presentations(self.services)
        presentations = self.services.workspace_presentations
        self.assertIsNotNone(presentations)
        with self.assertRaises(JarvisHTTPError):
            await presentations.resolver(None, "note", uuid4(), "v1")
        held = access.files.for_session(uuid4())
        await access.close()
        for check in (access.check, held.access_check):
            with self.assertRaises(SourceError) as raised:
                check("fixture-owner")
            self.assertEqual(raised.exception.code, "recording_access_closed")
        with self.assertRaises(SourceError):
            _repository(request, "fixture-owner")

    async def test_legacy_area_failure_does_not_offer_files_or_recording(self):
        self.services.memory_authority = MemoryAuthority.LEGACY
        self.services.knowledge_runtime = None
        with patch.object(RecordingSourceAccess, "create", new_callable=AsyncMock,
            side_effect=SourceError("knowledge_area_unavailable", "Missing owned area", 404)):
            await initialize_recording_runtime(self.services, flags=flags())
        self.assertEqual(self.services.recording_issue, "knowledge_area_unavailable")
        self.assertIsNone(self.services.recording_source_access)
        self.admit.assert_not_called()

    async def test_lifecycle_drains_sessions_then_fences_files_before_closing_pool(self):
        order = []
        async def record(name):
            order.append(name)
        services = JarvisRuntimeServices(settings=self.services.settings,
            realtime_session_manager=SimpleNamespace(close_all=lambda: record("sessions")),
            sensory_frame_window=None, latest_visual_frame_buffer=None, domain_event_bus=None, model_limiter=None,
            recording_source_access=SimpleNamespace(close=lambda: record("recordings")),
            jarvis_file_storage=SimpleNamespace(close=lambda: record("storage")),
            db_pool=SimpleNamespace(close=lambda: record("pool")))
        await services.close()
        await services.close()
        self.assertEqual(order, ["sessions", "recordings", "storage", "pool"])


class CaptureInspectorPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_inspector_cannot_be_used_as_generated_inspector(self):
        with tempfile.TemporaryDirectory(prefix="jarvis-recording-inspector-") as directory:
            path = Path(directory) / "fixture"
            content = b"never invoked fixture executable\n"
            path.write_bytes(content)
            path.chmod(0o700)
            config = MediaInspectorConfiguration(storage_max_file_bytes=40000000,
                ffprobe_path=str(path), ffprobe_sha256=hashlib.sha256(content).hexdigest())
            policy = FileTransferPolicy("mp4_capture_v1", 40000000)
            inspect = await prepare_media_inspector(config, policy)
            with self.assertRaises(MediaInspectionError):
                await inspect(object(), FileTransferPolicy("mp4_output_v1", 20000000))
            with patch("app.files.media_runtime.inspect_mp4_file", new_callable=AsyncMock, return_value="capture"):
                self.assertEqual(await inspect(object(), policy), "capture")
