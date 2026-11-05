import json
import asyncio
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.hermes_sessions.serving import configured_profile_runtime


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name).resolve()
        (self.home / "jarvis").mkdir()
        (self.home / "jarvis" / "callback.key").write_bytes(b"k" * 32)
        self.key_file = self.home / "provider-key"
        self.key_file.write_text("fixture-key")
        self.config = {"owner_id": "owner", "sandbox_policy": {"internet_access": True}}
        self.settings = SimpleNamespace(JARVIS_OWNER_ID="owner", PUBLIC_BASE_URL="https://example.invalid/")

    def configure(self):
        (self.home / "jarvis-adapter.json").write_text(json.dumps(self.config))
        return patch.dict(os.environ, {"JARVIS_HERMES_PROFILE_HOME": str(self.home),
            "JARVIS_FRONTIER_E2B_API_KEY_FILE": str(self.key_file)})

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(configured_profile_runtime())

    def test_uses_existing_origin_and_explicit_profile(self):
        with self.configure(), patch("app.hermes_sessions.sandbox_provider.RetainedE2BProvider") as provider, \
                patch("app.hermes_sessions.runtime.RetainedProfileRuntime") as runtime:
            result = configured_profile_runtime(settings=self.settings)
            self.assertIs(result, runtime.return_value)
            self.assertEqual(runtime.call_args.args, (self.home,))
            self.assertEqual(runtime.call_args.kwargs["callback_endpoint"],
                "https://example.invalid/hermes/background-wake")
            self.assertTrue(runtime.call_args.kwargs["sandbox_policy"].internet_access)
            self.assertEqual(runtime.call_args.kwargs["sandbox_policy"].template, "desktop")
            self.assertEqual(runtime.call_args.kwargs["sandbox_policy"].lifetime_seconds, 3600)
            self.assertEqual(provider.call_args.args[0].api_key.get_secret_value(), "fixture-key")

    def test_wrong_owner_and_uncomposed_domain_profile_refused(self):
        for changed in ({"owner_id": "someone-else"}, {"domain_tools": True}):
            with self.subTest(changed=changed):
                self.config = {"owner_id": "owner", **changed}
                with self.configure(), self.assertRaises(ValueError):
                    configured_profile_runtime(settings=self.settings)

    def test_existing_factory_connects_master_notices_without_signature_changes(self):
        manager = object()
        services = SimpleNamespace(realtime_session_manager=manager)
        with self.configure(), patch("app.hermes_sessions.sandbox_provider.RetainedE2BProvider"), \
                patch("app.hermes_sessions.runtime.RetainedProfileRuntime"), \
                patch("app.realtime.live_retained.RetainedMasterNotices") as notices:
            runtime = configured_profile_runtime(settings=self.settings, runtime_services=services)
            notices.assert_called_once_with(runtime, manager)
            self.assertIs(runtime.events.on_admitted, notices.return_value.admitted)
            self.assertIs(runtime.host.on_pump, notices.return_value.pump)

    def test_missing_persistent_key_is_not_silently_rotated(self):
        (self.home / "jarvis" / "callback.key").unlink()
        with self.configure(), self.assertRaises(FileNotFoundError):
            configured_profile_runtime(settings=self.settings)

    def test_domain_composition_uses_existing_files_and_scopes_without_memory(self):
        from app.knowledge.files import SourceFileRepository
        from app.runtime.feature_config import TaskLimits
        from app.tasks.repository import TaskRepository
        from test_work import ResultFixture
        pool = object()
        files = SourceFileRepository(SimpleNamespace(pool=pool), SimpleNamespace(bucket="fixture"),
                                     owner_id="owner", area_id=uuid4())
        services = SimpleNamespace(settings=self.settings, db_pool=pool,
            recording_source_access=SimpleNamespace(files=files, tasks=TaskRepository(pool, TaskLimits())),
            tool_modules=())
        self.config.update(domain_tools=True, launch={"source": "cli", "cwd": str(self.home), "toolsets": []})
        native = ResultFixture(self.home)
        with self.configure(), patch("app.hermes_sessions.sandbox_provider.RetainedE2BProvider") as provider, \
                patch("app.hermes_sessions.client.HermesGatewayClient", return_value=native), \
                patch("app.hermes_sessions.runtime.RetainedProfileRuntime") as constructor, \
                patch("app.sandbox.view_secrets.SandboxViewSecrets.configured", return_value="private-view-secrets"):
            runtime = configured_profile_runtime(settings=self.settings, runtime_services=services)
            self.assertEqual(provider.call_args.kwargs["view_secrets"], "private-view-secrets")
            self.assertIs(runtime.source_domain.files, files)
            self.assertIs(runtime.source_domain.adapter.contexts.publication.sources, files.sources)
            self.assertEqual(runtime.source_domain.adapter.contexts.publication.area_id, files.area_id)
            self.assertIs(runtime.source_domain.adapter.contexts.publication.selections, files.selections)
            published = json.loads((self.home / "jarvis/domain-tools.json").read_text())
            self.assertEqual({tool["name"] for tool in published["tools"]}, set(runtime.voice_allowed_tools))
            self.assertEqual(runtime.voice_allowed_tools, frozenset({"list_files", "get_file_metadata",
                "create_text_file", "sandbox_stage_file", "sandbox_publish_file", "sandbox_view_image"}))
            self.assertIs(constructor.call_args.kwargs["domain_adapter"], runtime.source_domain.adapter)
            async def exercise():
                master = await runtime.source_domain.adapter.service.master()
                scope = await runtime.prepare_voice_input(session_id=master.logical_id, input_id="voice-one",
                                                          jarvis_session_id=uuid4())
                self.assertIsNone(scope.request_event_id)
                wake = await constructor.call_args.kwargs["prepare_event_scope"](
                    session_id=master.logical_id, native_input_id="wake-one", event={"kind": "sandbox_message"})
                self.assertEqual(wake.jarvis_session_id, scope.jarvis_session_id)
                self.assertEqual(wake.allowed_tools, scope.allowed_tools)
                self.assertIsNone(wake.request_event_id)
                self.assertIsNone(wake.request_revision)
            asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
