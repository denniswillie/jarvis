"""The callback is a route on the ordinary Jarvis app, not a second server."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import httpx

from app.main import create_app
from app.hermes_sessions.background_events import BackgroundEvents
from app.hermes_sessions.bindings import SessionBindings
from app.hermes_sessions.sandbox_bindings import SessionSandboxBindings
from app.hermes_sessions.service import RetainedSessionService
from app.sandbox.models import SandboxPolicy
from test_service import NativeFixture


class SharedBackendCallbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jarvis-shared-callback-")
        self.home = Path(self.temp.name).resolve()
        bindings = SessionBindings(self.home / "sessions.sqlite", owner_id="owner", profile_home=self.home)
        self.native = NativeFixture(self.home)
        service = RetainedSessionService(self.native, bindings,
            launch={"source": "cli", "cwd": str(self.home), "toolsets": []})
        self.master = await service.master()
        self.resources = SessionSandboxBindings(self.home / "resources.sqlite", owner_id="owner", profile_id="profile")
        reserved = self.resources.reserve(self.master.logical_id, SandboxPolicy())
        self.resources.start_creation(reserved)
        self.resources.bind(reserved, "fixture-sandbox")
        self.events = BackgroundEvents(sessions=service, sandbox_bindings=self.resources, signing_key=b"k" * 32)
        self.body = {"session_id": self.master.logical_id, "event_id": "result-1", "message": "Result ready"}
        self.headers = {"Authorization": "Bearer " + self.events.credential(self.master.logical_id)}

    async def asyncTearDown(self):
        self.resources.close()
        self.temp.cleanup()

    async def test_shared_app_accepts_authenticated_callback_and_keeps_other_routes(self):
        runtime = SimpleNamespace(run=lambda stop: None, current_events=lambda: self.events)
        app = create_app(hermes_profile_runtime=runtime)
        paths = app.openapi()["paths"]
        self.assertIn("/health", paths)
        self.assertIn("/hermes/background-wake", paths)
        self.assertGreater(len(paths), 2)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://jarvis") as client:
            response = await client.post("/hermes/background-wake", json=self.body, headers=self.headers)
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()["session_id"], self.master.logical_id)
            self.assertEqual(len(self.native.inputs), 1)

    async def test_unconfigured_backend_does_not_start_a_profile_for_callback(self):
        app = create_app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://jarvis") as client:
            self.assertEqual((await client.post("/hermes/background-wake", json=self.body)).status_code, 401)
            self.assertEqual((await client.post("/hermes/background-wake", json=self.body, headers=self.headers)).status_code, 503)
        self.assertEqual(self.native.inputs, {})

    def test_runtime_cannot_be_selected_through_arbitrary_factory_values(self):
        with self.assertRaises(TypeError):
            create_app(hermes_profile_runtime="model-selected-profile")


if __name__ == "__main__":
    unittest.main()
