"""Application route composition without lifespan, providers, or SQL."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from fastapi.testclient import TestClient
from app.auth.static_bearer import require_static_bearer, JarvisPrincipal
from app.settings import JarvisSettings

class WorkspaceRouteRegistrationTests(unittest.TestCase):
    def setUp(self):
        from app.main import create_app
        self.settings = JarvisSettings(_env_file=None, JARVIS_OWNER_ID="test-owner",
                                      JARVIS_STATIC_BEARER_TOKEN="isolated-workspace-test-token")
        with patch("app.main.get_settings", return_value=self.settings):
            self.app = create_app()
        self.app.dependency_overrides[require_static_bearer] = lambda: JarvisPrincipal("test-owner")
        self.presentations = SimpleNamespace(list_requests=AsyncMock(return_value={"items": []}))
        self.services = SimpleNamespace(settings=self.settings, workspace_presentations=self.presentations)
        self.app.state.runtime_services = self.services
        self.client = TestClient(self.app)  # Deliberately no real lifespan/startup.
        self.addCleanup(self.client.close)
        self.session = uuid4()

    def test_presentations_bind_principal_session_without_note_service(self):
        response = self.client.get("/workspace/requests", params={"session_id": str(self.session)})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"items": []})
        ctx = self.presentations.list_requests.call_args.args[0]
        self.assertEqual((ctx.owner_id, ctx.session_id, ctx.operation_id), ("test-owner", self.session, None))
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertEqual(response.headers["vary"], "Authorization")

    def test_note_routes_are_absent(self):
        for path in ("/workspace/note", "/workspace/artifacts",
                     f"/workspace/artifacts/{uuid4()}", f"/workspace/artifacts/{uuid4()}/versions"):
            for method in ("get", "post"):
                with self.subTest(path=path, method=method):
                    response = getattr(self.client, method)(path, params={"session_id": str(self.session)})
                    self.assertEqual(response.status_code, 404)
        self.presentations.list_requests.assert_not_awaited()

    def test_unknown_presentation_is_not_reported_as_empty_success(self):
        self.services.workspace_presentations = None
        response = self.client.get("/workspace/requests", params={"session_id": str(self.session)})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "workspace_unavailable")

    def test_foreign_principal_cannot_reach_presentation_service(self):
        self.app.dependency_overrides[require_static_bearer] = lambda: JarvisPrincipal("foreign")
        response = self.client.get("/workspace/requests", params={"session_id": str(self.session)})
        self.assertEqual(response.status_code, 503)
        self.presentations.list_requests.assert_not_awaited()
