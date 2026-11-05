from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.auth.client_context import router, storage_namespace
from app.errors import JarvisHTTPError
from app.settings import get_settings


class ClientContextTests(unittest.TestCase):
    def setUp(self):
        self.session = uuid4()
        self.settings = SimpleNamespace(JARVIS_OWNER_ID="fixture-owner",
            JARVIS_STATIC_BEARER_TOKEN="fixture-context-bearer-secret",
            SUPABASE_DB_URL="postgresql://project_user:password@db.example.test:5432/postgres")
        self.connection = SimpleNamespace(fetchrow=AsyncMock(return_value={"schema_name": "public"}))
        manager = MagicMock()
        manager.__aenter__.return_value = self.connection
        pool = SimpleNamespace(acquire=MagicMock(return_value=manager))
        self.services = SimpleNamespace(settings=self.settings, db_pool=pool,
            knowledge_runtime=SimpleNamespace(sources=SimpleNamespace(pool=pool)), workspace_notes=object())
        app = FastAPI()
        app.include_router(router)
        app.state.runtime_services = self.services
        app.dependency_overrides[get_settings] = lambda: self.settings
        @app.exception_handler(JarvisHTTPError)
        async def error_handler(request, error):
            return JSONResponse({"code": error.code}, status_code=error.status_code)
        self.client = TestClient(app)

    def request(self, **kwargs):
        return self.client.get("/client-context", params={"session_id": str(self.session)},
            headers={"Authorization": "Bearer " + self.settings.JARVIS_STATIC_BEARER_TOKEN}, **kwargs)

    def test_success_uses_authenticated_owner_and_owned_session(self):
        response = self.request()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"owner_id": "fixture-owner", "session_id": str(self.session),
            "backend_namespace": storage_namespace(self.settings.SUPABASE_DB_URL)})
        self.assertEqual(self.connection.fetchrow.call_args.args[1:], ("fixture-owner", self.session))
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertNotIn("password", response.text)
        self.assertNotIn("db.example", response.text)

    def test_missing_bearer_does_not_access_storage(self):
        response = self.client.get("/client-context", params={"session_id": str(self.session)})
        self.assertEqual(response.status_code, 401)
        self.connection.fetchrow.assert_not_awaited()

    def test_wrong_bearer_does_not_access_storage(self):
        response = self.client.get("/client-context", params={"session_id": str(self.session)},
            headers={"Authorization": "Bearer wrong-fixture-secret"})
        self.assertEqual(response.status_code, 403)
        self.connection.fetchrow.assert_not_awaited()

    def test_foreign_session_cannot_unlock_namespace(self):
        self.connection.fetchrow.return_value = None
        self.assertEqual(self.request().status_code, 404)

    def test_legacy_source_pool_is_not_a_client_context_dependency(self):
        self.services.knowledge_runtime.sources.pool = object()
        self.assertEqual(self.request().status_code, 200)

    def test_disabled_legacy_knowledge_still_verifies_owned_session(self):
        self.services.knowledge_runtime = None
        self.assertEqual(self.request().status_code, 200)
        self.assertEqual(self.connection.fetchrow.call_args.args[1:], ("fixture-owner", self.session))

    def test_unavailable_database_is_not_verified(self):
        self.services.db_pool = None
        self.assertEqual(self.request().status_code, 503)
        self.connection.fetchrow.assert_not_awaited()

    def test_runtime_owner_must_match_authenticated_owner(self):
        self.services.settings = SimpleNamespace(JARVIS_OWNER_ID="other-owner")
        self.assertEqual(self.request().status_code, 503)
        self.connection.fetchrow.assert_not_awaited()

    def test_failed_database_verification_keeps_context_unavailable(self):
        self.connection.fetchrow.side_effect = OSError("fixture disconnected")
        self.assertEqual(self.request().status_code, 503)

    def test_bearer_and_database_password_rotation_preserve_namespace(self):
        first = self.request().json()["backend_namespace"]
        self.settings.JARVIS_STATIC_BEARER_TOKEN = "rotated-context-bearer-secret"
        self.settings.SUPABASE_DB_URL = self.settings.SUPABASE_DB_URL.replace(":password@", ":replacement@")
        self.assertEqual(self.request().json()["backend_namespace"], first)

    def test_different_storage_project_or_owner_stays_separate(self):
        original = self.request().json()
        self.settings.SUPABASE_DB_URL = self.settings.SUPABASE_DB_URL.replace("project_user", "other_project")
        changed = self.request().json()
        self.assertNotEqual(changed["backend_namespace"], original["backend_namespace"])
        self.settings.JARVIS_OWNER_ID = "other-owner"
        self.assertNotEqual(self.request().json()["owner_id"], original["owner_id"])

    def test_bad_database_route_has_no_guessed_namespace(self):
        for value in (None, "", "https://db.example.test/postgres", "postgresql://localhost/"):
            with self.subTest(value=value), self.assertRaises(JarvisHTTPError):
                storage_namespace(value)

    def test_actual_sql_schema_is_part_of_namespace(self):
        initial = self.request().json()["backend_namespace"]
        self.connection.fetchrow.return_value = {"schema_name": "separate_source_schema"}
        self.assertNotEqual(self.request().json()["backend_namespace"], initial)
