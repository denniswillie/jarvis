"""Controlled ASGI admission tests; no real memory/provider/frontend acceptance."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import unittest
from uuid import uuid4

from fastapi import FastAPI, Request, WebSocket
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.runtime.feature_config import MemoryAuthority, SecondBrainSettings
from app.settings import JarvisSettings
from evals.responsive_voice.frontend_host import (
    FrontendReplayHost, ReplayHostError, ReplayHostScope, scoped_database_url)


class FixturePool:
    def __init__(self, scope):
        self.scope, self.fresh, self.bank = scope, True, scope.memory_bank_id
        self.path = [scope.database_schema, "pg_catalog", "extensions"]

    @asynccontextmanager
    async def acquire(self):
        yield self

    async def fetchval(self, query, *args):
        if query == "select current_schema()":
            return self.path[0]
        if query == "select current_schemas(false)":
            return self.path
        if query == "select clock_timestamp()":
            return datetime.now(UTC)
        if "select exists(select 1 from jarvis_sessions" in query:
            assert args[0] == self.scope.owner_id
            return self.fresh
        raise AssertionError("Unexpected SQL")

    async def fetchrow(self, query, *args):
        assert "jarvis_memory_banks" in query
        assert args == (self.scope.owner_id, self.scope.area_id)
        return {"bank_name": self.bank, "status": "active"}


class FrontendHostTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(UTC)
        run = uuid4().hex
        self.scope = ReplayHostScope(run_id=run,
            owner_id=f"direct-image-eval-second-brain-{run}-a",
            database_schema=f"jarvis_sb_eval_{run}_0", area_id=uuid4(),
            memory_bank_id="jarvis_" + uuid4().hex, expires_at=self.now + timedelta(minutes=5))
        self.settings = JarvisSettings(_env_file=None, JARVIS_OWNER_ID=self.scope.owner_id,
            JARVIS_STATIC_BEARER_TOKEN="isolated-evaluation-token-" + uuid4().hex,
            SUPABASE_DB_URL=scoped_database_url("postgresql://fixture:private@localhost/example?sslmode=require",
                                              self.scope.database_schema))
        self.flags = SecondBrainSettings(_env_file=None, enabled="true", tasks_enabled="true",
            knowledge_enabled="true", attention_enabled="true", memory_authority="hindsight",
            knowledge_area_id=str(self.scope.area_id))
        self.pool = FixturePool(self.scope)
        knowledge = SimpleNamespace(owner_id=self.scope.owner_id,
            configuration=SimpleNamespace(area_id=self.scope.area_id),
            sources=SimpleNamespace(pool=self.pool),
            context_repository=SimpleNamespace(tasks=SimpleNamespace(pool=self.pool)),
            task=SimpleNamespace(done=lambda: False), _closed=False)
        self.runtime = SimpleNamespace(settings=self.settings, memory_authority=MemoryAuthority.HINDSIGHT,
            db_pool=self.pool, knowledge_runtime=knowledge, _closed=False,
            attention_service=SimpleNamespace(push=None))
        self.starts = []

    def host(self):
        @asynccontextmanager
        async def lifespan(app):
            app.state.runtime_services = self.runtime
            yield

        app = FastAPI(lifespan=lifespan)

        @app.post("/sessions/start")
        async def start(request: Request):
            body = await request.json()
            self.starts.append(body)
            return {"body": body, "provenance": request.state.jarvis_evaluation["provenance"]}

        @app.websocket("/realtime/audio/ws")
        async def audio(websocket: WebSocket):
            await websocket.accept()
            await websocket.send_json(websocket.scope["state"]["jarvis_evaluation"])
            await websocket.close()

        return FrontendReplayHost(app, self.scope, settings=self.settings,
                                 flags=self.flags, clock=lambda: self.now)

    def headers(self):
        return {"Authorization": "Bearer " + self.settings.JARVIS_STATIC_BEARER_TOKEN,
                "X-Jarvis-Evaluation-Run": self.scope.run_id}

    def endpoint(self):
        return f"/eval/replay/{self.scope.run_id}/scope"

    def test_scoped_dsn_preserves_existing_options_but_removes_public_fallback(self):
        url = scoped_database_url("postgres://u:p@localhost/db?search_path=public&sslmode=require",
                                  self.scope.database_schema)
        self.assertNotIn("public", url)
        self.assertIn("sslmode=require", url)
        self.assertEqual(url.count("search_path="), 1)

    def test_attestation_requires_lifespan_proof_and_exposes_no_credentials(self):
        host = self.host()
        with TestClient(host) as client:
            response = client.get(self.endpoint(), headers=self.headers())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), self.scope.public())
            self.assertNotIn(self.settings.JARVIS_STATIC_BEARER_TOKEN, response.text)
            self.assertNotIn("postgres", response.text)
        self.assertFalse(host.ready)

    def test_missing_wrong_or_duplicate_headers_are_rejected(self):
        with TestClient(self.host()) as client:
            for headers in ({}, {**self.headers(), "X-Jarvis-Evaluation-Run": uuid4().hex},
                            {**self.headers(), "Authorization": "Bearer wrong"},
                            list(self.headers().items()) + [("X-Jarvis-Evaluation-Run", self.scope.run_id)]):
                with self.subTest(headers_count=len(headers)):
                    self.assertNotEqual(client.get(self.endpoint(), headers=headers).status_code, 200)

    def test_only_explicit_false_resume_is_forwarded(self):
        with TestClient(self.host()) as client:
            for value in ({}, {"resume_active": True}, {"resume_active": None}, {"resume_active": 0}):
                with self.subTest(value=value):
                    self.assertEqual(client.post("/sessions/start", json=value,
                                                 headers=self.headers()).status_code, 403)
            response = client.post("/sessions/start", json={"resume_active": False, "title": "Fixture"},
                                   headers=self.headers())
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["provenance"], "frontend_replay")
            self.assertEqual(self.starts, [{"resume_active": False, "title": "Fixture"}])

    def test_oversized_and_invalid_start_body_cannot_reach_application(self):
        with TestClient(self.host()) as client:
            for body in (b"{" , b" " * 16385):
                self.assertEqual(client.post("/sessions/start", content=body,
                                             headers=self.headers()).status_code, 403)
        self.assertEqual(self.starts, [])

    def test_owner_and_unscoped_database_fail_before_startup(self):
        self.settings.JARVIS_OWNER_ID = "personal-owner"
        with self.assertRaisesRegex(ReplayHostError, "owner_configuration"):
            self.host()
        self.settings.JARVIS_OWNER_ID = self.scope.owner_id
        self.settings.SUPABASE_DB_URL = "postgresql://localhost/db"
        with self.assertRaisesRegex(ReplayHostError, "database_path"):
            self.host()

    def test_bank_and_schema_mismatch_prevent_attestation(self):
        self.pool.bank = "jarvis_" + uuid4().hex
        with self.assertRaisesRegex(ReplayHostError, "canonical_bank"):
            with TestClient(self.host()):
                pass
        self.pool.bank = self.scope.memory_bank_id
        self.pool.path.append("public")
        with self.assertRaisesRegex(ReplayHostError, "runtime_schema"):
            with TestClient(self.host()):
                pass

    def test_external_push_is_rejected_even_when_flag_claims_disabled(self):
        self.runtime.attention_service.push = object()
        with self.assertRaisesRegex(ReplayHostError, "notification_binding"):
            with TestClient(self.host()):
                pass

    def test_enabled_external_capability_is_rejected_before_startup(self):
        self.flags = self.flags.model_copy(update={"push_enabled": "true"})
        with self.assertRaisesRegex(ReplayHostError, "external_capability"):
            self.host()

    def test_wall_clock_expiry_and_runtime_owner_change_fence_later_requests(self):
        with TestClient(self.host()) as client:
            self.runtime.knowledge_runtime.owner_id = "wrong-owner"
            self.assertEqual(client.get(self.endpoint(), headers=self.headers()).status_code, 403)
            self.runtime.knowledge_runtime.owner_id = self.scope.owner_id
            self.now = self.scope.expires_at
            self.assertEqual(client.get(self.endpoint(), headers=self.headers()).status_code, 403)

    def test_websocket_requires_fresh_active_session_in_same_owner(self):
        with TestClient(self.host()) as client:
            path = "/realtime/audio/ws?session_id=" + str(uuid4())
            with client.websocket_connect(path, headers=self.headers()) as websocket:
                self.assertEqual(websocket.receive_json()["run_id"], self.scope.run_id)
            self.pool.fresh = False
            with self.assertRaises(WebSocketDisconnect) as caught:
                with client.websocket_connect(path, headers=self.headers()):
                    pass
            self.assertEqual(caught.exception.code, 1008)

    def test_scope_requires_exact_fixture_identity(self):
        with self.assertRaises(ValueError):
            ReplayHostScope(**{**self.scope.model_dump(), "owner_id": "default"})

