"""Fixture wiring/auth tests; native task/storage and interactive evidence remain separate."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx

from app.knowledge.files import SourceFileRepository
from app.settings import JarvisSettings
from app.tasks.models import TaskError
from evals.second_brain.task_file_phone_fixture import application, live_http


class TaskFilePhoneFixtureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.token = "synthetic-only-unit-token"
        self.settings = JarvisSettings(_env_file=None, JARVIS_OWNER_ID="synthetic-phone-owner",
                                      JARVIS_STATIC_BEARER_TOKEN=self.token)
        self.files = Mock(spec=SourceFileRepository)
        self.files.read_content = AsyncMock(return_value=b"Synthetic file\n")
        self.files.get_file = AsyncMock(return_value=SimpleNamespace(filename="note.txt", content_type="text/plain"))
        self.files.delete_source_file = AsyncMock(return_value={"deleted": True})
        self.tasks = AsyncMock()
        self.tasks.get.side_effect = TaskError("task_not_found", "No task for this owner.", 404)
        self.state, self.persist, self.stop = {"requests": []}, Mock(), Mock()
        self.app = application(self.settings, self.files, Mock(), SimpleNamespace(repository=self.tasks),
                               self.state, self.persist, stop=self.stop)
        self.client = await self.enterAsyncContext(httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://fixture"))
        self.headers = {"Authorization": "Bearer " + self.token}

    async def test_all_data_and_control_routes_require_authentication(self):
        identity = str(uuid4())
        for headers, expected_status, expected_code in (
                ({}, 401, "jarvis_auth_required"),
                ({"Authorization": "Bearer invalid-fixture-token"}, 403, "jarvis_auth_invalid")):
            for method, route in (("GET", "/tasks"), ("GET", "/attention"), ("GET", "/files"),
                    ("GET", "/files/" + identity + "/content"), ("GET", "/__fixture/status"),
                    ("POST", "/__fixture/revoke/" + identity), ("POST", "/__fixture/stop")):
                with self.subTest(route=route, expected_status=expected_status):
                    response = await self.client.request(method, route, headers=headers)
                    self.assertEqual(response.status_code, expected_status)
                    self.assertEqual(response.json()["error"]["code"], expected_code)
        self.files.read_content.assert_not_awaited()
        self.files.delete_source_file.assert_not_awaited()
        self.stop.assert_not_called()

    async def test_task_errors_use_the_real_authenticated_route(self):
        identity = uuid4()
        response = await self.client.get("/tasks/" + str(identity), headers=self.headers)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "task_not_found")
        self.tasks.get.assert_awaited_once_with(self.settings.JARVIS_OWNER_ID, identity)

    async def test_real_file_route_preserves_exact_bytes_and_private_download_headers(self):
        identity = uuid4()
        response = await self.client.get("/files/" + str(identity) + "/content", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"Synthetic file\n")
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertTrue(response.headers["content-disposition"].startswith("attachment;"))
        self.files.read_content.assert_awaited_once_with(owner_id=self.settings.JARVIS_OWNER_ID, file_id=identity)

    async def test_revocation_uses_the_configured_owner(self):
        identity = uuid4()
        response = await self.client.post("/__fixture/revoke/" + str(identity), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.files.delete_source_file.assert_awaited_once_with(owner_id=self.settings.JARVIS_OWNER_ID, file_id=identity)

    async def test_stop_requests_outer_cleanup_without_recording_credentials(self):
        response = await self.client.post("/__fixture/stop", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.stop.assert_called_once_with()
        self.assertTrue(self.state["requests"][-1]["authorized"])
        self.assertNotIn(self.token, json.dumps(self.state))

    async def test_loopback_server_has_a_bounded_owned_lifecycle(self):
        async with live_http(self.app) as running:
            async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                response = await client.get(running.url + "/health")
                self.assertEqual(response.json(), {"status": "ok", "synthetic_only": True})
            self.assertFalse(running.task.done())
        self.assertTrue(running.task.done())
