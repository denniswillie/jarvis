"""Actual authenticated phone routes; no native model, email, push or live voice."""
import json
import os
from pathlib import Path
import unittest
from uuid import uuid4

import httpx

from app.tasks.models import TaskCreate
from evals.second_brain.source_fixture import source_fixture
from evals.second_brain.work_phone_fixture import create_app, fixture_snapshot


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class WorkPhoneRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.app, self.ids = await create_app(self.fixture, "synthetic-work-token")
        self.client = await self.enterAsyncContext(httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://fixture.invalid",
            headers={"Authorization": "Bearer synthetic-work-token"}))

    async def test_fixture_final_snapshot_is_json_serializable(self):
        snapshot = await fixture_snapshot(self.app, self.fixture.owner)
        saved = json.loads(json.dumps(snapshot))
        self.assertEqual(len(saved["tasks"]), 3)
        self.assertEqual(len(saved["inbox"]), 3)
        self.assertEqual({row["state"] for row in saved["inbox"]}, {"pending", "unknown"})
        self.assertTrue(all(isinstance(row["id"], str) for row in saved["inbox"]))
        self.assertEqual((await self.client.get("/__fixture/status")).json(), saved)

    async def test_owner_filter_read_only_browse_and_exact_replay(self):
        before = (await self.client.get("/tasks")).json()
        foreign = await self.app.state.runtime_services.background_task_service.repository.create(
            self.fixture.other, TaskCreate(request_id=uuid4(), title="Foreign fixture", requested_outcome="Private fixture work."),
            model="fixture-model")
        self.assertEqual((await self.client.get("/tasks/" + str(foreign.id))).status_code, 404)
        self.assertEqual((await self.client.get("/attention", params={"task_id": str(foreign.id)})).json(), [])
        result = await self.client.get("/attention", params={"task_id": self.ids["task_id"]})
        self.assertEqual(result.status_code, 200)
        self.assertEqual([r["id"] for r in result.json()], [self.ids["unknown_intent_id"]])
        self.assertEqual((await self.client.get("/attention", params={"task_id": str(uuid4())})).json(), [])
        self.assertEqual((await self.client.get("/attention", headers={"Authorization": "Bearer wrong"})).status_code, 403)
        identity = self.ids["unknown_intent_id"]
        detail = (await self.client.get("/attention/" + identity)).json()
        self.assertEqual(detail["state"], "unknown")
        self.assertIn("mid = (low + high) // 2", detail["expert_result"]["answer"])
        self.assertEqual(detail["attempt_count"], 1)
        body = {"request_id": str(uuid4()), "expected_generation": detail["generation"]}
        first = await self.client.post("/attention/" + identity + "/replay", json=body)
        again = await self.client.post("/attention/" + identity + "/replay", json=body)
        self.assertEqual(first.status_code, 200)
        self.assertTrue(again.json()["replayed_request"])
        self.assertEqual(first.json()["generation"], again.json()["generation"])
        self.assertEqual((await self.client.get("/tasks")).json(), before)
        current = (await self.client.get("/attention/" + identity)).json()
        self.assertEqual(current["state"], "pending")
        self.assertEqual(current["attempt_count"], 1)
        self.assertEqual(current["generation"], detail["generation"] + 1)

    async def test_reply_supersedes_question_and_stale_update_is_rejected(self):
        identity = self.ids["waiting_task_id"]
        task = (await self.client.get("/tasks/" + identity)).json()
        self.assertEqual(task["status"], "waiting_for_input")
        body = {"request_id": str(uuid4()), "expected_revision": task["revision"],
                "message": "Use the synthetic rover and a simulated controller. No real wiring."}
        first = await self.client.post("/tasks/" + identity + "/updates", json=body)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["status"], "queued")
        again = await self.client.post("/tasks/" + identity + "/updates", json=body)
        self.assertEqual(first.json()["revision"], again.json()["revision"])
        stale = await self.client.post("/tasks/" + identity + "/updates", json=body | {"request_id": str(uuid4())})
        self.assertEqual(stale.status_code, 409)
        old = (await self.client.get("/attention/" + self.ids["question_intent_id"])).json()
        self.assertFalse(old["source_available"])
        self.assertIsNone(old["expert_result"])

    async def test_reminder_dismissal_is_not_task_cancellation_and_removed_source_is_hidden(self):
        rows = (await self.client.get("/attention")).json()
        reminder = next(row for row in rows if row["source_namespace"] == "reminder_event")
        async with self.fixture.pool.acquire() as connection:
            previous_status = await connection.fetchval("select status from jarvis_reminders where owner_id=$1", self.fixture.owner)
        response = await self.client.post("/attention/" + reminder["id"] + "/dismiss",
            json={"request_id": str(uuid4()), "expected_generation": reminder["generation"]})
        self.assertEqual(response.status_code, 200)
        visible = (await self.client.get("/attention")).json()
        self.assertNotIn(reminder["id"], [row["id"] for row in visible])
        hidden = (await self.client.get("/attention", params={"include_dismissed": True})).json()
        self.assertIn(reminder["id"], [row["id"] for row in hidden])
        async with self.fixture.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select status from jarvis_reminders where owner_id=$1", self.fixture.owner), previous_status)
        identity = self.ids["unknown_intent_id"]
        self.assertEqual((await self.client.post("/__fixture/revoke/" + identity)).status_code, 200)
        detail = (await self.client.get("/attention/" + identity)).json()
        self.assertFalse(detail["source_available"])
        self.assertIsNone(detail["expert_result"])
        replay = await self.client.post("/attention/" + identity + "/replay",
            json={"request_id": str(uuid4()), "expected_generation": detail["generation"]})
        self.assertEqual(replay.status_code, 409)
        task_id = self.ids["queued_task_id"]
        task = (await self.client.get("/tasks/" + task_id)).json()
        cancelled = await self.client.post("/tasks/" + task_id + "/cancel",
            json={"request_id": str(uuid4()), "expected_revision": task["revision"], "message": ""})
        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(cancelled.json()["status"], "cancelled")
