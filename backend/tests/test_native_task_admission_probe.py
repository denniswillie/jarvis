"""Durable native admission evidence remains distinct from returned callbacks."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskMutation, TaskVoiceSource
from app.tools.invocation import bind_tool_invocation
from evals.second_brain.native_expert_delivery import (
    ObservedAdmissionRepository, accepted_task_admissions,
)
from evals.second_brain.native_voice_tasks import ObservedExecutor
from evals.second_brain.source_fixture import source_fixture


class AdmissionProofTests(unittest.TestCase):
    def test_a_returned_callback_is_not_a_committed_receipt(self):
        row = {"tool": "update_background_task", "call_id": "callback",
               "result": {"accepted": True}}
        self.assertEqual(accepted_task_admissions([row], "update"), [])

    def test_only_exact_operation_and_call_are_accepted_without_duplicate_receipts(self):
        proof = {"request_id": str(uuid4()), "call_id": "update-call", "committed": True,
                 "basis": "exact_committed_task_request", "operation": "update"}
        row = {"tool": "update_background_task", "call_id": "update-call",
               "durable_admission": proof}
        self.assertEqual(accepted_task_admissions([row, dict(row)], "update"), [row])
        self.assertEqual(accepted_task_admissions([row], "create"), [])
        self.assertEqual(accepted_task_admissions([row], "update", memory_edit=True), [])
        for changes in ({"call_id": "different-call"}, {"tool": "start_background_task"}):
            with self.subTest(changes=changes):
                self.assertEqual(accepted_task_admissions([row | changes], "update"), [])
        for changes in ({"committed": False}, {"basis": "attempted_write"}, {"operation": "create"}):
            with self.subTest(proof_changes=changes):
                self.assertEqual(accepted_task_admissions(
                    [row | {"durable_admission": proof | changes}], "update"), [])


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class NativeAdmissionDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.observations = []
        self.tasks = ObservedAdmissionRepository(self.fixture.pool, TaskLimits(),
                                                observations=self.observations)
        self.history = ContextRepository(self.tasks)
        async with self.fixture.pool.acquire() as db:
            self.session = await db.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.fixture.owner)
        self.original = await self.request("Compare these two synthetic project versions.")
        with bind_tool_invocation(provider_call_id="create-call",
                request_event_id=self.original.id, request_revision=self.original.revision):
            self.task = await self.tasks.create(self.fixture.owner, TaskCreate(
                request_id=uuid4(), title="Admission receipt fixture",
                requested_outcome="Compare two synthetic project versions.",
                session_id=self.session, knowledge_area_id=self.fixture.area),
                model="fixture-model", source=self.source(self.original), mode="consult")
        self.clarification = await self.request("Version one only.")
        self.update = TaskMutation(request_id=uuid4(), expected_revision=self.task.revision,
                                   message="Use version one only.")

    def source(self, event):
        return TaskVoiceSource(event.id, event.revision, self.session)

    async def request(self, text):
        return await self.history.append(self.fixture.owner, ContextIngress(
            session_id=self.session, source_namespace="voice:native-admission-fixture",
            source_event_id=uuid4().hex, receipt_id=uuid4().hex,
            kind="user_transcript", trust="user_input", payload={"text": text}))

    async def update_task(self, *, call_id="update-call"):
        with bind_tool_invocation(provider_call_id=call_id,
                request_event_id=self.clarification.id,
                request_revision=self.clarification.revision):
            return await self.tasks.mutate(self.fixture.owner, self.task.id, "update",
                                           self.update, source=self.source(self.clarification))

    def tool_row(self):
        return {"tool": "update_background_task", "call_id": "update-call"}

    async def test_committed_update_survives_cancelled_callback_without_fabricating_result(self):
        async def execute(name, arguments, **kwargs):
            await self.update_task(call_id=kwargs["provider_call_id"])
            raise asyncio.CancelledError()
        rows = []
        observer = ObservedExecutor(SimpleNamespace(execute=execute), rows, time.monotonic())
        with self.assertRaises(asyncio.CancelledError):
            await observer.execute("update_background_task", {"task_ref": self.tasks.reference(self.task.id)},
                                   provider_call_id="update-call")
        self.assertEqual(len(rows), 1)
        self.assertNotIn("result", rows[0])
        self.assertEqual(accepted_task_admissions(rows, "update"), [])
        await self.tasks.refresh_admissions(self.fixture.owner, self.session, rows)
        self.assertEqual(accepted_task_admissions(rows, "update"), rows)
        proof = rows[0]["durable_admission"]
        self.assertEqual(proof["request_id"], str(self.update.request_id))
        self.assertEqual(proof["task_id"], str(self.task.id))
        self.assertEqual(proof["accepted_revision"], self.task.revision + 1)
        self.assertEqual(proof["voice_source"], self.source(self.clarification).parameters())
        self.assertNotIn("result", rows[0])
        self.assertEqual((await self.tasks.get(self.fixture.owner, self.task.id)).revision, 2)
        await self.tasks.refresh_admissions(self.fixture.owner, self.session, rows)
        self.assertEqual(len(accepted_task_admissions(rows, "update")), 1)

    async def test_rollback_does_not_turn_observed_write_into_acceptance(self):
        actual = self.tasks._save_request
        async def fail_after_observation(*args, **kwargs):
            await actual(*args, **kwargs)
            raise RuntimeError("fixture rollback after observation")
        with patch.object(self.tasks, "_save_request", fail_after_observation):
            with self.assertRaisesRegex(RuntimeError, "fixture rollback"):
                await self.update_task()
        attempted = [row for row in self.observations if row["operation"] == "update"]
        self.assertEqual(len(attempted), 1)
        rows = [self.tool_row()]
        await self.tasks.refresh_admissions(self.fixture.owner, self.session, rows)
        self.assertEqual(accepted_task_admissions(rows, "update"), [])
        self.assertNotIn("durable_admission", rows[0])
        self.assertFalse(attempted[0]["committed"])
        self.assertEqual((await self.tasks.get(self.fixture.owner, self.task.id)).revision, 1)
        async with self.fixture.pool.acquire() as db:
            self.assertFalse(await db.fetchval(
                "select exists(select 1 from jarvis_task_requests where owner_id=$1 and request_id=$2)",
                self.fixture.owner, self.update.request_id))

    async def test_owner_session_call_and_exact_receipt_identity_must_match(self):
        await self.update_task()
        original = dict(next(row for row in self.observations if row["operation"] == "update"))
        other_event = await self.request("A different valid request in the same session.")
        changes = (
            {"owner_id": "another-fixture-owner"},
            {"request_id": str(uuid4())},
            {"task_id": str(uuid4())},
            {"operation": "create"},
            {"parameters_digest": "not-the-committed-digest"},
            {"accepted_revision": original["accepted_revision"] + 1},
            {"call_id": "another-call"},
            {"request_event_id": str(other_event.id)},
            {"request_revision": original["request_revision"] + 1},
            {"voice_source": self.source(other_event).parameters()},
        )
        for change in changes:
            with self.subTest(change=change):
                self.observations[:] = [original | change]
                rows = [self.tool_row()]
                await self.tasks.refresh_admissions(self.fixture.owner, self.session, rows)
                self.assertEqual(accepted_task_admissions(rows, "update"), [])
                self.assertNotIn("durable_admission", rows[0])
        self.observations[:] = [dict(original)]
        for owner, session in (("another-fixture-owner", self.session), (self.fixture.owner, uuid4())):
            with self.subTest(owner=owner, session=session):
                rows = [self.tool_row()]
                await self.tasks.refresh_admissions(owner, session, rows)
                self.assertEqual(accepted_task_admissions(rows, "update"), [])
        rows = [self.tool_row()]
        await self.tasks.refresh_admissions(self.fixture.owner, self.session, rows)
        self.assertEqual(accepted_task_admissions(rows, "update"), rows)

    async def test_create_receipt_matches_without_claiming_memory_edit_authority(self):
        rows = [{"tool": "start_background_task", "call_id": "create-call"}]
        await self.tasks.refresh_admissions(self.fixture.owner, self.session, rows)
        self.assertEqual(accepted_task_admissions(rows, "create"), rows)
        self.assertEqual(accepted_task_admissions(rows, "create", memory_edit=True), [])
        self.assertNotIn("result", rows[0])
