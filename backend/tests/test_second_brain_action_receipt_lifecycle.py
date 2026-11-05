"""Receipt content removal must not reset an already submitted effect."""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.runtime.feature_config import TaskLimits
from app.tasks.actions import TaskActionRepository
from app.tasks.models import ClaimLost, TaskCreate, TaskError, digest, json_object
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain fixtures required",
)
class ActionReceiptLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id(),
        ))
        self.owner = self.fixture.owner
        self.tasks = TaskRepository(self.fixture.pool, TaskLimits())
        self.actions = TaskActionRepository(self.tasks)
        self.secret = "S20-SYNTHETIC-" + uuid4().hex
        parameters = {"body":self.secret}
        async with self.tasks.transaction(self.owner) as connection:
            session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.owner)
            request = await self.tasks.payload(connection, self.owner, {"text":"Approve the exact synthetic effect."})
            event = await connection.fetchval("""insert into jarvis_context_events(
                owner_id,session_id,source_namespace,source_event_id,kind,finality,trust,payload_id)
                values($1,$2,'fixture-receipt',$3,'user_request','final','user_input',$4) returning id""",
                self.owner, session, str(uuid4()), request)
            scope = await self.tasks.payload(connection, self.owner, parameters)
            grant = await connection.fetchval("""insert into jarvis_operation_grants(
                owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                values($1,'notes.create',$2,$3,1,1,$4,clock_timestamp()+interval '1 hour') returning id""",
                self.owner, "fixture-grant-" + uuid4().hex, event, scope)
        task = await self.tasks.create(self.owner, TaskCreate(
            request_id=uuid4(), title="Receipt fixture", requested_outcome="Perform the exact synthetic operation.",
            session_id=session, grant_ids=(grant,),
        ), model="fixture-model", source_event_id=event)
        claim = await self.tasks.claim_next("receipt-fixture-worker", ("frontier",))
        self.assertEqual(claim.task.id, task.id)
        action = await self.actions.prepare(claim, "fixture-effect", grant, "notes.create", parameters)
        await self.actions.authorize(claim, action["id"])
        token = await self.actions.begin_submission(claim, action["id"])
        self.run = SimpleNamespace(task=task, claim=claim, action_id=action["id"], scope=scope, token=token)
        self.receipt = {"tool":"fixture_effect", "parameters_digest":digest(parameters),
                        "result":{"status":"success", "body":self.secret}}

    async def record(self, status="accepted", receipt=None):
        await self.actions.record_receipt(self.owner, self.run.task.id, self.run.action_id,
                                          self.run.token, status, self.receipt if receipt is None else receipt)
        return await self.actions.get(self.owner, self.run.task.id, self.run.action_id)

    async def remove(self, payload, *, state="deleted"):
        async with self.fixture.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_context_payloads set availability=$3,content=null where owner_id=$1 and id=$2",
                self.owner, payload, state,
            )

    async def body(self, payload):
        async with self.fixture.pool.acquire() as connection:
            return await connection.fetchrow(
                "select availability,content from jarvis_context_payloads where owner_id=$1 and id=$2",
                self.owner, payload,
            )

    async def parents(self, payload):
        async with self.fixture.pool.acquire() as connection:
            return set(await connection.fetchval(
                "select array_agg(source_payload_id) from jarvis_context_payload_dependencies "
                "where owner_id=$1 and dependent_payload_id=$2", self.owner, payload,
            ) or ())

    async def assert_no_receipt_text(self):
        async with self.fixture.pool.acquire() as connection:
            self.assertFalse(await connection.fetchval(
                "select exists(select 1 from jarvis_context_payloads where owner_id=$1 and position($2 in content::text)>0)",
                self.owner, self.secret,
            ))

    async def test_parameters_reuse_scope_and_receipt_history_reuses_one_bound_body(self):
        action = await self.record()
        self.assertEqual(action["parameters_payload_id"], self.run.scope)
        self.assertNotEqual(action["receipt_payload_id"], self.run.scope)
        self.assertEqual(action["receipt_digest"], digest(self.receipt))
        self.assertEqual(await self.parents(action["receipt_payload_id"]), {self.run.scope})
        event = (await self.tasks.events(self.owner, self.run.task.id))[-1]
        self.assertEqual(event["event_type"], "action_accepted")
        self.assertEqual(event["payload_id"], action["receipt_payload_id"])
        self.assertEqual(json_object((await self.body(action["receipt_payload_id"]))["content"]), self.receipt)

    async def test_scope_removal_erases_receipt_text_but_preserves_outcome_token_and_fingerprint(self):
        before = await self.record()
        await self.remove(self.run.scope)
        after = await self.actions.get(self.owner, self.run.task.id, self.run.action_id)
        for key in ("id", "operation_key", "status", "submission_token", "receipt_digest", "receipt_payload_id"):
            self.assertEqual(after[key], before[key])
        self.assertIsNone((await self.body(after["receipt_payload_id"]))["content"])
        self.assertIsNone((await self.tasks.events(self.owner, self.run.task.id))[-1]["content"])
        await self.assert_no_receipt_text()

    async def test_late_accepted_receipt_records_outcome_without_inserting_removed_text(self):
        await self.remove(self.run.scope)
        action = await self.record()
        self.assertEqual(action["status"], "accepted")
        self.assertEqual(action["submission_token"], self.run.token)
        self.assertEqual(action["receipt_digest"], digest(self.receipt))
        body = await self.body(action["receipt_payload_id"])
        self.assertEqual(body["availability"], "unavailable")
        self.assertIsNone(body["content"])
        self.assertEqual(await self.parents(action["receipt_payload_id"]), set())
        await self.assert_no_receipt_text()
        async with self.fixture.pool.acquire() as connection:
            reservation = await connection.fetchrow(
                "select status,actual_tool_steps from jarvis_task_reservations where owner_id=$1 and task_id=$2 and operation_key=$3",
                self.owner, self.run.task.id, "action:" + str(self.run.action_id),
            )
        self.assertEqual(tuple(reservation.values()), ("settled", 1))

    async def test_identical_late_acknowledgments_are_idempotent_after_erasure(self):
        first = await self.record()
        await self.remove(self.run.scope)
        before = await self.tasks.events(self.owner, self.run.task.id)
        results = await asyncio.gather(self.record(), self.record())
        self.assertTrue(all(action["receipt_payload_id"] == first["receipt_payload_id"] for action in results))
        self.assertEqual(await self.tasks.events(self.owner, self.run.task.id), before)
        await self.assert_no_receipt_text()
        with self.assertRaises((TaskError, ClaimLost)):
            await self.actions.begin_submission(self.run.claim, self.run.action_id)

    async def test_changed_receipt_cannot_replace_erased_accepted_outcome(self):
        first = await self.record()
        await self.remove(self.run.scope)
        with self.assertRaises(TaskError) as raised:
            await self.record(receipt={"different":"unconfirmed outcome"})
        self.assertEqual(raised.exception.code, "task_action_receipt_conflict")
        after = await self.actions.get(self.owner, self.run.task.id, self.run.action_id)
        self.assertEqual(after["receipt_digest"], first["receipt_digest"])
        self.assertEqual(after["status"], "accepted")
        await self.assert_no_receipt_text()

    async def test_lost_ack_does_not_downgrade_accepted_outcome_after_removal(self):
        first = await self.record()
        await self.remove(self.run.scope)
        after = await self.record("outcome_unknown", {"error":"An acknowledgment was lost"})
        self.assertEqual(after["status"], "accepted")
        self.assertEqual(after["receipt_digest"], first["receipt_digest"])
        self.assertEqual(after["receipt_payload_id"], first["receipt_payload_id"])

    async def test_reconciliation_after_removal_keeps_known_outcome_without_proof_text(self):
        unknown = await self.record("outcome_unknown", {"error":self.secret})
        await self.remove(self.run.scope)
        proof = {"method":"Synthetic trusted reconciliation", "private_detail":self.secret}
        await self.actions.reconcile(self.owner, self.run.task.id, self.run.action_id,
                                     outcome="accepted", proof=proof)
        current = await self.actions.get(self.owner, self.run.task.id, self.run.action_id)
        self.assertEqual(current["status"], "reconciled")
        self.assertEqual(current["reconciled_outcome"], "accepted")
        self.assertEqual(current["submission_token"], self.run.token)
        self.assertEqual(current["receipt_digest"], digest(proof))
        self.assertIsNone((await self.body(unknown["receipt_payload_id"]))["content"])
        self.assertIsNone((await self.body(current["receipt_payload_id"]))["content"])
        await self.assert_no_receipt_text()

    async def test_reconciliation_proof_is_bound_when_scope_remains_available(self):
        await self.record("outcome_unknown", {"error":"Synthetic unconfirmed submission"})
        proof = {"method":"Synthetic independent lookup", "private_detail":self.secret}
        await self.actions.reconcile(self.owner, self.run.task.id, self.run.action_id,
                                     outcome="not_submitted", proof=proof)
        current = await self.actions.get(self.owner, self.run.task.id, self.run.action_id)
        self.assertEqual(current["reconciled_outcome"], "not_submitted")
        self.assertEqual(await self.parents(current["receipt_payload_id"]), {self.run.scope})
        await self.remove(self.run.scope)
        await self.assert_no_receipt_text()
        self.assertEqual((await self.actions.get(self.owner, self.run.task.id, self.run.action_id))["reconciled_outcome"], "not_submitted")

    async def test_legacy_retained_body_can_establish_fingerprint_without_new_copy(self):
        first = await self.record()
        async with self.fixture.pool.acquire() as connection:
            await connection.execute("update jarvis_task_actions set receipt_digest=null where owner_id=$1 and id=$2",
                                     self.owner, self.run.action_id)
        updated = await self.record()
        self.assertEqual(updated["receipt_digest"], digest(self.receipt))
        self.assertEqual(updated["receipt_payload_id"], first["receipt_payload_id"])
        await self.remove(self.run.scope)
        await self.record()
        await self.assert_no_receipt_text()

    async def test_legacy_removed_body_without_fingerprint_is_not_fabricated_from_retry(self):
        first = await self.record()
        async with self.fixture.pool.acquire() as connection:
            await connection.execute("update jarvis_task_actions set receipt_digest=null where owner_id=$1 and id=$2",
                                     self.owner, self.run.action_id)
        await self.remove(self.run.scope)
        with self.assertRaises(TaskError):
            await self.record()
        current = await self.actions.get(self.owner, self.run.task.id, self.run.action_id)
        self.assertEqual(current["status"], "accepted")
        self.assertIsNone(current["receipt_digest"])
        self.assertEqual(current["receipt_payload_id"], first["receipt_payload_id"])
        await self.assert_no_receipt_text()

    async def test_deletion_waits_for_receipt_transaction_then_removes_new_copy(self):
        removing = None
        try:
            async with self.tasks.transaction(self.owner) as connection:
                await self.actions.record_receipt_in_transaction(
                    connection, self.owner, self.run.task.id, self.run.action_id,
                    self.run.token, "accepted", self.receipt,
                )
                removing = asyncio.create_task(self.remove(self.run.scope))
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(removing), .1)
            await asyncio.wait_for(removing, 5)
        finally:
            if removing is not None and not removing.done():
                removing.cancel()
                await asyncio.gather(removing, return_exceptions=True)
        action = await self.actions.get(self.owner, self.run.task.id, self.run.action_id)
        self.assertEqual(action["status"], "accepted")
        self.assertIsNone((await self.body(action["receipt_payload_id"]))["content"])
        await self.assert_no_receipt_text()

    async def test_wrong_owner_or_submission_token_cannot_record_a_receipt(self):
        for owner, token in ((self.fixture.other, self.run.token), (self.owner, uuid4())):
            with self.subTest(owner=owner), self.assertRaises(TaskError):
                await self.actions.record_receipt(owner, self.run.task.id, self.run.action_id,
                                                  token, "accepted", self.receipt)
        action = await self.actions.get(self.owner, self.run.task.id, self.run.action_id)
        self.assertEqual(action["status"], "submitting")
        self.assertIsNone(action["receipt_digest"])
        self.assertIsNone(action["receipt_payload_id"])
