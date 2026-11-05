"""Historical receipt repair uses exact pointers, not matching private prose."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from app.runtime.feature_config import TaskLimits
from app.tasks.actions import TaskActionRepository
from app.tasks.models import TaskCreate, TaskError, TaskStatus, WorkerOutcome, digest
from app.tasks.receipt_maintenance import ReceiptCopyMaintenance
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


class ReceiptMaintenanceBoundsTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_scope_cursor_and_bounds_fail_before_database_access(self):
        service = ReceiptCopyMaintenance(SimpleNamespace())
        for owner, cursor, limit in (("",None,64),(" bad",None,64),("x"*201,None,64),
                                     ("fixture","not-a-uuid",64),("fixture",None,True),
                                     ("fixture",None,0),("fixture",None,129)):
            with self.subTest(owner=owner, limit=limit), self.assertRaises(ValueError):
                await service.audit_page(owner, after_action_id=cursor, limit=limit)
        with self.assertRaises(ValueError):
            await service.repair_page("fixture", expected_digest="missing")


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain fixtures required",
)
class HistoricalReceiptMaintenanceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id(),
        ))
        self.owner = self.fixture.owner
        self.tasks = TaskRepository(self.fixture.pool, TaskLimits())
        self.actions = TaskActionRepository(self.tasks)
        self.service = ReceiptCopyMaintenance(self.tasks)

    async def legacy_action(self, *, owner=None, status="accepted"):
        owner = owner or self.owner
        secret = "S21-SYNTHETIC-" + uuid4().hex
        parameters = {"body":secret}
        async with self.tasks.transaction(owner) as connection:
            session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", owner)
            request = await self.tasks.payload(connection, owner, {"text":"Authorize the synthetic scope"})
            event = await connection.fetchval("""insert into jarvis_context_events(
                owner_id,session_id,source_namespace,source_event_id,kind,finality,trust,payload_id)
                values($1,$2,'fixture-old-receipt',$3,'user_request','final','user_input',$4) returning id""",
                owner, session, str(uuid4()), request)
            scope = await self.tasks.payload(connection, owner, parameters)
            grant = await connection.fetchval("""insert into jarvis_operation_grants(
                owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                values($1,'notes.create',$2,$3,1,1,$4,clock_timestamp()+interval '1 hour') returning id""",
                owner, str(uuid4()), event, scope)
        task = await self.tasks.create(owner, TaskCreate(
            request_id=uuid4(), title="Historical receipt fixture", requested_outcome="Use the synthetic authorized scope",
            session_id=session, grant_ids=(grant,),
        ), model="fixture-model", source_event_id=event)
        claim = await self.tasks.claim_next("receipt-maintenance-fixture", ("frontier",), owner_id=owner)
        self.assertEqual(claim.task.id, task.id)
        action = await self.actions.prepare(claim, "fixture-action", grant, "notes.create", parameters)
        await self.actions.authorize(claim, action["id"])
        token = await self.actions.begin_submission(claim, action["id"])
        receipt = {"receipt_body":secret, "synthetic":True}
        async def old_writer(connection, owner_id, _action, content):
            return await self.tasks.payload(connection, owner_id, content)
        if status is not None:
            # Reproduce the old writer's lack of a source edge, while keeping
            # the real claimed/submitted action lifecycle and budget handling.
            with patch.object(self.actions, "_receipt_payload", new=old_writer):
                await self.actions.record_receipt(owner, task.id, action["id"], token, status, receipt)
            async with self.tasks.transaction(owner) as connection:
                await connection.execute("update jarvis_task_actions set receipt_digest=null where owner_id=$1 and id=$2",
                                         owner, action["id"])
        await self.tasks.finish(claim, WorkerOutcome(TaskStatus.PAUSED, {"fixture":"The synthetic attempt ended"}))
        current = await self.actions.get(owner, task.id, action["id"])
        return SimpleNamespace(owner=owner, task=task, action_id=action["id"], scope=scope, token=token,
                               receipt_id=current["receipt_payload_id"], receipt=receipt, secret=secret)

    async def body(self, owner, payload):
        async with self.fixture.pool.acquire() as connection:
            return await connection.fetchrow(
                "select availability,content from jarvis_context_payloads where owner_id=$1 and id=$2", owner, payload)

    async def parents(self, owner, payload):
        async with self.fixture.pool.acquire() as connection:
            return set(await connection.fetchval("""select array_agg(source_payload_id)
                from jarvis_context_payload_dependencies where owner_id=$1 and dependent_payload_id=$2""",
                owner, payload) or ())

    async def remove(self, owner, payload):
        async with self.fixture.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2",
                                     owner, payload)

    async def repair(self, page):
        return await self.service.repair_page(page["owner_id"], expected_digest=page["snapshot_digest"],
            after_action_id=UUID(page["after_action_id"]) if page["after_action_id"] else None, limit=page["limit"])

    async def test_audit_is_metadata_only_and_does_not_change_legacy_copy(self):
        old = await self.legacy_action()
        before = await self.actions.get(old.owner, old.task.id, old.action_id)
        page = await self.service.audit_page(self.owner)
        self.assertNotIn(old.secret, json.dumps(page))
        self.assertEqual(page["items"][0]["operation"], "bind")
        self.assertTrue(page["items"][0]["fingerprint_needed"])
        self.assertFalse(page["migration_accepted"])
        self.assertFalse(page["application_release_authorized"])
        self.assertEqual(await self.actions.get(old.owner, old.task.id, old.action_id), before)
        self.assertEqual(await self.parents(old.owner, old.receipt_id), set())

    async def test_repair_binds_exact_pointer_and_repeated_fresh_audit_is_noop(self):
        old = await self.legacy_action()
        page = await self.service.audit_page(self.owner)
        result = await self.repair(page)
        self.assertEqual((result["linked"], result["fingerprints_added"]), (1, 1))
        self.assertEqual(await self.parents(old.owner, old.receipt_id), {old.scope})
        action = await self.actions.get(old.owner, old.task.id, old.action_id)
        self.assertEqual(action["receipt_digest"], digest(old.receipt))
        self.assertEqual(action["submission_token"], old.token)
        with self.assertRaises(TaskError) as raised:
            await self.repair(page)
        self.assertEqual(raised.exception.code, "receipt_audit_stale")
        repeat = await self.repair(await self.service.audit_page(self.owner))
        self.assertEqual((repeat["linked"], repeat["redacted"], repeat["fingerprints_added"], repeat["unchanged"]),
                         (0, 0, 0, 1))
        await self.remove(old.owner, old.scope)
        self.assertIsNone((await self.body(old.owner, old.receipt_id))["content"])

    async def test_deleted_scope_repairs_retained_copy_without_erasing_outcome_identity(self):
        old = await self.legacy_action()
        await self.remove(old.owner, old.scope)
        self.assertIsNotNone((await self.body(old.owner, old.receipt_id))["content"])
        before = await self.actions.get(old.owner, old.task.id, old.action_id)
        result = await self.repair(await self.service.audit_page(self.owner))
        self.assertEqual((result["redacted"], result["fingerprints_added"]), (1, 1))
        after = await self.actions.get(old.owner, old.task.id, old.action_id)
        for key in ("id", "task_id", "status", "submission_token", "parameters_payload_id", "receipt_payload_id"):
            self.assertEqual(after[key], before[key])
        self.assertEqual(after["receipt_digest"], digest(old.receipt))
        self.assertIsNone((await self.body(old.owner, old.receipt_id))["content"])
        await self.actions.record_receipt(old.owner, old.task.id, old.action_id, old.token, "accepted", old.receipt)
        self.assertIsNone((await self.body(old.owner, old.receipt_id))["content"])

    async def test_scope_change_after_audit_refuses_stale_plan_without_partial_repair(self):
        old = await self.legacy_action()
        page = await self.service.audit_page(self.owner)
        await self.remove(old.owner, old.scope)
        with self.assertRaises(TaskError) as raised:
            await self.repair(page)
        self.assertEqual(raised.exception.code, "receipt_audit_stale")
        self.assertIsNone((await self.actions.get(old.owner, old.task.id, old.action_id))["receipt_digest"])
        self.assertIsNotNone((await self.body(old.owner, old.receipt_id))["content"])
        await self.repair(await self.service.audit_page(self.owner))
        self.assertIsNone((await self.body(old.owner, old.receipt_id))["content"])

    async def test_receipt_replacement_after_audit_cannot_repair_the_wrong_payload(self):
        old = await self.legacy_action(status="outcome_unknown")
        page = await self.service.audit_page(self.owner)
        await self.actions.record_receipt(old.owner, old.task.id, old.action_id, old.token,
                                          "accepted", {"synthetic":"Later trusted acknowledgment"})
        current = await self.actions.get(old.owner, old.task.id, old.action_id)
        with self.assertRaises(TaskError) as raised:
            await self.repair(page)
        self.assertEqual(raised.exception.code, "receipt_audit_stale")
        self.assertEqual(await self.actions.get(old.owner, old.task.id, old.action_id), current)
        self.assertEqual(await self.parents(old.owner, old.receipt_id), set())

    async def test_conflicting_fingerprint_prevents_every_write_in_the_page(self):
        first, second = await self.legacy_action(), await self.legacy_action()
        async with self.tasks.transaction(self.owner) as connection:
            await connection.execute("update jarvis_task_actions set receipt_digest=$3 where owner_id=$1 and id=$2",
                                     self.owner, second.action_id, "f" * 64)
        page = await self.service.audit_page(self.owner)
        self.assertTrue(any("receipt_fingerprint_conflict" in item["issues"] for item in page["items"]))
        with self.assertRaises(TaskError) as raised:
            await self.repair(page)
        self.assertEqual(raised.exception.code, "receipt_inventory_unresolved")
        self.assertEqual(await self.parents(self.owner, first.receipt_id), set())
        self.assertIsNone((await self.actions.get(self.owner, first.task.id, first.action_id))["receipt_digest"])

    async def test_unproved_creation_order_is_not_rewritten_or_guessed(self):
        old = await self.legacy_action()
        async with self.tasks.transaction(self.owner) as connection:
            await connection.execute("update jarvis_task_actions set receipt_payload_id=$3 where owner_id=$1 and id=$2",
                                     self.owner, old.action_id, old.scope)
        page = await self.service.audit_page(self.owner)
        self.assertIn("source_copy_order_unproved", page["items"][0]["issues"])
        with self.assertRaises(TaskError):
            await self.repair(page)
        self.assertIsNotNone((await self.body(old.owner, old.scope))["content"])
        self.assertEqual(await self.parents(old.owner, old.scope), set())

    async def test_pagination_and_foreign_owner_cursor_remain_scoped(self):
        originals = [await self.legacy_action() for _ in range(3)]
        foreign = await self.legacy_action(owner=self.fixture.other)
        cursor, seen = None, set()
        while True:
            page = await self.service.audit_page(self.owner, after_action_id=cursor, limit=1)
            self.assertLessEqual(len(page["items"]), 1)
            seen.update(item["action_id"] for item in page["items"])
            await self.repair(page)
            if not page["has_more"]:
                break
            cursor = UUID(page["next_after_action_id"])
        self.assertEqual(seen, {str(old.action_id) for old in originals})
        self.assertEqual(await self.parents(foreign.owner, foreign.receipt_id), set())
        with self.assertRaises(TaskError):
            await self.service.audit_page(self.owner, after_action_id=foreign.action_id)
        own = await self.service.audit_page(self.owner)
        with self.assertRaises(TaskError):
            await self.service.repair_page(foreign.owner, expected_digest=own["snapshot_digest"])

    async def test_removed_body_and_missing_receipt_do_not_invent_fingerprints(self):
        removed = await self.legacy_action()
        missing = await self.legacy_action(status=None)
        await self.remove(removed.owner, removed.receipt_id)
        result = await self.repair(await self.service.audit_page(self.owner))
        self.assertEqual((result["already_removed"], result["no_receipt"], result["fingerprints_added"]), (1, 1, 0))
        for old in (removed, missing):
            self.assertIsNone((await self.actions.get(old.owner, old.task.id, old.action_id))["receipt_digest"])

    async def test_unbound_old_history_is_reported_not_matched_by_prose(self):
        old = await self.legacy_action(status="outcome_unknown")
        await self.actions.record_receipt(old.owner, old.task.id, old.action_id, old.token,
                                          "accepted", {"receipt_body":old.secret, "synthetic":"Current receipt"})
        history = await self.service.unbound_history_page(self.owner, limit=1)
        self.assertEqual(len(history["items"]), 1)
        self.assertEqual(history["items"][0]["payload_id"], str(old.receipt_id))
        self.assertEqual(history["items"][0]["event_type"], "action_outcome_unknown")
        self.assertNotIn(old.secret, json.dumps(history))
        self.assertFalse(history["migration_accepted"])
        await self.repair(await self.service.audit_page(self.owner))
        self.assertIsNotNone((await self.body(self.owner, old.receipt_id))["content"])
        self.assertEqual(await self.parents(self.owner, old.receipt_id), set())
        foreign = await self.legacy_action(owner=self.fixture.other)
        foreign_event = (await self.tasks.events(foreign.owner, foreign.task.id))[0]["id"]
        with self.assertRaises(TaskError):
            await self.service.unbound_history_page(self.owner, after_event_id=foreign_event)
