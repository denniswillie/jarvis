import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import asyncpg
import httpx

from app.migrations.runner import apply_migrations, quote_schema
from app.runtime.feature_config import SecondBrainSettings, TaskLimits
from app.settings import JarvisSettings, get_settings
from app.tasks.actions import TaskActionRepository
from app.tasks.budgets import TaskBudgetRepository
from app.tasks.models import BudgetExceeded, ClaimLost, Reconciliation, TaskCreate, TaskError, TaskMutation, TaskStatus, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService, initialize_task_runtime
from evals.second_brain.fixtures import DatabaseFixture


class _Provider:
    model = "fixture-only"

    async def execute(self, context):
        return WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "Fixture result."})

    async def reconcile(self, context, *, cancel):
        return Reconciliation("unknown")


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit second-brain database fixture configuration required")
class TaskDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        self.fixture = DatabaseFixture(self.dsn, Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id())
        self.addAsyncCleanup(self.fixture.cleanup)
        connection = await asyncpg.connect(self.dsn, timeout=10, command_timeout=90)
        try:
            self.schema = await self.fixture.create_schema(connection)
            await apply_migrations(connection, schema=self.schema, fresh=True)
        finally:
            await connection.close()
        async def setup_fixture_connection(connection):
            await connection.execute(f'set search_path to {quote_schema(self.schema)},pg_catalog,public,extensions')
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=2, timeout=10, command_timeout=30,
            setup=setup_fixture_connection)
        self.addAsyncCleanup(self.pool.close)
        self.owner, self.other = self.fixture.owners
        self.limits = TaskLimits(owner_concurrency=1, global_concurrency=2, lease_seconds=30, poll_seconds=0.05)
        self.repository = TaskRepository(self.pool, self.limits)
        self.budgets = TaskBudgetRepository(self.repository)
        self.actions = TaskActionRepository(self.repository)

    def request(self, **changes):
        return TaskCreate(**({"request_id": uuid4(), "title": "Fixture", "requested_outcome": "Synthetic task, not personal knowledge."} | changes))

    async def create(self, *, owner=None, request=None):
        return await self.repository.create(owner or self.owner, request or self.request(), model="fixture-only")

    async def claim(self):
        result = await self.repository.claim_next("fixture-worker", ("frontier",))
        self.assertIsNotNone(result)
        return result

    async def mutation(self, task, operation, message="", *, request_id=None):
        return await self.repository.mutate(task.owner_id, task.id, operation,
            TaskMutation(request_id=request_id or uuid4(), expected_revision=task.revision, message=message))

    async def expire(self, claim):
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_task_attempts set lease_until=clock_timestamp()-interval '1 second' where owner_id=$1 and id=$2", claim.task.owner_id, claim.attempt_id)

    async def grant(self, *, scope=None):
        scope = scope or {"recipient": "fixture@example.invalid", "artifact": "synthetic-only"}
        async with self.repository.transaction(self.owner) as connection:
            event = await self.repository._request_event(connection, self.owner, uuid4(), {"literal_request": "Fixture-only action test; no real email."})
            payload = await self.repository.payload(connection, self.owner, scope)
            identity = await connection.fetchval("""insert into jarvis_operation_grants(owner_id,capability,operation_key,request_event_id,
                request_revision,scope_revision,scope_payload_id,expires_at) values($1,'fixture.send','fixture-send',$2,1,1,$3,
                clock_timestamp()+interval '1 hour') returning id""", self.owner, event["id"], payload)
        return identity, scope

    async def test_duplicate_request_replays_but_repeated_new_request_creates_work(self):
        request = self.request()
        first, replay = await asyncio.gather(self.create(request=request), self.create(request=request))
        self.assertEqual(first.id, replay.id)
        second = await self.create(request=self.request(title=request.title, requested_outcome=request.requested_outcome))
        self.assertNotEqual(first.id, second.id)
        with self.assertRaises(TaskError):
            await self.create(request=request.model_copy(update={"requested_outcome": "Different"}))
        self.assertEqual([event["event_type"] for event in await self.repository.events(self.owner, first.id)], ["accepted"])

    async def test_worker_concurrency_fairness_and_owner_limits(self):
        first = await self.create()
        await self.create()
        foreign = await self.create(owner=self.other)
        claims = await asyncio.gather(self.repository.claim_next("worker-a", ("frontier",)), self.repository.claim_next("worker-b", ("frontier",)))
        self.assertEqual({claim.task.id for claim in claims}, {first.id, foreign.id})
        self.assertIsNone(await self.repository.claim_next("worker-c", ("frontier",)))
        await self.repository.finish(claims[0], WorkerOutcome(TaskStatus.SUCCEEDED, {"done": True}))
        next_claim = await self.repository.claim_next("worker-c", ("frontier",))
        if claims[0].task.owner_id == self.owner:
            self.assertIsNotNone(next_claim)

    async def test_shared_limits_prevent_a_differently_configured_worker_from_oversubscribing(self):
        await self.create(); await self.create(); await self.create(owner=self.other)
        first = await self.claim()
        larger = TaskRepository(self.pool, self.limits.model_copy(update={"global_concurrency": 8, "owner_concurrency": 8}))
        second = await larger.claim_next("larger-config", ("frontier",))
        self.assertEqual(second.task.owner_id, self.other)
        self.assertIsNone(await larger.claim_next("larger-config", ("frontier",)))
        lower = TaskRepository(self.pool, self.limits.model_copy(update={"global_concurrency": 1}))
        self.assertEqual((await lower.configure_scheduler_limits())["active_attempts"], 2)
        await self.repository.finish(first, WorkerOutcome(TaskStatus.SUCCEEDED, {"done": True}))
        self.assertIsNone(await larger.claim_next("larger-config", ("frontier",)))
        await self.repository.finish(second, WorkerOutcome(TaskStatus.SUCCEEDED, {"done": True}))
        self.assertIsNotNone(await larger.claim_next("larger-config", ("frontier",)))

    async def test_expired_fences_cannot_renew_checkpoint_or_publish(self):
        task = await self.create()
        old = await self.claim()
        await self.expire(old)
        recovery = await self.repository.claim_recovery("restart")
        with self.assertRaises(ClaimLost):
            await self.repository.renew(old)
        with self.assertRaises(ClaimLost):
            await self.repository.checkpoint(old, {"bad": "stale checkpoint"})
        self.assertFalse(await self.repository.finish(old, WorkerOutcome(TaskStatus.SUCCEEDED, {"bad": "stale"})))
        await self.repository.resolve_recovery(recovery, Reconciliation("not_started"))
        fresh = await self.claim()
        self.assertNotEqual(fresh.attempt_id, old.attempt_id)
        await self.repository.finish(fresh, WorkerOutcome(TaskStatus.SUCCEEDED, {"good": "current"}))
        self.assertEqual((await self.repository.get(self.owner, task.id)).status, TaskStatus.SUCCEEDED)

    async def test_running_update_fences_old_result_then_requeues_new_revision(self):
        task = await self.create()
        claim = await self.claim()
        update_id = uuid4()
        updated = await self.mutation(task, "update", "A corrected task outcome.", request_id=update_id)
        self.assertEqual(updated.status, TaskStatus.PAUSED)
        self.assertEqual(updated.revision, 2)
        replay = await self.repository.mutate(self.owner, task.id, "update", TaskMutation(request_id=update_id, expected_revision=1, message="A corrected task outcome."))
        self.assertEqual(replay.revision, 2)
        self.assertFalse(await self.repository.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED, {"wrong": "old answer"})))
        recovery = await self.repository.claim_recovery("updater")
        await self.repository.resolve_recovery(recovery, Reconciliation("not_started"))
        fresh = await self.claim()
        self.assertEqual(fresh.task.requested_outcome, "A corrected task outcome.")
        self.assertEqual(fresh.task_revision, 2)
        self.assertIsNone(fresh.task.pending_status)

    async def test_deleted_area_blocks_publication_including_event_payload(self):
        async with self.pool.acquire() as connection:
            area = await connection.fetchval("insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Fixture area','testing') returning id", self.owner)
        task = await self.create(request=self.request(knowledge_area_id=area))
        claim = await self.claim()
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_knowledge_areas set status='deleted',revision=revision+1 where owner_id=$1 and id=$2", self.owner, area)
        await self.repository.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED, {"private_answer": "must-not-publish"}))
        current = await self.repository.get(self.owner, task.id)
        self.assertEqual(current.status, TaskStatus.PAUSED)
        self.assertIsNone(current.result_payload_id)
        events = await self.repository.events(self.owner, task.id)
        self.assertNotIn("must-not-publish", json.dumps(events, default=str))

    async def test_cancel_targets_only_one_task_and_blocks_old_worker(self):
        task = await self.create()
        other_task = await self.create()
        claim = await self.claim()
        cancelled = await self.mutation(task, "cancel")
        self.assertEqual(cancelled.status, TaskStatus.CANCEL_REQUESTED)
        with self.assertRaises(ClaimLost):
            await self.repository.renew(claim)
        recovery = await self.repository.claim_recovery("canceller")
        await self.repository.resolve_recovery(recovery, Reconciliation("not_started"))
        self.assertEqual((await self.repository.get(self.owner, task.id)).status, TaskStatus.CANCELLED)
        self.assertEqual((await self.repository.get(self.owner, other_task.id)).status, TaskStatus.QUEUED)
        with self.assertRaises(TaskError):
            await self.repository.get(self.other, task.id)
        with self.assertRaises(TaskError):
            await self.repository.mutate(self.owner, other_task.id, "cancel", TaskMutation(request_id=uuid4(), expected_revision=99))

    async def test_missing_request_payload_and_revoked_grant_stop_claim_or_renewal(self):
        task = await self.create()
        async with self.pool.acquire() as connection:
            await connection.execute("""update jarvis_context_payloads set content=null,availability='deleted'
                where owner_id=$1 and id=(select payload_id from jarvis_context_events where owner_id=$1 and id=$2)""", self.owner, task.request_event_id)
        self.assertIsNone(await self.repository.claim_next("fixture", ("frontier",)))
        self.assertEqual((await self.repository.get(self.owner, task.id)).status, TaskStatus.PAUSED)
        grant_id, _ = await self.grant()
        guarded = await self.create(request=self.request(grant_ids=(grant_id,)))
        claim = await self.claim()
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_operation_grants set status='revoked' where owner_id=$1 and id=$2", self.owner, grant_id)
        self.assertFalse(await self.repository.renew(claim))
        self.assertEqual((await self.repository.get(self.owner, guarded.id)).status, TaskStatus.PAUSED)

    async def test_session_close_does_not_cancel_durable_task(self):
        async with self.pool.acquire() as connection:
            session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.owner)
        task = await self.create(request=self.request(session_id=session))
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_sessions set status='closed' where id=$1", session)
        claim = await self.claim()
        await self.repository.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED, {"done": True}))
        self.assertEqual((await self.repository.get(self.owner, task.id)).status, TaskStatus.SUCCEEDED)

    async def test_live_handle_is_reconciled_instead_of_resubmitted(self):
        task = await self.create()
        claim = await self.claim()
        reservation = await self.budgets.reserve(claim, "provider-op", input_tokens=10, output_tokens=10)
        await self.budgets.submit(claim, reservation["id"], provider_dispatch=True)
        await self.repository.checkpoint(claim, {"phase": "accepted"}, provider_handle="fixture-live-handle")
        await self.expire(claim)
        recovery = await self.repository.claim_recovery("new-worker")
        self.assertEqual(recovery.provider_handle, "fixture-live-handle")
        await self.repository.resolve_recovery(recovery, Reconciliation("running"))
        self.assertIsNone(await self.repository.claim_next("no-duplicate", ("frontier",)))
        self.assertEqual((await self.repository.get(self.owner, task.id)).attempt_count, 1)
        await self.expire(recovery)
        recovery = await self.repository.claim_recovery("new-worker")
        await self.repository.resolve_recovery(recovery, Reconciliation("terminal", WorkerOutcome(TaskStatus.SUCCEEDED, {"receipt": "same-handle"})))
        self.assertEqual((await self.repository.get(self.owner, task.id)).status, TaskStatus.SUCCEEDED)

    async def test_dispatch_without_a_handle_stays_unknown_and_cannot_resume(self):
        task = await self.create()
        claim = await self.claim()
        reservation = await self.budgets.reserve(claim, "ambiguous", input_tokens=20, output_tokens=30)
        await self.budgets.submit(claim, reservation["id"], provider_dispatch=True)
        await self.repository.abandon(claim)
        unknown = await self.repository.get(self.owner, task.id)
        self.assertEqual(unknown.status, TaskStatus.UNKNOWN)
        with self.assertRaises(TaskError):
            await self.mutation(unknown, "resume")
        self.assertEqual(await self.budgets.usage(self.owner, task.id), {"input": 20, "output": 30, "steps": 0})
        recovery = await self.repository.claim_recovery("restart")
        await self.repository.resolve_recovery(recovery, Reconciliation("unknown"))
        self.assertEqual((await self.repository.get(self.owner, task.id)).attempt_count, 1)

    async def test_proved_not_started_releases_only_the_provider_reservation(self):
        task = await self.create(); claim = await self.claim()
        reservation = await self.budgets.reserve(claim, "provider", input_tokens=10, output_tokens=20)
        await self.budgets.submit(claim, reservation["id"], provider_dispatch=True)
        await self.repository.abandon(claim)
        recovery = await self.repository.claim_recovery("native-proof")
        await self.repository.resolve_recovery(recovery, Reconciliation("not_started"))
        self.assertEqual(await self.budgets.usage(self.owner, task.id), {"input": 0, "output": 0, "steps": 0})
        fresh = await self.claim()
        reservation2 = await self.budgets.reserve(fresh, "provider", input_tokens=10, output_tokens=20)
        self.assertEqual(reservation2["id"], reservation["id"])
        self.assertEqual(reservation2["attempt_id"], fresh.attempt_id)

    async def test_unknown_provider_can_recover_into_a_durable_clarification(self):
        task = await self.create(); claim = await self.claim()
        reservation = await self.budgets.reserve(claim, "provider", input_tokens=1)
        await self.budgets.submit(claim, reservation["id"], provider_dispatch=True)
        await self.repository.abandon(claim)
        recovery = await self.repository.claim_recovery("recovery")
        await self.repository.resolve_recovery(recovery, Reconciliation("terminal", WorkerOutcome(TaskStatus.WAITING, {"question": "Which fixture?"})))
        current = await self.repository.get(self.owner, task.id)
        self.assertEqual(current.status, TaskStatus.WAITING)
        self.assertIsNotNone(current.result_payload_id)

    async def test_budget_is_retry_stable_and_known_usage_settles_once(self):
        limits = self.limits.model_copy(update={"task_input_tokens": 10, "task_output_tokens": 10, "request_output_tokens": 10, "owner_daily_tokens": 20, "tool_steps": 2})
        self.repository = TaskRepository(self.pool, limits)
        self.budgets = TaskBudgetRepository(self.repository)
        task = await self.create()
        claim = await self.claim()
        first, replay = await asyncio.gather(self.budgets.reserve(claim, "same", input_tokens=8, output_tokens=8, tool_steps=1),
                                            self.budgets.reserve(claim, "same", input_tokens=8, output_tokens=8, tool_steps=1))
        self.assertEqual(first["id"], replay["id"])
        with self.assertRaises(BudgetExceeded):
            await self.budgets.reserve(claim, "too-much", input_tokens=3)
        await self.budgets.submit(claim, first["id"])
        with self.assertRaises(TaskError):
            await self.budgets.submit(claim, first["id"])
        await self.budgets.settle(self.owner, task.id, claim.attempt_id, first["id"], input_tokens=4, output_tokens=3, tool_steps=1)
        await self.budgets.settle(self.owner, task.id, claim.attempt_id, first["id"], input_tokens=4, output_tokens=3, tool_steps=1)
        await self.repository.abandon(claim)
        next_claim = await self.claim()
        self.assertEqual(await self.budgets.usage(self.owner, task.id), {"input": 4, "output": 3, "steps": 1})
        await self.budgets.reserve(next_claim, "next", input_tokens=6, output_tokens=7, tool_steps=1)
        with self.assertRaises(BudgetExceeded):
            await self.budgets.reserve(next_claim, "over-step", tool_steps=1)

    async def test_action_duplicate_ids_parameter_change_and_crash_boundary(self):
        grant_id, scope = await self.grant()
        task = await self.create(request=self.request(grant_ids=(grant_id,)))
        claim = await self.claim()
        first = await self.actions.prepare(claim, "logical-send", grant_id, "fixture.send", scope)
        replay = await self.actions.prepare(claim, "logical-send", grant_id, "fixture.send", scope)
        self.assertEqual(first["id"], replay["id"])
        with self.assertRaises(TaskError):
            await self.actions.prepare(claim, "logical-send", grant_id, "fixture.send", scope | {"recipient": "different.invalid"})
        await self.actions.authorize(claim, first["id"])
        token = await self.actions.begin_submission(claim, first["id"])
        with self.assertRaises(TaskError):
            await self.actions.begin_submission(claim, first["id"])
        await self.repository.abandon(claim)
        self.assertEqual((await self.actions.get(self.owner, task.id, first["id"]))["status"], "outcome_unknown")
        self.assertEqual((await self.repository.get(self.owner, task.id)).status, TaskStatus.UNKNOWN)
        await self.actions.record_receipt(self.owner, task.id, first["id"], token, "accepted", {"provider_receipt": "synthetic-receipt"})
        await self.actions.record_receipt(self.owner, task.id, first["id"], token, "accepted", {"provider_receipt": "synthetic-receipt"})
        self.assertEqual((await self.actions.get(self.owner, task.id, first["id"]))["status"], "accepted")

    async def test_action_submission_charges_tool_budget_and_grant_cannot_fan_out(self):
        self.repository = TaskRepository(self.pool, self.limits.model_copy(update={"tool_steps": 1}))
        self.budgets = TaskBudgetRepository(self.repository); self.actions = TaskActionRepository(self.repository)
        first_grant, scope = await self.grant(); second_grant, _ = await self.grant(scope=scope)
        task = await self.create(request=self.request(grant_ids=(first_grant, second_grant)))
        claim = await self.claim()
        first = await self.actions.prepare(claim, "one", first_grant, "fixture.send", scope)
        await self.actions.authorize(claim, first["id"])
        token = await self.actions.begin_submission(claim, first["id"])
        await self.actions.record_receipt(self.owner, task.id, first["id"], token, "accepted", {"receipt": "one"})
        second = await self.actions.prepare(claim, "two", second_grant, "fixture.send", scope)
        await self.actions.authorize(claim, second["id"])
        with self.assertRaises(BudgetExceeded):
            await self.actions.begin_submission(claim, second["id"])
        await self.actions.revoke_unsubmitted(claim, second["id"])
        self.assertTrue(await self.repository.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED, {"done": True})))
        another = await self.create(request=self.request(grant_ids=(first_grant,)))
        other_claim = await self.claim()
        self.assertEqual(other_claim.task.id, another.id)
        with self.assertRaises(TaskError) as caught:
            await self.actions.prepare(other_claim, "new-provider-call-id", first_grant, "fixture.send", scope)
        self.assertEqual(caught.exception.code, "task_grant_already_bound")

    async def test_late_receipt_completes_requested_cancellation_without_replaying_effect(self):
        grant, scope = await self.grant()
        task = await self.create(request=self.request(grant_ids=(grant,)))
        claim = await self.claim()
        action = await self.actions.prepare(claim, "send", grant, "fixture.send", scope)
        await self.actions.authorize(claim, action["id"])
        token = await self.actions.begin_submission(claim, action["id"])
        await self.mutation(task, "cancel")
        recovery = await self.repository.claim_recovery("cancel-recovery")
        await self.repository.resolve_recovery(recovery, Reconciliation("not_started"))
        self.assertEqual((await self.repository.get(self.owner, task.id)).status, TaskStatus.UNKNOWN)
        await self.actions.record_receipt(self.owner, task.id, action["id"], token, "accepted", {"receipt": "already submitted before cancellation"})
        self.assertEqual((await self.repository.get(self.owner, task.id)).status, TaskStatus.CANCELLED)

    async def test_prepared_action_prevents_false_success_and_explicit_resume_revalidates_it(self):
        grant, scope = await self.grant()
        task = await self.create(request=self.request(grant_ids=(grant,)))
        claim = await self.claim()
        action = await self.actions.prepare(claim, "send", grant, "fixture.send", scope)
        await self.repository.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED, {"incorrect_claim": "sent"}))
        paused = await self.repository.get(self.owner, task.id)
        self.assertEqual(paused.status, TaskStatus.PAUSED)
        self.assertEqual(paused.reason, "unsubmitted_actions")
        await self.mutation(paused, "resume")
        fresh = await self.claim()
        revalidated = await self.actions.prepare(fresh, "send", grant, "fixture.send", scope)
        self.assertEqual(action["id"], revalidated["id"])
        self.assertEqual(revalidated["task_revision"], fresh.task_revision)
        await self.actions.authorize(fresh, action["id"])
        token = await self.actions.begin_submission(fresh, action["id"])
        await self.actions.record_receipt(self.owner, task.id, action["id"], token, "accepted", {"receipt": "fixture"})
        await self.repository.finish(fresh, WorkerOutcome(TaskStatus.SUCCEEDED, {"done": True}))

    async def test_cancel_before_submission_revokes_prepared_action(self):
        grant_id, scope = await self.grant()
        task = await self.create(request=self.request(grant_ids=(grant_id,)))
        claim = await self.claim()
        action = await self.actions.prepare(claim, "logical-send", grant_id, "fixture.send", scope)
        await self.actions.authorize(claim, action["id"])
        await self.mutation(task, "cancel")
        with self.assertRaises(ClaimLost):
            await self.actions.begin_submission(claim, action["id"])
        self.assertEqual((await self.actions.get(self.owner, task.id, action["id"]))["status"], "revoked")

    async def test_cancel_and_update_keep_intent_after_an_uncertain_action_releases_its_attempt(self):
        for operation in ("cancel", "update"):
            with self.subTest(operation=operation):
                grant, scope = await self.grant()
                task = await self.create(request=self.request(grant_ids=(grant,)))
                claim = await self.claim()
                action = await self.actions.prepare(claim, "send", grant, "fixture.send", scope)
                await self.actions.authorize(claim, action["id"])
                token = await self.actions.begin_submission(claim, action["id"])
                await self.repository.abandon(claim)
                unknown = await self.repository.get(self.owner, task.id)
                self.assertIsNone(unknown.current_attempt_id)
                changed = await self.mutation(unknown, operation, "Corrected fixture task")
                self.assertEqual(changed.status, TaskStatus.UNKNOWN)
                self.assertEqual(changed.pending_status, "cancelled" if operation == "cancel" else "queued")
                await self.actions.record_receipt(self.owner, task.id, action["id"], token, "accepted", {"receipt": operation})
                current = await self.repository.get(self.owner, task.id)
                self.assertEqual(current.status, TaskStatus.CANCELLED if operation == "cancel" else TaskStatus.QUEUED)

    async def test_recovery_rechecks_revoked_context_and_active_time_budget(self):
        for cause in ("revoked", "exhausted"):
            with self.subTest(cause=cause):
                grant, _ = await self.grant()
                task = await self.create(request=self.request(grant_ids=(grant,)))
                claim = await self.claim()
                reserved = await self.budgets.reserve(claim, "remote", input_tokens=1)
                await self.budgets.submit(claim, reserved["id"], provider_dispatch=True)
                await self.repository.checkpoint(claim, {}, provider_handle="fixture-existing-handle")
                async with self.pool.acquire() as connection:
                    if cause == "revoked":
                        await connection.execute("update jarvis_operation_grants set status='revoked' where owner_id=$1 and id=$2", self.owner, grant)
                    else:
                        await connection.execute("update jarvis_background_tasks set active_seconds=$3 where owner_id=$1 and id=$2", self.owner, task.id, task.budget.active_seconds)
                await self.expire(claim)
                recovery = await self.repository.claim_recovery("restart")
                self.assertEqual(recovery.task.pending_status, "paused")
                calls = []
                class RecoveredProvider(_Provider):
                    async def reconcile(self, context, *, cancel):
                        calls.append(cancel)
                        return Reconciliation("terminal", WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "must-not-publish"}))
                await TaskRunner(self.repository, {"frontier": RecoveredProvider()})._execute(recovery)
                self.assertEqual(calls, [True])
                current = await self.repository.get(self.owner, task.id)
                self.assertEqual(current.status, TaskStatus.PAUSED)
                self.assertIsNone(current.result_payload_id)
                if cause == "revoked":
                    self.assertNotIn("must-not-publish", json.dumps(await self.repository.events(self.owner, task.id), default=str))

    async def test_success_claim_without_effect_receipt_stays_reconcilable(self):
        grant, scope = await self.grant()
        task = await self.create(request=self.request(grant_ids=(grant,)))
        claim = await self.claim()
        action = await self.actions.prepare(claim, "send", grant, "fixture.send", scope)
        await self.actions.authorize(claim, action["id"])
        await self.actions.begin_submission(claim, action["id"])
        await self.repository.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED, {"unsupported_claim": "I definitely sent it"}))
        self.assertEqual((await self.repository.get(self.owner, task.id)).status, TaskStatus.UNKNOWN)
        self.assertEqual((await self.actions.get(self.owner, task.id, action["id"]))["status"], "outcome_unknown")
        self.assertNotIn("I definitely sent it", json.dumps(await self.repository.events(self.owner, task.id), default=str))
        await self.actions.reconcile(self.owner, task.id, action["id"], outcome="not_submitted", proof={"fixture_native_lookup": "absent"})
        self.assertEqual((await self.repository.get(self.owner, task.id)).status, TaskStatus.PAUSED)

    async def test_cancel_waits_for_both_provider_and_action_reconciliation(self):
        grant, scope = await self.grant()
        task = await self.create(request=self.request(grant_ids=(grant,)))
        claim = await self.claim()
        reservation = await self.budgets.reserve(claim, "provider", input_tokens=1)
        await self.budgets.submit(claim, reservation["id"], provider_dispatch=True)
        await self.repository.checkpoint(claim, {}, provider_handle="fixture-live")
        action = await self.actions.prepare(claim, "send", grant, "fixture.send", scope)
        await self.actions.authorize(claim, action["id"])
        await self.actions.begin_submission(claim, action["id"])
        await self.mutation(task, "cancel")
        recovery = await self.repository.claim_recovery("restart")
        await self.repository.resolve_recovery(recovery, Reconciliation("terminal", WorkerOutcome(TaskStatus.FAILED, {"provider_cancelled": True})))
        current = await self.repository.get(self.owner, task.id)
        self.assertEqual(current.status, TaskStatus.UNKNOWN)
        self.assertEqual(current.pending_status, "cancelled")
        await self.actions.reconcile(self.owner, task.id, action["id"], outcome="accepted", proof={"fixture_native_receipt": "accepted"})
        self.assertEqual((await self.repository.get(self.owner, task.id)).status, TaskStatus.CANCELLED)

    async def test_attempt_and_active_time_limits_survive_retries(self):
        self.repository = TaskRepository(self.pool, self.limits.model_copy(update={"max_attempts": 1}))
        task = await self.create()
        claim = await self.claim()
        await self.repository.abandon(claim)
        self.assertIsNone(await self.repository.claim_next("retry", ("frontier",)))
        self.assertEqual((await self.repository.get(self.owner, task.id)).reason, "task_attempt_budget_reached")
        other = await self.create()
        active = await self.claim()
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_background_tasks set active_seconds=$3 where owner_id=$1 and id=$2", self.owner, other.id, other.budget.active_seconds)
        self.assertFalse(await self.repository.renew(active))
        self.assertEqual((await self.repository.get(self.owner, other.id)).reason, "task_active_budget_reached")

    async def test_daily_reservation_rollover_rechecks_aggregate_before_dispatch(self):
        self.repository = TaskRepository(self.pool, self.limits.model_copy(update={"owner_daily_tokens": 20}))
        self.budgets = TaskBudgetRepository(self.repository)
        await self.create()
        claim = await self.claim()
        yesterday = await self.budgets.reserve(claim, "yesterday", input_tokens=15)
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_task_reservations set accounting_day=accounting_day-1 where id=$1", yesterday["id"])
        await self.budgets.reserve(claim, "today", input_tokens=20)
        with self.assertRaises(BudgetExceeded) as caught:
            await self.budgets.submit(claim, yesterday["id"], provider_dispatch=True)
        self.assertEqual(caught.exception.code, "task_owner_daily_budget_reached")

    async def test_only_provider_usage_can_end_execution_and_actual_overruns_pause(self):
        self.repository = TaskRepository(self.pool, self.limits.model_copy(update={"task_input_tokens": 10}))
        self.budgets = TaskBudgetRepository(self.repository)
        task = await self.create()
        claim = await self.claim()
        reservation = await self.budgets.reserve(claim, "local", input_tokens=5)
        await self.budgets.submit(claim, reservation["id"])
        with self.assertRaises(TaskError) as caught:
            await self.budgets.settle(self.owner, task.id, claim.attempt_id, reservation["id"], input_tokens=5, output_tokens=0, tool_steps=0, provider_terminal=True)
        self.assertEqual(caught.exception.code, "task_usage_not_provider")
        await self.budgets.settle(self.owner, task.id, claim.attempt_id, reservation["id"], input_tokens=11, output_tokens=0, tool_steps=0)
        self.assertTrue(await self.repository.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED, {"result": "fixture"})))
        self.assertEqual((await self.repository.get(self.owner, task.id)).reason, "task_usage_overrun")

    async def test_http_auth_queue_backpressure_and_owner_scoped_events(self):
        from app.main import create_app
        repo = TaskRepository(self.pool, self.limits.model_copy(update={"owner_queue_limit": 1}))
        app = create_app()
        app.state.runtime_services = SimpleNamespace(background_task_service=TaskService(repo, {"frontier": _Provider()}))
        app.dependency_overrides[get_settings] = lambda: JarvisSettings(_env_file=None, JARVIS_OWNER_ID=self.owner, JARVIS_STATIC_BEARER_TOKEN="fixture-token")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            body = self.request().model_dump(mode="json")
            self.assertEqual((await client.post("/tasks", json=body)).status_code, 401)
            headers = {"Authorization": "Bearer fixture-token"}
            first = await client.post("/tasks", headers=headers, json=body)
            self.assertEqual(first.status_code, 200, first.text)
            self.assertEqual((await client.post("/tasks", headers=headers, json=body)).json()["id"], first.json()["id"])
            full = await client.post("/tasks", headers=headers, json=self.request().model_dump(mode="json"))
            self.assertEqual(full.status_code, 429)
            self.assertEqual(full.json()["error"]["code"], "task_queue_full")
            events = await client.get(f"/tasks/{first.json()['id']}/events", headers=headers)
            self.assertEqual(events.status_code, 200)
            foreign = await self.create(owner=self.other)
            self.assertEqual((await client.get(f"/tasks/{foreign.id}", headers=headers)).status_code, 404)

    async def test_runtime_enabled_with_no_provider_stays_truthful_and_stops_cleanly(self):
        settings = SecondBrainSettings(_env_file=None, enabled="true", tasks_enabled="true")
        services = SimpleNamespace(settings=SimpleNamespace(SUPABASE_DB_URL=self.dsn), db_pool=self.pool,
                                   background_task_providers={},assistant_context_providers=())
        with patch("app.tasks.service.get_second_brain_settings", return_value=settings):
            await initialize_task_runtime(services)
        self.assertIsNone(services.background_task_issue)
        try:
            request = self.request()
            accepted = await self.create(request=request)
            replay = await services.background_task_service.create(self.owner, request)
            self.assertEqual(replay.id, accepted.id)
            with self.assertRaises(TaskError):
                await services.background_task_service.create(self.owner, self.request())
        finally:
            self.assertFalse(hasattr(services,"background_task_runner"))
            self.assertFalse(hasattr(services,"background_task_runner_task"))

    async def test_worker_process_kill_restart_preserves_tasks_and_uncertain_external_effects(self):
        jobs = {}
        effects = []
        lock = threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def respond(self, value):
                body = json.dumps(value).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                with lock:
                    if self.path == "/jobs":
                        identity = str(uuid4()); jobs[identity] = json.loads(body)["task"] != str(slow.id)
                        self.respond({"id": identity})
                    elif self.path == "/effects":
                        effects.append(json.loads(body)["action"])
                        self.respond({"accepted": True})
                    else:
                        identity = self.path.split("/")[2]; jobs[identity] = True
                        self.respond({"state": "terminal"})
            def do_GET(self):
                with lock:
                    self.respond({"state": "terminal" if jobs[self.path.split("/")[2]] else "running"})
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        limits = self.limits.model_copy(update={"owner_concurrency": 1, "global_concurrency": 1, "lease_seconds": 5})
        self.repository = TaskRepository(self.pool, limits)
        waiting = await self.create(request=self.request(title="fixture-wait"))
        slow = await self.create(request=self.request(title="fixture-slow"))
        queued = await self.create(request=self.request(title="fixture-queued"))
        grant, _ = await self.grant()
        action_task = await self.create(request=self.request(title="fixture-action", grant_ids=(grant,)))
        def launch():
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(Path.cwd()), "PYTHONDONTWRITEBYTECODE": "1",
                   "JARVIS_SECOND_BRAIN_TEST_DB_URL": self.dsn, "JARVIS_TASK_PROBE_SCHEMA": self.schema,
                   "JARVIS_TASK_PROBE_URL": f"http://127.0.0.1:{server.server_port}", "JARVIS_TASK_PROBE_LIMITS": limits.model_dump_json()}
            output = open(self.fixture.path.with_suffix(f".{uuid4().hex}.worker.log"), "w")
            process = subprocess.Popen([sys.executable, "-m", "evals.second_brain.worker_probe"], env=env, stdout=output, stderr=subprocess.STDOUT)
            def stop():
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
                output.close()
            self.addCleanup(stop)
            return process
        async def until(predicate, *, process, timeout=30):
            end = time.monotonic()+timeout
            while time.monotonic()<end:
                self.assertIsNone(process.poll(), "fixture worker exited; inspect its retained worker.log")
                value = await predicate()
                if value: return value
                await asyncio.sleep(0.05)
            self.fail("fixture worker condition timed out")
        first = launch()
        async def live_handle():
            async with self.pool.acquire() as connection:
                return await connection.fetchval("select provider_handle from jarvis_task_attempts where owner_id=$1 and task_id=$2 and provider_state='live'", self.owner, slow.id)
        handle = await until(live_handle, process=first)
        self.assertIsNone(first.poll())
        first.kill(); first.wait(timeout=5)
        self.assertEqual((await self.repository.get(self.owner, waiting.id)).status, TaskStatus.WAITING)
        self.assertEqual((await self.repository.get(self.owner, queued.id)).status, TaskStatus.QUEUED)
        second = launch()
        async def reconciled_live_handle():
            return any(event["event_type"] == "provider_running" for event in await self.repository.events(self.owner, slow.id))
        await until(reconciled_live_handle, process=second)
        with lock:
            jobs[handle] = True
        async def finished():
            return (await self.repository.get(self.owner, queued.id)).status == TaskStatus.SUCCEEDED
        await until(finished, process=second)
        self.assertIsNone(second.poll())
        self.assertEqual((await self.repository.get(self.owner, slow.id)).attempt_count, 1)
        self.assertEqual((await self.repository.get(self.owner, waiting.id)).status, TaskStatus.WAITING)
        self.assertIn(handle, jobs)
        self.assertEqual(len(jobs), 2, "one remote slow job plus one queued job, not a replay of the killed worker's job")
        async def effect_submitted():
            with lock:
                return bool(effects)
        await until(effect_submitted, process=second)
        second.kill(); second.wait(timeout=5)
        third = launch()
        async def uncertain():
            return (await self.repository.get(self.owner, action_task.id)).status == TaskStatus.UNKNOWN
        await until(uncertain, process=third)
        async with self.pool.acquire() as connection:
            action_id = await connection.fetchval("select id from jarvis_task_actions where owner_id=$1 and task_id=$2", self.owner, action_task.id)
        self.assertEqual(effects, [str(action_id)], "a killed submitter must not repeat its external effect")
        await self.actions.reconcile(self.owner, action_task.id, action_id, outcome="accepted", proof={"fixture_server_effects": effects[:]})
        paused = await self.repository.get(self.owner, action_task.id)
        self.assertEqual(paused.status, TaskStatus.PAUSED)
        await self.mutation(paused, "resume")
        async def action_finished():
            return (await self.repository.get(self.owner, action_task.id)).status == TaskStatus.SUCCEEDED
        await until(action_finished, process=third)
        self.assertEqual(effects, [str(action_id)], "explicit resume must reuse the reconciled receipt, not resend")
