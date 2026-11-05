"""Offline orchestration boundary: production code + in-memory SQL transport + disk fake.

This deliberately does NOT qualify PostgreSQL constraints or actual E2B persistence.
Unknown SQL fails loudly; the existing production repository methods are exercised.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.sandbox.allocations import SandboxAllocationRepository
from app.sandbox.broker import SandboxCommandBroker
from app.sandbox.cleanup import SandboxCleaner, SandboxCleanupRepository
from app.sandbox.models import SandboxError, SandboxPolicy, SandboxInfo, E2BConfiguration
from app.tasks.models import ClaimLost, TaskError, TaskStatus, json_object


def now():
    return datetime.now(timezone.utc)


class Transport:
    def __init__(self):
        self.rows, self.views, self.attempts, self.grants = {}, {}, {}, {}
        self.unfinished, self.revoked, self.problem = False, set(), None
        self.lock = asyncio.Lock()
        self.sql = []

    @asynccontextmanager
    async def transaction(self, owner):
        async with self.lock:
            yield self

    def add_claim(self, *, task_id=None, revision=1):
        task = NS(id=task_id or uuid4(), owner_id="owner", status=TaskStatus.RUNNING,
            revision=revision, current_attempt_id=uuid4(), pending_status=None,
            active_seconds=0, budget=NS(active_seconds=300), fence=revision)
        claim = NS(task=task, attempt_id=task.current_attempt_id, task_revision=revision, fence=revision)
        self.views[task.id] = task
        self.attempts[claim.attempt_id] = True
        grant = uuid4()
        self.grants[grant] = dict(id=grant, capability="sandbox.execute", scope=SandboxPolicy().model_dump(),
            expires_at=now()+timedelta(minutes=5), task_id=task.id)
        return claim, grant

    async def lock_claim(self, c, claim):
        t = self.views[claim.task.id]
        if (t.status != TaskStatus.RUNNING or t.current_attempt_id != claim.attempt_id
                or t.fence != claim.fence or not self.attempts[claim.attempt_id]):
            raise ClaimLost()
        return t, {}

    async def load(self, c, owner, identity, **kw):
        return self.views[identity]

    async def context_problem(self, c, task):
        return self.problem

    async def validate_grant(self, c, owner, identity):
        if identity in self.revoked:
            raise TaskError("task_grant_unavailable", "revoked")
        return self.grants[identity]

    async def emit(self, *a, **kw):
        pass

    async def fetch(self, sql, *args):
        self.sql.append(sql)
        if "select id,reusable,phase" in sql:
            return [dict(r) for r in self.rows.values() if r["owner_id"]==args[0] and r["task_id"]==args[1]]
        if "(grant_id=$2 or (task_id=$3 and attempt_id=$4))" in sql:
            return [dict(r) for r in self.rows.values() if r["owner_id"] == args[0] and
                    (r["grant_id"] == args[1] or (r["task_id"], r["attempt_id"]) == args[2:4])]
        if "select id,task_id" in sql:
            return [dict(r) for r in self.rows.values() if r["phase"] != "cleaned" and
                    r["next_cleanup_at"] <= now() and not r.get("cleanup_live")]
        raise AssertionError(sql)

    async def fetchval(self, sql, *args):
        self.sql.append(sql)
        if "jarvis_task_grants" in sql:
            return self.grants[args[2]]["task_id"] == args[1]
        if "jarvis_task_attempts" in sql:
            return self.attempts.get(args[1], False)
        if "jarvis_sandbox_commands" in sql:
            return self.unfinished
        if "select fence from jarvis_background_tasks" in sql:
            return self.views[args[1]].fence
        raise AssertionError(sql)

    async def fetchrow(self, sql, *args):
        self.sql.append(sql)
        if "insert into jarvis_sandbox_allocations" in sql:
            owner, task, revision, attempt, grant, policy, digest, expiry, seconds, reusable = args
            identity = uuid4()
            row = dict(id=identity, owner_id=owner, task_id=task, task_revision=revision, attempt_id=attempt,
                grant_id=grant, policy=json_object(policy), policy_digest=digest, phase="reserved",
                deadline=min(expiry, now()+timedelta(seconds=seconds)), reusable=reusable, sandbox_id=None,
                template_id=None, provider_expires_at=None, submission_token=None, submission_fence=None,
                cleanup_token=None, cleanup_generation=0, cleanup_observed_ids=[], cleanup_live=False,
                next_cleanup_at=now(), reason=None)
            row["native_scope"] = SandboxAllocationRepository.scope(row).model_dump(mode="json")
            self.rows[identity] = row
            return dict(row)
        if "jarvis_sandbox_allocations" in sql:
            identity = args[-1]
            row = self.rows.get(identity)
            if row is None or row["owner_id"] != args[0]:
                return None
            if len(args) == 3 and row["task_id"] != args[1]:
                return None
            return dict(row, within_deadline=row["deadline"]>now())
        raise AssertionError(sql)

    async def execute(self, sql, *args):
        self.sql.append(sql)
        row = self.rows[args[1]]
        if "phase='creating'" in sql:
            row.update(phase="creating", submission_token=args[2], submission_fence=args[3])
        elif "set phase=$3,sandbox_id=$4" in sql:
            row.update(phase=args[2], sandbox_id=args[3], template_id=args[4], provider_expires_at=args[5])
        elif "task_id=$3,task_revision=$4" in sql:
            row.update(task_id=args[2], task_revision=args[3], attempt_id=args[4], grant_id=args[5],
                phase="resuming", deadline=min(row["deadline"], args[6]))
        elif "set phase='ready'" in sql:
            row["phase"] = "ready"
            if len(args)>2:
                row["provider_expires_at"] = args[2]
        elif "set phase=$8" in sql:
            row.update(phase=args[7], reason=args[2], cleanup_token=args[3], cleanup_worker=args[4],
                cleanup_generation=args[5], cleanup_live=True)
        elif "set phase='paused'" in sql:
            row.update(phase="paused", cleanup_token=None, cleanup_live=False,
                next_cleanup_at=now()+timedelta(seconds=args[2]))
        elif "set phase=$3,cleanup_error" in sql:
            row.update(phase=args[2], cleanup_error=args[3], cleanup_token=None, cleanup_live=False,
                next_cleanup_at=now()+timedelta(seconds=args[4]))
        elif "set phase=$3,reason=$4" in sql:
            row.update(phase=args[2], reason=args[3], next_cleanup_at=now())
        elif "set cleanup_observed_ids=$3" in sql:
            row["cleanup_observed_ids"] = args[2]
        elif "cleanup_lease_until=clock_timestamp()" in sql:
            row["cleanup_live"] = True
        elif "next_cleanup_at=least(deadline" in sql:
            row["next_cleanup_at"] = min(row["deadline"], now()+timedelta(seconds=args[2]))
        else:
            raise AssertionError(sql)


class DiskProvider:
    configuration = E2BConfiguration(api_key="offline-only", request_timeout_seconds=1)

    def __init__(self, path):
        self.path = path
        self.handle = None
        self.creates = self.pauses = self.resumes = self.kills = 0
        self.pause_gate = None
        self.pause_entered = asyncio.Event()
        self.resume_error = False

    async def create(self, scope, policy):
        self.creates += 1
        self.handle = SandboxInfo(sandbox_id="offline-vm", scope=scope, policy=policy,
            state="running", template_id="fixture-base", expires_at=now()+timedelta(minutes=5))
        return self.handle

    async def prepare(self, handle):
        return handle

    async def info(self, identity, scope, policy):
        if self.handle is None:
            raise SandboxError("sandbox_not_found")
        if scope != self.handle.provider_scope or policy != self.handle.policy:
            raise SandboxError("sandbox_scope_mismatch")
        return self.handle

    async def pause(self, handle):
        await self.info(handle.sandbox_id, handle.provider_scope, handle.policy)
        self.pauses += 1
        self.pause_entered.set()
        if self.pause_gate:
            await self.pause_gate.wait()
        self.handle = self.handle.model_copy(update={"state": "paused"})
        return self.handle

    async def resume(self, handle, *, timeout_seconds=None):
        self.resumes += 1
        if self.resume_error:
            raise SandboxError("sandbox_not_found")
        await self.info(handle.sandbox_id, handle.provider_scope, handle.policy)
        self.handle = handle.model_copy(update={"state":"running", "expires_at":now()+timedelta(seconds=timeout_seconds)})
        return self.handle

    async def discover(self, scope, policy):
        if self.handle is None:
            return []
        await self.info(self.handle.sandbox_id, scope, policy)
        return [self.handle.sandbox_id]

    async def kill(self, identity, scope, policy):
        await self.info(identity, scope, policy)
        self.kills += 1
        self.handle = None
        for p in self.path.iterdir():
            p.unlink()


class ContinuityTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.db = Transport()
        self.claim, self.grant = self.db.add_claim()
        self.provider = DiskProvider(self.path)
        self.first = self.broker(self.claim, self.grant, reusable=True)
        self.repo = SandboxCleanupRepository(self.first.allocations)
        self.cleaner = SandboxCleaner(self.repo, self.provider)
        self.row = await self.first._ready(await self.first.allocation())
        self.identity = self.row["id"]
        (self.path/"draft.txt").write_text("turn one retained contents")

    def broker(self, claim, grant, **kw):
        b = SandboxCommandBroker(NS(tasks=self.db, task=claim.task, claim=claim), self.provider, grant, **kw)
        b._managed = AsyncMock(return_value=None)
        return b

    def idle(self):
        self.claim.task.status = TaskStatus.SUCCEEDED
        self.db.attempts[self.claim.attempt_id] = False
        self.db.rows[self.identity]["next_cleanup_at"] = now()

    async def pause(self):
        self.idle()
        result = await self.cleaner.run_once("owner", "offline-cleaner")
        self.assertEqual(result["state"], "paused")

    def followup(self, *, same_task=False):
        claim, grant = self.db.add_claim(task_id=self.claim.task.id if same_task else None,
            revision=2 if same_task else 1)
        return self.broker(claim, grant, continuation_id=self.identity)

    async def test_turn_one_file_idle_pause_turn_two_resume_same_files(self):
        deadline = self.row["deadline"]
        await self.pause()
        second = self.followup()
        reused = await second._ready(await second.allocation())
        self.assertEqual(reused["id"], self.identity)
        self.assertEqual(reused["deadline"], deadline)
        self.assertEqual((self.path/"draft.txt").read_text(), "turn one retained contents")
        (self.path/"draft.txt").write_text("turn two edit")
        self.assertEqual((self.provider.creates, self.provider.pauses, self.provider.resumes, self.provider.kills), (1,1,1,0))
        self.assertEqual(second.allocations.handle(reused).scope.task_id, second.context.task.id)
        self.assertEqual(second.allocations.handle(reused).provider_scope.task_id, self.claim.task.id)
        with self.assertRaises(ClaimLost):
            await self.first.allocations.current(self.claim, self.identity)
        with self.assertRaises(TaskError) as caught:
            await self.first.allocations.request_cleanup("owner", self.identity)
        self.assertEqual(caught.exception.code, "sandbox_cleanup_stale")
        self.db.rows[self.identity]["next_cleanup_at"] = now()
        self.assertIsNone(await self.cleaner.run_once("owner", "old-cleanup-pass"))
        self.assertEqual((self.path/"draft.txt").read_text(), "turn two edit")

    async def test_same_task_new_attempt_reattaches(self):
        await self.pause()
        second = self.followup(same_task=True)
        reused = await second._ready(await second.allocation())
        self.assertEqual(reused["attempt_id"], second.context.claim.attempt_id)
        self.assertEqual(self.provider.creates, 1)

    async def test_active_budget_exhaustion_still_cleans(self):
        self.claim.task.active_seconds = 300
        result = await self.cleaner.run_once("owner", "cleaner")
        self.assertEqual(result["state"], "cleaned")
        self.assertEqual(self.provider.pauses, 0)

    async def test_superseded_but_still_active_attempt_is_not_retained(self):
        self.claim.task.revision = 2
        result = await self.cleaner.run_once("owner", "cleaner")
        self.assertEqual(result["state"], "cleaned")
        self.assertEqual(self.provider.pauses, 0)

    async def test_distinct_tasks_cannot_replay_same_short_command_key(self):
        from app.sandbox.commands import SandboxCommandRepository
        from app.tasks.models import digest
        first = SandboxCommandRepository.operation_digest(self.row, self.claim.task.id, "render")
        second = SandboxCommandRepository.operation_digest(self.row, uuid4(), "render")
        self.assertNotEqual(first, second)
        self.assertEqual(first, SandboxCommandRepository.operation_digest(self.row, self.claim.task.id, "render"))
        self.assertEqual(SandboxCommandRepository.operation_digest({}, uuid4(), "render"), digest({"key":"render"}))

    async def test_previous_task_command_identity_is_not_current_after_transfer(self):
        from app.sandbox.commands import SandboxCommandRepository
        await self.pause()
        second = self.followup()
        await second._ready(await second.allocation())
        commands = SandboxCommandRepository(self.db)
        commands._load = AsyncMock(return_value={"allocation_id":self.identity,
            "task_id":self.claim.task.id, "availability":"available"})
        with self.assertRaises(TaskError) as caught:
            await commands.current(second.context.claim, uuid4())
        self.assertEqual(caught.exception.code, "sandbox_command_not_current")
        with self.assertRaises(TaskError):
            await commands.begin_submission(second.context.claim, uuid4())

    async def test_resolver_uses_owned_task_not_latest_owner_workspace(self):
        unrelated = self.followup()
        with patch('app.live_state.admission.task_command_source', new=AsyncMock(return_value=None)) as source:
            self.assertIsNone(await unrelated.allocations.continuation_for_task(unrelated.context.claim))
            self.assertEqual(await self.first.allocations.continuation_for_task(self.claim), self.identity)
            self.assertEqual(source.await_count, 1)

    async def test_resolver_uses_only_validated_exact_predecessor(self):
        from app.tasks.media_editing import CorrectedMediaInput
        second = self.followup()
        pin = CorrectedMediaInput.model_construct(target=NS(task_id=self.claim.task.id))
        with patch('app.live_state.admission.task_command_source', new=AsyncMock(return_value=NS(content={'media_input':{}}))), \
             patch('app.tasks.media_editing.parse_media_input', return_value=pin):
            self.assertEqual(await second.allocations.continuation_for_task(second.context.claim), self.identity)
            self.db.rows[self.identity]['phase']='cleaned'
            with self.assertRaises(TaskError) as caught:
                await second.allocations.continuation_for_task(second.context.claim)
            self.assertEqual(caught.exception.code,'sandbox_continuation_unavailable')

    async def test_resolver_propagates_withdrawn_predecessor_validation(self):
        second=self.followup()
        with patch('app.live_state.admission.task_command_source',
                   new=AsyncMock(side_effect=TaskError('media_command_scope_unavailable','withdrawn'))):
            with self.assertRaises(TaskError) as caught:
                await second.allocations.continuation_for_task(second.context.claim)
            self.assertEqual(caught.exception.code,'media_command_scope_unavailable')

    async def test_generated_schema_contains_exact_continuity_migration(self):
        from app.migrations.runner import load_migrations, render_fresh_schema
        migrations=load_migrations()
        self.assertEqual(migrations[-1].name,'sandbox_continuity')
        self.assertEqual((Path(__file__).resolve().parents[1]/'app/schema.sql').read_text(), render_fresh_schema())

    async def test_active_work_is_not_paused(self):
        self.assertIsNone(await self.cleaner.run_once("owner", "cleaner"))
        self.assertEqual(self.provider.pauses, 0)

    async def test_attachment_waits_for_confirmed_pause_not_merely_pause_intent(self):
        self.idle()
        self.provider.pause_gate = asyncio.Event()
        pending = asyncio.create_task(self.cleaner.run_once("owner", "cleaner"))
        await self.provider.pause_entered.wait()
        try:
            with self.assertRaises(TaskError) as caught:
                await self.followup().allocation()
            self.assertEqual(caught.exception.code, "sandbox_continuation_busy")
        finally:
            self.provider.pause_gate.set()
            await pending

    async def test_resume_failure_never_creates_replacement_or_claims_ready(self):
        await self.pause()
        second = self.followup()
        self.provider.resume_error = True
        with self.assertRaises(SandboxError):
            await second._ready(await second.allocation())
        self.assertEqual(self.provider.creates, 1)
        self.assertEqual(self.db.rows[self.identity]["phase"], "resuming")
        with self.assertRaises(TaskError):
            await second.allocations.current(second.context.claim, self.identity)

    async def test_deadline_still_kills_paused_workspace(self):
        await self.pause()
        row = self.db.rows[self.identity]
        row.update(deadline=now()-timedelta(seconds=1), next_cleanup_at=now())
        result = await self.cleaner.run_once("owner", "cleaner")
        self.assertEqual(result["state"], "cleaned")
        self.assertEqual(self.provider.kills, 1)
        self.assertFalse((self.path/"draft.txt").exists())

    async def test_cancelled_current_holder_is_cleaned_after_transfer(self):
        await self.pause()
        second = self.followup()
        await second._ready(await second.allocation())
        second.context.task.status = TaskStatus.CANCELLED
        self.db.rows[self.identity]["next_cleanup_at"] = now()
        result = await self.cleaner.run_once("owner", "cleaner")
        self.assertEqual(result["state"], "cleaned")
        self.assertEqual(self.provider.kills, 1)

    async def test_revoked_source_cannot_transfer_or_retain(self):
        await self.pause()
        self.db.revoked.add(self.grant)
        with self.assertRaises(TaskError):
            await self.followup().allocation()
        self.db.rows[self.identity]["next_cleanup_at"] = now()
        self.assertEqual((await self.cleaner.run_once("owner", "cleaner"))["state"], "cleaned")

    async def test_unresolved_native_command_is_not_paused_and_reused(self):
        self.idle()
        self.db.unfinished = True
        result = await self.cleaner.run_once("owner", "cleaner")
        self.assertEqual(result["state"], "cleaned")
        self.assertEqual(self.provider.pauses, 0)

    async def test_old_cleanup_claim_cannot_touch_transferred_holder(self):
        self.idle()
        old = await self.repo.claim_next("owner", "old-cleaner")
        handle = await self.provider.pause(self.first.allocations.handle(self.db.rows[self.identity]))
        await self.repo.finish_pause(old, handle)
        second = self.followup()
        await second._ready(await second.allocation())
        with self.assertRaises(SandboxError) as caught:
            await self.repo.authorize(old)
        self.assertEqual(caught.exception.code, "sandbox_cleanup_claim_lost")
        self.assertEqual(self.provider.kills, 0)

    async def test_cleanup_intent_is_irreversible(self):
        await self.pause()
        await self.first.allocations.request_cleanup("owner", self.identity, revoke=True)
        with self.assertRaises(TaskError):
            await self.followup().allocation()
        self.assertEqual(self.db.rows[self.identity]["phase"], "cleaned")

    async def test_late_creation_receipt_cannot_regress_transferred_holder(self):
        original = self.first.allocations.handle(self.row)
        token = self.row["submission_token"]
        await self.pause()
        second = self.followup()
        ready = await second._ready(await second.allocation())
        receipt = await self.first.allocations.record_created("owner", self.identity, token, original)
        self.assertEqual(receipt["phase"], "ready")
        self.assertEqual(receipt["attempt_id"], ready["attempt_id"])

    async def test_pause_lease_loss_does_not_restore_execution(self):
        self.idle()
        claim = await self.repo.claim_next("owner", "lost-cleaner")
        row = self.db.rows[self.identity]
        row.update(cleanup_live=False, next_cleanup_at=now())
        with self.assertRaises(TaskError):
            await self.followup().allocation()
        self.assertEqual(self.db.rows[self.identity]["phase"], "cleaned")
        with self.assertRaises(SandboxError):
            await self.repo.authorize(claim)

    async def test_expired_workspace_cannot_attach(self):
        await self.pause()
        self.db.rows[self.identity]["deadline"] = now()-timedelta(seconds=1)
        with self.assertRaises(TaskError) as caught:
            await self.followup().allocation()
        self.assertEqual(caught.exception.code, "sandbox_continuation_unavailable")

    async def test_wrong_owner_cannot_attach(self):
        await self.pause()
        second = self.followup()
        second.context.task.owner_id = "other-owner"
        with self.assertRaises(TaskError) as caught:
            await second.allocation()
        self.assertEqual(caught.exception.code, "sandbox_allocation_not_found")

    async def test_changed_policy_cannot_attach(self):
        await self.pause()
        second = self.followup()
        self.db.grants[second.grant_id]["scope"] = SandboxPolicy(command_seconds=30).model_dump()
        with self.assertRaises(TaskError) as caught:
            await second.allocation()
        self.assertEqual(caught.exception.code, "sandbox_continuation_unavailable")

    async def test_current_attempt_old_fence_cannot_request_cleanup(self):
        self.claim.task.fence += 1
        with self.assertRaises(TaskError) as caught:
            await self.first.allocations.request_cleanup("owner", self.identity)
        self.assertEqual(caught.exception.code, "sandbox_cleanup_stale")
        self.assertEqual(self.db.rows[self.identity]["phase"], "ready")

    async def test_removed_context_is_cleaned(self):
        await self.pause()
        self.db.problem = "task_context_removed"
        self.db.rows[self.identity]["next_cleanup_at"] = now()
        result = await self.cleaner.run_once("owner", "cleaner")
        self.assertEqual(result["state"], "cleaned")
        self.assertEqual(self.provider.kills, 1)


class ProviderContinuityTests(IsolatedAsyncioTestCase):
    def setUp(self):
        from tests.test_sandbox_provider import SandboxProviderTests
        SandboxProviderTests.setUp(self)
        self.box.set_timeout = AsyncMock()

    async def test_resume_uses_immutable_metadata_and_remaining_deadline(self):
        handle = await self.provider.info(self.raw.sandbox_id, self.scope, self.policy)
        holder = self.scope.model_copy(update={"task_id":uuid4(), "attempt_id":uuid4()})
        current = handle.model_copy(update={"scope":holder, "native_scope":self.scope})
        resumed = await self.provider.resume(current, timeout_seconds=37)
        self.assertEqual(resumed.scope, holder)
        self.assertEqual(resumed.provider_scope, self.scope)
        self.assertEqual(self.api.connect.call_args.kwargs["timeout"], 37)
        self.box.set_timeout.assert_awaited_once()
        self.assertEqual(self.box.set_timeout.call_args.args, (37,))
        self.api.create.assert_not_awaited()

    async def test_missing_original_metadata_cannot_resume(self):
        handle = await self.provider.info(self.raw.sandbox_id, self.scope, self.policy)
        current = handle.model_copy(update={"scope":self.scope.model_copy(update={"task_id":uuid4()})})
        with self.assertRaises(SandboxError):
            await self.provider.resume(current, timeout_seconds=30)
        self.api.connect.assert_not_awaited()

    async def test_resume_never_extends_full_policy_from_excessive_timeout(self):
        handle = await self.provider.info(self.raw.sandbox_id, self.scope, self.policy)
        for timeout in (0, -1, 301, True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                await self.provider.resume(handle, timeout_seconds=timeout)
        self.api.connect.assert_not_awaited()

    async def test_pause_uses_original_provider_metadata_after_transfer(self):
        handle = await self.provider.info(self.raw.sandbox_id, self.scope, self.policy)
        current = handle.model_copy(update={"scope":self.scope.model_copy(update={"task_id":uuid4()}),
            "native_scope":self.scope})
        self.raw.state = "paused"
        receipt = await self.provider.pause(current)
        self.assertEqual(receipt.provider_scope, self.scope)
        self.api.pause.assert_awaited_once()
