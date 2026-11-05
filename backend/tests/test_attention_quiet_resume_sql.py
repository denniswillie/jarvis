"""Manifested schema-42 qualification with controlled transport, never providers.

Each named check is reported, including failures. Synthetic task/conversation
sources and owned fixture schemas are cleaned by DatabaseFixture's independent
audit. Delivery telemetry in this test is not hardware/hearing evidence.
"""
import asyncio
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import asyncpg

from app.attention.models import AttentionError, InboxMutation
from app.attention.push_models import DeviceRegistration, PushOutcome
from app.attention.push_repository import PushRepository
from app.attention.quiet import OwnerQuietRepository, QuietCommand, quiet_delivery_decision
from app.attention.quiet_resume import NAMESPACE
from app.attention.repository import AttentionRepository
from app.migrations.runner import apply_migrations, load_migrations, quote_schema, read_schema_readiness, attention_quiet_schema_ready
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskCreate, TaskMutation, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tools.invocation import ToolInvocationContext
from evals.second_brain.fixtures import DatabaseFixture
from test_notification_ledger_database import FixtureSpeechPreparation, cleanup_fixture_resources


async def record_qualification_check(report, failures, save, name, operation, *, prerequisite_ready=True):
    """A failed prerequisite is not evidence that dependent functionality ran."""
    if not prerequisite_ready:
        item = {"name": name, "status": "unreached", "reason": "migration_lifecycle_prerequisite_failed"}
    else:
        try:
            await operation()
        except Exception as error:
            item = {"name": name, "status": "failed", "error_type": type(error).__name__,
                    "sqlstate": getattr(error, "sqlstate", None)}
            if isinstance(error, AssertionError):
                item["assertion"] = str(error)[:1500]
            failures.append(item)
        else:
            item = {"name": name, "status": "passed"}
    report["checks"].append(item)
    save()
    return item["status"] == "passed"


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class QuietResumeSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_migration_release_bursts_and_recovery(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        fixture = DatabaseFixture(dsn, directory, test_name=self.id())
        report_path = directory / f"quiet-resume-results-{fixture.run_id}.json"
        report = {"format": "jarvis.quiet_resume.sql.v1", "run_id": fixture.run_id,
                  "status": "running", "checks": [], "cleanup": None,
                  "provider_calls": 0, "apns_calls": 0, "hardware_evidence": False}
        failures, wire = [], []
        connection = pool = None
        lifecycle_ready = False

        def save():
            fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")

        async def check(name, operation, *, requires_lifecycle=True):
            return await record_qualification_check(report, failures, save, name, operation,
                prerequisite_ready=not requires_lifecycle or lifecycle_ready)

        save()
        try:
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
            schema = await fixture.create_schema(connection)
            migrations = tuple(m for m in load_migrations() if m.version <= 42)
            await apply_migrations(connection, schema=schema, fresh=True, migrations=migrations[:41])
            path_sql = f"set search_path to {quote_schema(schema)},pg_catalog,extensions"
            await connection.execute(path_sql)

            async def setup(c):
                await c.execute(path_sql)
                await c.execute("set lock_timeout='8s'; set statement_timeout='45s'")

            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4, timeout=10,
                                            command_timeout=60, setup=setup)
            owner, other = fixture.owners
            session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", owner)
            other_session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", other)
            tasks = TaskRepository(pool, TaskLimits(lease_seconds=300))
            contexts, quiet = ContextRepository(tasks), OwnerQuietRepository(pool)
            attention = AttentionRepository(tasks, lease_seconds=300)
            push = PushRepository(attention, SimpleNamespace(environment="sandbox", topic="com.jarvis.quiet.fixture",
                fallback_seconds=0, max_attempts=3, lease_seconds=30))
            speech = FixtureSpeechPreparation()

            async def change(action="activate", **kwargs):
                record = await contexts.append(owner, ContextIngress(session_id=session,
                    source_namespace="quiet_resume_fixture", source_event_id=uuid4().hex, receipt_id=uuid4().hex,
                    kind="user_transcript", trust="user_input", finality="final",
                    payload={"text": "Resume interruptions." if action == "resume" else "Please do not disturb."}))
                authority = ToolInvocationContext(request_event_id=record.id, request_revision=record.revision)
                return await quiet.save(owner, session, authority, QuietCommand(action=action,
                    expected_revision=(await quiet.get(owner)).revision, **kwargs))

            async def work(*, who=owner, sid=session, status="succeeded"):
                task = await tasks.create(who, TaskCreate(request_id=uuid4(), title="Synthetic quiet release work",
                    requested_outcome="Return a synthetic read-only fixture result.", session_id=sid), model="fixture-model")
                claim = await tasks.claim_next("quiet-resume-work", ("frontier",), owner_id=who)
                self.assertIsNotNone(claim)
                self.assertEqual(claim.task.id, task.id)
                result = await HermesTaskResult(status=status, answer="Synthetic result remains readable.",
                    speech=[{"text": "Synthetic result remains readable."}], evidence=[],
                    completed_operation_refs=[], uncertainties=[]).verified(TaskRunner(tasks, {}).context(claim), SimpleNamespace())
                self.assertTrue(await tasks.finish(claim, result))
                identity = await connection.fetchval("select id from jarvis_attention_intents where owner_id=$1 and task_id=$2",
                                                     who, task.id)
                self.assertIsNotNone(identity)
                return await tasks.get(who, task.id), identity

            async def reset():
                # Synthetic test separation only; never a production release path.
                async with tasks.transaction(owner) as c:
                    await c.execute("""update jarvis_attention_attempts set
                        phase=case when submitted_at is null then 'failed' else 'unknown' end,ended_at=clock_timestamp()
                        where owner_id=$1 and phase in ('claimed','submitted','forwarding','awaiting_receipt')""", owner)
                    await c.execute("update jarvis_attention_intents set state='dismissed' where owner_id=$1", owner)
                if (await quiet.get(owner)).effective_state == "quiet":
                    await change("resume")
                wire.clear()

            async def emit(claim, *, start=0):
                async with attention.write_guard(claim, 0, start_sample=start, samples=8) as sent:
                    wire.append({"intent": str(claim.intent_id), "start": start, "samples": 8})
                    sent()

            async def summary_row():
                episode = (await quiet.get(owner)).policy.episode.episode_id
                return await connection.fetchrow("""select i.*,p.content from jarvis_attention_intents i
                    join jarvis_context_payloads p on p.owner_id=i.owner_id and p.id=i.source_payload_id
                    where i.owner_id=$1 and i.source_namespace=$2 and i.source_id=$3""", owner, NAMESPACE, episode)

            async def migration():
                nonlocal pool, tasks, contexts, quiet, attention, push
                self.assertFalse(await attention_quiet_schema_ready(connection))
                ready = await read_schema_readiness(connection, schema=schema, migrations=migrations)
                self.assertEqual(ready.pending, (42,))
                self.assertTrue((await read_schema_readiness(connection, schema=schema, migrations=migrations[:41])).ready)
                _, identity = await work()
                original = await attention.claim_next(owner, session, "pre42")
                self.assertEqual(original.intent_id, identity)
                await change()
                await change("resume")
                self.assertIsNone(await attention.claim_next(owner, session, "missing42"))
                with self.assertRaises(AttentionError) as error:
                    async with attention.guard(original):
                        self.fail("An old schema must not release a queued writer.")
                self.assertEqual(error.exception.code, "owner_quiet_release_unavailable")
                # Match explicit deployment ownership, not an in-transaction
                # cached-statement retry. Stop delivery, drain all old pool
                # borrowers, then migrate before creating the new runtime.
                await attention.settle(original, reason="deployment_shutdown")
                old_attempt = await connection.fetchrow("""select phase,submitted_at,ended_at
                    from jarvis_attention_attempts where owner_id=$1 and id=$2""", owner, original.attempt_id)
                self.assertEqual(old_attempt["phase"], "failed")
                self.assertIsNone(old_attempt["submitted_at"])
                self.assertIsNotNone(old_attempt["ended_at"])
                old_pool = pool
                borrowed = await old_pool.acquire()
                entry_query = "select * from jarvis_attention_intents where owner_id=$1 and id=$2"
                closing = None
                try:
                    before = await borrowed.fetchrow(entry_query, owner, identity)
                    self.assertEqual(before["id"], identity)
                    self.assertNotIn("quiet_episode_id", before)
                    closing = asyncio.create_task(old_pool.close())
                    with self.assertRaises(TimeoutError):
                        await asyncio.wait_for(asyncio.shield(closing), timeout=.05)
                    self.assertFalse(closing.done(), "Pool closure must wait for outstanding borrowers.")
                finally:
                    await old_pool.release(borrowed)
                    if closing is not None:
                        await asyncio.wait_for(closing, timeout=10)
                self.assertEqual(old_pool.get_size(), 0)
                with self.assertRaises(asyncpg.InterfaceError):
                    await borrowed.fetchrow(entry_query, owner, identity)
                pool = None
                self.assertEqual(await apply_migrations(connection, schema=schema, migrations=migrations), (42,))
                self.assertEqual(await apply_migrations(connection, schema=schema, migrations=migrations), ())
                self.assertTrue(await attention_quiet_schema_ready(connection))
                self.assertTrue((await read_schema_readiness(connection, schema=schema, migrations=migrations)).ready)
                rollback = connection.transaction()
                await rollback.start()
                try:
                    await connection.execute("alter table jarvis_attention_intents drop constraint jarvis_attention_quiet_pair")
                    self.assertFalse(await attention_quiet_schema_ready(connection))
                    self.assertIn("missing_attention_quiet_episode_schema",
                                  (await read_schema_readiness(connection, schema=schema, migrations=migrations)).issues)
                finally:
                    await rollback.rollback()
                self.assertTrue(await attention_quiet_schema_ready(connection))
                # The administrative connection never becomes a runtime pool.
                # New repositories must not retain the closed pre-upgrade pool.
                pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4, timeout=10,
                                                command_timeout=60, setup=setup)
                self.assertIsNot(pool, old_pool)
                async with pool.acquire() as fresh:
                    # A server-side pooler may reuse a PostgreSQL backend PID.
                    # Client ownership and the actual changed row shape matter.
                    self.assertIsNot(fresh, borrowed)
                    after = await fresh.fetchrow(entry_query, owner, identity)
                    self.assertEqual(after["id"], identity)
                    self.assertIn("quiet_episode_id", after)
                    self.assertIsNone(after["quiet_episode_id"])
                    self.assertIn("quiet_summary_intent_id", after)
                    self.assertIsNone(after["quiet_summary_intent_id"])
                    self.assertTrue(await attention_quiet_schema_ready(fresh))
                tasks = TaskRepository(pool, TaskLimits(lease_seconds=300))
                contexts, quiet = ContextRepository(tasks), OwnerQuietRepository(pool)
                attention = AttentionRepository(tasks, lease_seconds=300)
                push = PushRepository(attention, SimpleNamespace(environment="sandbox", topic="com.jarvis.quiet.fixture",
                    fallback_seconds=0, max_attempts=3, lease_seconds=30))
                self.assertIs(tasks.pool, pool)
                self.assertIs(attention.pool, pool)
                self.assertIs(attention.tasks, tasks)
                self.assertIs(quiet.pool, pool)
                self.assertIs(push.tasks, tasks)
                self.assertIs(push.attention, attention)
                recovered = await attention.claim_next(owner, session, "post42")
                self.assertEqual(recovered.source_namespace, NAMESPACE)
                with self.assertRaises(AttentionError) as error:
                    async with attention.guard(original):
                        self.fail("The stopped pre-upgrade delivery lease must not regain ownership.")
                self.assertEqual(error.exception.code, "owner_quiet_coalesced")
                self.assertIsNotNone((await attention.detail(owner, identity))["expert_result"])
            lifecycle_ready = await check("isolated_41_to_42_upgrade_retry_partial_readiness_and_recovery",
                                          migration, requires_lifecycle=False)

            async def before_claim(count):
                await reset()
                await change()
                results = [await work() for _ in range(5)]
                episode = (await quiet.get(owner)).policy.episode.episode_id
                for _ in range(count):
                    await change()
                    self.assertIsNone(await attention.claim_next(owner, session, "quiet-before-claim"))
                    self.assertIsNone(await push.claim_next(owner, "quiet-before-claim-push"))
                    self.assertTrue((await quiet_delivery_decision(None, owner, kind="solicited_reply", channel="voice")).allowed)
                    self.assertEqual((await OwnerQuietRepository(pool).get(owner)).policy.episode.episode_id, episode)
                await change("resume")
                claims = await asyncio.gather(*(AttentionRepository(tasks).claim_next(owner, session, f"burst-{i}") for i in range(3)))
                admitted = [claim for claim in claims if claim is not None]
                self.assertEqual(len(admitted), 1)
                claim = admitted[0]
                self.assertEqual(claim.source_namespace, NAMESPACE)
                row = await summary_row()
                content = json.loads(row["content"]) if isinstance(row["content"], str) else row["content"]
                self.assertEqual(len(content["members"]), 3)
                self.assertLess(len(content["message"].encode()), 400)
                requests = await attention.prepare(claim, speech)
                self.assertEqual(len(requests), 1)
                await attention.submit_chunk(claim, 0, requests[0])
                await emit(claim)
                await attention.settle(claim, reason="fixture_partial_output", interrupted=True)
                for _ in range(count):
                    await change("resume")
                    self.assertIsNone(await AttentionRepository(tasks).claim_next(owner, session, "reconnected"))
                    self.assertIsNone(await push.claim_next(owner, "no-phone-replay"))
                self.assertEqual(len(wire), 1)
                for task, identity in results:
                    self.assertEqual((await tasks.get(owner, task.id)).status, TaskStatus.SUCCEEDED)
                    self.assertIsNotNone((await attention.detail(owner, identity))["expert_result"])
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_attention_attempts where owner_id=$1 and intent_id=$2",
                                                          owner, claim.intent_id), 1)
            for count in (1, 2, 3):
                await check(f"{count}_cutoffs_before_claim_one_bounded_summary_concurrent_reconnect", lambda count=count: before_claim(count))

            async def writer_cutoffs(count, partial):
                await reset()
                task, identity = await work()
                claim = await attention.claim_next(owner, session, "writer-boundary")
                self.assertEqual(claim.intent_id, identity)
                requests = await attention.prepare(claim, speech)
                await attention.submit_chunk(claim, 0, requests[0])
                if partial:
                    await emit(claim)
                real_mutation = tasks.mutate
                with patch.object(tasks, "mutate", AsyncMock(wraps=real_mutation)) as cancel_hook:
                    for _ in range(count):
                        await change()
                        with self.assertRaises(AttentionError) as error:
                            await emit(claim, start=8 if partial else 0)
                        self.assertEqual(error.exception.code, "owner_quiet_quiet")
                        await attention.settle(claim, reason="owner_quiet_quiet", interrupted=True)
                    cancel_hook.assert_not_awaited()
                detail = await attention.detail(owner, identity)
                self.assertEqual(detail["state"], "unknown")
                self.assertEqual(detail["attempts"][0]["forwarded_samples"], 8 if partial else 0)
                self.assertIsNone(detail["attempts"][0]["played_samples"])
                self.assertEqual(detail["hearing"], "not_established_by_inbox_or_generation_state")
                await change("resume")
                self.assertIsNone(await attention.claim_next(owner, session, "no-ambiguous-replay"))
                self.assertIsNone(await push.claim_next(owner, "no-ambiguous-phone-replay"))
                self.assertEqual((await summary_row())["state"], "dismissed")
                self.assertEqual(len(wire), 1 if partial else 0)
                self.assertEqual((await tasks.get(owner, task.id)).status, TaskStatus.SUCCEEDED)
            for count in (1, 2, 3):
                for partial in (False, True):
                    await check(f"{count}_cutoffs_{'after_partial' if partial else 'at_writer_handoff'}_unknown_no_cancel_or_replay",
                                lambda count=count, partial=partial: writer_cutoffs(count, partial))

            async def unsubmitted_recovery():
                await reset()
                task, identity = await work()
                original = await attention.claim_next(owner, session, "prewrite")
                await attention.prepare(original, speech)
                await change()
                for _ in range(3):
                    await attention.settle(original, reason="speech_only_stop", interrupted=True)
                detail = await attention.detail(owner, identity)
                self.assertEqual(detail["state"], "failed")
                self.assertIsNone(detail["attempts"][0]["submitted_at"])
                self.assertTrue((await quiet.get(owner)).suppresses_unsolicited)
                await change("resume")
                claim = await attention.claim_next(owner, session, "safe-summary")
                self.assertEqual(claim.source_namespace, NAMESPACE)
                await connection.execute("update jarvis_attention_attempts set lease_until=clock_timestamp()-interval '1 second' where owner_id=$1 and id=$2",
                                         owner, claim.attempt_id)
                self.assertIsNone(await AttentionRepository(tasks).claim_next(owner, session, "recovery"))
                self.assertIsNone(await push.claim_next(owner, "recovery-phone"))
                self.assertEqual((await tasks.get(owner, task.id)).status, TaskStatus.SUCCEEDED)
                # Deliberate replay addresses one original, not the whole backlog.
                detail = await attention.detail(owner, identity)
                request = InboxMutation(request_id=uuid4(), expected_generation=detail["generation"])
                await attention.mutate(owner, identity, "replay", request)
                replayed = await attention.mutate(owner, identity, "replay", request)
                self.assertTrue(replayed["replayed_request"])
                self.assertEqual((await attention.claim_next(owner, session, "explicit-original")).intent_id, identity)
            await check("unsubmitted_stop_is_not_work_cancel_crash_recovery_no_backlog_explicit_replay_dedupes", unsubmitted_recovery)

            async def expiry():
                await reset()
                now = await connection.fetchval("select clock_timestamp()")
                deadline = now + timedelta(seconds=2)
                accepted = await change(expires_at=deadline, timezone="UTC")
                await work()
                self.assertIsNone(await attention.claim_next(owner, session, "pre-expiry"))
                await asyncio.sleep(max(0, (deadline - await connection.fetchval("select clock_timestamp()")).total_seconds()) + .05)
                fresh = OwnerQuietRepository(pool)
                self.assertEqual((await fresh.get(owner)).revision, accepted["accepted_revision"])
                self.assertEqual((await fresh.get(owner)).effective_state, "expired")
                claim = await attention.claim_next(owner, session, "expired")
                self.assertEqual(claim.source_namespace, NAMESPACE)
                self.assertIsNone(await AttentionRepository(tasks).claim_next(owner, session, "expiry-reconnect"))
            await check("exact_expiry_reconstructs_once_without_implicit_revision_or_deadline_reset", expiry)

            async def source_and_goal_fences():
                for mode in ("goal", "withdraw", "expiry", "defer"):
                    await reset()
                    await change()
                    task, identity = await work()
                    await change("resume")
                    claim = await attention.claim_next(owner, session, "source-fence")
                    self.assertEqual(claim.source_namespace, NAMESPACE)
                    if mode == "goal":
                        updated = await tasks.mutate(owner, task.id, "update", TaskMutation(
                            request_id=uuid4(), expected_revision=task.revision, message="Different synthetic goal."))
                        await tasks.mutate(owner, task.id, "cancel", TaskMutation(request_id=uuid4(), expected_revision=updated.revision))
                    elif mode == "withdraw":
                        await connection.execute("update jarvis_context_payloads set availability='unavailable',content=null where owner_id=$1 and id=$2",
                                                 owner, task.result_payload_id)
                    elif mode == "expiry":
                        await connection.execute("update jarvis_attention_intents set expires_at=clock_timestamp()-interval '1 second' where owner_id=$1 and id=$2", owner, identity)
                    else:
                        await connection.execute("update jarvis_attention_intents set not_before=clock_timestamp()+interval '1 hour' where owner_id=$1 and id=$2", owner, identity)
                    with self.assertRaises(AttentionError):
                        async with attention.guard(claim):
                            self.fail("Changed member reached writer.")
                    await attention.settle(claim, reason="fixture_source_changed")
                    self.assertIsNone(await attention.claim_next(owner, session, "no-replacement"))
                    self.assertIsNone(await push.claim_next(owner, "no-replacement-phone"))
                    self.assertEqual(wire, [])
            await check("source_withdrawal_goal_revision_member_expiry_and_defer_rechecked_at_writer", source_and_goal_fences)

            async def push_and_owner_fences():
                await reset()
                await change()
                _, identity = await work()
                await push.change_device(owner, uuid4(), DeviceRegistration(request_id=uuid4(), expected_revision=0,
                    environment="sandbox", topic="com.jarvis.quiet.fixture", token="cd" * 32,
                    permission="authorized", enabled=True))
                await change("resume")
                claim = await push.claim_next(owner, "summary-phone")
                self.assertIsNotNone(claim)
                self.assertEqual(claim.intent_id, (await summary_row())["id"])
                self.assertIsNone(await attention.claim_next(owner, session, "no-cross-channel"))
                request = await push.prepare_submission(claim)
                self.assertIsNotNone(request)
                self.assertTrue((await push.authorize_send(request)).allowed)
                await push.finish(claim, PushOutcome("unknown", "controlled_handoff_unknown"))
                self.assertIsNone(await push.claim_next(owner, "no-push-repeat"))
                self.assertIsNone(await attention.claim_next(owner, session, "no-voice-after-push"))
                self.assertIsNotNone((await attention.detail(owner, identity))["expert_result"])
                _, other_identity = await work(who=other, sid=other_session)
                with self.assertRaises(asyncpg.ForeignKeyViolationError):
                    async with connection.transaction():
                        await connection.execute("update jarvis_attention_intents set quiet_episode_id=$3,quiet_summary_intent_id=$4 where owner_id=$1 and id=$2",
                                                 other, other_identity, uuid4(), claim.intent_id)
                self.assertFalse((await quiet.get(other)).suppresses_unsolicited)
            await check("canonical_push_summary_cross_channel_dedupe_unknown_and_cross_owner_constraint", push_and_owner_fences)

            async def running_and_cancel():
                await reset()
                await change()
                running = await tasks.create(owner, TaskCreate(request_id=uuid4(), title="Unrelated running work",
                    requested_outcome="Remain running during quiet controls.", session_id=session), model="fixture-model")
                running_claim = await tasks.claim_next("unrelated-running", ("frontier",), owner_id=owner)
                self.assertEqual(running_claim.task.id, running.id)
                target, identity = await work(status="waiting_for_input")
                revision = (await quiet.get(owner)).revision
                request = TaskMutation(request_id=uuid4(), expected_revision=target.revision)
                first = await tasks.mutate(owner, target.id, "cancel", request)
                for _ in range(3):
                    repeated = await tasks.mutate(owner, target.id, "cancel", request)
                    self.assertEqual(repeated.revision, first.revision)
                    self.assertEqual((await quiet.get(owner)).revision, revision)
                    self.assertTrue((await quiet_delivery_decision(None, owner, kind="solicited_reply", channel="voice")).allowed)
                    self.assertIsNone(await attention.claim_next(owner, session, "quiet-reconnect"))
                unrelated = await tasks.get(owner, running.id)
                self.assertEqual(unrelated.status, TaskStatus.RUNNING)
                self.assertEqual(unrelated.current_attempt_id, running_claim.attempt_id)
                result = await HermesTaskResult(status="succeeded", answer="Unrelated work completed while quiet.",
                    speech=[{"text": "Unrelated work completed while quiet."}], evidence=[], completed_operation_refs=[], uncertainties=[]
                    ).verified(TaskRunner(tasks, {}).context(running_claim), SimpleNamespace())
                self.assertTrue(await tasks.finish(running_claim, result))
                self.assertTrue((await quiet.get(owner)).suppresses_unsolicited)
                await change("resume")
                summary = await attention.claim_next(owner, session, "unrelated-result")
                self.assertEqual(summary.source_namespace, NAMESPACE)
                row = await summary_row()
                content = json.loads(row["content"]) if isinstance(row["content"], str) else row["content"]
                self.assertNotIn(str(identity), [m["intent_id"] for m in content["members"]])
                self.assertEqual((await tasks.get(owner, target.id)).status, TaskStatus.CANCELLED)
            await check("repeated_real_task_cancel_is_separate_direct_questions_reconnect_and_unrelated_work_continue", running_and_cancel)
            report["status"] = "failed" if failures else "passed"
        except Exception as error:
            report.update(status="failed", setup_error_type=type(error).__name__, sqlstate=getattr(error, "sqlstate", None))
            raise AssertionError("Quiet resume fixture setup/execution failed: " + type(error).__name__) from None
        finally:
            await cleanup_fixture_resources(fixture, report, save, pool=pool, connection=connection)
            print("QUIET_RESUME_SQL_REPORT=" + str(report_path))
            print("QUIET_RESUME_SQL_SUMMARY=" + json.dumps(report))
        self.assertEqual(failures, [], "Named quiet resume checks failed; see sanitized report.")
