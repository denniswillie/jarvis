"""Actual disposable-schema qualification; no personal owner, provider or public migration."""
import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

import asyncpg

from app.attention.models import AttentionError, InboxMutation
from app.attention.push_models import DeviceRegistration, PushOutcome
from app.attention.push_repository import PushRepository
from app.attention.quiet import OwnerQuietRepository, QuietCommand, quiet_delivery_decision
from app.attention.quiet_tools import create_quiet_tools
from app.attention.repository import AttentionRepository
from app.migrations.runner import apply_migrations, load_migrations, quote_schema
from app.reminders.duties import ResponsibilityRepository
from app.reminders.duty_preferences import ResponsibilityPreferences, SetNotificationPreferences, NotificationPolicy, read_preferences
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskCreate, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tools.inputs import ToolInputError
from app.tools.invocation import ToolInvocationContext, bind_tool_invocation
from evals.second_brain.fixtures import DatabaseFixture
from test_notification_ledger_database import cleanup_fixture_resources


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class OwnerQuietSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_owner_quiet_canonical_lifecycle_and_delivery(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        fixture = DatabaseFixture(dsn, directory, test_name=self.id())
        report_path = directory / f"owner-quiet-results-{fixture.run_id}.json"
        report = {"format": "jarvis.owner_quiet.sql.v1", "run_id": fixture.run_id,
                  "status": "running", "checks": [], "cleanup": None,
                  "provider_calls": 0, "hardware_evidence": False}
        failures = []
        connection = pool = None

        def save_report():
            descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")

        async def check(name, operation):
            try:
                await operation()
            except Exception as error:
                item = {"name": name, "status": "failed", "error_type": type(error).__name__,
                        "sqlstate": getattr(error, "sqlstate", None)}
                failures.append(item)
            else:
                item = {"name": name, "status": "passed"}
            report["checks"].append(item)
            save_report()

        save_report()
        try:
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
            schema = await fixture.create_schema(connection)
            migrations = tuple(m for m in load_migrations() if m.version <= 42)
            await apply_migrations(connection, schema=schema, fresh=True, migrations=migrations)
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
            contexts = ContextRepository(tasks)
            quiet = OwnerQuietRepository(pool)
            attention = AttentionRepository(tasks, lease_seconds=300)
            push = PushRepository(attention, SimpleNamespace(environment="sandbox", topic="com.jarvis.quiet.fixture",
                fallback_seconds=0, max_attempts=3, lease_seconds=30))
            saved = {}

            async def source(*, who=owner, sid=session, finality="final", kind="user_transcript", text="Please do not disturb."):
                record = await contexts.append(who, ContextIngress(session_id=sid,
                    source_namespace="quiet_sql_conversation", source_event_id=uuid4().hex, receipt_id=uuid4().hex,
                    kind=kind, trust="user_input" if kind == "user_transcript" else "sensor",
                    finality=finality, payload={"text": text}))
                return record, ToolInvocationContext(provider_call_id="fixture", request_event_id=record.id,
                                                      request_revision=record.revision)

            async def change(action="activate", **kwargs):
                _, authority = await source(text="Resume interruptions." if action == "resume" else "Please do not disturb.")
                revision = (await quiet.get(owner)).revision
                return await quiet.save(owner, session, authority,
                    QuietCommand(action=action, expected_revision=revision, **kwargs))

            async def defaults():
                self.assertEqual((await quiet.get(owner)).revision, 0)
                self.assertFalse((await quiet.get(other)).suppresses_unsolicited)
            await check("empty_owner_state_and_cross_owner_isolation", defaults)

            async def untrusted():
                before = (await quiet.get(owner)).revision
                for options in ({"finality": "provisional"}, {"kind": "observation"}):
                    _, authority = await source(**options)
                    with self.assertRaises(ToolInputError) as error:
                        await quiet.save(owner, session, authority, QuietCommand(action="activate", expected_revision=before))
                    self.assertEqual(error.exception.code, "request_evidence_unavailable")
                self.assertEqual((await quiet.get(owner)).revision, before)
            await check("provisional_caption_and_sensor_cannot_admit_command", untrusted)

            async def finalized():
                record, authority = await source()
                setter = create_quiet_tools(repository=quiet, owner_id=owner, session_id=session)[1]
                revision = (await quiet.get(owner)).revision
                command = QuietCommand(action="activate", expected_revision=revision)
                with bind_tool_invocation(provider_call_id="fixture", request_event_id=record.id, request_revision=record.revision):
                    result = await setter.handler(command.model_dump(mode="json"))
                self.assertEqual(result["status"], "success")
                self.assertTrue((await quiet.get(owner)).suppresses_unsolicited)
                self.assertFalse((await quiet.get(other)).suppresses_unsolicited)
                saved.update(authority=authority, command=command, accepted=result["accepted_revision"])
            await check("finalized_authorized_tool_commits_owner_quiet", finalized)

            async def retry_and_conflicts():
                result = await quiet.save(owner, session, saved["authority"], saved["command"])
                self.assertTrue(result["request_replayed"])
                self.assertEqual(result["accepted_revision"], saved["accepted"])
                with self.assertRaises(ToolInputError) as error:
                    await quiet.save(owner, session, saved["authority"],
                        QuietCommand(action="resume", expected_revision=result["quiet"]["revision"]))
                self.assertEqual(error.exception.code, "request_conflict")
                _, new = await source()
                with self.assertRaises(ToolInputError) as error:
                    await quiet.save(owner, session, new, QuietCommand(action="resume", expected_revision=0))
                self.assertEqual(error.exception.code, "revision_conflict")
                for who, sid in ((other, other_session), (owner, other_session)):
                    with self.assertRaises(ToolInputError) as error:
                        await quiet.save(who, sid, saved["authority"], QuietCommand(action="resume", expected_revision=0))
                    self.assertEqual(error.exception.code, "request_evidence_unavailable")
            await check("retry_dedupe_conflict_stale_revision_wrong_owner_and_session", retry_and_conflicts)

            async def reconstruction():
                before = await quiet.get(owner)
                fresh_pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1, timeout=10,
                                                       command_timeout=60, setup=setup)
                try:
                    current = await OwnerQuietRepository(fresh_pool).get(owner)
                    self.assertEqual(current.revision, before.revision)
                    self.assertEqual(current.policy, before.policy)
                    self.assertTrue(current.suppresses_unsolicited)
                finally:
                    await asyncio.wait_for(fresh_pool.close(), 10)
                await change("resume")
                replay = await quiet.save(owner, session, saved["authority"], saved["command"])
                self.assertTrue(replay["request_replayed"])
                self.assertEqual(replay["quiet"]["effective_state"], "resumed")
            await check("new_connection_reconstruction_resume_and_old_retry_no_resurrection", reconstruction)

            async def concurrent():
                revision = (await quiet.get(owner)).revision
                _, first = await source()
                _, second = await source()
                command = QuietCommand(action="activate", expected_revision=revision)
                outcomes = await asyncio.gather(quiet.save(owner, session, first, command),
                                               quiet.save(owner, session, second, command), return_exceptions=True)
                self.assertEqual(sum(isinstance(r, dict) for r in outcomes), 1)
                losers = [r for r in outcomes if isinstance(r, ToolInputError)]
                self.assertEqual(len(losers), 1)
                self.assertEqual(losers[0].code, "revision_conflict")
            await check("real_owner_lock_serializes_concurrent_revision_writers", concurrent)

            async def expiry():
                await change("resume")
                now = await connection.fetchval("select clock_timestamp()")
                deadline = now + timedelta(seconds=3)
                result = await change(expires_at=deadline, timezone="UTC")
                self.assertEqual(result["quiet"]["effective_state"], "quiet")
                await asyncio.sleep(max(0, (deadline - await connection.fetchval("select clock_timestamp()")).total_seconds()) + .1)
                current = await OwnerQuietRepository(pool).get(owner)
                self.assertEqual(current.effective_state, "expired")
                self.assertEqual(current.policy.episode.expires_at, deadline)
                self.assertEqual(current.revision, result["accepted_revision"])
            await check("exact_database_time_expiry_without_timestamp_edits", expiry)

            async def duty_only():
                _, authority = await source(text="Pause duty reminders for an hour.")
                now = await connection.fetchval("select clock_timestamp()")
                prefs = ResponsibilityPreferences(ResponsibilityRepository(pool))
                result = await prefs.save(owner, session, authority, SetNotificationPreferences(expected_revision=0,
                    policy=NotificationPolicy(timezone="UTC", paused_until=now + timedelta(hours=1))))
                saved["duty"] = result["notification_preferences"]
                self.assertFalse((await quiet.get(owner)).suppresses_unsolicited)
            await check("duty_pause_does_not_become_owner_quiet", duty_only)

            async def complete_work():
                await change()
                task = await tasks.create(owner, TaskCreate(request_id=uuid4(), title="Synthetic owner quiet work",
                    requested_outcome="Return a synthetic read-only result, without external actions.", session_id=session),
                    model="fixture-model")
                claim = await tasks.claim_next("quiet-sql-worker", ("frontier",), owner_id=owner)
                self.assertEqual(claim.task.id, task.id)
                context = TaskRunner(tasks, {}).context(claim)
                outcome = await HermesTaskResult(status="succeeded", answer="Synthetic result remains available.",
                    speech=[{"text": "Synthetic result remains available."}], evidence=[],
                    completed_operation_refs=[], uncertainties=[]).verified(context, SimpleNamespace())
                self.assertTrue(await tasks.finish(claim, outcome))
                intent = await connection.fetchval("select id from jarvis_attention_intents where owner_id=$1 and task_id=$2",
                                                  owner, task.id)
                self.assertIsNotNone(intent)
                saved.update(task=task, intent=intent)
                device = uuid4()
                await push.change_device(owner, device, DeviceRegistration(request_id=uuid4(), expected_revision=0,
                    environment="sandbox", topic="com.jarvis.quiet.fixture", token="ab" * 32,
                    permission="authorized", enabled=True))
            await check("authorized_work_finishes_while_quiet_without_provider", complete_work)

            async def blocked_claims_and_reads():
                self.assertIsNone(await attention.claim_next(owner, session, "quiet-voice"))
                self.assertIsNone(await push.claim_next(owner, "quiet-push"))
                self.assertEqual(await connection.fetchval(
                    "select count(*) from jarvis_attention_attempts where owner_id=$1", owner), 0)
                self.assertEqual((await tasks.get(owner, saved["task"].id)).status, TaskStatus.SUCCEEDED)
                detail = await attention.detail(owner, saved["intent"])
                self.assertIsNotNone(detail["expert_result"])
                self.assertTrue(await attention.list(owner))
                async with tasks.transaction(owner) as c:
                    self.assertTrue((await quiet_delivery_decision(c, owner, kind="solicited_reply", channel="voice")).allowed)
                self.assertTrue((await quiet.get(owner)).suppresses_unsolicited)
            await check("canonical_voice_push_claims_block_but_task_result_and_direct_reads_survive", blocked_claims_and_reads)

            async def voice_write_race():
                await change("resume")
                claim = await attention.claim_next(owner, session, "quiet-voice-positive-control")
                self.assertIsNotNone(claim)
                await change()
                with self.assertRaises(AttentionError) as error:
                    async with attention.guard(claim):
                        self.fail("Quiet must fence the pending writer.")
                self.assertEqual(error.exception.code, "owner_quiet_quiet")
                await attention.settle(claim, reason="fixture_quiet_no_output")
                self.assertEqual((await tasks.get(owner, saved["task"].id)).status, TaskStatus.SUCCEEDED)
            await check("eligible_voice_positive_control_then_actual_guard_rechecks_quiet", voice_write_race)

            async def push_write_race():
                await change("resume")
                # Resume no longer replays withheld originals. This positive
                # control explicitly requests this exact original inbox item.
                detail = await attention.detail(owner, saved["intent"])
                await attention.mutate(owner, saved["intent"], "replay", InboxMutation(
                    request_id=uuid4(), expected_generation=detail["generation"]))
                claim = await push.claim_next(owner, "quiet-push-positive-control")
                self.assertIsNotNone(claim)
                await change()
                self.assertIsNone(await push.prepare_submission(claim))
                phase = await connection.fetchval(
                    "select phase from jarvis_attention_attempts where owner_id=$1 and id=$2", owner, claim.attempt_id)
                self.assertEqual(phase, "claimed")
                await push.finish(claim, PushOutcome("failed", "fixture_quiet_no_submission", retry_seconds=1))
                self.assertIsNotNone((await attention.detail(owner, saved["intent"]))["expert_result"])
                async with pool.acquire() as c:
                    self.assertEqual(await read_preferences(c, owner), saved["duty"])
            await check("eligible_push_positive_control_then_preparation_block_duty_policy_unchanged", push_write_race)

            async def superseded_source():
                record, authority = await source()
                await contexts.append(owner, ContextIngress(session_id=session, source_namespace=record.source_namespace,
                    source_event_id=record.source_event_id, receipt_id=uuid4().hex, expected_revision=record.revision,
                    kind="user_transcript", trust="user_input", finality="revised", payload={"text": "Do not change my settings."}))
                revision = (await quiet.get(owner)).revision
                with self.assertRaises(ToolInputError) as error:
                    await quiet.save(owner, session, authority, QuietCommand(action="resume", expected_revision=revision))
                self.assertEqual(error.exception.code, "request_evidence_unavailable")
                self.assertEqual((await quiet.get(owner)).revision, revision)
            await check("superseded_request_cannot_resume", superseded_source)

            report["status"] = "failed" if failures else "passed"
        except Exception as error:
            report.update(status="failed", setup_error_type=type(error).__name__, sqlstate=getattr(error, "sqlstate", None))
            raise AssertionError("Quiet SQL setup/execution failed: " + type(error).__name__) from None
        finally:
            await cleanup_fixture_resources(fixture, report, save_report, pool=pool, connection=connection)
            print("QUIET_SQL_REPORT=" + str(report_path))
            print("QUIET_SQL_SUMMARY=" + json.dumps({"status": report["status"], "checks": report["checks"], "cleanup": report["cleanup"]}))
        self.assertEqual(failures, [], "Named SQL checks failed; see sanitized qualification report.")

