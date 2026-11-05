"""Actual manifested PostgreSQL readonly cancellation/recovery qualification.

No SQL mocks, provider, Hermes execution or public migration. An owned inert
child supplies a genuine local cleanup proof to the trusted reconciliation
boundary; this is not another agent runtime or native-Hermes acceptance claim.
Missing explicit DB gates fail, rather than count skipped tests as acceptance.
"""
import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from uuid import uuid4

import asyncpg

from app.live_state.models import SessionScope
from app.migrations.runner import apply_migrations, load_migrations, quote_schema
from app.realtime.live_session_capture import LiveCaptionCapture
from app.runtime.feature_config import TaskLimits
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_repository import ContextRepository
from app.tasks.models import ClaimLost, Reconciliation, TaskError, TaskStatus, WorkerOutcome
from app.tasks.readonly import CAPABILITY, ReadonlyTaskAdmission
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from app.voice_routes.contracts import RouteIdentity, TranscriptRevision
from app.voice_routes.live_adapter import LiveTranscriptPart
from app.voice_routes.live_protocol import LiveDelegationRequest, LiveTranscriptFragment
from evals.second_brain.fixtures import DatabaseFixture
from test_notification_ledger_database import cleanup_fixture_resources


class ReadonlyCancelSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_manifested_cancellation_recovery_and_rollback(self):
        dsn = os.environ.get("JARVIS_SECOND_BRAIN_TEST_DB_URL")
        directory = os.environ.get("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR")
        self.assertTrue(dsn and directory, "Explicit DB and manifested-fixture gates are required; no skipped acceptance")
        fixture = DatabaseFixture(dsn, Path(directory), test_name=self.id())
        report_path = Path(directory) / ("readonly-cancel-sql-" + fixture.run_id + ".json")
        report = {"format":"jarvis.readonly_cancel.sql.v1", "run_id":fixture.run_id, "status":"running",
            "cases":[], "cleanup":None, "sql_mocks":False, "public_migration":False,
            "provider_calls":0, "hermes_executions":0, "owned_inert_children":0,
            "cleanup_proof_boundary":"owned inert child exit; not a Hermes/provider receipt"}
        connection = pool = child = capture = captions = None
        failed = None

        def save():
            fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")

        async def check(name, operation):
            nonlocal failed
            try:
                async with asyncio.timeout(60):
                    evidence = await operation()
            except Exception as error:
                failed = {"name":name, "status":"failed", "error_type":type(error).__name__,
                          "sqlstate":getattr(error, "sqlstate", None)}
                report["cases"].append(failed)
                save()
                raise
            report["cases"].append({"name":name, "status":"passed", "evidence":evidence})
            save()

        save()
        try:
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=300)
            schema = await fixture.create_schema(connection)
            migrations = load_migrations()
            report["migration_ceiling"] = migrations[-1].version
            save()
            await asyncio.wait_for(apply_migrations(connection, schema=schema, fresh=True, migrations=migrations), 300)
            path_sql = f"set search_path to {quote_schema(schema)},pg_catalog,extensions"
            await connection.execute(path_sql)

            async def setup(c):
                await c.execute(path_sql)
                await c.execute("set lock_timeout='8s'; set statement_timeout='30s'")

            pool = await asyncpg.create_pool(dsn, min_size=2, max_size=4, timeout=10,
                                            command_timeout=40, setup=setup)
            tasks = TaskRepository(pool, TaskLimits(lease_seconds=300))
            rival = TaskRepository(pool, tasks.limits)
            sources = ContextRepository(tasks)
            registry = TaskWorkerRegistry(pool)
            providers = {"frontier":ConfiguredTaskProvider("fixture-readonly-sql")}
            service = TaskService(tasks, providers, registry=registry)
            admission = ReadonlyTaskAdmission(service)
            owner, other = fixture.owners
            session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", owner)
            scope = SessionScope(owner_id=owner, session_id=session)
            route = RouteIdentity(scope=scope, route_id=uuid4(), route_generation=1)
            audio = str(uuid4())
            capture = SessionContextCapture(sources, owner_id=owner, session_id=session, provider="openai", audio_session_id=audio)
            captions = LiveCaptionCapture(capture, SimpleNamespace(owner_id=owner, jarvis_session_id=session,
                audio_session_id=audio), observation_backed=True)
            await registry.advertise(worker_id="readonly-sql-worker", owner_id=owner, host_id="readonly-sql-host",
                providers=providers, ttl_seconds=300, capabilities=(CAPABILITY,))
            identity = uuid4()
            parts = (LiveTranscriptPart(identity, 1, LiveTranscriptFragment("synthetic-sql-speech", "user",
                                                                          "What number is visible?", 1, 90)),)
            self.assertTrue(captions.transcript(TranscriptRevision(route=route, transcript_id=identity,
                revision=1, observed_at=datetime.now(UTC), role="user", text=parts[0].fragment.delta, final=False), parts))
            originals = await captions.observations(parts)
            self.assertTrue(await capture.flush())
            request = LiveDelegationRequest("synthetic-sql-delegation", "sql-d1", 90)
            receipt = claim = recovery = proof = None
            initial_deadline = None

            async def admit():
                async with tasks.transaction(owner) as c:
                    return await admission.admit(c, route=route, audio_session_id=audio, delegation=request,
                        observations=originals, visual_context="No legible current number is established.")

            async def snapshot():
                row = await connection.fetchrow("""select t.status,t.pending_status,t.revision,t.fence,
                    t.current_attempt_id,t.result_payload_id,t.attempt_count,
                    a.status as attempt_status,a.provider_state,a.worker_id,a.fence as attempt_fence,
                    a.lease_until,a.lease_until>clock_timestamp() as lease_valid,
                    (select count(*) from jarvis_context_events where owner_id=t.owner_id) as source_events,
                    (select count(*) from jarvis_context_payloads where owner_id=t.owner_id) as payloads,
                    (select count(*) from jarvis_task_events where owner_id=t.owner_id and task_id=t.id) as task_events
                    from jarvis_background_tasks t left join jarvis_task_attempts a
                    on a.owner_id=t.owner_id and a.id=$3 where t.owner_id=$1 and t.id=$2""",
                    owner, receipt.task.id, claim.attempt_id)
                return dict(row)

            async def install_abort_trigger(kind):
                condition = ("new.lease_until < old.lease_until" if kind == "lease"
                             else "new.provider_state='terminal' and old.provider_state<>'terminal'")
                await connection.execute(f"""create function readonly_sql_abort() returns trigger language plpgsql as $$
                    begin
                        if new.id=tg_argv[0]::uuid and {condition} then
                            raise exception 'synthetic readonly transaction rollback' using errcode='P0001';
                        end if;
                        return new;
                    end $$""")
                await connection.execute(f"""create trigger readonly_sql_abort_trigger after update on jarvis_task_attempts
                    for each row execute function readonly_sql_abort('{claim.attempt_id}')""")

            async def drop_abort_trigger():
                await connection.execute("drop trigger readonly_sql_abort_trigger on jarvis_task_attempts")
                await connection.execute("drop function readonly_sql_abort()")

            async def canonical_admission_and_claim():
                nonlocal receipt, claim, initial_deadline, child
                receipt = await admit()
                replay = await admit()
                self.assertEqual(replay.task.id, receipt.task.id)
                self.assertFalse(receipt.source.authorizes_request)
                self.assertTrue(all(not item.authorizes_request for item in originals))
                self.assertIsNone(await tasks.claim_next("missing-capability", ("frontier",),
                    owner_id=owner, worker_capabilities=()))
                claim = await tasks.claim_next("readonly-sql-worker", ("frontier",), owner_id=owner,
                    provider_models={"frontier":"fixture-readonly-sql"}, worker_capabilities=(CAPABILITY,))
                self.assertIsNotNone(claim)
                self.assertEqual(claim.task.id, receipt.task.id)
                # No agent/model/gateway: this owned child only waits for EOF.
                child = await asyncio.create_subprocess_exec(sys.executable, "-c", "import sys; sys.stdin.buffer.read()",
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                report["owned_inert_children"] = 1
                await tasks.start_runtime(claim, "fixture-cleanup:"+uuid4().hex,
                    {"format":"fixture.cleanup.v1", "phase":"owned_child_running"}, host_id="readonly-sql-host")
                state = await snapshot()
                self.assertEqual((state["status"], state["attempt_status"], state["provider_state"]),
                                 ("running", "running", "live"))
                self.assertTrue(state["lease_valid"])
                initial_deadline = state["lease_until"]
                self.assertIsNone(await tasks.claim_recovery("premature-recovery", owner_id=owner,
                    host_id="readonly-sql-host", backend_kinds=("frontier",)))
                return {"same_task_on_replay":True, "real_registry":True, "real_claim":True,
                        "provisional_sources_do_not_authorize_actions":True, "premature_recovery_rejected":True}

            async def cancellation_rollback():
                before = await snapshot()
                await install_abort_trigger("lease")
                try:
                    with self.assertRaises(asyncpg.RaiseError) as raised:
                        await admission.cancel(receipt, reason="cancelled")
                    self.assertEqual(raised.exception.sqlstate, "P0001")
                finally:
                    await drop_abort_trigger()
                self.assertEqual(await snapshot(), before)
                original = await sources.get(owner, receipt.source.id, session_id=session)
                self.assertTrue(original.is_current)
                self.assertEqual(original.revision, receipt.source.revision)
                return {"sql_trigger_error":"P0001", "task_source_events_payloads_lease_rolled_back":True}

            async def committed_cancellation():
                before = await snapshot()
                self.assertTrue(await admission.cancel(receipt, reason="cancelled"))
                state = await snapshot()
                self.assertEqual((state["status"], state["pending_status"]), ("cancel_requested", "cancelled"))
                self.assertEqual((state["revision"], state["fence"]), (before["revision"]+1, before["fence"]+1))
                self.assertEqual(state["current_attempt_id"], claim.attempt_id)
                self.assertEqual((state["attempt_status"], state["provider_state"]), ("running", "live"))
                self.assertFalse(state["lease_valid"])
                self.assertLess(state["lease_until"], initial_deadline)
                self.assertIsNone(state["result_payload_id"])
                self.assertIsNone(child.returncode)
                original = await sources.get(owner, receipt.source.id, session_id=session)
                self.assertFalse(original.is_current)
                return {"status":state["status"], "pending_status":state["pending_status"],
                        "same_attempt":True, "lease_expired_without_wait":True, "child_not_claimed_complete":True}

            async def exclusive_recovery():
                nonlocal recovery
                claims = await asyncio.gather(*[repository.claim_recovery(name, owner_id=owner,
                    host_id="readonly-sql-host", backend_kinds=("frontier",))
                    for repository, name in ((tasks,"controller-one"), (rival,"controller-two"))])
                winners = [value for value in claims if value is not None]
                self.assertEqual(len(winners), 1)
                recovery = winners[0]
                self.assertEqual(recovery.attempt_id, claim.attempt_id)
                self.assertTrue(recovery.recovering)
                self.assertEqual(recovery.task.pending_status, "cancelled")
                state = await snapshot()
                self.assertEqual(state["attempt_status"], "recovering")
                self.assertTrue(state["lease_valid"])
                self.assertEqual(state["fence"], state["attempt_fence"])
                self.assertEqual(state["fence"], recovery.fence)
                with self.assertRaises(ClaimLost):
                    await tasks.renew(claim)
                self.assertFalse(await tasks.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED,
                    {"answer":"Synthetic stale answer must never become current."})))
                self.assertIsNone((await tasks.get(owner, receipt.task.id)).result_payload_id)
                self.assertIsNone(await rival.claim_recovery("third-controller", owner_id=owner,
                    host_id="readonly-sql-host", backend_kinds=("frontier",)))
                self.assertIsNone(await rival.claim_recovery("foreign-controller", owner_id=other,
                    host_id="readonly-sql-host", backend_kinds=("frontier",)))
                return {"concurrent_controllers":2, "exclusive_winners":1, "same_attempt":True,
                        "old_worker_renew_and_result_rejected":True, "foreign_owner_rejected":True}

            async def joined_cleanup_and_recovery_rollback():
                nonlocal proof
                self.assertIsNone(child.returncode)
                self.assertEqual((await tasks.get(owner, receipt.task.id)).status, TaskStatus.CANCEL_REQUESTED)
                child.stdin.close()
                await asyncio.wait_for(child.wait(), 5)
                self.assertEqual(child.returncode, 0)
                proof = Reconciliation("terminal", WorkerOutcome(TaskStatus.PAUSED,
                    {"format":"fixture.cleanup.v1", "owned_child_joined":True, "physical_completion":False},
                    reason="owned_inert_child_joined"))
                before = await snapshot()
                await install_abort_trigger("terminal")
                try:
                    with self.assertRaises(asyncpg.RaiseError):
                        await tasks.resolve_recovery(recovery, proof)
                finally:
                    await drop_abort_trigger()
                self.assertEqual(await snapshot(), before)
                self.assertEqual(before["status"], "cancel_requested")
                self.assertEqual(before["attempt_status"], "recovering")
                return {"owned_child_exit_code":child.returncode, "cleanup_proof_before_terminal_commit":True,
                        "failed_recovery_commit_rolled_back":True}

            async def terminal_and_no_replay():
                self.assertEqual(child.returncode, 0)
                await tasks.resolve_recovery(recovery, proof)
                state = await snapshot()
                self.assertEqual((state["status"], state["pending_status"], state["current_attempt_id"],
                                  state["result_payload_id"]), ("cancelled", None, None, None))
                self.assertEqual((state["attempt_status"], state["provider_state"]), ("interrupted", "terminal"))
                with self.assertRaises(ClaimLost):
                    await tasks.renew(recovery)
                self.assertFalse(await admission.cancel(receipt, reason="cancelled"))
                with self.assertRaises(TaskError):
                    await admit()
                self.assertIsNone(await tasks.claim_recovery("after-terminal", owner_id=owner,
                    host_id="readonly-sql-host", backend_kinds=("frontier",)))
                self.assertIsNone(await tasks.claim_next("after-terminal", ("frontier",), owner_id=owner,
                                                       worker_capabilities=(CAPABILITY,)))
                counts = await connection.fetchrow("""select
                    (select count(*) from jarvis_background_tasks where owner_id=$1) as tasks,
                    (select count(*) from jarvis_task_attempts where owner_id=$1) as attempts,
                    (select count(*) from jarvis_task_actions where owner_id=$1) as actions,
                    (select count(*) from jarvis_task_grants where owner_id=$1) as grants,
                    (select count(*) from jarvis_background_tasks where owner_id=$1 and result_payload_id is not null) as results""", owner)
                self.assertEqual(dict(counts), {"tasks":1, "attempts":1, "actions":0, "grants":0, "results":0})
                return dict(counts) | {"status":"cancelled", "attempt_provider_state":"terminal",
                                      "pending_status":None, "replay_and_reclaim_rejected":True}

            for name, operation in (
                ("canonical_admission_claim_and_early_recovery_rejection", canonical_admission_and_claim),
                ("cancellation_transaction_rollback", cancellation_rollback),
                ("committed_cancellation_expires_same_attempt_lease", committed_cancellation),
                ("exclusive_recovery_and_stale_worker_fencing", exclusive_recovery),
                ("joined_cleanup_proof_and_recovery_rollback", joined_cleanup_and_recovery_rollback),
                ("cancelled_terminal_without_duplicate_attempt_effect_or_result", terminal_and_no_replay),
            ):
                await check(name, operation)
            report["status"] = "passed"
        except Exception as error:
            report.update(status="failed", error_type=type(error).__name__, sqlstate=getattr(error,"sqlstate",None))
            if failed is None:
                failed = {"name":"bootstrap_or_setup", "error_type":type(error).__name__,
                          "sqlstate":getattr(error,"sqlstate",None)}
            save()
        finally:
            try:
                if child is not None and child.returncode is None:
                    child.terminate()
                    try:
                        await asyncio.wait_for(child.wait(), 5)
                    except TimeoutError:
                        child.kill()
                        await child.wait()
                if captions is not None:
                    await captions.close()
                if capture is not None:
                    await capture.close()
            finally:
                await cleanup_fixture_resources(fixture, report, save, pool=pool, connection=connection)
            self.assertEqual(report["cleanup"]["remaining_schemas"], 0)
            self.assertEqual(report["cleanup"]["remaining_public_owner_rows"], 0)
            print(json.dumps({"report":str(report_path), "status":report["status"],
                "sql_cases_passed":sum(case["status"] == "passed" for case in report["cases"]),
                "cleanup_remaining_schemas":report["cleanup"]["remaining_schemas"],
                "cleanup_remaining_public_owner_rows":report["cleanup"]["remaining_public_owner_rows"]}))
        self.assertIsNone(failed, json.dumps(failed))
        self.assertEqual(len(report["cases"]), 6)


if __name__ == "__main__":
    unittest.main()
