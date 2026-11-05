"""Real isolated PostgreSQL ASR correction; controlled inputs, no model calls."""
import asyncio
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

import asyncpg

from app.attention.notification_ledger.admission import TaskAdmissionAdapter
from app.live_state.models import SessionScope
from app.migrations.runner import apply_migrations, load_migrations, quote_schema
from app.runtime.feature_config import TaskLimits
from app.tasks.actions import TaskActionRepository
from app.tasks.catalog import ConfiguredTaskProvider
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError, TaskMutation, TaskStatus, TaskVoiceSource, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from app.tasks.voice_tools import BackgroundTaskToolModule, StartArguments
from app.tools.invocation import CanonicalRequestIdentity, bind_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry
from evals.second_brain.fixtures import DatabaseFixture
from test_notification_ledger_database import cleanup_fixture_resources


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
    os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Explicit disposable database configuration required")
class AdmissionCorrectionSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_canonical_revision_fences_replays_and_effect_barriers(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        fixture = DatabaseFixture(dsn, directory, test_name=self.id())
        report_path = directory / ("admission-corrections-" + fixture.run_id + ".json")
        report = {"format": "jarvis.admission_corrections.sql.v1", "run_id": fixture.run_id,
            "status": "running", "cases": [], "cleanup": None, "provider_calls": 0,
            "public_migration": False, "sql_mocks": False}
        connection = pool = None

        def save():
            fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")

        async def check(name, operation):
            try:
                async with asyncio.timeout(70):
                    evidence = await operation()
            except Exception as error:
                report["cases"].append({"name": name, "status": "failed", "error_type": type(error).__name__,
                                        "sqlstate": getattr(error, "sqlstate", None)})
                save()
                raise
            report["cases"].append({"name": name, "status": "passed", "evidence": evidence})
            save()

        save()
        try:
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=300)
            schema = await fixture.create_schema(connection)
            migrations = load_migrations()
            report["migration_ceiling"] = migrations[-1].version
            save()
            await asyncio.wait_for(apply_migrations(connection, schema=schema, fresh=True,
                                                   migrations=migrations), 300)
            path_sql = f"set search_path to {quote_schema(schema)},pg_catalog,extensions"
            await connection.execute(path_sql)

            async def setup(c):
                await c.execute(path_sql)
                await c.execute("set lock_timeout='8s'; set statement_timeout='30s'")

            pool = await asyncpg.create_pool(dsn, min_size=2, max_size=4, timeout=10,
                                            command_timeout=40, setup=setup)
            tasks = TaskRepository(pool, TaskLimits(lease_seconds=300))
            sources = ContextRepository(tasks)
            service = TaskService(tasks, {"frontier": ConfiguredTaskProvider("offline-fixture")})
            owner = fixture.owners[0]
            session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", owner)
            scope = SessionScope(owner_id=owner, session_id=session)
            adapter = TaskAdmissionAdapter(service)
            registry = create_tool_registry(modules=(BackgroundTaskToolModule(service),),
                context=ToolBuildContext(None, owner, session))
            args = StartArguments(title="Research the red book", requested_outcome="Research the red book.")
            revised = args.model_copy(update={"title": "Research the blue book", "requested_outcome": "Research the blue book."})
            first = updated = claim = action = token = original_source = revised_source = None
            actions = TaskActionRepository(tasks)

            async def source(key, revision, *, provisional=False):
                return await sources.append(owner, ContextIngress(session_id=session,
                    source_namespace="qualification:asr", source_event_id=key, receipt_id=str(revision),
                    expected_revision=revision - 1, kind="user_transcript", trust="user_input",
                    finality="provisional" if provisional else "final" if revision == 1 else "revised",
                    payload={"text": f"Synthetic canonical instruction {key} revision {revision}."}))

            async def admit(record, arguments=args):
                return await adapter.create(scope, record.id, record.revision, arguments)

            async def voice(record, arguments=args):
                with bind_tool_invocation(provider_call_id=str(uuid4()), request_event_id=record.id,
                        request_revision=record.revision, request_identity=CanonicalRequestIdentity.from_record(record)):
                    return await registry.get("start_background_task").handler(arguments.model_dump(mode="json"))

            async def original_two_origins():
                nonlocal first, original_source
                original_source = await source("shared", 1)
                first, native = await asyncio.gather(admit(original_source), voice(original_source))
                self.assertEqual(native["status"], "success", native)
                self.assertEqual(native["task"]["task_ref"], tasks.reference(first.id))
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_background_tasks where owner_id=$1", owner), 1)
                return {"tasks": 1, "revision": first.revision}

            async def running_correction_with_submitted_effect():
                nonlocal claim, action, token, updated, revised_source
                claim = await tasks.claim_next("fixture-worker", ("frontier",), owner_id=owner)
                self.assertIsNotNone(claim)
                self.assertEqual(claim.task.id, first.id)
                # Fixture-only exact operation grant, followed by real action
                # admission/submission receipts. No external effect is executed.
                async with tasks.transaction(owner) as c:
                    payload = await tasks.payload(c, owner, {"synthetic_operation": "no external execution"})
                    grant = await c.fetchval("""insert into jarvis_operation_grants(owner_id,capability,
                        operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                        values($1,'fixture_effect','fixture-effect',$2,$3,1,$4,clock_timestamp()+interval '1 hour')
                        returning id""", owner, original_source.id, original_source.revision, payload)
                    await c.execute("insert into jarvis_task_grants(owner_id,task_id,grant_id) values($1,$2,$3)", owner, first.id, grant)
                action = await actions.prepare(claim, "fixture-effect", grant, "fixture_effect",
                                               {"synthetic_operation": "no external execution"})
                await actions.authorize(claim, action["id"])
                token = await actions.begin_submission(claim, action["id"])
                revised_source = await source("shared", 2)
                updated, native = await asyncio.gather(admit(revised_source, revised), voice(revised_source, revised))
                self.assertEqual(native["status"], "success", native)
                self.assertEqual(native["task"]["revision"], 2)
                self.assertEqual((updated.id, updated.revision), (first.id, 2))
                self.assertGreater(updated.fence, claim.fence)
                self.assertEqual(updated.status, TaskStatus.UNKNOWN)
                self.assertEqual(updated.pending_status, "queued")
                self.assertEqual(updated.request_event_id, revised_source.id)
                self.assertEqual(updated.requested_outcome, revised.requested_outcome)
                self.assertFalse(await tasks.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "stale"})))
                row = await connection.fetchrow("select status,submission_token from jarvis_task_actions where owner_id=$1 and id=$2", owner, action["id"])
                self.assertEqual(dict(row), {"status": "submitting", "submission_token": token})
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_actions where owner_id=$1", owner), 1)
                return {"task_revision": 2, "old_worker_rejected": True, "submission_unchanged": True, "state": updated.status.value}

            async def restarted_json_replay():
                # New repository/service instances, default asyncpg JSON text.
                restarted = TaskAdmissionAdapter(TaskService(TaskRepository(pool, tasks.limits), {}))
                replay = await restarted.create(scope, revised_source.id, 2, revised)
                self.assertEqual((replay.id, replay.revision), (first.id, 2))
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_requests where owner_id=$1 and task_id=$2", owner, first.id), 2)
                return {"revision": replay.revision, "receipts": 2, "provider_configuration_required_for_replay": False}

            async def conflicts_and_stale_authority():
                for change in ({"requested_outcome": "Unrelated work"}, {"mode": "consult"},
                               {"requested_capabilities": ["sandbox_network"]}, {"knowledge_area_id": uuid4()}):
                    with self.assertRaises(TaskError):
                        await admit(revised_source, revised.model_copy(update=change))
                with self.assertRaises(TaskError):
                    await admit(original_source)
                self.assertEqual((await tasks.get(owner, first.id)).revision, 2)
                return {"conflicting_or_stale_proposals_rejected": 5}

            async def settled_effect_not_replayed():
                await actions.record_receipt(owner, first.id, action["id"], token, "accepted", {"fixture": "receipt only"})
                third = await source("shared", 3)
                with self.assertRaises(TaskError) as caught:
                    await admit(third, revised)
                self.assertEqual(caught.exception.code, "task_reconciliation_required")
                self.assertEqual((await tasks.get(owner, first.id)).revision, 2)
                return {"settled_submission_requires_explicit_review": True}

            async def unrelated_edit_wins():
                other = await source("explicit-other-edit", 1)
                task = await service.mutate(owner, first.id, "update", TaskMutation(request_id=uuid4(),
                    expected_revision=2, message="Explicit independent newer instructions."),
                    source=TaskVoiceSource(other.id, other.revision, session))
                fourth = await source("shared", 4)
                with self.assertRaises(TaskError):
                    await admit(fourth, revised)
                current = await tasks.get(owner, first.id)
                self.assertEqual(current.request_event_id, other.id)
                self.assertEqual(current.revision, task.revision)
                return {"unrelated_current_instructions_preserved": True}

            async def provisional_does_not_authorize():
                provisional = await source("provisional", 1, provisional=True)
                with self.assertRaises(TaskError) as caught:
                    await admit(provisional)
                self.assertEqual(caught.exception.code, "task_request_not_finalized")
                native = await voice(provisional)
                self.assertNotEqual(native["status"], "success")
                return {"router_and_voice_rejected_provisional": True}

            async def cancel_not_undone():
                original = await source("cancelled", 1)
                task = await admit(original)
                await service.mutate(owner, task.id, "cancel", TaskMutation(request_id=uuid4(), expected_revision=1),
                    source=TaskVoiceSource(original.id, 1, session))
                newer = await source("cancelled", 2)
                with self.assertRaises(TaskError):
                    await admit(newer, revised)
                current = await tasks.get(owner, task.id)
                self.assertEqual((current.status, current.revision), (TaskStatus.CANCELLED, 2))
                return {"cancelled_task_not_reopened": True}

            async def conflicting_corrections_share_one_winner():
                original = await source("competing", 1)
                task = await admit(original)
                newer = await source("competing", 2)
                outcomes = await asyncio.gather(admit(newer, revised),
                    admit(newer, revised.model_copy(update={"requested_outcome": "A competing objective."})),
                    return_exceptions=True)
                self.assertEqual(sum(isinstance(item, TaskError) for item in outcomes), 1, outcomes)
                self.assertEqual((await tasks.get(owner, task.id)).revision, 2)
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_requests where owner_id=$1 and task_id=$2", owner, task.id), 2)
                return {"one_revision_winner": True, "receipt_count": 2}

            async def withdrawal_at_mutation_boundary():
                original = await source("withdrawal", 1)
                task = await admit(original)
                newer = await source("withdrawal", 2)
                original_mutate = tasks.mutate

                async def withdraw_then_mutate(*values, **kwargs):
                    async with tasks.transaction(owner) as c:
                        await c.execute("""update jarvis_context_payloads set availability='deleted',content=null,
                            removed_at=clock_timestamp() where owner_id=$1 and id=$2""", owner, original.payload_id)
                    return await original_mutate(*values, **kwargs)

                with patch.object(tasks, "mutate", side_effect=withdraw_then_mutate):
                    with self.assertRaises(TaskError):
                        await admit(newer, revised)
                self.assertEqual((await tasks.get(owner, task.id)).revision, 1)
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_requests where owner_id=$1 and task_id=$2", owner, task.id), 1)
                return {"withdrawal_rechecked_under_owner_lock": True, "correction_not_committed": True}

            async def receipt_failure_rolls_back_revision():
                original = await source("rollback", 1)
                task = await admit(original)
                newer = await source("rollback", 2)
                await connection.execute("""create function asr_fixture_abort() returns trigger language plpgsql as $$
                    begin if new.operation='update' and new.task_id=tg_argv[0]::uuid then
                        raise exception 'synthetic correction receipt failure' using errcode='P0001';
                    end if; return new; end $$""")
                await connection.execute(f"""create trigger asr_fixture_abort after insert on jarvis_task_requests
                    for each row execute function asr_fixture_abort('{task.id}')""")
                try:
                    with self.assertRaises(asyncpg.RaiseError):
                        await admit(newer, revised)
                finally:
                    await connection.execute("drop trigger asr_fixture_abort on jarvis_task_requests")
                    await connection.execute("drop function asr_fixture_abort()")
                current = await tasks.get(owner, task.id)
                self.assertEqual((current.revision, current.fence, current.request_event_id),
                                 (task.revision, task.fence, original.id))
                retried = await admit(newer, revised)
                self.assertEqual(retried.revision, 2)
                return {"failed_transaction_rolled_back": True, "explicit_retry_revision": 2}

            for name, operation in (
                ("two_origins_one_task", original_two_origins),
                ("actual_revision_old_worker_fence_and_uncertain_action_barrier", running_correction_with_submitted_effect),
                ("restart_json_text_idempotent_replay", restarted_json_replay),
                ("incompatible_and_stale_proposals", conflicts_and_stale_authority),
                ("settled_submission_no_blind_replay", settled_effect_not_replayed),
                ("unrelated_task_update_preserved", unrelated_edit_wins),
                ("provisional_caption_no_authority", provisional_does_not_authorize),
                ("cancel_not_undone", cancel_not_undone),
                ("competing_corrections_one_winner", conflicting_corrections_share_one_winner),
                ("source_withdrawal_at_effect_boundary", withdrawal_at_mutation_boundary),
                ("receipt_failure_atomic_rollback", receipt_failure_rolls_back_revision),
            ):
                await check(name, operation)
            report["status"] = "passed"
        except Exception as error:
            report.update(status="failed", error_type=type(error).__name__, sqlstate=getattr(error, "sqlstate", None))
            raise
        finally:
            await cleanup_fixture_resources(fixture, report, save, pool=pool, connection=connection)
        self.assertEqual(report["cleanup"]["remaining_schemas"], 0)
        self.assertEqual(report["cleanup"]["remaining_public_owner_rows"], 0)
        print(json.dumps({"report": str(report_path), "manifest": str(fixture.path),
                          "cases_passed": len(report["cases"]), "cleanup": report["cleanup"]}))
