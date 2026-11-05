"""P2: actual PostgreSQL, independent writer process, real Live State owners.

No SQL/runtime/binder mocks. Synthetic worker output and visual observations are
inputs, not proof of model reasoning, visual detection, delivery or hardware.
Explicit fixture gates fail rather than silently skipping acceptance.
"""
import asyncio
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import sys
import time
import unittest
from uuid import uuid4

import asyncpg

from app.knowledge.models import SourceError
from app.knowledge.observations import ObservationPublication
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.sources import SourceRepository
from app.live_state.models import SessionScope
from app.live_state.projection import select_voice_context
from app.live_state.runtime import LiveStateRuntime
from app.migrations.runner import apply_migrations, load_migrations, quote_schema
from app.perception.models import ObservationFrameLink, VisualObservation
from app.perception.repository import PerceptionRepository
from app.runtime.events import DomainEventBus
from app.runtime.feature_config import TaskLimits
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate
from app.tasks.repository import TaskRepository
from evals.second_brain.fixtures import DatabaseFixture
from evals.second_brain.lost_wakeup_writer import sanitized
from test_notification_ledger_database import cleanup_fixture_resources


BOUND_SECONDS = 8.0


def now():
    return datetime.now(UTC)


class ObservedBus(DomainEventBus):
    """Observe the real process-local bus; never forward the child commit ACK."""
    def __init__(self):
        super().__init__()
        self.publishes = 0

    async def publish(self, event):
        self.publishes += 1
        return await super().publish(event)


class LiveStateLostWakeupSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_cross_process_polling_reconstruction_and_expiry(self):
        dsn = os.environ.get("JARVIS_SECOND_BRAIN_TEST_DB_URL")
        directory = os.environ.get("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR")
        self.assertTrue(dsn and directory, "Explicit manifested PostgreSQL is required; no skipped acceptance")
        fixture = DatabaseFixture(dsn, Path(directory), test_name=self.id())
        report_path = Path(directory) / ("live-state-lost-wakeup-" + fixture.run_id + ".json")
        report = {"format": "jarvis.live_state.lost_wakeup.sql.v1", "run_id": fixture.run_id,
            "status": "running", "cases": [], "cleanup": None, "reader_pid": os.getpid(),
            "writer": None, "sql_mocks": False, "runtime_mocks": False, "provider_calls": 0,
            "public_migration": False, "bus_forwarding": False,
            "refresh_interval_seconds": 1.0, "refresh_timeout_seconds": 2.0,
            "catchup_bound_seconds": BOUND_SECONDS, "reconstructed_process": False}
        connection = pool = child = stderr_drain = None
        runtimes, buses = [], []
        current = first = second = None
        failure = None
        replies, bound = {}, {}

        def save():
            descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, default=str)
                stream.write("\n")

        async def check(name, operation):
            try:
                async with asyncio.timeout(45):
                    evidence = await operation()
            except Exception as error:
                report["cases"].append({"name": name, "status": "failed", **sanitized(error)})
                save()
                raise
            report["cases"].append({"name": name, "status": "passed", "evidence": evidence})
            save()

        async def wait_for(predicate, *, runtime, since=None):
            started = time.monotonic_ns() if since is None else since
            while (time.monotonic_ns() - started) / 1e9 < BOUND_SECONDS:
                value = predicate()
                if value:
                    return value, (time.monotonic_ns() - started) / 1e6
                if runtime.failure is not None or runtime.requires_reconstruction:
                    self.fail("Production Live State requires reconstruction: " + repr(runtime.failure))
                await asyncio.sleep(.025)
            self.fail("Canonical catch-up exceeded bound; worker=" + repr(runtime.worker.last_error)
                      + "; perception=" + repr(runtime.perception.worker.last_error))

        def task_entry(runtime, identity, status=None):
            return next((entry for entry in runtime.service.snapshot().entries
                         if entry.kind == "task" and entry.task.aggregate_id == identity
                         and entry.validity == "active" and (status is None or entry.status == status)), None)

        def observation(runtime):
            return next((entry for entry in runtime.service.snapshot().entries
                         if entry.kind == "observation" and entry.validity == "active"), None)

        async def bind_tasks(runtime):
            selection = select_voice_context(runtime.service.snapshot(), now=now())
            self.assertIn("task:", " ".join(selection.selected_entry_ids))
            binding = await runtime.bind(selection)
            async with runtime.guard(binding):
                pass
            return selection, binding

        async def reply():
            line = await asyncio.wait_for(child.stdout.readline(), 20)
            self.assertTrue(line, "Independent writer closed its result channel")
            value = json.loads(line)
            self.assertNotEqual(value["kind"], "failed", json.dumps(value))
            self.assertEqual(value["pid"], child.pid)
            return value

        async def command(operation, identity=None):
            child.stdin.write((json.dumps({"operation": operation, "task": str(identity)}) + "\n").encode())
            await child.stdin.drain()
            value = await reply()
            self.assertEqual(value["kind"], "stopped" if operation == "stop" else "committed")
            return value

        async def drain_stderr():
            size = 0
            while block := await child.stderr.read(4096):
                size += len(block)
                if size > 65536:
                    raise AssertionError("Writer diagnostic budget exceeded")
            return size  # Do not persist potentially sensitive third-party diagnostics.

        async def close_runtime(runtime):
            await asyncio.wait_for(runtime.close(), 10)
            for owned in (runtime.worker._task, runtime.perception.worker._task,
                          runtime._monitor, runtime.service._worker, runtime._close_task):
                self.assertTrue(owned.done(), "Runtime close left an owned task unjoined")
            self.assertFalse(runtime.worker.bus._handlers, "Runtime close leaked bus subscriptions")

        save()
        try:
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=300)
            schema = await fixture.create_schema(connection)
            migrations = load_migrations()
            report["migration_ceiling"] = migrations[-1].version
            save()
            await asyncio.wait_for(apply_migrations(connection, schema=schema, fresh=True,
                                                   migrations=migrations), 300)
            search_path = f"set search_path to {quote_schema(schema)},pg_catalog,extensions"
            await connection.execute(search_path)

            async def setup(c):
                await c.execute(search_path)
                await c.execute("set lock_timeout='5s'; set statement_timeout='15s'")

            pool = await asyncpg.create_pool(dsn, min_size=2, max_size=4, timeout=10,
                                            command_timeout=20, setup=setup)
            owner = fixture.owners[0]
            session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", owner)
            area = await connection.fetchval("insert into jarvis_knowledge_areas(owner_id,name,domain) "
                "values($1,'Lost wakeup fixture','test') returning id", owner)
            sources = SourceRepository(pool)
            tasks = TaskRepository(pool, TaskLimits(lease_seconds=300))
            context = ContextRepository(tasks, publication=CanonicalSourcePublication(sources, area))
            perceptions = PerceptionRepository(pool,
                observation_publication=ObservationPublication(context, owner_id=owner))
            scope = SessionScope(owner_id=owner, session_id=session)
            created = [await tasks.create(owner, TaskCreate(request_id=uuid4(), session_id=session,
                title="Synthetic task " + str(index), requested_outcome="Return an isolated test result."),
                model="lost-wakeup-" + uuid4().hex) for index in (1, 2)]

            async def start_runtime():
                bus = ObservedBus()
                buses.append(bus)
                runtime = LiveStateRuntime(scope=scope, sources=sources, context=context,
                    area_selections=None, bus=bus, access_check=lambda requested: self.assertEqual(requested, owner),
                    tasks=tasks, working_memory=perceptions, clock=now)
                runtimes.append(runtime)
                await runtime.start()
                await wait_for(lambda: runtime.service.snapshot().readiness == "ready", runtime=runtime)
                return runtime

            async def initial_state_and_writer():
                nonlocal current, first, child, stderr_drain
                first = current = await start_runtime()
                for task in created:
                    self.assertIsNotNone(task_entry(first, task.id, "queued"))
                _, bound["queued"] = await bind_tasks(first)
                environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1]), "PYTHONUNBUFFERED": "1",
                    "JARVIS_SECOND_BRAIN_TEST_DB_URL": dsn}
                child = await asyncio.create_subprocess_exec(sys.executable, "-m",
                    "evals.second_brain.lost_wakeup_writer", env=environment,
                    cwd=Path(__file__).resolve().parents[1], stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=65536)
                report["writer"] = {"pid": child.pid, "phase": "started"}
                save()
                stderr_drain = asyncio.create_task(drain_stderr())
                child.stdin.write((json.dumps({"manifest": str(fixture.path.resolve()), "owner": owner,
                    "session": str(session), "tasks": [str(task.id) for task in created]}) + "\n").encode())
                await child.stdin.drain()
                ready = await reply()
                self.assertEqual(ready["kind"], "ready")
                self.assertNotEqual(child.pid, os.getpid())
                return {"independent_pid": True, "queued_tasks": 2, "canonical_bind_guard": True}

            async def claim_without_wakeup():
                replies["running"] = await command("claim", created[0].id)
                with self.assertRaises(SourceError) as rejected:
                    async with first.guard(bound["queued"]):
                        self.fail("Old queued state survived an independently committed claim")
                self.assertEqual(rejected.exception.code, "live_task_snapshot_changed")
                entry, elapsed = await wait_for(lambda: task_entry(first, created[0].id, "running"),
                    runtime=first, since=replies["running"]["committed_monotonic_ns"])
                self.assertEqual(entry.task.revision, created[0].revision)
                self.assertEqual(entry.observed_at,
                    datetime.fromisoformat(replies["running"]["task"]["updated_at"]))
                _, bound["running"] = await bind_tasks(first)
                return {"catchup_ms": elapsed, "unchanged_intent_revision": True,
                        "stale_queued_binding_rejected": rejected.exception.code, "bus_publishes": buses[0].publishes}

            async def result_without_wakeup():
                replies["first_result"] = await command("finish", created[0].id)
                with self.assertRaises(SourceError) as rejected:
                    async with first.guard(bound["running"]):
                        self.fail("Old running state survived committed completion")
                self.assertEqual(rejected.exception.code, "live_task_snapshot_changed")
                entry, elapsed = await wait_for(lambda: task_entry(first, created[0].id, "succeeded"),
                    runtime=first, since=replies["first_result"]["committed_monotonic_ns"])
                self.assertIn("Synthetic result 37.", entry.summary)
                recorded = datetime.fromisoformat(replies["first_result"]["result_recorded_at"])
                self.assertIn("recorded_at=" + recorded.isoformat(), entry.summary)
                self.assertEqual(entry.observed_at, datetime.fromisoformat(replies["first_result"]["task"]["updated_at"]))
                _, bound["completed"] = await bind_tasks(first)
                return {"catchup_ms": elapsed, "original_result_time": recorded.isoformat(),
                        "verified_manifest_result": True, "stale_running_binding_rejected": rejected.exception.code}

            async def current_visual_original():
                captured = now()
                frame = await connection.fetchval("""insert into visual_frames
                    (owner_id,session_id,visual_session_id,frame_sequence,source,captured_at,
                     content_type,width,height,storage_bucket,storage_path,raw_image_expires_at)
                    values($1,$2,$3,0,'glasses',$4,'image/jpeg',32,32,'fixture','not-uploaded',
                           clock_timestamp()+interval '1 hour') returning id""", owner, session, uuid4(), captured)
                original = VisualObservation(id=uuid4(), owner_id=owner, session_id=session, source="glasses",
                    started_at=captured, ended_at=captured, primary_visual_frame_id=frame,
                    summary="Synthetic marker visibility remains uncertain.", scene=None, activity=None,
                    entities=(), visible_text=(), state_changes=(), memory_tags=(), keyframe_candidates=(),
                    interrupt_candidate=None, starts_new_event=False, boundary_reason=None, confidence=.5,
                    analyzer_provider="fixture", analyzer_model="no-model-call", analyzer_contract_version=1,
                    compacted_at=None, created_at=captured)
                await perceptions.insert_observation(observation=original,
                    frame_links=(ObservationFrameLink(frame, 0, "primary"),))
                result = await perceptions.compare_and_swap_session_working_memory(owner_id=owner,
                    session_id=session, expected_revision=0, observation_id=original.id,
                    state={"summary": original.summary}, expires_at=captured + timedelta(minutes=2))
                self.assertTrue(result.applied)
                entry, elapsed = await wait_for(lambda: observation(first), runtime=first)
                self.assertEqual(entry.observed_at, captured)
                self.assertEqual(entry.frames[0].captured_at, captured)
                bound["visual_selection"] = select_voice_context(first.service.snapshot(), now=now())
                await first.bind(bound["visual_selection"])
                replies["visual_expiry"] = entry.expires_at
                replies["observation_id"] = original.id
                return {"catchup_ms": elapsed, "original_capture_time": captured.isoformat(),
                        "freshness_deadline": entry.expires_at.isoformat(), "model_executions": 0}

            async def close_and_commit_without_reader():
                await close_runtime(first)
                with self.assertRaises(SourceError) as rejected:
                    async with first.guard(bound["completed"]):
                        self.fail("Closed runtime admitted an old bound context")
                self.assertEqual(rejected.exception.code, "live_state_runtime_unavailable")
                replies["closed_running"] = await command("claim", created[1].id)
                replies["closed_result"] = await command("finish", created[1].id)
                self.assertEqual(replies["closed_result"]["task"]["status"], "succeeded")
                # Let the original visual freshness window lapse, not a fabricated
                # revised timestamp or a rewritten/removed observation row.
                await asyncio.sleep(max(0, (replies["visual_expiry"] - now()).total_seconds()) + .05)
                return {"owned_tasks_joined": True, "subscriptions_removed": True,
                        "result_committed_while_runtime_closed": True, "expired_original_not_rewritten": True}

            async def reconstruct_from_authorities():
                nonlocal second, current
                reconstructed_at = now()
                started = time.monotonic_ns()
                second = current = await start_runtime()
                self.assertNotEqual(first.service.snapshot().cursor.epoch, second.service.snapshot().cursor.epoch)
                self.assertIsNot(first.task_context, second.task_context)
                self.assertIsNot(first.binder, second.binder)
                for task, key in zip(created, ("first_result", "closed_result"), strict=True):
                    entry = task_entry(second, task.id, "succeeded")
                    self.assertIsNotNone(entry)
                    receipt = replies[key]
                    observed = datetime.fromisoformat(receipt["task"]["updated_at"])
                    recorded = datetime.fromisoformat(receipt["result_recorded_at"])
                    self.assertLess(observed, reconstructed_at)
                    self.assertLess(recorded, reconstructed_at)
                    self.assertEqual(entry.observed_at, observed)
                    self.assertIn("recorded_at=" + recorded.isoformat(), entry.summary)
                    self.assertIn(receipt["result"]["answer"], entry.summary)
                    durable = await tasks.get(owner, task.id)
                    self.assertEqual(durable.created_at, task.created_at)
                    self.assertEqual(durable.updated_at, observed)
                    self.assertEqual(str(durable.result_payload_id), receipt["task"]["result_payload_id"])
                selection, bound["reconstructed"] = await bind_tasks(second)
                self.assertTrue(set(bound["completed"].payload_root_ids).isdisjoint(
                    bound["reconstructed"].payload_root_ids))
                with self.assertRaises(SourceError) as rejected:
                    async with second.guard(bound["completed"]):
                        self.fail("Pre-close queued second task survived its completion")
                self.assertEqual(rejected.exception.code, "live_task_snapshot_changed")
                self.assertIsNone(observation(second))
                self.assertNotIn("Synthetic marker", selection.text)
                with self.assertRaises(SourceError) as expired:
                    await second.bind(bound["visual_selection"])
                self.assertEqual(expired.exception.code, "live_state_context_expired")
                state = await perceptions.get_session_working_memory(owner_id=owner, session_id=session)
                self.assertEqual(state.last_observation_id, replies["observation_id"])
                self.assertGreater(state.expires_at, now())
                baseline = second.perception.worker.refresh_count
                await wait_for(lambda: second.perception.worker.refresh_count >= baseline + 2, runtime=second)
                self.assertIsNone(observation(second))
                elapsed = (time.monotonic_ns() - started) / 1e6
                self.assertLess(elapsed, BOUND_SECONDS * 1000)
                return {"reconstruction_and_two_visual_polls_ms": elapsed, "durable_results": 2,
                        "original_task_and_result_timestamps_preserved": True, "new_epoch_and_binder": True,
                        "stale_pre_close_binding_rejected": rejected.exception.code,
                        "expired_visual_selection_rejected": expired.exception.code,
                        "retained_working_row_not_rejuvenated": True}

            async def correction_after_reconstruction():
                corrected = await command("update", created[0].id)
                with self.assertRaises(SourceError) as rejected:
                    async with second.guard(bound["reconstructed"]):
                        self.fail("Superseded completed result remained admissible")
                # Intent correction revokes the prior result's source roots.
                # The canonical source guard runs before snapshot comparison;
                # rejecting there is stronger than waiting for task projection.
                self.assertEqual(rejected.exception.code, "live_state_context_revoked")
                entry, elapsed = await wait_for(lambda: task_entry(second, created[0].id, "queued"),
                    runtime=second, since=corrected["committed_monotonic_ns"])
                self.assertGreater(entry.task.revision, created[0].revision)
                selection, _ = await bind_tasks(second)
                self.assertNotIn("Synthetic result 37.", selection.text)
                self.assertIn("Synthetic result 41.", selection.text)
                return {"catchup_ms": elapsed, "corrected_result_absent": True,
                        "unrelated_durable_result_preserved": True, "stale_result_rejected": rejected.exception.code}

            async def joined_final_state():
                await close_runtime(second)
                stopped = await command("stop")
                self.assertTrue(stopped["pool_closed"])
                await asyncio.wait_for(child.wait(), 10)
                self.assertEqual(child.returncode, 0)
                report["writer"].update(phase="joined", exit_code=child.returncode,
                    diagnostic_bytes=await stderr_drain, pool_closed=True)
                counts = dict(await connection.fetchrow("""select
                    (select count(*) from jarvis_background_tasks where owner_id=$1) as tasks,
                    (select count(*) from jarvis_task_attempts where owner_id=$1) as attempts,
                    (select count(*) from jarvis_task_events where owner_id=$1 and event_type='succeeded') as completions,
                    (select count(*) from jarvis_background_tasks where owner_id=$1 and result_payload_id is not null) as current_results,
                    (select count(*) from jarvis_task_actions where owner_id=$1) as actions""", owner))
                self.assertEqual(counts, {"tasks": 2, "attempts": 2, "completions": 2, "current_results": 1, "actions": 0})
                self.assertEqual(sum(bus.publishes for bus in buses), 0)
                for runtime in runtimes:
                    self.assertEqual(runtime.worker.failure_count, 0)
                    self.assertEqual(runtime.perception.worker.failure_count, 0)
                return counts | {"writer_joined": True, "all_runtime_tasks_joined": True,
                    "bus_publishes": 0, "canonical_refresh_failures": 0}

            for name, operation in (
                ("actual_runtime_and_independent_writer", initial_state_and_writer),
                ("committed_claim_poll_catchup_and_stale_guard", claim_without_wakeup),
                ("committed_result_poll_catchup_and_original_time", result_without_wakeup),
                ("canonical_visual_original_projection", current_visual_original),
                ("joined_close_and_commits_without_reader", close_and_commit_without_reader),
                ("reconstruction_preserves_results_not_expired_visuals", reconstruct_from_authorities),
                ("corrected_result_invalidates_reconstructed_binding", correction_after_reconstruction),
                ("joined_writer_runtime_and_no_duplicate_effects", joined_final_state),
            ):
                await check(name, operation)
            report["status"] = "passed"
        except Exception as error:
            failure = sanitized(error)
            report.update(status="failed", failure=failure)
            save()
        finally:
            close_errors = []
            for runtime in runtimes:
                try:
                    await close_runtime(runtime)
                except Exception as error:
                    close_errors.append(sanitized(error))
            if child is not None:
                if child.returncode is None:
                    child.stdin.close()
                    try:
                        await asyncio.wait_for(child.wait(), 12)
                    except TimeoutError:
                        child.terminate()
                        try:
                            await asyncio.wait_for(child.wait(), 5)
                        except TimeoutError:
                            child.kill()
                            await child.wait()
                report["writer"].update(phase="joined", exit_code=child.returncode)
                if stderr_drain is not None:
                    try:
                        report["writer"]["diagnostic_bytes"] = await asyncio.wait_for(stderr_drain, 5)
                    except Exception as error:
                        close_errors.append(sanitized(error))
            if close_errors:
                report.update(status="failed", close_errors=close_errors)
            if fixture.manifest["schemas"]:
                await cleanup_fixture_resources(fixture, report, save, pool=pool, connection=connection)
            else:
                if connection is not None:
                    await connection.close(timeout=5)
                report["cleanup"] = {"status": "not_required_no_resources_allocated"}
                save()
            print(json.dumps({"report": str(report_path), "status": report["status"],
                "cases": report["cases"], "writer": report["writer"], "failure": failure,
                "cleanup": report["cleanup"]}, default=str))
        self.assertIsNone(failure, json.dumps(failure))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["cases"]), 8)
        self.assertEqual(report["cleanup"]["remaining_schemas"], 0)
        self.assertEqual(report["cleanup"]["remaining_public_owner_rows"], 0)


if __name__ == "__main__":
    unittest.main()
