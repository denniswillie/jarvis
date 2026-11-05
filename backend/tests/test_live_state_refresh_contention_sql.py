"""One manifested SQL diagnostic, not a latency acceptance or native qualification.

Only the canonical worker is started: baseline, two bounded observation writes
overlapping refresh, then post-write refresh. No provider/model/storage client.
The production worker retains its configured read/apply/guard deadlines. Source text,
query arguments, SQL text, exception messages and credentials are never emitted.
"""
import argparse
import asyncio
from contextlib import ExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from time import monotonic
import unittest
from unittest.mock import patch

import asyncpg

from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.live_state.models import SessionScope
from app.live_state.refresh import CanonicalRefreshWorker
from app.live_state.runtime import LiveStateRuntime
from app.runtime.events import DomainEventBus
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from evals.second_brain.fixtures import _owner_tables, _quote, validate_manifest
from evals.second_brain.source_fixture import source_fixture


LANE = ContextVar("refresh_diagnostic_lane", default="fixture_setup")
STAGE = ContextVar("refresh_diagnostic_stage", default="fixture_setup")
SOURCE_TIME = datetime(2026, 11, 1, 12, tzinfo=UTC)


class Trace:
    def __init__(self):
        self.started = monotonic()
        self.stages, self.queries = [], []
        self.dropped = 0
        self.observation_entered = asyncio.Event()

    def add(self, target, row):
        if len(target) < 1024:
            target.append(row)
        else:
            self.dropped += 1

    @contextmanager
    def span(self, name, *, pid=None):
        token = STAGE.set(name)
        start, outcome = monotonic(), "ok"
        try:
            yield
        except BaseException as error:
            outcome = type(error).__name__
            raise
        finally:
            end = monotonic()
            self.add(self.stages, dict(lane=LANE.get(), stage=name, pid=pid,
                start_ms=round((start - self.started) * 1000, 3),
                duration_ms=round((end - start) * 1000, 3), outcome=outcome))
            STAGE.reset(token)

    def wrap(self, stack, owner, method, name):
        original = getattr(owner, method)

        async def measured(*args, **kwargs):
            with self.span(name):
                return await original(*args, **kwargs)

        stack.enter_context(patch.object(owner, method, measured))

    async def pool_init(self, connection):
        pid = connection.get_server_pid()

        def query(record):
            sql = record.query
            lock = ("memory_owner" if ":jarvis-memory-owner:" in sql else
                    "task_owner" if ":jarvis-task-owner:" in sql else None)
            self.add(self.queries, dict(lane=LANE.get(), stage=STAGE.get(), pid=pid,
                end_ms=round((monotonic() - self.started) * 1000, 3),
                duration_ms=round(record.elapsed * 1000, 3), lock=lock,
                fingerprint=sha256(sql.encode()).hexdigest()[:12],
                tables=sorted(set(re.findall(r"\bjarvis_[a-z_]+\b", sql)))[:12],
                outcome=type(record.exception).__name__ if record.exception else "ok"))

        connection.add_query_logger(query)

    def transactions(self, stack, sources):
        original = sources.transaction

        @asynccontextmanager
        async def measured(*args, **kwargs):
            manager = original(*args, **kwargs)
            with self.span("memory_transaction_enter"):
                connection = await manager.__aenter__()
            if LANE.get().startswith("observation_"):
                self.observation_entered.set()
            # Includes transaction exit/rollback; nested timings are inclusive.
            with self.span("memory_owner_hold", pid=connection.get_server_pid()):
                try:
                    yield connection
                except BaseException:
                    if not await manager.__aexit__(*sys.exc_info()):
                        raise
                else:
                    await manager.__aexit__(None, None, None)

        stack.enter_context(patch.object(sources, "transaction", measured))


def health(runtime):
    worker = runtime.worker
    return dict(healthy=runtime._healthy, readiness=runtime.service.snapshot().readiness,
        needs_resync=runtime.service.needs_resync, entries=len(runtime._entry_ids),
        refresh_count=worker.refresh_count, failure_count=worker.failure_count,
        last_error=worker.last_error, last_error_stage=worker.last_error_stage,
        last_error_code=worker.last_error_code)


async def audit_residue(dsn, manifest):
    """Fresh connection, exact fixture identities; no cleanup or storage writes."""
    validate_manifest(manifest)
    connection = await asyncpg.connect(dsn, timeout=10, command_timeout=30)
    try:
        schemas = await connection.fetchval(
            "select count(*) from pg_namespace where nspname=any($1::text[])", manifest["schemas"])
        tables = await _owner_tables(connection, "public")
        parts = [f"select count(*) as n from public.{_quote(row['relname'])} "
                 "where owner_id::text=any($1::text[])" for row in tables]
        public_rows = (await connection.fetchval(
            "select sum(n)::bigint from (" + " union all ".join(parts) + ") audited",
            manifest["owners"]) if parts else 0)
        storage_present = await connection.fetchval("select to_regclass('storage.objects') is not null")
        storage_rows = (await connection.fetchval("""select count(*) from storage.objects o
            where to_jsonb(o)->>'owner_id'=any($1::text[])
               or to_jsonb(o)->>'owner'=any($1::text[])
               or o.name like $2""", manifest["owners"], "%" + manifest["run_id"] + "%")
            if storage_present else 0)
        return dict(remaining_schemas=schemas, remaining_public_owner_rows=public_rows,
                    remaining_storage_objects=storage_rows, storage_catalog_present=storage_present)
    finally:
        await connection.close(timeout=5)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
    os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Explicit manifested SQL fixture required")
class LiveStateRefreshContentionSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_refresh_contention_diagnostic(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        trace, fixture = Trace(), None
        report = dict(format="jarvis.live_state.refresh_contention.v1", phases=[], writes=[],
            boundaries=dict(read_seconds=2.0, apply_seconds=10.0, canonical_guard_seconds=2.0,
                observation_seconds=3.0, observation_count=2,
                pool_max_size=3, canonical_worker_only=True, native_or_model_calls=False,
                readiness_monitor_started=False, seed_is_fixture_setup=True,
                timings_inclusive=True, latency_acceptance=False))
        try:
            async with source_fixture(dsn, directory, test_name=self.id(), pool_init=trace.pool_init) as f:
                fixture = f.fixture
                report["fixture_manifest"] = str(fixture.path)
                tasks = TaskRepository(f.pool, TaskLimits())
                selections = SessionKnowledgeAreas(f.sources, tasks, f.area)
                publication = CanonicalSourcePublication(f.sources, f.area, selections)
                context = ContextRepository(tasks, publication=publication)
                async with f.pool.acquire() as connection:
                    session = await connection.fetchval(
                        "insert into jarvis_sessions(owner_id) values($1) returning id", f.owner)
                runtime = LiveStateRuntime(scope=SessionScope(owner_id=f.owner, session_id=session,
                    area_id=f.area), sources=f.sources, context=context, area_selections=selections,
                    bus=DomainEventBus(), access_check=lambda owner: None, tasks=tasks)
                self.assertEqual(runtime.worker.timeout_seconds, 2.0)
                self.assertEqual(runtime.worker.apply_timeout_seconds, runtime.ttl_seconds)
                self.assertEqual(runtime.binder.timeout_seconds, 2.0)
                pending = []
                await runtime.service.start()
                runtime.producer.start()
                runtime._started = True  # Trusted seam: no unrelated background workers.
                try:
                    with ExitStack() as stack:
                        trace.transactions(stack, f.sources)
                        for owner, method, label in (
                            (runtime, "_read", "read"), (runtime, "_apply", "apply"),
                            (runtime.binder, "validate_scope", "scope_validation"),
                            (context, "snapshot", "context_snapshot"),
                            (context, "append_in_transaction", "canonical_append"),
                            (f.sources, "current_snapshot", "memory_snapshot"),
                            (f.sources, "publish_in_transaction", "source_publication"),
                            (selections, "read_in_transaction", "selection_read"),
                            (selections, "bind_context", "selection_bind"),
                            (runtime.task_context, "read_ids", "task_id_read"),
                            (runtime.task_context, "project", "task_projection"),
                            (runtime.binder, "bind_record", "bind_record"),
                            (runtime.binder, "bind_memory", "bind_memory"),
                            (runtime.binder, "available", "cached_binding_guard"),
                            (runtime.binder, "_validate_dependencies", "dependency_guard"),
                        ):
                            trace.wrap(stack, owner, method, label)
                        for repository in {id(tasks): tasks, id(runtime.binder.tasks): runtime.binder.tasks}.values():
                            trace.wrap(stack, repository, "lock_owner", "task_owner_lock")

                        # Durable synthetic precondition, not a capture latency claim.
                        with trace.span("fixture_seed"):
                            async with asyncio.timeout(30):
                                request = await context.append(f.owner, ContextIngress(session_id=session,
                                    source_namespace="fixture:refresh-contention", source_event_id="request",
                                    receipt_id="final", kind="user_transcript", trust="user_input",
                                    source_time=SOURCE_TIME, payload={"text": "Synthetic bounded diagnostic request",
                                        "input_id": "synthetic-input", "transcript_status": "finalized",
                                        "capture_input_version": 1}))
                        self.assertTrue(request.authorizes_request)

                        async def phase(name):
                            token = LANE.set(name)
                            before = health(runtime)

                            async def apply(value):
                                try:
                                    await runtime._apply(value)
                                finally:
                                    worker._closed = True

                            async def invalidate(reason):
                                try:
                                    await runtime._invalidate(reason)
                                finally:
                                    worker._closed = True

                            worker = CanonicalRefreshWorker(scope=runtime.scope, bus=runtime.worker.bus,
                                event_names=runtime.worker.event_names, read=runtime._read,
                                apply=apply, invalidate=invalidate,
                                timeout_seconds=runtime.worker.timeout_seconds,
                                apply_timeout_seconds=runtime.worker.apply_timeout_seconds)
                            runtime.worker = worker
                            guard = "not_attempted"
                            try:
                                worker.start()
                                await worker._task
                                cached = runtime._cache.get("conversation:request")
                                guard_kind = "request_binding" if cached else "scope_only"
                                try:
                                    with trace.span("guard_" + guard_kind):
                                        manager = (runtime.guard(cached[2], wait_for_owner=False) if cached else
                                                   runtime.scope_guard(wait_for_owner=False))
                                        async with manager:
                                            pass  # No provider write or tool effect.
                                    guard = "ok"
                                except Exception as error:
                                    guard = type(error).__name__
                                report["phases"].append(dict(name=name, before=before, after=health(runtime),
                                    guard_kind=guard_kind, guard_outcome=guard))
                                self.assertEqual(worker.refresh_count + worker.failure_count, 1)
                            finally:
                                await worker.close()
                                LANE.reset(token)

                        async def observations():
                            for index in range(2):
                                token = LANE.set(f"observation_{index}")
                                outcome = "ok"
                                try:
                                    with trace.span("observation_append"):
                                        async with asyncio.timeout(3.0):
                                            await context.append(f.owner, ContextIngress(session_id=session,
                                                source_namespace="fixture:refresh-contention",
                                                source_event_id=f"observation-{index}", receipt_id="final",
                                                kind="observation", trust="sensor", source_time=SOURCE_TIME,
                                                payload={"text": "Synthetic optional observation", "images": []}))
                                except Exception as error:
                                    outcome = type(error).__name__
                                finally:
                                    report["writes"].append(dict(index=index, outcome=outcome))
                                    trace.observation_entered.set()
                                    LANE.reset(token)

                        await phase("baseline")
                        writer = asyncio.create_task(observations(), name="diagnostic-optional-writes")
                        pending.append(writer)
                        await trace.observation_entered.wait()
                        await phase("contended")
                        await writer
                        await phase("post_writes")
                finally:
                    for task in pending:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    await runtime.worker.close()
                    runtime._closing = True
                    await runtime.service.close()
            report["audit"] = await audit_residue(dsn, fixture.manifest)
            self.assertEqual({key: report["audit"][key] for key in (
                "remaining_schemas", "remaining_public_owner_rows", "remaining_storage_objects")},
                dict(remaining_schemas=0, remaining_public_owner_rows=0, remaining_storage_objects=0))
        except BaseException as error:
            report["error_type"] = type(error).__name__
            raise
        finally:
            await asyncio.sleep(0)  # Drain already-scheduled asyncpg query logger callbacks.
            report.update(stages=trace.stages, queries=trace.queries, trace_dropped=trace.dropped,
                          cleanup_status=fixture.manifest["status"] if fixture else "fixture_not_yielded")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "live-state-refresh-contention.json"
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
            print("REFRESH_CONTENTION " + json.dumps(dict(evidence=str(path), phases=report["phases"],
                writes=report["writes"], audit=report.get("audit"), error_type=report.get("error_type"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configured-db", action="store_true", required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()
    from app.settings import JarvisSettings
    try:
        dsn = JarvisSettings().SUPABASE_DB_URL
        if not dsn:
            raise ValueError("Configured database required")
    except Exception as error:
        print(json.dumps({"configuration_error_type": type(error).__name__}))
        return 2
    os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"] = dsn
    os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"] = str(args.manifest_dir)
    LiveStateRefreshContentionSQLTests.__unittest_skip__ = False
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LiveStateRefreshContentionSQLTests)
    result = unittest.TestResult()  # Do not print arbitrary database exception messages.
    suite.run(result)
    print(json.dumps(dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                          skipped=len(result.skipped))))
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
