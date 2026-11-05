"""Real v39 -> v40 episode-owner qualification, confined to a manifested schema.

Uses the existing explicit DB/manifest gates. Synthetic canonical user ingress,
real repositories, real constraints and real locks; no model or public migration.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import UUID, uuid4

import asyncpg

from app.experience.repository import ToolExecutionRepository
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.sources import SourceRepository
from app.live_state.models import SessionScope
from app.live_state.perception import CheckResolution, EvidenceReference
from app.migrations.runner import apply_migrations, load_migrations, quote_schema
from app.reminders.duties import ChangeDuty, DutySpec, ResponsibilityRepository, SaveDuty
from app.reminders.episode_checks import EpisodeCheckOwner, reference
from app.reminders.episode_models import AdmitEpisodeCheck, ChangeEpisodeCheck, EpisodeCheckDraft, EpisodeCheckSpec
from app.reminders.episode_tools import EpisodeCheckToolModule
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError
from app.tasks.repository import TaskRepository
from app.tools.inputs import ToolInputError
from app.tools.executor import JarvisToolExecutor
from app.tools.invocation import ToolInvocationContext, bind_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.provenance import SourceBoundToolResult, ToolResultCapture, restore_source_result
from app.tools.registry import JarvisToolRegistry
from evals.second_brain.fixtures import DatabaseFixture
from test_notification_ledger_database import cleanup_fixture_resources
from test_responsive_perception_sql import ObservedTransactionPool, sanitized


def now():
    return datetime.now(UTC)


class DiagnosticPool:
    """Retain sanitized SQL evidence when an owner deliberately masks DB errors.

    No added transaction, lock, retry or pool lease. All repository statements
    and transaction operations execute on the original leased connection.
    """

    def __init__(self, pool, errors):
        self.pool, self.errors = pool, errors

    @asynccontextmanager
    async def acquire(self, *args, **kwargs):
        async with self.pool.acquire(*args, **kwargs) as connection:
            errors = self.errors

            class Connection:
                def __getattr__(self, name):
                    method = getattr(connection, name)
                    if name not in {"execute", "executemany", "fetch", "fetchrow", "fetchval"}:
                        return method

                    async def query(sql, *args, **kwargs):
                        try:
                            return await method(sql, *args, **kwargs)
                        except asyncpg.PostgresError as error:
                            errors.append({"sqlstate": error.sqlstate,
                                "error_type": type(error).__name__, "operation": name,
                                "query_head": " ".join(sql.split())[:180],
                                "constraint": getattr(error, "constraint_name", None),
                                "column": getattr(error, "column_name", None)})
                            raise
                    return query

            yield Connection()


class RollbackProbe(Exception):
    pass


class FailOnceReceiptCapture(SessionContextCapture):
    """Rollback a real receipt transaction after its insert, not the local effect."""

    fail_next_receipt = True

    async def bound_tool_outcome(self, *args, **kwargs):
        record = await super().bound_tool_outcome(*args, **kwargs)
        if self.fail_next_receipt:
            self.fail_next_receipt = False
            raise RollbackProbe("Synthetic after-commit receipt failure")
        return record


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class EpisodeCheckSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_v40_episode_owner_and_invariants(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        fixture = DatabaseFixture(dsn, directory, test_name=self.id())
        report_path = directory / f"episode-v40-{fixture.run_id}.json"
        report = {"format": "jarvis.episode_owner.sql.v1", "run_id": fixture.run_id,
            "status": "running", "migration_ceiling": 40, "checks": [], "cleanup": None,
            "model_calls": False, "public_migration": False, "network_retries": 0}
        connection = pool = rival_pool = None
        failures, sql_errors, captures = [], [], []

        def save():
            fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")

        async def check(name, operation):
            start = len(sql_errors)
            try:
                async with asyncio.timeout(50):
                    evidence = await operation()
            except Exception as error:
                entry = {"name": name, "status": "failed", **sanitized(error),
                         "repository_sql_errors": sql_errors[start:]}
                failures.append(entry)
                if isinstance(error, TimeoutError):
                    entry["failure_kind"] = "local_deadline"
                elif isinstance(error, (OSError, asyncpg.ConnectionDoesNotExistError,
                                        asyncpg.CannotConnectNowError)):
                    report["checks"].append(entry)
                    save()
                    raise
            else:
                entry = {"name": name, "status": "passed", "evidence": evidence,
                         "repository_sql_errors": sql_errors[start:]}
            report["checks"].append(entry)
            save()

        save()
        try:
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
            schema = await fixture.create_schema(connection)
            migrations = tuple(m for m in load_migrations() if m.version <= 40)
            self.assertEqual((migrations[-1].version, migrations[-1].name),
                             (40, "responsibility_episode_checks"))
            report["migration_40_checksum"] = migrations[-1].checksum
            await asyncio.wait_for(apply_migrations(connection, schema=schema, fresh=True,
                migrations=tuple(m for m in migrations if m.version <= 39)), timeout=240)
            path_sql = f"set search_path to {quote_schema(schema)},pg_catalog,extensions"
            await connection.execute(path_sql)
            await connection.execute("set lock_timeout='8s'; set statement_timeout='20s'")

            async def setup(c):
                await c.execute(path_sql)
                await c.execute("set lock_timeout='8s'; set statement_timeout='20s'")

            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, timeout=10,
                                            command_timeout=25, setup=setup)
            rival_pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1, timeout=10,
                                                  command_timeout=25, setup=setup)
            inspected = DiagnosticPool(pool, sql_errors)
            repository = ResponsibilityRepository(inspected)
            sources = SourceRepository(inspected)
            owner, other = fixture.owners
            area = await connection.fetchval("insert into jarvis_knowledge_areas(owner_id,name,domain) "
                "values($1,'Episode SQL fixture','test') returning id", owner)
            tasks = TaskRepository(inspected, TaskLimits())
            context = ContextRepository(tasks, publication=CanonicalSourcePublication(sources, area))

            async def new_scope(selected_owner=owner):
                session = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", selected_owner)
                return SessionScope(owner_id=selected_owner, session_id=session)

            async def capture(scope, *, at=None, previous=None, text="Track the synthetic check.", repo=context):
                record = await repo.append(scope.owner_id, ContextIngress(session_id=scope.session_id,
                    source_namespace="qualification:episode", source_event_id=(previous.source_event_id
                        if previous else str(uuid4())), receipt_id=str(uuid4()),
                    expected_revision=previous.revision if previous else 0,
                    kind="user_transcript", trust="user_input", finality="revised" if previous else "final",
                    source_time=at or now(), payload={"text": text}))
                return ToolInvocationContext(request_event_id=record.id, request_revision=record.revision), record

            async def source_identity(record):
                value = await connection.fetchval("select source_id from jarvis_knowledge_source_versions "
                    "where owner_id=$1 and payload_id=$2", owner, record.payload_id)
                self.assertIsNotNone(value)
                return value

            # Seed an actual scheduled duty under v39 before changing its schema.
            legacy_scope = await new_scope()
            legacy_source, _ = await capture(legacy_scope)
            legacy_request = SaveDuty(spec=DutySpec(title="Synthetic laundry reminder",
                first_at=now()-timedelta(minutes=1), every_days=1))
            legacy = await repository.save(owner, legacy_scope.session_id, legacy_source, legacy_request)
            legacy_id = UUID(legacy["responsibility"]["id"])
            await connection.execute("update jarvis_responsibilities set spec=spec-'kind' "
                                     "where owner_id=$1 and id=$2", owner, legacy_id)
            legacy_rule = dict(await connection.fetchrow("select * from jarvis_responsibilities "
                "where owner_id=$1 and id=$2", owner, legacy_id))
            legacy_occurrence = dict(await connection.fetchrow("select * from jarvis_responsibility_occurrences "
                "where owner_id=$1 and responsibility_id=$2", owner, legacy_id))

            async def upgrade():
                self.assertEqual(await apply_migrations(connection, schema=schema, migrations=migrations), (40,))
                await pool.expire_connections()
                await rival_pool.expire_connections()
                self.assertEqual(await apply_migrations(connection, schema=schema, migrations=migrations), ())
                rule = dict(await connection.fetchrow("select * from jarvis_responsibilities "
                    "where owner_id=$1 and id=$2", owner, legacy_id))
                occurrence = dict(await connection.fetchrow("select * from jarvis_responsibility_occurrences "
                    "where owner_id=$1 and id=$2", owner, legacy_occurrence["id"]))
                self.assertEqual(rule, legacy_rule)
                self.assertEqual({k: occurrence[k] for k in legacy_occurrence}, legacy_occurrence)
                self.assertEqual(occurrence["check_revision"], 1)
                self.assertIsNone(occurrence["episode_id"])
                actor = EpisodeCheckOwner(repository, enabled=True)
                self.assertEqual((await actor.initialize()).status, "available")
                replay = await repository.save(owner, legacy_scope.session_id, legacy_source, legacy_request)
                self.assertTrue(replay["request_replayed"])
                body = legacy_request.model_dump(mode="json")
                body["spec"].pop("kind")
                old_hash = sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
                self.assertEqual(await connection.fetchval("select arguments_hash from jarvis_responsibility_requests "
                    "where owner_id=$1 and responsibility_id=$2 and operation='save'", owner, legacy_id), old_hash)
                return {"applied_only": [40], "idempotent": True,
                        "old_rule_occurrence_and_request_fingerprint_preserved": True}

            await check("v39_scheduled_upgrade_preservation_and_retry", upgrade)
            if failures:
                raise AssertionError("Upgrade gate failed; later writes not attempted")
            await connection.execute("create table episode_sql_effects(owner_id text not null,id uuid primary key)")

            def accepted(result):
                self.assertEqual(result.outcome.status, "accepted")
                self.assertIsNotNone(result.check)
                return result

            async def scenario(*, dependencies=()):
                scope, clock = await new_scope(), [now()]
                actor = EpisodeCheckOwner(repository, enabled=True, clock=lambda: clock[0])
                self.assertEqual((await actor.initialize()).status, "available")
                source, record = await capture(scope, at=clock[0])
                request = AdmitEpisodeCheck(spec=EpisodeCheckSpec(title="Synthetic departure check"),
                    check=EpisodeCheckDraft(question="What state is the synthetic switch in?",
                        object_association="fixture:switch", dependencies=dependencies,
                        observation_window_expires_at=clock[0]+timedelta(minutes=5)))
                result = accepted(await actor.admit(scope, source, request))
                return SimpleNamespace(scope=scope, clock=clock, actor=actor, source=source,
                                       record=record, request=request, result=result)

            def refs(result):
                return {"responsibility": result.check.canonical_owner_ref, "check": result.outcome.check_ref,
                        "episode": result.check.episode_or_object_ref}

            def change_request(result, operation, **kwargs):
                value = result.check
                return ChangeEpisodeCheck(responsibility_id=value.canonical_owner_ref.source_id,
                    expected_revision=value.canonical_owner_ref.revision, occurrence_id=value.id,
                    expected_check_revision=value.revision, episode_id=value.episode_or_object_ref.source_id,
                    expected_episode_revision=value.episode_or_object_ref.revision, operation=operation, **kwargs)

            async def row(s, result=None):
                return dict(await connection.fetchrow("select * from jarvis_responsibility_occurrences "
                    "where owner_id=$1 and id=$2", owner, (result or s.result).check.id))

            async def error(code, operation):
                with self.assertRaises(ToolInputError) as caught:
                    await operation
                self.assertEqual(caught.exception.code, code)

            async def guarded(s, result=None):
                async with s.actor.guard(s.scope, **refs(result or s.result)) as (c, snapshot):
                    self.assertTrue(c.is_in_transaction())
                    return snapshot

            async def unreadable(s):
                with self.assertRaises(ToolInputError):
                    await guarded(s)
                read = await s.actor.read(s.scope, include_inactive=True, limit=100)
                self.assertEqual(read.capability.status, "available")
                self.assertNotIn(s.result.check.id, {snapshot.check.id for snapshot in read.snapshots})
                self.assertIn("episode_check_evidence_unavailable:"+str(s.result.check.id), read.gaps)

            def resolution(result, record, *, basis="user_report"):
                episode = result.check.episode_or_object_ref
                evidence = EvidenceReference(source=reference("context_event", record.id, record.revision),
                    captured_at=record.source_time or record.recorded_at, episode_or_object_ref=episode,
                    frame_id=uuid4() if basis == "visual_observation" else None,
                    visibility="visible" if basis == "visual_observation" else None)
                return CheckResolution(basis=basis, source_refs=(evidence,),
                    applicable_episode_or_object_ref=episode, resolved_at=evidence.captured_at)

            async def admission_retry():
                before = await connection.fetchval("select count(*) from jarvis_reminders where owner_id=$1", owner)
                s = await scenario()
                replay = accepted(await s.actor.admit(s.scope, s.source, s.request))
                self.assertTrue(replay.replayed)
                self.assertEqual(replay.check, s.result.check)
                occurrence = await row(s)
                self.assertIsNone(occurrence["reminder_id"])
                self.assertIsNone(await connection.fetchval("select next_due_at from jarvis_responsibilities "
                    "where owner_id=$1 and id=$2", owner, s.result.check.canonical_owner_ref.source_id))
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_reminders where owner_id=$1", owner), before)
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_responsibility_occurrences "
                    "where owner_id=$1 and responsibility_id=$2", owner, occurrence["responsibility_id"]), 1)
                await error("request_conflict", s.actor.admit(s.scope, s.source,
                    s.request.model_copy(update={"check": EpisodeCheckDraft(question="Conflicting question")})))
                return {"one_episode": True, "idempotent_retry": True, "placeholder_reminders": 0}

            async def concurrent_admission():
                s = await scenario()
                source, _ = await capture(s.scope, at=s.clock[0])
                first, second = await asyncio.gather(s.actor.admit(s.scope, source, s.request),
                                                    s.actor.admit(s.scope, source, s.request))
                accepted(first), accepted(second)
                self.assertEqual(first.check.id, second.check.id)
                self.assertEqual(int(first.replayed)+int(second.replayed), 1)
                return {"concurrent_same_identity": "one_committed_effect"}

            async def cross_session_read():
                s = await scenario()
                second = await new_scope()
                read = await s.actor.read(second, limit=100)
                snapshot = next(item for item in read.snapshots if item.check.id == s.result.check.id)
                self.assertEqual(snapshot.check.scope, second)
                self.assertIn(s.record.payload_id, snapshot.source_payload_ids)
                async with s.actor.guard(second, **refs(s.result)) as (_, current):
                    self.assertEqual(current.check.id, s.result.check.id)
                foreign = await new_scope(other)
                await ContextRepository(tasks).append(other, ContextIngress(session_id=foreign.session_id,
                    source_namespace="qualification:foreign", source_event_id=str(uuid4()), receipt_id=str(uuid4()),
                    kind="user_request", trust="user_input", payload={"text": "Synthetic foreign owner"}))
                self.assertFalse((await s.actor.read(foreign)).snapshots)
                with self.assertRaises(ToolInputError):
                    async with s.actor.guard(foreign, **refs(s.result)):
                        self.fail("Foreign owner acquired episode")
                return {"same_owner_cross_session": True, "original_evidence_preserved": True,
                        "cross_owner_rejected": True}

            async def revision_conflicts():
                s = await scenario()
                original = await row(s)
                request = change_request(s.result, "cancel")
                for key, value, code in (("expected_revision", 2, "revision_conflict"),
                    ("expected_check_revision", 2, "check_revision_conflict"),
                    ("expected_episode_revision", 2, "check_revision_conflict"),
                    ("episode_id", uuid4(), "check_revision_conflict")):
                    source, _ = await capture(s.scope, at=s.clock[0])
                    await error(code, s.actor.change(s.scope, source, request.model_copy(update={key: value})))
                self.assertEqual(await row(s), original)
                return {"independent_rule_check_episode_revision_and_identity_fences": 4}

            async def next_episode():
                s = await scenario()
                s.clock[0] += timedelta(seconds=1)
                source, _ = await capture(s.scope, at=s.clock[0])
                following = accepted(await s.actor.admit(s.scope, source, s.request.model_copy(update={
                    "responsibility_id": s.result.check.canonical_owner_ref.source_id, "expected_revision": 1})))
                self.assertNotEqual(following.check.id, s.result.check.id)
                self.assertNotEqual(following.check.episode_or_object_ref.source_id,
                                    s.result.check.episode_or_object_ref.source_id)
                self.assertEqual((await row(s))["status"], "open")
                self.assertEqual((await row(s))["check_revision"], 1)
                self.assertEqual(following.check.canonical_owner_ref.revision, 2)
                return {"new_episode_identity": True, "older_unresolved_episode_preserved": True}

            async def rebind_episode():
                s = await scenario()
                s.clock[0] += timedelta(seconds=1)
                source, _ = await capture(s.scope, at=s.clock[0])
                changed = accepted(await s.actor.change(s.scope, source, change_request(s.result, "revise",
                    check=EpisodeCheckDraft(question="Inspect the replacement object", object_association="fixture:other"))))
                self.assertEqual(changed.check.episode_or_object_ref.source_id, s.result.check.episode_or_object_ref.source_id)
                self.assertEqual(changed.check.episode_or_object_ref.revision, 2)
                self.assertEqual(changed.check.revision, 2)
                source, record = await capture(s.scope, at=s.clock[0])
                await error("resolution_episode_conflict", s.actor.change(s.scope, source,
                    change_request(changed, "resolve", resolution=resolution(s.result, record))))
                return {"explicit_rebinding_advances_episode_revision": True, "old_episode_evidence_rejected": True}

            async def resolved_scenario():
                s = await scenario()
                s.clock[0] += timedelta(seconds=1)
                source, record = await capture(s.scope, at=s.clock[0], text="I switched the synthetic control off.")
                request = change_request(s.result, "resolve", resolution=resolution(s.result, record))
                result = accepted(await s.actor.change(s.scope, source, request))
                return s, source, record, request, result

            async def user_report():
                s, source, record, request, result = await resolved_scenario()
                self.assertEqual(result.check.lifecycle, "resolved")
                self.assertEqual(result.check.resolution.basis, "user_report")
                self.assertEqual(result.check.resolution.source_refs[0].captured_at, record.source_time)
                self.assertEqual(result.check.evidence_availability, "available")
                self.assertEqual((await row(s))["report_state"], "user_reported_done")
                replay = accepted(await s.actor.change(s.scope, source, request))
                self.assertTrue(replay.replayed)
                self.assertEqual(replay.check, result.check)
                async with pool.acquire() as c:
                    view = await repository._view(c, owner, result.check.canonical_owner_ref.source_id)
                self.assertEqual(len(view["report_history"]), 1)
                self.assertEqual(view["report_history"][0]["report_state"], "user_reported_done")
                self.assertIsNone(view["occurrences"][0]["reminder_status"])
                return {"original_user_report_time_and_basis": True, "retry_one_report": True,
                        "observed_physical_completion_claimed": False}

            async def unqualified_resolution():
                s = await scenario()
                original = await row(s)
                for basis in ("visual_observation", "trusted_source"):
                    source, record = await capture(s.scope, at=s.clock[0])
                    await error("resolution_validator_unavailable", s.actor.change(s.scope, source,
                        change_request(s.result, "resolve", resolution=resolution(s.result, record, basis=basis))))
                self.assertEqual(await row(s), original)
                return {"unsupported_evidence_bases_rejected": 2, "row_unchanged": True}

            async def cancellation():
                s = await scenario()
                source, _ = await capture(s.scope, at=s.clock[0])
                result = accepted(await s.actor.change(s.scope, source, change_request(s.result, "cancel")))
                self.assertEqual(result.check.lifecycle, "dismissed")
                self.assertIsNone(result.check.resolution)
                occurrence = await row(s)
                self.assertEqual((occurrence["status"], occurrence["report_state"]), ("cancelled", "unknown"))
                source, record = await capture(s.scope, at=s.clock[0])
                await error("episode_check_inactive", s.actor.change(s.scope, source,
                    change_request(result, "resolve", resolution=resolution(result, record))))
                return {"cancellation_is_not_resolution": True, "late_resolution_rejected": True}

            async def snooze():
                s = await scenario()
                source, _ = await capture(s.scope, at=s.clock[0])
                until = s.clock[0] + timedelta(minutes=1)
                result = accepted(await s.actor.change(s.scope, source,
                    change_request(s.result, "snooze", until=until)))
                self.assertEqual(result.check.lifecycle, "tracking")
                self.assertIsNone(result.check.resolution)
                with self.assertRaises(ToolInputError) as caught:
                    await guarded(s, result)
                self.assertEqual(caught.exception.code, "episode_check_inactive")
                s.clock[0] = until + timedelta(seconds=1)
                await guarded(s, result)
                self.assertEqual((await row(s))["status"], "open")
                return {"tracking_remains_unresolved": True, "guard_resumes_after_snooze": True}

            async def expiry():
                s = await scenario()
                source, _ = await capture(s.scope, at=s.clock[0])
                await error("observation_window_active", s.actor.change(s.scope, source,
                    change_request(s.result, "expire")))
                s.clock[0] += timedelta(minutes=6)
                source, _ = await capture(s.scope, at=s.clock[0])
                result = accepted(await s.actor.change(s.scope, source, change_request(s.result, "expire")))
                self.assertEqual(result.check.lifecycle, "expired")
                self.assertIsNone(result.check.resolution)
                self.assertEqual((await row(s))["status"], "open")
                self.assertEqual(await connection.fetchval("select status from jarvis_responsibilities "
                    "where owner_id=$1 and id=$2", owner, result.check.canonical_owner_ref.source_id), "active")
                return {"window_expired": True, "obligation_still_open": True}

            async def source_correction():
                s = await scenario()
                await guarded(s)
                await capture(s.scope, previous=s.record, text="Correction: the original request was mistaken.")
                await unreadable(s)
                self.assertEqual((await row(s))["status"], "open")
                return {"corrected_admission_not_usable_as_current_evidence": True}

            async def dependency_correction():
                source_scope = await new_scope()
                _, dependency = await capture(source_scope, text="Synthetic object association.")
                s = await scenario(dependencies=(reference("context_event", dependency.id, dependency.revision),))
                await guarded(s)
                await capture(source_scope, previous=dependency, text="Correction: different object.")
                await unreadable(s)
                return {"superseded_dependency_invalidates_read_and_guard": True}

            async def source_deletion():
                s = await scenario()
                await sources.delete_source(owner, area, await source_identity(s.record), expected_revision=1)
                await unreadable(s)
                self.assertEqual((await row(s))["report_state"], "unknown")
                return {"real_source_deletion_invalidates_read_and_guard": True}

            async def report_deletion():
                s, _, record, _, result = await resolved_scenario()
                await sources.delete_source(owner, area, await source_identity(record), expected_revision=1)
                read = await s.actor.read(s.scope, include_inactive=True, limit=100)
                self.assertNotIn(result.check.id, {item.check.id for item in read.snapshots})
                self.assertIn("episode_check_evidence_unavailable:"+str(result.check.id), read.gaps)
                async with pool.acquire() as c:
                    view = await repository._view(c, owner, result.check.canonical_owner_ref.source_id)
                self.assertEqual(view["occurrences"][0]["report_state"], "unknown")
                self.assertEqual(view["report_history"][0]["report_state"], "unknown")
                return {"deleted_report_is_not_current_reassurance": True}

            async def scheduled_compatibility():
                s = await scenario()
                episode_before = await row(s)
                count = await repository.materialize(owner, now=now()+timedelta(days=2))
                self.assertGreaterEqual(count, 1)
                self.assertEqual(await row(s), episode_before)
                scope = await new_scope()
                source, _ = await capture(scope)
                duty = await repository.save(owner, scope.session_id, source,
                    SaveDuty(spec=DutySpec(title="Synthetic scheduled check", first_at=now())))
                occurrence = duty["occurrences"][0]
                self.assertIsNone(occurrence["episode_id"])
                self.assertIsNotNone(occurrence["reminder_id"])
                source, _ = await capture(scope, text="I completed the synthetic scheduled action.")
                changed = await repository.change(owner, scope.session_id, source, ChangeDuty(
                    responsibility_id=UUID(duty["responsibility"]["id"]),
                    expected_revision=duty["responsibility"]["revision"], operation="report_done",
                    occurrence_id=UUID(occurrence["id"])))
                self.assertEqual(changed["occurrences"][0]["report_state"], "user_reported_done")
                return {"pre_kind_scheduler_preserved": True, "episode_ignored_by_scheduler": True,
                        "new_scheduled_save_and_report_work": True}

            async def scheduled_bypass_rejected():
                s = await scenario()
                source, _ = await capture(s.scope)
                await error("episode_versions_required", repository.change(owner, s.scope.session_id, source,
                    ChangeDuty(responsibility_id=s.result.check.canonical_owner_ref.source_id,
                        expected_revision=1, operation="cancel")))
                await error("episode_versions_required", repository.save(owner, s.scope.session_id, source,
                    SaveDuty(responsibility_id=s.result.check.canonical_owner_ref.source_id, expected_revision=1,
                        spec=DutySpec(title="Wrong conversion", first_at=now()))))
                self.assertEqual((await row(s))["check_revision"], 1)
                return {"legacy_api_cannot_bypass_episode_versions": True}

            async def rejected(sql, *args):
                transaction = connection.transaction()
                await transaction.start()
                try:
                    try:
                        await connection.execute(sql, *args)
                    except asyncpg.PostgresError as caught:
                        self.assertIn(caught.sqlstate, {"23514", "23505", "23503"})
                        return {"sqlstate": caught.sqlstate, "constraint": caught.constraint_name}
                    self.fail("Database accepted the invalid episode mutation")
                finally:
                    await transaction.rollback()

            async def shape_constraints():
                s = await scenario()
                checks = []
                checks.append(await rejected("update jarvis_responsibilities set next_due_at=clock_timestamp() "
                    "where owner_id=$1 and id=$2", owner, s.result.check.canonical_owner_ref.source_id))
                checks.append(await rejected("update jarvis_responsibilities set spec=jsonb_set(spec,'{kind}',"
                    "'\"scheduled\"'::jsonb) where owner_id=$1 and id=$2", owner, s.result.check.canonical_owner_ref.source_id))
                checks.append(await rejected("update jarvis_responsibility_occurrences set check_payload='{}'::jsonb,"
                    "check_revision=check_revision+1 where owner_id=$1 and id=$2", owner, s.result.check.id))
                checks.append(await rejected("update jarvis_responsibility_occurrences set reminder_id=null "
                    "where owner_id=$1 and id=$2", owner, legacy_occurrence["id"]))
                checks.append(await rejected("update jarvis_responsibility_occurrences set responsibility_id=$3 "
                    "where owner_id=$1 and id=$2", owner, s.result.check.id, legacy_id))
                return {"rejected_shape_kind_mutations": checks}

            async def null_origin_constraint():
                s = await scenario()
                return await rejected("update jarvis_responsibility_occurrences set "
                    "check_payload=jsonb_set(check_payload,'{origin}','null'::jsonb),"
                    "check_revision=check_revision+1 where owner_id=$1 and id=$2", owner, s.result.check.id)

            async def null_resolution_basis_constraint():
                s = await scenario()
                _, record = await capture(s.scope, at=s.clock[0])
                invalid = resolution(s.result, record).model_dump(mode="json")
                invalid["basis"] = None
                return await rejected("update jarvis_responsibility_occurrences set check_lifecycle='resolved',"
                    "check_resolution=$3::jsonb,check_revision=check_revision+1 "
                    "where owner_id=$1 and id=$2", owner, s.result.check.id, json.dumps(invalid))

            async def version_constraints():
                s = await scenario()
                checks = []
                for fragment, args in (
                    ("check_payload=jsonb_set(check_payload,'{question}','\"Changed without revision\"'::jsonb)", ()),
                    ("check_lifecycle='dismissed',check_revision=check_revision+2", ()),
                    ("episode_revision=episode_revision+2,check_revision=check_revision+1", ()),
                    ("episode_id=$3,check_revision=check_revision+1", (uuid4(),)),
                    ("check_revision=0", ())):
                    checks.append(await rejected("update jarvis_responsibility_occurrences set "+fragment+
                        " where owner_id=$1 and id=$2", owner, s.result.check.id, *args))
                return {"rejected_semantic_identity_version_mutations": checks}

            async def borrowed_guard_rollback():
                s = await scenario()
                marker = uuid4()
                async with pool.acquire() as c:
                    with self.assertRaises(RollbackProbe):
                        async with c.transaction():
                            async with s.actor.guard(s.scope, connection=c, **refs(s.result)) as (borrowed, snapshot):
                                self.assertIs(borrowed, c)
                                self.assertEqual(snapshot.check.id, s.result.check.id)
                                await borrowed.execute("insert into episode_sql_effects(owner_id,id) values($1,$2)", owner, marker)
                                raise RollbackProbe()
                self.assertFalse(await connection.fetchval("select exists(select 1 from episode_sql_effects where id=$1)", marker))
                await guarded(s)
                return {"borrowed_same_connection": True, "rolled_back_effect_rows": 0, "locks_released": True}

            async def concurrent_invalidation(operation):
                s = await scenario()
                identity = await source_identity(s.record)
                observed = ObservedTransactionPool(rival_pool)
                rival_sources = SourceRepository(observed)
                rival_context = ContextRepository(TaskRepository(observed, TaskLimits()),
                    publication=CanonicalSourcePublication(rival_sources, area))
                rival = None
                try:
                    async with s.actor.guard(s.scope, **refs(s.result)) as (c, _):
                        holder_pid = await c.fetchval("select pg_backend_pid()")
                        if operation == "correct":
                            rival = asyncio.create_task(capture(s.scope, previous=s.record,
                                text="Corrected synthetic request", repo=rival_context))
                        else:
                            rival = asyncio.create_task(rival_sources.delete_source(owner, area, identity, expected_revision=1))
                        await asyncio.wait_for(observed.ready.wait(), timeout=4)
                        self.assertNotEqual(holder_pid, observed.pid)
                        async with asyncio.timeout(4):
                            while not await connection.fetchval("select $1::int=any(pg_blocking_pids($2::int))",
                                                                 holder_pid, observed.pid):
                                if rival.done():
                                    await rival
                                    self.fail("Invalidation completed without the expected active guard lock")
                                await asyncio.sleep(0.05)
                        self.assertFalse(rival.done())
                    await asyncio.wait_for(rival, timeout=15)
                    await unreadable(s)
                    return {"physical_active_transaction_pids": True, "pg_blocking_pids_confirmed": True,
                            "invalidation_finishes_after_guard": True, "next_guard_rejected": True}
                finally:
                    if rival is not None:
                        if not rival.done():
                            rival.cancel()
                        try:
                            await rival
                        except BaseException:
                            pass

            async def tool_bundle(*, fail_receipt=False):
                scope = await new_scope(other)
                tool_context = ContextRepository(tasks,
                    publication=CanonicalSourcePublication(sources, tool_area))
                capture_type = FailOnceReceiptCapture if fail_receipt else SessionContextCapture
                voice_capture = capture_type(tool_context, owner_id=other, session_id=scope.session_id,
                    provider="openai", audio_session_id="episode-sql-"+uuid4().hex, write_timeout=15)
                captures.append(voice_capture)
                input_id, call_id = "input-"+uuid4().hex, "call-"+uuid4().hex
                voice_capture.input_final(input_id, "Track the synthetic object check for this fixture.")
                voice_capture.bind_call(call_id, input_id)
                self.assertTrue(await voice_capture.flush())
                original = await voice_capture.request_for_call(call_id)
                self.assertIsNotNone(original)
                actor = EpisodeCheckOwner(repository, enabled=True)
                self.assertEqual((await actor.initialize()).status, "available")
                definitions = {tool.name: tool for tool in EpisodeCheckToolModule(actor).build_tools(
                    ToolBuildContext(settings=SimpleNamespace(RESPONSIVE_VOICE_ENABLED=True),
                                     owner_id=other, session_id=scope.session_id))}
                registry = JarvisToolRegistry()
                registry.register_many(definitions.values())
                executor = JarvisToolExecutor(registry, execution_repository=ToolExecutionRepository(inspected),
                    owner_id=other, session_id=scope.session_id, context_capture=voice_capture,
                    require_source_tracking=True)
                return SimpleNamespace(scope=scope, context=tool_context, capture=voice_capture,
                    input_id=input_id, call_id=call_id, original=original, actor=actor,
                    definitions=definitions, executor=executor,
                    arguments={"spec": {"title": "Synthetic tool check"},
                               "check": {"question": "Is the synthetic marker visible?"}})

            async def execute_track(b):
                result = await b.executor.execute("track_episode_check", b.arguments, provider_call_id=b.call_id)
                self.assertIsInstance(result, SourceBoundToolResult)
                self.assertEqual(result["status"], "success")
                return result

            async def unbound_track(b):
                with bind_tool_invocation(provider_call_id=b.call_id, request_event_id=b.original.id,
                                          request_revision=b.original.revision):
                    result = await b.definitions["track_episode_check"].handler(b.arguments)
                self.assertEqual(result["status"], "success")
                return result

            async def receipt_edges(result):
                roots = await connection.fetch("select source_payload_id from jarvis_context_payload_dependencies "
                    "where owner_id=$1 and dependent_payload_id=$2", other, result.payload_id)
                edges = await connection.fetch("select event_id,event_revision,source_payload_id,source_usage "
                    "from jarvis_context_event_dependencies where owner_id=$1 and dependent_payload_id=$2",
                    other, result.payload_id)
                return {item["source_payload_id"] for item in roots}, [dict(item) for item in edges]

            async def tool_source_id(b):
                identity = await connection.fetchval("select source_id from jarvis_knowledge_source_versions "
                    "where owner_id=$1 and payload_id=$2", other, b.original.payload_id)
                self.assertIsNotNone(identity)
                return identity

            async def invalidate_tool_input(b, operation):
                if operation == "correct":
                    b.capture.input_final(b.input_id, "Correction: do not rely on the original synthetic request.")
                    self.assertTrue(await b.capture.flush())
                else:
                    await sources.delete_source(other, tool_area, await tool_source_id(b),
                                                expected_revision=b.original.revision)

            async def successful_receipt_count(b, call_id=None):
                return await connection.fetchval("""select count(*) from jarvis_context_events e
                    join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
                    where e.owner_id=$1 and e.session_id=$2 and e.kind='tool_outcome'
                    and p.content->>'provider_call_id'=$3 and p.content#>>'{result,status}'='success'""",
                    other, b.scope.session_id, call_id or b.call_id)

            async def tool_executor_receipt():
                b = await tool_bundle()
                result = await execute_track(b)
                roots, edges = await receipt_edges(result)
                self.assertEqual(roots, {b.original.payload_id})
                self.assertEqual(edges, [{"event_id": b.original.id, "event_revision": b.original.revision,
                    "source_payload_id": b.original.payload_id, "source_usage": "current"}])
                record = await b.context.get(other, result.event_id, session_id=b.scope.session_id)
                self.assertEqual(record.content["result"], dict(result))
                replay = await b.executor.execute("track_episode_check", b.arguments, provider_call_id=b.call_id)
                self.assertIsInstance(replay, SourceBoundToolResult)
                self.assertEqual(replay.event_id, result.event_id)
                self.assertEqual(await successful_receipt_count(b), 1)
                self.assertFalse(result["physical_action_verified"])
                return {"real_executor_and_session_capture": True, "exact_original_roots_and_current_edges": True,
                        "durable_execution_replays_same_receipt": True}

            async def tool_cross_session_receipt():
                b = await tool_bundle()
                admitted = await execute_track(b)
                later = await tool_bundle()
                result = await later.executor.execute("list_episode_checks", {"limit": 8},
                                                      provider_call_id=later.call_id)
                self.assertIsInstance(result, SourceBoundToolResult)
                self.assertIn(admitted["check"]["id"], {item["id"] for item in result["checks"]})
                record = await later.context.get(other, result.event_id, session_id=later.scope.session_id)
                self.assertNotEqual(record.session_id, b.original.session_id)
                roots, edges = await receipt_edges(result)
                self.assertIn(b.original.payload_id, roots)
                self.assertIn({"event_id": b.original.id, "event_revision": b.original.revision,
                    "source_payload_id": b.original.payload_id, "source_usage": "current"}, edges)
                return {"receipt_in_new_session": True, "prior_session_originals_preserved": True}

            async def tool_receipt_invalidation(operation):
                b = await tool_bundle()
                result = await execute_track(b)
                async with tasks.transaction(other) as c:
                    descendant = await tasks.payload(c, other, {"fixture": "downstream receipt copy"},
                                                    source_payload_ids=(result.payload_id,))
                await invalidate_tool_input(b, operation)
                with self.assertRaises(TaskError) as caught:
                    await restore_source_result(b.capture, owner_id=other, session_id=b.scope.session_id,
                        call_id=b.call_id, name="track_episode_check", saved=result.audit_receipt())
                self.assertEqual(caught.exception.code, "tool_result_source_unavailable")
                async with connection.transaction():
                    self.assertFalse(await connection.fetchval("select lock_jarvis_copy_sources($1,$2::uuid[])",
                        other, [result.payload_id, descendant]))
                replay = await b.executor.execute("track_episode_check", b.arguments, provider_call_id=b.call_id)
                self.assertEqual(replay["error"]["code"], "tool_result_source_unavailable")
                return {"canonical_invalidation_reaches_receipt_and_descendant": True,
                        "durable_replay_refuses_removed_evidence": True}

            async def post_handler_invalidation(operation):
                b = await tool_bundle()
                raw = await unbound_track(b)
                await invalidate_tool_input(b, operation)
                with self.assertRaises((TaskError, ToolInputError)):
                    await b.definitions["track_episode_check"].result_capture(raw,
                        ToolResultCapture(b.capture, b.call_id, "track_episode_check"))
                self.assertEqual(await successful_receipt_count(b), 0)
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_responsibility_occurrences "
                    "where owner_id=$1 and id=$2", other, UUID(raw["check"]["id"])), 1)
                return {"post_handler_source_change_rejects_receipt": True,
                        "committed_mutation_not_misrepresented_as_rolled_back": True}

            async def post_handler_check_change():
                b = await tool_bundle()
                raw = await unbound_track(b)
                current = raw["check"]
                request = ChangeEpisodeCheck(responsibility_id=current["canonical_owner_ref"]["source_id"],
                    expected_revision=current["canonical_owner_ref"]["revision"], occurrence_id=current["id"],
                    expected_check_revision=current["revision"], episode_id=current["episode_or_object_ref"]["source_id"],
                    expected_episode_revision=current["episode_or_object_ref"]["revision"], operation="cancel")
                source = ToolInvocationContext(request_event_id=b.original.id, request_revision=b.original.revision)
                accepted(await b.actor.change(b.scope, source, request))
                with self.assertRaises((TaskError, ToolInputError)):
                    await b.definitions["track_episode_check"].result_capture(raw,
                        ToolResultCapture(b.capture, b.call_id, "track_episode_check"))
                self.assertEqual(await successful_receipt_count(b), 0)
                return {"current_rule_check_revalidated_after_handler": True}

            async def after_commit_capture_failure():
                b = await tool_bundle(fail_receipt=True)
                with self.assertLogs("app.tools.executor", level="ERROR"):
                    result = await b.executor.execute("track_episode_check", b.arguments, provider_call_id=b.call_id)
                self.assertEqual(result["error"]["code"], "tool_receipt_unavailable_after_commit")
                self.assertTrue(result["error"]["local_effect_reported"])
                self.assertTrue(await b.capture.flush())
                self.assertEqual(await successful_receipt_count(b), 0)
                async def occurrence_count():
                    return await connection.fetchval("""select count(*) from jarvis_responsibility_occurrences o
                        join jarvis_responsibilities r on r.owner_id=o.owner_id and r.id=o.responsibility_id
                        where r.owner_id=$1 and r.session_id=$2""", other, b.scope.session_id)
                self.assertEqual(await occurrence_count(), 1)
                same_call = await b.executor.execute("track_episode_check", b.arguments, provider_call_id=b.call_id)
                self.assertEqual(same_call["error"]["code"], "tool_receipt_unavailable_after_commit")
                retry_call = "retry-"+uuid4().hex
                b.capture.bind_call(retry_call, b.input_id)
                retried = await b.executor.execute("track_episode_check", b.arguments, provider_call_id=retry_call)
                self.assertIsInstance(retried, SourceBoundToolResult)
                self.assertTrue(retried["replayed"])
                self.assertEqual(await occurrence_count(), 1)
                self.assertEqual(await successful_receipt_count(b, retry_call), 1)
                return {"real_receipt_insert_rolled_back": True, "truthful_local_effect_error": True,
                        "new_provider_id_reuses_canonical_mutation": True, "occurrences": 1}

            async def tool_report_retry():
                b = await tool_bundle()
                admitted = await execute_track(b)
                report_input, report_call = "report-"+uuid4().hex, "report-call-"+uuid4().hex
                b.capture.input_final(report_input, "I completed the synthetic action.")
                b.capture.bind_call(report_call, report_input)
                self.assertTrue(await b.capture.flush())
                original = await b.capture.request_for_call(report_call)
                value = admitted["check"]
                arguments = {"responsibility_id": value["canonical_owner_ref"]["source_id"],
                    "expected_revision": value["canonical_owner_ref"]["revision"], "occurrence_id": value["id"],
                    "expected_check_revision": value["revision"], "episode_id": value["episode_or_object_ref"]["source_id"],
                    "expected_episode_revision": value["episode_or_object_ref"]["revision"], "operation": "report_done"}
                result = await b.executor.execute("change_episode_check", arguments, provider_call_id=report_call)
                self.assertIsInstance(result, SourceBoundToolResult)
                report_time = original.source_time or original.recorded_at
                self.assertEqual(datetime.fromisoformat(result["check"]["resolution"]["resolved_at"]), report_time)
                self.assertEqual(result["check"]["resolution"]["basis"], "user_report")
                b.actor.clock = lambda: report_time + timedelta(minutes=1)
                retry_call = "report-retry-"+uuid4().hex
                b.capture.bind_call(retry_call, report_input)
                retry = await b.executor.execute("change_episode_check", arguments, provider_call_id=retry_call)
                self.assertIsInstance(retry, SourceBoundToolResult)
                self.assertTrue(retry["replayed"])
                self.assertEqual(retry["check"], result["check"])
                roots, edges = await receipt_edges(retry)
                self.assertEqual(roots, {b.original.payload_id, original.payload_id})
                self.assertEqual({item["event_id"] for item in edges}, {b.original.id, original.id})
                return {"server_uses_original_report_time": True, "new_call_and_later_clock_retry_stable": True,
                        "admission_and_report_originals_bound": True}

            for name, operation in (
                ("episode_admit_retry_without_placeholder_reminder", admission_retry),
                ("concurrent_admission_one_effect", concurrent_admission),
                ("cross_session_read_and_cross_owner_exclusion", cross_session_read),
                ("exact_rule_check_episode_revision_conflicts", revision_conflicts),
                ("new_episode_preserves_unresolved_old_episode", next_episode),
                ("object_rebinding_rejects_old_episode_evidence", rebind_episode),
                ("user_report_resolution_and_stable_retry", user_report),
                ("unqualified_visual_trusted_resolution_rejected", unqualified_resolution),
                ("cancel_does_not_resolve_or_allow_late_resolution", cancellation),
                ("snooze_preserves_tracking_and_gates_effects", snooze),
                ("expiry_preserves_unresolved_obligation", expiry),
                ("admission_source_correction_invalidates_projection", source_correction),
                ("dependency_correction_invalidates_projection", dependency_correction),
                ("source_deletion_invalidates_projection", source_deletion),
                ("resolution_source_deletion_removes_reassurance", report_deletion),
                ("scheduled_materialize_save_report_compatibility", scheduled_compatibility),
                ("scheduled_api_cannot_bypass_episode_fences", scheduled_bypass_rejected),
                ("database_shape_and_kind_constraints", shape_constraints),
                ("database_rejects_json_null_origin", null_origin_constraint),
                ("database_rejects_json_null_resolution_basis", null_resolution_basis_constraint),
                ("database_semantic_revision_and_identity_constraints", version_constraints),
                ("borrowed_guard_rolls_back_and_releases_locks", borrowed_guard_rollback),
            ):
                await check(name, operation)
            for operation in ("correct", "delete"):
                await check(f"concurrent_source_{operation}_serializes_with_guard",
                            lambda operation=operation: concurrent_invalidation(operation))
            # Use the second manifested owner only after the foundation tests'
            # cross-owner exclusion check; do not affect their existing fixtures.
            tool_area = await connection.fetchval("insert into jarvis_knowledge_areas(owner_id,name,domain) "
                "values($1,'Episode tool SQL fixture','test') returning id", other)
            for name, operation in (
                ("tool_executor_receipt_original_roots_and_current_edges", tool_executor_receipt),
                ("tool_cross_session_read_receipt_preserves_original_sources", tool_cross_session_receipt),
                ("tool_receipt_correction_invalidates_descendants", lambda: tool_receipt_invalidation("correct")),
                ("tool_receipt_deletion_invalidates_descendants", lambda: tool_receipt_invalidation("delete")),
                ("post_handler_correction_blocks_receipt", lambda: post_handler_invalidation("correct")),
                ("post_handler_deletion_blocks_receipt", lambda: post_handler_invalidation("delete")),
                ("post_handler_check_change_blocks_receipt", post_handler_check_change),
                ("after_commit_capture_failure_truthful_and_retry_no_duplicate", after_commit_capture_failure),
                ("tool_report_done_uses_original_time_on_new_call_retry", tool_report_retry),
            ):
                await check(name, operation)
            report["status"] = "failed" if failures else "passed"
            save()
        except Exception as caught:
            report.update(status="failed", fatal=sanitized(caught), repository_sql_errors=sql_errors)
            save()
            raise AssertionError("Episode SQL qualification stopped; inspect sanitized private report") from None
        finally:
            try:
                for voice_capture in captures:
                    try:
                        await asyncio.wait_for(voice_capture.close(), timeout=20)
                    except Exception as caught:
                        report.update(status="failed", capture_close=sanitized(caught))
                if rival_pool is not None:
                    try:
                        await asyncio.wait_for(rival_pool.close(), timeout=10)
                    except Exception as caught:
                        rival_pool.terminate()
                        report.update(status="failed", rival_close=sanitized(caught))
                if fixture.manifest["schemas"]:
                    await cleanup_fixture_resources(fixture, report, save, pool=pool, connection=connection)
                else:
                    if connection is not None:
                        await connection.close(timeout=5)
                    report["cleanup"] = {"status": "not_required_no_resources_allocated",
                        "schemas_requested": 0, "independent_remote_audit": "not_attempted"}
            finally:
                save()
                print(json.dumps({"report": str(report_path), "status": report["status"],
                    "checks": [{"name": c["name"], "status": c["status"],
                        **({key: c.get(key) for key in ("error_type", "code", "sqlstate", "repository_sql_errors")}
                           if c["status"] == "failed" else {})} for c in report["checks"]],
                    "cleanup": report["cleanup"]}))
        self.assertFalse(failures, json.dumps(failures))
