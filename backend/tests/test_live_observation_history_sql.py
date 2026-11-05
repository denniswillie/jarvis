"""Real canonical authority versus application-only observational freshness.

Run with evals.second_brain.run_tests --live-db and --manifest-dir. One SQL
case owns one disposable source_fixture schema. No provider, Hindsight worker,
storage upload, native client or model answer is involved. The materialized
voice copy is real SQL; provider publication/ACK and route leases are NOT tested.

Only the application projection/binder clock is advanced. clock_timestamp()
samples PostgreSQL's actual clock; neither database nor provider TTL is mocked.
"""
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceError
from app.knowledge.publication import CanonicalSourcePublication
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.models import (
    GuardedPayloadBinding, LiveSnapshot, ObservationEntry, SessionScope,
    SourceDependency, StateCursor,
)
from app.live_state.projection import prepare_voice_context, select_voice_context
from app.live_state.reducer import ProjectionState, expire
from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError
from app.tasks.repository import TaskRepository
from app.voice_routes.contracts import RouteIdentity
from app.voice_routes.live_context_content import encode_live_context_content
from evals.second_brain.source_fixture import source_fixture


def observation_snapshot(binding, observed_at, summary):
    """No authority is manufactured here: SQL callers supply a real binding."""
    deadline = observed_at + timedelta(seconds=10)
    entry = ObservationEntry(
        entry_id="fixture-observation", scope=binding.scope,
        received_at=observed_at, observed_at=observed_at,
        fresh_until=deadline, expires_at=deadline,
        sources=binding.dependencies, guard_binding=binding,
        evidence_kind="observed", summary=summary,
    )
    return LiveSnapshot(scope=binding.scope,
        cursor=StateCursor(epoch=uuid4(), revision=1), as_of=observed_at,
        readiness="ready", entries=(entry,))


def history_selection(snapshot, now):
    return select_voice_context(snapshot, now=now, max_encoded_bytes=500,
                                preserve_observation_history=True)


class ObservationHistoryContractOfflineTests(unittest.IsolatedAsyncioTestCase):
    """Contract/helper checks only. These fixture roots are not SQL proof."""

    def setUp(self):
        self.now = datetime(2026, 10, 31, 12, tzinfo=UTC)
        scope = SessionScope(owner_id="fixture-offline-only", session_id=uuid4())
        dependency = SourceDependency(source_namespace="context_event",
                                      source_id=uuid4(), revision=1)
        self.binding = GuardedPayloadBinding(scope=scope,
            payload_root_ids=(uuid4(),), dependencies=(dependency,))
        self.snapshot = observation_snapshot(self.binding, self.now, "Synthetic dial reads 42.")
        self.route = RouteIdentity(scope=scope, route_id=uuid4(), route_generation=1)

    async def test_history_preparation_keeps_dates_without_changing_legacy_expiry(self):
        selection = history_selection(self.snapshot, self.now)
        self.assertEqual(selection.selected_entry_ids, ("fixture-observation",))
        bind = AsyncMock(return_value=self.binding)
        update = await prepare_voice_context(selection, self.route, bind=bind, clock=lambda: self.now)
        self.assertIsNotNone(update)
        self.assertIsNone(update.expires_at)
        self.assertEqual(len(update.observation_history), 1)
        window = update.observation_history[0]
        self.assertEqual(window.observed_at, self.now)
        self.assertEqual(window.fresh_until, selection.expires_at)
        self.assertEqual(window.dependencies, self.binding.dependencies)
        self.assertLessEqual(len(encode_live_context_content(update.text).encode("utf-8")), 500)
        legacy = select_voice_context(self.snapshot, now=self.now)
        legacy_update = await prepare_voice_context(legacy, self.route, bind=bind, clock=lambda: self.now)
        self.assertEqual(legacy_update.observation_history, ())
        self.assertEqual(legacy_update.expires_at, window.fresh_until)

    async def test_expired_view_is_unknown_and_stale_new_preparation_never_binds(self):
        selection = history_selection(self.snapshot, self.now)
        later = self.snapshot.entries[0].fresh_until + timedelta(seconds=1)
        reduced = expire(ProjectionState(self.snapshot), now=later)
        self.assertTrue(reduced.applied)
        self.assertEqual(reduced.state.snapshot.entries, ())
        self.assertGreater(reduced.state.snapshot.cursor.revision, self.snapshot.cursor.revision)
        current = history_selection(reduced.state.snapshot, later)
        self.assertEqual(current.selected_entry_ids, ())
        self.assertEqual(current.dependencies, ())
        self.assertEqual(current.observation_history, ())
        self.assertIn("unknown", current.text.lower())
        bind = AsyncMock(return_value=self.binding)
        self.assertIsNone(await prepare_voice_context(selection, self.route, bind=bind, clock=lambda: later))
        bind.assert_not_awaited()


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "requires existing isolated --live-db wrapper and manifest directory")
class ObservationHistorySQLTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]) / self._testMethodName
        self.evidence = {
            "test": self.id(), "completed_checks": [],
            "clock_scope": {"application_clock_advanced": True,
                            "postgres_clock_modified": False, "provider_clock_tested": False},
            "provider_publication_tested": False, "route_lease_loss_tested": False,
            "external_memory_calls": False, "storage_uploads": False,
        }
        self.addAsyncCleanup(self._write_evidence)
        self.stack = AsyncExitStack()
        self.addAsyncCleanup(self.stack.aclose)
        self.fixture = await self.stack.enter_async_context(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"], self.directory, test_name=self.id()))
        self.owner = self.fixture.owner
        self.tasks = TaskRepository(self.fixture.pool, TaskLimits())
        self.context = ContextRepository(self.tasks,
            publication=CanonicalSourcePublication(self.fixture.sources, self.fixture.area))
        self.session = await SessionRepository(self.fixture.pool).create_session(self.owner, "History SQL fixture")
        self.scope = SessionScope(owner_id=self.owner, session_id=self.session.id, area_id=self.fixture.area)
        self.observed_at = self.now = await self._database_time()
        self.binder = CanonicalContextBinder(self.fixture.sources, self.scope, clock=lambda: self.now)
        self.route = RouteIdentity(scope=self.scope, route_id=uuid4(), route_generation=1)
        self.evidence.update(fixture_manifest=str(self.fixture.fixture.path),
            database_before=self.observed_at.isoformat(),
            application_before=self.now.isoformat(), session_id=str(self.session.id))

    async def _database_time(self):
        async with self.fixture.pool.acquire() as connection:
            return await connection.fetchval("select clock_timestamp()")

    async def _write_evidence(self):
        fixture = getattr(self, "fixture", None)
        if fixture is None:
            return  # source_fixture still owns its pre-allocation failure manifest.
        self.evidence.update(fixture_status=fixture.fixture.manifest["status"],
                             cleanup=fixture.fixture.manifest.get("cleanup"))
        path = self.directory / ("authority-checks-" + fixture.fixture.run_id + ".json")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(self.evidence, output, indent=2)

    async def _append(self, identity, summary, *, revision=0):
        return await self.context.append(self.owner, ContextIngress(
            session_id=self.session.id, source_namespace="fixture:observational-history-sql",
            source_event_id=identity, receipt_id=str(uuid4()), expected_revision=revision,
            kind="observation", trust="derived", finality="revised" if revision else "final",
            source_time=self.observed_at, payload={"summary": summary}))

    async def _prepared(self, record):
        binding = await self.binder.bind_record(record, record.content["summary"])
        snapshot = observation_snapshot(binding, record.source_time, record.content["summary"])
        selection = history_selection(snapshot, self.now)
        self.assertEqual(selection.selected_entry_ids, ("fixture-observation",))
        update = await prepare_voice_context(selection, self.route, bind=self.binder, clock=lambda: self.now)
        self.assertIsNotNone(update)
        self.assertIsNotNone(update.guard_binding)
        self.assertIsNone(update.expires_at)
        self.assertEqual(update.observation_history[0].observed_at, record.source_time)
        self.assertTrue(await self.binder.available(update.guard_binding))
        # History-as-dated-content MUST NOT change SQL dependency semantics.
        async with self.fixture.pool.acquire() as connection:
            usage = await connection.fetchval("select source_usage from jarvis_context_event_dependencies "
                "where owner_id=$1 and dependent_payload_id=$2 and event_id=$3",
                self.owner, binding.payload_root_ids[0], record.id)
        self.assertEqual(usage, "current")
        return snapshot, selection, update

    async def _locked(self, owner, binding):
        async with self.fixture.pool.acquire() as connection, connection.transaction():
            return await connection.fetchval("select lock_jarvis_copy_sources($1,$2::uuid[])",
                                             owner, list(binding.payload_root_ids))

    async def _assert_revoked(self, binding):
        self.assertFalse(await self._locked(self.owner, binding))
        with self.assertRaises(SourceError) as raised:
            async with self.binder.guard(binding):
                self.fail("A revoked canonical copy must not enter the publication guard")
        self.assertEqual(raised.exception.code, "live_state_context_revoked")
        async with self.fixture.pool.acquire() as connection:
            rows = await connection.fetch("select availability,content from jarvis_context_payloads "
                "where owner_id=$1 and id=any($2::uuid[])", self.owner, list(binding.payload_root_ids))
        self.assertEqual(len(rows), len(binding.payload_root_ids))
        self.assertTrue(all(row["availability"] != "available" and row["content"] is None for row in rows))

    async def test_dated_history_survives_freshness_but_not_canonical_authority_changes(self):
        correctable = await self._append("correctable", "Synthetic dial reads 42.")
        deletable = await self._append("deletable", "Synthetic indicator is blue.")
        prepared = [await self._prepared(record) for record in (correctable, deletable)]
        self.now = self.observed_at + timedelta(seconds=11)
        self.evidence["application_after"] = self.now.isoformat()
        for snapshot, selection, update in prepared:
            reduced = expire(ProjectionState(snapshot), now=self.now)
            self.assertTrue(reduced.applied)
            current = history_selection(reduced.state.snapshot, self.now)
            self.assertEqual(current.selected_entry_ids, ())
            self.assertEqual(current.dependencies, ())
            self.assertIn("unknown", current.text.lower())
            self.assertLess(update.observation_history[0].fresh_until, self.now)
            self.assertIsNone(await prepare_voice_context(selection, self.route,
                                                         bind=self.binder, clock=lambda: self.now))
            self.assertTrue(await self._locked(self.owner, update.guard_binding))
            self.assertTrue(await self.binder.available(update.guard_binding))
        self.evidence["completed_checks"].append("working_view_expired_canonical_history_still_authorized")

        # A different authenticated owner cannot inherit this history. Both the
        # normal immutable-scope check and an attempted re-scoping hit denial;
        # the latter reaches the REAL SQL root guard under a valid other session.
        history = prepared[0][2].guard_binding
        other_session = await SessionRepository(self.fixture.pool).create_session(self.fixture.other, "Other SQL fixture")
        other_scope = SessionScope(owner_id=self.fixture.other, session_id=other_session.id)
        other = CanonicalContextBinder(self.fixture.sources, other_scope, clock=lambda: self.now)
        with self.assertRaises(SourceError) as raised:
            async with other.guard(history):
                self.fail("An ownership change must not inherit the prior scope")
        self.assertEqual(raised.exception.code, "live_state_scope_changed")
        with self.assertRaises(SourceError) as raised:
            async with other.guard(history.model_copy(update={"scope": other_scope})):
                self.fail("Re-scoping metadata cannot authorize another owner's SQL roots")
        self.assertEqual(raised.exception.code, "live_state_context_revoked")
        self.assertFalse(await self._locked(self.fixture.other, history))
        self.assertTrue(await self.binder.available(history))
        self.evidence["completed_checks"].append("changed_owner_denied_without_revoking_rightful_owner")

        corrected = await self._append("correctable", "Correction: synthetic dial reads 24.", revision=1)
        self.assertEqual(corrected.revision, 2)
        await self._assert_revoked(history)
        replacement = await self.binder.bind_record(corrected, corrected.content["summary"])
        self.assertTrue(await self.binder.available(replacement))
        await self._assert_revoked(history)  # New authorized content cannot revive the old copy.
        self.evidence["completed_checks"].append("true_correction_revokes_prepared_dated_history")

        async with self.fixture.pool.acquire() as connection:
            source_id = await connection.fetchval("select source_id from jarvis_knowledge_source_versions "
                "where owner_id=$1 and payload_id=$2", self.owner, deletable.payload_id)
        self.assertIsNotNone(source_id)
        await self.fixture.sources.delete_source(self.owner, self.fixture.area, source_id, expected_revision=1)
        await self._assert_revoked(prepared[1][2].guard_binding)
        with self.assertRaises(TaskError) as raised:
            await self._append("deletable", "Late observation replay", revision=1)
        self.assertEqual(raised.exception.code, "context_source_removed")
        self.evidence["completed_checks"].append("true_deletion_revokes_history_and_blocks_resurrection")
        self.evidence["database_after"] = (await self._database_time()).isoformat()
        self.evidence["canonical_event_ids"] = [str(correctable.id), str(corrected.id), str(deletable.id)]
        self.evidence["prepared_payload_roots"] = [str(root)
            for _, _, update in prepared for root in update.guard_binding.payload_root_ids]


if __name__ == "__main__":
    unittest.main()
