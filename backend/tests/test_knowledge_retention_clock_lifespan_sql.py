"""Explicit-policy SQL qualification, not native time or model-memory acceptance."""
from __future__ import annotations

import calendar
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app import main
from app.knowledge.lifecycle import SourceLifecycleService
from app.knowledge.models import SourceError
from app.runtime.feature_config import MemoryAuthority, SecondBrainSettings
from app.settings import JarvisSettings
from app.tasks.context_events import ContextIngress
from app.tasks.models import TaskVoiceSource
from evals.second_brain.source_fixture import source_fixture


class ControlledRetentionClock:
    """Declared in-process test control; never changes stored event timestamps."""
    def __init__(self, instant):
        self.instant = instant
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.instant

    def advance_to(self, instant):
        if (not isinstance(instant, datetime) or instant.tzinfo is None
                or instant.utcoffset() is None or instant < self.instant):
            raise ValueError("Test retention time must be aware and nondecreasing")
        self.instant = instant


def next_month(instant):
    year, month = (instant.year + 1, 1) if instant.month == 12 else (instant.year, instant.month + 1)
    return instant.replace(year=year, month=month,
                           day=min(instant.day, calendar.monthrange(year, month)[1]))


def test_declared_clock_advances_calendar_month_without_changing_original():
    original = datetime(2027, 1, 31, 12, tzinfo=UTC)
    clock = ControlledRetentionClock(original)
    clock.advance_to(next_month(original))
    assert clock() == datetime(2027, 2, 28, 12, tzinfo=UTC)
    assert original == datetime(2027, 1, 31, 12, tzinfo=UTC)


@pytest.mark.parametrize("invalid", [datetime(2026, 1, 1), "2027-01-01", datetime(2025, 1, 1, tzinfo=UTC)])
def test_declared_clock_rejects_ambiguous_or_backward_control(invalid):
    clock = ControlledRetentionClock(datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="aware and nondecreasing"):
        clock.advance_to(invalid)


class NoExternalTransport:
    """Only lifecycle methods exist; no live bank, storage or model operations."""
    def __init__(self):
        self.closed = 0
        self.external_calls = 0

    async def close(self):
        self.closed += 1

    def diagnostics(self):
        return {"stubbed_external_transport": True, "external_calls": self.external_calls}


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested database required",
)
class RetentionClockLifespanDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"], self.directory, test_name=self.id()))
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select current_schema()"), self.f.fixture.manifest["schemas"][0])
            self.anchor = await db.fetchval("select clock_timestamp()")
            self.session = await db.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
        self.clock = ControlledRetentionClock(self.anchor)
        self.transport = NoExternalTransport()
        self.observations = []
        self.runtime = None
        settings = JarvisSettings(
            _env_file=None, APP_ENV="test", JARVIS_OWNER_ID=self.f.owner,
            JARVIS_STATIC_BEARER_TOKEN="isolated-sql-no-http-admission",
            SUPABASE_DB_URL=self.f.fixture.dsn, OPENAI_API_KEY=None, GEMINI_API_KEY=None,
            VISUAL_REALTIME_ENABLED=False, VISUAL_FAST_PERCEPTION_ENABLED=False,
            JARVIS_RESTORE_IN_PROGRESS=False,
        )
        flags = SecondBrainSettings(
            _env_file=None, enabled="true", knowledge_enabled="true",
            hindsight_url="http://127.0.0.1:19999", hindsight_api_key="synthetic-no-live-service",
            knowledge_area_id=str(self.f.area), knowledge_shutdown_seconds="1",
        )

        async def initialize_base(services):
            services.db_pool = self.f.pool
            services.memory_authority = MemoryAuthority.HINDSIGHT

        async def ingestion_without_transport(worker, owner, worker_id, stop, **kwargs):
            self.assertEqual(owner, self.f.owner)
            await stop.wait()

        for target, replacement in (
            ("app.main.get_settings", lambda: settings),
            ("app.main.configure_logging", lambda *args, **kwargs: None),
            ("app.main._initialize_runtime_services", initialize_base),
            ("app.main.initialize_task_runtime", AsyncMock()),
            ("app.main.configure_selected_memory", lambda services: None),
            ("app.main.initialize_attention_runtime", AsyncMock()),
            ("app.main._initialize_reminder_services", AsyncMock()),
            ("app.knowledge.runtime.get_second_brain_settings", lambda: flags),
            ("app.knowledge.runtime.HindsightClient", lambda *args, **kwargs: self.transport),
            ("app.knowledge.runtime.IngestionWorker.run", ingestion_without_transport),
        ):
            self.enterContext(patch(target, new=replacement))
        self.addAsyncCleanup(self.record_joined_case)
        injected = self._testMethodName != "test_default_database_clock"
        self.app = main.create_app(**({"retention_clock": self.clock} if injected else {}))
        await self.enterAsyncContext(self.app.router.lifespan_context(self.app))
        services = self.app.state.runtime_services
        self.runtime = services.knowledge_runtime
        self.assertIsNotNone(self.runtime, services.knowledge_issue)
        self.assertIs(self.runtime.retention._clock, self.clock if injected else None)
        self.assertIs(self.runtime.retention.sources, self.runtime.sources)
        self.service = self.runtime.retention
        self.tasks = self.runtime.context_repository.tasks
        self.request = await self.new_request()

    async def record_joined_case(self):
        runtime = self.runtime
        joined = runtime is not None and runtime._closed and all(
            task is not None and task.done()
            for task in (runtime.task, runtime.file_task, runtime.retention_task))
        receipt = {
            "format": "jarvis.retention_clock.lifespan_sql_case.v1", "test": self.id(),
            "fixture_manifest": str(self.f.fixture.path), "run_id": self.f.fixture.run_id,
            "owner_id": self.f.owner, "area_id": str(self.f.area),
            "physical_anchor": self.anchor.isoformat(), "logical_test_time": self.clock.instant.isoformat(),
            "clock_is_declared_in_process_test_control": True,
            "observations": self.observations, "runtime_joined": joined,
            "transport_closed": self.transport.closed, "external_calls": self.transport.external_calls,
            "native_phase_or_model_acceptance": "not_measured",
        }
        with (self.directory / ("retention-case-" + self.f.fixture.run_id + ".json")).open("x") as handle:
            json.dump(receipt, handle, indent=2, default=str)
            handle.write("\n")
        if runtime is not None:
            self.assertTrue(joined)
            self.assertEqual(self.transport.closed, 1)

    async def new_request(self):
        event = await self.runtime.context_repository.append(self.f.owner, ContextIngress(
            session_id=self.session, source_namespace="fixture:retention-clock-authority",
            source_event_id=uuid4().hex, receipt_id=uuid4().hex,
            kind="user_transcript", trust="user_input", source_time=self.anchor,
            payload={"text": "Synthetic explicit policy request for an isolated retention test."}))
        return TaskVoiceSource(event.id, event.revision, self.session)

    async def publish(self, label):
        return await self.f.publish(label, source_key=uuid4().hex, source_time=self.anchor)

    async def schedule(self, source, deadline, **changes):
        return await self.service.set_policy(self.f.owner, self.f.area, source.source_id, **(dict(
            expected_source_revision=source.revision, expected_source_generation=source.generation,
            expected_policy_revision=0, expires_at=deadline, request=self.request) | changes))

    async def physical_snapshot(self, source):
        async with self.f.pool.acquire() as db:
            created = await db.fetchval(
                "select created_at from jarvis_knowledge_sources where owner_id=$1 and id=$2",
                self.f.owner, source.source_id)
            versions = await db.fetch("""select revision,recorded_at,source_time from jarvis_knowledge_source_versions
                where owner_id=$1 and source_id=$2 order by revision""", self.f.owner, source.source_id)
        self.assertIsNotNone(created)
        return {"source_created_at": created, "versions": [dict(row) for row in versions]}

    async def pass_at(self, instant):
        self.clock.advance_to(instant)
        result = await self.service.process_due(self.f.owner, limit=16)
        self.observations.append({"test_clock": instant, "retention_results": result})
        return result

    async def assert_available_unchanged(self, source, before):
        current = await self.runtime.sources.read(self.f.owner, self.f.area, source.source_id)
        self.assertEqual((current.revision, current.content_digest), (source.revision, source.content_digest))
        self.assertEqual(await self.physical_snapshot(source), before)

    async def test_persistent_and_dependent_sources_survive_day_and_month(self):
        persistent = await self.publish("Persistent synthetic fact; no expiry policy.")
        held = await self.publish("Synthetic fact with an available dependent original.")
        before = {str(source.source_id): await self.physical_snapshot(source) for source in (persistent, held)}
        async with self.tasks.transaction(self.f.owner) as db:
            dependent = await self.tasks.payload(db, self.f.owner, {"text": "Protected dependent evidence."},
                                                  source_payload_ids=(held.payload_id,))
        await self.schedule(held, self.anchor + timedelta(days=1))
        for instant in (self.anchor + timedelta(days=1), next_month(self.anchor)):
            results = await self.pass_at(instant)
            self.assertEqual([(row["source_id"], row["status"], row["reason"]) for row in results],
                             [(str(held.source_id), "scheduled", "retained_dependents")])
            for source in (persistent, held):
                await self.assert_available_unchanged(source, before[str(source.source_id)])
        state = await self.service.inspect(self.f.owner, self.f.area, persistent.source_id)
        self.assertEqual(state["retention"]["status"], "not_scheduled")
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval(
                "select availability from jarvis_context_payloads where owner_id=$1 and id=$2", self.f.owner, dependent), "available")
        self.observations.append({"immutable_timestamps": before, "dependent_kept": str(dependent)})

    async def assert_expiry_threshold(self, deadline):
        source = await self.publish("Unreferenced synthetic source with explicit expiry.")
        before = await self.physical_snapshot(source)
        await self.schedule(source, deadline)
        self.assertEqual(await self.pass_at(deadline - timedelta(microseconds=1)), [])
        await self.assert_available_unchanged(source, before)
        result = await self.pass_at(deadline)
        self.assertEqual([(row["source_id"], row["status"], row["reason"]) for row in result],
                         [(str(source.source_id), "expired", "originals_deleted_provider_cleanup_queued")])
        self.assertFalse(result[0]["full_erasure_verified"])
        self.assertEqual(await self.service.process_due(self.f.owner), [])
        state = await self.service.inspect(self.f.owner, self.f.area, source.source_id)
        self.assertEqual(state["source_state"], "deleted")
        self.assertEqual(await self.physical_snapshot(source), before)
        with self.assertRaises(SourceError):
            await self.runtime.sources.read(self.f.owner, self.f.area, source.source_id)
        async with self.f.pool.acquire() as db:
            self.assertIsNone(await db.fetchval(
                "select content from jarvis_context_payloads where owner_id=$1 and id=$2", self.f.owner, source.payload_id))
            self.assertEqual(await db.fetchval("""select count(*) from jarvis_memory_outbox
                where owner_id=$1 and document_id=$2 and operation='delete'""", self.f.owner, source.document_id), 1)
            audit = dict(await db.fetchrow("""select checked_at,updated_at,next_check_at,clock_timestamp() as physical_now
                from jarvis_source_retention where owner_id=$1 and source_id=$2""", self.f.owner, source.source_id))
        self.assertLessEqual(audit["checked_at"], audit["physical_now"])
        self.assertLessEqual(audit["updated_at"], audit["physical_now"])
        self.assertLess(audit["checked_at"], deadline)
        self.assertEqual(audit["next_check_at"], deadline + timedelta(minutes=1))
        self.observations.append({"immutable_timestamps": before, "physical_audit": audit,
                                  "provider_cleanup": "queued_not_executed"})

    async def test_explicit_expiry_at_day_threshold(self):
        await self.assert_expiry_threshold(self.anchor + timedelta(days=1))

    async def test_explicit_expiry_at_month_threshold(self):
        await self.assert_expiry_threshold(next_month(self.anchor))

    async def test_corrected_source_cannot_inherit_old_revision_expiry(self):
        source = await self.publish("Synthetic original before correction.")
        before = await self.physical_snapshot(source)
        await self.schedule(source, self.anchor + timedelta(days=1))
        corrected = await SourceLifecycleService(self.runtime.sources, self.tasks).correct(
            self.f.owner, self.f.area, source.source_id, expected_revision=source.revision,
            replacement="Synthetic correction requiring new retention authority.", request=await self.new_request())
        self.assertEqual(corrected["revision"], 2)
        result = await self.pass_at(next_month(self.anchor))
        self.assertEqual([(row["status"], row["reason"]) for row in result], [("requires_review", "source_changed")])
        current = await self.runtime.sources.read(self.f.owner, self.f.area, source.source_id)
        self.assertEqual(current.revision, 2)
        after = await self.physical_snapshot(source)
        self.assertEqual(after["source_created_at"], before["source_created_at"])
        self.assertEqual(after["versions"][:1], before["versions"])
        self.assertLess(current.recorded_at, self.anchor + timedelta(days=1))
        self.observations.append({"before": before, "after": after, "current_revision": current.revision})

    async def test_stale_candidate_cannot_expire_replaced_policy(self):
        source = await self.publish("Synthetic source with a subsequently disabled policy.")
        before = await self.physical_snapshot(source)
        await self.schedule(source, self.anchor + timedelta(days=1))
        async with self.f.pool.acquire() as db:
            candidate = dict(await db.fetchrow("""select source_id,policy_revision from jarvis_source_retention
                where owner_id=$1 and source_id=$2""", self.f.owner, source.source_id))
        await self.schedule(source, None, expected_policy_revision=1, request=await self.new_request())
        self.clock.advance_to(next_month(self.anchor))
        self.assertIsNone(await self.service._expire(self.f.owner, candidate))
        self.assertEqual(await self.service.process_due(self.f.owner), [])
        state = await self.service.inspect(self.f.owner, self.f.area, source.source_id)
        self.assertEqual((state["retention"]["policy_revision"], state["retention"]["status"]), (2, "disabled"))
        await self.assert_available_unchanged(source, before)
        self.observations.append({"selected_candidate": candidate, "policy": state["retention"], "immutable_timestamps": before})

    async def test_default_database_clock(self):
        self.assertIsNone(self.service._clock)
        source = await self.publish("Default database-time synthetic source.")
        before = await self.physical_snapshot(source)
        await self.schedule(source, self.anchor + timedelta(days=1))
        self.clock.advance_to(next_month(self.anchor))
        self.assertEqual(await self.service.process_due(self.f.owner), [])
        self.assertEqual(self.clock.calls, 0)
        await self.assert_available_unchanged(source, before)
        self.observations.append({"unused_test_clock": self.clock.instant, "consumer_clock": "database_time",
                                  "immutable_timestamps": before})
