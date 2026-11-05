"""Exact quantities reuse canonical originals and real isolated source transactions."""
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.knowledge.measurements import (
    Measurement, MeasurementCalculation, MeasurementQuery, MeasurementReference,
    MeasurementRepository, MeasurementToolModule, SaveMeasurement, calculate_values,
)
from app.knowledge.models import SourceError
from app.knowledge.lifecycle import SourceLifecycleService
from app.tasks.models import TaskVoiceSource
from app.runtime.feature_config import TaskLimits
from app.tasks.repository import TaskRepository
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from evals.second_brain.source_fixture import source_fixture


def entry(value, *, unit="kg", day="2026-09-12", metric="body mass"):
    return Measurement(subject="Dennis", metric=metric, value=value, unit=unit, measured_on=day)


class MeasurementContractTests(unittest.TestCase):
    def test_decimal_values_and_arithmetic_are_exact(self):
        self.assertEqual(calculate_values([entry("0.1"), entry("0.2")], "sum"), "0.3")
        self.assertEqual(calculate_values([entry("80.2"), entry("81.35")], "difference"), "1.15")
        for value in ("NaN", "Infinity", "1e5000", "0.0000000000001", 0.1):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                entry(value)

    def test_normalized_values_round_trip(self):
        for value in ("1e12", "9" * 18 + "e12", "1e-12", "-123.45"):
            with self.subTest(value=value):
                stored = entry(value)
                self.assertEqual(Measurement.model_validate(stored.model_dump()).value, stored.value)
        with self.assertRaises(ValidationError):
            entry("9" * 30 + "e12")

    def test_units_and_series_are_not_silently_mixed(self):
        for other in (entry("2", unit="lb"), entry("2", metric="bench press")):
            with self.subTest(other=other), self.assertRaises(SourceError):
                calculate_values([entry("1"), other], "sum")

    def test_inputs_require_exact_revision_and_distinct_records(self):
        identity = uuid4()
        with self.assertRaises(ValidationError):
            SaveMeasurement(measurement=entry("1"), measurement_id=identity)
        with self.assertRaises(ValidationError):
            MeasurementCalculation(operation="difference", records=[
                MeasurementReference(measurement_id=identity, revision=1)])
        with self.assertRaises(ValidationError):
            MeasurementCalculation(operation="sum", records=[
                MeasurementReference(measurement_id=identity, revision=1)]*2)
        with self.assertRaises(ValidationError):
            MeasurementQuery(from_date="2026-09-14", through_date="2026-09-12")


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class MeasurementDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.repo = MeasurementRepository(self.f.sources, self.tasks, owner_id=self.f.owner, area_id=self.f.area)
        async with self.f.pool.acquire() as connection:
            self.session, self.later = [await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner) for _ in range(2)]
            self.foreign = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.other)

    async def request(self):
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "insert into jarvis_context_streams(owner_id) values($1) on conflict do nothing", self.f.owner)
            payload = await connection.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available',$2::jsonb) returning id""",
                self.f.owner, json.dumps({"text":"Explicit synthetic measurement request."}))
            return await connection.fetchval("""insert into jarvis_context_events
                (owner_id,session_id,source_namespace,source_event_id,kind,finality,trust,payload_id)
                values($1,$2,'measurement-fixture',$3,'user_transcript','final','user_input',$4) returning id""",
                self.f.owner, self.session, str(uuid4()), payload)

    async def save(self, measurement, *, event=None, **changes):
        event = event or await self.request()
        request = SaveMeasurement(measurement=measurement, **changes)
        with bind_tool_invocation(provider_call_id="call-"+uuid4().hex, request_event_id=event, request_revision=1):
            result = await self.repo.save(self.session, request)
        return result, event

    @staticmethod
    def reference(result):
        row = result["records"][0]
        return MeasurementReference(measurement_id=row["measurement_id"], revision=row["revision"])

    async def test_replay_and_cross_session_exact_history(self):
        first, event = await self.save(entry("80.2"))
        replay, _ = await self.save(entry("80.2"), event=event)
        self.assertTrue(replay["request_replayed"])
        self.assertEqual(first["records"][0]["measurement_id"], replay["records"][0]["measurement_id"])
        with self.assertRaises(SourceError):
            await self.save(entry("81"), event=event)
        second, _ = await self.save(entry("81.35", day="2026-09-14"))
        history = await self.repo.listing(self.later, MeasurementQuery(metric="body mass"))
        self.assertEqual([r["measurement"]["value"] for r in history["records"]], ["80.2", "81.35"])
        result = await self.repo.calculate(self.later, MeasurementCalculation(
            operation="difference", records=[self.reference(first), self.reference(second)]))
        self.assertEqual(result["calculation"]["value"], "1.15")
        self.assertEqual(len(result["sources"]), 2)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_memory_outbox where owner_id=$1 and operation='retain'", self.f.owner), 2)

    async def test_correction_recomputes_preserves_history_and_unrelated_record(self):
        first, _ = await self.save(entry("80.2"))
        second, original_request = await self.save(entry("81.35", day="2026-09-14"))
        unrelated, _ = await self.save(entry("3", unit="km", metric="running distance"))
        identity = second["records"][0]["measurement_id"]
        correction_request = await self.request()
        lifecycle = SourceLifecycleService(self.f.sources, self.tasks)
        await lifecycle.correct(self.f.owner, self.f.area, UUID(identity), expected_revision=1,
            replacement=entry("81.05", day="2026-09-14").model_dump_json(),
            request=TaskVoiceSource(correction_request, 1, self.session))
        corrected = await self.repo.listing(self.later,
            MeasurementQuery(metric="body mass", from_date="2026-09-14", through_date="2026-09-14"))
        self.assertEqual(corrected["records"][0]["revision"], 2)
        with self.assertRaises(SourceError):
            await self.save(entry("81.35", day="2026-09-14"), event=original_request)
        with self.assertRaises(SourceError):
            await self.repo.calculate(self.later, MeasurementCalculation(operation="difference",
                records=[self.reference(first), self.reference(second)]))
        result = await self.repo.calculate(self.later, MeasurementCalculation(operation="difference",
            records=[self.reference(first), self.reference(corrected)]))
        self.assertEqual(result["calculation"]["value"], "0.85")
        old = await self.f.sources.read(self.f.owner, self.f.area, UUID(identity), revision=1)
        self.assertEqual(old.content["measurement"]["value"], "81.35")
        control = await self.repo.calculate(self.later, MeasurementCalculation(
            operation="sum", records=[self.reference(unrelated)]))
        self.assertEqual(control["calculation"]["value"], "3")

    async def test_text_correction_cannot_destroy_exact_record(self):
        first, _ = await self.save(entry("80.2"))
        identity = UUID(first["records"][0]["measurement_id"])
        event = await self.request()
        with self.assertRaises(SourceError) as caught:
            await SourceLifecycleService(self.f.sources, self.tasks).correct(
                self.f.owner, self.f.area, identity, expected_revision=1,
                replacement="It should be eighty kilograms.",
                request=TaskVoiceSource(event, 1, self.session))
        self.assertEqual(caught.exception.code, "measurement_requires_structured_correction")
        current = await self.repo.listing(self.later, MeasurementQuery())
        self.assertEqual(current["records"][0]["revision"], 1)
        self.assertEqual(current["records"][0]["measurement"]["value"], "80.2")

    async def test_pages_and_dates_do_not_claim_complete_totals(self):
        for day in ("2026-09-12", "2026-09-13", "2026-09-14"):
            await self.save(entry("1.1", day=day))
        page = await self.repo.listing(self.later, MeasurementQuery(limit=2))
        self.assertTrue(page["has_more"])
        self.assertEqual(page["next_offset"], 2)
        tail = await self.repo.listing(self.later, MeasurementQuery(limit=2, offset=2))
        self.assertFalse(tail["has_more"])
        self.assertEqual(len(tail["records"]), 1)
        selected = await self.repo.listing(self.later,
            MeasurementQuery(from_date="2026-09-13", through_date="2026-09-13"))
        self.assertEqual(len(selected["records"]), 1)
        total = await self.repo.calculate(self.later, MeasurementCalculation(operation="sum",
            records=[MeasurementReference(measurement_id=r["measurement_id"], revision=r["revision"])
                     for r in [*page["records"], *tail["records"]]]))
        self.assertEqual(total["calculation"]["value"], "3.3")

    async def test_removed_foreign_and_mixed_unit_records_are_not_calculated(self):
        first, _ = await self.save(entry("1"))
        other_unit, _ = await self.save(entry("2", unit="lb"))
        with self.assertRaises(SourceError):
            await self.repo.calculate(self.later, MeasurementCalculation(operation="sum",
                records=[self.reference(first), self.reference(other_unit)]))
        with self.assertRaises(SourceError):
            await self.repo.listing(self.foreign, MeasurementQuery())
        identity = UUID(first["records"][0]["measurement_id"])
        await self.f.sources.delete_source(self.f.owner, self.f.area, identity, expected_revision=1)
        with self.assertRaises(SourceError):
            await self.repo.calculate(self.later, MeasurementCalculation(
                operation="sum", records=[self.reference(first)]))
        self.assertEqual(len((await self.repo.listing(self.later, MeasurementQuery()))["records"]), 1)

    async def test_tools_require_real_request_and_frontier_reads_have_no_write_bypass(self):
        tools = {tool.name:tool for tool in MeasurementToolModule(self.repo).build_tools(
            ToolBuildContext(None, self.f.owner, self.session))}
        result = await tools["save_measurement"].handler(
            SaveMeasurement(measurement=entry("80.2")).model_dump(mode="json"))
        self.assertEqual(result["error"]["code"], "measurement_request_required")
        event = await self.request()
        with bind_tool_invocation(provider_call_id="tool-test", request_event_id=event, request_revision=1):
            saved = await tools["save_measurement"].handler(
                SaveMeasurement(measurement=entry("80.2")).model_dump(mode="json"))
        self.assertEqual(saved["status"], "success")
        readonly = MeasurementRepository(self.f.sources, self.tasks, owner_id=self.f.owner,
            area_id=self.f.area, task_evidence=SimpleNamespace())
        names = {tool.name for tool in MeasurementToolModule(readonly).build_tools(
            ToolBuildContext(None, self.f.owner, self.session))}
        self.assertEqual(names, {"list_measurements", "calculate_measurements"})
