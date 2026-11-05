"""Batched source reads retain exact scope, revision and copy validation."""
from datetime import UTC, datetime
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceError, SourceVersion
from app.knowledge.read_provenance import SourceReadProvenance
from app.knowledge.sources import SourceRepository


def source(**changes):
    now = datetime(2026, 10, 30, tzinfo=UTC)
    return SourceVersion(**(dict(owner_id="batch-fixture", area_id=uuid4(),
        source_id=uuid4(), revision=1, generation=1, payload_id=uuid4(),
        document_id="synthetic-document", bank_name="synthetic-bank",
        content_digest="a" * 64, source_type="user_transcript", trust="user_input",
        source_time=now, time_basis="synthetic fixture", recorded_at=now,
        content={"text": "Synthetic source"}, is_current=True, ingestion_status="pending") | changes))


def row(value, **changes):
    return value.model_dump() | dict(current_revision=value.revision,
        context_is_current=True, source_status="active", availability="available",
        current_generation=value.generation) | changes


class SourceReadBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_query_preserves_current_and_explicit_history_order(self):
        current = source(revision=2, generation=2)
        historical = source(source_id=current.source_id, generation=2, is_current=False)
        connection = SimpleNamespace(fetch=AsyncMock(return_value=[
            row(current), row(historical, current_revision=2)]))
        repository = SourceRepository(None)
        values = await repository._read_many(connection, current.owner_id, current.area_id,
            [(current.source_id, None), (current.source_id, 1)])
        self.assertEqual(values, [current, historical])
        connection.fetch.assert_awaited_once()
        self.assertEqual(connection.fetch.call_args.args[1:],
            (current.owner_id, current.area_id, [current.source_id] * 2, [None, 1]))

    async def test_empty_request_does_not_query(self):
        connection = SimpleNamespace(fetch=AsyncMock())
        self.assertEqual(await SourceRepository(None)._read_many(connection, "owner", uuid4(), []), [])
        connection.fetch.assert_not_awaited()

    async def test_missing_or_unavailable_member_fails_the_entire_read(self):
        value = source()
        for invalid in ({"source_id": None}, row(value, availability="deleted"),
                        row(value, source_status="deleted"), row(value, context_is_current=False)):
            with self.subTest(invalid=invalid), self.assertRaises(SourceError):
                connection = SimpleNamespace(fetch=AsyncMock(return_value=[row(value), invalid]))
                await SourceRepository(None)._read_many(connection, value.owner_id, value.area_id,
                    [(value.source_id, None), (uuid4(), None)])

    def binding(self, values):
        connection = SimpleNamespace(is_in_transaction=lambda: True,
            fetch=AsyncMock(return_value=[]), fetchval=AsyncMock(return_value=True),
            executemany=AsyncMock())
        sources = SimpleNamespace(pool=None, _area=AsyncMock(),
            _read_many=AsyncMock(return_value=values))
        binding = SourceReadProvenance(sources)
        binding.tasks.lock_owner = AsyncMock()
        materialize = AsyncMock(return_value=("bound", uuid4()))
        return connection, sources, binding, materialize

    async def test_eight_sources_use_two_bulk_reads_and_keep_both_lock_checks(self):
        values = [source() for _ in range(8)]
        connection, sources, binding, materialize = self.binding(values)
        result = await binding.bind_in_transaction(connection, "owner", uuid4(),
            [value.original_evidence() for value in values], materialize)
        self.assertEqual(result, "bound")
        self.assertEqual(sources._read_many.await_count, 2)
        self.assertEqual(len(sources._read_many.call_args.args[-1]), 8)
        binding.tasks.lock_owner.assert_awaited_once()
        sources._area.assert_awaited_once()
        self.assertIn("lock_jarvis_copy_sources", connection.fetchval.call_args.args[0])
        self.assertIn("for share", connection.fetch.call_args.args[0])
        materialize.assert_awaited_once()
        connection.executemany.assert_awaited_once()

    async def test_changed_reference_before_or_after_locks_never_materializes(self):
        value = source()
        changed = value.model_copy(update={"generation": 2})
        for reads in ([ [changed] ], [ [value], [changed] ]):
            with self.subTest(reads=len(reads)), self.assertRaises(SourceError) as caught:
                connection, sources, binding, materialize = self.binding([value])
                sources._read_many.side_effect = reads
                await binding.bind_in_transaction(connection, "owner", uuid4(),
                    [value.original_evidence()], materialize)
            self.assertEqual(caught.exception.code, "knowledge_read_evidence_changed")
            materialize.assert_not_awaited()
            connection.executemany.assert_not_awaited()

    async def test_unavailable_copy_stops_before_second_validation(self):
        value = source()
        connection, sources, binding, materialize = self.binding([value])
        connection.fetchval.return_value = False
        with self.assertRaises(SourceError) as caught:
            await binding.bind_in_transaction(connection, "owner", uuid4(),
                [value.original_evidence()], materialize)
        self.assertEqual(caught.exception.code, "knowledge_read_evidence_unavailable")
        sources._read_many.assert_awaited_once()
        materialize.assert_not_awaited()

    async def test_repeated_reference_is_resolved_and_recorded_once(self):
        value = source()
        connection, sources, binding, materialize = self.binding([value])
        await binding.bind_in_transaction(connection, "owner", uuid4(),
            [value.original_evidence()] * 2, materialize)
        self.assertEqual(sources._read_many.call_args.args[-1], [(value.source_id, None)])
        self.assertEqual(len(connection.executemany.call_args.args[1]), 1)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class SourceReadBatchDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_sql_matches_single_reads_and_preserves_lifecycle(self):
        from evals.second_brain.source_fixture import source_fixture

        async with source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
                Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),
                test_name=self.id()) as fixture:
            values = [await fixture.publish(source_key=f"batch-{i}") for i in range(8)]
            requests = [(value.source_id, None) for value in values]
            async with fixture.sources.transaction(fixture.owner) as connection:
                expected = [await fixture.sources._read(connection, fixture.owner, fixture.area, identity)
                            for identity, _ in requests]
                actual = await fixture.sources._read_many(connection, fixture.owner, fixture.area, requests)
                self.assertEqual(actual, expected)
                for owner, area in ((fixture.other, fixture.area), (fixture.owner, uuid4())):
                    with self.assertRaises(SourceError):
                        await fixture.sources._read_many(connection, owner, area, requests)
                with self.assertRaises(SourceError):
                    await fixture.sources._read_many(connection, fixture.owner, fixture.area,
                        requests + [(uuid4(), None)])

            binding = SourceReadProvenance(fixture.sources)
            payload = {"sources": [value.original_evidence() for value in values]}
            roots = await binding.bind_context(fixture.owner, fixture.area, payload)
            self.assertEqual(len(roots), 1)
            self.assertEqual(await binding.bind_context(fixture.owner, fixture.area, payload), roots)

            replacement = await fixture.publish("Corrected synthetic source", source_key="batch-0", expected_revision=1)
            with self.assertRaises(SourceError) as caught:
                await binding.bind_context(fixture.owner, fixture.area, payload)
            self.assertEqual(caught.exception.code, "knowledge_read_evidence_changed")
            old = await fixture.sources.read(fixture.owner, fixture.area, values[0].source_id, revision=1)
            async with fixture.sources.transaction(fixture.owner) as connection:
                mixed = await fixture.sources._read_many(connection, fixture.owner, fixture.area,
                    [(replacement.source_id, None), (replacement.source_id, 1)])
                self.assertEqual(mixed, [replacement, old])

            payload["sources"][0] = replacement.original_evidence()
            await binding.bind_context(fixture.owner, fixture.area, payload)
            await fixture.sources.delete_source(fixture.owner, fixture.area, replacement.source_id,
                expected_revision=2)
            with self.assertRaises(SourceError):
                await binding.bind_context(fixture.owner, fixture.area, payload)

            async with fixture.pool.acquire() as connection:
                await connection.execute("update jarvis_knowledge_areas set status='archived', revision=revision+1 where owner_id=$1 and id=$2",
                    fixture.owner, fixture.area)
            with self.assertRaises(SourceError) as caught:
                await binding.bind_context(fixture.owner, fixture.area,
                    {"sources": [values[1].original_evidence()]})
            self.assertEqual(caught.exception.code, "knowledge_area_unavailable")
