"""Cleanup transaction boundaries; actual SQL checks remain separately qualified."""
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from evals.second_brain.fixtures import DatabaseFixture, cleanup_manifest


class FixtureCleanupTransactionTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, directory):
        fixture = DatabaseFixture("unused-local-test", Path(directory), test_name=self.id())
        fixture.manifest["schemas"] = [f"jarvis_sb_eval_{fixture.run_id}_{i}" for i in range(3)]
        fixture.persist()
        return fixture

    async def exercise(self, fixture, *, fail_schema=None, foreign=False):
        transactions = []
        dropped = []
        events = []
        active = None

        @asynccontextmanager
        async def transaction():
            nonlocal active
            token = object()
            active = token
            transactions.append(token)
            events.append("begin")
            try:
                yield
            except BaseException:
                events.append("rollback")
                raise
            else:
                events.append("commit")
            finally:
                active = None

        async def execute(sql, *args, **kwargs):
            self.assertIsNotNone(active)
            if sql.startswith("drop schema"):
                schema = next(item for item in fixture.manifest["schemas"] if item in sql)
                if schema == fail_schema:
                    raise RuntimeError("synthetic second-schema failure")
                self.assertNotIn(active, [token for _, token in dropped])
                dropped.append((schema, active))

        connection = SimpleNamespace(transaction=transaction, execute=AsyncMock(side_effect=execute),
                                     fetchval=AsyncMock(return_value=foreign), close=AsyncMock())
        audit = SimpleNamespace(fetchval=AsyncMock(return_value=0), close=AsyncMock())
        with patch("evals.second_brain.fixtures.asyncpg.connect", AsyncMock(side_effect=[connection, audit])), \
             patch("evals.second_brain.fixtures._owner_tables", AsyncMock(return_value=[])):
            if fail_schema or foreign:
                with self.assertRaises(RuntimeError):
                    await fixture.cleanup()
            else:
                result = await cleanup_manifest("unused-local-test", fixture.manifest)
                self.assertEqual(result["remaining_schemas"], 0)
                self.assertEqual(result["remaining_public_owner_rows"], 0)
                audit.close.assert_awaited_once()
        connection.close.assert_awaited_once()
        return events, dropped, audit

    async def test_three_schema_drops_release_locks_between_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            events, dropped, _ = await self.exercise(fixture)
            self.assertEqual(events, ["begin", "commit"] * 3)
            self.assertEqual([schema for schema, _ in dropped], fixture.manifest["schemas"])

    async def test_partial_failure_keeps_full_manifest_and_does_not_claim_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            schemas = list(fixture.manifest["schemas"])
            events, dropped, audit = await self.exercise(fixture, fail_schema=schemas[1])
            self.assertEqual(events, ["begin", "commit", "begin", "rollback"])
            self.assertEqual([schema for schema, _ in dropped], schemas[:1])
            self.assertEqual(fixture.manifest["schemas"], schemas)
            self.assertEqual(fixture.manifest["status"], "cleanup_failed")
            audit.fetchval.assert_not_awaited()
            # Retry still enumerates the original manifest; DROP IF EXISTS makes
            # an already committed first-schema removal harmless.
            events, dropped, _ = await self.exercise(fixture)
            self.assertEqual(len(dropped), 3)

    async def test_external_reference_still_refuses_drop(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            events, dropped, audit = await self.exercise(fixture, foreign=True)
            self.assertEqual(events, ["begin", "rollback"])
            self.assertEqual(dropped, [])
            audit.fetchval.assert_not_awaited()
