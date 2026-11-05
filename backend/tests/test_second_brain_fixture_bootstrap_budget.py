"""Schema DDL has bounded setup/teardown budgets, not relaxed application limits."""
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from evals.second_brain.fixtures import DatabaseFixture, cleanup_manifest
from evals.second_brain.source_fixture import source_fixture


class FixtureBootstrapBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_longer_bootstrap_budget_does_not_change_pool_query_deadlines(self):
        connection = SimpleNamespace(close=AsyncMock(), fetchval=AsyncMock(return_value=uuid4()))
        @asynccontextmanager
        async def acquire():
            yield connection
        pool = SimpleNamespace(acquire=acquire, close=AsyncMock())
        with TemporaryDirectory() as directory, \
             patch("evals.second_brain.source_fixture.asyncpg.connect", new_callable=AsyncMock, return_value=connection) as connect, \
             patch("evals.second_brain.source_fixture.asyncpg.create_pool", new_callable=AsyncMock, return_value=pool) as create_pool, \
             patch("evals.second_brain.source_fixture.apply_migrations", new_callable=AsyncMock) as migrate, \
             patch.object(DatabaseFixture, "create_schema", new_callable=AsyncMock, return_value="jarvis_fixture_budget") as create, \
             patch.object(DatabaseFixture, "cleanup", new_callable=AsyncMock):
            async with source_fixture("fixture-only-dsn", Path(directory), test_name=self.id()):
                self.assertEqual(connect.call_args.kwargs, {"timeout": 10, "command_timeout": 300})
                self.assertEqual(create_pool.call_args.kwargs["command_timeout"], 30)
                migrate.assert_awaited_once_with(connection, schema="jarvis_fixture_budget", fresh=True, migrations=None)
                create.assert_awaited_once_with(connection)
        pool.close.assert_awaited_once()
        connection.close.assert_awaited_once_with(timeout=5)

    async def test_only_owned_schema_drop_has_the_longer_cleanup_deadline(self):
        @asynccontextmanager
        async def transaction():
            yield
        connection = SimpleNamespace(transaction=transaction, fetchval=AsyncMock(return_value=False),
                                     execute=AsyncMock(), close=AsyncMock())
        audit = SimpleNamespace(fetchval=AsyncMock(return_value=0), close=AsyncMock())
        with TemporaryDirectory() as directory:
            fixture = DatabaseFixture("fixture-only-dsn", Path(directory), test_name=self.id())
            schema = "jarvis_sb_eval_" + fixture.run_id + "_0"
            fixture.manifest["schemas"] = [schema]
            with patch("evals.second_brain.fixtures.asyncpg.connect", new_callable=AsyncMock,
                       side_effect=[connection, audit]) as connect, \
                 patch("evals.second_brain.fixtures._owner_tables", new_callable=AsyncMock, return_value=[]):
                result = await cleanup_manifest("fixture-only-dsn", fixture.manifest)
        self.assertEqual(result["remaining_schemas"], 0)
        self.assertEqual(result["remaining_public_owner_rows"], 0)
        self.assertTrue(all(call.kwargs["command_timeout"] == 30 for call in connect.call_args_list))
        connection.execute.assert_awaited_once_with(f'drop schema if exists "{schema}" cascade', timeout=120)
