"""Migration packaging/readiness checks, not PostgreSQL execution evidence."""
from contextlib import asynccontextmanager
import unittest

from app.migrations.runner import load_migrations, read_schema_readiness, render_fresh_schema


class WorkspaceMigrationTests(unittest.IsolatedAsyncioTestCase):
    def test_append_only_migration_is_packaged_and_rendered(self):
        migrations = load_migrations()
        migration = next(item for item in migrations if item.version == 43)
        self.assertEqual(migration.name, "workspace_receipts")
        for table in ("jarvis_note_operations", "jarvis_workspace_requests"):
            self.assertIn("create table " + table, migration.sql)
        self.assertIn("reject_jarvis_foundation_update()", migration.sql)
        rendered = render_fresh_schema(migrations)
        self.assertIn(migration.sql.rstrip(), rendered)
        self.assertIn(migration.checksum, rendered)

    async def test_ledger_alone_does_not_hide_missing_workspace_tables(self):
        migrations = load_migrations()

        class Connection:
            missing = None

            @asynccontextmanager
            async def transaction(self, **kwargs):
                yield self

            async def fetch(self, *args):
                return [{"version": item.version, "name": item.name, "checksum": item.checksum}
                        for item in migrations]

            async def fetchval(self, sql, *args):
                return not (args and args[0] == self.missing)

        connection = Connection()
        self.assertTrue((await read_schema_readiness(connection)).ready)
        for table in ("jarvis_note_operations", "jarvis_workspace_requests"):
            connection.missing = "public." + table
            readiness = await read_schema_readiness(connection)
            self.assertFalse(readiness.ready)
            self.assertIn("missing_table:" + table, readiness.issues)
