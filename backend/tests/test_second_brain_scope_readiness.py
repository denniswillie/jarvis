"""A migration ledger alone does not establish the session scope tables exist."""
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from app.migrations.runner import load_migrations, read_schema_readiness


class SessionScopeReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_scope_table_is_required_after_migration_32(self):
        migrations=load_migrations()
        @asynccontextmanager
        async def transaction(**kwargs):
            yield
        connection=SimpleNamespace(transaction=transaction,
            fetch=AsyncMock(return_value=[{"version":m.version,"name":m.name,"checksum":m.checksum} for m in migrations]),
            fetchval=AsyncMock(return_value=True))
        complete=await read_schema_readiness(connection,schema="fixture")
        self.assertTrue(complete.ready)
        for table in ("jarvis_session_knowledge_selections","jarvis_knowledge_context_scopes"):
            with self.subTest(table=table):
                async def exists(query,name):
                    return name!="fixture."+table
                connection.fetchval.side_effect=exists
                missing=await read_schema_readiness(connection,schema="fixture")
                self.assertFalse(missing.ready)
                self.assertEqual(missing.pending,())
                self.assertEqual(missing.issues,("missing_table:"+table,))
