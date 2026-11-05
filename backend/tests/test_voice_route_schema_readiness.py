"""Offline readiness-contract checks; not PostgreSQL migration qualification."""
from contextlib import asynccontextmanager
import unittest

from app.migrations.runner import load_migrations, read_schema_readiness


class ReadinessConnection:
    def __init__(self, migrations, *, missing=()):
        self.rows = [dict(version=m.version, name=m.name, checksum=m.checksum)
                     for m in migrations]
        self.missing = set(missing)
        self.lookups = []
        self.readonly = None

    @asynccontextmanager
    async def transaction(self, *, readonly):
        self.readonly = readonly
        yield self

    async def fetch(self, query):
        return self.rows

    async def fetchval(self, query, relation):
        self.lookups.append(relation)
        return relation not in self.missing


class VoiceRouteSchemaReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_v39_requires_evidence_relation_in_selected_schema(self):
        migrations = load_migrations()
        relation = "qualification.jarvis_attention_route_evidence"
        connection = ReadinessConnection(migrations, missing=(relation,))
        result = await read_schema_readiness(connection, schema="qualification", migrations=migrations)
        self.assertFalse(result.ready)
        self.assertEqual(result.issues, ("missing_table:jarvis_attention_route_evidence",))
        self.assertIn(relation, connection.lookups)
        self.assertNotIn("public.jarvis_attention_route_evidence", connection.lookups)
        self.assertTrue(connection.readonly)

    async def test_complete_v39_schema_is_ready(self):
        migrations = load_migrations()
        connection = ReadinessConnection(migrations)
        result = await read_schema_readiness(connection, migrations=migrations)
        self.assertTrue(result.ready)
        self.assertEqual(result.pending, ())
        self.assertIn("public.jarvis_attention_route_evidence", connection.lookups)

    async def test_explicit_v38_target_does_not_require_v39_relation(self):
        migrations = tuple(m for m in load_migrations() if m.version <= 38)
        relation = "public.jarvis_attention_route_evidence"
        connection = ReadinessConnection(migrations, missing=(relation,))
        result = await read_schema_readiness(connection, migrations=migrations)
        self.assertTrue(result.ready)
        self.assertNotIn(relation, connection.lookups)

    async def test_unapplied_v39_remains_pending(self):
        migrations = load_migrations()
        connection = ReadinessConnection(tuple(m for m in migrations if m.version <= 38))
        result = await read_schema_readiness(connection, migrations=migrations)
        self.assertFalse(result.ready)
        self.assertIn(39, result.pending)
        self.assertNotIn("public.jarvis_attention_route_evidence", connection.lookups)
