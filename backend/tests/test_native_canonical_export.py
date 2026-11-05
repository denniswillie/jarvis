"""Offline boundaries for the native fixture snapshot; SQL acceptance is separate."""
from contextlib import asynccontextmanager
import json
import unittest
from unittest.mock import AsyncMock, patch

from evals.responsive_voice import native_canonical_export as exporter


RUN_ID = "a" * 32
RESOURCES = {"run_id": RUN_ID, "owner_id": f"direct-image-eval-second-brain-{RUN_ID}-a",
             "database_schema": f"jarvis_sb_eval_{RUN_ID}_0"}
TABLES = ["jarvis_context_events", "jarvis_knowledge_sources", "jarvis_sessions"]


class Database:
    def __init__(self, rows=None, names=None, read_only="on"):
        self.rows = rows or {}
        self.names = TABLES if names is None else names
        self.read_only = read_only
        self.queries = []
        self.transaction_options = None

    @asynccontextmanager
    async def transaction(self, **options):
        self.transaction_options = options
        yield self

    async def execute(self, query):
        self.queries.append((query, ()))

    async def fetchval(self, query):
        return self.read_only

    async def fetch(self, query, *arguments):
        self.queries.append((query, arguments))
        if "pg_catalog.pg_class" in query:
            return [{"table_name": name} for name in self.names]
        name = next(name for name in self.names if f'."{name}"' in query)
        return self.rows.get(name, [])


def encoded(value):
    text = json.dumps(value)
    return {"row_bytes": len(text.encode()), "row_data": text}


class CanonicalExportTests(unittest.IsolatedAsyncioTestCase):
    async def test_owned_snapshot_is_read_only_repeatable_and_bound(self):
        row = {"owner_id": RESOURCES["owner_id"], "text": "Synthetic onion observation."}
        db = Database({"jarvis_context_events": [encoded(row)]})
        result = await exporter.collect(db, RESOURCES)
        self.assertEqual(result["status"], "captured")
        self.assertEqual(result["row_count"], 1)
        self.assertEqual(result["selected_tables"], 3)
        self.assertEqual(result["transaction_read_only"], "on")
        self.assertEqual(db.transaction_options, {"isolation": "repeatable_read", "readonly": True})
        self.assertEqual(result["tables"]["jarvis_context_events"], [row])
        for query, arguments in db.queries:
            if 'from "jarvis_' in query:
                self.assertEqual(arguments[0], RESOURCES["owner_id"])
                self.assertIn("t.owner_id::text=$1", query)
                self.assertNotIn('"public"', query)

    async def test_scope_rejection_occurs_before_connection(self):
        for changes in ({"run_id": "not-a-run"}, {"owner_id": "personal-owner"},
                        {"owner_id": "direct-image-eval-second-brain-" + "b" * 32 + "-a"},
                        {"database_schema": "public"}, {"database_schema": 'jarvis_x";drop schema public'}):
            with self.subTest(changes=changes), patch.object(exporter.asyncpg, "connect", new_callable=AsyncMock) as connect:
                with self.assertRaises(exporter.ExportError):
                    await exporter.export("unused", RESOURCES | changes)
                connect.assert_not_awaited()

    async def test_read_write_transaction_is_not_accepted(self):
        with self.assertRaisesRegex(exporter.ExportError, "transaction_not_read_only"):
            await exporter.collect(Database(read_only="off"), RESOURCES)

    async def test_missing_or_excessive_catalog_never_claims_capture(self):
        for names in ([], ["jarvis_sessions"], TABLES + [f"table_{i}" for i in range(126)],
                      TABLES + [TABLES[0]], TABLES + ['invalid"name']):
            with self.subTest(names=names), self.assertRaises(exporter.ExportError):
                await exporter.collect(Database(names=names), RESOURCES)

    async def test_current_schema_cardinality_fits_without_unbounded_export(self):
        db = Database(names=TABLES + [f"fixture_table_{i}" for i in range(91)])
        result = await exporter.collect(db, RESOURCES)
        self.assertEqual(result["selected_tables"], 94)
        self.assertEqual(result["limits"]["tables"], 128)
        self.assertEqual(result["excluded_relations"], {"google_oauth_tokens": "credential_store_not_exported"})
        self.assertIn("c.relname<>'google_oauth_tokens'", db.queries[1][0])

    async def test_foreign_or_oversized_rows_fail_closed(self):
        bad_rows = [encoded({"owner_id": "foreign"}),
                    {"row_bytes": exporter.MAX_ROW_BYTES + 1, "row_data": None},
                    encoded({"owner_id": RESOURCES["owner_id"], "text": "x" * (exporter.MAX_STRING + 1)})]
        for row in bad_rows:
            with self.subTest(reason=row["row_bytes"]), self.assertRaises(exporter.ExportError):
                await exporter.collect(Database({TABLES[0]: [row]}), RESOURCES)

    async def test_row_and_output_caps_are_not_reported_as_full_evidence(self):
        row = encoded({"owner_id": RESOURCES["owner_id"]})
        with self.assertRaisesRegex(exporter.ExportError, "table_row_bound_exceeded"):
            await exporter.collect(Database({TABLES[0]: [row] * 501}), RESOURCES)
        with patch.object(exporter, "MAX_ROWS", 1), self.assertRaisesRegex(exporter.ExportError, "total_row_bound_exceeded"):
            await exporter.collect(Database({TABLES[0]: [row], TABLES[1]: [row]}), RESOURCES)
        with patch.object(exporter, "MAX_OUTPUT_BYTES", 1), self.assertRaisesRegex(exporter.ExportError, "output_byte_bound_exceeded"):
            await exporter.collect(Database({TABLES[0]: [row]}), RESOURCES)


if __name__ == "__main__":
    unittest.main()
