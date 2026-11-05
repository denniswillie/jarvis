"""Bounded, read-only evidence export for an isolated native replay fixture.

This is an evaluation utility, not a production backup or a memory reader tool.
It never deletes data or treats the snapshot as a provider-closure receipt.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import re

import asyncpg

from evals.responsive_voice.native_replay_run import read_json, save

MAX_TABLES = 128
MAX_ROWS_PER_TABLE = 500
MAX_ROWS = 5000
MAX_ROW_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 12 * 1024 * 1024
MAX_STRING = 65536


class ExportError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise ExportError(code)


def scope(resources):
    run_id = resources.get("run_id")
    owner = resources.get("owner_id")
    schema = resources.get("database_schema")
    require(isinstance(run_id, str) and re.fullmatch(r"[0-9a-f]{32}", run_id), "invalid_run_id")
    require(owner == f"direct-image-eval-second-brain-{run_id}-a", "non_fixture_owner")
    require(schema == f"jarvis_sb_eval_{run_id}_0", "non_fixture_schema")
    return {"run_id": run_id, "owner_id": owner, "schema": schema}


def check_strings(value):
    if isinstance(value, str):
        require(len(value) <= MAX_STRING, "string_bound_exceeded")
    elif isinstance(value, dict):
        for key, item in value.items():
            check_strings(key)
            check_strings(item)
    elif isinstance(value, list):
        for item in value:
            check_strings(item)


async def collect(connection, resources):
    binding = scope(resources)
    report = {"format": "jarvis.native_replay.canonical_evidence.v1", "status": "collecting",
              **binding, "started_at": datetime.now(UTC).isoformat(), "tables": {},
              "row_count": 0, "selected_tables": 0, "truncated": False,
              "excluded_relations": {"google_oauth_tokens": "credential_store_not_exported"},
              "limits": {"tables": MAX_TABLES, "rows_per_table": MAX_ROWS_PER_TABLE,
                         "rows_total": MAX_ROWS, "row_bytes": MAX_ROW_BYTES,
                         "output_bytes": MAX_OUTPUT_BYTES, "string_characters": MAX_STRING},
              "meaning": "Owned canonical rows at one read-only snapshot; not provider closure or hearing."}
    async with connection.transaction(isolation="repeatable_read", readonly=True):
        await connection.execute("set local statement_timeout='8s'")
        report["transaction_read_only"] = await connection.fetchval("show transaction_read_only")
        require(report["transaction_read_only"] == "on", "transaction_not_read_only")
        tables = await connection.fetch("""select c.relname as table_name
            from pg_catalog.pg_class c
            join pg_catalog.pg_namespace n on n.oid=c.relnamespace
            where n.nspname=$1 and c.relkind in ('r','p') and c.relname<>'google_oauth_tokens'
                and exists(select 1 from pg_catalog.pg_attribute a
                    where a.attrelid=c.oid and a.attname='owner_id'
                        and a.attnum>0 and not a.attisdropped)
            order by c.relname limit $2""", binding["schema"], MAX_TABLES + 1)
        names = [row["table_name"] for row in tables]
        require(0 < len(names) <= MAX_TABLES, "table_bound_exceeded_or_empty")
        require({"jarvis_sessions", "jarvis_knowledge_sources", "jarvis_context_events"}.issubset(names),
                "canonical_schema_incomplete")
        require(len(names) == len(set(names)), "duplicate_catalog_relation")
        report["selected_tables"] = len(names)
        for name in names:
            require(re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", name), "unsupported_catalog_identifier")
            # Catalog-validated identifiers only; owner remains a bound value.
            rows = await connection.fetch(f'''select octet_length(to_jsonb(t)::text) as row_bytes,
                case when octet_length(to_jsonb(t)::text)<=$2 then to_jsonb(t) else null end as row_data
                from "{binding['schema']}"."{name}" t
                where t.owner_id::text=$1 order by t.ctid limit $3''',
                binding["owner_id"], MAX_ROW_BYTES, MAX_ROWS_PER_TABLE + 1)
            require(len(rows) <= MAX_ROWS_PER_TABLE, "table_row_bound_exceeded")
            require(report["row_count"] + len(rows) <= MAX_ROWS, "total_row_bound_exceeded")
            values = []
            for row in rows:
                require(row["row_bytes"] <= MAX_ROW_BYTES and row["row_data"] is not None,
                        "row_byte_bound_exceeded")
                value = json.loads(row["row_data"]) if isinstance(row["row_data"], str) else row["row_data"]
                require(isinstance(value, dict) and str(value.get("owner_id")) == binding["owner_id"],
                        "row_owner_mismatch")
                check_strings(value)
                values.append(value)
            report["tables"][name] = values
            report["row_count"] += len(values)
            require(len((json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False)+"\n").encode()) <= MAX_OUTPUT_BYTES,
                    "output_byte_bound_exceeded")
    report.update(status="captured", finished_at=datetime.now(UTC).isoformat())
    require(len((json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False)+"\n").encode()) <= MAX_OUTPUT_BYTES,
            "output_byte_bound_exceeded")
    return report


async def export(dsn, resources):
    scope(resources)  # Reject foreign or unbound input before making a connection.
    connection = None
    try:
        async with asyncio.timeout(50):
            connection = await asyncpg.connect(dsn, timeout=8, command_timeout=8)
            return await collect(connection, resources)
    finally:
        if connection is not None:
            await connection.close(timeout=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "output_already_exists")
    resources = read_json(args.resource_manifest)
    binding = scope(resources)
    from app.settings import JarvisSettings
    try:
        report = asyncio.run(export(JarvisSettings().SUPABASE_DB_URL, resources))
    except Exception as error:
        report = {"format": "jarvis.native_replay.canonical_evidence.v1", "status": "failed", **binding,
                  "error_type": type(error).__name__,
                  "reason": str(error) if isinstance(error, ExportError) else "export_failed_without_raw_exception"}
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    save(args.output, report)
    print(json.dumps({"status": report["status"], "selected_tables": report.get("selected_tables"),
                      "row_count": report.get("row_count"), "output": str(args.output)}))
    return 0 if report["status"] == "captured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
