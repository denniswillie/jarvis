"""Explicit real-DB qualification; no providers, services or public migrations.

Use a fresh private output directory. Credentials remain in this process and
redacted logs. The existing manifested fixture owns schema teardown; a separate
connection audits all fixture schemas and public owner rows afterward.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import unittest

import asyncpg
from dotenv import load_dotenv

from app.settings import JarvisSettings, is_configured_secret
from app.migrations.runner import load_migrations
from evals.second_brain.fixtures import DatabaseFixture, _owner_tables, _quote, validate_manifest


class RedactedStream:
    def __init__(self, stream, secrets):
        self.stream, self.secrets = stream, secrets

    def write(self, text):
        for secret in self.secrets:
            text = text.replace(secret, "[redacted]")
        self.stream.write(text)
        self.stream.flush()

    def flush(self):
        self.stream.flush()


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subcases, self.completed = [], []

    def addSubTest(self, test, subtest, error):
        self.subcases.append({"test": test.id(), "parameters": dict(subtest.params),
                              "status": "passed" if error is None else "failed",
                              "error_type": error[0].__name__ if error else None})
        super().addSubTest(test, subtest, error)

    def addSuccess(self, test):
        self.completed.append(test.id())
        super().addSuccess(test)


async def probe(dsn):
    connection = await asyncpg.connect(dsn, timeout=10, command_timeout=10)
    try:
        async with connection.transaction(readonly=True):
            if await connection.fetchval("select 1") != 1:
                raise RuntimeError("database_probe_failed")
    finally:
        await connection.close(timeout=5)


async def independent_cleanup_audit(dsn, directory):
    manifests, cleanup_errors = [], []
    for path in sorted(directory.glob("second-brain-*.json")):
        data = json.loads(path.read_text())
        validate_manifest(data)
        if data["status"] != "cleaned":
            try:
                await DatabaseFixture.load(dsn, path).cleanup()
            except Exception as error:
                cleanup_errors.append({"manifest": path.name, "error_type": type(error).__name__})
        manifests.append(data)
    if not manifests:
        return {"status": "no_fixture_allocated", "manifest_count": 0,
                "remaining_schemas": 0, "remaining_public_owner_rows": 0, "cleanup_errors": cleanup_errors}
    schemas = [schema for data in manifests for schema in data["schemas"]]
    owners = [owner for data in manifests for owner in data["owners"]]
    connection = await asyncpg.connect(dsn, timeout=10, command_timeout=30)
    try:
        async with connection.transaction(readonly=True):
            remaining = await connection.fetchval(
                "select count(*) from pg_namespace where nspname=any($1::text[])", schemas)
            public_rows = 0
            for table in await _owner_tables(connection, "public"):
                public_rows += await connection.fetchval(
                    f"select count(*) from public.{_quote(table['relname'])} where owner_id::text=any($1::text[])", owners)
        return {"status": "passed" if not (remaining or public_rows or cleanup_errors) else "failed",
                "manifest_count": len(manifests), "remaining_schemas": remaining,
                "remaining_public_owner_rows": public_rows, "cleanup_errors": cleanup_errors}
    finally:
        await connection.close(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    os.umask(0o077)
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixtures = output / "fixtures"
    fixtures.mkdir()
    root = Path(__file__).resolve().parents[3]
    load_dotenv(root / "backend/.env", override=False)
    settings = JarvisSettings(_env_file=root / "backend/.env")
    dsn = settings.SUPABASE_DB_URL
    if not is_configured_secret(dsn):
        (output / "preflight.json").write_text(json.dumps({"database_probe": "not_configured"}) + "\n")
        return 2
    try:
        asyncio.run(probe(dsn))
        migrations = load_migrations()
        preflight = {"database_probe": "passed", "migration_versions": [m.version for m in migrations],
                     "quiet_42_discovered": any(m.version == 42 and m.name == "attention_quiet_episodes" for m in migrations)}
    except Exception as error:
        (output / "preflight.json").write_text(json.dumps({"database_probe": "failed",
                                                         "error_type": type(error).__name__}) + "\n")
        print(json.dumps({"status": "preflight_failed", "error_type": type(error).__name__}))
        return 2
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"] = dsn
    os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"] = str(fixtures)
    sys.path.insert(0, str(root / "backend/tests"))
    names = ["test_task_interruption_bursts.TaskInterruptionBurstTests",
             "test_task_interruption_effect_bursts_sql.TaskInterruptionEffectSQLTests"]
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    secrets = sorted({value for key, value in os.environ.items() if len(value) > 5 and
                      any(word in key.upper() for word in ("KEY", "TOKEN", "PASSWORD", "DB_URL"))}, key=len, reverse=True)
    start = time.monotonic()
    result = None
    try:
        with (output / "tests.log").open("x") as stream:
            result = unittest.TextTestRunner(stream=RedactedStream(stream, secrets), verbosity=2,
                                            resultclass=RecordedResult).run(suite)
    finally:
        try:
            cleanup = asyncio.run(independent_cleanup_audit(dsn, fixtures))
        except Exception as error:
            cleanup = {"status": "failed", "error_type": type(error).__name__}
        (output / "cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")
    report = {"executed_at": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.monotonic() - start,
              "tests_collected": result.testsRun, "passed_methods": result.completed,
              "failures": [test.id() for test, detail in result.failures],
              "errors": [test.id() for test, detail in result.errors],
              "skipped": [{"test": test.id(), "reason": reason} for test, reason in result.skipped],
              "subcases": result.subcases, "cleanup": cleanup,
              "quiet_42_discovered": preflight["quiet_42_discovered"],
              "model_provider_calls": 0, "native_agent_launches": 0}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), "tests": result.testsRun,
                      "failures": len(result.failures), "errors": len(result.errors), "skips": len(result.skipped),
                      "subcases": len(result.subcases), "cleanup": cleanup}))
    return 0 if result.wasSuccessful() and not result.skipped and cleanup["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
