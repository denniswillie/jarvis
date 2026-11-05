"""Bounded operator CLI for the stopped-writer legacy migration."""
import argparse
import asyncio
import json
from uuid import UUID, uuid4

import asyncpg

from app.knowledge.backfill import LegacyBackfill
from app.knowledge.sources import SourceRepository
from app.knowledge.legacy_populations import SOURCE_TABLES
from app.settings import JarvisSettings


async def execute(args):
    settings = JarvisSettings()
    pool = await asyncpg.create_pool(settings.SUPABASE_DB_URL, min_size=1, max_size=3,
                                     timeout=10, command_timeout=45)
    try:
        importer = LegacyBackfill(SourceRepository(pool))
        if args.command == "inspect":
            async with importer.snapshot(settings.JARVIS_OWNER_ID, max_rows=args.max_rows) as snapshot:
                return {
                    "input_digest": snapshot.input_digest, "population_counts": snapshot.counts,
                    "candidate_sources": sum(row.table in SOURCE_TABLES for row in snapshot.rows.values()),
                    "writes": False, "provider_contacted": False, "ready_for_cutover": False,
                }
        return await importer.run(
            settings.JARVIS_OWNER_ID, args.area, run_id=args.run_id,
            batch_size=args.batch_size, max_rows=args.max_rows)
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "import"))
    parser.add_argument("--area", type=UUID)
    parser.add_argument("--run-id", type=UUID, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--legacy-writers-stopped", action="store_true",
        help="Attest that the old application/writers remain stopped between batches.")
    args = parser.parse_args()
    if args.command == "import":
        if args.area is None or not args.legacy_writers_stopped:
            parser.error("import requires --area and --legacy-writers-stopped")
        args.run_id = args.run_id or uuid4()
    try:
        result = asyncio.run(execute(args))
    except Exception as error:
        # No connection string, SQL detail, source prose or credentials in output.
        print(json.dumps({"ok": False, "run_id": str(args.run_id) if args.run_id else None,
                          "error_type": type(error).__name__, "code": getattr(error, "code", None),
                          "sqlstate": getattr(error, "sqlstate", None)}))
        return 1
    print(json.dumps({"ok": True, **result}))
    return 2 if result.get("status") == "requires_attention" else 0


if __name__ == "__main__":
    raise SystemExit(main())
