"""Explicit CLI. It never logs connection strings or provider credentials."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import sys

import asyncpg

from app.migrations.runner import apply_migrations, read_schema_readiness, render_fresh_schema
from app.settings import JarvisSettings, is_configured_secret


async def _run(args) -> int:
    settings = JarvisSettings()
    if not is_configured_secret(settings.SUPABASE_DB_URL):
        print(json.dumps({"ok": False, "error": "database_not_configured"}))
        return 1
    connection = await asyncpg.connect(settings.SUPABASE_DB_URL, timeout=10, command_timeout=90)
    try:
        if args.command == "check":
            result = await read_schema_readiness(connection, schema=args.schema)
            print(json.dumps(asdict(result)))
            return 0 if result.ready else 1
        added = await apply_migrations(connection, schema=args.schema, fresh=args.command == "init")
        print(json.dumps({"ok": True, "applied": added}))
        return 0
    finally:
        await connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "apply", "init", "render"))
    parser.add_argument("--schema", default="public")
    args = parser.parse_args()
    if args.command == "render":
        sys.stdout.write(render_fresh_schema())
        return 0
    try:
        return asyncio.run(_run(args))
    except Exception as error:
        # A database exception can include SQL payloads or connection details.
        print(json.dumps({"ok": False, "error_type": type(error).__name__,
                          "sqlstate": getattr(error, "sqlstate", None)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
