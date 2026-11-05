from __future__ import annotations

import argparse
import asyncio
import json
import os

import asyncpg

from app.diagnostics.router import collect_memory_diagnostics


async def inspect(database_url: str) -> dict[str, object]:
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)
    try:
        return await collect_memory_diagnostics(
            pool=pool,
            sensory_frame_window=None,
        )
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect aggregate Jarvis memory health without memory content"
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("JARVIS_TEST_DB_URL") or os.getenv("SUPABASE_DB_URL"),
    )
    args = parser.parse_args()
    if not args.database_url:
        parser.error("--database-url, JARVIS_TEST_DB_URL, or SUPABASE_DB_URL is required")
    result = asyncio.run(inspect(args.database_url))
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
