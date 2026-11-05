"""Explicit administration for shared task admission limits; not a model tool."""
import argparse
import asyncio
import json

from app.db import create_pool
from app.runtime.feature_config import SecondBrainSettings, TaskLimits
from app.settings import JarvisSettings
from app.tasks.repository import TaskRepository


async def run():
    raw = SecondBrainSettings()
    limits = TaskLimits(**{name: getattr(raw, name) for name in TaskLimits.model_fields})
    pool = await create_pool(JarvisSettings())
    try:
        print(json.dumps(await TaskRepository(pool, limits).configure_scheduler_limits()))
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configure-limits",))
    parser.parse_args()
    try:
        asyncio.run(run())
    except Exception as error:
        print(json.dumps({"ok": False, "error_type": type(error).__name__}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
