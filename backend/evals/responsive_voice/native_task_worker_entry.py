"""Explicit fixture entrypoint; normal app.tasks.worker CLI remains unchanged."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from app.tasks.worker import run_command, startup_failure_receipt
from evals.responsive_voice.native_task_release_control import (
    BoundReleaseWaiter, ReleaseControlError, load_config, read_private,
)
from evals.responsive_voice.native_task_result_hold import ResultHoldProvider


def configured_control(environ):
    raw = environ.get("JARVIS_NATIVE_RESULT_HOLD_CONFIG")
    if not raw:
        raise ReleaseControlError("explicit_hold_config_required")
    config = load_config(raw)
    expected = {"JARVIS_OWNER_ID": config.scope.owner_id,
        "SECOND_BRAIN_WORKER_HOST_ID": config.scope.worker_host_id,
        "SECOND_BRAIN_FRONTIER_MODEL": config.scope.model}
    if any(environ.get(key) != value for key, value in expected.items()):
        raise ReleaseControlError("hold_environment_scope_mismatch")
    native = Path(environ.get("SECOND_BRAIN_HERMES_WORK_ROOT", ""))
    secret = Path(environ.get("JARVIS_FRONTIER_DB_URL_FILE", ""))
    if (not native.is_absolute() or native.resolve() != config.worker_root / "native"
            or not secret.is_absolute() or secret.parent.resolve() != config.worker_root or secret.name != "db.secret"):
        raise ReleaseControlError("hold_environment_path_mismatch")
    parts = urlsplit(read_private(config.worker_root, "db.secret").decode().strip())
    query = parse_qs(parts.query, keep_blank_values=True)
    if (parts.scheme not in {"postgres", "postgresql"} or "options" in query
            or query.get("search_path") != [f'"{config.scope.database_schema}",pg_catalog,extensions']):
        raise ReleaseControlError("hold_environment_schema_mismatch")
    return config


async def run_entry(*, environ=None):
    config = configured_control(os.environ if environ is None else environ)
    waiter = BoundReleaseWaiter(config)

    def transform(provider):
        if provider.model != config.scope.model:
            raise ReleaseControlError("hold_provider_model_mismatch")
        return ResultHoldProvider(provider, waiter, max_hold_seconds=config.max_hold_seconds)

    return await run_command("run", provider_transform=transform)


def main():
    try:
        return asyncio.run(run_entry())
    except Exception as error:
        print(json.dumps(startup_failure_receipt(error)), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
