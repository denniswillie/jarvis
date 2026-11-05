"""Small experiment-only protocol and effect helpers; not production policy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3


def tool_result(payload):
    """Hermes ordinary handlers return text, not arbitrary Python dictionaries."""
    return payload if isinstance(payload, str) else json.dumps(payload, allow_nan=False)


def fixture_effect(database: Path, operation: str, payload: dict):
    """One transactional fixture effect per trusted logical operation, across processes."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    with sqlite3.connect(database, timeout=10) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS effects (operation TEXT PRIMARY KEY, digest TEXT NOT NULL, payload TEXT NOT NULL)")
        inserted = connection.execute(
            "INSERT OR IGNORE INTO effects VALUES (?, ?, ?)", (operation, digest, encoded)
        ).rowcount == 1
        row = connection.execute("SELECT digest FROM effects WHERE operation = ?", (operation,)).fetchone()
        if row[0] != digest:
            raise ValueError("Logical operation reused with different content")
    return {"operation": operation, "created": inserted, "receipt": digest}


def assert_tool_receipts(output):
    """Inspect native receipts rather than trusting agent completion flags."""
    for message in output.get("messages", []):
        if message.get("role") != "tool":
            continue
        content = message.get("content", "")
        if "tool_result_contract" in str(content):
            raise AssertionError("Hermes rejected a tool result")
