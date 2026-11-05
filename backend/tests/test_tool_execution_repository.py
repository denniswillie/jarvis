from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.experience.models import UnsafeToolPayloadError
from app.experience.repository import ToolExecutionRepository


class _AsyncContext:
    def __init__(self, value: object) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(self, *_: object) -> None:
        return None


class _Connection:
    def __init__(self, results: list[dict[str, Any] | None]) -> None:
        self.results = list(results)
        self.calls: list[str] = []

    def transaction(self) -> _AsyncContext:
        return _AsyncContext(self)

    async def fetchrow(self, query: str, *_: object) -> dict[str, Any] | None:
        self.calls.append(query)
        return self.results.pop(0)


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self) -> _AsyncContext:
        return _AsyncContext(self.connection)


class ToolExecutionRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_execution_is_idempotent_for_same_provider_call(self) -> None:
        row = _tool_row()
        connection = _Connection([None, row])
        repository = ToolExecutionRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.start_execution(
            owner_id=row["owner_id"],
            session_id=row["session_id"],
            provider_call_id=row["provider_call_id"],
            tool_name=row["tool_name"],
            arguments={"query": "weather"},
            started_at=row["started_at"],
        )

        self.assertEqual(result.id, row["id"])
        self.assertIn("on conflict (session_id, provider_call_id) do nothing", connection.calls[0])

    async def test_finalize_execution_records_durable_safe_result(self) -> None:
        row = _tool_row(
            status="completed",
            result={"status": "success", "temperature": 18},
        )
        connection = _Connection([row])
        repository = ToolExecutionRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.finalize_execution(
            owner_id=row["owner_id"],
            session_id=row["session_id"],
            provider_call_id=row["provider_call_id"],
            status="completed",
            completed_at=row["completed_at"],
            result={"status": "success", "temperature": 18},
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.result["temperature"], 18)

    async def test_unsafe_result_is_rejected_before_database_access(self) -> None:
        connection = _Connection([])
        repository = ToolExecutionRepository(_Pool(connection))  # type: ignore[arg-type]

        with self.assertRaises(UnsafeToolPayloadError):
            await repository.finalize_execution(
                owner_id="owner",
                session_id=uuid4(),
                provider_call_id="call-1",
                status="completed",
                completed_at=datetime.now(UTC),
                result={"signed_url": "https://example.test/private"},
            )

        self.assertEqual(connection.calls, [])


def _tool_row(
    *,
    status: str = "started",
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "id": uuid4(),
        "owner_id": "owner",
        "session_id": uuid4(),
        "provider_call_id": "call-1",
        "tool_name": "web_search",
        "mutates_state": False,
        "arguments": json.dumps({"query": "weather"}),
        "result": json.dumps(result) if result is not None else None,
        "status": status,
        "error": None,
        "started_at": now,
        "completed_at": now if status != "started" else None,
        "created_at": now,
    }
