from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.diagnostics.router import collect_memory_diagnostics, router
from app.errors import JarvisHTTPError
from app.settings import JarvisSettings, get_settings


NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


def _summary(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "observation_count": 4,
        "detailed_observation_count": 3,
        "working_state_count": 1,
        "expired_working_state_count": 0,
        "latest_working_state_at": NOW - timedelta(seconds=2),
        "open_event_count": 1,
        "awaiting_consolidation_count": 2,
        "latest_consolidation_at": NOW - timedelta(seconds=5),
        "active_memory_items_missing_embeddings": 0,
        "consolidated_events_missing_embeddings": 0,
        "dead_job_count": 0,
        "oldest_active_job_at": NOW - timedelta(seconds=7),
        "retrieval_count": 3,
        "retrieval_mean_latency_ms": 12.5,
        "retrieval_p95_latency_ms": 18.0,
    }
    values.update(overrides)
    return values


class _Connection:
    def __init__(self, *, summary: dict[str, Any] | None = None) -> None:
        self.summary = summary or _summary()

    async def fetchrow(self, _query: str, *_args: object) -> dict[str, Any]:
        return self.summary

    async def fetch(self, query: str, *_args: object) -> list[dict[str, Any]]:
        if "from memory_jobs" in query:
            return [{"status": "queued", "count": 2}]
        if "visual_frame_retention_reasons" in query:
            return [
                {
                    "scope": "visual_frame",
                    "reason_type": "important",
                    "count": 1,
                }
            ]
        return [
            {"status": "open", "count": 1},
            {"status": "closed", "count": 2},
        ]


class _AcquireContext:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _Connection:
        return self.connection

    async def __aexit__(self, *_args: object) -> None:
        return None


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self.connection)


class MemoryDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = _Connection()
        self.pool = _Pool(self.connection)
        app = FastAPI()
        app.include_router(router)
        app.state.runtime_services = SimpleNamespace(
            db_pool=self.pool,
            sensory_frame_window=None,
        )
        app.dependency_overrides[get_settings] = lambda: JarvisSettings(
            _env_file=None,
            JARVIS_STATIC_BEARER_TOKEN="diagnostics-token",
            JARVIS_OWNER_ID="owner",
        )

        @app.exception_handler(JarvisHTTPError)
        async def handle_error(
            _request: Request,
            error: JarvisHTTPError,
        ) -> JSONResponse:
            return JSONResponse(
                status_code=error.status_code,
                content={
                    "error": {
                        "code": error.code,
                        "message": error.message,
                        "details": error.details,
                    }
                },
            )

        self.app = app
        self.client = TestClient(app)

    def test_memory_diagnostics_requires_authentication(self) -> None:
        response = self.client.get("/diagnostics/memory")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "jarvis_auth_required")

    def test_memory_diagnostics_returns_aggregate_health_without_content(self) -> None:
        response = self.client.get(
            "/diagnostics/memory",
            headers={"Authorization": "Bearer diagnostics-token"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["perception"]["observation_count"], 4)
        self.assertEqual(payload["jobs"]["counts_by_status"], {"queued": 2})
        self.assertEqual(payload["retrieval"]["p95_latency_ms"], 18.0)
        serialized = str(payload)
        self.assertNotIn("image_bytes", serialized)
        self.assertNotIn("storage_path", serialized)
        self.assertNotIn("content", serialized)

    def test_memory_diagnostics_reports_actionable_degradation(self) -> None:
        self.connection.summary = _summary(
            dead_job_count=1,
            active_memory_items_missing_embeddings=2,
            consolidated_events_missing_embeddings=3,
        )

        payload = asyncio.run(
            collect_memory_diagnostics(
                pool=self.pool,
                sensory_frame_window=None,
                now=NOW,
            )
        )

        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(
            payload["issues"],
            [
                "dead_memory_jobs",
                "active_memory_embeddings_missing",
                "experience_event_embeddings_missing",
            ],
        )

    def test_memory_diagnostics_fails_clearly_without_database(self) -> None:
        self.app.state.runtime_services.db_pool = None

        response = self.client.get(
            "/diagnostics/memory",
            headers={"Authorization": "Bearer diagnostics-token"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["error"]["code"],
            "jarvis_database_not_configured",
        )


if __name__ == "__main__":
    unittest.main()
