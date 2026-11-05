from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError
from app.memory.router import router
from app.memory.projection import MemoryProfile


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


class _MemoryService:
    def __init__(self) -> None:
        self.owner_ids = []

    async def list_recent(self, **_kwargs):
        self.owner_ids.append(_kwargs["owner_id"])
        return [
            SimpleNamespace(
                id=UUID(int=1),
                session_id=UUID(int=2),
                memory_type="goal",
                status="active",
                title="Reach the station",
                content="Reach the station before noon.",
                summary="Reach the station.",
                source_experience_event_id=UUID(int=8),
                occurred_at=NOW,
                valid_from=NOW,
                valid_to=None,
                importance=0.8,
                confidence=1.0,
                source="explicit_agent_memory",
                is_explicit=True,
                superseded_by_memory_id=None,
                created_at=NOW,
                updated_at=NOW,
            )
        ]

    async def working_state(self, **_kwargs):
        self.owner_ids.append(_kwargs["owner_id"])
        return None, None

    async def explain(self, **_kwargs):
        self.owner_ids.append(_kwargs["owner_id"])
        memory = (await self.list_recent(owner_id=_kwargs["owner_id"]))[0]
        evidence = SimpleNamespace(
            evidence=SimpleNamespace(
                id=UUID(int=9),
                memory_item_id=memory.id,
                evidence_type="frame",
                evidence_ref_id=str(UUID(int=10)),
                evidence_time=NOW,
                excerpt="Expired frame",
                confidence=1.0,
                created_at=NOW,
            ),
            source_available=True,
            retained_image_available=False,
        )
        return SimpleNamespace(
            memory=memory,
            evidence=(evidence,),
            links=(),
            related_memories=(),
        )

    async def evidence(self, **_kwargs):
        explanation = await self.explain(**_kwargs)
        return explanation.evidence


class _ProjectionService:
    def __init__(self) -> None:
        self.owner_ids: list[str] = []

    async def get_profile(self, *, owner_id: str) -> MemoryProfile:
        self.owner_ids.append(owner_id)
        return MemoryProfile(
            owner_id=owner_id,
            markdown="# Jarvis Memory Profile\n",
            content_hash="a" * 64,
            generated_at=NOW,
            source_revision=3,
            projected_revision=3,
        )


class MemoryRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(router)
        self.service = _MemoryService()
        self.projection_service = _ProjectionService()
        app.state.runtime_services = SimpleNamespace(
            memory_service=self.service,
            memory_projection_service=self.projection_service,
        )
        app.dependency_overrides[require_static_bearer] = lambda: JarvisPrincipal(
            owner_id="owner"
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

        self.client = TestClient(app)

    def test_recent_memory_is_owner_scoped_and_cursor_shaped(self) -> None:
        response = self.client.get("/memory/recent")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["items"][0]["memory_type"], "goal")
        self.assertIsNone(payload["next_cursor"])
        serialized = str(payload)
        self.assertNotIn("storage_path", serialized)
        self.assertNotIn("embedding", serialized)
        self.assertEqual(self.service.owner_ids, ["owner"])

    def test_invalid_cursor_has_stable_error_code(self) -> None:
        response = self.client.get("/memory/recent?cursor=not-a-cursor")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["error"]["code"],
            "invalid_memory_cursor",
        )

    def test_detail_and_evidence_report_expired_frame_provenance(self) -> None:
        memory_id = str(UUID(int=1))
        detail = self.client.get(f"/memory/{memory_id}")
        evidence = self.client.get(f"/memory/{memory_id}/evidence")

        self.assertEqual(detail.status_code, 200)
        self.assertEqual(evidence.status_code, 200)
        self.assertEqual(
            detail.json()["memory"]["source_experience_event_id"],
            str(UUID(int=8)),
        )
        self.assertIn("expired", detail.json()["uncertainty"])
        self.assertFalse(evidence.json()["items"][0]["retained_image_available"])

    def test_working_state_uses_authenticated_owner(self) -> None:
        response = self.client.get("/memory/working-state")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"session": None, "owner": None})
        self.assertEqual(self.service.owner_ids[-1], "owner")

    def test_profile_uses_authenticated_owner_and_projection_contract(self) -> None:
        response = self.client.get("/memory/profile")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["profile"]["markdown"],
            "# Jarvis Memory Profile\n",
        )
        self.assertFalse(response.json()["profile"]["is_stale"])
        self.assertEqual(self.projection_service.owner_ids, ["owner"])


if __name__ == "__main__":
    unittest.main()
