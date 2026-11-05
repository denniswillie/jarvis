"""Actual HTTP callers over manifested source storage; no Hindsight or legacy fallback."""
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.models import SourceError
from app.memory.router import router
from app.runtime.feature_config import MemoryAuthority
from evals.second_brain.source_fixture import source_fixture


def application(services, owner):
    app = FastAPI()
    app.include_router(router)
    app.state.runtime_services = services
    app.dependency_overrides[require_static_bearer] = lambda: JarvisPrincipal(owner_id=owner)

    @app.exception_handler(JarvisHTTPError)
    async def error_handler(request: Request, error: JarvisHTTPError):
        return JSONResponse(status_code=error.status_code, content={
            "error": {"code": error.code, "message": error.message, "details": error.details}})

    return app


class SelectedMemoryUnavailableHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_unavailable_selected_memory_never_calls_legacy(self):
        legacy = SimpleNamespace(list_recent=AsyncMock(side_effect=AssertionError("legacy")),
            explain=AsyncMock(side_effect=AssertionError("legacy")))
        working = SimpleNamespace(get_session_working_memory=AsyncMock(return_value=None),
            get_owner_working_memory=AsyncMock(return_value=None))
        services = SimpleNamespace(memory_authority=MemoryAuthority.HINDSIGHT, knowledge_runtime=None,
            settings=SimpleNamespace(JARVIS_OWNER_ID="owner"), memory_service=legacy,
            working_memory_service=working, memory_projection_service=Mock())
        app = application(services, "owner")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            for path in ("/memory/recent", "/memory/" + str(uuid4()), "/memory/" + str(uuid4()) + "/evidence"):
                response = await client.get(path)
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["error"]["code"], "selected_memory_unavailable")
            response = await client.get("/memory/working-state", params={"session_id": str(uuid4())})
            self.assertEqual(response.json(), {"session": None, "owner": None})
            self.assertEqual(response.headers["cache-control"], "no-store")
            profile = (await client.get("/memory/profile")).json()
            self.assertIsNone(profile["profile"])
            self.assertEqual(profile["status"], "not_materialized")
            self.assertIn("does not mean nothing", profile["meaning"])
        legacy.list_recent.assert_not_awaited()
        legacy.explain.assert_not_awaited()
        services.memory_projection_service.get_profile.assert_not_called()

    async def test_invalid_authority_is_not_a_legacy_route(self):
        services = SimpleNamespace(memory_authority="invalid", memory_service=Mock())
        app = application(services, "owner")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            result = await client.get("/memory/recent")
            self.assertEqual(result.status_code, 503)
            self.assertEqual(result.json()["error"]["code"], "memory_authority_unavailable")


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested source fixtures required")
class SourceMemoryHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id())
        self.f = await self.context.__aenter__()
        self.addAsyncCleanup(self.context.__aexit__, None, None, None)
        self.provider = SimpleNamespace(sources=self.f.sources, recall=AsyncMock(side_effect=AssertionError("provider I/O")))
        self.reader = SourceMemoryReadBackend(self.f.sources, self.provider,
            owner_id=self.f.owner, area_id=self.f.area, page_characters=1000)
        runtime = SimpleNamespace(sources=self.f.sources, configuration=SimpleNamespace(area_id=self.f.area),
            memory_read_backend_for=lambda owner: self.reader, context_for=lambda owner: None)
        self.services = SimpleNamespace(memory_authority=MemoryAuthority.HINDSIGHT,
            knowledge_runtime=runtime, settings=SimpleNamespace(JARVIS_OWNER_ID=self.f.owner),
            memory_service=Mock(), memory_projection_service=Mock(),
            working_memory_service=SimpleNamespace(get_owner_working_memory=AsyncMock(return_value=None)))
        self.app = application(self.services, self.f.owner)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://fixture")
        self.addAsyncCleanup(self.client.aclose)

    async def test_recent_originals_have_honest_null_confidence_and_unknown_time(self):
        source = await self.f.publish("Synthetic reported action, not proof", source_time=None, trust="user_input")
        response = await self.client.get("/memory/recent")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        item = body["items"][0]
        self.assertEqual(item["id"], str(source.source_id))
        self.assertIsNone(item["confidence"])
        self.assertIsNone(item["importance"])
        self.assertIsNone(item["occurred_at"])
        self.assertEqual(item["source_evidence"]["trust"], "user_input")
        self.assertEqual(item["source_evidence"]["ingestion_status"], "pending")
        self.assertIn("not proof", item["content"])
        self.assertNotIn("bank_name", response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["vary"], "Authorization")
        self.assertFalse(body["provider_contacted"])
        self.provider.recall.assert_not_awaited()

    async def test_cursor_paginates_same_time_sources_and_is_bound_to_filters_and_area(self):
        originals = [await self.f.publish(str(n), source_key=str(n)) for n in range(5)]
        expected = sorted((str(s.source_id) for s in originals), reverse=True)
        found, cursor = [], None
        while True:
            params = {"limit": 2} | ({"cursor": cursor} if cursor else {})
            response = await self.client.get("/memory/recent", params=params)
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            found.extend(item["id"] for item in body["items"])
            cursor = body["next_cursor"]
            if not cursor:
                break
        self.assertEqual(found, expected)
        first = (await self.client.get("/memory/recent", params={"limit": 1})).json()
        bad = await self.client.get("/memory/recent", params={
            "cursor": first["next_cursor"], "source_types": "different"})
        self.assertEqual(bad.status_code, 400)
        legacy_cursor = "eyJ0aW1lIjoiMjAyNi0wOS0xNFQxMjowMDowMCswMDowMCIsImlkIjoiMDAwMDAwMDAtMDAwMC0wMDAwLTAwMDAtMDAwMDAwMDAwMDAwIn0"
        self.assertEqual((await self.client.get("/memory/recent", params={"cursor": legacy_cursor})).status_code, 400)

    async def test_source_type_and_original_time_filters_do_not_invent_legacy_categories(self):
        now = datetime(2026, 9, 14, 12, tzinfo=UTC)
        match = await self.f.publish("A", source_key="a", source_type="experience", source_time=now)
        await self.f.publish("B", source_key="b", source_type="note", source_time=now)
        await self.f.publish("Unknown", source_key="unknown", source_time=None)
        await self.f.publish("Boundary", source_key="end", source_time=now + timedelta(hours=1))
        result = await self.client.get("/memory/recent", params={"source_types": "experience",
            "start_time": now.isoformat(), "end_time": (now + timedelta(hours=1)).isoformat()})
        self.assertEqual([x["id"] for x in result.json()["items"]], [str(match.source_id)])
        self.assertEqual((await self.client.get("/memory/recent", params={"memory_types": "fact"})).status_code, 400)
        self.assertEqual((await self.client.get("/memory/recent", params={"start_time": "2026-09-14T12:00:00"})).status_code, 400)
        self.assertEqual((await self.client.get("/memory/recent", params={"start_time": now.isoformat(), "end_time": now.isoformat()})).status_code, 400)
        self.assertEqual((await self.client.get("/memory/recent", params={"session_id": str(uuid4())})).json()["items"], [])

    async def test_detail_pages_return_exact_original_and_require_revision_digest(self):
        content = {"text": "Synthetic original " * 250, "reported": True}
        source = await self.f.publish_payload(content)
        url = "/memory/" + str(source.source_id)
        result = (await self.client.get(url)).json()
        self.assertIsNone(result["memory"]["confidence"])
        self.assertIsNone(result["evidence"][0]["retained_image_available"])
        self.assertIn("not independent verification", result["uncertainty"])
        original = result["memory"]["content"]
        while result["content_page"]["next_offset"] is not None:
            response = await self.client.get(url, params={
                "offset": result["content_page"]["next_offset"], "revision": source.revision,
                "expected_digest": source.content_digest})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            original += result["memory"]["content"]
        self.assertEqual(json.loads(original), content)
        self.assertEqual((await self.client.get(url, params={"offset": 1000})).status_code, 400)
        self.assertEqual((await self.client.get(url, params={"offset": 1000, "revision": 1, "expected_digest": "0" * 64})).status_code, 409)
        self.provider.recall.assert_not_awaited()

    async def test_correction_current_and_historical_reads_and_deletion(self):
        first = await self.f.publish("Old statement")
        second = await self.f.publish("Corrected statement", expected_revision=1)
        url = "/memory/" + str(first.source_id)
        current = (await self.client.get(url)).json()
        self.assertIn("Corrected", current["memory"]["content"])
        self.assertEqual(current["memory"]["source_evidence"]["revision"], 2)
        history = (await self.client.get(url, params={"revision": 1})).json()
        self.assertIn("Old statement", history["memory"]["content"])
        self.assertFalse(history["memory"]["source_evidence"]["is_current"])
        await self.f.sources.delete_source(self.f.owner, self.f.area, second.source_id, expected_revision=2)
        self.assertEqual((await self.client.get("/memory/recent")).json()["items"], [])
        for path in (url, url + "?revision=1", url + "/evidence"):
            response = await self.client.get(path)
            self.assertIn(response.status_code, (404, 409))
            self.assertNotIn("Old statement", response.text)
            self.assertNotIn("Corrected statement", response.text)

    async def test_archived_scope_and_wrong_owner_never_fall_back(self):
        source = await self.f.publish()
        self.app.dependency_overrides[require_static_bearer] = lambda: JarvisPrincipal(owner_id=self.f.other)
        for path in ("/memory/recent", "/memory/" + str(source.source_id), "/memory/profile", "/memory/working-state"):
            self.assertEqual((await self.client.get(path)).status_code, 403)
        self.app.dependency_overrides[require_static_bearer] = lambda: JarvisPrincipal(owner_id=self.f.owner)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_knowledge_areas set status='archived',revision=revision+1 where id=$1", self.f.area)
        self.assertEqual((await self.client.get("/memory/recent")).status_code, 404)
        self.assertEqual((await self.client.get("/memory/" + str(source.source_id))).status_code, 404)
        self.services.memory_service.list_recent.assert_not_called()

    async def test_skipped_unavailable_versions_do_not_end_pagination_early(self):
        values = [await self.f.publish(str(n), source_key=str(n)) for n in range(7)]
        ordered = sorted(values, key=lambda s: str(s.source_id), reverse=True)
        hidden = {s.source_id for s in ordered[:3]}
        convert = self.f.sources._version
        def reject(row, **kwargs):
            if row["source_id"] in hidden:
                raise SourceError("knowledge_source_unavailable", "Synthetic lifecycle race")
            return convert(row, **kwargs)
        with patch.object(self.f.sources, "_version", side_effect=reject):
            body = (await self.client.get("/memory/recent", params={"limit": 2})).json()
            self.assertEqual([x["id"] for x in body["items"]], [str(s.source_id) for s in ordered[3:5]])
            self.assertEqual(body["unavailable_sources_skipped"], 3)
            tail = (await self.client.get("/memory/recent", params={"limit": 2, "cursor": body["next_cursor"]})).json()
            self.assertEqual([x["id"] for x in tail["items"]], [str(s.source_id) for s in ordered[5:]])
            self.assertIsNone(tail["next_cursor"])

    async def test_imported_session_filter_uses_normalized_authority_through_http(self):
        from app.knowledge.backfill import LegacyBackfill
        async with self.f.pool.acquire() as c:
            sessions = [await c.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
                        for _ in range(2)]
            for session in sessions:
                await c.execute("""insert into jarvis_messages(owner_id,session_id,role,modality,content)
                    values($1,$2,'user','audio','Synthetic imported HTTP source')""", self.f.owner, session)
        report = await LegacyBackfill(self.f.sources).run(self.f.owner, self.f.area)
        self.assertEqual(report["session_association_issues"], 0)
        response = await self.client.get("/memory/recent", params={
            "session_id": str(sessions[0]), "source_types": "legacy:jarvis_messages"})
        self.assertEqual(response.status_code, 200, response.text)
        items = response.json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["session_id"], str(sessions[0]))
        self.services.memory_service.list_recent.assert_not_called()
        self.provider.recall.assert_not_awaited()
