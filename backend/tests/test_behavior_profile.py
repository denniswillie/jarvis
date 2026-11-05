"""Offline profile/API/tool tests. The fake is not PostgreSQL race evidence."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import unittest

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import httpx
from pydantic import ValidationError

from app.assistant.models import BehaviorProfileRevisionConflict, BehaviorProfileUpdate
from app.assistant.profile_router import router
from app.assistant.profile_service import BehaviorProfileService
from app.assistant.repository import BehaviorProfileRepository
from app.assistant.tools import create_behavior_tools
from app.errors import JarvisHTTPError
from app.settings import get_settings


class ProfilePool:
    """Minimal SQL-shaped fixture with explicit commit/release visibility."""

    def __init__(self):
        self.rows = {}
        self.lock = asyncio.Lock()
        self.borrowers = 0
        self.in_transaction = False
        self.writes = 0
        self.fail_commit = False

    @asynccontextmanager
    async def acquire(self):
        self.borrowers += 1
        try:
            yield self
        finally:
            self.borrowers -= 1

    @asynccontextmanager
    async def transaction(self):
        async with self.lock:
            before = {key: dict(value) for key, value in self.rows.items()}
            self.in_transaction = True
            try:
                yield self
                if self.fail_commit:
                    raise RuntimeError("fixture commit failed")
            except BaseException:
                self.rows = before
                raise
            finally:
                self.in_transaction = False

    async def fetchrow(self, sql, owner, *args):
        query = sql.strip().lower()
        if query.startswith("select"):
            row = self.rows.get(owner)
            return dict(row) if row else None
        if query.startswith("insert"):
            if owner in self.rows:
                return None
            row = dict(owner_id=owner, behavior_instructions=args[0], revision=1,
                       created_at=datetime(2026, 11, 2, tzinfo=UTC),
                       updated_at=datetime(2026, 11, 2, tzinfo=UTC))
        elif query.startswith("update"):
            old = self.rows.get(owner)
            if old is None or old["revision"] != args[1]:
                return None
            row = dict(old, behavior_instructions=args[0], revision=old["revision"] + 1,
                       updated_at=old["updated_at"] + timedelta(seconds=1))
        else:
            raise AssertionError("Unexpected fixture SQL")
        self.rows[owner] = row
        self.writes += 1
        return dict(row)

    async def fetchval(self, sql, owner):
        row = self.rows.get(owner)
        return row["revision"] if row else None


class BehaviorProfileTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pool = ProfilePool()
        self.events = []

        def committed(event):
            self.assertFalse(self.pool.in_transaction)
            self.assertEqual(self.pool.borrowers, 0)
            self.events.append(event.as_payload())

        self.repo = BehaviorProfileRepository(self.pool, on_committed=committed)
        self.service = BehaviorProfileService(self.repo)
        self.owner = "synthetic-profile-owner"

    async def write(self, text, revision=0, owner=None):
        return await self.service.update(owner_id=owner or self.owner,
            update=BehaviorProfileUpdate(behavior_instructions=text, expected_revision=revision))

    async def test_get_missing_does_not_create_or_publish(self):
        profile = await self.service.get(owner_id=self.owner)
        self.assertEqual(profile, dict(owner_id=self.owner, behavior_instructions="", revision=0,
                                       created_at=None, updated_at=None))
        self.assertEqual(self.pool.writes, 0)
        self.assertEqual(self.events, [])

    async def test_revisions_clear_and_owner_isolation(self):
        one = await self.write("  Keep answers short  ")
        two = await self.write("", 1)
        self.assertEqual(one["behavior_instructions"], "Keep answers short")
        self.assertEqual(two["behavior_instructions"], "")
        self.assertEqual((one["revision"], two["revision"]), (1, 2))
        self.assertEqual(one["created_at"], two["created_at"])
        self.assertNotEqual(one["updated_at"], two["updated_at"])
        other = await self.write("Other owner", owner="synthetic-other")
        self.assertEqual(other["revision"], 1)
        self.assertEqual((await self.service.get(owner_id=self.owner))["revision"], 2)
        event = self.events[1]
        self.assertEqual(event, dict(type="behavior_profile.committed",
            event_id=f"behavior-profile:{self.owner}:2", owner_id=self.owner, revision=2,
            behavior_instructions="", committed_at=two["updated_at"]))

    async def test_stale_and_future_updates_never_write_or_emit(self):
        await self.write("Current")
        for revision in (0, 2):
            with self.subTest(revision=revision), self.assertRaises(BehaviorProfileRevisionConflict) as error:
                await self.write("Lost update", revision)
            self.assertEqual(error.exception.current_revision, 1)
        with self.assertRaises(BehaviorProfileRevisionConflict) as error:
            await self.write("Missing", 1, "missing-owner")
        self.assertEqual(error.exception.current_revision, 0)
        self.assertEqual(self.pool.writes, 1)
        self.assertEqual(len(self.events), 1)

    async def test_failed_commit_does_not_emit_and_callback_failure_does_not_undo_commit(self):
        self.pool.fail_commit = True
        with self.assertRaisesRegex(RuntimeError, "fixture commit failed"):
            await self.write("Not committed")
        self.assertEqual(self.events, [])
        self.assertIsNone(await self.repo.get(owner_id=self.owner))
        self.pool.fail_commit = False

        def unavailable(event):
            raise RuntimeError("fixture notification unavailable")

        repository = BehaviorProfileRepository(self.pool, on_committed=unavailable)
        with self.assertLogs("app.assistant.repository", level="WARNING"):
            result = await repository.set(owner_id=self.owner, behavior_instructions="Saved",
                                          expected_revision=0)
        self.assertEqual(result.revision, 1)
        self.assertEqual((await repository.get(owner_id=self.owner)).behavior_instructions, "Saved")

    async def test_voice_and_app_share_cas_and_no_consumption_claim(self):
        getter, setter = create_behavior_tools(repository=self.repo, owner_id=self.owner)
        initial = await getter.handler({})
        await self.write("App edit")
        stale = await setter.handler(dict(behavior_instructions="Voice edit",
            expected_revision=initial["behavior_profile"]["revision"]))
        self.assertEqual(stale["error"]["code"], "behavior_profile_revision_conflict")
        current = await getter.handler({})
        saved = await setter.handler(dict(behavior_instructions="Voice edit",
            expected_revision=current["behavior_profile"]["revision"]))
        self.assertTrue(saved["effect_committed"])
        self.assertNotIn("applies_to", saved)
        self.assertEqual((await self.service.get(owner_id=self.owner))["behavior_instructions"], "Voice edit")
        self.assertEqual(len(self.events), 2)
        for args in ({"behavior_instructions": "No revision"},
                     {"behavior_instructions": "Wrong type", "expected_revision": True},
                     {"behavior_instructions": "Foreign", "expected_revision": 2, "owner_id": "other"}):
            self.assertEqual((await setter.handler(args))["error"]["code"], "invalid_tool_arguments")
        self.assertIn("expected_revision", setter.input_schema["required"])

    def test_validation_strict_types_limits_and_clear(self):
        for revision in (True, -1, "0", 1.0, None):
            with self.subTest(revision=revision), self.assertRaises(ValidationError):
                BehaviorProfileUpdate(behavior_instructions="x", expected_revision=revision)
        for value in (None, 2, "x" * 12001):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValidationError):
                BehaviorProfileUpdate(behavior_instructions=value, expected_revision=0)
        self.assertEqual(BehaviorProfileUpdate(behavior_instructions="x" * 12000,
                                              expected_revision=0).expected_revision, 0)
        self.assertEqual(BehaviorProfileUpdate(behavior_instructions=" \n ",
                                              expected_revision=0).behavior_instructions, "")


class BehaviorProfileHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pool = ProfilePool()
        self.repo = BehaviorProfileRepository(self.pool)
        self.app = FastAPI()
        self.app.state.runtime_services = SimpleNamespace(behavior_profile_service=BehaviorProfileService(self.repo))
        self.app.dependency_overrides[get_settings] = lambda: SimpleNamespace(
            JARVIS_OWNER_ID="fixture-profile-owner", JARVIS_STATIC_BEARER_TOKEN="fixture-profile-bearer")

        @self.app.exception_handler(JarvisHTTPError)
        async def error_handler(request, error):
            return JSONResponse({"error": {"code": error.code, "message": error.message,
                                           "details": error.details}}, status_code=error.status_code)

        self.app.include_router(router)
        self.client = await self.enterAsyncContext(httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://fixture"))
        self.auth = {"Authorization": "Bearer fixture-profile-bearer"}

    async def test_authenticated_read_and_conditional_write(self):
        url = "/assistant/profile"
        self.assertEqual((await self.client.get(url)).status_code, 401)
        self.assertEqual((await self.client.get(url, headers={"Authorization": "Bearer wrong"})).status_code, 403)
        initial = await self.client.get(url, headers=self.auth)
        self.assertEqual(initial.json()["revision"], 0)
        self.assertEqual(self.pool.writes, 0)
        self.assertIn("no-store", initial.headers["Cache-Control"])
        saved = await self.client.put(url, headers=self.auth, json={"behavior_instructions": "Brief",
                                                                  "expected_revision": 0})
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["revision"], 1)
        stale = await self.client.put(url, headers=self.auth, json={"behavior_instructions": "Stale",
                                                                  "expected_revision": 0})
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["error"]["details"], {"expected_revision": 0, "current_revision": 1})
        self.assertEqual((await self.client.get(url + "?owner_id=other", headers=self.auth)).json(), saved.json())
        for payload in ({"behavior_instructions": "x"},
                        {"behavior_instructions": "x", "expected_revision": True},
                        {"behavior_instructions": "x" * 12001, "expected_revision": 1},
                        {"behavior_instructions": "x", "expected_revision": 1, "owner_id": "other"}):
            response = await self.client.put(url, headers=self.auth, json=payload)
            self.assertEqual(response.status_code, 422)
        self.assertEqual(self.pool.writes, 1)
        clear = await self.client.put(url, headers=self.auth, json={"behavior_instructions": "",
                                                                  "expected_revision": 1})
        self.assertEqual(clear.json()["behavior_instructions"], "")
        self.assertEqual(clear.json()["revision"], 2)

    async def test_unwired_service_fails_closed(self):
        self.app.state.runtime_services = SimpleNamespace()
        response = await self.client.get("/assistant/profile", headers=self.auth)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "behavior_profile_unavailable")


if __name__ == "__main__":
    unittest.main()
