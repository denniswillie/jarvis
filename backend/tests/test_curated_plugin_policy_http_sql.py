"""Opt-in real SQL + authenticated HTTP + production policy authority composition.

The initial policy is explicitly seeded inside the disposable fixture. This
does not qualify production bootstrap, mounted runtime or managed-job fences.
"""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import httpx

from app.errors import JarvisHTTPError
from app.plugins.authority import PluginAuthority
from app.plugins.models import PolicySnapshot
from app.plugins.policy import CATEGORIES, UI_NAMESPACE, PluginPolicyChange, changed_policy
from app.plugins.repository import PluginPolicyRepository
from app.plugins.router import router
from app.plugins.service import PluginPolicyService
from app.runtime.feature_config import TaskLimits
from app.settings import get_settings
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Core's explicit manifested database lane is required")
class PluginPolicyHTTPSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_authenticated_http_mutation_replay_cas_and_withdrawal(self):
        fixture = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        tasks = TaskRepository(fixture.pool, TaskLimits(lease_seconds=300))
        contexts = ContextRepository(tasks)
        repository = PluginPolicyRepository(fixture.pool)
        authority = PluginAuthority(repository, contexts, tasks)
        # Test-only activation, never the production runtime configuration.
        service = PluginPolicyService(repository, authority=authority, changes_enabled=True)

        async def client(owner):
            app = FastAPI()
            app.include_router(router)
            app.state.runtime_services = SimpleNamespace(plugin_policy_service=service)
            app.dependency_overrides[get_settings] = lambda: SimpleNamespace(
                JARVIS_STATIC_BEARER_TOKEN="synthetic-plugin-http", JARVIS_OWNER_ID=owner)

            @app.exception_handler(JarvisHTTPError)
            async def errors(request, error):
                return JSONResponse({"error": {"code": error.code, "message": error.message}},
                                    status_code=error.status_code)

            return await self.enterAsyncContext(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://fixture.invalid"))

        http = await client(fixture.owner)
        other = await client(fixture.other)
        headers = {"Authorization": "Bearer synthetic-plugin-http"}

        async def counts():
            async with fixture.pool.acquire() as connection:
                return tuple([await connection.fetchval("select count(*) from " + table + " where owner_id=$1", fixture.owner)
                    for table in ("jarvis_context_events", "jarvis_context_payloads",
                                  "jarvis_context_ingress_receipts", "jarvis_capability_preferences")])

        def change(revision, plugin, enabled):
            return PluginPolicyChange(mutation_id=uuid4(), expected_revision=revision,
                enabled_changes=[dict(plugin_id=plugin, enabled=enabled)])

        async def put(command):
            return await http.put("/plugins/policy", headers=headers, json=command.model_dump(mode="json"))

        response = await http.get("/plugins", headers=headers)
        self.assertFalse(response.json()["policy_available"])
        self.assertFalse(any(x["can_change"] for x in response.json()["items"]))
        before = await counts()
        self.assertEqual((await http.put("/plugins/policy", json=change(0, "browser", True).model_dump(mode="json"))).status_code, 401)
        self.assertEqual(await counts(), before)

        # Explicit synthetic setup through the existing canonical source/ledger
        # seams; no absent-policy fallback is passed to production authority.
        seed = PluginPolicyChange(mutation_id=uuid4(), expected_revision=0,
            enabled_changes=[dict(plugin_id=plugin, enabled=True) for plugin in CATEGORIES],
            default_changes=[dict(category="video_editing", plugin_id="video_editor")])
        async with tasks.transaction(fixture.owner) as connection:
            event = await contexts.append_in_transaction(connection, fixture.owner, ContextIngress(
                session_id=None, source_namespace=UI_NAMESPACE,
                source_event_id=str(seed.mutation_id), receipt_id=str(seed.mutation_id),
                expected_revision=0, kind="user_request", trust="user_input", finality="final",
                payload=seed.ingress_payload()))
            baseline = PolicySnapshot(revision=0, entries=())
            previous = await repository.read(connection, fixture.owner, baseline=baseline, lock=True)
            await repository.append(connection, fixture.owner, seed, event, previous,
                                    changed_policy(baseline, seed, event.id))

        catalog = (await http.get("/plugins", headers=headers)).json()
        self.assertEqual(catalog["policy_revision"], 1)
        self.assertTrue(all(x["can_change"] for x in catalog["items"]))
        original_config = catalog["items"][0]["configuration_id"]
        disable = change(1, "video_editor", False)
        response = await put(disable)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("no-store", response.headers["cache-control"])
        receipt = response.json()
        self.assertEqual((receipt["mutation_id"], receipt["applied_revision"], receipt["request_replayed"]),
                         (str(disable.mutation_id), 2, False))
        video = receipt["catalog"]["items"][0]
        self.assertFalse(video["desired_enabled"])
        self.assertTrue(video["selected_default"])
        self.assertNotEqual(video["configuration_id"], original_config)
        self.assertEqual(video["installed"], "unknown")
        self.assertFalse((await other.get("/plugins", headers=headers)).json()["policy_available"])

        self.assertEqual((await put(change(2, "browser", False))).status_code, 200)
        before = await counts()
        replay = await put(disable)
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual((replay.json()["applied_revision"], replay.json()["request_replayed"],
                          replay.json()["catalog"]["policy_revision"]), (2, True, 3))
        self.assertEqual(await counts(), before)
        conflict = await put(disable.model_copy(update={"expected_revision": 3}))
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.json()["error"]["code"], "plugin_policy_mutation_conflict")
        self.assertEqual(await counts(), before)

        simultaneous = await asyncio.gather(put(change(3, "video_editor", True)), put(change(3, "files_data", False)))
        self.assertEqual(sorted(x.status_code for x in simultaneous), [200, 409])
        loser = next(x for x in simultaneous if x.status_code == 409)
        self.assertEqual(loser.json()["error"]["code"], "plugin_policy_revision_conflict")
        self.assertEqual((await counts())[0], 4, "Rejected CAS leaves no orphan UI event")

        async with tasks.transaction(fixture.owner) as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_sessions where owner_id=$1", fixture.owner), 0)
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=(select payload_id from jarvis_context_events
                    where owner_id=$1 and source_namespace=$2 and source_event_id=$3)""",
                fixture.owner, UI_NAMESPACE, str(disable.mutation_id))
        withdrawn = (await http.get("/plugins", headers=headers)).json()
        self.assertFalse(withdrawn["policy_available"])
        self.assertFalse(any(x["can_change"] for x in withdrawn["items"]))
        denied_replay = await put(disable)
        self.assertEqual(denied_replay.status_code, 409, "Removed source must not impersonate account auth loss")
        self.assertEqual(denied_replay.json()["error"]["code"], "plugin_policy_source_unavailable")
        before = await counts()
        rejected = await put(change(4, "browser", True))
        self.assertEqual(rejected.status_code, 409)
        self.assertEqual(rejected.json()["error"]["code"], "plugin_policy_unavailable")
        self.assertEqual(await counts(), before, "Unavailable policy rolls back provisional ingress")


if __name__ == "__main__":
    unittest.main()
