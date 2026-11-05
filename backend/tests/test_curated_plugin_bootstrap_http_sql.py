"""Opt-in absent-ledger initialization through real HTTP, authority and SQL.

Uses Core's explicit immutable deployment carrier; no synthetic user seed.
The runtime activation is fixture-only and does not qualify actual job dispatch.
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
from app.plugins.deployment import CURATED_DEPLOYMENT_BASELINE_V1
from app.plugins.policy import PluginPolicyChange, UI_NAMESPACE
from app.plugins.repository import PluginPolicyRepository
from app.plugins.router import router
from app.plugins.service import PluginPolicyService
from app.runtime.feature_config import TaskLimits
from app.settings import get_settings
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Core's explicit manifested database lane is required")
class PluginBootstrapHTTPSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_absent_account_first_edits_replay_and_withdrawal(self):
        fixture = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        tasks = TaskRepository(fixture.pool, TaskLimits(lease_seconds=300))
        contexts = ContextRepository(tasks)
        repository = PluginPolicyRepository(fixture.pool)
        baseline = CURATED_DEPLOYMENT_BASELINE_V1
        authority = PluginAuthority(repository, contexts, tasks, deployment_baseline=baseline)
        service = PluginPolicyService(repository, authority=authority, changes_enabled=True)

        async def client(owner):
            app = FastAPI()
            app.include_router(router)
            app.state.runtime_services = SimpleNamespace(plugin_policy_service=service)
            app.dependency_overrides[get_settings] = lambda: SimpleNamespace(
                JARVIS_STATIC_BEARER_TOKEN="synthetic-plugin-bootstrap", JARVIS_OWNER_ID=owner)

            @app.exception_handler(JarvisHTTPError)
            async def errors(request, error):
                return JSONResponse({"error": {"code": error.code, "message": error.message}},
                                    status_code=error.status_code)

            return await self.enterAsyncContext(httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://fixture.invalid"))

        http, other = await client(fixture.owner), await client(fixture.other)
        headers = {"Authorization": "Bearer synthetic-plugin-bootstrap"}

        async def counts():
            async with fixture.pool.acquire() as connection:
                return tuple([await connection.fetchval("select count(*) from " + table + " where owner_id=$1", fixture.owner)
                    for table in ("jarvis_context_events", "jarvis_context_payloads",
                                  "jarvis_context_ingress_receipts", "jarvis_capability_preferences")])

        def disable(plugin, revision=0):
            return PluginPolicyChange(mutation_id=uuid4(), expected_revision=revision,
                enabled_changes=[dict(plugin_id=plugin, enabled=False)])

        async def put(command):
            return await http.put("/plugins/policy", headers=headers, json=command.model_dump(mode="json"))

        before = await counts()
        # source_fixture creates one knowledge area. Migration 0033 retains
        # its metadata payload; that is neither user ingress nor plugin policy.
        self.assertEqual(before, (0, 1, 0, 0))
        async with fixture.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("""select count(*)
                from jarvis_context_payloads p join jarvis_knowledge_area_versions h
                  on h.owner_id=p.owner_id and h.metadata_payload_id=p.id
                where p.owner_id=$1 and h.area_id=$2 and h.revision=1 and h.origin='created'""",
                fixture.owner, fixture.area), 1)
        response = await http.get("/plugins", headers=headers)
        self.assertEqual(response.status_code, 200)
        catalog = response.json()
        self.assertEqual(catalog["policy_revision"], 0)
        self.assertTrue(catalog["policy_available"])
        self.assertTrue(all(x["desired_enabled"] and x["selected_default"] and x["can_change"]
                            for x in catalog["items"]))
        self.assertTrue(all(x["installed"] == "unknown" and x["readiness"] == "unknown"
                            for x in catalog["items"]))
        self.assertEqual({x["plugin_id"]: x["configuration_id"] for x in catalog["items"]},
                         {x.plugin_id: str(x.configuration_id) for x in baseline.policy.entries})
        self.assertEqual(await counts(), before, "Reading deployment defaults creates no user event or policy row")

        first = [disable("video_editor"), disable("browser")]
        self.assertEqual((await http.put("/plugins/policy", json=first[0].model_dump(mode="json"))).status_code, 401)
        self.assertEqual(await counts(), before)
        bindings = {plugin: await authority.select_new_work(owner_id=fixture.owner, plugin_id=plugin)
                    for plugin in ("video_editor", "browser")}
        self.assertTrue(all(binding.policy_revision == 0 and binding.deployment_baseline == baseline.identity
                            for binding in bindings.values()))
        self.assertEqual(await counts(), before, "Baseline selection is not a fabricated preference mutation")

        replies = await asyncio.gather(*(put(command) for command in first))
        self.assertEqual(sorted(x.status_code for x in replies), [200, 409])
        winner_index = next(i for i, response in enumerate(replies) if response.status_code == 200)
        winner, loser = first[winner_index], first[1 - winner_index]
        receipt = replies[winner_index].json()
        self.assertEqual((receipt["applied_revision"], receipt["request_replayed"]), (1, False))
        self.assertEqual(receipt["mutation_id"], str(winner.mutation_id))
        self.assertEqual(replies[1 - winner_index].json()["error"]["code"], "plugin_policy_revision_conflict")
        self.assertEqual((await counts())[0], 1, "Losing first-write ingress must roll back")
        self.assertEqual((await counts())[3], 1)
        cards = {x["plugin_id"]: x for x in receipt["catalog"]["items"]}
        disabled_plugin = winner.enabled_changes[0].plugin_id
        untouched_plugin = loser.enabled_changes[0].plugin_id
        self.assertFalse(cards[disabled_plugin]["desired_enabled"])
        self.assertTrue(cards[disabled_plugin]["selected_default"])
        self.assertTrue(cards[untouched_plugin]["desired_enabled"])

        async with tasks.transaction(fixture.owner) as connection:
            with self.assertRaises(JarvisHTTPError) as denied:
                await authority.require_new_work(connection, owner_id=fixture.owner,
                                                 binding=bindings[disabled_plugin])
            self.assertEqual(denied.exception.code, "plugin_disabled")
            await authority.require_new_work(connection, owner_id=fixture.owner,
                                             binding=bindings[untouched_plugin])

        before = await counts()
        retry = await put(winner)
        self.assertEqual(retry.status_code, 200)
        self.assertEqual((retry.json()["applied_revision"], retry.json()["request_replayed"]), (1, True))
        self.assertEqual(await counts(), before)
        self.assertEqual((await put(loser)).status_code, 409)
        self.assertEqual(await counts(), before)
        self.assertEqual((await other.get("/plugins", headers=headers)).json()["policy_revision"], 0,
                         "An untouched account retains only its deployment baseline")

        later = await put(disable("files_data", revision=1))
        self.assertEqual(later.status_code, 200)
        before = await counts()
        replay = (await put(winner)).json()
        self.assertEqual((replay["applied_revision"], replay["request_replayed"],
                          replay["catalog"]["policy_revision"]), (1, True, 2))
        self.assertEqual(await counts(), before)

        async with tasks.transaction(fixture.owner) as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_sessions where owner_id=$1", fixture.owner), 0)
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=(select payload_id from jarvis_context_events
                    where owner_id=$1 and source_namespace=$2 and source_event_id=$3)""",
                fixture.owner, UI_NAMESPACE, str(winner.mutation_id))
        withdrawn = (await http.get("/plugins", headers=headers)).json()
        self.assertFalse(withdrawn["policy_available"])
        self.assertIsNone(withdrawn["policy_revision"])
        self.assertTrue(all(x["desired_enabled"] is None and not x["can_change"] for x in withdrawn["items"]))
        before = await counts()
        rejected = await put(disable(untouched_plugin, revision=2))
        self.assertEqual(rejected.status_code, 409)
        self.assertEqual(rejected.json()["error"]["code"], "plugin_policy_unavailable")
        self.assertEqual(await counts(), before)
        async with tasks.transaction(fixture.owner) as connection:
            with self.assertRaises(JarvisHTTPError) as denied:
                await authority.require_new_work(connection, owner_id=fixture.owner,
                                                 binding=bindings[untouched_plugin])
            self.assertEqual(denied.exception.code, "plugin_policy_unavailable")


if __name__ == "__main__":
    unittest.main()
