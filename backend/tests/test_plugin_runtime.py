"""Actual app/lifespan composition with offline repository I/O doubles.

This does not start a database, provider, worker or voice session. Real ledger
semantics are covered separately by the bootstrap HTTP/SQL integration case.
"""
from contextlib import ExitStack, asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import httpx

from app.plugins.deployment import CURATED_DEPLOYMENT_BASELINE_V1
from app.plugins.runtime import initialize_plugin_runtime
from app.runtime.feature_config import TaskLimits
from app.settings import JarvisSettings, get_settings
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository


class PluginRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.connection = SimpleNamespace(fetchrow=AsyncMock(return_value=None))

        @asynccontextmanager
        async def acquire():
            yield self.connection

        self.pool = SimpleNamespace(acquire=Mock(side_effect=acquire), close=AsyncMock())
        self.tasks = TaskRepository(self.pool, TaskLimits(lease_seconds=300))
        self.contexts = ContextRepository(self.tasks)
        self.services = SimpleNamespace(db_pool=self.pool,
            background_task_context_service=SimpleNamespace(repository=self.contexts),
            knowledge_runtime=None, plugin_policy_service=None, plugin_authority=None)

    def test_reuses_existing_owners_without_io_or_write_activation(self):
        initialize_plugin_runtime(self.services)
        authority = self.services.plugin_authority
        service = self.services.plugin_policy_service
        self.assertIs(authority.contexts, self.contexts)
        self.assertIs(authority.tasks, self.tasks)
        self.assertIs(authority.repository.pool, self.pool)
        self.assertIs(authority.deployment_baseline, CURATED_DEPLOYMENT_BASELINE_V1)
        self.assertIs(service.authority, authority)
        self.assertIs(service.repository, authority.repository)
        self.assertFalse(service.changes_enabled)
        self.pool.acquire.assert_not_called()

    def test_knowledge_context_is_reused_when_task_context_is_absent(self):
        self.services.background_task_context_service = None
        self.services.knowledge_runtime = SimpleNamespace(context_repository=self.contexts)
        initialize_plugin_runtime(self.services)
        self.assertIs(self.services.plugin_authority.contexts, self.contexts)
        self.pool.acquire.assert_not_called()

    def test_missing_or_mismatched_owner_dependencies_clear_existing_service(self):
        for change in (
            dict(db_pool=None),
            dict(background_task_context_service=None),
            dict(background_task_context_service=SimpleNamespace(repository=object())),
            dict(db_pool=object()),
        ):
            with self.subTest(change=tuple(change)):
                services = SimpleNamespace(**(vars(self.services) | change))
                services.plugin_policy_service = object()
                services.plugin_authority = object()
                initialize_plugin_runtime(services)
                self.assertIsNone(services.plugin_policy_service)
                self.assertIsNone(services.plugin_authority)
        self.pool.acquire.assert_not_called()

    async def test_real_app_lifespan_mounts_authenticated_read_only_catalog(self):
        from app import main

        settings = JarvisSettings(_env_file=None, JARVIS_RESTORE_IN_PROGRESS=False,
            JARVIS_STATIC_BEARER_TOKEN="synthetic-plugin-runtime", JARVIS_OWNER_ID="runtime-owner",
            LOG_LEVEL="CRITICAL")
        manager = Mock(close_all=AsyncMock(), prepare_for_resource_close=AsyncMock())

        async def database(services):
            services.db_pool = self.pool

        async def tasks(services):
            services.background_task_context_service = SimpleNamespace(repository=self.contexts)

        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "get_settings", return_value=settings))
            stack.enter_context(patch.object(main, "RealtimeSessionManager", return_value=manager))
            stack.enter_context(patch.object(main, "_initialize_runtime_services", side_effect=database))
            stack.enter_context(patch.object(main, "initialize_task_runtime", side_effect=tasks))
            for name in ("initialize_knowledge_runtime", "initialize_recording_runtime",
                         "initialize_attention_runtime", "_initialize_reminder_services"):
                stack.enter_context(patch.object(main, name, new_callable=AsyncMock))
            stack.enter_context(patch.object(main, "configure_selected_memory"))
            stack.enter_context(patch.object(main, "initialize_workspace_notes"))
            stack.enter_context(patch("app.runtime.workspace_delivery.initialize_workspace_presentations"))
            app = main.create_app()
            app.dependency_overrides[get_settings] = lambda: settings
            async with main.lifespan(app):
                service = app.state.runtime_services.plugin_policy_service
                self.assertIsNotNone(service)
                self.assertIs(service.authority.contexts, self.contexts)
                self.pool.acquire.assert_not_called()
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                             base_url="http://fixture.invalid") as client:
                    self.assertEqual((await client.get("/plugins")).status_code, 401)
                    self.pool.acquire.assert_not_called()
                    headers = {"Authorization": "Bearer synthetic-plugin-runtime"}
                    response = await client.get("/plugins", headers=headers)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers["cache-control"], "private, no-store")
                    catalog = response.json()
                    self.assertTrue(catalog["policy_available"])
                    self.assertEqual(catalog["policy_revision"], 0)
                    self.assertEqual(len(catalog["items"]), 3)
                    self.assertTrue(all(item["desired_enabled"] and item["selected_default"]
                                        and not item["can_change"] for item in catalog["items"]))
                    self.assertTrue(all(item["installed"] == item["readiness"] == "unknown"
                                        for item in catalog["items"]))
                    self.assertEqual(self.connection.fetchrow.call_args.args[1], "runtime-owner")
                    read_count = self.pool.acquire.call_count
                    response = await client.put("/plugins/policy", headers=headers, json={
                        "mutation_id": str(UUID(int=1)), "expected_revision": 0,
                        "enabled_changes": [{"plugin_id": "browser", "enabled": False}]})
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(response.json()["error"]["code"], "plugin_policy_changes_unavailable")
                    self.assertEqual(self.pool.acquire.call_count, read_count)
                    self.connection.fetchrow.return_value = dict(revision=4,
                        policy_payload_id=UUID(int=2), availability="withdrawn", content=None)
                    response = await client.get("/plugins", headers=headers)
                    self.assertEqual(response.status_code, 200)
                    self.assertFalse(response.json()["policy_available"])
                    self.assertTrue(all(not item["can_change"] for item in response.json()["items"]))
                    app.state.runtime_services.plugin_policy_service = None
                    self.assertEqual((await client.get("/plugins", headers=headers)).status_code, 503)
                    self.assertEqual(self.pool.acquire.call_count, read_count + 1)
        self.pool.close.assert_awaited_once()
        manager.close_all.assert_awaited_once()
