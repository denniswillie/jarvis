"""Viewer composition is independent of task execution and recording capture."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.runtime.sandbox_viewing import initialize_sandbox_views
from app.tasks.repository import TaskRepository


class ViewerRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pool = Mock()
        self.resolve = AsyncMock(return_value="existing-content")
        self.services = SimpleNamespace(
            settings=SimpleNamespace(JARVIS_OWNER_ID="viewer-owner"),
            db_pool=self.pool,
            background_task_service=None,
            workspace_presentations=SimpleNamespace(resolver=self.resolve),
            recording_ingress=object(),
        )
        self.provider = Mock()
        self.secrets = Mock()

    def initialize(self, **kwargs):
        initialize_sandbox_views(
            self.services, provider=self.provider, secrets=self.secrets, **kwargs)

    async def test_disabled_tasks_do_not_disable_read_only_viewing(self):
        recording = self.services.recording_ingress
        self.initialize()
        views = self.services.sandbox_views
        self.assertIsInstance(views.access.tasks, TaskRepository)
        self.assertIs(views.access.tasks.pool, self.pool)
        self.assertIsNone(self.services.background_task_service)
        self.assertIs(self.services.recording_ingress, recording)
        self.assertFalse(self.pool.mock_calls)
        self.assertFalse(self.provider.mock_calls)
        await views.close()

    async def test_existing_task_repository_is_reused(self):
        repository = object()
        self.services.background_task_service = SimpleNamespace(repository=repository)
        self.initialize()
        self.assertIs(self.services.sandbox_views.access.tasks, repository)
        await self.services.sandbox_views.close()

    async def test_explicit_repository_takes_precedence(self):
        repository = object()
        self.services.background_task_service = SimpleNamespace(repository=object())
        self.initialize(repository=repository)
        self.assertIs(self.services.sandbox_views.access.tasks, repository)
        await self.services.sandbox_views.close()

    async def test_existing_presentations_remain_with_their_owner(self):
        self.initialize()
        ctx, artifact = object(), uuid4()
        for kind in ("note", "file"):
            result = await self.services.workspace_presentations.resolver(ctx, kind, artifact, "version")
            self.assertEqual(result, "existing-content")
            self.resolve.assert_awaited_with(ctx, kind, artifact, "version")
        await self.services.sandbox_views.close()

    async def test_sandbox_presentation_uses_exact_viewer_resource(self):
        self.initialize()
        ctx, artifact = object(), uuid4()
        views = self.services.sandbox_views
        views.resolve_presentation = AsyncMock(return_value="sandbox-generation")
        result = await self.services.workspace_presentations.resolver(ctx, "sandbox", artifact, "generation")
        self.assertEqual(result, "sandbox-generation")
        views.resolve_presentation.assert_awaited_once_with(ctx, artifact, "generation")
        self.resolve.assert_not_awaited()
        await views.close()

    def test_no_opt_in_keeps_existing_resolver(self):
        with patch("app.runtime.sandbox_viewing.SandboxViewSecrets.configured", return_value=None):
            initialize_sandbox_views(self.services, provider=self.provider)
        self.assertIsNone(self.services.sandbox_views)
        self.assertIs(self.services.workspace_presentations.resolver, self.resolve)
        self.assertFalse(self.pool.mock_calls)

    def test_missing_database_keeps_viewer_unavailable(self):
        self.services.db_pool = None
        self.initialize()
        self.assertIsNone(self.services.sandbox_views)
        self.assertFalse(self.provider.mock_calls)

    def test_missing_presentations_keeps_viewer_unavailable(self):
        self.services.workspace_presentations = None
        self.initialize()
        self.assertIsNone(self.services.sandbox_views)
        self.assertFalse(self.pool.mock_calls)
