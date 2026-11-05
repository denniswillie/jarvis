"""Composition contract: real services, no database/provider calls at construction."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.hermes_sessions.bindings import BindingError
from app.hermes_sessions.domain_scope import DomainInputScope
from app.hermes_sessions.source_domain import compose_source_domain
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.retrieval import SourceMemoryRetrieval
from app.runtime.feature_config import TaskLimits
from app.settings import JarvisSettings
from app.tasks.repository import TaskRepository


class SourceDomainTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pool = object()
        self.sources = SimpleNamespace(pool=self.pool)
        self.requests = TaskRepository(self.pool, TaskLimits())
        self.area = uuid4()
        self.provider = Mock()
        self.reader = SourceMemoryReadBackend(self.sources,
            SourceMemoryRetrieval(self.sources, self.provider), owner_id="owner", area_id=self.area)
        self.service = SimpleNamespace(bindings=SimpleNamespace(owner_id="owner", profile_home=Path("/fixture")))
        self.scope = DomainInputScope("owner", "master", "input", uuid4(),
                                     frozenset({"remember", "create_text_file"}))
        self.resolver = AsyncMock()
        self.storage = SimpleNamespace(bucket="fixture")

    def compose(self, **changes):
        return compose_source_domain(self.service, **({
            "scope_for_input": self.resolver, "sources": self.sources,
            "requests": self.requests, "memory_reader": self.reader,
            "storage": self.storage, "area_id": self.area,
            "enable_legacy_memory": True,
        } | changes))

    async def test_composition_is_lazy_and_uses_existing_services(self):
        composed = self.compose()
        self.provider.assert_not_called()
        self.resolver.assert_not_called()
        self.assertIs(composed.adapter.contexts.tasks, self.requests)
        self.assertIs(composed.adapter.contexts.publication.sources, composed.files.sources)
        self.assertEqual(composed.adapter.contexts.publication.area_id, composed.files.area_id)
        self.assertIs(composed.adapter.executions._pool, self.pool)
        self.assertIsNone(composed.files.request_authority)
        self.assertEqual(composed.adapter.qualified_write_tools, frozenset({"remember", "create_text_file"}))
        modules = await composed.adapter.modules_for_scope(self.scope)
        self.assertIs(modules[0].read_backend, self.reader)
        self.assertIs(modules[0].write_backend.sources, self.sources)
        self.assertIs(modules[1].repository, composed.files)
        self.provider.assert_not_called()

    async def test_memory_is_optional_and_retained_files_do_not_require_user_request(self):
        composed = self.compose(enable_legacy_memory=False, memory_reader=None)
        modules = await composed.adapter.modules_for_scope(self.scope)
        self.assertEqual(len(modules), 1)
        self.assertEqual(modules[0].name, "retained_files")
        self.assertEqual(composed.adapter.retained_write_tools, frozenset({"create_text_file"}))
        self.assertNotIn("remember", composed.allowed_tools)

    async def test_access_check_is_not_replaced_by_cached_permission(self):
        check = Mock()
        composed = self.compose(access_check=check)
        await composed.adapter.modules_for_scope(self.scope)
        await composed.adapter.modules_for_scope(self.scope)
        self.assertEqual(check.call_count, 2)
        with self.assertRaises(BindingError):
            await composed.adapter.modules_for_scope(replace(self.scope, owner_id="other"))

    def test_mismatched_database_is_rejected(self):
        with self.assertRaises(ValueError):
            self.compose(requests=TaskRepository(object(), TaskLimits()))

    def test_legacy_reader_and_foreign_area_are_rejected(self):
        with self.assertRaises(ValueError):
            self.compose(memory_reader=SimpleNamespace())
        with self.assertRaises(ValueError):
            self.compose(area_id=uuid4())

    def test_foreign_owner_and_fixed_session_are_rejected(self):
        foreign = SourceMemoryReadBackend(self.sources,
            SourceMemoryRetrieval(self.sources, self.provider), owner_id="other", area_id=self.area)
        with self.assertRaises(ValueError):
            self.compose(memory_reader=foreign)
        fixed = SourceMemoryReadBackend(self.sources,
            SourceMemoryRetrieval(self.sources, self.provider), owner_id="owner", area_id=self.area,
            session_id=uuid4())
        with self.assertRaises(ValueError):
            self.compose(memory_reader=fixed)

    def test_foreign_settings_and_selection_are_rejected(self):
        with self.assertRaises(ValueError):
            self.compose(settings=JarvisSettings(_env_file=None, JARVIS_OWNER_ID="other"))
        with self.assertRaises(ValueError):
            self.compose(selections=object())
