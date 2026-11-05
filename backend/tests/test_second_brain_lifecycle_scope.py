"""Existing source identities remain addressable after session scope changes."""
from contextlib import asynccontextmanager
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.knowledge.lifecycle_tools import SourceLifecycleToolModule
from app.knowledge.models import SourceError
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from evals.second_brain.source_fixture import source_fixture


class LifecycleScopeContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner = "lifecycle-scope-fixture"
        self.default, self.target, self.session, self.source = (uuid4() for _ in range(4))
        self.connection = SimpleNamespace(fetchval=AsyncMock())
        pool = object()

        @asynccontextmanager
        async def transaction(owner):
            yield self.connection

        self.sources = SimpleNamespace(pool=pool, transaction=transaction, delete_source_in_transaction=AsyncMock())
        self.requests = SimpleNamespace(pool=pool, lock_owner=AsyncMock(), _voice_source=AsyncMock())
        self.selection = SimpleNamespace(sources=self.sources, default_area_id=self.default,
            source_area=AsyncMock(return_value=self.target), read=AsyncMock(side_effect=AssertionError("No default lookup")))
        self.lifecycle = SimpleNamespace(**{name: AsyncMock(return_value={"source_id": str(self.source), "state": "active"})
            for name in ("inspect", "correct", "forget", "retry_cleanup")})
        self.deletions = SimpleNamespace(inspect=AsyncMock(return_value={"source_id": str(self.source)}),
                                        documents=AsyncMock(return_value={"source_id": str(self.source), "versions": []}))
        self.guard = Mock()

    def tools(self, *, scoped=True, session=True):
        module = SourceLifecycleToolModule(self.sources, self.requests, self.owner, self.default,
            access_check=self.guard, selections=self.selection if scoped else None)
        with patch("app.knowledge.lifecycle_tools.SourceLifecycleService", return_value=self.lifecycle), \
             patch("app.knowledge.lifecycle_tools.SourceDeletionService", return_value=self.deletions):
            return {tool.name: tool for tool in module.build_tools(
                ToolBuildContext(None, self.owner, self.session if session else None))}

    async def test_all_operations_use_the_original_target_area_not_the_session_default(self):
        tools = self.tools()
        calls = [
            ("inspect_memory_source_state", {}, self.lifecycle.inspect),
            ("inspect_memory_source_deletion", {}, self.deletions.inspect),
            ("list_memory_source_deletion_records", {}, self.deletions.documents),
            ("correct_memory_source", {"expected_revision": 1, "replacement": "A synthetic correction."}, self.lifecycle.correct),
            ("forget_memory_source", {"expected_revision": 1}, self.lifecycle.forget),
            ("retry_memory_source_cleanup", {"expected_generation": 2}, self.lifecycle.retry_cleanup),
        ]
        with bind_tool_invocation(provider_call_id="scope-call", request_event_id=uuid4(), request_revision=1):
            for name, extra, operation in calls:
                with self.subTest(name=name):
                    result = await tools[name].handler({"source_id": str(self.source), **extra})
                    self.assertEqual(result["status"], "success")
                    self.assertEqual(operation.call_args.args[:3], (self.owner, self.target, self.source))
                    self.assertEqual(result["knowledge_scope"]["area_id"], str(self.target))
                    self.assertEqual(result["knowledge_scope"]["origin"], "existing_source")
            deletion = await tools["delete_memory_source"].handler(
                {"source_id": str(self.source), "expected_revision": 1})
        self.assertEqual(deletion["status"], "success")
        self.assertEqual(self.sources.delete_source_in_transaction.call_args.args[1:],
                         (self.owner, self.target, self.source))
        self.deletions.inspect.assert_awaited_with(self.owner, self.target, self.source)
        self.selection.source_area.assert_awaited_with(self.owner, self.session, self.source)
        self.selection.read.assert_not_awaited()

    async def test_a_target_lookup_change_during_operation_does_not_retarget_its_result(self):
        async def inspect(owner, area, source):
            self.selection.source_area.return_value = self.default
            return {"source_id": str(source), "state": "forgotten"}
        self.lifecycle.inspect.side_effect = inspect
        result = await self.tools()["inspect_memory_source_state"].handler({"source_id": str(self.source)})
        self.assertEqual(result["knowledge_scope"]["area_id"], str(self.target))
        self.assertEqual(self.lifecycle.inspect.call_args.args[1], self.target)

    async def test_fixed_area_and_sessionless_read_contracts_remain_unchanged(self):
        fixed = self.tools(scoped=False, session=False)
        self.assertEqual(set(fixed), {"inspect_memory_source_state", "inspect_memory_source_deletion",
                                     "list_memory_source_deletion_records"})
        result = await fixed["inspect_memory_source_state"].handler({"source_id": str(self.source)})
        self.assertEqual(self.lifecycle.inspect.call_args.args[1], self.default)
        self.assertNotIn("knowledge_scope", result)
        self.selection.source_area.assert_not_awaited()

    async def test_foreign_target_or_invalid_model_scope_cannot_reach_operations(self):
        tools = self.tools()
        invalid = await tools["forget_memory_source"].handler(
            {"source_id": str(self.source), "expected_revision": 1, "area_id": str(self.target)})
        self.assertEqual(invalid["error"]["code"], "invalid_tool_input")
        self.selection.source_area.assert_not_awaited()
        self.selection.source_area.side_effect = SourceError("knowledge_source_not_found", "No owned source", 404)
        result = await tools["inspect_memory_source_state"].handler({"source_id": str(self.source)})
        self.assertEqual(result["error"]["code"], "knowledge_source_not_found")
        self.lifecycle.inspect.assert_not_awaited()
        self.lifecycle.forget.assert_not_awaited()

    async def test_missing_finalized_request_still_prevents_mutation(self):
        result = await self.tools()["forget_memory_source"].handler(
            {"source_id": str(self.source), "expected_revision": 1})
        self.assertEqual(result["error"]["code"], "knowledge_request_not_finalized")
        self.lifecycle.forget.assert_not_awaited()

    async def test_runtime_revocation_after_scope_lookup_prevents_operation(self):
        permitted = True
        def guard(owner):
            if not permitted:
                raise SourceError("knowledge_intake_closed", "Runtime closed", 503)
        self.guard.side_effect = guard
        async def revoke(*args):
            nonlocal permitted
            permitted = False
            return self.target
        self.selection.source_area.side_effect = revoke
        result = await self.tools()["inspect_memory_source_state"].handler({"source_id": str(self.source)})
        self.assertEqual(result["error"]["code"], "knowledge_intake_closed")
        self.lifecycle.inspect.assert_not_awaited()

    async def test_source_area_requires_owned_session_and_source_without_active_area_lookup(self):
        resolver = SessionKnowledgeAreas(self.sources, self.requests, self.default)
        self.connection.fetchval.side_effect = [True, self.target]
        self.assertEqual(await resolver.source_area(self.owner, self.session, self.source), self.target)
        self.assertEqual(self.connection.fetchval.call_args.args[1:], (self.owner, self.source))
        self.assertIn("owner_id=$1", self.connection.fetchval.call_args.args[0])
        self.connection.fetchval.side_effect = [False]
        with self.assertRaises(SourceError) as caught:
            await resolver.source_area(self.owner, self.session, self.source)
        self.assertEqual(caught.exception.code, "knowledge_session_unavailable")
        self.connection.fetchval.side_effect = [True, None]
        with self.assertRaises(SourceError) as caught:
            await resolver.source_area(self.owner, self.session, self.source)
        self.assertEqual(caught.exception.code, "knowledge_source_not_found")
        with self.assertRaises(ValueError):
            await resolver.source_area(self.owner, self.session, "not-a-source")

    def test_mismatched_resolver_pool_or_default_is_rejected(self):
        for changes in ({"sources": SimpleNamespace(pool=object())}, {"default_area_id": uuid4()}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                SourceLifecycleToolModule(self.sources, self.requests, self.owner, self.default,
                    selections=SimpleNamespace(**(vars(self.selection) | changes)))


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested source fixtures required")
class LifecycleScopeDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_area_inspection_and_cleanup_survive_default_change_and_archive(self):
        async with source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()) as f:
            tasks = TaskRepository(f.pool, TaskLimits())
            selections = SessionKnowledgeAreas(f.sources, tasks, f.area)
            async with f.pool.acquire() as connection:
                session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", f.owner)
                foreign_session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", f.other)
                target = await connection.fetchval(
                    "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Old project','robotics') returning id", f.owner)
            original = await ContextRepository(tasks, publication=CanonicalSourcePublication(f.sources, target)).append(
                f.owner, ContextIngress(session_id=session, source_namespace="voice:lifecycle-scope",
                    source_event_id="original", receipt_id="original", kind="user_transcript",
                    trust="user_input", payload={"text": "A synthetic old project source."}))
            request = await ContextRepository(tasks, publication=CanonicalSourcePublication(f.sources, f.area)).append(
                f.owner, ContextIngress(session_id=session, source_namespace="voice:lifecycle-scope",
                    source_event_id="request", receipt_id="request", kind="user_transcript",
                    trust="user_input", payload={"text": "Forget, then delete the synthetic original."}))
            async with f.pool.acquire() as connection:
                source = await connection.fetchval(
                    "select source_id from jarvis_knowledge_context_sources where owner_id=$1 and event_id=$2",
                    f.owner, original.id)
                await connection.execute(
                    "update jarvis_knowledge_areas set status='archived',revision=revision+1 where owner_id=$1 and id=$2",
                    f.owner, target)
            module = SourceLifecycleToolModule(f.sources, tasks, f.owner, f.area, selections=selections)
            tools = {tool.name: tool for tool in module.build_tools(ToolBuildContext(None, f.owner, session))}
            arguments = {"source_id": str(source)}
            with bind_tool_invocation(provider_call_id="owned-lifecycle", request_event_id=request.id, request_revision=request.revision):
                state = await tools["inspect_memory_source_state"].handler(arguments)
                self.assertEqual(state["status"], "success", state)
                self.assertEqual(state["knowledge_scope"]["area_id"], str(target))
                forgotten = await tools["forget_memory_source"].handler(arguments | {"expected_revision": 1})
                self.assertEqual(forgotten["status"], "success", forgotten)
                self.assertEqual(forgotten["source_state"]["state"], "forgotten")
                deleted = await tools["delete_memory_source"].handler(arguments | {"expected_revision": 1})
                self.assertEqual(deleted["status"], "success", deleted)
                self.assertTrue(deleted["accepted"])
                recorded = await tools["inspect_memory_source_state"].handler(arguments)
                self.assertEqual(recorded["source_state"]["state"], "deleted")
                retried = await tools["retry_memory_source_cleanup"].handler(
                    arguments | {"expected_generation": recorded["source_state"]["generation"]})
                self.assertEqual(retried["status"], "success", retried)
            self.assertEqual((await selections.read(f.owner, session))["area_id"], str(f.area))
            with self.assertRaises(SourceError):
                await selections.source_area(f.other, None, source)
            with self.assertRaises(SourceError):
                await selections.source_area(f.owner, foreign_session, source)
