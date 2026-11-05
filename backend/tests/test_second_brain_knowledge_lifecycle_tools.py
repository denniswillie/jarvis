"""Conversational deletion acceptance over isolated canonical originals.

No model, production owner, external effect or Hindsight endpoint is contacted.
"""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.knowledge.lifecycle_tools import (
    SourceLifecycleToolModule, audit_source_lifecycle_arguments, audit_source_lifecycle_result,
)
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.models import SourceError, SourcePublication
from app.knowledge.runtime import KnowledgeRuntime
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.memory.tools import MemoryToolModule
from app.runtime.feature_config import TaskLimits
from app.tasks.models import TaskVoiceSource
from app.tasks.repository import TaskRepository
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry
from evals.second_brain.source_fixture import source_fixture


class SourceLifecycleToolContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pool = object()
        self.sources = SimpleNamespace(pool=self.pool)
        self.requests = SimpleNamespace(pool=self.pool)
        self.owner, self.area, self.session = "fixture", uuid4(), uuid4()
        self.module = SourceLifecycleToolModule(self.sources, self.requests, self.owner, self.area)
        self.context = ToolBuildContext(None, self.owner, self.session)
        self.tools = {tool.name: tool for tool in self.module.build_tools(self.context)}

    async def test_registry_is_explicit_and_does_not_replace_logical_forgetting(self):
        self.assertEqual(set(self.tools), {
            "inspect_memory_source_deletion", "list_memory_source_deletion_records", "delete_memory_source",
            "inspect_memory_source_state", "correct_memory_source", "forget_memory_source",
            "retry_memory_source_cleanup",
            "delete_memory_source_group",
        })
        self.assertEqual({tool.name for tool in self.tools.values() if tool.mutates_state},
                         {"delete_memory_source", "correct_memory_source", "forget_memory_source",
                          "retry_memory_source_cleanup", "delete_memory_source_group"})
        for tool in self.tools.values():
            self.assertFalse(tool.input_schema["additionalProperties"])
            self.assertNotIn("owner_id", tool.input_schema["properties"])
            self.assertNotIn("area_id", tool.input_schema["properties"])
            self.assertNotIn("request_event_id", tool.input_schema["properties"])
        readonly = self.module.build_tools(ToolBuildContext(None, self.owner, None))
        self.assertEqual(len(readonly), 3)
        self.assertTrue(all(not tool.mutates_state for tool in readonly))
        with self.assertRaises(SourceError):
            self.module.build_tools(ToolBuildContext(None, "foreign", self.session))
        with self.assertRaises(ValueError):
            SourceLifecycleToolModule(self.sources, SimpleNamespace(pool=object()), self.owner, self.area)

    def test_delegated_catalog_removes_only_foreground_replacement(self):
        from dataclasses import replace
        delegated = replace(self.module, allow_foreground_correction=False)
        tools = {tool.name: tool for tool in delegated.build_tools(self.context)}
        self.assertEqual(set(tools), set(self.tools) - {"correct_memory_source"})
        self.assertTrue(tools["forget_memory_source"].mutates_state)
        self.assertTrue(tools["delete_memory_source"].mutates_state)
        self.assertEqual(len(delegated.build_tools(ToolBuildContext(None, self.owner, None))), 3)

    async def test_invalid_arguments_cannot_supply_scope_or_reach_database(self):
        valid = {"source_id": str(uuid4()), "expected_revision": 1}
        for arguments in (
            {**valid, "source_id": "not-a-source"}, {**valid, "expected_revision": True},
            {**valid, "expected_revision": 0}, {**valid, "expected_revision": "1"},
            {**valid, "owner_id": "foreign"}, {**valid, "area_id": str(uuid4())},
            {**valid, "request_event_id": str(uuid4())}, {**valid, "confirmed": True},
        ):
            with self.subTest(arguments=arguments):
                result = await self.tools["delete_memory_source"].handler(arguments)
                self.assertEqual(result["error"]["code"], "invalid_tool_input")
        for arguments in ({"source_id": str(uuid4()), "limit": 129},
                          {"source_id": str(uuid4()), "after_revision": False}):
            result = await self.tools["list_memory_source_deletion_records"].handler(arguments)
            self.assertEqual(result["error"]["code"], "invalid_tool_input")

    async def test_missing_user_request_and_worker_context_are_not_deletion_authority(self):
        args = {"source_id": str(uuid4()), "expected_revision": 1}
        result = await self.tools["delete_memory_source"].handler(args)
        self.assertEqual(result["error"]["code"], "knowledge_request_not_finalized")
        with bind_tool_invocation(provider_call_id="fixture-call", request_event_id=uuid4(), request_revision=1,
                                 task=TaskToolInvocation(uuid4(), 1, uuid4())):
            result = await self.tools["delete_memory_source"].handler(args)
        self.assertEqual(result["error"]["code"], "knowledge_request_not_finalized")

    async def test_audit_omits_unvalidated_prose_and_private_result_bodies(self):
        secret = "S22 private malformed input"
        arguments = audit_source_lifecycle_arguments({"source_id": secret, "content": secret, "limit": secret})
        self.assertNotIn(secret, json.dumps(arguments))
        receipt = audit_source_lifecycle_result({
            "status": "failed", "deletion": {"source_id": str(uuid4()), "content": secret},
            "error": {"code": "fixture_refusal", "message": secret},
        })
        self.assertNotIn(secret, json.dumps(receipt))
        self.assertTrue(receipt["content_omitted"])

    async def test_runtime_factory_composes_with_selected_reads_and_stops_when_closed(self):
        # Exercise the factory without starting a worker or constructing an HTTP client.
        runtime = object.__new__(KnowledgeRuntime)
        runtime.owner_id, runtime.sources, runtime._closed = self.owner, self.sources, False
        runtime.configuration = SimpleNamespace(area_id=self.area)
        runtime.context_repository = SimpleNamespace(tasks=self.requests)
        runtime.area_selections = SessionKnowledgeAreas(self.sources, self.requests, self.area)
        self.assertIs(runtime.lifecycle_tool_module_for(self.owner).selections, runtime.area_selections)
        selected = SourceMemoryReadBackend(
            self.sources, SimpleNamespace(sources=self.sources), owner_id=self.owner, area_id=self.area,
        )
        registry = create_tool_registry(modules=[
            MemoryToolModule(read_backend=selected), runtime.lifecycle_tool_module_for(self.owner),
        ], context=self.context)
        self.assertIsNotNone(registry.get("search_memory"))
        self.assertIsNotNone(registry.get("explain_memory"))
        self.assertIsNotNone(registry.get("delete_memory_source"))
        self.assertIsNone(registry.get("forget_memory"))
        runtime._closed = True
        result = await registry.get("delete_memory_source").handler(
            {"source_id": str(uuid4()), "expected_revision": 1})
        self.assertEqual(result["error"]["code"], "knowledge_intake_closed")


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain fixtures required",
)
class SourceLifecycleVoiceDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id(),
        ))
        self.owner, self.area = self.fixture.owner, self.fixture.area
        self.sources = self.fixture.sources
        self.tasks = TaskRepository(self.fixture.pool, TaskLimits())
        self.sessions = {}
        async with self.fixture.pool.acquire() as connection:
            for owner in (self.owner, self.fixture.other):
                self.sessions[owner] = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", owner)
            self.other_area = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Other scope','test') returning id",
                self.owner)
        self.session = self.sessions[self.owner]
        self.module = SourceLifecycleToolModule(self.sources, self.tasks, self.owner, self.area)
        self.tools = {tool.name: tool for tool in self.module.build_tools(ToolBuildContext(None, self.owner, self.session))}

    async def request(self, *, owner=None, session=None, trust="user_input", kind="user_request",
                      finality="final", previous=None):
        owner = owner or self.owner
        async with self.tasks.transaction(owner) as connection:
            payload = await self.tasks.payload(connection, owner, {"text": "Delete the identified stored originals"})
            return await connection.fetchrow("""insert into jarvis_context_events(
                owner_id,session_id,source_namespace,source_event_id,revision,supersedes_event_id,kind,finality,trust,payload_id)
                values($1,$2,'fixture-memory-command',$3,$4,$5,$6,$7,$8,$9) returning *""",
                owner, session or self.sessions[owner],
                previous["source_event_id"] if previous else str(uuid4()),
                previous["revision"] + 1 if previous else 1, previous["id"] if previous else None,
                kind, finality, trust, payload)

    async def invoke(self, name, arguments, *, request=None, call_id=None):
        with bind_tool_invocation(provider_call_id=call_id or str(uuid4()),
                                 request_event_id=request["id"] if request else None,
                                 request_revision=request["revision"] if request else None):
            return await self.tools[name].handler(arguments)

    async def delete(self, source, request, *, revision=None, call_id=None):
        return await self.invoke("delete_memory_source", {
            "source_id": str(source.source_id),
            "expected_revision": source.revision if revision is None else revision,
        }, request=request, call_id=call_id)

    async def inspect(self, source):
        return await self.invoke("inspect_memory_source_deletion", {"source_id": str(source.source_id)})

    async def content(self, payload_id):
        async with self.fixture.pool.acquire() as connection:
            return await connection.fetchval("select content from jarvis_context_payloads where owner_id=$1 and id=$2",
                                             self.owner, payload_id)

    async def cleanup_rows(self, source):
        async with self.fixture.pool.acquire() as connection:
            return [dict(row) for row in await connection.fetch("""select o.id,o.document_id,o.status,o.generation
                from jarvis_memory_outbox o join jarvis_knowledge_source_versions v
                    on v.owner_id=o.owner_id and v.document_id=o.document_id
                where v.owner_id=$1 and v.source_id=$2 and o.operation='delete' order by v.revision""",
                self.owner, source.source_id)]

    async def test_readonly_status_and_manifest_work_without_user_request_or_provider(self):
        source = await self.fixture.publish("S22 private original")
        await self.fixture.publish("S22 private revision", expected_revision=1)
        result = await self.inspect(source)
        self.assertEqual(result["deletion"]["state"], "not_requested")
        self.assertFalse(result["deletion"]["full_erasure_verified"])
        first = await self.invoke("list_memory_source_deletion_records",
            {"source_id": str(source.source_id), "limit": 1})
        self.assertEqual(first["manifest"]["next_after_revision"], 1)
        second = await self.invoke("list_memory_source_deletion_records",
            {"source_id": str(source.source_id), "after_revision": 1, "limit": 1})
        self.assertIsNone(second["manifest"]["next_after_revision"])
        self.assertNotIn("S22 private", json.dumps([result, first, second]))
        self.assertIsNotNone(await self.content(source.payload_id))

    async def test_explicit_deletion_removes_versions_and_linked_copy_not_unrelated_original(self):
        first = await self.fixture.publish("S22 source revision one")
        current = await self.fixture.publish("S22 source revision two", expected_revision=1)
        unrelated = await self.fixture.publish("S22 unrelated source", source_key="unrelated")
        async with self.tasks.transaction(self.owner) as connection:
            copy = await self.tasks.payload(connection, self.owner, {"text": "S22 linked private copy"},
                                            source_payload_ids=(first.payload_id,))
        request = await self.request()
        result = await self.delete(current, request)
        self.assertTrue(result["accepted"])
        receipt = result["deletion"]
        self.assertEqual((receipt["read_barrier"], receipt["state"]), ("deleted", "pending"))
        self.assertEqual(receipt["local_originals"]["remaining_content_payloads"], 0)
        self.assertEqual(receipt["provider"]["pending"], 2)
        self.assertFalse(receipt["full_erasure_verified"])
        for payload in (first.payload_id, current.payload_id, copy):
            self.assertIsNone(await self.content(payload))
        self.assertIsNotNone(await self.content(unrelated.payload_id))
        self.assertIsNotNone(await self.content(request["payload_id"]))
        with self.assertRaises(SourceError):
            await self.sources.read(self.owner, self.area, current.source_id)
        self.assertNotIn("S22 source", json.dumps(result))

    async def test_repeated_and_concurrent_provider_call_ids_preserve_one_deletion(self):
        source = await self.fixture.publish()
        request = await self.request()
        results = await asyncio.gather(
            self.delete(source, request, call_id="first-call"),
            self.delete(source, request, call_id="new-id-after-interruption"),
        )
        self.assertTrue(all(result["accepted"] for result in results))
        self.assertEqual({result["deletion"]["generation"] for result in results}, {2})
        before = await self.cleanup_rows(source)
        self.assertEqual(len(before), 1)
        repeat = await self.delete(source, request, call_id="another-new-provider-id")
        self.assertTrue(repeat["accepted"])
        self.assertEqual(await self.cleanup_rows(source), before)

    async def test_stale_source_revision_cannot_delete_corrected_content(self):
        source = await self.fixture.publish("Earlier original")
        current = await self.fixture.publish("Corrected original", expected_revision=1)
        before = await self.cleanup_rows(source)
        result = await self.delete(source, await self.request())
        self.assertEqual(result["error"]["code"], "knowledge_source_revision_conflict")
        self.assertIsNotNone(await self.content(current.payload_id))
        self.assertEqual(await self.cleanup_rows(source), before)
        self.assertEqual((await self.sources.read(self.owner, self.area, source.source_id)).revision, 2)

    async def test_non_user_and_unfinished_context_cannot_authorize_deletion(self):
        source = await self.fixture.publish()
        for options in (
            {"trust": "external_content", "kind": "observation"},
            {"trust": "assistant_output", "kind": "assistant_output"},
            {"trust": "tool_receipt", "kind": "tool_outcome"},
            {"finality": "provisional"},
        ):
            with self.subTest(options=options):
                result = await self.delete(source, await self.request(**options))
                self.assertEqual(result["error"]["code"], "task_request_evidence_missing")
        self.assertIsNotNone(await self.content(source.payload_id))
        self.assertEqual(await self.cleanup_rows(source), [])

    async def test_superseded_removed_and_foreign_user_requests_are_refused(self):
        source = await self.fixture.publish()
        superseded = await self.request()
        await self.request(previous=superseded, finality="revised")
        removed = await self.request()
        async with self.fixture.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2",
                                     self.owner, removed["payload_id"])
        foreign = await self.request(owner=self.fixture.other)
        async with self.fixture.pool.acquire() as connection:
            other_session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.owner)
        different_session = await self.request(session=other_session)
        for request in (superseded, removed, foreign, different_session):
            with self.subTest(request=request["id"]):
                result = await self.delete(source, request)
                self.assertEqual(result["error"]["code"], "task_request_evidence_missing")
        self.assertIsNotNone(await self.content(source.payload_id))
        self.assertEqual(await self.cleanup_rows(source), [])

    async def test_source_and_area_scope_are_server_owned(self):
        source = await self.fixture.publish()
        outside = await self.fixture.publish("Different area original", area_id=self.other_area, source_key="other-area")
        request = await self.request()
        for source_id in (outside.source_id, uuid4()):
            result = await self.invoke("delete_memory_source",
                {"source_id": str(source_id), "expected_revision": 1}, request=request)
            self.assertEqual(result["error"]["code"], "knowledge_source_not_found")
            read = await self.invoke("inspect_memory_source_deletion", {"source_id": str(source_id)})
            self.assertEqual(read["error"]["code"], "knowledge_source_not_found")
        self.assertIsNotNone(await self.content(source.payload_id))
        self.assertIsNotNone(await self.content(outside.payload_id))

    async def test_shared_original_refusal_is_atomic_and_explains_scope(self):
        source = await self.fixture.publish("Shared retained evidence")
        await self.sources.publish(self.owner, SourcePublication(
            area_id=self.other_area, namespace="fixture", source_key="shared",
            payload_id=source.payload_id, source_type=source.source_type, trust=source.trust,
            source_time=source.source_time, time_basis=source.time_basis, expected_revision=0,
        ))
        result = await self.delete(source, await self.request())
        self.assertEqual(result["error"]["code"], "knowledge_source_shared_original")
        self.assertIsNotNone(await self.content(source.payload_id))
        self.assertEqual(await self.cleanup_rows(source), [])
        self.assertEqual((await self.inspect(source))["deletion"]["read_barrier"], "active")

    async def test_runtime_closure_before_commit_rolls_back_deletion_and_outbox(self):
        source = await self.fixture.publish()
        closed = False
        def access(owner):
            if closed:
                raise SourceError("knowledge_intake_closed", "Fixture runtime closed.", 503)
        self.tools = {tool.name: tool for tool in SourceLifecycleToolModule(
            self.sources, self.tasks, self.owner, self.area, access_check=access,
        ).build_tools(ToolBuildContext(None, self.owner, self.session))}
        original = self.sources.delete_source_in_transaction
        async def close_after_write(*args, **kwargs):
            nonlocal closed
            result = await original(*args, **kwargs)
            closed = True
            return result
        with patch.object(self.sources, "delete_source_in_transaction", side_effect=close_after_write):
            result = await self.delete(source, await self.request())
        self.assertEqual(result["error"]["code"], "knowledge_intake_closed")
        self.assertIsNotNone(await self.content(source.payload_id))
        self.assertEqual(await self.cleanup_rows(source), [])
        self.assertEqual((await self.sources.read(self.owner, self.area, source.source_id)).generation, 1)

    async def test_current_request_lock_is_held_until_deletion_commit(self):
        source, request = await self.fixture.publish(), await self.request()
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.tasks._voice_source
        async def hold_request(*args, **kwargs):
            result = await original(*args, **kwargs)
            entered.set()
            await release.wait()
            return result
        with patch.object(self.tasks, "_voice_source", side_effect=hold_request):
            deleting = asyncio.create_task(self.delete(source, request))
            revising = None
            try:
                await asyncio.wait_for(entered.wait(), 5)
                revising = asyncio.create_task(self.request(previous=request, finality="revised"))
                with self.assertRaises(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(revising), 0.1)
                release.set()
                result = await asyncio.wait_for(deleting, 5)
                await asyncio.wait_for(revising, 5)
                self.assertTrue(result["accepted"])
            finally:
                release.set()
                await asyncio.gather(deleting, *([revising] if revising else []), return_exceptions=True)
        self.assertIsNone(await self.content(source.payload_id))

    async def test_transactional_deletion_seam_refuses_autocommit(self):
        source = await self.fixture.publish()
        async with self.fixture.pool.acquire() as connection:
            with self.assertRaises(RuntimeError):
                await self.sources.delete_source_in_transaction(connection, self.owner, self.area,
                                                                source.source_id, expected_revision=1)
        self.assertIsNotNone(await self.content(source.payload_id))
