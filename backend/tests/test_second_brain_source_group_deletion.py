"""Explicit shared-original deletion; no automatic expansion of user authority."""
import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from app.knowledge.deletion import SourceDeletionService
from app.knowledge.lifecycle_tools import (
    SourceLifecycleToolModule, audit_source_lifecycle_arguments, audit_source_lifecycle_result,
)
from app.knowledge.models import SourceError, SourcePublication
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation
from app.tools.module import ToolBuildContext
from evals.second_brain.source_fixture import source_fixture


class SourceGroupInputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pool = object()
        module = SourceLifecycleToolModule(SimpleNamespace(pool=pool), SimpleNamespace(pool=pool), "fixture", uuid4())
        self.tools = {tool.name: tool for tool in module.build_tools(ToolBuildContext(None, "fixture", uuid4()))}
        self.first = {"source_id": str(uuid4()), "expected_revision": 1}
        self.second = {"source_id": str(uuid4()), "expected_revision": 1}

    async def test_group_is_bounded_unique_and_cannot_supply_authority(self):
        groups = [[], [self.first], [self.first, self.first],
                  [self.first, self.second | {"expected_revision": True}],
                  [self.first, self.second | {"area_id": str(uuid4())}],
                  [self.first, self.second | {"source_id": "private prose"}],
                  [{"source_id": str(uuid4()), "expected_revision": 1} for _ in range(33)]]
        for sources in groups:
            with self.subTest(sources=sources):
                result = await self.tools["delete_memory_source_group"].handler({"sources": sources})
                self.assertEqual(result["error"]["code"], "invalid_tool_input")
        result = await self.tools["delete_memory_source_group"].handler(
            {"sources": [self.first, self.second], "confirmed": True})
        self.assertEqual(result["error"]["code"], "invalid_tool_input")

    async def test_finalized_user_request_not_worker_or_missing_context_is_required(self):
        arguments = {"sources": [self.first, self.second]}
        result = await self.tools["delete_memory_source_group"].handler(arguments)
        self.assertEqual(result["error"]["code"], "knowledge_request_not_finalized")
        with bind_tool_invocation(provider_call_id="fixture", request_event_id=uuid4(), request_revision=1,
                                  task=TaskToolInvocation(uuid4(), 1, uuid4())):
            result = await self.tools["delete_memory_source_group"].handler(arguments)
        self.assertEqual(result["error"]["code"], "knowledge_request_not_finalized")

    async def test_shared_scope_inspection_rejects_invalid_cursors_and_bounds(self):
        for changes in ({"limit": True}, {"limit": 0}, {"limit": 129}, {"after_source_id": "private prose"}):
            result = await self.tools["inspect_memory_source_deletion"].handler(
                {"source_id": self.first["source_id"], **changes})
            self.assertEqual(result["error"]["code"], "invalid_tool_input")

    def test_group_audit_keeps_outcome_without_originals_or_invalid_prose(self):
        secret = "private synthetic body must not enter audit"
        arguments = audit_source_lifecycle_arguments(
            {"sources": [self.first | {"content": secret}, {"source_id": secret}, secret]})
        self.assertNotIn(secret, json.dumps(arguments))
        result = audit_source_lifecycle_result({"status": "success", "deletions": [
            {"source_id": self.first["source_id"], "revision": 1, "read_barrier": "deleted", "content": secret}]})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["deletions"][0]["status"], "success")
        self.assertEqual(result["deletions"][0]["read_barrier"], "deleted")
        self.assertNotIn(secret, json.dumps(result))


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested shared-source fixtures required")
class SourceGroupDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.history = ContextRepository(self.tasks)
        self.selections = SessionKnowledgeAreas(self.f.sources, self.tasks, self.f.area)
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.other_area = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Other fixture','test') returning id",
                self.f.owner)
        self.tools = self.make_tools()

    def make_tools(self, access=None):
        module = SourceLifecycleToolModule(self.f.sources, self.tasks, self.f.owner, self.f.area,
                                           access_check=access, selections=self.selections)
        return {tool.name: tool for tool in module.build_tools(ToolBuildContext(None, self.f.owner, self.session))}

    async def alias(self, original, *, area=None):
        return await self.f.sources.publish(self.f.owner, SourcePublication(
            area_id=area or self.f.area, namespace="fixture", source_key=uuid4().hex,
            payload_id=original.payload_id, source_type=original.source_type, trust=original.trust,
            source_time=original.source_time, time_basis=original.time_basis, expected_revision=0))

    async def request(self, *, finality="final", session=None):
        return await self.history.append(self.f.owner, ContextIngress(
            session_id=session or self.session, source_namespace="voice:shared-deletion",
            source_event_id=uuid4().hex, receipt_id=uuid4().hex, kind="user_transcript", trust="user_input",
            finality=finality, source_time=datetime.now(UTC),
            payload={"text": "Delete the originals and histories of both explicitly named synthetic sources, including their shared copy."}))

    async def invoke(self, sources, *, request=None, tools=None):
        request = request or await self.request()
        with bind_tool_invocation(provider_call_id=uuid4().hex, request_event_id=request.id, request_revision=request.revision):
            return await (tools or self.tools)["delete_memory_source_group"].handler({
                "sources": [{"source_id": str(source.source_id), "expected_revision": source.revision} for source in sources]})

    async def intact(self, *sources):
        for source in sources:
            current = await self.f.sources.read(self.f.owner, source.area_id, source.source_id)
            self.assertEqual(current.document_id, source.document_id)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_memory_outbox where owner_id=$1 and operation='delete'", self.f.owner), 0)

    async def test_shared_inspection_is_paginated_metadata_across_owned_areas(self):
        original = await self.f.publish("Synthetic private shared marker")
        siblings = [await self.alias(original, area=self.other_area) for _ in range(3)]
        result = await self.tools["inspect_memory_source_deletion"].handler(
            {"source_id": str(original.source_id), "limit": 2})
        page = result["deletion"]["shared_original_sources"]
        self.assertEqual(len(page["items"]), 2)
        self.assertIsNotNone(page["next_after_source_id"])
        next_result = await self.tools["inspect_memory_source_deletion"].handler({
            "source_id": str(original.source_id), "limit": 2, "after_source_id": page["next_after_source_id"]})
        final = next_result["deletion"]["shared_original_sources"]
        self.assertEqual(len(final["items"]), 1)
        self.assertIsNone(final["next_after_source_id"])
        self.assertEqual({row["source_id"] for row in page["items"] + final["items"]},
                         {str(source.source_id) for source in siblings})
        self.assertNotIn("Synthetic private shared marker", json.dumps(result))
        with self.assertRaises(SourceError):
            await SourceDeletionService(self.f.sources).inspect(self.f.other, self.f.area, original.source_id)
        await self.intact(original, *siblings)

    async def test_explicit_cross_area_group_deletes_history_but_preserves_unrelated_original(self):
        original = await self.f.publish("Shared old original")
        sibling = await self.alias(original, area=self.other_area)
        newer = await self.f.publish("Revised current original", expected_revision=1)
        control = await self.f.publish("Unrelated retained control", source_key="control")
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_knowledge_areas set status='archived',revision=revision+1 where owner_id=$1 and id=$2",
                self.f.owner, self.other_area)
        result = await self.invoke([newer, sibling])
        self.assertEqual(result["status"], "success", result)
        self.assertFalse(result["scope_expanded"])
        self.assertEqual(set(result["knowledge_scope"]["area_ids"]), {str(self.f.area), str(self.other_area)})
        self.assertTrue(all(row["read_barrier"] == "deleted" and
                            row["local_originals"]["remaining_content_payloads"] == 0 and
                            row["full_erasure_verified"] is False for row in result["deletions"]))
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_context_payloads
                where owner_id=$1 and id=any($2::uuid[]) and content is not null""",
                self.f.owner, [original.payload_id, newer.payload_id]), 0)
            before = await connection.fetch("""select id,status,generation from jarvis_memory_outbox
                where owner_id=$1 and operation='delete' order by id""", self.f.owner)
        self.assertEqual(len(before), 3)
        self.assertEqual((await self.f.sources.read(self.f.owner, self.f.area, control.source_id)).content, control.content)
        repeated = await self.invoke([newer, sibling])
        self.assertEqual(repeated["status"], "success", repeated)
        async with self.f.pool.acquire() as connection:
            after = await connection.fetch("""select id,status,generation from jarvis_memory_outbox
                where owner_id=$1 and operation='delete' order by id""", self.f.owner)
        self.assertEqual(before, after)
        self.assertEqual((await self.selections.read(self.f.owner, self.session))["area_id"], str(self.f.area))

    async def test_unlisted_shared_source_refuses_all_members_without_expanding_scope(self):
        original = await self.f.publish()
        sibling = await self.alias(original)
        outside = await self.alias(original, area=self.other_area)
        result = await self.invoke([original, sibling])
        self.assertEqual(result["error"]["code"], "knowledge_source_shared_original")
        await self.intact(original, sibling, outside)

    async def test_bad_revision_or_missing_member_rolls_back_entire_group(self):
        first = await self.f.publish(source_key="first")
        second = await self.f.publish(source_key="second")
        for member, code in ((SimpleNamespace(source_id=second.source_id, revision=2), "knowledge_source_revision_conflict"),
                             (SimpleNamespace(source_id=uuid4(), revision=1), "knowledge_source_not_found")):
            with self.subTest(code=code):
                result = await self.invoke([first, member])
                self.assertEqual(result["error"]["code"], code)
                await self.intact(first, second)

    async def test_unfinalized_and_wrong_session_requests_do_not_authorize_group(self):
        first = await self.f.publish()
        second = await self.alias(first)
        async with self.f.pool.acquire() as connection:
            other_session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
        for request in (await self.request(finality="provisional"), await self.request(session=other_session)):
            result = await self.invoke([first, second], request=request)
            self.assertEqual(result["error"]["code"], "task_request_evidence_missing")
        await self.intact(first, second)

    async def test_runtime_closure_after_group_mutation_rolls_back_every_original(self):
        first = await self.f.publish()
        second = await self.alias(first)
        closed = False
        def access(owner):
            if closed:
                raise SourceError("knowledge_intake_closed", "Synthetic runtime closed", 503)
        tools = self.make_tools(access)
        original = self.f.sources.delete_sources_in_transaction
        async def close_after(*args, **kwargs):
            nonlocal closed
            result = await original(*args, **kwargs)
            closed = True
            return result
        with patch.object(self.f.sources, "delete_sources_in_transaction", side_effect=close_after):
            result = await self.invoke([first, second], tools=tools)
        self.assertEqual(result["error"]["code"], "knowledge_intake_closed")
        await self.intact(first, second)

    async def test_group_serializes_against_a_late_new_shared_mapping(self):
        first = await self.f.publish()
        second = await self.alias(first)
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.tasks._voice_source
        async def hold(*args, **kwargs):
            result = await original(*args, **kwargs)
            entered.set()
            await release.wait()
            return result
        deleting = adding = None
        try:
            with patch.object(self.tasks, "_voice_source", side_effect=hold):
                deleting = asyncio.create_task(self.invoke([first, second]))
                await asyncio.wait_for(entered.wait(), 5)
                adding = asyncio.create_task(self.alias(first, area=self.other_area))
                with self.assertRaises(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(adding), .1)
                release.set()
                result = await asyncio.wait_for(deleting, 10)
                self.assertEqual(result["status"], "success", result)
                with self.assertRaises(SourceError):
                    await asyncio.wait_for(adding, 10)
        finally:
            release.set()
            await asyncio.gather(*(task for task in (deleting, adding) if task), return_exceptions=True)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_knowledge_sources where owner_id=$1", self.f.owner), 2)

    async def test_repository_group_seam_requires_transaction_and_exact_unique_targets(self):
        first = await self.f.publish()
        second = await self.alias(first)
        targets = [(self.f.area, first.source_id, 1), (self.f.area, second.source_id, 1)]
        async with self.f.pool.acquire() as connection:
            with self.assertRaises(RuntimeError):
                await self.f.sources.delete_sources_in_transaction(connection, self.f.owner, targets)
        async with self.f.sources.transaction(self.f.owner) as connection:
            for invalid in (targets[:1], [targets[0], targets[0]], [targets[0], (self.f.area, second.source_id, True)]):
                with self.assertRaises(ValueError):
                    await self.f.sources.delete_sources_in_transaction(connection, self.f.owner, invalid)
        await self.intact(first, second)

