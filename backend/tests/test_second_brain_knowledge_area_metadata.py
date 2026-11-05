"""Conversational area metadata with actual request identity and isolated SQL."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import asyncpg
from pydantic import ValidationError

from app.knowledge.areas import AREA_OPERATIONS, KnowledgeAreaService
from app.knowledge.area_tools import KnowledgeAreaToolModule, audit_area_arguments, audit_area_result
from app.knowledge.models import KnowledgeAreaDefinition, SourceError, SourcePublication
from app.runtime.feature_config import TaskLimits
from app.tasks.models import TaskVoiceSource
from app.tasks.repository import TaskRepository
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation
from app.tools.module import ToolBuildContext
from evals.second_brain.source_fixture import source_fixture


class AreaMetadataContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.owner, self.session = "metadata-fixture", uuid4()
        self.service = SimpleNamespace(create=AsyncMock(), update=AsyncMock())
        self.module = KnowledgeAreaToolModule(self.service, self.owner)
        self.tools = {tool.name: tool for tool in self.module.build_tools(ToolBuildContext(None, self.owner, self.session))}

    async def test_metadata_is_bounded_descriptive_text_not_authority(self):
        for fields in ({"name": " ", "domain": "test"}, {"name": "one", "domain": "\x00"},
                       {"name": "one", "domain": "test", "description": "x" * 4001},
                       {"name": "one", "domain": "test", "description": "\x00"},
                       {"name": "one", "domain": "test", "policy": {"allow_all": True}}):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                KnowledgeAreaDefinition.model_validate(fields)
        definition = KnowledgeAreaDefinition(name="  Project  ", domain=" robotics ")
        self.assertEqual((definition.name, definition.domain, definition.description), ("Project", "robotics", ""))
        for args in ({"areas": []}, {"areas": [{"name": "one", "domain": "test"}] * 9},
                     {"areas": [{"name": "one", "domain": "test", "owner_id": "foreign"}]}):
            result = await self.tools["create_knowledge_areas"].handler(args)
            self.assertEqual(result["error"]["code"], "invalid_tool_input")
        self.service.create.assert_not_awaited()

    async def test_new_mutations_require_finalized_foreground_request(self):
        operations = {"create_knowledge_areas": {"areas": [{"name": "one", "domain": "test"}]},
            "update_knowledge_area": {"area_id": str(uuid4()), "expected_revision": 1,
                                      "definition": {"name": "one", "domain": "test"}}}
        for name, arguments in operations.items():
            with self.subTest(name=name):
                result = await self.tools[name].handler(arguments)
                self.assertEqual(result["error"]["code"], "knowledge_request_not_finalized")
                with bind_tool_invocation(provider_call_id="worker", request_event_id=uuid4(), request_revision=1,
                        task=TaskToolInvocation(uuid4(), 1, uuid4())):
                    result = await self.tools[name].handler(arguments)
                self.assertEqual(result["error"]["code"], "knowledge_request_not_finalized")
        self.service.create.assert_not_awaited()
        self.service.update.assert_not_awaited()

    async def test_audits_omit_batch_names_and_descriptions(self):
        private = "Private business plan"
        arguments = audit_area_arguments({"areas": [{"name": private, "domain": private, "description": private}]})
        result = audit_area_result({"status": "success", "areas": [{"id": str(uuid4()), "revision": 1,
            "status": "active", "name": private, "domain": private, "description": private}]})
        self.assertNotIn(private, json.dumps([arguments, result]))
        self.assertEqual(len(result["areas"]), 1)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Manifested SQL fixtures required")
class AreaMetadataDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.service = KnowledgeAreaService(self.f.sources, self.tasks)
        self.sessions = {}
        async with self.f.pool.acquire() as connection:
            for owner in (self.f.owner, self.f.other):
                self.sessions[owner] = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", owner)
        self.tools = {tool.name: tool for tool in KnowledgeAreaToolModule(self.service, self.f.owner).build_tools(
            ToolBuildContext(None, self.f.owner, self.sessions[self.f.owner]))}

    async def request(self, *, owner=None, previous=None, trust="user_input", finality="final"):
        owner = owner or self.f.owner
        async with self.tasks.transaction(owner) as connection:
            payload = await self.tasks.payload(connection, owner, {"text": "Explicit synthetic area metadata request"})
            row = await connection.fetchrow("""insert into jarvis_context_events(owner_id,session_id,source_namespace,
                source_event_id,revision,supersedes_event_id,kind,finality,trust,payload_id)
                values($1,$2,'fixture-area-metadata',$3,$4,$5,$6,$7,$8,$9) returning *""", owner,
                self.sessions[owner], previous["source_event_id"] if previous else str(uuid4()),
                previous["revision"] + 1 if previous else 1, previous["id"] if previous else None,
                "user_request" if trust == "user_input" else "observation", finality, trust, payload)
        return row

    async def invoke(self, name, arguments, request):
        with bind_tool_invocation(provider_call_id=str(uuid4()), request_event_id=request["id"],
                                 request_revision=request["revision"]):
            return await self.tools[name].handler(arguments)

    async def create(self, areas, request=None):
        return await self.invoke("create_knowledge_areas", {"areas": areas}, request or await self.request())

    async def update(self, area, *, name="Renamed project", request=None, revision=None, description=None):
        definition = {key: area[key] for key in ("name", "domain", "description")}
        definition["name"] = name
        if description is not None:
            definition["description"] = description
        return await self.invoke("update_knowledge_area", {"area_id": area["id"],
            "expected_revision": revision or area["revision"], "definition": definition}, request or await self.request())

    async def counts(self):
        async with self.f.pool.acquire() as connection:
            return tuple(await connection.fetchrow("""select
                (select count(*) from jarvis_knowledge_areas where owner_id=$1) as areas,
                (select count(*) from jarvis_context_events where owner_id=$1 and source_namespace=$2) as receipts""",
                self.f.owner, AREA_OPERATIONS))

    async def test_one_request_creates_distinct_domain_areas_and_scoped_originals(self):
        result = await self.create([{"name": "Project", "domain": domain, "description": domain + " work"}
                                   for domain in ("software", "robotics", "fitness", "business")])
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["replayed"])
        self.assertEqual(len({area["id"] for area in result["areas"]}), 4)
        originals = []
        for area in result["areas"]:
            async with self.f.pool.acquire() as connection:
                payload = await self.tasks.payload(connection, self.f.owner, {"text": area["domain"] + " original"})
            source = await self.f.sources.publish(self.f.owner, SourcePublication(area_id=UUID(area["id"]),
                namespace="area-metadata-fixture", source_key="same-source-name", payload_id=payload,
                source_type="text", trust="user_input", source_time=None, time_basis="fixture record", expected_revision=0))
            originals.append(source)
        self.assertEqual(len({source.bank_name for source in originals}), 4)
        for source in originals:
            actual = await self.f.sources.read(self.f.owner, source.area_id, source.source_id)
            self.assertEqual(actual.content, source.content)
            self.assertEqual(actual.trust.value, "user_input")
            with self.assertRaises(SourceError):
                await self.f.sources.read(self.f.other, source.area_id, source.source_id)
            with self.assertRaises(SourceError):
                await self.f.sources.read(self.f.owner, self.f.area, source.source_id)

    async def test_concurrent_new_call_ids_recover_one_creation_batch(self):
        request = await self.request()
        definitions = [{"name": "One", "domain": "test"}, {"name": "Two", "domain": "test"}]
        first, second = await asyncio.gather(self.create(definitions, request), self.create(definitions, request))
        self.assertEqual([area["id"] for area in first["areas"]], [area["id"] for area in second["areas"]])
        self.assertEqual(sorted([first["replayed"], second["replayed"]]), [False, True])
        self.assertEqual(await self.counts(), (3, 1))
        changed = await self.create([{"name": "Changed spelling", "domain": "test"}], request)
        self.assertEqual(changed["error"]["code"], "knowledge_area_request_conflict")
        self.assertEqual(await self.counts(), (3, 1))

    async def test_later_request_and_same_name_are_not_silently_merged(self):
        definitions = [{"name": "Same name", "domain": "test"}]
        first, second = await self.create(definitions), await self.create(definitions)
        self.assertNotEqual(first["areas"][0]["id"], second["areas"][0]["id"])
        duplicate = await self.create(definitions * 2)
        self.assertEqual(duplicate["error"]["code"], "knowledge_area_duplicate_input")

    async def test_receipt_failure_rolls_back_the_entire_creation_batch(self):
        with patch.object(self.service, "_record", side_effect=RuntimeError("synthetic receipt failure")):
            with self.assertRaises(RuntimeError):
                await self.create([{"name": "One", "domain": "test"}, {"name": "Two", "domain": "test"}])
        self.assertEqual(await self.counts(), (1, 0))

    async def test_rename_preserves_identity_sources_and_receipt_retries(self):
        source = await self.f.publish("Original project decision")
        area = await self.service.read(self.f.owner, self.f.area)
        request = await self.request()
        result = await self.update(area, request=request, description="A current project description")
        retry = await self.update(area, request=request, description="A current project description")
        self.assertEqual(result["area"]["id"], str(self.f.area))
        self.assertEqual(result["area"]["revision"], area["revision"] + 1)
        self.assertTrue(result["changed"])
        self.assertTrue(retry["replayed"])
        self.assertEqual(retry["applied_areas"], result["applied_areas"])
        actual = await self.f.sources.read(self.f.owner, self.f.area, source.source_id)
        self.assertEqual((actual.document_id, actual.payload_id, actual.generation),
                         (source.document_id, source.payload_id, source.generation))
        conflict = await self.update(area, name="Unexpected alternative", request=request)
        self.assertEqual(conflict["error"]["code"], "knowledge_area_request_conflict")

    async def test_replayed_receipt_separates_applied_revision_from_current_state(self):
        request = await self.request()
        definitions = [{"name": "Original label", "domain": "test"}]
        first = await self.create(definitions, request)
        changed = await self.update(first["areas"][0], name="Current label")
        retry = await self.create(definitions, request)
        self.assertTrue(retry["replayed"])
        self.assertEqual(retry["areas"][0]["name"], "Current label")
        self.assertEqual(retry["areas"][0]["revision"], changed["area"]["revision"])
        self.assertEqual(retry["applied_areas"][0]["revision"], 1)

    async def test_stale_updates_conflict_and_unchanged_metadata_does_not_advance_revision(self):
        area = await self.service.read(self.f.owner, self.f.area)
        unchanged = await self.update(area, name=area["name"])
        self.assertFalse(unchanged["changed"])
        self.assertEqual(unchanged["area"]["revision"], area["revision"])
        await self.update(area)
        stale = await self.update(area)
        self.assertEqual(stale["error"]["code"], "knowledge_area_revision_conflict")

    async def test_metadata_changes_do_not_reactivate_archived_areas(self):
        request = await self.request()
        await self.service.set_status(self.f.owner, self.f.area, "archived", expected_revision=1,
            request=TaskVoiceSource(request["id"], request["revision"], self.sessions[self.f.owner]))
        area = await self.service.read(self.f.owner, self.f.area)
        result = await self.update(area, description="This text does not grant access or activate an area")
        self.assertEqual(result["area"]["status"], "archived")
        with self.assertRaises(SourceError):
            await self.f.sources.current_snapshot(self.f.owner, self.f.area)

    async def test_invalid_request_sources_cannot_create_or_edit_areas(self):
        old = await self.request()
        await self.request(previous=old, finality="revised")
        removed = await self.request()
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where id=$1",
                                     removed["payload_id"])
        area = await self.service.read(self.f.owner, self.f.area)
        for request in (old, removed, await self.request(owner=self.f.other), await self.request(finality="provisional"),
                        await self.request(trust="external_content")):
            with self.subTest(request=request["id"]):
                for result in (await self.create([{"name": "Rejected", "domain": "test"}], request),
                               await self.update(area, request=request)):
                    self.assertEqual(result["error"]["code"], "task_request_evidence_missing")
        self.assertEqual(await self.counts(), (1, 0))

    async def test_foreign_and_deleted_area_ids_are_not_editable_or_resurrected(self):
        async with self.f.pool.acquire() as connection:
            foreign = await connection.fetchval("insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Other','test') returning id", self.f.other)
        result = await self.update({"id": str(foreign), "name": "Other", "domain": "test", "description": "", "revision": 1})
        self.assertEqual(result["error"]["code"], "knowledge_area_unavailable")
        request = await self.request()
        definitions = [{"name": "Deleted later", "domain": "test"}]
        created = await self.create(definitions, request)
        identity = UUID(created["areas"][0]["id"])
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_knowledge_areas set status='deleted',revision=revision+1 where owner_id=$1 and id=$2", self.f.owner, identity)
        replay = await self.create(definitions, request)
        self.assertEqual(replay["error"]["code"], "knowledge_area_unavailable")
        self.assertEqual(await self.counts(), (2, 1))

    async def test_removed_receipt_cannot_recreate_effect_and_request_removal_propagates(self):
        request = await self.request()
        definitions = [{"name": "Receipt removal", "domain": "test"}]
        result = await self.create(definitions, request)
        async with self.f.pool.acquire() as connection:
            payload = await connection.fetchval("select payload_id from jarvis_context_events where id=$1", UUID(result["receipt_event_id"]))
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where id=$1", payload)
        refused = await self.create(definitions, request)
        self.assertEqual(refused["error"]["code"], "knowledge_area_receipt_unavailable")
        self.assertEqual(await self.counts(), (2, 1))
        next_request = await self.request()
        next_result = await self.create(definitions, next_request)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where id=$1", next_request["payload_id"])
            content = await connection.fetchval("""select p.content from jarvis_context_events e
                join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id where e.id=$1""",
                UUID(next_result["receipt_event_id"]))
        self.assertIsNone(content)

    async def test_access_revocation_rolls_back_metadata_and_receipt(self):
        request = await self.request()
        count = 0
        def authorize(owner):
            nonlocal count
            count += 1
            if count == 2:
                raise SourceError("fixture_revoked", "Synthetic access revocation", 403)
        with self.assertRaises(SourceError):
            await self.service.create(self.f.owner, [{"name": "Rollback", "domain": "test"}],
                request=TaskVoiceSource(request["id"], request["revision"], self.sessions[self.f.owner]), access_check=authorize)
        self.assertEqual(await self.counts(), (1, 0))

    async def test_database_description_limit_is_enforced(self):
        async with self.f.pool.acquire() as connection:
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute("insert into jarvis_knowledge_areas(owner_id,name,domain,description) values($1,'Bounded','test',$2)", self.f.owner, "x" * 4001)
