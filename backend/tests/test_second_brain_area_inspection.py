"""Area inspection, removable metadata history, and stable source identity."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import asyncpg

from app.knowledge.areas import KnowledgeAreaService
from app.knowledge.area_tools import KnowledgeAreaToolModule, audit_area_result
from app.knowledge.models import SourceError
from app.migrations.runner import apply_migrations, load_migrations, read_schema_readiness
from app.runtime.feature_config import TaskLimits
from app.tasks.repository import TaskRepository
from app.tools.module import ToolBuildContext
from evals.second_brain.source_fixture import source_fixture


class AreaInspectionContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.area = "inspection-fixture", uuid4()
        self.service = SimpleNamespace(history=AsyncMock(return_value={"history": []}),
                                       memberships=AsyncMock(return_value={"memberships": []}))
        self.module = KnowledgeAreaToolModule(self.service, self.owner)
        self.tools = {t.name: t for t in self.module.build_tools(ToolBuildContext(None, self.owner, None))}

    async def test_inspection_is_read_only_without_model_selected_authority(self):
        for name in ("get_knowledge_area_history", "list_knowledge_area_sources"):
            tool = self.tools[name]
            self.assertFalse(tool.mutates_state)
            self.assertFalse(tool.input_schema["additionalProperties"])
            self.assertNotIn("owner_id", tool.input_schema["properties"])
            self.assertNotIn("session_id", tool.input_schema["properties"])
            self.assertEqual((await tool.handler({"area_id": str(self.area)}))["status"], "success")
        self.service.history.assert_awaited_once_with(self.owner, self.area, after_revision=0, limit=32)
        self.service.memberships.assert_awaited_once_with(self.owner, self.area, after_id=None, limit=32)

    async def test_history_and_membership_arguments_are_bounded_before_service_access(self):
        for name, bad in (("get_knowledge_area_history", {"after_revision": True}),
                         ("get_knowledge_area_history", {"after_revision": -1}),
                         ("list_knowledge_area_sources", {"after_id": "bad"})):
            self.assertEqual((await self.tools[name].handler({"area_id": str(self.area)} | bad))["error"]["code"],
                             "invalid_tool_input")
        for name in ("get_knowledge_area_history", "list_knowledge_area_sources"):
            for limit in (0, 65, True, "2"):
                self.assertEqual((await self.tools[name].handler({"area_id": str(self.area), "limit": limit}))["error"]["code"],
                                 "invalid_tool_input")
        self.service.history.assert_not_awaited()
        self.service.memberships.assert_not_awaited()

    async def test_authorization_is_rechecked_after_inspection(self):
        def deny(owner):
            raise SourceError("fixture_revoked", "Access revoked.", 403)
        module = KnowledgeAreaToolModule(self.service, self.owner)
        tools = {t.name: t for t in module.build_tools(ToolBuildContext(None, self.owner, None))}
        async def revoke(*args, **kwargs):
            object.__setattr__(module, "access_check", deny)
            return {"history": []}
        self.service.history.side_effect = revoke
        result = await tools["get_knowledge_area_history"].handler({"area_id": str(self.area)})
        self.assertEqual(result["error"]["code"], "fixture_revoked")

    def test_effective_policy_is_descriptive_and_follows_actual_visibility(self):
        for status in ("active", "archived"):
            area = KnowledgeAreaService._payload({"id": self.area, "owner_id": self.owner, "status": status, "revision": 1})
            self.assertNotIn("owner_id", area)
            self.assertEqual(area["policy"]["revision"], area["revision"])
            self.assertEqual(area["policy"]["ordinary_source_access_enabled"], status == "active")
            self.assertFalse(area["policy"]["descriptions_grant_authority"])
            self.assertEqual(area["policy"]["scope"], "private_owner")

    def test_history_and_membership_audits_do_not_copy_private_metadata(self):
        source = str(uuid4())
        result = audit_area_result({"status": "success", "history": [{"revision": 2, "status": "active",
            "origin": "updated", "metadata_availability": "available", "metadata": {"name": "PRIVATE"}}],
            "memberships": [{"source_id": source, "revision": 2, "status": "active",
                             "ordinary_content_available": True, "time_basis": "PRIVATE"}]})
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertEqual(result["history"][0]["revision"], 2)
        self.assertEqual(result["memberships"][0]["source_id"], source)

    async def test_service_page_bounds_are_checked_before_database_access(self):
        pool = object()
        service = KnowledgeAreaService(SimpleNamespace(pool=pool), SimpleNamespace(pool=pool))
        for kwargs in ({"after_revision": True}, {"after_revision": -1}, {"limit": 0}, {"limit": 65}):
            with self.assertRaises(ValueError):
                await service.history(self.owner, self.area, **kwargs)
        for kwargs in ({"after_id": "bad"}, {"limit": True}, {"limit": 65}):
            with self.assertRaises(ValueError):
                await service.memberships(self.owner, self.area, **kwargs)


LIVE = bool(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"))


@unittest.skipUnless(LIVE, "Manifested SQL fixtures required")
class AreaInspectionDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from tests.test_second_brain_knowledge_area_metadata import AreaMetadataDatabaseTests
        await AreaMetadataDatabaseTests.asyncSetUp(self)
        for method in ("request", "invoke", "create", "update"):
            setattr(self, method, getattr(AreaMetadataDatabaseTests, method).__get__(self))

    async def history(self, area=None, **kwargs):
        result = await self.tools["get_knowledge_area_history"].handler(
            {"area_id": str(area or self.f.area)} | kwargs)
        self.assertEqual(result["status"], "success")
        return result

    async def members(self, area=None, **kwargs):
        result = await self.tools["list_knowledge_area_sources"].handler(
            {"area_id": str(area or self.f.area)} | kwargs)
        self.assertEqual(result["status"], "success")
        return result

    async def state(self, area, state):
        return await self.invoke("set_knowledge_area_state", {"area_id": area["id"], "state": state,
            "expected_revision": area["revision"]}, await self.request())

    async def test_complete_future_history_pagination_and_visibility_policy(self):
        created = await self.create([{"name": "Bench", "domain": "robotics", "description": "Initial description"}])
        area = created["areas"][0]
        area = (await self.update(area, name="Robot bench", description="Revised description"))["area"]
        archived = (await self.state(area, "archived"))["area"]
        self.assertFalse(archived["policy"]["ordinary_source_access_enabled"])
        area = (await self.state(archived, "active"))["area"]
        unchanged = await self.update(area, name=area["name"], description=area["description"])
        self.assertFalse(unchanged["changed"])
        first = await self.history(area["id"], limit=2)
        self.assertEqual([h["revision"] for h in first["history"]], [1, 2])
        self.assertEqual([h["metadata"]["name"] for h in first["history"]], ["Bench", "Robot bench"])
        self.assertEqual(first["history"][0]["origin"], "created")
        self.assertFalse(first["earlier_history_unavailable"])
        second = await self.history(area["id"], after_revision=first["next_after_revision"], limit=2)
        self.assertEqual([h["status"] for h in second["history"]], ["archived", "active"])
        self.assertIsNone(second["next_after_revision"])
        self.assertTrue(second["area"]["policy"]["ordinary_source_access_enabled"])
        self.assertEqual(second["area"]["revision"], 4)

    async def test_request_removal_erases_its_history_copy_and_area_deletion_erases_all_copies(self):
        area = await self.service.read(self.f.owner, self.f.area)
        request = await self.request()
        updated = await self.update(area, name="Private historical label", request=request)
        self.assertEqual(updated["status"], "success")
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where id=$1",
                                     request["payload_id"])
        history = await self.history()
        self.assertIsNone(history["history"][1]["metadata"])
        self.assertEqual(history["history"][1]["metadata_availability"], "deleted")
        self.assertIsNotNone(history["history"][0]["metadata"])
        async with self.f.pool.acquire() as connection:
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute("update jarvis_knowledge_area_versions set origin='updated' where owner_id=$1", self.f.owner)
            await connection.execute("update jarvis_knowledge_areas set status='deleted',revision=revision+1 where owner_id=$1 and id=$2",
                                     self.f.owner, self.f.area)
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_knowledge_area_versions h
                join jarvis_context_payloads p on p.owner_id=h.owner_id and p.id=h.metadata_payload_id
                where h.owner_id=$1 and p.content is not null""", self.f.owner), 0)
        result = await self.tools["get_knowledge_area_history"].handler({"area_id": str(self.f.area)})
        self.assertEqual(result["error"]["code"], "knowledge_area_unavailable")

    async def test_membership_pages_preserve_tombstones_without_exposing_original_content(self):
        originals = [await self.f.publish("PRIVATE original "+str(i), source_key=str(i)) for i in range(3)]
        await self.f.sources.delete_source(self.f.owner, self.f.area, originals[0].source_id, expected_revision=1)
        changed = await self.f.publish("PRIVATE revised", source_key="1", expected_revision=1)
        first = await self.members(limit=2)
        second = await self.members(after_id=first["next_after_id"], limit=2)
        items = first["memberships"] + second["memberships"]
        self.assertEqual({m["source_id"] for m in items}, {str(s.source_id) for s in originals})
        self.assertNotIn("PRIVATE", json.dumps(items))
        self.assertEqual(next(m for m in items if m["source_id"] == str(changed.source_id))["revision"], 2)
        deleted = next(m for m in items if m["status"] == "deleted")
        self.assertFalse(deleted["ordinary_content_available"])
        area = await self.service.read(self.f.owner, self.f.area)
        archived = (await self.state(area, "archived"))["area"]
        self.assertTrue(all(not m["ordinary_content_available"] for m in (await self.members())["memberships"]))
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.f.area, changed.source_id)
        await self.state(archived, "active")
        self.assertEqual(sum(m["ordinary_content_available"] for m in (await self.members())["memberships"]), 2)

    async def test_inspection_and_cursors_remain_owner_and_area_scoped(self):
        area = (await self.create([{"name": "Other project", "domain": "business"}]))["areas"][0]
        source = await self.f.publish("Owned other area", area_id=UUID(area["id"]))
        async with self.f.pool.acquire() as connection:
            foreign = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Foreign','test') returning id", self.f.other)
        for name in ("get_knowledge_area_history", "list_knowledge_area_sources"):
            result = await self.tools[name].handler({"area_id": str(foreign)})
            self.assertEqual(result["error"]["code"], "knowledge_area_unavailable")
        cursor = await self.tools["list_knowledge_area_sources"].handler(
            {"area_id": str(self.f.area), "after_id": str(source.source_id)})
        self.assertEqual(cursor["error"]["code"], "knowledge_area_source_cursor")
        history = await self.tools["get_knowledge_area_history"].handler(
            {"area_id": str(self.f.area), "after_revision": 999})
        self.assertEqual(history["error"]["code"], "knowledge_area_history_cursor")

    async def test_receipt_failure_rolls_back_area_history_and_private_payload(self):
        request = await self.request()
        async with self.f.pool.acquire() as connection:
            before = await connection.fetchval("select count(*) from jarvis_context_payloads where owner_id=$1", self.f.owner)
        with patch.object(self.service, "_record", side_effect=RuntimeError("synthetic receipt failure")):
            with self.assertRaises(RuntimeError):
                await self.create([{"name": "Must roll back", "domain": "test"}], request)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_context_payloads where owner_id=$1", self.f.owner), before)
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_knowledge_area_versions where owner_id=$1", self.f.owner), 1)

    async def test_concurrent_updates_and_retry_produce_one_revision_snapshot(self):
        area = await self.service.read(self.f.owner, self.f.area)
        requests = [await self.request(), await self.request()]
        results = await asyncio.gather(*(self.update(area, name="Label "+str(i), request=r)
                                         for i, r in enumerate(requests)))
        self.assertEqual(sum(r["status"] == "success" for r in results), 1)
        winner = next(i for i, r in enumerate(results) if r["status"] == "success")
        replay = await self.update(area, name="Label "+str(winner), request=requests[winner])
        self.assertTrue(replay["replayed"])
        self.assertEqual([r["revision"] for r in (await self.history())["history"]], [1, 2])

    async def test_path_move_rename_and_filename_reuse_do_not_merge_source_identity(self):
        key = str(uuid4())
        original = {"filename": "code.py", "path": "old/code.py", "text": "value = 1"}
        first = await self.f.publish_payload(original, source_key=key, source_type="document", trust="external_content")
        moved = await self.f.publish_payload(original | {"filename": "solver.py", "path": "new/solver.py"},
            source_key=key, source_type="document", trust="external_content", expected_revision=1)
        replacement = await self.f.publish_payload(original | {"text": "value = 2"},
            source_key=str(uuid4()), source_type="document", trust="external_content")
        self.assertEqual((first.source_id, moved.source_id), (first.source_id, first.source_id))
        self.assertEqual(moved.revision, 2)
        self.assertNotEqual(replacement.source_id, first.source_id)
        self.assertNotEqual(first.content_digest, moved.content_digest)
        old = await self.f.sources.read(self.f.owner, self.f.area, first.source_id, revision=1)
        self.assertEqual(old.content, original)
        self.assertFalse(old.is_current)
        current = await self.f.sources.read(self.f.owner, self.f.area, first.source_id)
        self.assertEqual(current.content["path"], "new/solver.py")
        memberships = (await self.members())["memberships"]
        self.assertEqual(len(memberships), 2)
        self.assertTrue(all(m["trust"] == "external_content" for m in memberships))
        async with self.f.pool.acquire() as connection:
            operations = await connection.fetch("""select operation,status from jarvis_memory_outbox
                where owner_id=$1 and document_id=$2""", self.f.owner, first.document_id)
        self.assertEqual({(r["operation"], r["status"]) for r in operations},
                         {("retain", "cancelled"), ("delete", "pending")})


@unittest.skipUnless(LIVE, "Manifested SQL fixtures required")
class AreaHistoryMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_upgrade_records_only_known_baseline_and_readiness_requires_history(self):
        migrations = load_migrations()
        self.assertEqual(migrations[-1].version, 33)
        async with source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
                Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id(),
                migrations=migrations[:-1]) as f:
            async with f.pool.acquire() as connection:
                for _ in range(2):
                    await connection.execute("update jarvis_knowledge_areas set name='Before history',revision=revision+1 where owner_id=$1 and id=$2",
                                             f.owner, f.area)
                deleted = await connection.fetchval(
                    "insert into jarvis_knowledge_areas(owner_id,name,domain,status) values($1,'Do not copy','test','deleted') returning id", f.owner)
                schema = f.fixture.manifest["schemas"][0]
                self.assertEqual(await apply_migrations(connection, schema=schema), (33,))
                self.assertEqual(await apply_migrations(connection, schema=schema), ())
                self.assertTrue((await read_schema_readiness(connection, schema=schema)).ready)
                row = await connection.fetchrow("select * from jarvis_knowledge_area_versions where owner_id=$1 and area_id=$2",
                                                f.owner, deleted)
                self.assertIsNone(row["metadata_payload_id"])
            service = KnowledgeAreaService(f.sources, TaskRepository(f.pool, TaskLimits()))
            history = await service.history(f.owner, f.area)
            self.assertEqual(history["first_recorded_revision"], 3)
            self.assertTrue(history["earlier_history_unavailable"])
            self.assertEqual([r["origin"] for r in history["history"]], ["migration_baseline"])
            self.assertEqual(history["history"][0]["metadata"]["name"], "Before history")
            async with f.pool.acquire() as connection:
                await connection.execute("drop table jarvis_knowledge_area_versions cascade")
                state = await read_schema_readiness(connection, schema=schema)
                self.assertIn("missing_table:jarvis_knowledge_area_versions", state.issues)
