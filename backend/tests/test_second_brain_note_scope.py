"""Session defaults route new notes; existing originals and receipts keep identity."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from app.knowledge.models import SourceError
from app.knowledge.notes import SourceNoteRepository, SourceNoteSearchResult
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.notes.models import JarvisNote
from legacy_note_tool_fixture import NoteToolModule, create_note_tools
from app.runtime.feature_config import TaskLimits
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskVoiceSource
from app.tasks.repository import TaskRepository
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.provenance import ToolResultCapture
from evals.second_brain.source_fixture import source_fixture


class NoteScopeContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.area, self.project, self.session = "note-scope-fixture", uuid4(), uuid4(), uuid4()
        self.connection = SimpleNamespace(fetch=AsyncMock(return_value=[]), fetchrow=AsyncMock(),
                                          fetchval=AsyncMock(return_value=False))
        @asynccontextmanager
        async def transaction(owner):
            yield self.connection
        self.sources = SimpleNamespace(pool=object(), transaction=transaction, _area=AsyncMock(),
                                       delete_source_in_transaction=AsyncMock())
        self.selections = SimpleNamespace(sources=self.sources, default_area_id=self.area, _session=AsyncMock(),
            read_in_transaction=AsyncMock(return_value={"area_id":str(self.project), "area_status":"active"}))
        self.root = SourceNoteRepository(self.sources, owner_id=self.owner, area_id=self.area, selections=self.selections)
        self.notes = self.root.for_session(self.session)
        now = datetime.now(UTC)
        self.note = JarvisNote(id=uuid4(), session_id=self.session, title="Fixture", body="Original",
            source="assistant", created_at=now, updated_at=now,
            memory_source={"source_id":str(uuid4()), "area_id":str(self.project)})

    async def test_existing_note_uses_recorded_area_without_reading_current_default(self):
        with patch("app.knowledge.notes.note_sources", new_callable=AsyncMock,
                   return_value=[{"area_id":self.area}]):
            self.assertEqual(await self.notes._area_for(self.connection, self.note.id), self.area)
        self.selections._session.assert_awaited_once_with(self.connection, self.owner, self.session)
        self.selections.read_in_transaction.assert_not_awaited()
        self.assertEqual(self.root.area_id, self.area)
        self.assertIsNone(self.root.session_id)

    async def test_new_note_and_search_use_current_session_selection(self):
        with patch("app.knowledge.notes.note_sources", new_callable=AsyncMock, return_value=[]):
            self.assertEqual(await self.notes._area_for(self.connection, uuid4()), self.project)
        result = await self.notes.search_notes(owner_id=self.owner, query="fixture", limit=8)
        self.assertEqual(result, [])
        self.assertEqual(result.knowledge_scope["area_id"], str(self.project))
        self.assertEqual(self.connection.fetch.call_args.args[3], self.project)
        self.sources._area.assert_awaited_with(self.connection, self.owner, self.project)
        self.selections.read_in_transaction.assert_awaited_with(self.connection, self.owner, self.session)

    async def test_empty_search_scope_survives_tool_payload_conversion(self):
        tools = {t.name:t for t in NoteToolModule(self.root).build_tools(ToolBuildContext(None,self.owner,self.session))}
        result = await tools["search_notes"].handler({"query":"absent"})
        self.assertEqual(result["notes"], [])
        self.assertEqual(result["knowledge_scope"]["area_id"], str(self.project))
        self.selections.read_in_transaction.return_value = {"area_id":str(self.area)}
        later = await tools["search_notes"].handler({})
        self.assertEqual(later["knowledge_scope"]["area_id"], str(self.area))
        self.assertEqual(result["knowledge_scope"]["area_id"], str(self.project))

    async def test_competing_area_mappings_are_not_guessed(self):
        with patch("app.knowledge.notes.note_sources", new_callable=AsyncMock,
                   return_value=[{"area_id":self.area},{"area_id":self.project}]):
            with self.assertRaises(SourceError) as caught:
                await self.notes._area_for(self.connection, self.note.id)
        self.assertEqual(caught.exception.code, "note_source_conflict")
        self.selections.read_in_transaction.assert_not_awaited()

    async def test_create_retry_uses_existing_original_area_in_one_transaction(self):
        self.connection.fetchrow.return_value = {"id":self.note.id}
        with patch.object(self.notes,"_area_for",new_callable=AsyncMock,return_value=self.project), \
             patch("app.knowledge.notes._note_from_record",return_value=self.note), \
             patch.object(self.notes,"_expose",new_callable=AsyncMock,return_value=self.note) as expose:
            result = await self.notes.create_note(owner_id=self.owner,session_id=self.session,
                                                 title=self.note.title,body=self.note.body)
        self.assertEqual(result.id,self.note.id)
        expose.assert_awaited_once_with(self.connection,self.note,area_id=self.project)
        self.sources._area.assert_awaited_with(self.connection,self.owner,self.project)

    async def test_update_and_delete_preserve_existing_area(self):
        self.connection.fetchrow.return_value = {"id":self.note.id}
        with patch.object(self.notes,"_area_for",new_callable=AsyncMock,return_value=self.project), \
             patch("app.knowledge.notes._note_from_record",return_value=self.note), \
             patch.object(self.notes,"_mapping",new_callable=AsyncMock,return_value={"id":uuid4(),"current_revision":1}) as mapping, \
             patch.object(self.notes,"_expose",new_callable=AsyncMock,return_value=self.note) as expose:
            unchanged = await self.notes.update_note(owner_id=self.owner,note_id=self.note.id,title=None,body=self.note.body)
            self.assertEqual(unchanged.id,self.note.id)
            expose.assert_awaited_once_with(self.connection,self.note,area_id=self.project)
            self.connection.fetchval.return_value = True
            self.assertTrue(await self.notes.delete_note(owner_id=self.owner,note_id=self.note.id))
            self.assertEqual(self.sources.delete_source_in_transaction.call_args.args[2], self.project)
            mapping.assert_awaited_with(self.connection,self.note.id,area_id=self.project)

    async def test_result_capture_preserves_empty_and_saved_scope_after_switch(self):
        context = SimpleNamespace(capture=SimpleNamespace(owner_id=self.owner,session_id=self.session,
            repository=SimpleNamespace(pool=self.sources.pool)),call_id="captured",tool_name="search_notes")
        result = {"notes":[],"knowledge_scope":{"area_id":str(self.project)}}
        with patch.object(self.notes.provenance,"bind",new_callable=AsyncMock) as bind:
            await self.notes.capture_tool_result(result,context)
            self.assertEqual(bind.call_args.args[:3],(self.owner,self.project,[]))
            context.tool_name="create_note"
            await self.notes.capture_tool_result({"note":{"memory_source":{"area_id":str(self.area)}}},context)
            self.assertEqual(bind.call_args.args[1],self.area)
        self.selections.read_in_transaction.assert_not_awaited()

    async def test_mixed_or_foreign_session_capture_is_refused(self):
        context = SimpleNamespace(capture=SimpleNamespace(owner_id=self.owner,session_id=uuid4(),
            repository=SimpleNamespace(pool=self.sources.pool)),call_id="bad",tool_name="search_notes")
        result = {"notes":[],"knowledge_scope":{"area_id":str(self.project)}}
        with self.assertRaises(SourceError):
            await self.notes.capture_tool_result(result,context)
        context.capture.session_id=self.session
        result["notes"]=[{"memory_source":{"area_id":str(self.area)}}]
        with self.assertRaises(SourceError) as caught:
            await self.notes.capture_tool_result(result,context)
        self.assertEqual(caught.exception.code,"knowledge_result_scope_mismatch")

    async def test_wrong_owner_or_create_session_is_rejected_before_storage(self):
        for owner,session in (("foreign",self.session),(self.owner,uuid4())):
            with self.assertRaises(SourceError):
                await self.notes.create_note(owner_id=owner,session_id=session,title="No",body="Write")
        self.selections._session.assert_not_awaited()
        self.connection.fetchrow.assert_not_awaited()

    def test_fixed_and_worker_backends_are_not_rebound(self):
        fixed=SourceNoteRepository(self.sources,owner_id=self.owner,area_id=self.area)
        self.assertIs(fixed.for_session(self.session),fixed)
        for changes in ({"sources":SimpleNamespace(pool=object())},{"default_area_id":uuid4()}):
            with self.assertRaises(ValueError):
                SourceNoteRepository(self.sources,owner_id=self.owner,area_id=self.area,
                    selections=SimpleNamespace(**(vars(self.selections)|changes)))
        with self.assertRaises(ValueError):
            SourceNoteRepository(self.sources,owner_id=self.owner,area_id=self.area,
                                 selections=self.selections,task_evidence=object())
        with self.assertRaises(ValueError):
            self.root.for_session("untrusted-id")


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class NoteScopeDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_scoped_note_identity_receipts_history_and_deletion(self):
        async with source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),test_name=self.id()) as f:
            tasks=TaskRepository(f.pool,TaskLimits())
            selections=SessionKnowledgeAreas(f.sources,tasks,f.area)
            history=ContextRepository(tasks,publication=CanonicalSourcePublication(f.sources,f.area,selections))
            root=SourceNoteRepository(f.sources,owner_id=f.owner,area_id=f.area,selections=selections)
            async with f.pool.acquire() as connection:
                first=await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id",f.owner)
                second=await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id",f.owner)
                project=await connection.fetchval(
                    "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Bench','robotics') returning id",f.owner)
                fitness=await connection.fetchval(
                    "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Training','fitness') returning id",f.owner)
            async def request(text):
                return await history.append(f.owner,ContextIngress(session_id=first,source_namespace="voice:note-scope",
                    source_event_id=str(uuid4()),receipt_id=str(uuid4()),kind="user_transcript",trust="user_input",payload={"text":text}))
            async def select(area):
                event=await request("Select a synthetic area.")
                current=await selections.read(f.owner,first)
                await selections.select(f.owner,area,request=TaskVoiceSource(event.id,event.revision,first),
                                        expected_revision=current["selection_revision"])
            notes=root.for_session(first)
            await select(project)
            event=await request("Save the synthetic note.")
            with bind_tool_invocation(provider_call_id="first",request_event_id=event.id,request_revision=event.revision):
                original=await notes.create_note(owner_id=f.owner,session_id=first,title="Bench note",body="Calibration pending.")
            source_id=UUID(original.memory_source["source_id"])
            self.assertEqual(original.memory_source["area_id"],str(project))
            self.assertEqual(await root.for_session(second).search_notes(owner_id=f.owner,query=None,limit=8),[])
            capture=SessionContextCapture(history,owner_id=f.owner,session_id=first,provider="openai",audio_session_id="note-scope")
            tools={t.name:t for t in create_note_tools(repository=root,owner_id=f.owner,session_id=first)}
            found=await tools["search_notes"].handler({"query":"Bench"})
            empty=await tools["search_notes"].handler({"query":"Not present"})
            await select(fitness)
            found_receipt=await notes.capture_tool_result(found,ToolResultCapture(capture,"found","search_notes"))
            empty_receipt=await notes.capture_tool_result(empty,ToolResultCapture(capture,"empty","search_notes"))
            with bind_tool_invocation(provider_call_id="retry-new-id",request_event_id=event.id,request_revision=event.revision):
                replay=await notes.create_note(owner_id=f.owner,session_id=first,title="Bench note",body="Calibration pending.")
            self.assertEqual(replay.id,original.id)
            self.assertEqual(replay.memory_source["area_id"],str(project))
            self.assertEqual((await tools["search_notes"].handler({}))["notes"],[])
            async with f.pool.acquire() as connection:
                for receipt in (found_receipt,empty_receipt):
                    self.assertEqual(await connection.fetchval("""select s.area_id from jarvis_knowledge_context_sources cs
                        join jarvis_knowledge_sources s on s.owner_id=cs.owner_id and s.id=cs.source_id
                        where cs.owner_id=$1 and cs.event_id=$2""",f.owner,receipt.event_id),project)
                self.assertEqual(await connection.fetchval("select count(*) from jarvis_notes where owner_id=$1",f.owner),1)
            revised=await notes.update_note(owner_id=f.owner,note_id=original.id,title=None,body="Calibration paused, not completed.")
            self.assertEqual(revised.memory_source["source_id"],str(source_id))
            self.assertEqual(revised.memory_source["area_id"],str(project))
            self.assertEqual(revised.memory_source["revision"],2)
            self.assertEqual((await f.sources.read(f.owner,project,source_id,revision=1)).content["record"]["body"],original.body)
            async with f.pool.acquire() as connection:
                self.assertNotEqual(await connection.fetchval("select availability from jarvis_context_payloads where owner_id=$1 and id=$2",
                                                              f.owner,found_receipt.payload_id),"available")
                await connection.execute("update jarvis_knowledge_areas set status='archived',revision=revision+1 where owner_id=$1 and id=$2",
                                         f.owner,project)
            self.assertTrue(await notes.delete_note(owner_id=f.owner,note_id=original.id))
            with bind_tool_invocation(provider_call_id="retry-after-delete",request_event_id=event.id,request_revision=event.revision):
                with self.assertRaises(SourceError):
                    await notes.create_note(owner_id=f.owner,session_id=first,title="Bench note",body="Calibration pending.")
            await capture.close()
