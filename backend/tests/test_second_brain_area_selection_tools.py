"""Conversational area selection uses exact requests, not provider call identity."""
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.knowledge.area_tools import KnowledgeAreaToolModule, audit_area_arguments, audit_area_result
from app.knowledge.models import SourceError
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.voice_tools import BackgroundTaskToolModule
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry


class AreaSelectionContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.session, self.area = "selection-fixture", uuid4(), uuid4()
        pool = object()
        self.sources, self.requests = SimpleNamespace(pool=pool), SimpleNamespace(pool=pool)
        self.service = SimpleNamespace(sources=self.sources, requests=self.requests,
            list=AsyncMock(return_value={"areas": [], "next_after_id": None}))
        self.state = {"area_id": str(self.area), "area_status": "active", "selection_revision": 0}
        self.scopes = SimpleNamespace(sources=self.sources, requests=self.requests,
            read=AsyncMock(return_value=self.state),
            select=AsyncMock(return_value={"selection": self.state, "applied_area_id": str(self.area),
                                           "applied_revision": 1, "replayed": False}))
        self.module = KnowledgeAreaToolModule(self.service, self.owner, selections=self.scopes)
        self.registry = create_tool_registry(modules=(self.module,),
            context=ToolBuildContext(None, self.owner, self.session))
        self.args = {"area_id": str(self.area), "expected_selection_revision": 0}

    async def test_schema_has_only_area_and_separate_strict_selection_counter(self):
        definition = self.registry.get("select_knowledge_area")
        self.assertTrue(definition.mutates_state)
        self.assertEqual(set(definition.input_schema["properties"]), {"area_id", "expected_selection_revision"})
        self.assertFalse(definition.input_schema["additionalProperties"])
        self.assertEqual(definition.input_schema["properties"]["expected_selection_revision"]["minimum"], 0)
        for bad in (True, -1, "0", 0.0, None):
            result = await definition.handler(self.args | {"expected_selection_revision": bad})
            self.assertEqual(result["error"]["code"], "invalid_tool_input")
        self.scopes.select.assert_not_awaited()

    def test_fixed_and_worker_catalogs_do_not_offer_session_selection(self):
        fixed = KnowledgeAreaToolModule(self.service, self.owner)
        for module, session in ((fixed, self.session), (self.module, None)):
            names = {item.name for item in module.build_tools(ToolBuildContext(None, self.owner, session))}
            self.assertNotIn("select_knowledge_area", names)
        self.assertEqual({item.name for item in self.module.build_tools(ToolBuildContext(None, self.owner, None))},
                         {"list_knowledge_areas", "get_knowledge_area",
                          "get_knowledge_area_history", "list_knowledge_area_sources"})

    def test_owner_and_shared_store_are_required(self):
        with self.assertRaises(SourceError):
            self.module.build_tools(ToolBuildContext(None, "foreign-owner", self.session))
        for key in ("sources", "requests"):
            scopes = SimpleNamespace(**vars(self.scopes))
            setattr(scopes, key, SimpleNamespace(pool=object()))
            with self.assertRaises(ValueError):
                KnowledgeAreaToolModule(self.service, self.owner, selections=scopes)

    async def test_listing_exposes_current_selection_without_requiring_user_mutation(self):
        result = await self.registry.get("list_knowledge_areas").handler({})
        self.assertEqual(result["selection"], self.state)
        self.scopes.read.assert_awaited_once_with(self.owner, self.session)
        self.scopes.select.assert_not_awaited()
        fixed = create_tool_registry(modules=(KnowledgeAreaToolModule(self.service, self.owner),),
            context=ToolBuildContext(None, self.owner, self.session))
        self.assertNotIn("selection", await fixed.get("list_knowledge_areas").handler({}))

    async def test_selection_forwards_exact_request_and_trusted_session(self):
        event = uuid4()
        with bind_tool_invocation(provider_call_id="first-native-id", request_event_id=event, request_revision=3):
            result = await self.registry.get("select_knowledge_area").handler(self.args)
        self.assertEqual(result["status"], "success")
        args, kwargs = self.scopes.select.await_args
        self.assertEqual(args, (self.owner, self.area))
        self.assertEqual((kwargs["request"].event_id, kwargs["request"].revision,
                          kwargs["request"].session_id), (event, 3, self.session))
        self.assertEqual(kwargs["expected_revision"], 0)
        kwargs["access_check"](self.owner)
        with self.assertRaises(SourceError):
            kwargs["access_check"]("foreign-owner")

    async def test_worker_and_missing_request_cannot_change_foreground_scope(self):
        for invocation in (SimpleNamespace(task=object(), request_event_id=uuid4(), request_revision=1),
                           SimpleNamespace(task=None, request_event_id=None, request_revision=None)):
            with patch("app.knowledge.area_tools.current_tool_invocation", return_value=invocation):
                result = await self.registry.get("select_knowledge_area").handler(self.args)
            self.assertEqual(result["error"]["code"], "knowledge_request_not_finalized")
        self.scopes.select.assert_not_awaited()

    def test_audit_distinguishes_current_from_applied_without_private_metadata(self):
        result = {"status": "success", "selection": self.state | {"description": "PRIVATE_DESCRIPTION"},
            "applied_area_id": str(uuid4()), "applied_revision": 1, "replayed": True,
            "areas": [{"id": str(self.area), "revision": 5, "status": "active", "name": "PRIVATE_NAME"}]}
        receipt = audit_area_result(result)
        self.assertEqual(receipt["selection"], self.state)
        self.assertEqual(receipt["applied_area_id"], result["applied_area_id"])
        self.assertTrue(receipt["replayed"])
        self.assertNotIn("PRIVATE", json.dumps(receipt))
        self.assertEqual(audit_area_arguments(self.args)["expected_selection_revision"], 0)
        self.assertNotIn("expected_selection_revision",
                         audit_area_arguments(self.args | {"expected_selection_revision": True}))
        self.assertNotIn("replayed", audit_area_result({"status": "success", "replayed": True, "areas": []}))


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested source fixtures required")
class AreaSelectionDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Reuse the accepted real runtime/service composition, not another fake engine.
        from tests.test_second_brain_task_area_scope import TaskAreaDatabaseTests
        await TaskAreaDatabaseTests.asyncSetUp(self)
        self.module = self.runtime.area_tool_module_for(self.f.owner)
        self.registry = self.registry_for(self.session)

    def access(self, owner):
        if owner != self.f.owner:
            raise SourceError("knowledge_owner_mismatch", "The fixture owner does not match.", 403)
        return self.history

    def registry_for(self, session):
        return create_tool_registry(modules=(self.module, BackgroundTaskToolModule(self.service)),
            context=ToolBuildContext(None, self.f.owner, session))

    async def capture(self, text="Select a synthetic knowledge area.", *, session=None, previous=None,
                      finality="final", history=None):
        return await (history or self.history).append(self.f.owner, ContextIngress(
            session_id=session or self.session, source_namespace="voice:area-selection-tools",
            source_event_id=previous.source_event_id if previous else str(uuid4()), receipt_id=str(uuid4()),
            expected_revision=previous.revision if previous else 0, finality=finality,
            kind="user_transcript", trust="user_input", payload={"text": text}))

    async def invoke(self, name, args, *, event=None, registry=None, call_id=None):
        with bind_tool_invocation(provider_call_id=call_id or str(uuid4()),
                request_event_id=event.id if event else None, request_revision=event.revision if event else None):
            return await (registry or self.registry).get(name).handler(args)

    async def select(self, area, *, event=None, expected=None, **kwargs):
        if expected is None:
            current = await self.scopes.read(self.f.owner, self.session)
            expected = current["selection_revision"]
        return await self.invoke("select_knowledge_area",
            {"area_id": str(area), "expected_selection_revision": expected},
            event=event or await self.capture(), **kwargs)

    async def area_for(self, event):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchval("""select area_id from jarvis_knowledge_context_scopes
                where owner_id=$1 and source_namespace=$2 and source_event_id=$3""",
                self.f.owner, event.source_namespace, event.source_event_id)

    async def test_runtime_selection_routes_new_sources_without_moving_history_or_other_sessions(self):
        self.assertIs(self.module.selections, self.scopes)
        listing = await self.invoke("list_knowledge_areas", {})
        self.assertEqual(listing["selection"]["selection_revision"], 0)
        self.assertEqual(listing["selection"]["area_id"], str(self.f.area))
        self.assertNotIn(str(self.foreign), json.dumps(listing))
        old = await self.capture("Earlier personal observation.")
        selected = await self.select(self.project)
        self.assertEqual(selected["status"], "success")
        new = await self.capture("New robotics observation.")
        self.assertEqual(await self.area_for(old), self.f.area)
        self.assertEqual(await self.area_for(new), self.project)
        other = await self.capture("Independent conversation.", session=self.other_session)
        self.assertEqual(await self.area_for(other), self.f.area)
        current = await self.invoke("list_knowledge_areas", {})
        self.assertEqual(current["selection"]["selection_revision"], 1)
        self.assertEqual(current["selection"]["area_id"], str(self.project))
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_background_tasks where owner_id=$1", self.f.owner), 0)

    async def test_provider_retry_recovers_receipt_without_restoring_an_old_selection(self):
        original = await self.capture()
        args = {"event": original, "expected": 0}
        first = await self.select(self.project, **args, call_id="lost-provider-id")
        retry = await self.select(self.project, **args, call_id="new-provider-id")
        self.assertEqual(first["receipt_event_id"], retry["receipt_event_id"])
        self.assertTrue(retry["replayed"])
        await self.select(self.f.area, expected=1)
        late = await self.select(self.project, **args, call_id="third-provider-id")
        self.assertEqual(late["selection"]["area_id"], str(self.f.area))
        self.assertEqual(late["selection"]["selection_revision"], 2)
        self.assertEqual((late["applied_area_id"], late["applied_revision"]), (str(self.project), 1))
        self.assertTrue(late["replayed"])
        conflict = await self.select(self.f.area, **args)
        self.assertEqual(conflict["error"]["code"], "knowledge_selection_request_conflict")
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_session_knowledge_selections where owner_id=$1", self.f.owner), 2)

    async def test_selection_requires_current_finalized_owned_session_request(self):
        args = {"area_id": str(self.project), "expected_selection_revision": 0}
        absent = await self.invoke("select_knowledge_area", args)
        self.assertEqual(absent["error"]["code"], "knowledge_request_not_finalized")
        provisional = await self.capture(finality="provisional")
        self.assertIn("error", await self.invoke("select_knowledge_area", args, event=provisional))
        original = await self.capture("Original transcription.")
        await self.capture("Corrected transcription.", previous=original, finality="revised")
        self.assertIn("error", await self.invoke("select_knowledge_area", args, event=original))
        valid = await self.capture()
        self.assertIn("error", await self.invoke("select_knowledge_area", args, event=valid,
                                                registry=self.registry_for(self.other_session)))
        self.assertIn("error", await self.invoke("select_knowledge_area", args,
                                                event=SimpleNamespace(id=uuid4(), revision=1)))
        with self.assertRaises(SourceError):
            self.module.build_tools(ToolBuildContext(None, self.f.other, self.session))
        self.assertEqual((await self.scopes.read(self.f.owner, self.session))["selection_revision"], 0)
        self.assertEqual((await self.scopes.read(self.f.owner, self.other_session))["selection_revision"], 0)

    async def test_foreign_archived_conflicting_and_older_requests_cannot_change_selection(self):
        self.assertIn("error", await self.select(self.foreign, expected=0))
        archive = await self.invoke("set_knowledge_area_state", {"area_id": str(self.project),
            "state": "archived", "expected_revision": 1}, event=await self.capture("Archive the fixture."))
        self.assertEqual(archive["status"], "success")
        self.assertIn("error", await self.select(self.project, expected=0))
        unarchive = await self.invoke("set_knowledge_area_state", {"area_id": str(self.project),
            "state": "active", "expected_revision": archive["area"]["revision"]},
            event=await self.capture("Reactivate the fixture."))
        self.assertEqual(unarchive["status"], "success")
        wrong_counter = await self.select(self.project, expected=unarchive["area"]["revision"])
        self.assertEqual(wrong_counter["error"]["code"], "knowledge_selection_revision_conflict")
        older = await self.capture("An older unsent selection.")
        self.assertEqual((await self.select(self.project, expected=0))["status"], "success")
        stale = await self.select(self.f.area, event=older, expected=1)
        self.assertEqual(stale["error"]["code"], "knowledge_selection_request_stale")
        self.assertEqual((await self.scopes.read(self.f.owner, self.session))["area_id"], str(self.project))

    async def test_compound_request_selection_governs_new_tasks_but_later_utterances_do_not(self):
        event = await self.capture("Switch to robotics and delegate three independent synthetic tasks.")
        self.assertEqual((await self.select(self.project, event=event, expected=0))["status"], "success")
        args = {"title": "Synthetic analysis", "requested_outcome": "Analyze the synthetic request.", "request_slot": 1}
        first = await self.invoke("start_background_task", args, event=event)
        self.assertEqual(first["status"], "success")
        self.assertEqual(first["task"]["knowledge_area_id"], str(self.project))
        await self.select(self.f.area, expected=1)
        replay = await self.invoke("start_background_task", args, event=event)
        self.assertEqual(replay["task"]["task_ref"], first["task"]["task_ref"])
        self.assertEqual(replay["task"]["knowledge_area_id"], str(self.project))
        second = await self.invoke("start_background_task", args | {"request_slot": 2}, event=event)
        self.assertEqual(second["status"], "success")
        self.assertEqual(second["task"]["knowledge_area_id"], str(self.project))
        explicit = await self.invoke("start_background_task",
            args | {"request_slot": 3, "knowledge_area_id": str(self.f.area)}, event=event)
        self.assertEqual(explicit["status"], "success")
        self.assertEqual(explicit["task"]["knowledge_area_id"], str(self.f.area))
        self.assertEqual(await self.area_for(event), self.f.area)

    async def test_selection_does_not_reassign_a_task_already_accepted_under_the_same_request(self):
        event = await self.capture("Switch areas and delegate two synthetic tasks.")
        args = {"title": "Already admitted", "requested_outcome": "Analyze a synthetic question."}
        first = await self.invoke("start_background_task", args, event=event)
        self.assertEqual(first["task"]["knowledge_area_id"], str(self.f.area))
        await self.select(self.project, event=event, expected=0)
        replay = await self.invoke("start_background_task", args, event=event)
        self.assertEqual(replay["task"]["task_ref"], first["task"]["task_ref"])
        self.assertEqual(replay["task"]["knowledge_area_id"], str(self.f.area))
        second = await self.invoke("start_background_task", args | {"request_slot": 2}, event=event)
        self.assertEqual(second["task"]["knowledge_area_id"], str(self.project))
