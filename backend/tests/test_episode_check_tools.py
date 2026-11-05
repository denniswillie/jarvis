"""Offline real-owner tool qualification; not PostgreSQL or physical acceptance."""
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.live_state.models import SessionScope
from app.reminders.duties import ResponsibilityRepository
from app.reminders.duty_runtime import initialize_responsibilities
from app.reminders.episode_checks import EpisodeCheckOwner
from app.reminders.episode_tools import EpisodeCheckToolModule
from app.tasks.models import TaskError
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.provenance import SourceBoundToolResult, ToolResultCapture
from app.tools.registry import JarvisToolRegistry
from test_episode_check_owner import NOW, Store


class CaptureStore(Store):
    def __init__(self):
        super().__init__()
        self.dependencies = []

    async def executemany(self, query, values):
        self.sql(query)
        assert "jarvis_context_event_dependencies" in query
        self.dependencies.extend(values)


class Capture:
    def __init__(self, store, scope):
        self.store = store
        self.owner_id, self.session_id = scope.owner_id, scope.session_id
        self.repository = SimpleNamespace(publication=None, tasks=self)
        self.roots = ()

    @asynccontextmanager
    async def transaction(self, owner):
        assert owner == self.owner_id
        async with self.store.acquire() as connection, connection.transaction():
            yield connection

    async def bound_tool_outcome(self, connection, call_id, name, result, roots):
        assert connection is self.store and connection.active
        assert call_id and name and result["status"] == "success"
        self.roots = roots
        return SimpleNamespace(payload_id=uuid4(), id=uuid4())


class EpisodeToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = CaptureStore()
        self.scope = SessionScope(owner_id="episode-tools", session_id=uuid4())
        self.source = self.store.event(self.scope.owner_id, self.scope.session_id)
        self.now = NOW
        self.owner = EpisodeCheckOwner(ResponsibilityRepository(self.store), enabled=True,
            clock=lambda: self.now)
        await self.owner.initialize()
        self.settings = SimpleNamespace(RESPONSIVE_VOICE_ENABLED=True)
        self.module = EpisodeCheckToolModule(self.owner)
        self.context = ToolBuildContext(settings=self.settings, owner_id=self.scope.owner_id,
            session_id=self.scope.session_id)
        self.tools = {tool.name: tool for tool in self.module.build_tools(self.context)}
        self.capture = Capture(self.store, self.scope)

    async def invoke(self, name, arguments, *, source=None):
        source = source or self.source
        with bind_tool_invocation(provider_call_id=source.provider_call_id,
                request_event_id=source.request_event_id, request_revision=source.request_revision):
            return await self.tools[name].handler(arguments)

    async def admit(self):
        return await self.invoke("track_episode_check", {
            "spec": {"title": "Leaving check"}, "check": {"question": "Are the keys visible?"}})

    def command(self, result, operation, **changes):
        check = result["check"]
        return dict(responsibility_id=check["canonical_owner_ref"]["source_id"],
            occurrence_id=check["id"], expected_revision=check["canonical_owner_ref"]["revision"],
            expected_check_revision=check["revision"], episode_id=check["episode_or_object_ref"]["source_id"],
            expected_episode_revision=check["episode_or_object_ref"]["revision"], operation=operation, **changes)

    async def capture_result(self, name, result, capture=None):
        return await self.tools[name].result_capture(result,
            ToolResultCapture(capture or self.capture, str(uuid4()), name))

    def test_registration_is_explicit_and_source_capture_is_valid(self):
        registry = JarvisToolRegistry()
        registry.register_many(self.tools.values())
        self.assertEqual({tool.name for tool in registry.list()}, {
            "track_visual_memory", "track_episode_check", "change_episode_check", "list_episode_checks"})
        for session, settings in ((None, self.settings), (self.scope.session_id, None),
                                  (self.scope.session_id, SimpleNamespace(RESPONSIVE_VOICE_ENABLED=False))):
            self.assertEqual(self.module.build_tools(ToolBuildContext(
                settings=settings, owner_id=self.scope.owner_id, session_id=session)), [])

    async def test_admission_requires_actual_captured_request(self):
        result = await self.tools["track_episode_check"].handler({
            "spec": {"title": "Keys"}, "check": {"question": "Keys visible?"}})
        self.assertEqual(result["error"]["code"], "request_evidence_required")
        self.assertFalse(self.store.rules)

    async def test_admit_retry_has_one_occurrence_and_no_placeholder_reminder(self):
        first, replay = await self.admit(), await self.admit()
        self.assertEqual(first["status"], "success")
        self.assertTrue(replay["replayed"])
        self.assertEqual(first["check"]["id"], replay["check"]["id"])
        self.assertEqual(len(self.store.occurrences), 1)
        self.assertFalse(self.store.reminders)
        bound = await self.capture_result("track_episode_check", first)
        self.assertIsInstance(bound, SourceBoundToolResult)
        self.assertEqual(set(self.capture.roots), {self.store.events[self.source.request_event_id]["payload_id"]})
        self.assertEqual(self.store.dependencies[0][2], self.source.request_event_id)

    async def test_report_evidence_uses_original_time_and_retries_stably(self):
        admitted = await self.admit()
        self.now += timedelta(minutes=1)
        report = self.store.event(self.scope.owner_id, self.scope.session_id, at=self.now)
        command = self.command(admitted, "report_done")
        result = await self.invoke("change_episode_check", command, source=report)
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["physical_action_verified"])
        resolution = result["check"]["resolution"]
        self.assertEqual(resolution["basis"], "user_report")
        self.assertEqual(resolution["source_refs"][0]["source"]["source_id"], str(report.request_event_id))
        self.assertEqual(resolution["resolved_at"], result["check"]["evidence_refs"][0]["captured_at"])
        self.now += timedelta(minutes=5)
        replay = await self.invoke("change_episode_check", command, source=report)
        self.assertEqual(replay["status"], "success")
        self.assertTrue(replay["replayed"])
        await self.capture_result("change_episode_check", result)
        self.assertEqual(len(self.capture.roots), 2)

    async def test_model_cannot_supply_resolution_or_fake_report_evidence(self):
        admitted = await self.admit()
        for extra in ({"resolution": {}}, {"captured_at": NOW.isoformat()}, {"source_refs": []}):
            with self.subTest(extra=extra):
                result = await self.invoke("change_episode_check", self.command(admitted, "report_done", **extra))
                self.assertEqual(result["error"]["code"], "invalid_tool_input")
        self.assertEqual(next(iter(self.store.occurrences.values()))["check_lifecycle"], "tracking")

    async def test_cancellation_is_not_resolution(self):
        admitted = await self.admit()
        result = await self.invoke("change_episode_check", self.command(admitted, "cancel"))
        self.assertEqual(result["check"]["lifecycle"], "dismissed")
        self.assertIsNone(result["check"]["resolution"])
        await self.capture_result("change_episode_check", result)

    async def test_list_across_sessions_preserves_original_sources(self):
        admitted = await self.admit()
        later = SessionScope(owner_id=self.scope.owner_id, session_id=uuid4())
        tools = {tool.name: tool for tool in self.module.build_tools(ToolBuildContext(
            settings=self.settings, owner_id=later.owner_id, session_id=later.session_id))}
        result = await tools["list_episode_checks"].handler({})
        self.assertEqual(result["checks"][0]["id"], admitted["check"]["id"])
        capture = Capture(self.store, later)
        await tools["list_episode_checks"].result_capture(result, ToolResultCapture(capture, "later", "list_episode_checks"))
        self.assertEqual(capture.roots, (self.store.events[self.source.request_event_id]["payload_id"],))

    async def test_unavailable_owner_does_not_promise_admission_or_empty_history(self):
        self.owner._unavailable()
        for name, arguments in (("track_episode_check", {"spec": {"title": "Keys"},
                "check": {"question": "Keys visible?"}}), ("list_episode_checks", {})):
            with self.subTest(name=name):
                result = await self.invoke(name, arguments)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["code"], "episode_tracking_unavailable")
        self.assertFalse(self.store.rules)

    async def test_changed_check_and_removed_source_prevent_receipt_copy(self):
        admitted = await self.admit()
        result = await self.invoke("change_episode_check", self.command(admitted, "cancel"))
        with self.assertRaises(Exception):
            await self.capture_result("track_episode_check", admitted)
        self.store.events[self.source.request_event_id]["availability"] = "deleted"
        with self.assertRaises(Exception):
            await self.capture_result("change_episode_check", result)
        self.assertFalse(self.store.dependencies)

    async def test_receipt_scope_and_mutated_json_are_rejected(self):
        result = await self.admit()
        with self.assertRaises(TaskError):
            await self.capture_result("track_episode_check", result,
                Capture(self.store, SessionScope(owner_id="another", session_id=self.scope.session_id)))
        result["physical_action_verified"] = True
        with self.assertRaises(TaskError):
            await self.capture_result("track_episode_check", result)

    async def test_list_validation_is_strict(self):
        for arguments in ({"limit": True}, {"limit": 0}, {"include_inactive": "yes"}, {"owner_id": "other"}):
            with self.subTest(arguments=arguments):
                result = await self.invoke("list_episode_checks", arguments)
                self.assertEqual(result["error"]["code"], "invalid_tool_input")

    async def test_runtime_gates_schema_probe_and_keeps_scheduled_duties_available(self):
        flags = SimpleNamespace(resolve=lambda *a, **k: SimpleNamespace(enabled=True, configured=True, issue=None))
        for enabled, schema in ((False, True), (True, True), (True, False)):
            with self.subTest(enabled=enabled, schema=schema):
                store = CaptureStore()
                store.schema = schema
                services = SimpleNamespace(settings=SimpleNamespace(SUPABASE_DB_URL="fixture", JARVIS_OWNER_ID="fixture",
                    RESPONSIVE_VOICE_ENABLED=enabled), db_pool=store, tool_modules=())
                with patch("app.reminders.duty_runtime.ResponsibilityRuntime.start"):
                    runtime = await initialize_responsibilities(services, flags)
                self.assertIsNotNone(runtime)
                self.assertIsNone(services.responsibility_issue)
                self.assertEqual(runtime.episode_checks.capability.status,
                    "disabled" if not enabled else "available" if schema else "unavailable")
                self.assertEqual([module.name for module in services.tool_modules],
                    ["responsibilities", "episode_checks"] if enabled else ["responsibilities"])
                self.assertEqual(any("limit 0" in q for q in store.queries), enabled)
