"""Offline contract tests: no database, providers, model calls or servers."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.assistant.models import BehaviorProfileRevisionConflict, JarvisBehaviorProfile
from app.assistant.profile_service import BehaviorProfileService
from app.knowledge.models import SourceError
from app.realtime.live_domain_commands import LiveDomainCommands
from app.recordings.voice import LeanRecordingDispatcher
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation, current_tool_invocation


class Tasks:
    def __init__(self):
        self.connection = object()
        self.active = False
        self.owners = []
        self.emit = AsyncMock()

    @asynccontextmanager
    async def transaction(self, owner):
        self.owners.append(owner)
        self.active = True
        try:
            yield self.connection
        finally:
            self.active = False


class Invocation:
    def __init__(self, context):
        self.task = SimpleNamespace(id=uuid4(), owner_id=context.owner_id, session_id=context.jarvis_session_id)
        self.claim = SimpleNamespace(attempt_id=uuid4())
        self.tasks = Tasks()
        self.operation_id, self.request_payload_id = uuid4(), uuid4()
        self.source = SimpleNamespace(payload_id=uuid4())
        self.roots = (self.source.payload_id, self.request_payload_id)
        self.authorize = AsyncMock(return_value=self.roots)


def file_fixture():
    return SimpleNamespace(id=uuid4(), session_id=uuid4(), filename="example.csv", content_type="text/csv",
        storage_bucket="private", storage_path="must-not-be-exposed", size_bytes=7, description="Fixture",
        source="generated", source_visual_frame_id=None, created_at=datetime.now(timezone.utc),
        metadata={}, task_evidence=None,
        memory_source={"document_id": "file:v1", "payload_id": str(uuid4())})


class LiveDomainCommandsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = SimpleNamespace(owner_id="fixture-owner", jarvis_session_id=uuid4())
        self.invocation = Invocation(self.context)
        self.services, self.application = SimpleNamespace(), SimpleNamespace(recording=None)
        self.current = True

    def adapter(self):
        return LiveDomainCommands(self.services, self.application, self.context)

    async def call(self, adapter, name, arguments):
        return await adapter.execute(name, arguments, invocation=self.invocation,
            provider_call_id="actual-provider-call", backend_session_key="actual-hermes-session",
            admit_if=lambda: self.current)

    def presentations(self, *, notes=None, files=None):
        service = SimpleNamespace(resolver=SimpleNamespace(notes=notes, files=files),
            present=AsyncMock(), operation_receipt=AsyncMock(return_value=None))
        self.services.workspace_presentations = service
        return service

    def file_access(self):
        file = file_fixture()
        repository = SimpleNamespace(list_files=AsyncMock(return_value=[file]))
        files = SimpleNamespace(get=AsyncMock(return_value=file), scoped=AsyncMock(return_value=repository))
        self.presentations(files=files)
        return files, file, repository

    def test_no_domain_services_still_offer_explicit_reply_and_notice(self):
        self.assertEqual({tool["name"] for tool in self.adapter().catalog()},
                         {"reply_to_user", "notify_user"})
        self.invocation.tasks.emit.assert_not_awaited()

    def test_removed_note_services_cannot_advertise_tools(self):
        self.services.workspace_notes = SimpleNamespace(read_account_snapshot_in_transaction=AsyncMock(), publish_account_edit=AsyncMock())
        adapter = self.adapter()
        self.assertEqual({item["name"] for item in adapter.catalog()}, {"reply_to_user", "notify_user"})
        catalog = adapter.catalog()
        catalog[0]["parameters"]["properties"]["injected"] = {}
        self.assertNotIn("injected", adapter.catalog()[0]["parameters"]["properties"])

    async def test_owner_or_session_mismatch_never_reaches_domain(self):
        self.application.read_live_state = AsyncMock(return_value=({}, ()))
        adapter = self.adapter()
        for key, wrong in (("owner_id", "other-owner"), ("session_id", uuid4())):
            original = getattr(self.invocation.task, key)
            setattr(self.invocation.task, key, wrong)
            with self.subTest(field=key), self.assertRaises(SourceError):
                await self.call(adapter, "read_live_state", {})
            setattr(self.invocation.task, key, original)
        self.application.read_live_state.assert_not_awaited()

    async def test_stale_route_and_failed_authorization_never_read(self):
        self.application.read_live_state = AsyncMock(return_value=({}, ()))
        adapter = self.adapter()
        self.current = False
        with self.assertRaises(SourceError):
            await self.call(adapter, "read_live_state", {})
        self.invocation.authorize.assert_not_awaited()
        self.current = True
        self.invocation.authorize.side_effect = SourceError("claim_lost", "Changed")
        with self.assertRaises(SourceError):
            await self.call(adapter, "read_live_state", {})
        self.application.read_live_state.assert_not_awaited()

    async def test_preferences_use_existing_revisioned_service(self):
        profile = {"behavior_instructions": "Be concise", "revision": 3}
        service = SimpleNamespace(get=AsyncMock(return_value=profile), update=AsyncMock(return_value=profile))
        self.services.behavior_profile_service = service
        adapter = self.adapter()
        read = await self.call(adapter, "read_preferences", {})
        self.assertEqual(read.result["preferences"], profile)
        result = await self.call(adapter, "edit_preferences",
                                 {"behavior_instructions": " Be concise ", "expected_revision": 2})
        self.assertTrue(result.result["effect_committed"])
        self.assertEqual(service.update.await_args.kwargs["owner_id"], self.context.owner_id)
        self.assertEqual(service.update.await_args.kwargs["update"].expected_revision, 2)
        self.assertEqual(service.update.await_args.kwargs["update"].behavior_instructions, "Be concise")
        service.update.side_effect = BehaviorProfileRevisionConflict(2, 4)
        conflict = await self.call(adapter, "edit_preferences",
                                  {"behavior_instructions": "New", "expected_revision": 2})
        self.assertFalse(conflict.result["effect_committed"])
        self.assertEqual(conflict.result["outcome"], "conflict")
        self.assertEqual(service.update.await_count, 2)

    async def test_real_preference_service_returns_json_dictionaries(self):
        now = datetime.now(timezone.utc)
        profile = JarvisBehaviorProfile(self.context.owner_id, "Be concise", now, now, revision=3)
        repository = SimpleNamespace(get=AsyncMock(return_value=profile), set=AsyncMock(return_value=profile))
        self.services.behavior_profile_service = BehaviorProfileService(repository)
        adapter = self.adapter()
        read = await self.call(adapter, "read_preferences", {})
        edit = await self.call(adapter, "edit_preferences",
                               {"behavior_instructions": "Be concise", "expected_revision": 2})
        for receipt in (read, edit):
            self.assertIs(type(receipt.result), dict)
            self.assertIs(type(receipt.result["preferences"]), dict)
            self.assertEqual(receipt.result["preferences"]["revision"], 3)
            self.assertIs(type(receipt.result["preferences"]["updated_at"]), str)

    async def test_files_are_independent_of_notes_and_do_not_expose_storage(self):
        files, file, repository = self.file_access()
        adapter = self.adapter()
        names = {item["name"] for item in adapter.catalog()}
        self.assertNotIn("read_account_note", names)
        self.assertIn("read_workspace_file", names)
        receipt = await self.call(adapter, "read_workspace_file", {"file_id": str(file.id), "version_id": "file:v1"})
        files.get.assert_awaited_once_with(self.context.owner_id, self.context.jarvis_session_id, file.id, "file:v1")
        self.assertEqual(receipt.source_payload_ids, (UUID(file.memory_source["payload_id"]),))
        self.assertNotIn("storage_path", receipt.result["file"])
        self.assertNotIn("storage_bucket", receipt.result["file"])
        result = await self.call(adapter, "search_workspace_files", {"query": "example", "limit": 1})
        self.assertTrue(result.result["page_full"])
        repository.list_files.assert_awaited_once_with(owner_id=self.context.owner_id, query="example", limit=1)

    def test_recording_files_are_available_without_presentation_or_notes(self):
        access = SimpleNamespace(check=Mock(), files=object())
        self.services.recording_source_access = access
        adapter = self.adapter()
        self.assertEqual({item["name"] for item in adapter.catalog()},
                         {"read_workspace_file", "search_workspace_files", "reply_to_user", "notify_user"})
        access.check.assert_called_once_with(self.context.owner_id)

    async def test_open_presentation_uses_exact_version_and_keeps_pending_receipt(self):
        files, file, _ = self.file_access()
        presentations = self.services.workspace_presentations
        pending = {"outcome": "accepted", "effect_committed": True,
            "artifact": {"kind": "file", "artifact_id": str(file.id), "version_id": "file:v1"},
            "presentation": {"state": "pending"}}
        presentations.present.return_value = pending
        adapter = self.adapter()
        definition = next(item for item in adapter.catalog() if item["name"] == "present_workspace_resource")
        self.assertEqual(definition["parameters"]["properties"]["artifact_kind"]["enum"], ["file"])
        result = await self.call(adapter, "present_workspace_resource",
            {"artifact_kind": "file", "artifact_id": str(file.id), "version_id": "file:v1"})
        self.assertEqual(result.result, pending)
        self.assertEqual(result.source_payload_ids, (UUID(file.memory_source["payload_id"]),))
        args, kwargs = presentations.present.await_args
        self.assertEqual(args[0].operation_id, self.invocation.operation_id)
        self.assertEqual(kwargs, {"action": "open", "artifact_kind": "file", "artifact_id": file.id, "version_id": "file:v1"})
        with self.assertRaises(ValueError):
            await self.call(adapter, "present_workspace_resource",
                {"artifact_kind": "note", "artifact_id": str(uuid4()), "version_id": "note:v1"})
        self.assertEqual(presentations.present.await_count, 1)

    async def test_operation_lookup_never_invents_a_completed_effect(self):
        presentations = self.presentations()
        adapter = self.adapter()
        identity = uuid4()
        missing = await self.call(adapter, "read_workspace_operation", {"domain": "presentation", "operation_id": str(identity)})
        self.assertIsNone(missing.result["effect_committed"])
        self.assertEqual(missing.result["outcome"], "outcome_unknown")
        actual = {"effect_committed": True, "outcome": "completed", "operation_id": str(identity)}
        presentations.operation_receipt.return_value = actual
        result = await self.call(adapter, "read_workspace_operation", {"domain": "presentation", "operation_id": str(identity)})
        self.assertEqual(result.result, actual)
        presentations.present.assert_not_awaited()

    async def test_recording_passes_actual_call_identity_and_preserves_worker_scope(self):
        dispatcher = LeanRecordingDispatcher(owner_id=self.context.owner_id,
            session_id=self.context.jarvis_session_id, admissions=None, policy=None,
            provider_session_key=lambda: "unused", is_active=lambda: True, on_command=lambda value: True)
        self.application.recording = dispatcher
        unconfirmed = {"outcome": "outcome_unknown", "effect_committed": None}
        scope = TaskToolInvocation(self.invocation.task.id, 1, self.invocation.claim.attempt_id)

        async def execute(arguments, **kwargs):
            self.assertIs(current_tool_invocation().task, scope)
            self.assertEqual(kwargs["provider_call_id"], "actual-provider-call")
            self.assertEqual(kwargs["backend_session_key"], "actual-hermes-session")
            self.assertTrue(kwargs["admit_if"]())
            return unconfirmed

        self.application.execute_recording = AsyncMock(side_effect=execute)
        adapter = self.adapter()
        self.assertIn("live_recording", {item["name"] for item in adapter.catalog()})
        with bind_tool_invocation(provider_call_id="outer", task=scope):
            receipt = await self.call(adapter, "live_recording", {"action": "status"})
            self.assertIs(current_tool_invocation().task, scope)
        self.assertEqual(receipt.result, unconfirmed)

    async def test_sandbox_view_only_uses_exact_present_seam(self):
        presentations = self.presentations()
        pending = {"outcome": "accepted", "effect_committed": True, "presentation": {"state": "pending"}}
        viewer = SimpleNamespace(present=AsyncMock(return_value=pending))
        self.services.sandbox_views = viewer
        adapter = self.adapter()
        self.assertIn("view_sandbox", {item["name"] for item in adapter.catalog()})
        task_id, allocation_id = uuid4(), uuid4()
        result = await self.call(adapter, "view_sandbox", {"task_id": str(task_id), "allocation_id": str(allocation_id)})
        self.assertEqual(result.result, pending)
        args, kwargs = viewer.present.await_args
        self.assertIs(args[1], presentations)
        self.assertEqual(args[0].operation_id, self.invocation.operation_id)
        self.assertEqual(kwargs, {"task_id": task_id, "allocation_id": allocation_id})

    async def test_live_state_is_only_advertised_when_backed_and_never_prefetched(self):
        self.assertNotIn("read_live_state", {item["name"] for item in self.adapter().catalog()})
        self.application.read_live_state = AsyncMock(return_value=({"revision": 7}, (uuid4(),)))
        adapter = self.adapter()
        definition = next(item for item in adapter.catalog() if item["name"] == "read_live_state")
        selector = definition["parameters"]["properties"]["section"]
        self.assertEqual(selector["enum"], ["human", "system", "all"])
        self.assertEqual(selector["default"], "all")
        self.assertNotIn("section", definition["parameters"].get("required", []))
        self.assertFalse(definition["parameters"]["additionalProperties"])
        self.application.read_live_state.assert_not_awaited()
        self.invocation.tasks.emit.assert_not_awaited()
        await self.call(adapter, "read_live_state", {})
        self.application.read_live_state.assert_awaited_once_with()
        self.invocation.tasks.emit.assert_not_awaited()

    async def test_explicit_live_state_read_preserves_snapshot_and_original_roots(self):
        state = {"revision": 8, "observations": [{"text": "Keys may be on the desk",
            "observed_at": "2026-11-03T01:00:00Z", "uncertain": True}],
            "memory": {"status": "unavailable"}, "tasks": [{"status": "running"}]}
        roots = (uuid4(), uuid4())

        async def read():
            self.assertFalse(self.invocation.tasks.active)
            self.invocation.authorize.assert_awaited_once_with(self.invocation.tasks.connection)
            return state, roots

        self.application.read_live_state = AsyncMock(side_effect=read)
        receipt = await self.call(self.adapter(), "read_live_state", {})
        self.assertEqual(receipt.result["live_state"], state)
        self.assertEqual(receipt.source_payload_ids, roots)
        self.assertEqual(receipt.result["outcome"], "available")
        self.assertFalse(receipt.result["effect_committed"])
        self.application.read_live_state.assert_awaited_once_with()
        self.invocation.tasks.emit.assert_not_awaited()

    async def test_failed_live_state_read_does_not_invent_an_empty_snapshot(self):
        self.application.read_live_state = AsyncMock(
            side_effect=SourceError("state_unavailable", "Current source evidence is unavailable", 503))
        with self.assertRaises(SourceError):
            await self.call(self.adapter(), "read_live_state", {})
        self.invocation.tasks.emit.assert_not_awaited()

    async def test_section_selection_uses_one_same_ram_read_and_original_roots(self):
        state = {"revision": 8, "human": {"stale": False, "revision": 2},
                 "system": {"reminders": []}}
        roots = (uuid4(),)
        self.application.read_live_state = AsyncMock(return_value=(state, roots))
        receipt = await self.call(self.adapter(), "read_live_state", {"section": "human"})
        self.assertEqual(receipt.result["live_state"],
                         {"revision": 8, "human": {"stale": False, "revision": 2}})
        self.assertEqual(receipt.source_payload_ids, roots)
        self.assertIn("system", state)
        self.application.read_live_state.assert_awaited_once_with()

    async def test_invalid_section_is_rejected_before_authorization_or_read(self):
        self.application.read_live_state = AsyncMock()
        adapter = self.adapter()
        for section in ("history", None, 1):
            with self.subTest(section=section), self.assertRaises(ValueError):
                await self.call(adapter, "read_live_state", {"section": section})
        self.invocation.authorize.assert_not_awaited()
        self.application.read_live_state.assert_not_awaited()

    async def test_live_state_rejects_forged_arguments_and_revoked_claim_before_read(self):
        self.application.read_live_state = AsyncMock(return_value=({}, ()))
        adapter = self.adapter()
        with self.assertRaises(ValueError):
            await self.call(adapter, "read_live_state", {"source_payload_ids": [str(uuid4())]})
        self.invocation.authorize.assert_not_awaited()
        self.invocation.authorize.side_effect = SourceError("claim_lost", "The work is no longer authorized", 403)
        with self.assertRaises(SourceError):
            await self.call(adapter, "read_live_state", {})
        self.application.read_live_state.assert_not_awaited()
        self.invocation.tasks.emit.assert_not_awaited()

    async def test_explicit_reply_and_notice_persist_source_bound_events_not_playback(self):
        self.application.read_live_state = AsyncMock(return_value=({}, ()))
        adapter = self.adapter()

        async def persist(connection, task, event_type, **kwargs):
            self.assertTrue(self.invocation.tasks.active)
            self.assertIs(connection, self.invocation.tasks.connection)
            self.assertIs(task, self.invocation.task)
            self.assertGreater(self.invocation.authorize.await_count, 0)

        self.invocation.tasks.emit.side_effect = persist
        cases = (("reply_to_user", "live_voice_reply", "Which saved file should I use?"),
                 ("notify_user", "live_voice_notice", "You asked to be reminded before leaving."))
        for name, event_type, text in cases:
            with self.subTest(name=name):
                receipt = await self.call(adapter, name, {"text": text})
                self.invocation.tasks.emit.assert_awaited_with(self.invocation.tasks.connection,
                    self.invocation.task, event_type, attempt_id=self.invocation.claim.attempt_id,
                    payload={"text": text, "operation_id": str(self.invocation.operation_id),
                             "agent_selected": True},
                    source_payload_ids=(self.invocation.request_payload_id,))
                self.assertEqual(receipt.result["outcome"], "accepted")
                self.assertFalse(receipt.result["effect_committed"])
                self.assertEqual(receipt.result["operation_id"], str(self.invocation.operation_id))
                self.assertIn("not confirmed", receipt.result["summary"])
        self.assertEqual(self.invocation.tasks.emit.await_count, 2)
        self.assertEqual([call.args[2] for call in self.invocation.tasks.emit.await_args_list],
                         ["live_voice_reply", "live_voice_notice"])
        self.application.read_live_state.assert_not_awaited()

    async def test_explicit_voice_request_validates_text_and_rejects_forged_authority(self):
        adapter = self.adapter()
        invalid = [{"text": ""}, {"text": "x" * 4001}, {"text": "bad\x00text"}, {"text": 123},
                   {"text": "Reply", "operation_id": str(uuid4())},
                   {"text": "Notice", "agent_selected": False}]
        for name in ("reply_to_user", "notify_user"):
            for arguments in invalid:
                with self.subTest(name=name, fields=list(arguments)), self.assertRaises(ValueError):
                    await self.call(adapter, name, arguments)
        self.invocation.authorize.assert_not_awaited()
        self.invocation.tasks.emit.assert_not_awaited()

    async def test_explicit_voice_events_require_current_transport_and_authorized_work(self):
        adapter = self.adapter()
        self.current = False
        for name in ("reply_to_user", "notify_user"):
            with self.subTest(name=name, boundary="transport"), self.assertRaises(SourceError):
                await self.call(adapter, name, {"text": "Message"})
        self.invocation.authorize.assert_not_awaited()
        self.current = True
        self.invocation.authorize.side_effect = SourceError("claim_lost", "Work authority changed", 403)
        for name in ("reply_to_user", "notify_user"):
            with self.subTest(name=name, boundary="claim"), self.assertRaises(SourceError):
                await self.call(adapter, name, {"text": "Message"})
        self.invocation.tasks.emit.assert_not_awaited()

    async def test_durable_voice_event_failure_is_not_reported_as_acceptance(self):
        self.invocation.tasks.emit.side_effect = RuntimeError("Event persistence unavailable")
        adapter = self.adapter()
        for name in ("reply_to_user", "notify_user"):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                await self.call(adapter, name, {"text": "Message"})
        self.assertEqual(self.invocation.tasks.emit.await_count, 2)

    async def test_ordinary_domain_completion_does_not_pull_state_or_publish_progress(self):
        self.application.read_live_state = AsyncMock(return_value=({"revision": 9}, ()))
        adapter = self.adapter()
        with self.assertRaises(ValueError):
            await self.call(adapter, "edit_account_note", {"changes": []})
        self.application.read_live_state.assert_not_awaited()
        self.invocation.tasks.emit.assert_not_awaited()

    async def test_unsupported_command_is_rejected_without_authorization_or_effects(self):
        with self.assertRaises(ValueError):
            await self.call(self.adapter(), "live_note", {"instruction": "Create another note agent"})
        self.invocation.authorize.assert_not_awaited()


from uuid import UUID


if __name__ == "__main__":
    unittest.main()
