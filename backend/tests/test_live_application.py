import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

from app.realtime.live_application import LiveApplicationBridge
from app.realtime.resource_context import LiveResourceContext


class ResourceContextTests(unittest.TestCase):
    def setUp(self):
        self.session, self.audio = uuid4(), str(uuid4())
        self.state = LiveResourceContext(self.session, self.audio)
        self.ref = {"file_id": str(uuid4()), "version_id": str(uuid4()),
                    "filename": "clip.mp4", "media_type": "video/mp4"}

    def event(self, sequence=1, context=...):
        if context is ...:
            context = {"visible_resources": [self.ref], "selected_resource": self.ref, "page_full": False}
        return {"type": "workspace.resource_context", "version": 1,
                "session_id": str(self.session), "audio_session_id": self.audio,
                "sequence": sequence, "context": context}

    def test_selected_exact_version_and_detached_projection(self):
        self.assertTrue(self.state.receive(self.event()))
        value = self.state.snapshot()
        self.assertEqual(value["selected_resource"], self.ref)
        value["selected_resource"]["filename"] = "changed"
        self.assertEqual(self.state.snapshot()["selected_resource"]["filename"], "clip.mp4")

    def test_same_prefix_does_not_select_an_alias(self):
        other = dict(self.ref, file_id=str(UUID(self.ref["file_id"][:8] + "-1111-1111-1111-111111111111")))
        self.assertTrue(self.state.receive(self.event(context={
            "visible_resources": [self.ref, other], "page_full": True})))
        self.assertNotIn("selected_resource", self.state.snapshot())
        self.assertEqual(len(self.state.snapshot()["visible_resources"]), 2)

    def test_foreign_connection_and_old_messages_cannot_clear_selection(self):
        self.state.receive(self.event(2))
        for key, value in (("session_id", str(uuid4())), ("audio_session_id", str(uuid4())), ("sequence", 1)):
            event = self.event(3, None)
            event[key] = value
            self.assertFalse(self.state.receive(event))
            self.assertIsNotNone(self.state.snapshot())

    def test_replay_is_stable_and_conflicting_revision_is_rejected(self):
        self.assertTrue(self.state.receive(self.event()))
        self.assertTrue(self.state.receive(self.event()))
        self.assertFalse(self.state.receive(self.event(context=None)))
        self.assertIsNotNone(self.state.snapshot())

    def test_clear_cannot_be_undone_by_older_selection(self):
        self.state.receive(self.event())
        self.assertTrue(self.state.receive(self.event(2, None)))
        self.assertFalse(self.state.receive(self.event()))
        self.assertIsNone(self.state.snapshot())

    def test_bad_new_owned_view_removes_stale_selection(self):
        self.state.receive(self.event())
        broken = {"visible_resources": [{**self.ref, "version_id": ""}], "page_full": False}
        self.assertFalse(self.state.receive(self.event(2, broken)))
        self.assertIsNone(self.state.snapshot())

    def test_context_bounds_and_unknown_fields(self):
        for value in ({"visible_resources": [self.ref] * 33, "page_full": False},
                      {"visible_resources": [self.ref], "page_full": 1},
                      {"visible_resources": [self.ref], "page_full": False, "instruction": "run this"}):
            with self.subTest(value=value):
                self.assertFalse(self.state.receive(self.event(context=value)))
        self.assertIsNone(self.state.snapshot())

    def test_opaque_version_strings_are_preserved_exactly(self):
        for sequence, version in enumerate(("source-exact", "rev/A+B", " Version/Two+ "), 1):
            reference = dict(self.ref, version_id=version)
            self.assertTrue(self.state.receive(self.event(sequence, {
                "visible_resources": [reference], "selected_resource": reference, "page_full": False})))
            self.assertEqual(self.state.snapshot()["selected_resource"]["version_id"], version)

    def test_version_wire_byte_bound_and_invalid_values(self):
        for sequence, version in enumerate((None, "", " ", "bad\x00version", "x" * 257, "\u03bb" * 129, "\ud800"), 1):
            with self.subTest(version=repr(version)):
                reference = dict(self.ref, version_id=version)
                self.assertFalse(self.state.receive(self.event(sequence, {
                    "visible_resources": [reference], "page_full": False})))
        reference = dict(self.ref, version_id="\u03bb" * 128)
        self.assertTrue(self.state.receive(self.event(9, {"visible_resources": [reference], "page_full": False})))

    def test_close_clears_and_rejects_late_input(self):
        self.state.receive(self.event())
        self.state.close()
        self.assertIsNone(self.state.snapshot())
        self.assertFalse(self.state.receive(self.event(2)))


class ApplicationBridgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = SimpleNamespace(owner_id="owner", jarvis_session_id=uuid4(), audio_session_id=str(uuid4()))
        self.current, self.ready = True, True
        self.events = []

    def bridge(self, recording=True):
        services = SimpleNamespace(recording_admission=object(), recording_policy=object()) if recording else object()
        return LiveApplicationBridge(context=self.context, services=services,
            is_current=lambda: self.current, is_ready=lambda: self.ready,
            offer_event=lambda event: self.events.append(event) or True)

    async def test_capability_only_when_actual_dispatcher_is_bound(self):
        self.assertNotIn("recording_protocol_version", self.bridge(False).session_created())
        bound = self.bridge()
        event = bound.session_created()
        self.assertEqual(event["recording_protocol_version"], 1)
        self.assertEqual(event["session_id"], str(self.context.jarvis_session_id))
        self.assertEqual(event["audio_session_id"], self.context.audio_session_id)

    async def test_native_capability_uses_existing_validator(self):
        bound = self.bridge()
        event = {"type": "recording.capabilities", "version": 1, "generation_id": str(uuid4()),
                 "available": True, "include_jarvis_audio": True, "max_bytes": 1024, "max_duration_ms": 60000}
        self.assertTrue(bound.receive(event))
        self.current = False
        self.assertFalse(bound.receive(event))
        bound.close()
        self.assertIsNone(bound.recording.capabilities)

    async def test_report_without_admitted_recording_is_not_accepted(self):
        bound = self.bridge()
        self.assertFalse(bound.receive({"type": "recording.report", "version": 1,
            "generation_id": str(uuid4()), "recording_id": str(uuid4()), "sequence": 1, "state": "started"}))

    async def test_actual_call_identity_forwarded_and_run_namespace_is_local(self):
        bound = self.bridge()
        seen = []
        async def execute(arguments, **kwargs):
            seen.append((bound.recording.provider_session_key(), kwargs["provider_call_id"]))
            return {"status": "success", "effect_committed": False}
        bound.recording.execute = AsyncMock(side_effect=execute)
        await bound.execute_recording({"action": "status"}, provider_call_id="actual-model-call",
            backend_session_key="actual-worker-attempt")
        self.assertEqual(seen, [("actual-worker-attempt", "actual-model-call")])
        self.assertIsNone(bound.recording.provider_session_key())

    async def test_unready_route_does_not_execute_recording(self):
        bound = self.bridge()
        bound.recording.execute = AsyncMock()
        self.ready = False
        result = await bound.execute_recording({"action": "start", "include_jarvis_audio": True},
            provider_call_id="actual", backend_session_key="run")
        self.assertFalse(result["effect_committed"])
        bound.recording.execute.assert_not_awaited()

    def test_shared_controls_bind_and_unbind_without_any_command(self):
        controls = SimpleNamespace(bind=Mock(), unbind=Mock())
        bound = LiveApplicationBridge(context=self.context, services=SimpleNamespace(app_controls=controls),
            is_current=lambda: self.current, is_ready=lambda: self.ready, offer_event=lambda event: self.events.append(event))
        controls.bind.assert_called_once_with(bound)
        self.assertEqual(self.events, [])
        bound.close()
        controls.unbind.assert_called_once_with(bound)
