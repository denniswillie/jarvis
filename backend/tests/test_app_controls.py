"""Offline shared intents, using existing domain/phone owners as substitutes."""
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from pydantic import ValidationError
from app.app_controls import AppCommand, AppControlRequest, AppControlService
from app.errors import JarvisHTTPError


class AppControlTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.session, self.operation = "owner", uuid4(), uuid4()
        self.connection = SimpleNamespace(fetchval=AsyncMock(return_value=True))

        @asynccontextmanager
        async def acquire():
            yield self.connection

        self.views = SimpleNamespace(cached_inventory=Mock(return_value={"items": []}), current=Mock(return_value=None),
            present=AsyncMock(return_value={"outcome": "accepted", "summary": "Pending display",
                "presentation": {"request_id": str(self.operation), "state": "pending"}}), stop=Mock(return_value=True))
        self.presentations = object()
        self.controls = AppControlService(owner_id=self.owner, pool=SimpleNamespace(acquire=acquire),
            sandbox_views=self.views, presentations=self.presentations)
        self.application = SimpleNamespace(context=SimpleNamespace(owner_id=self.owner,
            jarvis_session_id=self.session, audio_session_id="phone-audio"), is_current=lambda: True,
            is_ready=lambda: True, offer_event=Mock(return_value=True),
            snapshot=Mock(return_value={"recording": {"recording_id": None, "last_client_report": None}}),
            execute_recording=AsyncMock(return_value={"outcome": "admitted", "summary": "Capture unconfirmed", "effect_committed": True}))
        self.controls.bind(self.application)

    async def dispatch(self, command, **extra):
        return await self.controls.dispatch(owner_id=self.owner, session_id=self.session,
                                           operation_id=self.operation, command=command, **extra)

    async def test_recording_forwards_to_existing_dispatcher_without_claiming_capture(self):
        result = await self.dispatch({"action": "recording.start", "include_jarvis_audio": False})
        self.assertEqual(result["outcome"], "admitted")
        args, kwargs = self.application.execute_recording.call_args
        self.assertEqual(args[0], {"action": "start", "include_jarvis_audio": False})
        self.assertEqual(kwargs["provider_call_id"], str(self.operation))
        self.assertEqual(kwargs["backend_session_key"], "app-control:" + str(self.session))
        self.assertTrue(kwargs["admit_if"]())
        self.assertNotIn("saved", result)

    async def test_recording_stop_uses_exact_id(self):
        identity = uuid4()
        await self.dispatch({"action": "recording.stop", "recording_id": str(identity)})
        self.assertEqual(self.application.execute_recording.call_args.args[0],
                         {"action": "stop", "recording_id": str(identity)})

    async def test_disconnected_recording_never_admits(self):
        self.controls.unbind(self.application)
        result = await self.dispatch({"action": "recording.start", "include_jarvis_audio": True})
        self.assertEqual(result["outcome"], "unavailable")
        self.application.execute_recording.assert_not_awaited()

    async def test_stream_start_uses_exact_presentation_and_existing_intent(self):
        allocation = uuid4()
        result = await self.dispatch({"action": "sandbox.start", "allocation_id": str(allocation),
                                      "generation": "a" * 64, "resume": True})
        self.assertEqual(result["outcome"], "accepted")
        call = self.views.present.call_args
        self.assertEqual(call.kwargs, {"allocation_id": allocation, "generation": "a" * 64, "resume": True})
        self.assertIs(call.args[1], self.presentations)
        self.assertEqual(call.args[0].operation_id, self.operation)
        self.application.offer_event.assert_called_once_with({"type": "workspace.presentation.intent",
            "request_id": str(self.operation), "session_id": str(self.session), "audio_session_id": "phone-audio"})

    async def test_completed_presentation_is_not_reopened(self):
        self.views.present.return_value = {"outcome": "completed", "summary": "Previously displayed",
                                           "presentation": {"request_id": str(self.operation), "state": "displayed"}}
        await self.dispatch({"action": "sandbox.start", "allocation_id": str(uuid4()), "generation": "a" * 64})
        self.application.offer_event.assert_not_called()

    async def test_stop_is_exact_server_effect_not_fabricated_phone_ack(self):
        selected = uuid4()
        result = await self.dispatch({"action": "sandbox.stop", "view_id": str(selected)})
        self.views.stop.assert_called_once_with(self.owner, selected)
        self.assertEqual(result["outcome"], "server_view_closed")
        self.assertIn("not confirmed", result["summary"])
        self.assertEqual(self.application.offer_event.call_args.args[0]["view_id"], str(selected))
        self.views.stop.return_value = False
        self.assertEqual((await self.dispatch({"action": "sandbox.stop", "view_id": str(selected)}))["outcome"], "view_not_active")

    async def test_snapshot_is_detached_ram_only(self):
        self.connection.fetchval.reset_mock()
        first = self.controls.snapshot(self.owner, self.session)
        first["recording"]["recording_id"] = "changed"
        self.assertIsNone(self.controls.snapshot(self.owner, self.session)["recording"]["recording_id"])
        self.connection.fetchval.assert_not_awaited()
        self.application.execute_recording.assert_not_awaited()
        self.views.present.assert_not_awaited()

    async def test_foreign_owner_or_session_cannot_dispatch(self):
        with self.assertRaises(JarvisHTTPError):
            await self.controls.dispatch(owner_id="other", session_id=self.session, operation_id=self.operation,
                                         command={"action": "sandbox.stop", "view_id": str(uuid4())})
        self.connection.fetchval.return_value = False
        with self.assertRaises(JarvisHTTPError):
            await self.dispatch({"action": "recording.start", "include_jarvis_audio": True})
        self.application.execute_recording.assert_not_awaited()

    def test_no_implicit_audio_choice_or_cross_action_arguments(self):
        for command in ({"action": "recording.start"}, {"action": "recording.stop"},
                        {"action": "recording.start", "include_jarvis_audio": True, "resume": True},
                        {"action": "sandbox.stop", "view_id": str(uuid4()), "recording_id": str(uuid4())}):
            with self.assertRaises(ValidationError): AppCommand.model_validate(command)
        with self.assertRaises(ValidationError):
            AppControlRequest.model_validate({"owner_id": "forged", "session_id": str(self.session),
                "operation_id": str(self.operation), "command": {"action": "sandbox.stop", "view_id": str(uuid4())}})

    def test_old_connection_close_does_not_remove_replacement(self):
        replacement = SimpleNamespace(**vars(self.application))
        self.controls.bind(replacement)
        self.controls.unbind(self.application)
        self.assertTrue(self.controls.snapshot(self.owner, self.session)["phone_connected"])
        self.application.offer_event.assert_not_called()
