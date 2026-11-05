"""Link popup contracts through the shared Phone/Voice/Hermes dispatcher."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
from fastapi import FastAPI
from pydantic import ValidationError

from app.app_controls import AppCommand, AppControlService, router
from app.auth.static_bearer import require_static_bearer
from app.errors import JarvisHTTPError
from app.live_state.companion_tools import CompanionTools


class AppLinkTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.session, self.operation = "link-owner", uuid4(), uuid4()
        self.command = {"action": "show_link", "url": "https://example.com/tickets?day=friday",
                        "description": "The show you asked about."}
        self.connection = SimpleNamespace(fetchval=AsyncMock(return_value=True))

        @asynccontextmanager
        async def acquire():
            yield self.connection

        self.controls = AppControlService(owner_id=self.owner, pool=SimpleNamespace(acquire=acquire),
                                          sandbox_views=None, presentations=None)
        self.offered = asyncio.Event()
        self.application = SimpleNamespace(
            context=SimpleNamespace(owner_id=self.owner, jarvis_session_id=self.session, audio_session_id="audio-1"),
            is_current=lambda: True, is_ready=lambda: True, snapshot=lambda: {},
            execute_recording=AsyncMock(), offer_event=Mock(side_effect=self.offer))
        self.controls.bind(self.application)
        self.pending = []

    def offer(self, event):
        self.offered.set()
        return True

    async def asyncTearDown(self):
        self.controls.unbind(self.application)
        for task in self.pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.pending, return_exceptions=True)

    async def dispatch(self, **overrides):
        return await self.controls.dispatch(owner_id=self.owner, session_id=self.session,
                                            operation_id=self.operation, command=self.command | overrides)

    async def start(self):
        task = asyncio.create_task(self.dispatch())
        self.pending.append(task)
        await asyncio.wait_for(self.offered.wait(), timeout=1)
        return task

    def ack(self, outcome="displayed"):
        return self.controls.acknowledge_link(self.owner, self.session, self.operation, outcome)

    async def test_phone_event_and_confirmed_display_without_recording_or_browser_action(self):
        task = await self.start()
        self.assertFalse(task.done())
        self.application.offer_event.assert_called_once_with({
            "type": "app.link", "session_id": str(self.session), "audio_session_id": "audio-1",
            "operation_id": str(self.operation), "url": self.command["url"],
            "description": self.command["description"]})
        self.ack()
        result = await task
        self.assertEqual(result["outcome"], "link_displayed")
        self.assertTrue(result["effect_committed"])
        self.application.execute_recording.assert_not_awaited()

    async def test_duplicate_request_and_ack_do_not_present_twice(self):
        task = await self.start()
        self.ack()
        expected = await task
        self.assertEqual(await self.dispatch(), expected)
        self.assertEqual(self.ack(), expected)
        self.application.offer_event.assert_called_once()
        with self.assertRaises(JarvisHTTPError) as error:
            await self.dispatch(url="https://example.com/different")
        self.assertEqual(error.exception.status_code, 409)

    async def test_no_ack_returns_unconfirmed_then_late_ack_updates_state(self):
        result = await self.dispatch()
        self.assertEqual(result["outcome"], "link_sent")
        self.assertIsNone(result["effect_committed"])
        self.ack()
        self.assertEqual(self.controls.snapshot(self.owner, self.session)["last_command"]["outcome"], "link_displayed")
        self.assertEqual((await self.dispatch())["outcome"], "link_displayed")
        self.application.offer_event.assert_called_once()

    async def test_unavailable_phone_never_claims_display(self):
        self.controls.unbind(self.application)
        result = await self.dispatch()
        self.assertEqual(result["outcome"], "unavailable")
        self.application.offer_event.assert_not_called()

    async def test_full_phone_queue_does_not_wait_or_claim_display(self):
        self.application.offer_event.side_effect = None
        self.application.offer_event.return_value = False
        self.assertEqual((await self.dispatch())["outcome"], "unavailable")

    async def test_disconnect_releases_pending_request(self):
        task = await self.start()
        self.controls.unbind(self.application)
        self.assertEqual((await task)["outcome"], "unavailable")

    async def test_replacement_connection_cannot_ack_old_request(self):
        task = await self.start()
        replacement = SimpleNamespace(**vars(self.application))
        self.controls.bind(replacement)
        self.assertEqual((await task)["outcome"], "unavailable")
        self.assertEqual(self.ack()["outcome"], "unavailable")
        self.controls.unbind(replacement)

    async def test_wrong_owner_session_and_operation_cannot_ack(self):
        task = await self.start()
        for owner, session, operation in (("other", self.session, self.operation),
                                          (self.owner, uuid4(), self.operation),
                                          (self.owner, self.session, uuid4())):
            with self.assertRaises(JarvisHTTPError):
                self.controls.acknowledge_link(owner, session, operation, "displayed")
        self.assertFalse(task.done())
        self.ack("unavailable")
        self.assertEqual((await task)["outcome"], "unavailable")

    async def test_foreign_session_cannot_send(self):
        self.connection.fetchval.return_value = False
        with self.assertRaises(JarvisHTTPError):
            await self.dispatch()
        self.application.offer_event.assert_not_called()

    async def test_authenticated_ack_endpoint_matches_phone_contract(self):
        app = FastAPI()
        app.state.runtime_services = SimpleNamespace(app_controls=self.controls)
        app.dependency_overrides[require_static_bearer] = lambda: SimpleNamespace(owner_id=self.owner)
        app.include_router(router)
        task = await self.start()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            body = {"session_id": str(self.session), "operation_id": str(self.operation), "outcome": "displayed"}
            invalid = await client.post("/app-controls/link-ack", json=body | {"outcome": "sent"})
            self.assertEqual(invalid.status_code, 422)
            response = await client.post("/app-controls/link-ack", json=body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["outcome"], "link_displayed")
        self.assertEqual((await task)["outcome"], "link_displayed")

    async def test_shared_companion_tool_reaches_same_dispatcher(self):
        tools = CompanionTools(SimpleNamespace(owner_id=self.owner), app_controls=self.controls)
        schema = next(item for item in tools.voice_tools() if item["name"] == "control_app")["parameters"]
        self.assertIn("show_link", schema["properties"]["action"]["enum"])
        self.assertIn("url", schema["required"])
        task = asyncio.create_task(tools.invoke("control_app", self.command, owner_id=self.owner,
                                                session_id=self.session, operation_id=self.operation))
        self.pending.append(task)
        await asyncio.wait_for(self.offered.wait(), timeout=1)
        self.ack()
        self.assertEqual((await task)["outcome"], "link_displayed")

    async def start_agent_link(self):
        self.controls.send_link_notice = AsyncMock(return_value={
            "status": "provider_received", "provider_received": True})
        task = asyncio.create_task(self.controls.dispatch(owner_id=self.owner,
            session_id=self.session, operation_id=self.operation, command=self.command,
            notify_voice=True))
        self.pending.append(task)
        await asyncio.wait_for(self.offered.wait(), timeout=1)
        return task

    async def test_agent_confirmation_waits_for_phone_and_duplicate_ack_does_not_repeat(self):
        task = await self.start_agent_link()
        self.controls.send_link_notice.assert_not_called()
        self.ack()
        entry = self.controls._link_requests[self.operation]
        await entry.notice_task
        self.ack()
        result = await task
        self.assertEqual(result["outcome"], "link_displayed")
        self.assertEqual(result["voice_confirmation"]["owner"], "backend")
        self.controls.send_link_notice.assert_awaited_once_with(self.owner, entry.notice)
        self.assertEqual(entry.notice["kind"], "app_action_result")
        self.assertEqual(entry.notice["audio_session_id"], "audio-1")
        self.assertTrue(self.controls.notice_current(entry.notice))

    async def test_direct_luna_path_does_not_add_another_voice_confirmation(self):
        self.controls.send_link_notice = AsyncMock()
        task = await self.start()
        self.ack()
        self.assertEqual((await task)["outcome"], "link_displayed")
        self.controls.send_link_notice.assert_not_called()

    async def test_late_ack_still_notifies_without_repeating_the_phone_command(self):
        task = await self.start_agent_link()
        result = await task
        self.assertEqual(result["outcome"], "link_sent")
        self.controls.send_link_notice.assert_not_called()
        self.ack()
        await self.controls._link_requests[self.operation].notice_task
        self.application.offer_event.assert_called_once()
        self.controls.send_link_notice.assert_awaited_once()

    async def test_phone_unavailable_is_not_announced_as_displayed(self):
        task = await self.start_agent_link()
        self.ack("unavailable")
        entry = self.controls._link_requests[self.operation]
        await entry.notice_task
        self.assertEqual((await task)["outcome"], "unavailable")
        self.assertIn("could not display", entry.notice["text"])

    async def test_disconnected_phone_receipt_is_not_current(self):
        task = await self.start_agent_link()
        self.ack()
        entry = self.controls._link_requests[self.operation]
        self.controls.unbind(self.application)
        await asyncio.gather(entry.notice_task, return_exceptions=True)
        await task
        self.assertFalse(self.controls.notice_current(entry.notice))
        self.controls.send_link_notice.assert_not_called()

    async def test_voice_delivery_failure_does_not_replay_or_change_phone_success(self):
        task = await self.start_agent_link()
        self.controls.send_link_notice.side_effect = RuntimeError("connection lost")
        self.ack()
        entry = self.controls._link_requests[self.operation]
        await entry.notice_task
        self.assertEqual((await task)["outcome"], "link_displayed")
        self.assertEqual(entry.notice_status, "unknown")
        self.ack()
        self.controls.send_link_notice.assert_awaited_once()

    async def test_agent_tool_registration_enables_backend_confirmation(self):
        from app.tools.module import ToolBuildContext

        self.controls.send_link_notice = AsyncMock(return_value={"status": "provider_received"})
        tools = CompanionTools(SimpleNamespace(owner_id=self.owner), app_controls=self.controls)
        tool = next(item for item in tools.build_tools(ToolBuildContext(None, self.owner, self.session))
                    if item.name == "control_app")
        task = asyncio.create_task(tool.handler(self.command))
        self.pending.append(task)
        await asyncio.wait_for(self.offered.wait(), timeout=1)
        operation = next(iter(self.controls._link_requests))
        self.controls.acknowledge_link(self.owner, self.session, operation, "displayed")
        await self.controls._link_requests[operation].notice_task
        self.assertEqual((await task)["voice_confirmation"]["owner"], "backend")
        self.controls.send_link_notice.assert_awaited_once()

    def test_only_http_links_with_exact_action_fields(self):
        for url in ("https://example.com/a?b=1#details", "http://localhost:9000/page"):
            self.assertEqual(AppCommand.model_validate(self.command | {"url": url}).url, url)
        for url in ("javascript:alert(1)", "file:///tmp/a", "ftp://example.com", "https:///missing-host",
                    "https://user:password@example.com", "https://example.com\n", "https://example.com/a b",
                    "https://example.com\\path", "https://example.com:invalid", "https://example.com:99999"):
            with self.subTest(url=url), self.assertRaises(ValidationError):
                AppCommand.model_validate(self.command | {"url": url})
        for extra in ({"description": "  "}, {"url": None}, {"description": "x" * 1001},
                      {"include_jarvis_audio": True}, {"resume": True}, {"media_id": str(uuid4())}):
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                AppCommand.model_validate(self.command | extra)
        with self.assertRaises(ValidationError):
            AppCommand.model_validate({"action": "media.start", "url": "https://example.com"})
