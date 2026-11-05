"""Offline HTTP -> retained host -> actual Live session/sender/ACK proof.

Native execution and provider HTTP/WebSocket are fake. No live model, sandbox,
microphone, speaker, real profile, or heard-speech claim is involved.
"""
import asyncio
from datetime import timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
from fastapi import FastAPI

from app.hermes_sessions.background_events import BackgroundEvents, background_wake_router
from app.hermes_sessions.bindings import SessionBindings
from app.hermes_sessions.host import RetainedProfileHost
from app.hermes_sessions.sandbox_bindings import SessionSandboxBindings
from app.hermes_sessions.service import RetainedSessionService
from app.hermes_sessions.serving import _connect_master_notices
from app.hermes_sessions.work import WorkDeliveries, WorkSessionBridge
from app.realtime.live_retained import RetainedLiveWork
from app.realtime.live_session import LiveAudioSession
from app.realtime.session_manager import RealtimeSessionManager
from app.sandbox.models import SandboxPolicy
from test_live_voice_adapter import LiveFixture
from test_service import NativeFixture


class AsyncNative(NativeFixture):
    def __init__(self, home):
        super().__init__(home)
        self.results, self.reads, self.read_counts = {}, {}, {}
        self.change_sequence = 0
        self.changed = asyncio.Condition()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def request(self, method, **params):
        if method == "prompt.receipt":
            admission = params["admission_id"]
            result = self.results.get(admission, {"status": "started"})
            self.reads[admission] = result
            self.read_counts[admission] = self.read_counts.get(admission, 0) + 1
            return result
        return await super().request(method, **params)

    async def finish(self, admission, text, *, status="terminal"):
        self.results[admission] = {"status": status, "outcome": "completed",
            "result": {"final_response": text, "private_reasoning": "never sent"}}
        async with self.changed:
            self.change_sequence += 1
            self.changed.notify_all()

    async def wait_for_change(self, after, *, timeout=None):
        async with asyncio.timeout(timeout):
            async with self.changed:
                await self.changed.wait_for(lambda: self.change_sequence > after)
                return self.change_sequence


class ReadySession(LiveAudioSession):
    @property
    def ready(self):
        return self.fixture_ready


class RetainedMasterNoticeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jarvis-master-notice-test-")
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name).resolve()
        self.f = LiveFixture()
        self.provider_events = []
        receive = self.f.wire.socket.recv

        async def recorded_receive():
            raw = await receive()
            self.provider_events.append(json.loads(raw))
            return raw

        self.f.wire.socket.recv = recorded_receive
        await self.f.start()
        self.addAsyncCleanup(self.f.close)
        owner = self.f.identity.scope.owner_id
        bindings = SessionBindings(self.home / "bindings.sqlite", owner_id=owner, profile_home=self.home)
        self.native = AsyncNative(self.home)
        self.service = RetainedSessionService(self.native, bindings, launch={"source": "cli", "toolsets": []})
        self.master = await self.service.master()
        self.bridge = WorkSessionBridge(self.service, WorkDeliveries(self.home / "work.sqlite", bindings))
        self.resources = SessionSandboxBindings(self.home / "resources.sqlite", owner_id=owner, profile_id="fixture")
        self.addCleanup(self.resources.close)
        record = self.resources.reserve(self.master.logical_id, SandboxPolicy())
        self.resources.start_creation(record)
        self.resources.bind(record, "fake-sandbox")
        self.events = BackgroundEvents(sessions=self.service, sandbox_bindings=self.resources, signing_key=b"x" * 32)
        self.host = RetainedProfileHost(self.home, poll_seconds=.02, retry_seconds=.02,
            client_factory=lambda _: self.native, bridge_factory=lambda *_: self.bridge)
        self.manager = RealtimeSessionManager()
        self.runtime = _connect_master_notices(NS(bridge=self.bridge, events=self.events, host=self.host,
            wait_ready=self.host.wait_ready), NS(realtime_session_manager=self.manager))
        self.notices = self.runtime.master_notices
        self.notices.clock = lambda: self.f.now
        self.session = object.__new__(ReadySession)
        self.session.fixture_ready = True
        self.session.context = NS(owner_id=owner)
        self.session.services = NS(hermes_profile_runtime=self.runtime, companion_state=None)
        self.session.adapter = self.f.route
        self.session.request = NS(route=self.f.identity)
        await self.register()
        app = FastAPI()
        app.include_router(background_wake_router(lambda: self.events))
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture")
        self.addAsyncCleanup(self.http.aclose)
        self.stop = asyncio.Event()
        self.host_task = asyncio.create_task(self.host.run(self.stop))
        self.addAsyncCleanup(self.stop_host)
        await self.host.wait_ready(timeout=1)

    async def register(self):
        await self.manager.register(audio_session_id="fixture-live", owner_id=self.session.context.owner_id,
            jarvis_session_id=self.f.identity.scope.session_id, session=self.session)

    async def stop_host(self):
        self.stop.set()
        await asyncio.wait_for(self.host_task, timeout=1)

    async def wake(self, event_id="wake-one"):
        response = await self.http.post("/hermes/background-wake",
            headers={"Authorization": "Bearer " + self.events.credential(self.master.logical_id)},
            json={"session_id": self.master.logical_id, "event_id": event_id, "message": "Untrusted raw callback"})
        self.assertEqual(response.status_code, 202)
        return response.json()

    def appends(self):
        return [event for event in self.f.wire.socket.sent if event["type"] == "session.commentary.append"]

    async def test_actual_async_http_wake_waits_for_master_final_and_receives_fake_provider_ack_once(self):
        receipt = await self.wake()
        admission = receipt["admission_id"]
        await self.f.until(lambda: admission in self.native.reads)
        self.assertEqual(self.appends(), [])
        await self.native.finish(admission, "Partial answer", status="started")
        await self.f.until(lambda: self.native.reads.get(admission) == self.native.results[admission])
        self.assertEqual(self.appends(), [])
        await self.native.finish(admission, "The background result is ready.")
        await self.f.until(lambda: self.notices.last_status == "provider_received")
        self.assertEqual(receipt, await self.wake())
        await self.notices.pump(self.bridge, [])
        sent = self.appends()
        self.assertEqual(len(sent), 1)
        self.assertIsNone(sent[0]["delegation_id"])
        self.assertEqual(json.loads(sent[0]["content"]),
            {"kind": "hermes_completion", "text": "The background result is ready."})
        acknowledgments = [event for event in self.provider_events if event["type"] == "session.commentary.appended"]
        self.assertEqual(len(acknowledgments), 1)
        self.assertEqual(acknowledgments[0]["client_event_id"], sent[0]["event_id"])
        self.assertEqual(len(self.native.inputs), 1)

    async def test_terminal_empty_or_non_string_replies_retire_without_notice_or_further_polling(self):
        for index, text in enumerate(("", "  ", None, 17)):
            with self.subTest(text=text):
                event_id = f"no-answer-{index}"
                receipt = await self.wake(event_id)
                admission = receipt["admission_id"]
                await self.native.finish(admission, text)
                await self.f.until(lambda: admission not in self.notices._pending)
                reads = self.native.read_counts[admission]
                self.assertEqual(receipt, await self.wake(event_id))
                await self.notices.pump(self.bridge, [])
                self.assertEqual(self.native.read_counts[admission], reads)
                self.assertNotIn(admission, self.notices._pending)
                self.assertEqual(self.appends(), [])
        self.assertIsNone(self.notices.last_status)

    async def test_child_work_completion_waits_for_separate_master_answer(self):
        child = await self.bridge.tool(self.master.native_id, "create-one",
            {"action": "create", "label": "Offline child", "message": "Do fixture work"})
        await self.native.finish(child["admission_id"], "Raw child result")
        await self.f.until(lambda: bool(self.notices._pending))
        parent = next(iter(self.notices._pending.values()))
        self.assertEqual(parent.session_id, self.master.logical_id)
        self.assertNotEqual(parent.admission_id, child["admission_id"])
        self.assertEqual(self.appends(), [])
        await self.native.finish(parent.admission_id, "I finished the requested work.")
        await self.f.until(lambda: self.notices.last_status == "provider_received")
        await self.notices.pump(self.bridge, [])
        self.assertEqual(len(self.appends()), 1)
        self.assertEqual(json.loads(self.appends()[0]["content"])["text"], "I finished the requested work.")
        self.assertEqual(self.bridge.deliveries.pending(), [])

    async def test_voice_delegated_admission_stays_on_its_original_watcher(self):
        voice = RetainedLiveWork(self.session, self.runtime)
        self.addAsyncCleanup(voice.close)
        publish = AsyncMock()
        packet = NS(route=self.f.identity, window=NS(request=NS(delegation_id="voice-one")),
            as_context=lambda: {"dialogue": [{"context_event_id": "utterance-one", "sequence": 1,
                "speaker": "user", "delta": "Do the requested work"}], "window": {}})
        await voice(packet, publish=publish, is_current=lambda: True)
        admission = voice.last_receipt["admission_id"]
        await self.native.finish(admission, "Requested voice reply")
        await self.f.until(lambda: publish.await_count == 1)
        await self.notices.pump(self.bridge, [])
        self.assertEqual(publish.await_args.args[0].text, "Requested voice reply")
        self.assertNotIn(admission, self.notices._seen)
        self.assertEqual(self.appends(), [])

    async def test_missing_or_unready_owned_voice_never_sends_to_another_owner_or_replays(self):
        foreign = NS(ready=True, speak_companion_notice=AsyncMock())
        await self.manager.register(audio_session_id="foreign", owner_id="other", jarvis_session_id=uuid4(), session=foreign)
        self.session.fixture_ready = False
        receipt = await self.wake()
        await self.native.finish(receipt["admission_id"], "No current recipient")
        await self.f.until(lambda: self.notices.last_status == "unavailable")
        self.session.fixture_ready = True
        await self.wake()
        await self.notices.pump(self.bridge, [])
        foreign.speak_companion_notice.assert_not_awaited()
        self.assertEqual(self.appends(), [])

    async def test_child_background_admission_is_not_directly_spoken(self):
        child = await self.service.create_work(self.master.logical_id, request_id="child", label="Child")
        receipt = await self.service.send(child.logical_id, input_id="child-event", text="Child event")
        self.events.on_admitted(receipt)
        await self.native.finish(receipt.admission_id, "Child-only answer")
        await self.notices.pump(self.bridge, [])
        self.assertEqual(self.notices._pending, {})
        self.assertEqual(self.appends(), [])

    async def test_notice_guard_rejects_unregistered_completion(self):
        result = await self.session.speak_companion_notice({"id": str(uuid4()), "kind": "hermes_completion",
            "text": "Not admitted", "expires_at": (self.f.now + timedelta(seconds=30)).isoformat()})
        self.assertFalse(result["provider_received"])
        self.assertEqual(self.appends(), [])

    async def test_delivery_exception_does_not_restart_host_or_replay_uncertain_write(self):
        self.session.speak_companion_notice = AsyncMock(side_effect=OSError("unconfirmed transport"))
        receipt = await self.wake()
        await self.native.finish(receipt["admission_id"], "Actual final answer")
        await self.f.until(lambda: self.notices.last_status == "unknown")
        await self.wake()
        await self.notices.pump(self.bridge, [])
        self.session.speak_companion_notice.assert_awaited_once()
        self.assertEqual(self.host.connections, 1)
