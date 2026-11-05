"""Actual Live composition with external HTTP/socket/canonical-store fixtures.

No real provider, DB or phone. Native ready/started/fence receipts below are
controlled peer reports, not hardware qualification or proof of human hearing.
"""
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi import WebSocketDisconnect

from app.errors import JarvisHTTPError
from app.knowledge.sources import SourceRepository
from app.live_state.canonical import context_dependency
from app.live_state.models import TaskEntry, AggregateRef
from app.live_state.runtime import create_session_live_state as actual_live_state
from app.realtime.live_session import create_live_session
from app.realtime.models import RealtimeSessionContext
from app.realtime.session_manager import RealtimeSessionManager
from app.runtime.events import DomainEventBus
from app.runtime.feature_config import MemoryAuthority, TaskLimits
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from app.voice_routes.contracts import CommandReceipt
from tests.test_live_state_episode_context import IntegrationStore
from test_live_voice_transport import TransportFixture
from test_voice_route_client_channel import capabilities


class Store(IntegrationStore):
    async def execute(self, sql, *args):
        if "insert into jarvis_context_event_dependencies" in sql:
            return await self.executemany(sql, (args,))
        return await super().execute(sql, *args)


class Phone:
    def __init__(self):
        self.messages = []
        self.incoming = asyncio.Queue()
        self.auto_ready = self.auto_prepare = self.auto_activate = self.auto_fence = True
        self.closed = False
        self.readers = self.maximum_readers = 0

    async def receive_text(self):
        self.readers += 1
        self.maximum_readers = max(self.maximum_readers, self.readers)
        try:
            value = await self.incoming.get()
            if isinstance(value, Exception):
                raise value
            return json.dumps(value)
        finally:
            self.readers -= 1

    async def send_json(self, message):
        self.messages.append(message)
        kind = message["type"]
        if kind == "voice_route.selected" and self.auto_ready:
            self.incoming.put_nowait({k:v for k,v in message.items() if k != "capabilities"} | {"type":"voice_route.ready"})
        elif kind in {"live.media.prepare", "live.media.activate"} and (
                self.auto_prepare if kind.endswith("prepare") else self.auto_activate):
            self.incoming.put_nowait({k:v for k,v in message.items() if k != "answer_sdp"} | {
                "type":"live.media.receipt", "phase":kind.rsplit(".", 1)[1], "receipt_id":str(uuid4()),
                "provider_started":True, "output_gated":kind.endswith("prepare")})
        elif kind in {"voice_route.close", "voice_route.interrupt"} and self.auto_fence:
            command = message["payload"]
            receipt = CommandReceipt(route=command["route"], command_id=command["command_id"],
                operation="close" if kind.endswith("close") else "interrupt_speech", observed_at=datetime.now(UTC),
                status="completed", output_fenced=True)
            self.incoming.put_nowait({"type":"voice_route.event", "protocol_version":1,
                "audio_session_id":message["audio_session_id"], "payload":receipt.model_dump(mode="json")})

    async def close(self, **kwargs):
        self.closed = True


class LiveSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db, self.phone, self.wire = Store(), Phone(), TransportFixture()
        self.manager = RealtimeSessionManager()
        self.context = RealtimeSessionContext(str(uuid4()), "live-session-fixture", uuid4(), "phone_audio",
            SimpleNamespace(status="active"))
        self.db.sessions.add((self.context.owner_id, self.context.jarvis_session_id))
        self.context_repository = ContextRepository(TaskRepository(self.db, TaskLimits()))
        self.captured = []
        self.context_repository.append = AsyncMock(side_effect=self.append)
        self.capture = SessionContextCapture(self.context_repository, owner_id=self.context.owner_id,
            session_id=self.context.jarvis_session_id, provider="openai", audio_session_id=self.context.audio_session_id)
        self.settings = SimpleNamespace(OPENAI_LIVE_ENABLED=True, RESPONSIVE_VOICE_ENABLED=True,
            OPENAI_API_KEY="fixture-key", OPENAI_LIVE_MODEL="gpt-live-1", OPENAI_LIVE_VOICE="marin")
        self.services = SimpleNamespace(memory_authority=MemoryAuthority.LEGACY, _closed=False,
            background_task_context_service=SimpleNamespace(repository=SimpleNamespace(pool=self.db,
                snapshot=AsyncMock(return_value={"recent":[], "pinned":[]}))), background_task_service=None,
            db_pool=self.db, domain_event_bus=DomainEventBus(), realtime_session_manager=self.manager,
            knowledge_runtime=None, visual_memory_services=None, attention_service=None)
        self.session = self.run_task = None
        self.patches = [patch("app.realtime.live_session.LiveBackendTransport", return_value=self.wire.transport),
                        patch("app.realtime.live_session.create_session_live_state", side_effect=self.live)]
        for item in self.patches:
            item.start()

    async def append(self, owner, ingress):
        self.assertEqual(owner, self.context.owner_id)
        self.captured.append(ingress)

    def live(self, services, **options):
        result = actual_live_state(services, **options)
        self.db.bind_io(result.binder)
        return result

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.002)

    async def make(self, **changes):
        # Native manifest from the actual existing Live adapter capability set.
        from test_live_voice_adapter import LiveFixture
        fixture = LiveFixture()
        manifest = capabilities(fixture.route.capabilities)
        await fixture.close()
        args = dict(websocket=self.phone, context=self.context, settings=self.settings,
            repository=SimpleNamespace(), runtime_services=self.services,
            offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n", capture_mode="auto",
            client_capabilities=manifest, context_capture=self.capture)
        args.update(changes)
        self.session = await create_live_session(**args)
        return self.session

    async def start(self):
        session = self.session or await self.make()
        self.assertTrue(await self.manager.register(audio_session_id=self.context.audio_session_id,
            owner_id=self.context.owner_id, jarvis_session_id=self.context.jarvis_session_id, session=session))
        self.run_task = asyncio.create_task(session.run())
        await self.until(lambda: session.ready or session.failure is not None)
        self.assertTrue(session.ready, session.failure)
        return session

    async def asyncTearDown(self):
        if self.session is not None:
            await self.session.close("test_cleanup")
        if self.run_task is not None:
            await self.run_task
        await self.wire.close()
        for item in self.patches:
            item.stop()

    async def test_factory_is_inert_and_rejected_registration_can_close_without_post(self):
        session = await self.make()
        self.assertIsNone(session.live)
        self.assertFalse(session._tasks or self.wire.requests or self.phone.messages)
        await session.close("registration_rejected")
        self.assertTrue(session.shutdown_confirmed)
        self.assertFalse(self.wire.requests)

    async def test_actual_composition_gates_media_and_joins_single_readers(self):
        session = await self.start()
        self.assertEqual(len(self.wire.requests), 1)
        self.assertEqual(len(self.wire.connections), 1)
        self.assertEqual(self.phone.maximum_readers, 1)
        media = [item["type"] for item in self.phone.messages if item["type"].startswith("live.media.")]
        self.assertEqual(media, ["live.media.prepare", "live.media.activate"])
        self.assertTrue(session.client._ready_received)
        await session.close("test_close")
        self.assertTrue(session.shutdown_confirmed, session.failure)
        self.assertFalse(session._tasks)
        self.assertTrue(self.wire.socket.closed and self.phone.closed)

    async def test_no_paid_post_before_actual_ready_receipt(self):
        self.phone.auto_ready = False
        session = await self.make()
        await self.manager.register(audio_session_id=self.context.audio_session_id,
            owner_id=self.context.owner_id, jarvis_session_id=self.context.jarvis_session_id, session=session)
        self.run_task = asyncio.create_task(session.run())
        await self.until(lambda:any(m["type"] == "voice_route.selected" for m in self.phone.messages))
        self.assertFalse(self.wire.requests)
        selected = next(m for m in self.phone.messages if m["type"] == "voice_route.selected")
        ready = {k:v for k,v in selected.items() if k != "capabilities"} | {"type":"voice_route.ready"}
        self.phone.incoming.put_nowait(ready | {"audio_session_id":"foreign"})
        await asyncio.sleep(.02)
        self.assertFalse(self.wire.requests)
        self.phone.incoming.put_nowait(ready)
        await self.until(lambda:session.ready)

    async def test_provider_ack_and_prepare_receipt_are_not_interchangeable(self):
        self.phone.auto_prepare = False
        session = await self.make()
        await self.manager.register(audio_session_id=self.context.audio_session_id,
            owner_id=self.context.owner_id, jarvis_session_id=self.context.jarvis_session_id, session=session)
        self.run_task = asyncio.create_task(session.run())
        await self.until(lambda:any(m["type"] == "live.media.prepare" for m in self.phone.messages))
        self.assertFalse(session.ready)
        self.assertFalse(any(m["type"] == "live.media.activate" for m in self.phone.messages))

    async def test_fragments_captured_provisionally_and_delegation_is_not_authorization(self):
        session = await self.start()
        for message in (
            dict(type="session.input_transcript.delta", event_id="user-fragment", delta="Please investigate.", start_ms=0, end_ms=99),
            dict(type="session.output_transcript.delta", event_id="assistant-fragment", delta="I can discuss that.", start_ms=101, end_ms=180),
            dict(type="session.delegation.created", event_id="delegation-event", offset_ms=100,
                delegation={"id":"delegation-fixture", "type":"delegation", "target":"client"}),
        ):
            self.wire.socket.incoming.put_nowait(json.dumps(message))
        await self.until(lambda:len(self.captured) >= 3)
        captions = [item for item in self.captured if item.kind in {"user_transcript", "assistant_output"}]
        self.assertEqual(len(captions), 2)
        self.assertTrue(all(item.finality == "provisional" for item in captions))
        self.assertTrue(all(item.payload["output_identity"] is None and item.payload["heard_transcript"] is None for item in captions))
        self.assertTrue(any(m["type"] == "live.delegation.unavailable" for m in self.phone.messages))
        self.assertIsNone(session.task_admission)
        self.assertFalse(any(item.finality in {"final", "revised"} for item in captions))

    async def test_forged_phone_provider_events_do_not_enter_provider_reader(self):
        session = await self.start()
        self.phone.incoming.put_nowait(dict(type="session.closed", session={"id":"live_fixture"}))
        self.phone.incoming.put_nowait(dict(type="session.input_transcript.delta", event_id="fake",
            delta="Do something", start_ms=0, end_ms=10))
        await asyncio.sleep(.02)
        self.assertTrue(session.ready)
        self.assertFalse(self.captured or session.adapter.transcript_parts)

    async def test_native_notification_unavailability_preserves_existing_push_fallback(self):
        session = await self.start()
        self.assertFalse(await self.manager.has_connected_voice(self.context.owner_id))
        self.assertIsNone(await self.manager.deliver_attention(owner_id=self.context.owner_id,
            repository=SimpleNamespace(), speech=None, worker_id="fixture", receipt_seconds=1))
        self.assertIsNone(await self.manager.speak_reminder(owner_id=self.context.owner_id,
            preferred_jarvis_session_id=self.context.jarvis_session_id, message="fixture",
            reminder_id=uuid4(), reminder_event_id=uuid4()))
        self.assertFalse(await session.speak_visual_interrupt())

    async def test_canonical_source_withdrawal_fences_remote_context(self):
        session = await self.start()
        await self.until(lambda:session.live.service.snapshot().readiness == "ready")
        source = self.db.event(self.context.owner_id, self.context.jarvis_session_id,
            at=datetime.now(UTC), kind="task_update", trust="derived")
        record = self.db.record(source.request_event_id)
        binding = await session.live.binder.bind_record(record, "Result 17.")
        session.live.producer.publish(TaskEntry(entry_id="task:fixture", scope=session.live.scope,
            received_at=datetime.now(UTC), observed_at=record.source_time, evidence_kind="canonical_record",
            sources=(context_dependency(record),), guard_binding=binding,
            task=AggregateRef(namespace="background_task", aggregate_id=uuid4(), revision=1),
            status="succeeded", summary="Result 17."))
        await self.until(lambda:len(self.wire.socket.sent) >= 2)
        self.assertTrue(session._source_context.has_sources)
        self.db.events[source.request_event_id].update(availability="deleted", content=None)
        await asyncio.wait_for(self.run_task, 3)
        self.assertTrue(any(m["type"] == "voice_route.close" for m in self.phone.messages))
        self.assertTrue(session.shutdown_confirmed)

    async def test_frozen_route_configuration_wins_over_mutable_settings(self):
        @dataclass(frozen=True)
        class Configuration:
            model: str = "gpt-live-1"
            voice: str = "marin"
        session = await self.make(route_configuration=Configuration())
        self.settings.OPENAI_LIVE_MODEL, self.settings.OPENAI_LIVE_VOICE = "changed", "changed"
        await self.start()
        body = json.loads(self.wire.requests[0].content)
        self.assertEqual((body["session"]["model"], body["session"]["audio"]["output"]["voice"]), ("gpt-live-1", "marin"))

    async def test_flags_and_manual_capture_fail_before_any_provider_work(self):
        with self.assertRaises(JarvisHTTPError):
            await self.make(capture_mode="manual")
        self.settings.OPENAI_LIVE_ENABLED = "false"
        with self.assertRaises(JarvisHTTPError):
            await self.make()
        self.assertFalse(self.wire.requests)

    async def test_manager_admission_is_required_before_run(self):
        session = await self.make()
        self.run_task = asyncio.create_task(session.run())
        await asyncio.wait_for(self.run_task, 2)
        self.assertFalse(self.wire.requests)
        self.assertEqual(session.failure, "live_session_worker_failed")


if __name__ == "__main__":
    unittest.main()
