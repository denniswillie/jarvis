"""Actual app writer/channel/broker/startup composition with a controlled peer.

The peer receipts are fixtures, not native/provider or physical-silence evidence.
"""
import asyncio
from dataclasses import replace
from hashlib import sha256
import unittest
from uuid import uuid4

from app.realtime.live_client_channel import LiveClientChannel, LiveClientChannelError
from app.realtime.socket_sender import WebSocketEventSender
from app.voice_routes.client_channel import ClientChannelError
from app.voice_routes.contracts import CloseRouteCommand, CommandReceipt, InterruptSpeechCommand
from app.voice_routes.live_adapter import LiveMediaCommand, LiveVoiceRoute
from test_live_voice_adapter import LiveFixture, NOW
from test_voice_route_client_channel import capabilities


class Peer:
    def __init__(self):
        self.messages = []
        self.auto_ready = self.auto_media = self.auto_fence = True
        self.wrote = asyncio.Event()
        self.before_send = None
        self.receive = None

    async def send_json(self, message):
        if self.before_send is not None:
            await self.before_send(message)
        self.messages.append(message)
        self.wrote.set()
        kind = message["type"]
        if kind == "voice_route.selected" and self.auto_ready:
            self.receive({k:v for k,v in message.items() if k != "capabilities"} | {"type": "voice_route.ready"})
        elif kind.startswith("live.media.") and self.auto_media:
            self.receive({k:v for k,v in message.items() if k != "answer_sdp"} | {
                "type": "live.media.receipt", "phase": kind.rsplit(".", 1)[1],
                "receipt_id": str(uuid4()), "provider_started": True,
                "output_gated": kind == "live.media.prepare"})
        elif kind in {"voice_route.interrupt", "voice_route.close"} and self.auto_fence:
            command = message["payload"]
            receipt = CommandReceipt(route=command["route"], command_id=command["command_id"],
                operation="close" if kind == "voice_route.close" else "interrupt_speech",
                observed_at=NOW, status="completed", output_fenced=True)
            self.receive({"type": "voice_route.event", "protocol_version": 1,
                "audio_session_id": message["audio_session_id"], "payload": receipt.model_dump(mode="json")})


class LiveClientChannelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = LiveFixture()
        self.connection = uuid4()
        self.current_connection = self.connection
        self.peer = Peer()
        self.sender = WebSocketEventSender(self.peer)
        self.interrupts = []
        self.client = LiveClientChannel(route=self.fixture.identity, audio_session_id="fixture-audio",
            connection_id=self.connection, offer_sdp="v=0 fixture", capabilities=self.fixture.route.capabilities,
            sender=self.sender, current_connection=lambda: self.current_connection,
            on_interrupt=self.interrupts.append, ready_timeout=.3, control_timeout=.15)
        self.peer.receive = lambda payload: self.client.receive(payload, connection_id=self.connection)
        self.manifest = capabilities(self.fixture.route.capabilities)
        self.tasks = []

    async def asyncTearDown(self):
        await self.client.aclose()
        for task in self.tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.fixture.close()

    def ready(self):
        return dict(type="voice_route.ready", protocol_version=1, audio_session_id="fixture-audio",
            route=self.fixture.identity.model_dump(mode="json"))

    def command(self, *, close=False):
        model = CloseRouteCommand if close else InterruptSpeechCommand
        return model(route=self.fixture.identity, command_id=uuid4(), requested_at=NOW, reason="fixture")

    async def prepare(self):
        await self.client.select(self.manifest)
        request = self.fixture.start_request()
        proof = await self.client.bootstrap_gate(request)
        return request, proof

    async def test_proof_requires_actual_selection_and_received_ready(self):
        request = self.fixture.start_request()
        self.assertFalse(self.peer.receive(self.ready()))
        with self.assertRaises(LiveClientChannelError):
            await self.client.bootstrap_gate(request)
        self.peer.auto_ready = False
        selection = asyncio.create_task(self.client.select(self.manifest))
        self.tasks.append(selection)
        await asyncio.wait_for(self.peer.wrote.wait(), 1)
        with self.assertRaises(LiveClientChannelError):
            await self.client.bootstrap_gate(request)
        self.assertTrue(self.peer.receive(self.ready()))
        await selection
        proof = await self.client.bootstrap_gate(request)
        self.assertEqual(proof.offer_sha256, sha256(b"v=0 fixture").hexdigest())
        self.assertEqual(proof.connection_id, self.connection)
        self.assertTrue(proof.selection_written and proof.client_ready_received and proof.output_gated)
        self.assertTrue(proof.is_current() and proof.is_gated())

    async def test_foreign_readiness_does_not_establish_a_gate(self):
        self.peer.auto_ready = False
        selection = asyncio.create_task(self.client.select(self.manifest))
        self.tasks.append(selection)
        await asyncio.wait_for(self.peer.wrote.wait(), 1)
        self.assertFalse(self.client.receive(self.ready(), connection_id=uuid4()))
        self.assertFalse(self.peer.receive(self.ready() | {"audio_session_id": "foreign"}))
        self.assertFalse(self.peer.receive(self.ready() | {"protocol_version": True}))
        self.assertTrue(self.peer.receive(self.ready()))
        await selection

    async def test_offer_digest_and_route_are_not_rebound(self):
        request, proof = await self.prepare()
        wrong = request.model_copy(update={"route": request.route.model_copy(update={"route_generation": 2})})
        with self.assertRaises(LiveClientChannelError):
            await self.client.bootstrap_gate(wrong)
        self.assertNotEqual(proof.offer_sha256, sha256(b"v=0 different peer").hexdigest())

    async def test_actual_media_broker_opens_gate_only_at_activation_submission(self):
        request, proof = await self.prepare()
        prepare = LiveMediaCommand(self.fixture.identity, "fixture-audio", request.command_id,
            "live_fixture", "prepare", "v=0 fixture-answer")
        receipt = await self.client.media_control(prepare, proof.is_current)
        self.assertTrue(receipt.output_gated and proof.is_gated())
        active = await self.client.media_control(replace(prepare, phase="activate", answer_sdp=None), proof.is_current)
        self.assertFalse(active.output_gated or proof.is_gated())
        self.assertTrue(proof.is_current())
        self.assertTrue(self.peer.receive(self.ready()))
        self.assertFalse(proof.is_gated())

    async def test_gate_becomes_unknown_at_activation_write_not_after_ack(self):
        request, proof = await self.prepare()
        prepare = LiveMediaCommand(self.fixture.identity, "fixture-audio", request.command_id,
            "live_fixture", "prepare", "v=0 answer")
        await self.client.media_control(prepare, proof.is_current)
        self.peer.auto_media = False
        self.peer.wrote.clear()
        active = asyncio.create_task(self.client.media_control(
            replace(prepare, phase="activate", answer_sdp=None), proof.is_current))
        self.tasks.append(active)
        await asyncio.wait_for(self.peer.wrote.wait(), 1)
        self.assertFalse(proof.is_gated())
        self.assertFalse(active.done())
        self.client.close()
        await asyncio.gather(active, return_exceptions=True)

    async def test_permanent_fence_invalidates_proof_but_accepts_real_stop_receipt(self):
        _, proof = await self.prepare()
        receipt = await self.client.fence_output(self.command())
        self.assertEqual(receipt.status, "completed")
        self.assertTrue(receipt.output_fenced)
        self.assertFalse(proof.is_current() or proof.is_gated())
        self.assertFalse(self.peer.receive(self.ready()))
        self.assertEqual(self.peer.messages[-1]["type"], "voice_route.interrupt")

    async def test_local_interrupt_invalidates_synchronously_and_enqueues_once(self):
        _, proof = await self.prepare()
        command = self.command()
        payload = dict(type="voice_route.interrupt", protocol_version=1, audio_session_id="fixture-audio",
            payload=command.model_dump(mode="json"))
        self.assertTrue(self.peer.receive(payload))
        self.assertFalse(proof.is_current())
        self.assertEqual(self.interrupts, [command])
        self.assertFalse(self.peer.receive(payload))
        self.assertEqual(len(self.interrupts), 1)
        self.assertEqual(len(self.peer.messages), 1)

    async def test_replaced_connection_invalidates_live_proof_without_reconstruction(self):
        _, proof = await self.prepare()
        self.current_connection = uuid4()
        self.assertFalse(proof.is_current() or proof.is_gated())
        self.assertFalse(self.peer.receive(self.ready()))
        before = len(self.peer.messages)
        receipt = await self.client.fence_output(self.command(close=True))
        self.assertEqual(receipt.status, "unknown")
        self.assertEqual(len(self.peer.messages), before)

    async def test_codec_mismatch_is_rejected_not_silently_translated(self):
        with self.assertRaises(ClientChannelError):
            await self.client.select(self.manifest | {"input_codecs": ["opus"], "output_codecs": ["opus"]})
        self.assertFalse(self.peer.messages)

    async def test_actual_writer_lock_rechecks_connection_before_selection(self):
        await self.sender._lock.acquire()
        selected = asyncio.create_task(self.client.select(self.manifest))
        self.tasks.append(selected)
        await asyncio.sleep(0)
        self.current_connection = uuid4()
        self.sender._lock.release()
        with self.assertRaises(ClientChannelError):
            await selected
        self.assertFalse(self.peer.messages)

    async def test_full_adapter_start_uses_real_channel_and_media_broker_no_start_fence(self):
        await self.client.select(self.manifest)
        bindings = replace(self.fixture.route._b, client_connection_id=self.connection,
            bootstrap_gate=self.client.bootstrap_gate, media_control=self.client.media_control,
            fence_output=self.client.fence_output)
        self.fixture.route = LiveVoiceRoute(bindings, io_timeout=.2, ack_timeout=.3, start_timeout=1,
            clock=lambda: self.fixture.now)
        receipt = await self.fixture.start()
        self.assertEqual(receipt.status, "completed")
        self.assertEqual([message["type"] for message in self.peer.messages],
            ["voice_route.selected", "live.media.prepare", "live.media.activate"])
        closed = await self.fixture.route.close(self.command(close=True))
        self.assertTrue(closed.output_fenced)
        self.assertEqual(self.peer.messages[-1]["type"], "voice_route.close")

    async def test_disconnection_before_bootstrap_prevents_provider_creation(self):
        await self.client.select(self.manifest)
        bindings = replace(self.fixture.route._b, client_connection_id=self.connection,
            bootstrap_gate=self.client.bootstrap_gate, media_control=self.client.media_control,
            fence_output=self.client.fence_output)
        self.fixture.route = LiveVoiceRoute(bindings, io_timeout=.2, ack_timeout=.3, start_timeout=1,
            clock=lambda: self.fixture.now)
        self.sender.mark_closed()
        receipt = await self.fixture.start()
        self.assertEqual(receipt.status, "failed")
        self.assertFalse(self.fixture.wire.transport.creation_submitted)
