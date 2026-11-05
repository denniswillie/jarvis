"""Offline phone-wire qualification without provider, database or native audio."""
import asyncio
import unittest
from uuid import uuid4

from app.voice_routes.client_channel import ClientChannelError, VoiceRouteClientChannel
from app.voice_routes.contracts import CloseRouteCommand, CommandReceipt, InterruptSpeechCommand, OutputIdentity
from test_realtime_voice_route import NOW, FakeControl, make_adapter, make_identity, make_offer


def capabilities(manifest):
    return dict(supported_versions=[1], supported_transports=[manifest.transport.value],
        supported_media_topologies=[manifest.media_topology], input_codecs=[manifest.input_codec],
        output_codecs=[manifest.output_codec], local_output_fencing=True,
        client_playback_boundaries=["client_render_lower_bound"]
        if manifest.client_playback_evidence.value == "supported" else [])


class ClientChannelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.identity = make_identity()
        self.messages = []
        self.wrote = asyncio.Event()
        async def send(message):
            self.messages.append(message)
            self.wrote.set()
        self.send = send
        self.manifest = make_adapter(FakeControl(self.identity)).capabilities
        self.channel = VoiceRouteClientChannel(self.identity, "audio-fixture", self.manifest,
            send, clock=lambda: NOW, ready_timeout=0.05, control_timeout=0.05)

    async def asyncTearDown(self):
        await self.channel.aclose()

    def outer(self, kind, *, payload=None):
        message = dict(type=kind, protocol_version=1, audio_session_id="audio-fixture")
        message["route" if payload is None else "payload"] = (
            self.identity.model_dump(mode="json") if payload is None else payload)
        return message

    async def ready(self):
        await self.channel.select(capabilities(self.manifest))
        self.assertTrue(self.channel.accept_ready(self.outer("voice_route.ready")))
        self.assertTrue(await self.channel.wait_ready())
        self.wrote.clear()

    def command(self, *, close=False):
        model = CloseRouteCommand if close else InterruptSpeechCommand
        return model(route=self.identity, command_id=uuid4(), requested_at=NOW, reason="fixture")

    def receipt(self, command, **changes):
        value = CommandReceipt(route=self.identity, command_id=command.command_id,
            operation="close" if isinstance(command, CloseRouteCommand) else "interrupt_speech",
            observed_at=NOW, status="completed", output_fenced=True, provider_stopped=None)
        return self.outer("voice_route.event", payload=value.model_copy(update=changes).model_dump(mode="json"))

    async def test_selection_write_and_matching_client_ready_are_both_required(self):
        self.assertFalse(self.channel.accept_ready(self.outer("voice_route.ready")))
        await self.channel.select(capabilities(self.manifest))
        self.assertFalse(self.channel.ready)
        self.assertEqual(self.messages[0]["type"], "voice_route.selected")
        self.assertEqual(self.messages[0]["capabilities"], self.manifest.model_dump(mode="json"))
        self.assertTrue(self.channel.accept_ready(self.outer("voice_route.ready")))
        self.assertTrue(await self.channel.wait_ready())

    async def test_mismatched_generation_and_boolean_version_cannot_admit_media(self):
        await self.channel.select(capabilities(self.manifest))
        wrong = self.outer("voice_route.ready")
        wrong["route"] = make_identity(2).model_dump(mode="json")
        self.assertFalse(self.channel.accept_ready(wrong))
        wrong = self.outer("voice_route.ready") | {"protocol_version": True}
        self.assertFalse(self.channel.accept_ready(wrong))
        self.assertFalse(self.channel.ready)

    async def test_incompatible_pcm_route_is_not_silently_relabelled(self):
        client = capabilities(self.manifest)
        client["supported_transports"] = ["chained_voice"]
        with self.assertRaises(ClientChannelError):
            await self.channel.select(client)
        self.assertEqual(self.messages, [])

    async def test_client_without_local_fencing_is_rejected(self):
        client = capabilities(self.manifest) | {"local_output_fencing": False}
        with self.assertRaises(ClientChannelError):
            await self.channel.select(client)

    async def test_selection_timeout_never_falls_back_to_legacy(self):
        await self.channel.select(capabilities(self.manifest))
        self.assertFalse(await self.channel.wait_ready())
        self.assertFalse(self.channel.accept_ready(self.outer("voice_route.ready")))
        with self.assertRaises(ClientChannelError):
            self.channel.media_envelope({"type": "audio.webrtc.started"})

    async def test_media_envelope_does_not_rebind_foreign_identity(self):
        await self.ready()
        message = self.channel.media_envelope({"type": "audio.webrtc.started", "response_id": "r1"})
        self.assertEqual(message["route"], self.identity.model_dump(mode="json"))
        self.assertEqual(message["response_id"], "r1")
        with self.assertRaises(ClientChannelError):
            self.channel.media_envelope({"type": "audio.output.delta", "audio_session_id": "different"})

    async def test_native_buffer_event_names_are_exact(self):
        await self.ready()
        for suffix in ("started", "stopped", "cleared"):
            with self.subTest(suffix=suffix):
                kind = f"audio.webrtc.{suffix}"
                message = self.channel.media_envelope({"type": kind, "response_id": "r1"})
                self.assertEqual(message["type"], kind)
                self.assertEqual(message["route"], self.identity.model_dump(mode="json"))
                self.assertEqual(message["response_id"], "r1")
                with self.assertRaises(ClientChannelError):
                    self.channel.media_envelope({"type": f"audio.webrtc.output_{suffix}"})

    async def test_matching_fence_receipt_not_socket_write_confirms_fence(self):
        await self.ready()
        command = self.command()
        pending = asyncio.create_task(self.channel.request_fence(command))
        await self.wrote.wait()
        self.assertFalse(pending.done())
        self.assertTrue(self.channel.accept_receipt(self.receipt(command)))
        result = await pending
        self.assertEqual(result.status, "completed")
        self.assertTrue(result.output_fenced)
        self.assertIsNone(result.provider_stopped)

    async def test_fence_timeout_and_late_receipt_remain_unknown(self):
        await self.ready()
        command = self.command()
        result = await self.channel.request_fence(command)
        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.output_fenced)
        self.assertFalse(self.channel.accept_receipt(self.receipt(command)))

    async def test_duplicate_command_shares_one_write_and_receipt(self):
        await self.ready()
        command = self.command()
        first = asyncio.create_task(self.channel.request_fence(command))
        await self.wrote.wait()
        second = asyncio.create_task(self.channel.request_fence(command))
        self.channel.accept_receipt(self.receipt(command))
        left, right = await asyncio.gather(first, second)
        self.assertEqual(left, right)
        self.assertEqual(sum(item["type"] == "voice_route.interrupt" for item in self.messages), 1)
        with self.assertRaises(ClientChannelError):
            await self.channel.request_fence(command.model_copy(update={"reason": "different"}))

    async def test_wrong_operation_or_provider_claim_does_not_complete_fence(self):
        await self.ready()
        command = self.command()
        pending = asyncio.create_task(self.channel.request_fence(command))
        await self.wrote.wait()
        self.assertFalse(self.channel.accept_receipt(self.receipt(command, operation="close")))
        self.assertFalse(self.channel.accept_receipt(self.receipt(command, provider_stopped=True)))
        self.assertTrue(self.channel.accept_receipt(self.receipt(command)))
        await pending

    async def test_close_command_fences_new_media_before_receipt(self):
        await self.ready()
        command = self.command(close=True)
        pending = asyncio.create_task(self.channel.request_fence(command))
        await self.wrote.wait()
        self.assertFalse(self.channel.ready)
        with self.assertRaises(ClientChannelError):
            self.channel.media_envelope({"type": "audio.output.delta"})
        self.assertTrue(self.channel.accept_receipt(self.receipt(command)))
        self.assertEqual((await pending).status, "completed")

    async def test_disconnect_resolves_pending_as_unknown(self):
        await self.ready()
        command = self.command()
        pending = asyncio.create_task(self.channel.request_fence(command))
        await self.wrote.wait()
        self.channel.close()
        self.assertEqual((await pending).status, "unknown")

    async def test_local_interrupt_accepts_transport_metadata_not_foreign_route(self):
        await self.ready()
        command = self.command()
        message = self.outer("voice_route.interrupt", payload=command.model_dump(mode="json"))
        message["client_event_id"] = "client-event-1"
        self.assertEqual(self.channel.parse_interrupt(message), command)
        message["payload"]["route"] = make_identity(2).model_dump(mode="json")
        with self.assertRaises(ClientChannelError):
            self.channel.parse_interrupt(message)


class PlaybackChannelTests(ClientChannelTests):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.channel.aclose()
        self.manifest = make_adapter(FakeControl(self.identity), media_transport="websocket", playback=True).capabilities
        self.channel = VoiceRouteClientChannel(self.identity, "audio-fixture", self.manifest,
            self.send, clock=lambda: NOW, ready_timeout=0.05, control_timeout=0.05)

    async def bound(self):
        await self.ready()
        offer = make_offer(self.identity)
        output = OutputIdentity(output_id=uuid4(), provider_response_ref="r1", provider_item_ref="i1")
        await self.channel.bind_output(offer, output)
        return offer, output

    def playback(self, offer, output, milliseconds=20):
        return self.outer("voice_route.playback") | {
            "offer_id": str(offer.offer_id), "attempt_id": str(offer.attempt.attempt_id),
            "attempt_generation": offer.attempt.attempt_generation, "output_id": str(output.output_id),
            "playback": {"receipt_id": str(uuid4()), "observed_at": NOW.isoformat(),
                         "boundary": "client_render_lower_bound", "rendered_ms": milliseconds}}

    async def test_bound_render_receipt_keeps_attempt_generation(self):
        offer, output = await self.bound()
        evidence = self.channel.parse_playback(self.playback(offer, output))
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.attempt.attempt_generation, offer.attempt.attempt_generation)
        self.assertNotEqual(evidence.attempt.attempt_generation, self.identity.route_generation)
        self.assertEqual(evidence.playback.rendered_ms, 20)

    async def test_unbound_or_duplicate_or_regressing_playback_is_rejected(self):
        offer, output = await self.bound()
        message = self.playback(offer, output)
        self.assertIsNotNone(self.channel.parse_playback(message))
        self.assertIsNone(self.channel.parse_playback(message))
        self.assertIsNone(self.channel.parse_playback(self.playback(offer, output, 10)))
        wrong = self.playback(offer, output) | {"output_id": str(uuid4())}
        self.assertIsNone(self.channel.parse_playback(wrong))

    async def test_interruption_retires_old_render_binding(self):
        offer, output = await self.bound()
        command = self.command()
        self.channel.parse_interrupt(self.outer("voice_route.interrupt", payload=command.model_dump(mode="json")))
        self.assertIsNone(self.channel.parse_playback(self.playback(offer, output)))
        with self.assertRaises(ClientChannelError):
            await self.channel.bind_output(offer, output)

    async def test_boolean_attempt_generation_is_not_an_integer_identity(self):
        offer, output = await self.bound()
        message = self.playback(offer, output) | {"attempt_generation": True}
        self.assertIsNone(self.channel.parse_playback(message))
