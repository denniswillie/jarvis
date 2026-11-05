"""Real signaling broker and wire parser; no provider or physical media claims."""
import asyncio
from dataclasses import replace
import unittest
from uuid import uuid4

from app.live_state.models import SessionScope
from app.voice_routes.contracts import RouteIdentity
from app.voice_routes.live_adapter import LiveMediaCommand
from app.voice_routes.live_media import LiveMediaSignaling, LiveMediaSignalingError


class LiveMediaTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.route = RouteIdentity(scope=SessionScope(owner_id="media-fixture", session_id=uuid4()),
            route_id=uuid4(), route_generation=1)
        self.connection = uuid4()
        self.command = LiveMediaCommand(self.route, "fixture-audio", uuid4(), "live_fixture", "prepare", "v=0 fixture")
        self.written = asyncio.Event()
        self.release = asyncio.Event()
        self.before_write = None
        self.block_before = self.block_after = False
        self.writer_result = True
        self.current = True
        self.calls = []
        self.broker = LiveMediaSignaling(route=self.route, audio_session_id="fixture-audio",
            connection_id=self.connection, send=self.send, timeout_seconds=.3)
        self.tasks = []

    async def asyncTearDown(self):
        await self.broker.aclose()
        self.release.set()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def send(self, payload, current):
        if self.before_write is not None:
            self.before_write()
        if self.block_before:
            await self.release.wait()
        if not current():
            self.written.set()
            return False
        self.calls.append(payload)
        self.written.set()
        if self.block_after:
            await self.release.wait()
        return self.writer_result

    def receipt(self, command=None, **changes):
        command = command or self.command
        return dict(type="live.media.receipt", protocol_version=1,
            route=command.route.model_dump(mode="json"), audio_session_id=command.audio_session_id,
            command_id=str(command.command_id), provider_session_id=command.provider_session_id,
            phase=command.phase, receipt_id=str(uuid4()), provider_started=True,
            output_gated=command.phase == "prepare", **changes)

    async def begin(self, command=None):
        self.written.clear()
        task = asyncio.create_task(self.broker.request(command or self.command, lambda: self.current))
        self.tasks.append(task)
        await asyncio.wait_for(self.written.wait(), 1)
        return task

    def receive(self, payload):
        return self.broker.receive(payload, connection_id=self.connection)

    async def prepare(self):
        task = await self.begin()
        payload = self.receipt()
        self.assertTrue(self.receive(payload))
        return await task, payload

    async def test_two_phase_handshake_uses_actual_correlated_client_receipts(self):
        prepared, _ = await self.prepare()
        activate = replace(self.command, phase="activate", answer_sdp=None)
        task = await self.begin(activate)
        self.assertFalse(task.done())
        self.assertTrue(self.receive(self.receipt(activate)))
        active = await task
        self.assertTrue(prepared.output_gated)
        self.assertFalse(active.output_gated)
        self.assertNotEqual(prepared.receipt_id, active.receipt_id)
        self.assertEqual([p["type"] for p in self.calls], ["live.media.prepare", "live.media.activate"])
        self.assertNotIn("answer_sdp", self.calls[1])

    async def test_write_does_not_become_readiness(self):
        task = await self.begin()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertEqual(self.broker.pending_count, 1)
        self.receive(self.receipt())
        await task

    async def test_wrong_scope_and_unsupported_receipts_do_not_consume_current_waiter(self):
        task = await self.begin()
        payload = self.receipt()
        cases = ({"audio_session_id": "foreign"}, {"command_id": str(uuid4())},
            {"provider_session_id": "foreign"}, {"provider_started": False},
            {"output_gated": False}, {"protocol_version": True}, {"extra": "private"},
            {"phase": "activate"}, {"receipt_id": "not-a-uuid"})
        for change in cases:
            with self.subTest(change=change):
                self.assertFalse(self.receive(payload | change))
        self.assertFalse(self.broker.receive(payload, connection_id=uuid4()))
        self.assertFalse(task.done())
        self.assertTrue(self.receive(payload))
        await task

    async def test_receipt_before_actual_writer_gate_is_rejected(self):
        self.before_write = lambda: self.assertFalse(self.receive(self.receipt()))
        task = await self.begin()
        self.assertTrue(self.receive(self.receipt()))
        await task

    async def test_receipt_while_write_returns_is_buffered_not_prematurely_completed(self):
        self.block_after = True
        task = await self.begin()
        self.assertTrue(self.receive(self.receipt()))
        self.assertFalse(task.done())
        self.release.set()
        await task

    async def test_failed_write_cannot_be_rescued_by_a_receipt(self):
        self.block_after = True
        self.writer_result = False
        task = await self.begin()
        self.assertTrue(self.receive(self.receipt()))
        self.release.set()
        with self.assertRaises(LiveMediaSignalingError):
            await task
        self.assertTrue(self.broker.closed)

    async def test_duplicate_prepare_is_idempotent_without_new_write(self):
        receipt, payload = await self.prepare()
        self.assertTrue(self.receive(payload))
        self.assertFalse(self.receive(payload | {"receipt_id": str(uuid4())}))
        self.assertEqual(await self.broker.request(self.command, lambda: True), receipt)
        self.assertEqual(len(self.calls), 1)
        with self.assertRaises(LiveMediaSignalingError):
            await self.broker.request(replace(self.command, answer_sdp="v=0 changed"), lambda: True)

    async def test_receipt_id_cannot_be_reused_across_phases(self):
        prepared, _ = await self.prepare()
        command = replace(self.command, phase="activate", answer_sdp=None)
        task = await self.begin(command)
        payload = self.receipt(command)
        self.assertFalse(self.receive(payload | {"receipt_id": str(prepared.receipt_id)}))
        self.assertTrue(self.receive(payload))
        await task

    async def test_activation_requires_matching_completed_preparation(self):
        with self.assertRaises(LiveMediaSignalingError):
            await self.broker.request(replace(self.command, phase="activate", answer_sdp=None), lambda: True)
        await self.prepare()
        for change in ({"command_id": uuid4()}, {"provider_session_id": "live_foreign"}):
            with self.subTest(change=change), self.assertRaises(LiveMediaSignalingError):
                await self.broker.request(replace(self.command, phase="activate", answer_sdp=None, **change), lambda: True)
        self.assertEqual(len(self.calls), 1)

    async def test_duplicate_pending_command_is_not_replayed(self):
        task = await self.begin()
        with self.assertRaises(LiveMediaSignalingError):
            await self.broker.request(self.command, lambda: True)
        self.assertTrue(self.receive(self.receipt()))
        await task
        self.assertEqual(len(self.calls), 1)

    async def test_close_unblocks_waiter_and_rejects_late_receipts(self):
        task = await self.begin()
        self.broker.close()
        with self.assertRaises(LiveMediaSignalingError):
            await task
        self.assertFalse(self.receive(self.receipt()))
        self.assertEqual(self.broker.pending_count, 0)

    async def test_close_cancels_blocked_writer_without_output_fence_claim(self):
        self.block_after = True
        task = await self.begin()
        await self.broker.aclose()
        with self.assertRaises(LiveMediaSignalingError):
            await task
        self.assertFalse(self.release.is_set())
        self.assertFalse(any(p["type"].startswith("voice_route.") for p in self.calls))

    async def test_caller_cancellation_propagates_and_prevents_replay(self):
        task = await self.begin()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.broker.closed)
        with self.assertRaises(LiveMediaSignalingError):
            await self.broker.request(self.command, lambda: True)

    async def test_stale_generation_at_actual_write_cannot_submit(self):
        self.before_write = lambda: setattr(self, "current", False)
        task = await self.begin()
        with self.assertRaises(LiveMediaSignalingError):
            await task
        self.assertFalse(self.calls)

    async def test_stale_generation_at_receipt_cannot_activate(self):
        task = await self.begin()
        self.current = False
        self.assertFalse(self.receive(self.receipt()))
        with self.assertRaises(LiveMediaSignalingError):
            await task

    async def test_missing_receipt_times_out_without_replay(self):
        self.broker._timeout = .02
        task = await self.begin()
        with self.assertRaisesRegex(LiveMediaSignalingError, "media_receipt_timeout"):
            await task
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.broker.closed)

    async def test_command_scope_and_shape_fail_before_writing(self):
        for changes in ({"audio_session_id": "foreign"}, {"answer_sdp": "private invalid SDP"},
                        {"command_id": "not-a-uuid"}, {"provider_session_id": ""}, {"phase": "unknown"}):
            with self.subTest(changes=changes), self.assertRaises(LiveMediaSignalingError):
                await self.broker.request(replace(self.command, **changes), lambda: True)
        self.assertFalse(self.calls)
