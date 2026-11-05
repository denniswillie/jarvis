"""Shared user-cancel handler must not await a native notification's fence write."""
import asyncio
import unittest
from unittest.mock import AsyncMock

from app.attention.native_voice import NativeAttentionSink
import test_native_attention_delivery as attention_fixtures
import test_realtime_image_tools as session_fixtures


class NativeAttentionControlLivenessTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_during_prestart_notification_keeps_reader_responsive(self):
        session_fixture = session_fixtures.RealtimeImageToolTests()
        await session_fixture.asyncSetUp()
        attention_fixture = attention_fixtures.NativeDeliveryTests()
        await attention_fixture.asyncSetUp()
        session, runtime = session_fixture.session, attention_fixture.runtime
        runtime.before_session_close = AsyncMock(return_value=True)
        runtime.after_session_close = AsyncMock(return_value=True)
        session._voice_route_runtime = runtime
        sink = NativeAttentionSink(session, runtime, attention_fixture.f, AsyncMock(),
                                   clock=attention_fixture.sink.clock)
        entered = asyncio.Event()
        original_interrupt = runtime.coordinator.interrupt_speech.side_effect

        async def writer_fence(command):
            entered.set()
            await session.openai.send_event({"type": "output_audio_buffer.clear"})
            return await original_interrupt(command)

        runtime.coordinator.interrupt_speech.side_effect = writer_fence
        # A real delivery is owned, but its producer has not entered yet. This
        # is a legitimate barge-in boundary; no fake delivery completion needed.
        delivery = sink.create_delivery(attention_fixture.f, None, attention_fixture.f.claim,
                                        session, receipt_seconds=1)
        session._attention_delivery = delivery
        lock = session.openai._send_lock
        await lock.acquire()
        cancel = asyncio.create_task(session._handle_client_event({"type": "response.cancel"}))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            done, _ = await asyncio.wait({cancel}, timeout=0.05)
            responsive = bool(done)
            admission_free = not session._response_admission_lock.locked()
        finally:
            lock.release()
            await asyncio.wait_for(cancel, 1)
            await sink.close()
            await attention_fixture.asyncTearDown()
            await session_fixture.asyncTearDown()
        self.assertTrue(responsive, "Native attention fence still blocks the client control handler")
        self.assertTrue(admission_free, "Native attention fence still holds response admission")
