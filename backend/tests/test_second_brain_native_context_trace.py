import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from evals.second_brain.context_trace import ProbePhone, close_probe_resources, completed_answer


class NativeContextTraceContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_is_not_played_retained_or_mistaken_for_a_playback_ack(self):
        phone = ProbePhone()
        await phone.send_json({"type": "audio.output.delta", "audio_base64": "AQACAA=="})
        event = phone.events[0]
        self.assertEqual(event["audio_bytes"], 4)
        self.assertEqual(len(event["audio_sha256"]), 64)
        self.assertNotIn("audio_base64", event)
        self.assertNotIn("playback_status", event)

    async def test_text_transport_uses_the_same_audio_redaction(self):
        phone = ProbePhone()
        await phone.send_text(json.dumps({"type": "audio.output.delta", "audio_base64": "AQACAA=="}))
        self.assertEqual(phone.events[0]["audio_bytes"], 4)
        self.assertNotIn("audio_base64", phone.events[0])

    async def test_shutdown_errors_do_not_skip_task_pool_or_fixture_cleanup(self):
        order, result = [], {}
        async def fail(label):
            order.append(label)
            raise RuntimeError(label)
        async def cleanup_fixture():
            order.append("fixture")
            return {"status": "cleaned"}
        voice = SimpleNamespace(close=AsyncMock(side_effect=lambda reason: None))
        voice.close.side_effect = RuntimeError("voice")
        pool = SimpleNamespace(close=AsyncMock(side_effect=lambda: None))
        async def close_voice(reason):
            await fail("voice")
        async def close_pool():
            await fail("pool")
        voice.close.side_effect = close_voice
        pool.close.side_effect = close_pool
        fixture = SimpleNamespace(cleanup=AsyncMock(side_effect=cleanup_fixture))
        running = asyncio.create_task(asyncio.sleep(60))
        with self.assertRaises(RuntimeError):
            await close_probe_resources(result, fixture=fixture, pool=pool, running=running, voice=voice)
        self.assertTrue(running.cancelled())
        self.assertEqual(order, ["voice", "pool", "fixture"])
        self.assertEqual(result["cleanup"], {"status": "cleaned"})

    def test_tool_only_or_unrelated_completion_does_not_end_the_probe(self):
        transcript = {"type": "transcript.done", "response_id": "spoken", "text": "Fixture answer"}
        self.assertEqual(completed_answer([{"type": "response.done", "response_id": "tool"}]), "")
        self.assertEqual(completed_answer([transcript, {"type": "response.done", "response_id": "tool"}]), "")
        self.assertEqual(completed_answer([transcript, {"type": "response.done", "response_id": "spoken", "status": "cancelled"}]), "")
        self.assertEqual(completed_answer([transcript, {"type": "response.done", "response_id": "spoken", "status": "completed"}]), "Fixture answer")
