import json
import unittest

from evals.gemini_live.routing_probe import RoutedPhone, completed_after_resume, completed_spoken_answer, successful_execution


class RoutedPhoneTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_asgi_handshake_then_phone_event_and_disconnect(self):
        phone = RoutedPhone()
        self.assertEqual(await phone.asgi_receive(), {"type": "websocket.connect"})
        await phone.send("response.create")
        event = await phone.asgi_receive()
        self.assertEqual(event["type"], "websocket.receive")
        self.assertEqual(json.loads(event["text"])["type"], "response.create")
        await phone.disconnect()
        self.assertEqual((await phone.asgi_receive())["type"], "websocket.disconnect")

    async def test_asgi_server_control_events_are_not_model_responses(self):
        phone = RoutedPhone()
        await phone.asgi_send({"type": "websocket.accept"})
        self.assertTrue(phone.accepted)
        self.assertEqual(phone.events, [])
        await phone.asgi_send({"type": "websocket.send", "text": '{"type":"session.created"}'})
        self.assertEqual(phone.events, [{"type": "session.created"}])
        await phone.asgi_send({"type": "websocket.close", "code": 1008})
        self.assertTrue(phone.closed)
        self.assertEqual(phone.close_code, 1008)


class RoutingEvidenceTests(unittest.TestCase):
    def test_resumption_alone_does_not_prove_a_post_resume_answer(self):
        answer = [{"type": "audio.output.delta"}, {"type": "response.done"}]
        resumed = {"type": "audio.provider.reconnected", "provider": "gemini", "resumed": True}
        self.assertFalse(completed_after_resume(answer + [resumed]))
        self.assertFalse(completed_after_resume([resumed, {"type": "response.done"}]))
        self.assertTrue(completed_after_resume([resumed] + answer))
        self.assertFalse(completed_after_resume([resumed | {"resumed": False}] + answer))

    def test_tool_only_completion_cannot_finish_a_spoken_answer(self):
        events = [{"type": "response.done"}, {"type": "audio.output.delta"}]
        self.assertFalse(completed_spoken_answer(events))
        self.assertTrue(completed_spoken_answer(events + [{"type": "response.done"}]))

    def test_retrieval_requires_a_successful_result_for_the_correct_record(self):
        snapshot = {"executions": [{"tool_name": "explain_memory",
                    "arguments": '{"memory_id":"right-id"}', "result": '{"status":"success"}'}]}
        self.assertTrue(successful_execution(snapshot, "explain_memory", memory_id="right-id"))
        self.assertFalse(successful_execution(snapshot, "explain_memory", memory_id="wrong-id"))
        snapshot["executions"][0]["result"] = '{"status":"failed"}'
        self.assertFalse(successful_execution(snapshot, "explain_memory", memory_id="right-id"))
