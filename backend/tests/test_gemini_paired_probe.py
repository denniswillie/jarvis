import unittest
import json

from evals.gemini_live.paired_probe import ExpectedOpenAIAudio, PairedPhone, latency_summary, observed_openai_write, pcm_evidence, safe_event
from evals.gemini_live.gpt_domain_speech_probe import completed_new_audio_response, successful_memory_receipt
from evals.gemini_live.interruption import tool_result_succeeded


class PairedProbeTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_sink_accepts_both_production_websocket_send_forms(self):
        phone = PairedPhone()
        await phone.send_text(json.dumps({"type": "session.created"}))
        await phone.send_json({"type": "response.done"})
        self.assertEqual([event["type"] for event in phone.events], ["session.created", "response.done"])

    async def test_tracing_preserves_lazy_async_factory_and_writer_result(self):
        order, observations = [], []

        async def factory():
            order.append("resolved")
            return {"type": "response.create"}

        async def write(resolve):
            order.append("inside_real_writer")
            payload = await resolve()
            self.assertEqual(payload, {"type": "response.create"})
            return True

        sent = await observed_openai_write(write, factory, lambda *args: observations.append(args))
        self.assertIs(sent, True)
        self.assertEqual(order, ["inside_real_writer", "resolved"])
        self.assertEqual(observations, [("sent", {"type": "response.create"})])

    async def test_rejected_write_is_not_reported_as_sent(self):
        observations = []

        async def write(resolve):
            self.assertIsNone(await resolve())
            return False

        sent = await observed_openai_write(write, lambda: None, lambda *args: observations.append(args))
        self.assertIs(sent, False)
        self.assertFalse(observations)


class PairedProbeReportingTests(unittest.TestCase):
    def test_failed_tool_result_is_not_a_successful_commit_boundary(self):
        self.assertTrue(tool_result_succeeded({"status": "success"}))
        for value in (None, {}, {"status": "failed"}, {"status": "pending"}, "success"):
            self.assertFalse(tool_result_succeeded(value))

    def test_receipt_must_belong_to_a_successful_memory_call(self):
        receipt = {"type": "conversation.item.create", "item": {
            "type": "function_call_output", "call_id": "saved"}}
        self.assertTrue(successful_memory_receipt(receipt, {"saved"}))
        self.assertFalse(successful_memory_receipt(receipt, {"other"}))
        self.assertFalse(successful_memory_receipt(receipt, set()))

    def test_old_completion_does_not_finish_the_new_control_turn(self):
        events = [{"type": "audio.output.delta", "response_id": "old"},
                  {"type": "response.done", "response_id": "old"},
                  {"type": "audio.output.delta", "response_id": "new"}]
        self.assertFalse(completed_new_audio_response(events, {"old"}))
        events.append({"type": "response.done", "response_id": "new"})
        self.assertTrue(completed_new_audio_response(events, {"old"}))

    def test_only_established_commentary_suppression_is_excluded(self):
        expected = ExpectedOpenAIAudio()
        expected.observe_item({"id": "preamble", "phase": "commentary"})
        expected.observe_item({"id": "answer", "phase": "final_answer"})
        self.assertFalse(expected.admits_chunk("preamble"))
        self.assertTrue(expected.admits_chunk("answer"))
        self.assertTrue(expected.admits_chunk("legacy"))
        expected.observe_item({"id": "legacy", "phase": "commentary"})
        self.assertTrue(expected.admits_chunk("legacy"))

    def test_missing_answer_audio_cannot_hide_behind_commentary_filter(self):
        raw = {"commentary": b"aa", "answer": b"bbcc"}
        eligible = {"answer": b"bbcc"}
        result = pcm_evidence(raw, b"bbcc", set(raw), set(), eligible_audio=eligible)
        self.assertTrue(result["exact_match"])
        self.assertEqual(result["suppressed_native_bytes"], 2)
        self.assertFalse(pcm_evidence(raw, b"bb", set(raw), set(), eligible_audio=eligible)["exact_match"])

    def test_identical_partial_audio_is_not_native_completion(self):
        audio = {"answer": bytearray(b"\x01\x02")}
        self.assertFalse(pcm_evidence(audio, b"\x01\x02", set(), set())["native_completed"])
        self.assertFalse(pcm_evidence(audio, b"\x01\x02", {"answer"}, {"answer"})["native_completed"])
        self.assertTrue(pcm_evidence(audio, b"\x01\x02", {"answer"}, set())["native_completed"])
        self.assertFalse(pcm_evidence({}, b"", set(), set())["exact_match"])

    def test_redacts_audio_images_and_resume_handles_without_mutating_source(self):
        event = {"type": "response.output_audio.delta", "delta": "private-pcm",
                 "audio_base64": "phone-pcm", "nested": {"data": "image", "newHandle": "secret"},
                 "item": {"content": [{"type": "input_image", "image_url": "data:private-image"}]}}
        result = safe_event(event)
        self.assertNotIn("private", str(result))
        self.assertNotIn("secret", str(result))
        self.assertEqual(event["delta"], "private-pcm")
        self.assertEqual(result["type"], event["type"])

    def test_text_delta_is_preserved(self):
        event = {"type": "response.output_audio_transcript.delta", "delta": "spoken words"}
        self.assertEqual(safe_event(event), event)

    def test_session_update_records_only_a_digest(self):
        event = {"type": "session.update", "session": {"instructions": "fixture context"}}
        result = safe_event(event)
        self.assertEqual(set(result), {"type", "sha256"})
        self.assertEqual(len(result["sha256"]), 64)
        self.assertEqual(result, safe_event(event))

    def test_latency_retains_slow_failures_and_missing_measurements(self):
        results = [
            {"provider": "gemini", "transport_passed": True,
             "latency": {"input_end_to_app_audio_seconds": 1.0}},
            {"provider": "gemini", "transport_passed": False,
             "latency": {"input_end_to_app_audio_seconds": 9.0}},
            {"provider": "gemini", "transport_passed": False, "latency": {}},
        ]
        result = latency_summary(results)
        self.assertEqual(result["gemini"], {"attempts": 3, "transport_failures": 2,
            "measured_samples": 2, "missing_samples": 1, "median_seconds": 5.0,
            "max_seconds": 9.0, "p95_nearest_rank_seconds": 9.0})
        self.assertIsNone(result["openai"]["median_seconds"])
