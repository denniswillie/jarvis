import asyncio
from copy import deepcopy
from contextlib import ExitStack
from datetime import UTC, datetime
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.realtime.gemini.client import GeminiTransportError
from app.tasks.model_gateway import ModelEvent, ModelGatewayError, TaskModelGateway
from app.tasks.models import digest
from app.tasks.repository import TaskRepository
from evals.second_brain.voice_phase_profile import CURRENT_CALL, Profile

from evals.second_brain.native_voice_tasks import (
    MissingTranscriptCapture, ObservedExecutor, ObservedGeminiConnection, ObservedPhone, ObservedResponsesUpstream, ObservedTaskModelGateway,
    completed_audio_ids, native_metadata, paired_context_overhead, safe_app_event, speech_texts, terminal_metadata,
)


class NativeVoiceObservationTests(unittest.IsolatedAsyncioTestCase):

    async def test_native_tool_profiling_is_call_local_and_restores_parent_after_failure(self):
        seen, rows = {}, []
        both_entered = asyncio.Event()
        class Real:
            async def execute(self, name, arguments, **kwargs):
                seen[name] = CURRENT_CALL.get()
                if len(seen) == 2:
                    both_entered.set()
                await both_entered.wait()
                self_check = CURRENT_CALL.get()
                if self_check != seen[name]:
                    raise AssertionError("Concurrent profile contexts leaked")
                if name == "fail":
                    raise ValueError("PRIVATE-FAILURE")
                return {"status": "success"}
        wrapper = ObservedExecutor(Real(), rows, 0, profile=Profile())
        parent = CURRENT_CALL.set("parent")
        try:
            results = await asyncio.gather(wrapper.execute("ok", {}),
                                           wrapper.execute("fail", {}), return_exceptions=True)
            self.assertEqual(results[0], {"status": "success"})
            self.assertIsInstance(results[1], ValueError)
            self.assertEqual(CURRENT_CALL.get(), "parent")
            self.assertEqual(len(set(seen.values())), 2)
            self.assertTrue(all(value.startswith("native:") for value in seen.values()))
            self.assertEqual({row["profile_call"] for row in rows}, set(seen.values()))
            class Cancelled:
                async def execute(self, *args, **kwargs):
                    raise asyncio.CancelledError()
            with self.assertRaises(asyncio.CancelledError):
                await ObservedExecutor(Cancelled(), [], 0, profile=Profile()).execute("cancel", {})
            self.assertEqual(CURRENT_CALL.get(), "parent")
        finally:
            CURRENT_CALL.reset(parent)

    def test_native_profile_reuses_digest_only_logger_and_restorable_patches(self):
        profile = Profile()
        original = TaskRepository.create
        with ExitStack() as patches:
            profile.install(patches)
            self.assertIsNot(TaskRepository.create, original)
        self.assertIs(TaskRepository.create, original)
        record = SimpleNamespace(query="select 'PRIVATE-QUERY'", elapsed=.1,
                                 exception=ValueError("PRIVATE-ERROR"), args=("PRIVATE-ARGUMENT",))
        profile.query(record)
        self.assertEqual(profile.queries, [])
        token = CURRENT_CALL.set("owned-call")
        try:
            profile.query(record)
        finally:
            CURRENT_CALL.reset(token)
        self.assertEqual(len(profile.queries), 1)
        self.assertEqual(profile.queries[0]["call"], "owned-call")
        self.assertEqual(profile.queries[0]["error_type"], "ValueError")
        self.assertNotIn("PRIVATE", json.dumps(profile.queries))


    def context_request(self):
        return {"owner_id": "fixture", "session": None, "recent_messages": [],
                "current_utc_time": datetime(2026, 9, 15, tzinfo=UTC)}

    async def test_context_pair_uses_same_payload_and_alternates_real_builder_order(self):
        class Tasks:
            async def context_summaries(self, owner, *, limit):
                return {"total": 2, "tasks": [], "terminal_total": 0, "recent_terminal_tasks": []}
        report = await paired_context_overhead(SimpleNamespace(tasks=Tasks()), self.context_request(),
                                               lambda: ["b", "a"], samples=3)
        self.assertEqual(report["qualified_pairs"], 3)
        self.assertTrue(report["all_requested_pairs_qualified"])
        self.assertEqual(report["required_native_tasks"], 2)
        self.assertEqual([row["measurements"][0]["task_context_enabled"] for row in report["rows"]],
                         [False, True, False])
        self.assertTrue(all(row["baseline_payload_equal"] for row in report["rows"]))
        self.assertEqual(report["added_application_context"]["n"], 3)

    async def test_context_pair_does_not_count_stopped_native_work_or_fallback(self):
        class Tasks:
            async def context_summaries(self, owner, *, limit):
                return {"total": 2, "tasks": [], "terminal_total": 0, "recent_terminal_tasks": []}
        sources = SimpleNamespace(tasks=Tasks())
        empty = await paired_context_overhead(sources, self.context_request(), lambda: ["a"], samples=2)
        self.assertEqual(empty["rows"], [])
        self.assertFalse(empty["target_met"])
        calls = 0
        def ends_during_pair():
            nonlocal calls
            calls += 1
            return ["a", "b"] if calls < 3 else ["a"]
        ended = await paired_context_overhead(sources, self.context_request(), ends_during_pair, samples=2)
        self.assertEqual(len(ended["rows"]), 1)
        self.assertEqual(ended["qualified_pairs"], 0)
        self.assertIsNone(ended["added_application_context"])
        class FailedTasks:
            async def context_summaries(self, owner, *, limit):
                raise ValueError("PRIVATE-FIXTURE-ERROR")
        with self.assertLogs("app.assistant.context", level="WARNING"):
            failed = await paired_context_overhead(
                SimpleNamespace(tasks=FailedTasks()), self.context_request(), lambda: ["a", "b"], samples=1)
        self.assertEqual(failed["qualified_pairs"], 0)
        self.assertNotIn("PRIVATE-FIXTURE-ERROR", json.dumps(failed))
        self.assertFalse(failed["target_met"])

    async def test_native_result_observer_covers_actual_serialized_write_without_changing_payload(self):
        rows, sent = [], []
        class Socket:
            async def send(self, wire):
                sent.append(wire)
        connection = ObservedGeminiConnection(rows=rows, clock=0,
            config=SimpleNamespace(write_timeout_seconds=1, close_timeout_seconds=1))
        connection._socket = Socket()
        payload = {"status": "success", "task": {"task_ref": "task_0123456789ab", "revision": 2, "state": "succeeded"},
            "expert_output_available": True,
            "answer_delivery": {"route": "direct_tts", "state": "unavailable",
                                "reason": "attention_coordinator_unavailable"}}
        event = {"toolResponse": {"functionResponses": [{"id": "native", "name": "get_background_task",
                                                         "response": payload}]}}
        original = deepcopy(event)
        await connection._write(event)
        self.assertEqual(event, original)
        self.assertEqual(json.loads(sent[0]), original)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["socket_write_completed"])
        self.assertEqual(rows[0]["result_digest"], digest(payload))
        self.assertEqual(rows[0]["answer_delivery"], payload["answer_delivery"])
        self.assertNotIn("answer_digest", rows[0])
        self.assertEqual(rows[0]["task_state"], "succeeded")
        self.assertNotIn("expert_result", json.dumps(rows))
        await connection._write({"realtimeInput": {"audio": "PRIVATE-AUDIO"}})
        self.assertEqual(len(rows), 1)

    async def test_native_result_send_failure_is_not_recorded_as_delivered_or_retried(self):
        rows, sent, closed = [], [], []
        class Socket:
            async def send(self, wire):
                sent.append(wire)
                raise OSError("PRIVATE-SOCKET-ERROR")
            async def close(self):
                closed.append(True)
        connection = ObservedGeminiConnection(rows=rows, clock=0,
            config=SimpleNamespace(write_timeout_seconds=1, close_timeout_seconds=1))
        connection._socket = Socket()
        with self.assertRaises(GeminiTransportError):
            await connection._write({"toolResponse": {"functionResponses": [
                {"id": "native", "name": "get_background_task", "response": {"status": "success"}}]}})
        self.assertEqual(len(sent), 1)
        self.assertEqual(closed, [True])
        self.assertFalse(rows[0]["socket_write_completed"])
        self.assertEqual(rows[0]["error_code"], "gemini_write_failed")
        self.assertNotIn("PRIVATE-SOCKET-ERROR", json.dumps(rows))

    def test_missing_transcript_fixture_uses_production_unavailable_capture(self):
        capture = object.__new__(MissingTranscriptCapture)
        with patch.object(capture, "input_unavailable") as unavailable:
            capture.input_final("synthetic-input", "A transcript the fixture must suppress.")
        unavailable.assert_called_once_with("synthetic-input", reason="synthetic_missing_provider_transcript")

    def test_tool_only_done_is_not_a_spoken_answer(self):
        events = [{"type": "response.done", "response_id": "tool"},
                  {"type": "audio.output.delta", "response_id": "answer"}]
        self.assertEqual(completed_audio_ids(events), set())
        events.append({"type": "response.done", "response_id": "answer"})
        self.assertEqual(completed_audio_ids(events), {"answer"})

    def test_transcript_selection_and_pcm_redaction(self):
        events = [{"type": "transcript.done", "response_id": "old", "text": "Earlier"},
                  {"type": "transcript.done", "response_id": "new", "text": "Current"}]
        self.assertEqual(speech_texts(events, {"new"}), ["Current"])
        self.assertEqual(safe_app_event({"type": "audio.output.delta", "audio_base64": "secret-pcm"}),
                         {"type": "audio.output.delta"})

    async def test_executor_preserves_exact_result_and_failure(self):
        value = {"status": "success", "answer": "Exact expert text"}
        class Real:
            async def execute(self, name, arguments, **kwargs):
                self.call = name, arguments, kwargs
                if name == "fail":
                    raise ValueError("synthetic")
                return value
        real, rows = Real(), []
        wrapper = ObservedExecutor(real, rows, 0)
        self.assertIs(await wrapper.execute("get", {"ref": "exact"}, provider_call_id="native"), value)
        self.assertEqual(real.call, ("get", {"ref": "exact"}, {"provider_call_id": "native"}))
        with self.assertRaises(ValueError):
            await wrapper.execute("fail", {})
        self.assertEqual(rows[1]["error_type"], "ValueError")
        self.assertIn("duration_ms", rows[1])

    async def test_phone_observes_native_state_without_fabricating_playback(self):
        phone = ObservedPhone(lambda: ["task-a"])
        await phone.send_text('{"type":"response.done","response_id":"r"}')
        self.assertEqual(phone.observations[0]["native_running_tasks"], ["task-a"])
        self.assertEqual(phone.events[0]["type"], "response.done")
        self.assertTrue(phone.input.empty(), "A fixture must not synthesize heard receipts")

    def test_native_diagnostics_omit_raw_bodies_and_unknown_exit_messages(self):
        self.assertEqual(native_metadata({"completed": False, "failed": True, "partial": False,
            "interrupted": False, "api_calls": 1, "context_compressions": 0, "turn_exit_reason": "max_tokens",
            "final_response": "private original", "api_key": "not retained"}), {
            "completed": False, "failed": True, "partial": False, "interrupted": False,
            "api_calls": 1, "context_compressions": 0, "turn_exit_reason": "max_tokens"})
        self.assertNotIn("turn_exit_reason", native_metadata({"turn_exit_reason": "provider body / private"}))

    async def test_interrupt_is_one_real_client_cancel_after_accepted_audio(self):
        phone = ObservedPhone(lambda: ["task-a"])
        admitted = False
        phone.interrupt_when = lambda: admitted
        event = {"type": "audio.output.delta", "response_id": "r", "item_id": "i", "audio_base64": "AAA="}
        await phone.send_json(event)
        self.assertTrue(phone.input.empty())
        admitted = True
        await phone.send_json(event)
        await phone.send_json(event)
        self.assertEqual(phone.input.qsize(), 1)
        self.assertEqual((await phone.input.get())["type"], "response.cancel")
        self.assertFalse(phone.interruption["response_already_completed"])
        self.assertEqual(phone.interruption["native_running_tasks"], ["task-a"])
        self.assertEqual(phone.interruption["playback"], "unknown")

    def test_cancelled_or_failed_audio_cannot_complete_the_next_answer(self):
        for final in ({"interrupted": True}, {"status": "cancelled"}, {"status": "failed"},
                      {"status": "incomplete"}):
            with self.subTest(final=final):
                self.assertEqual(completed_audio_ids([
                    {"type": "audio.output.delta", "response_id": "old"},
                    {"type": "response.done", "response_id": "old", **final},
                ]), set())

    def test_terminal_receipts_keep_actual_usage_not_response_bodies(self):
        event = {"type": "response.incomplete", "response": {
            "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"input_tokens": 27, "output_tokens": 4096, "total_tokens": 4123,
                      "private": "not retained"}, "output": "private response",
            "error": {"code": "provider_error", "message": "private error"}}}
        self.assertEqual(terminal_metadata(event), {
            "event_type": "response.incomplete", "status": "incomplete",
            "incomplete_reason": "max_output_tokens", "error_code": "provider_error",
            "provider_usage": {"input_tokens": 27, "output_tokens": 4096, "total_tokens": 4123}})
        self.assertIsNone(terminal_metadata({"type": "response.output_text.delta", "delta": "private"}))
        self.assertEqual(terminal_metadata({"type": "response.failed", "response": {
            "usage": {"input_tokens": True, "output_tokens": -1},
            "error": {"code": "raw private body / never retain"}}})["provider_usage"], {})

    async def test_upstream_observer_preserves_payload_events_and_failure(self):
        payload, rows = {"private": "original"}, []
        original = ModelEvent({"type": "response.completed", "response": {
            "status": "completed", "usage": {"input_tokens": 3, "output_tokens": 7}}}, b"original-wire")
        failure = ModelGatewayError("model_provider_stream_incomplete")
        class Upstream:
            async def count(self, body):
                self.counted = body
                return 3
            async def events(self, body):
                self.streamed = body
                yield original
                raise failure
        real = Upstream()
        observed = ObservedResponsesUpstream(real, rows, 0, lambda: {"attempt_id": "owned"})
        self.assertEqual(await observed.count(payload), 3)
        stream = observed.events(payload)
        self.assertIs(await anext(stream), original)
        with self.assertRaises(ModelGatewayError) as caught:
            await anext(stream)
        self.assertIs(caught.exception, failure)
        self.assertIs(real.counted, payload)
        self.assertIs(real.streamed, payload)
        self.assertEqual(rows[1]["terminal_receipt"]["provider_usage"]["output_tokens"], 7)
        self.assertEqual(rows[1]["error_code"], failure.code)
        self.assertNotIn("private", str(rows))

    async def test_gateway_observer_preserves_uncertain_failure_without_retry(self):
        context = SimpleNamespace(task=SimpleNamespace(id="task"), claim=SimpleNamespace(attempt_id="attempt"))
        rows, calls, body = [], [], {"unchanged": True}
        failure = ModelGatewayError("model_usage_unknown")
        async def stream(gateway, payload, *, auxiliary=False):
            calls.append((payload, auxiliary))
            gateway._requests += 1
            gateway._uncertain = True
            yield b"original"
            raise failure
        gateway = ObservedTaskModelGateway(context, object(), diagnostics=rows, clock=0)
        with patch.object(TaskModelGateway, "stream", stream):
            output = gateway.stream(body, auxiliary=True)
            self.assertEqual(await anext(output), b"original")
            with self.assertRaises(ModelGatewayError) as caught:
                await anext(output)
        self.assertIs(caught.exception, failure)
        self.assertEqual(calls, [(body, True)])
        self.assertIs(calls[0][0], body)
        self.assertTrue(rows[0]["uncertain"])
        self.assertEqual(rows[0]["request_index"], 1)
        self.assertEqual(rows[0]["error_code"], "model_usage_unknown")
        self.assertNotIn("completed", rows[0])

    async def test_consuming_terminal_then_closing_is_not_provider_failure(self):
        event = ModelEvent({"type": "response.completed", "response": {
            "status": "completed", "usage": {"input_tokens": 2, "output_tokens": 5}}}, b"terminal")
        closed, rows = [], []
        class Upstream:
            async def events(self, payload):
                try:
                    yield event
                finally:
                    closed.append(True)
        observer = ObservedResponsesUpstream(Upstream(), rows, 0, lambda: {})
        stream = observer.events({})
        self.assertIs(await anext(stream), event)
        await stream.aclose()
        self.assertEqual(closed, [True])
        self.assertTrue(rows[0]["closed_by_consumer"])
        self.assertNotIn("error_type", rows[0])
        self.assertEqual(rows[0]["terminal_receipt"]["status"], "completed")
