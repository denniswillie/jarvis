"""Canonical capture contracts and hooks through both production voice sessions."""
import asyncio
import base64
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from uuid import uuid4

import test_generic_visual_speech as gpt_fixture
import test_gemini_audio_session as gemini_fixture

from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextRecord
from app.tasks.models import TaskError
from app.tools.executor import JarvisToolExecutor
from app.tools.invocation import current_tool_invocation
from app.tools.registry import JarvisToolRegistry, ToolDefinition


AUDIO = base64.b64encode(b"\x01\x00" * 2400).decode()


class CaptureRepository:
    """Isolated append/replay fixture, including the production finality fence."""

    def __init__(self):
        self.rows = {}
        self.receipts = {}
        self.events = []
        self.fail = False

    async def append(self, owner, event):
        if self.fail:
            raise RuntimeError("fixture database unavailable")
        key = owner, event.source_namespace, event.source_event_id
        receipt = (*key, event.receipt_id)
        if receipt in self.receipts:
            return self.receipts[receipt]
        prior = self.rows.get(key)
        if prior and prior.finality in {"final", "revised"} and event.finality in {"provisional", "unavailable"}:
            raise TaskError("context_finality_conflict", "Fixture finalized source cannot regress")
        row = ContextRecord(id=uuid4(), owner_id=owner, revision=prior.revision + 1 if prior else 1,
            finality=event.finality, content=event.payload, session_id=event.session_id,
            source_event_id=event.source_event_id, kind=event.kind, trust=event.trust,
            source_namespace=event.source_namespace, sequence=len(self.events) + 1,
            origin_sequence=prior.origin_sequence if prior else len(self.events) + 1,
            supersedes_event_id=prior.id if prior else None, source_time=event.source_time,
            recorded_at=datetime.now(UTC), payload_id=uuid4(), availability="available",
            content_digest=None, is_current=True)
        self.rows[key] = self.receipts[receipt] = row
        self.events.append(event)
        return row

    async def request_evidence(self, owner, session, *, source_namespace, source_event_id):
        row = self.rows.get((owner, source_namespace, source_event_id))
        if row is None or row.session_id != session or row.finality not in {"final", "revised"}:
            raise TaskError("context_request_not_finalized", "Fixture missing exact transcript")
        return row

    def of_kind(self, kind):
        return [row for row in self.rows.values() if row.kind == kind]


class VoiceCaptureTests(unittest.IsolatedAsyncioTestCase):
    def capture(self, provider="openai", **kwargs):
        repo = CaptureRepository()
        capture = SessionContextCapture(repo, owner_id="fixture", session_id=uuid4(),
                                        provider=provider, audio_session_id="fixture-audio", **kwargs)
        self.addAsyncCleanup(capture.close)
        return capture, repo

    async def test_late_transcript_and_duplicate_ack_never_authorize_the_wrong_input(self):
        capture, repo = self.capture()
        capture.admit_openai_commit("first", 1)
        capture.admit_openai_commit("second", 2)
        capture.openai_committed("provider-first")
        capture.openai_committed("provider-first")
        capture.openai_committed("provider-second")
        capture.bind_openai_response("response-first", 1)
        capture.bind_openai_call("call-first", "response-first")
        capture.openai_transcript("provider-second", "A different later request")
        self.assertIsNone(await capture.request_for_call("call-first"))
        capture.openai_transcript("provider-first", "The actual original request")
        request = await capture.request_for_call("call-first")
        self.assertEqual(request.content["text"], "The actual original request")
        self.assertEqual(request.content["provider_item_id"], "provider-first")
        self.assertEqual(len(repo.of_kind("user_transcript")), 2)
        capture.bind_openai_response("proactive", None)
        capture.bind_openai_call("call-proactive", "proactive")
        self.assertIsNone(await capture.request_for_call("call-proactive"))

    async def test_rejected_commit_does_not_steal_next_ack(self):
        capture, _ = self.capture()
        capture.admit_openai_commit("bad", 1)
        capture.admit_openai_commit("good", 2)
        capture.openai_commit_rejected("bad")
        capture.openai_committed("accepted")
        capture.openai_transcript("accepted", "Accepted question")
        capture.bind_openai_response("response", 2)
        capture.bind_openai_call("call", "response")
        self.assertEqual((await capture.request_for_call("call")).content["text"], "Accepted question")

    async def test_corrections_back_to_previous_text_are_new_revisions_not_old_receipts(self):
        capture, repo = self.capture()
        for text in ("A", "B", "A", "B"):
            capture.input_final("input", text)
            self.assertTrue(await capture.flush())
        row = repo.of_kind("user_transcript")[0]
        self.assertEqual((row.content["text"], row.revision), ("B", 4))
        capture.input_final("input", "B")
        await capture.flush()
        self.assertEqual(len(repo.events), 4)
        self.assertEqual(len({event.source_time for event in repo.events}), 1)
        self.assertIsNotNone(repo.events[0].source_time)

    async def test_partial_audio_and_playback_boundary_do_not_invent_heard_text(self):
        capture, repo = self.capture()
        capture.observe_output({"type": "audio.output.delta", "item_id": "output", "response_id": "response",
                                "audio_base64": AUDIO}, forwarded=True)
        capture.observe_output({"type": "transcript.delta", "item_id": "output", "response_id": "response",
                                "text": "A partial answer"}, forwarded=True)
        self.assertFalse(capture.playback("output", end_ms=101))
        self.assertFalse(capture.playback("unknown", end_ms=0))
        self.assertFalse(capture.playback("output", end_ms=True))
        self.assertTrue(capture.playback("output", end_ms=50))
        capture.finish_response("response", interrupted=True)
        await capture.flush()
        output = repo.of_kind("assistant_output")[0].content
        self.assertEqual(output["text"], "A partial answer")
        self.assertEqual(output["completion_state"], "interrupted")
        self.assertEqual(output["playback_status"], "unknown")
        self.assertIsNone(output["heard_transcript"])
        self.assertNotIn("audio_base64", str(repo.events))
        self.assertEqual(repo.of_kind("played_output_boundary")[0].content["audio_end_ms"], 50)

    async def test_close_retains_uncheckpointed_partial_text_and_missing_user_input(self):
        capture, repo = self.capture("gemini")
        capture.input_admitted("missing")
        for text in ("first ", "last"):
            capture.observe_output({"type": "transcript.delta", "item_id": "item", "response_id": "response",
                                    "text": text}, forwarded=False)
            await capture.flush()
        await capture.close()
        self.assertEqual(repo.of_kind("assistant_output")[0].content["text"], "first last")
        self.assertEqual(repo.of_kind("user_transcript")[0].content["transcript_status"], "session_closed_without_transcript")
        self.assertIsNone(await capture.request_for_call("unbound"))

    async def test_capture_failure_is_explicit_and_revokes_request_authority(self):
        capture, repo = self.capture()
        repo.fail = True
        capture.input_final("input", "A real request")
        capture.bind_call("call", "input")
        self.assertFalse(await capture.flush())
        self.assertIsNone(await capture.request_for_call("call"))
        repo.fail = False
        capture.input_admitted("next")
        await capture.close()
        self.assertIsNotNone(capture.diagnostics()["issue"])
        self.assertEqual(len(repo.of_kind("context_gap")), 1)

    async def test_native_gemini_snapshot_retains_audio_generated_after_interruption(self):
        capture, repo = self.capture("gemini")
        output = SimpleNamespace(item_id="item", response_id="response", transcript="Generated but not heard",
            provider_audio_bytes=9600, forwarded_audio_bytes=4800, interrupted=True, completed=True)
        capture.gemini_output(output)
        await capture.flush()
        self.assertEqual(repo.of_kind("assistant_output")[0].content["provider_audio_bytes"], 9600)
        self.assertEqual(repo.of_kind("assistant_output")[0].content["forwarded_audio_bytes"], 4800)
        self.assertFalse(capture.playback("item", end_ms=101))
        capture.gemini_output(output)
        await capture.flush()
        self.assertEqual(len(repo.events), 1)

    async def test_queue_overflow_keeps_explicit_gap_and_does_not_block_ingress(self):
        capture, repo = self.capture(max_pending=3)
        for i in range(8):
            capture.input_final(str(i), "Distinct request")
        self.assertLessEqual(len(capture._pending), 3)
        await capture.flush()
        self.assertEqual(len(repo.of_kind("context_gap")), 1)
        self.assertGreater(capture.failed_events, 0)

    async def test_executor_uses_trusted_call_context_not_arguments_and_records_actual_result(self):
        capture, repo = self.capture()
        capture.input_final("input", "Remember my fixture preference")
        capture.bind_call("call", "input")
        contexts = []

        async def handler(arguments):
            contexts.append(current_tool_invocation())
            return {"status": "success", "value": "stored fixture"}

        registry = JarvisToolRegistry()
        registry.register(ToolDefinition(name="fixture", description="Fixture", input_schema={"type": "object"}, handler=handler))
        executor = JarvisToolExecutor(registry, context_capture=capture)
        result = await executor.execute("fixture", {"request_event_id": str(uuid4())}, provider_call_id="call")
        await capture.flush()
        request = await capture.request_for_call("call")
        self.assertEqual(contexts[0].request_event_id, request.id)
        self.assertEqual(repo.of_kind("tool_outcome")[0].content["result"], result)
        await executor.execute("fixture", {}, provider_call_id="unrelated")
        self.assertIsNone(contexts[-1].request_event_id)
        self.assertIsNone(current_tool_invocation().request_event_id)


class VoiceCaptureProviderHookTests(unittest.IsolatedAsyncioTestCase):
    def attach(self, session, provider):
        repo = CaptureRepository()
        capture = SessionContextCapture(repo, owner_id=session.context.owner_id,
            session_id=session.context.jarvis_session_id, provider=provider, audio_session_id=session.context.audio_session_id)
        session._context_capture = capture
        executor = session.tool_executor if provider == "openai" else session.operations.executor
        executor._context_capture = capture
        self.addAsyncCleanup(capture.close)
        return capture, repo

    async def test_gpt_production_hooks_keep_partial_output_and_exact_late_user_reference(self):
        h = gpt_fixture.GenericVisualSpeechTests()
        await h.asyncSetUp()
        self.addAsyncCleanup(h.asyncTearDown)
        capture, repo = self.attach(h.session, "openai")
        await h.start_user_response()
        await h.session._handle_openai_event({"type": "input_audio_buffer.committed", "item_id": "input-1"})
        await h.acknowledge(response_id="response-1")
        capture.bind_openai_call("fixture-call", "response-1")
        self.assertIsNone(await capture.request_for_call("fixture-call"))
        await h.session._handle_openai_event({"type": "conversation.item.input_audio_transcription.completed",
            "item_id": "input-1", "transcript": "The exact spoken request"})
        self.assertEqual((await capture.request_for_call("fixture-call")).content["text"], "The exact spoken request")
        for event in ({"type": "response.output_audio.delta", "delta": AUDIO},
                      {"type": "response.output_audio_transcript.delta", "delta": "Interrupted explanation"}):
            await h.session._handle_openai_event({**event, "item_id": "output-1", "response_id": "response-1"})
        await h.session._handle_client_event({"type": "playback.truncated", "item_id": "output-1", "audio_end_ms": 50})
        await h.session._handle_client_event({"type": "response.cancel"})
        await capture.flush()
        self.assertEqual(repo.of_kind("assistant_output")[0].content["text"], "Interrupted explanation")
        self.assertEqual(repo.of_kind("assistant_output")[0].content["completion_state"], "interrupted")
        self.assertEqual(len(repo.of_kind("played_output_boundary")), 1)

    async def test_gemini_production_hooks_survive_partial_output_and_missing_next_transcript(self):
        h = gemini_fixture.GeminiAudioSessionTests()
        await h.asyncSetUp()
        self.addAsyncCleanup(h.asyncTearDown)
        capture, repo = self.attach(h.session, "gemini")
        h.session.journal.submit = Mock()
        await h.start()
        await h.append("first")
        await h.commit("first")
        await h.wires[-1].publish({"inputTranscription": {"text": "First spoken request"}})
        await gemini_fixture.until(lambda: bool(h.events("user.transcript.done")))
        first = h.session._turns["first"]
        capture.bind_call("call", first.input_item_id)
        self.assertEqual((await capture.request_for_call("call")).content["text"], "First spoken request")
        await h.answer(text="Partial explanation", terminal=False)
        await gemini_fixture.until(lambda: bool(h.events("audio.output.delta")))
        h.session._record_playback({"item_id": first.output.item_id, "audio_end_ms": 5})
        await h.append("missing")
        await h.session.close("fixture_disconnect")
        self.assertEqual(len(repo.of_kind("user_transcript")), 2)
        self.assertIn("Partial explanation", [row.content["text"] for row in repo.of_kind("assistant_output")])
        self.assertEqual(len(repo.of_kind("played_output_boundary")), 1)
        self.assertEqual(capture.diagnostics()["missing_transcripts"], 1)
        self.assertIsNone(capture.issue)
