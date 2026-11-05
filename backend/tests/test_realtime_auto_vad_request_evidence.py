"""Native auto-VAD request provenance through actual session/tool handlers.

Only provider transport, application output and canonical persistence are local
fixtures. No capture binding methods are called by the tests, no user transcript
is inferred from timing, and no real tool effect/provider/SQL/native I/O occurs.
"""

import asyncio
from contextlib import AsyncExitStack
import inspect
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.models import RealtimeSessionContext
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tools.invocation import current_tool_invocation
from app.tools.registry import ToolDefinition
from tests.test_second_brain_capture import CaptureRepository
from test_attention_quiet import LedgerFixture


class CanonicalFixtureRepository(CaptureRepository):
    """The shared fixture already returns the immutable canonical source DTO."""


class NativeProvider:
    capture_mode = "auto"
    transport = "webrtc"

    def __init__(self):
        self.events = []
        self.lock = asyncio.Lock()
        self.close = AsyncMock()

    async def send_event(self, event, *, write_guard=None):
        async with self.lock, AsyncExitStack() as stack:
            if write_guard is not None:
                await stack.enter_async_context(write_guard())
            event = event() if callable(event) else event
            if inspect.isawaitable(event):
                event = await event
            if event is None:
                return False
            self.events.append(event)
            return True


class NativeAutoVADRequestEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        quiet = LedgerFixture()
        quiet.is_in_transaction = lambda: quiet.lock.locked()
        repository = SimpleNamespace(_pool=quiet,
            create_message=AsyncMock(return_value=SimpleNamespace(id=uuid4())),
            get_recent_messages=AsyncMock(return_value=[]))
        self.session = DirectRealtimeAudioSession(
            offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            websocket=SimpleNamespace(close=AsyncMock(), send_text=AsyncMock()),
            context=RealtimeSessionContext(audio_session_id="auto-vad-fixture", owner_id="fixture",
                jarvis_session_id=uuid4(), client_kind="rayban_audio", session=SimpleNamespace()),
            settings=JarvisSettings(_env_file=None), repository=repository,
            tool_modules=(), context_providers=())
        self.provider = NativeProvider()
        self.session.openai = self.provider
        self.session._realtime_context_ready.set()
        self.session._send_app_event = AsyncMock(return_value=True)
        self.store = CanonicalFixtureRepository()
        self.capture = SessionContextCapture(self.store,
            owner_id=self.session.context.owner_id, session_id=self.session.context.jarvis_session_id,
            provider="openai", audio_session_id=self.session.context.audio_session_id)
        self.session._context_capture = self.capture
        self.session.tool_executor._context_capture = self.capture
        # Canonical capture DTOs now carry payload roots, so their real follow-up
        # guard needs the maintained source-storage fixture, not the quiet ledger.
        from test_realtime_tool_followup_delivery import _Sources
        self.session._source_context.pool = _Sources()
        self.invocations = []

        async def inspect_request(arguments):
            self.invocations.append((current_tool_invocation(), arguments))
            return {"status": "success", "meaning": "fixture evidence inspection only"}

        self.session.tool_registry.register(ToolDefinition(name="fixture_request_evidence",
            description="Inspect request provenance without side effects.",
            input_schema={"type": "object"}, handler=inspect_request))

    async def asyncTearDown(self):
        await self.session.close("auto_vad_fixture_complete")
        await self.capture.close()

    async def until(self, predicate):
        async with asyncio.timeout(1):
            while not predicate():
                await asyncio.sleep(0)

    def creates(self):
        return [event for event in self.provider.events if event["type"] == "response.create"]

    async def event(self, kind, **fields):
        await self.session._handle_openai_event({"type": kind, **fields})

    async def start(self, item_id):
        await self.event("input_audio_buffer.speech_started", item_id=item_id)

    async def commit_and_ack_response(self, item_id, response_id):
        before = len(self.creates())
        await self.event("input_audio_buffer.speech_stopped", item_id=item_id)
        await self.event("input_audio_buffer.committed", item_id=item_id)
        await self.until(lambda: len(self.creates()) > before)
        request = self.creates()[-1]
        await self.event("response.created", response={
            "id": response_id, "metadata": request["response"]["metadata"]})
        self.assertIn(response_id, self.session._active_responses)
        return request

    async def transcript(self, item_id, text):
        await self.event("conversation.item.input_audio_transcription.completed",
            item_id=item_id, transcript=text)
        self.assertTrue(await self.capture.flush())

    async def tool(self, response_id, call_id, arguments=None):
        item = {"id": call_id + "-item", "type": "function_call", "call_id": call_id,
            "name": "fixture_request_evidence", "arguments": json.dumps(arguments or {})}
        await self.event("response.output_item.done", response_id=response_id, item=item)
        await self.event("response.done", response={
            "id": response_id, "status": "completed", "output": [item]})

    def assert_no_explicit_commit(self):
        self.assertFalse(any(event["type"] == "input_audio_buffer.commit" for event in self.provider.events))

    async def test_native_auto_vad_binds_exact_final_request_through_tool_executor(self):
        await self.start("input-auto")
        generation = self.session._turn_generation
        await self.start("input-auto")  # Duplicate start cannot create authority twice.
        self.assertEqual(self.session._turn_generation, generation)
        await self.commit_and_ack_response("input-auto", "response-auto")
        await self.event("input_audio_buffer.committed", item_id="input-auto")
        self.assertEqual(len(self.creates()), 1)
        await self.transcript("input-auto", "Compare three fixture layouts in the background.")
        forged_id = str(uuid4())
        await self.tool("response-auto", "call-auto", {"request_event_id": forged_id})
        await self.until(lambda: len(self.invocations) == 1)
        invocation, arguments = self.invocations[0]
        evidence = await self.capture.request_for_call("call-auto")
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.content["provider_item_id"], "input-auto")
        self.assertEqual(evidence.finality, "final")
        self.assertEqual(invocation.provider_call_id, "call-auto")
        self.assertEqual(invocation.request_event_id, evidence.id)
        self.assertEqual(invocation.request_revision, evidence.revision)
        self.assertEqual(invocation.request_identity.namespace, self.capture.namespace)
        self.assertEqual(invocation.request_identity.event_id, evidence.source_event_id)
        self.assertNotEqual(str(invocation.request_event_id), arguments["request_event_id"])
        self.assertEqual(len(self.store.of_kind("user_transcript")), 1)
        self.assert_no_explicit_commit()

    async def test_tool_before_final_transcript_has_no_authority_or_retroactive_execution(self):
        await self.start("input-late")
        await self.commit_and_ack_response("input-late", "response-late")
        await self.tool("response-late", "call-late")
        await self.until(lambda: len(self.invocations) == 1)
        invocation = self.invocations[0][0]
        self.assertIsNone(invocation.request_event_id)
        self.assertIsNone(invocation.request_revision)
        self.assertIsNone(invocation.request_identity)
        self.assertIsNone(await self.capture.request_for_call("call-late"))
        await self.transcript("input-late", "The exact delayed request.")
        evidence = await self.capture.request_for_call("call-late")
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.content["provider_item_id"], "input-late")
        self.assertEqual(evidence.content["text"], "The exact delayed request.")
        self.assertEqual(len(self.invocations), 1)
        self.assertIsNone(invocation.request_event_id)
        self.assert_no_explicit_commit()

    async def test_interrupted_old_turn_and_reversed_transcripts_do_not_steal_new_authority(self):
        from app.realtime.source_context import LiveResponseSources
        from test_realtime_unwritten_request_sources import CanonicalStore, Sources
        sources = Sources()
        self.store = CanonicalStore(sources)
        self.capture.repository = self.store
        self.session._source_context = LiveResponseSources(sources,
            owner_id=self.session.context.owner_id, session_id=self.session.context.jarvis_session_id,
            audio_session_id=self.session.context.audio_session_id)
        await self.start("input-old")
        await self.commit_and_ack_response("input-old", "response-old")
        await self.start("input-new")
        await self.event("response.done", response={"id": "response-old", "status": "cancelled", "output": []})
        await self.commit_and_ack_response("input-new", "response-new")
        # Equal prose and reversed transcription arrival must not join identities.
        for item_id in ("input-new", "input-old"):
            await self.transcript(item_id, "Check that fixture plan.")
        # The cancelled empty response now contributes an unresolved reference,
        # not authority over the new utterance. Choose the actual current input.
        offer = self.session._active_responses["response-new"].request_sources
        current = next(source for source in offer.requests if source.kind == "current")
        retained = next(source for source in offer.requests if source.kind == "retained_cancelled_empty")
        self.assertIsNone(retained.record)  # Its late text cannot alter the written offer.
        await self.tool("response-new", "call-new", {"jarvis_request_source": current.reference})
        await self.until(lambda: len(self.invocations) == 1)
        evidence = await self.capture.request_for_call("call-new")
        self.assertEqual(evidence.content["provider_item_id"], "input-new")
        old = next(row for row in self.store.rows.values()
                   if row.kind == "user_transcript" and row.content["provider_item_id"] == "input-old")
        self.assertNotEqual(old.id, evidence.id)
        self.assertEqual(self.invocations[0][0].request_event_id, evidence.id)
        await self.tool("response-old", "call-stale")
        self.assertIsNone(await self.capture.request_for_call("call-stale"))
        self.assertEqual(len(self.invocations), 1)
        self.assert_no_explicit_commit()

    async def test_untracked_commit_and_transcript_do_not_invent_an_admitted_generation(self):
        await self.event("input_audio_buffer.committed", item_id="untracked")
        await self.transcript("untracked", "An untracked fixture request.")
        self.assertEqual(self.creates(), [])
        self.assertEqual(self.session._direct_inputs, {})
        self.assertEqual(self.capture._generations, {})
        self.assertIsNone(await self.capture.request_for_call("unbound-call"))
        self.assertEqual(len(self.store.of_kind("user_transcript")), 1)
        self.assertEqual(self.invocations, [])
        self.assert_no_explicit_commit()
