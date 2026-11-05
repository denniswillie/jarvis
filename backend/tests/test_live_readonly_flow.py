"""Owner -> restricted Hermes preparation/verification -> guarded Live wire.

Storage, received peer events and the structured model answer are controlled
boundaries. No native agent process, provider/model calls or service starts.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.realtime.live_delegation import current_result, result_guard
from app.realtime.live_session import LiveAudioSession
from app.realtime.source_context import LiveResponseSources
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskError, TaskStatus
from app.tasks.context_events import ContextIngress
from app.voice_routes.contracts import TranscriptRevision, DeliveryEvidence
from app.voice_routes.live_adapter import LiveVoiceRoute, LiveTranscriptPart
from app.voice_routes.live_protocol import LiveTranscriptFragment
from dataclasses import replace
from test_live_readonly_delegation import ReadonlyOwnerTests
from test_live_voice_adapter import LiveFixture


class GuardHost:
    """External composition fixture; production result/cancel guards run intact."""
    def __init__(self, owner, fixture, receipt):
        self.adapter = fixture.route
        self.request = SimpleNamespace(route=owner.route)
        self.readonly_admission = owner.admission
        self._readonly_receipts = {receipt.delegation_id: receipt}
        self.live = SimpleNamespace(scope_guard=lambda: owner.tasks.transaction(owner.scope.owner_id))
        self._source_context = LiveResponseSources(owner.db, owner_id=owner.scope.owner_id,
            session_id=owner.scope.session_id, audio_session_id=owner.context.audio_session_id)

    @property
    def ready(self):
        return self.adapter._started and self.adapter._open() and not self.adapter.requires_fenced_replacement


class ReadonlyFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await ReadonlyOwnerTests.asyncSetUp(self)
        # Existing source/transport fixtures have no preference SQL engine.
        # Quiet admission itself is qualified in dedicated policy/SQL/writer tests.
        gate = patch("app.realtime.live_delegation.assert_unsolicited_allowed", AsyncMock())
        gate.start()
        self.addCleanup(gate.stop)
    admit = ReadonlyOwnerTests.admit
    prepare = ReadonlyOwnerTests.prepare

    async def asyncTearDown(self):
        if hasattr(self, "wire"):
            await self.wire.close()
        await ReadonlyOwnerTests.asyncTearDown(self)

    async def wire_start(self):
        self.wire = LiveFixture()
        self.wire.identity = self.route
        self.wire.route = LiveVoiceRoute(replace(self.wire.route._b, identity=self.route),
            clock=lambda: self.wire.now, io_timeout=.3, ack_timeout=.3, start_timeout=1)
        result = await self.wire.start()
        self.assertEqual(result.status, "completed")

    async def finish_model_boundary(self, receipt, *, answer="The available visual evidence does not establish the number.",
                                    status="waiting_for_input"):
        context, prepared = await self.prepare(receipt)
        # Only the model answer is supplied by the fixture. Canonical citation
        # checks, action verification, ownership, manifest and finish are real.
        proposed = HermesTaskResult(status=status, answer=answer,
            speech=[{"text": answer}], evidence=[{"event_id": self.observations[0].id, "revision": 1}],
            completed_operation_refs=[], uncertainties=["No legible current number is established."])
        outcome = await proposed.verified(context, prepared.tools)
        self.assertTrue(await self.tasks.finish(context.claim, outcome))
        async with self.tasks.transaction(self.scope.owner_id) as connection:
            result = await current_result(self.tasks, connection, receipt, self.route)
        return context, prepared, result

    async def real_flow(self, *, answer=None, status="waiting_for_input"):
        await self.wire_start()
        await self.wire.inject(type="session.input_transcript.delta", event_id="actual-user-fragment",
                               delta="What number is visible?", start_ms=1, end_ms=90)
        await self.wire.until(lambda: any(isinstance(e, TranscriptRevision) for e in self.wire.events))
        caption = next(e for e in self.wire.events if isinstance(e, TranscriptRevision))
        self.assertTrue(self.captions.transcript(caption, self.wire.route.transcript_parts))
        self.observations = await self.captions.observations(self.wire.route.transcript_parts)
        await self.wire.inject(type="session.delegation.created", event_id="actual-delegation", offset_ms=90,
            delegation={"id": "d1", "type": "delegation", "target": "client"})
        stream = self.wire.route.delegation_requests()
        async with asyncio.timeout(1):
            self.request = await anext(stream)
        await stream.aclose()
        receipt = await self.admit()
        context, prepared, result = await self.finish_model_boundary(
            receipt, status=status, **({"answer": answer} if answer else {}))
        host = GuardHost(self, self.wire, receipt)
        return receipt, context, prepared, result, host

    async def test_actual_admission_restricted_hermes_verified_finish_guarded_reply(self):
        receipt, context, prepared, result, host = await self.real_flow()
        response = await self.wire.route.submit_delegation_result(result,
            guard=lambda expected: result_guard(host, receipt, expected))
        self.assertTrue(response.provider_received)
        self.assertEqual((await self.tasks.get(self.scope.owner_id, receipt.task.id)).status, TaskStatus.WAITING)
        self.assertFalse(prepared.tools.actions)
        self.assertFalse(self.db.grants)
        self.assertTrue(host._source_context.has_sources)
        self.assertFalse(any(isinstance(event, DeliveryEvidence) for event in self.wire.events))
        self.assertFalse(any(record.authorizes_request for record in self.observations))
        stored = self.db.payloads[result.result_payload_id]["content"]
        self.assertEqual(stored["authorship"]["attempt_id"], str(context.claim.attempt_id))
        self.assertEqual(stored["verified_operations"], [])
        self.assertFalse(any(r["kind"] == "user_transcript" and r["finality"] == "final" for r in self.db.rows(self.scope.owner_id)))

    async def test_observation_waits_for_consumer_and_storage_commit_before_admission(self):
        part = LiveTranscriptPart(self.caption_id, 2, LiveTranscriptFragment("not-yet-enqueued", "user", " Clarify that.", 95, 120))
        parts = self.parts + (part,)
        entered, release = asyncio.Event(), asyncio.Event()
        self.db.fragment_gate = (entered, release)
        completed = asyncio.Event()
        async def waiting_admission():
            observations = await self.captions.observations(parts)
            admitted = await self.admit(observations=observations)
            completed.set()
            return admitted
        waiter = asyncio.create_task(waiting_admission())
        try:
            await asyncio.sleep(0)
            self.assertFalse(waiter.done())  # No capture writer was running yet.
            event = self.event.model_copy(update={"revision": 2, "text": self.event.text + part.fragment.delta})
            self.assertTrue(self.captions.transcript(event, parts))
            async with asyncio.timeout(1):
                await entered.wait()
            self.assertFalse(completed.is_set())
            self.assertFalse(self.db.tasks)
            release.set()
            async with asyncio.timeout(1):
                receipt = await waiter
            self.assertEqual(len(receipt.source.content["speech_sources"]), 2)
        finally:
            release.set()
            if not waiter.done():
                waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    async def test_close_wakes_pending_capture_without_admitting_a_task(self):
        part = LiveTranscriptPart(self.caption_id, 2, LiveTranscriptFragment("never-arrived", "user", " More.", 95, 120))
        waiter = asyncio.create_task(self.captions.observations(self.parts + (part,)))
        await asyncio.sleep(0)
        await self.captions.close()
        with self.assertRaises(TaskError):
            await waiter
        self.assertFalse(self.db.tasks)

    async def test_provisional_caption_update_preserves_its_available_originals(self):
        part = LiveTranscriptPart(self.caption_id, 2, LiveTranscriptFragment("more", "user", " More.", 95, 120))
        self.captions.transcript(self.event.model_copy(update={"revision": 2, "text": self.event.text + part.fragment.delta}),
                                 self.parts + (part,))
        await self.captions.observations(self.parts + (part,))
        self.assertTrue(await self.capture.flush())
        self.assertIsNone(self.capture.issue)
        current = [row for row in self.db.rows(self.scope.owner_id) if row["kind"] == "user_transcript" and row["is_current"]]
        self.assertEqual(current[0]["revision"], 2)
        self.assertEqual(current[0]["availability"], "not_finalized")

    async def test_completed_unsent_reply_cancelled_at_actual_writer_guard(self):
        receipt, _, _, result, host = await self.real_flow()
        entered, release = asyncio.Event(), asyncio.Event()
        @asynccontextmanager
        async def delayed_guard(expected):
            entered.set()
            await release.wait()
            async with result_guard(host, receipt, expected):
                yield
        send = asyncio.create_task(self.wire.route.submit_delegation_result(result, guard=delayed_guard))
        try:
            await entered.wait()
            self.assertTrue(await LiveAudioSession.cancel_readonly_delegation(host, "d1", expected_revision=1))
            release.set()
            response = await send
            self.assertFalse(response.provider_received)
            self.assertFalse(any(r["type"] == "session.commentary.append" for r in self.wire.wire.socket.sent))
            async with self.tasks.transaction(self.scope.owner_id) as connection:
                with self.assertRaises(TaskError):
                    await current_result(self.tasks, connection, receipt, self.route)
        finally:
            release.set()
            await asyncio.gather(send, return_exceptions=True)

    async def test_succeeded_history_preserved_but_delivery_authority_revoked(self):
        receipt, _, _, result, host = await self.real_flow()
        # A terminal computation is factual history, not permission to speak.
        self.db.tasks[receipt.task.id]["status"] = "succeeded"
        self.assertTrue(await LiveAudioSession.cancel_readonly_delegation(host, "d1", expected_revision=1, reason="superseded"))
        self.assertEqual((await self.tasks.get(self.scope.owner_id, receipt.task.id)).status, TaskStatus.SUCCEEDED)
        self.assertFalse(await self.admission.cancel(receipt, reason="superseded"))
        response = await self.wire.route.submit_delegation_result(result, guard=lambda r: result_guard(host, receipt, r))
        self.assertEqual(response.reason, "delegation_result_revoked")

    async def test_canonical_correction_before_send_blocks_actual_reply(self):
        receipt, _, _, result, host = await self.real_flow()
        original = self.observations[0]
        await self.sources.append(self.scope.owner_id, ContextIngress(session_id=self.scope.session_id,
            source_namespace=original.source_namespace, source_event_id=original.source_event_id,
            receipt_id="corrected", expected_revision=original.revision, kind="observation", trust="derived",
            payload=original.content | {"delta": "That number was not the question."}))
        response = await self.wire.route.submit_delegation_result(result, guard=lambda r: result_guard(host, receipt, r))
        self.assertFalse(response.provider_received)
        self.assertFalse(any(r["type"] == "session.commentary.append" for r in self.wire.wire.socket.sent))

    async def test_revocation_after_quiet_chunk_fences_without_final_commentary(self):
        receipt, _, _, result, host = await self.real_flow(answer="This is uncertain; inspect the original carefully. " * 30)
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0
        @asynccontextmanager
        async def second_chunk_guard(expected):
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
                await release.wait()
            async with result_guard(host, receipt, expected):
                yield
        send = asyncio.create_task(self.wire.route.submit_delegation_result(result, guard=second_chunk_guard))
        try:
            await entered.wait()
            await LiveAudioSession.cancel_readonly_delegation(host, "d1", expected_revision=1)
            self.assertTrue(self.wire.route.requires_fenced_replacement)
            release.set()
            await send
            await self.wire.until(lambda: bool(self.wire.fences))
            self.assertFalse(any(r["type"] == "session.commentary.append" for r in self.wire.wire.socket.sent))
        finally:
            release.set()
            await asyncio.gather(send, return_exceptions=True)

    async def test_wrong_revision_cannot_cancel_or_latch_the_result(self):
        receipt, _, _, result, host = await self.real_flow()
        with self.assertRaises(TaskError):
            await LiveAudioSession.cancel_readonly_delegation(host, "d1", expected_revision=2)
        response = await self.wire.route.submit_delegation_result(result, guard=lambda r: result_guard(host, receipt, r))
        self.assertTrue(response.provider_received)
