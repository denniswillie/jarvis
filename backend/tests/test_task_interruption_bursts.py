"""Task-side interruption bursts, not acoustic or PostgreSQL acceptance.

Real admission/capture/context/result owners, coordinator, Live route and session
event consumer. Existing SQL/transport boundaries are controlled. Worker claim
arrival is the existing fixture boundary, not scheduler/lease qualification.
No action grants, effect ledger, native child or real provider is substituted.
"""
import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.realtime.live_delegation import current_result, result_guard
from app.realtime.live_session import LiveAudioSession
from app.realtime.source_context import LiveResponseSources
from app.tasks.context_events import ContextIngress
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import ClaimLost, TaskError, TaskStatus, WorkerOutcome
from app.voice_routes.coordinator import VoiceRouteCoordinator
from app.voice_routes.live_adapter import LiveVoiceRoute
from readonly_live_store import ReadonlyStore
from test_live_readonly_delegation import ReadonlyOwnerTests
from test_live_voice_adapter import LiveFixture


class BurstStorageBoundary(ReadonlyStore):
    """Two missing SQL operations; lifecycle decisions stay in production owners."""
    async def execute(self, sql, *args):
        q = " ".join(sql.split()).lower()
        if q.startswith("update jarvis_task_attempts set lease_until=clock_timestamp()"):
            attempt = self.attempts[args[1]]
            assert self.tasks[attempt["task_id"]]["owner_id"] == args[0]
            attempt["lease_until"] = datetime.now(timezone.utc)
            return
        return await super().execute(sql, *args)

    async def fetchrow(self, sql, *args):
        row = await super().fetchrow(sql, *args)
        if row is not None and "lease_until>clock_timestamp() as lease_valid" in sql:
            row["lease_valid"] = row["lease_until"] > datetime.now(timezone.utc)
        return row

    async def fetchval(self, sql, *args):
        if "exists(select 1 from jarvis_task_attempts" in " ".join(sql.split()).lower():
            attempt = self.attempts.get(args[2])
            return bool(attempt and attempt["task_id"] == args[1]
                        and self.tasks[args[1]]["owner_id"] == args[0])
        return await super().fetchval(sql, *args)


class Barrier:
    def __init__(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def hold(self):
        self.entered.set()
        await asyncio.wait_for(self.release.wait(), 5)

    async def reached(self):
        await asyncio.wait_for(self.entered.wait(), 2)


class EventHost:
    """Composition boundary for the real session event consumer, not a task FSM."""
    def __init__(self, test):
        self.adapter, self.capture, self.context = test.wire.route, test.captions, test.context
        self.request = SimpleNamespace(route=test.route)
        self.readonly_admission = test.admission
        self.live = SimpleNamespace(scope_guard=lambda: test.tasks.transaction(test.scope.owner_id))
        self._source_context = LiveResponseSources(test.db, owner_id=test.scope.owner_id,
            session_id=test.scope.session_id, audio_session_id=test.context.audio_session_id)
        self._readonly_receipts = {}
        self._closed = asyncio.Event()
        self._attention_sink = None
        self._proactive_interrupt_gate = SimpleNamespace(reset_for_user_turn=lambda: None)
        self.messages, self.closing_reasons = [], []
        self._app_sender = SimpleNamespace(send_event=AsyncMock(side_effect=self.send_event))

    async def send_event(self, event, **kwargs):
        self.messages.append(event)

    def _mark_closing(self, reason):
        self.closing_reasons.append(reason)
        self._closed.set()

    @property
    def ready(self):
        return self.adapter._started and self.adapter._open() and not self.adapter.requires_fenced_replacement


class TaskInterruptionBurstTests(unittest.IsolatedAsyncioTestCase):
    admit = ReadonlyOwnerTests.admit
    prepare = ReadonlyOwnerTests.prepare

    async def asyncSetUp(self):
        self.jobs, self.barriers = [], []
        with patch("test_live_readonly_delegation.ReadonlyStore", BurstStorageBoundary):
            await ReadonlyOwnerTests.asyncSetUp(self)
        self.addAsyncCleanup(self.cleanup)
        # Existing storage fixture has no quiet-preference SQL. DND is not
        # accepted by this test; Euclid owns that independent delivery gate.
        gate = patch("app.realtime.live_delegation.assert_unsolicited_allowed", AsyncMock())
        gate.start()
        self.addCleanup(gate.stop)
        self.wire = LiveFixture()
        self.wire.identity = self.route
        self.wire.route = LiveVoiceRoute(replace(self.wire.route._b, identity=self.route),
            clock=lambda: self.wire.now, io_timeout=.5, ack_timeout=.5, start_timeout=2)
        self.host = EventHost(self)
        self.wire.pump = asyncio.create_task(LiveAudioSession._route_events(self.host))
        async def valid(update):
            return True  # Bootstrap authorization boundary, not result validation.
        self.coordinator = VoiceRouteCoordinator(self.scope, validity=valid,
            control_timeout=2, clock=lambda: self.wire.now)
        await asyncio.wait_for(self.coordinator.start(self.wire.route, self.wire.start_request()), 3)
        self.origins = {}

    async def cleanup(self):
        for barrier in self.barriers:
            barrier.release.set()
        for job in self.jobs:
            if not job.done():
                job.cancel()
        if self.jobs:
            await asyncio.wait_for(asyncio.gather(*self.jobs, return_exceptions=True), 3)
            self.assertTrue(all(job.done() for job in self.jobs))
        if hasattr(self, "coordinator"):
            await asyncio.wait_for(self.coordinator.close(), 3)
        if hasattr(self, "wire"):
            await asyncio.wait_for(self.wire.close(), 3)
            self.assertTrue(self.wire.wire.socket.closed)
            self.assertTrue(self.wire.pump.done())
        await ReadonlyOwnerTests.asyncTearDown(self)
        self.assertEqual(self.db.depth, 0)
        self.assertFalse(self.db.lock.locked())

    def barrier(self):
        barrier = Barrier()
        self.barriers.append(barrier)
        return barrier

    def launch(self, coroutine):
        job = asyncio.create_task(coroutine)
        self.jobs.append(job)
        return job

    async def say(self, text):
        before = len(self.wire.route.transcript_parts)
        identity = "speech-" + uuid4().hex
        await self.wire.inject(type="session.input_transcript.delta", event_id=identity,
            delta=text, start_ms=before * 100, end_ms=before * 100 + 90)
        await self.wire.until(lambda: len(self.wire.route.transcript_parts) == before + 1)
        observations = await asyncio.wait_for(self.captions.observations(self.wire.route.transcript_parts), 2)
        self.assertTrue(await self.capture.flush())
        self.assertTrue(all(not row.authorizes_request for row in observations))
        return (observations[-1],)

    async def question(self, label, *, observations=None):
        observations = observations or await self.say(" Investigate synthetic question " + label + ".")
        await self.wire.inject(type="session.delegation.created", event_id="event-" + label,
            offset_ms=90, delegation={"id": label, "type": "delegation", "target": "client"})
        stream = self.wire.route.delegation_requests()
        try:
            request = await asyncio.wait_for(anext(stream), 2)
        finally:
            await stream.aclose()
        receipt = await self.admit(delegation=request, observations=observations)
        self.host._readonly_receipts[label] = receipt
        self.origins[receipt.task.id] = observations[0]
        return receipt, request

    async def state(self, receipt):
        return await self.tasks.get(self.scope.owner_id, receipt.task.id)

    async def complete(self, receipt, prepared=None):
        context, bundle = prepared or await self.prepare(receipt)
        original = self.origins[receipt.task.id]
        answer = "Source-bound answer for " + receipt.delegation_id + ". No physical action was performed."
        proposed = HermesTaskResult(status="succeeded", answer=answer,
            speech=[{"text": answer}], evidence=[{"event_id": original.id, "revision": original.revision}],
            completed_operation_refs=[], uncertainties=[])
        outcome = await proposed.verified(context, bundle.tools)
        self.assertTrue(await self.tasks.finish(context.claim, outcome))
        async with self.tasks.transaction(self.scope.owner_id) as connection:
            result = await current_result(self.tasks, connection, receipt, self.route)
        return context, bundle, result

    async def send(self, receipt, result, *, barrier=None):
        @asynccontextmanager
        async def guarded(expected):
            if barrier is not None:
                await barrier.hold()
            async with result_guard(self.host, receipt, expected):
                yield
        return await self.wire.route.submit_delegation_result(result, guard=guarded)

    def commentary(self, delegation_id):
        return [item for item in self.wire.wire.socket.sent
                if item["type"] == "session.commentary.append" and item.get("delegation_id") == delegation_id]

    async def cancel(self, receipt, *, reason="cancelled"):
        return await LiveAudioSession.cancel_readonly_delegation(self.host, receipt.delegation_id,
            expected_revision=receipt.task.revision, reason=reason)

    def assert_serialized_unattributed(self, records):
        # Inspect the actual JSON-shaped app boundary, not Python event classes.
        serialized = json.loads(json.dumps(records))
        events = [record["payload"] for record in serialized if record["type"] == "voice_route.event"]
        self.assertTrue(events, "No serialized route records were observed")
        transcripts = [event for event in events if event.get("kind") == "transcript_revision"]
        self.assertTrue(transcripts, "No actual transcript records were inspected")
        for transcript in transcripts:
            self.assertIs(transcript["final"], False)
            self.assertIn("output", transcript)
            self.assertIsNone(transcript["output"])
        deliveries = [event for event in events if event.get("kind") == "delivery_evidence"]
        for event in events:
            self.assertIsNone(event.get("heard_transcript"))
            self.assertIn(event.get("heard"), (None, False))
        for delivery in deliveries:
            self.assertNotIn(delivery["state"], {
                "output_started", "output_finished", "client_playback_observed"})
            self.assertIn("output", delivery)
            self.assertIn("playback", delivery)
            self.assertIsNone(delivery["output"])
            self.assertIsNone(delivery["playback"])
        return transcripts, deliveries

    async def test_actual_serialized_ack_and_caption_are_not_hearing(self):
        primary, _ = await self.question("serialized-evidence")
        _, _, result = await self.complete(primary)
        self.assertTrue((await self.send(primary, result)).provider_received)
        await self.wire.inject(type="session.output_transcript.delta", event_id="assistant-fragment",
            delta="A provisional output fragment.", start_ms=101, end_ms=190)
        await self.wire.until(lambda: any(message.get("payload", {}).get("role") == "assistant"
                                         for message in self.host.messages))
        # This controlled notification transport supplies real serialized ACK
        # events through the adapter and session consumer, not a ledger/hearing
        # acceptance claim. Its offer identity is the existing transport fixture.
        offer = self.wire.offer()
        await self.wire.route.offer_notification(offer)
        await self.wire.until(lambda: any(message.get("payload", {}).get("state") == "provider_received"
                                         for message in self.host.messages))
        transcripts, deliveries = self.assert_serialized_unattributed(self.host.messages)
        self.assertTrue(any(record["role"] == "user" for record in transcripts))
        self.assertTrue(any(record["role"] == "assistant" for record in transcripts))
        self.assertEqual([record["state"] for record in deliveries],
                         ["offered", "submitted", "provider_received"])
        ack = deliveries[-1]
        self.assertEqual(ack["offer_id"], str(offer.offer_id))
        self.assertEqual(ack["attempt"], offer.attempt.model_dump(mode="json"))
        for change in ({"state": "client_playback_observed"}, {"output": {"invented": True}},
                       {"playback": {"invented": True}}, {"heard": True},
                       {"heard_transcript": "Invented hearing receipt."}):
            with self.subTest(invalid_serialized_claim=tuple(change)):
                corrupted = json.loads(json.dumps(self.host.messages))
                target = next(message["payload"] for message in corrupted
                              if message.get("payload", {}).get("state") == "provider_received")
                target.update(change)
                with self.assertRaises(AssertionError):
                    self.assert_serialized_unattributed(corrupted)

    async def test_serialized_evidence_assertion_rejects_empty_samples(self):
        with self.assertRaises(AssertionError):
            self.assert_serialized_unattributed([])

    async def scenario(self, kind, count):
        async with asyncio.timeout(12):
            primary, request = await self.question("primary")
            unrelated, _ = await self.question("unrelated")
            untouched = await self.state(unrelated)
            if kind == "speech_queued":
                initial = await self.state(primary)
                for index in range(count):
                    await self.say(" Wait, one moment.")
                    self.assertEqual(await self.state(primary), initial)
                    replay = await self.admit(delegation=request, observations=(self.origins[primary.task.id],))
                    self.assertEqual(replay.task.id, primary.task.id)
                    with self.assertRaises(TaskError):
                        await self.admit(delegation=replace(request, event_id="retry-" + str(index)),
                            observations=(self.origins[primary.task.id],))
                self.assertEqual(len(self.db.tasks), 2)
                self.assertFalse(self.wire.fences or self.host.closing_reasons)
            elif kind == "additional_question_running":
                prepared = await self.prepare(primary)
                held = self.barrier()
                async def work():
                    await held.hold()
                    return await self.complete(primary, prepared)
                worker = self.launch(work())
                await held.reached()
                initial = await self.state(primary)
                for index in range(count):
                    followup, _ = await self.question("followup-" + str(index))
                    _, _, reply = await self.complete(followup)
                    self.assertTrue((await self.send(followup, reply)).provider_received)
                    self.assertEqual(len(self.commentary(followup.delegation_id)), 1)
                    self.assertEqual(await self.state(primary), initial)
                    self.assertFalse(worker.done())
                held.release.set()
                _, _, reply = await worker
                self.assertTrue((await self.send(primary, reply)).provider_received)
                self.assertEqual((await self.state(primary)).status, TaskStatus.SUCCEEDED)
            elif kind in {"cancel_queued", "cancel_running"}:
                prepared = await self.prepare(primary) if kind == "cancel_running" else None
                initial = await self.state(primary)
                await self.say(" Cancel the primary task.")
                self.assertEqual(await self.state(primary), initial)  # Speech is not command authority.
                for index in range(count):
                    self.assertEqual(await self.cancel(primary), index == 0)
                state = await self.state(primary)
                self.assertEqual(state.revision, 2)
                self.assertIsNone(state.result_payload_id)
                if prepared is None:
                    self.assertEqual(state.status, TaskStatus.CANCELLED)
                    self.assertIsNone(state.current_attempt_id)
                else:
                    self.assertEqual(state.status, TaskStatus.CANCEL_REQUESTED)
                    self.assertEqual(state.pending_status, "cancelled")
                    self.assertIsNotNone(state.current_attempt_id)
                    with self.assertRaises(ClaimLost):
                        await prepared[1].tools._check()
                    self.assertFalse(await self.tasks.finish(prepared[0].claim,
                        WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "obsolete callback"})))
                    self.assertEqual((await self.state(primary)).status, TaskStatus.CANCEL_REQUESTED)
                self.assertFalse(self.commentary(primary.delegation_id))
            elif kind == "replacement_after_result":
                old_context, _, old_result = await self.complete(primary)
                original = self.origins[primary.task.id]
                superseded = [(primary, old_result)]
                latest = None
                for index in range(count):
                    corrected = await self.sources.append(self.scope.owner_id, ContextIngress(
                        session_id=self.scope.session_id, source_namespace=original.source_namespace,
                        source_event_id=original.source_event_id, receipt_id="correction-" + str(index),
                        expected_revision=original.revision, kind="observation", trust="derived",
                        payload=original.content | {"delta": "Corrected read-only question " + str(index)}))
                    original = corrected
                    if latest is not None:
                        superseded.append(latest)
                    replacement, _ = await self.question("replacement-" + str(index), observations=(corrected,))
                    _, _, result = await self.complete(replacement)
                    latest = replacement, result
                    for previous, _ in superseded:
                        async with self.tasks.transaction(self.scope.owner_id) as connection:
                            with self.assertRaises(TaskError):
                                await current_result(self.tasks, connection, previous, self.route)
                self.assertFalse(await self.tasks.finish(old_context.claim,
                    WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "late old result"})))
                replacement, result = latest
                self.assertNotEqual(replacement.task.id, primary.task.id)
                self.assertTrue((await self.send(replacement, result)).provider_received)
                self.assertFalse((await self.send(primary, old_result)).provider_received)
                self.assertEqual(len(self.commentary(replacement.delegation_id)), 1)
                self.assertFalse(self.commentary(primary.delegation_id))
                self.assertEqual((await self.state(replacement)).result_payload_id, result.result_payload_id)
            elif kind == "cancel_delivery_queued":
                _, _, result = await self.complete(primary)
                persisted = await self.state(primary)
                held = self.barrier()
                sending = self.launch(self.send(primary, result, barrier=held))
                await held.reached()
                for index in range(count):
                    await self.say(" Do not announce the primary answer.")
                    self.assertEqual(await self.cancel(primary), index == 0)
                held.release.set()
                self.assertFalse((await sending).provider_received)
                state = await self.state(primary)
                self.assertEqual(state.status, TaskStatus.SUCCEEDED)
                self.assertEqual(state.result_payload_id, persisted.result_payload_id)
                self.assertFalse(self.commentary(primary.delegation_id))
                self.assertFalse((await self.send(primary, result)).provider_received)
            elif kind == "already_delivered_duplicates":
                _, _, result = await self.complete(primary)
                delivered = await self.send(primary, result)
                self.assertTrue(delivered.provider_received)
                for index in range(count):
                    await self.say(" Another brief interruption.")
                    self.assertEqual(await self.send(primary, result), delivered)
                self.assertEqual(len(self.commentary(primary.delegation_id)), 1)
                self.assertEqual((await self.state(primary)).status, TaskStatus.SUCCEEDED)
            else:
                self.fail("Unknown burst scenario")
            self.assertEqual(await self.state(unrelated), untouched)
            self.assertFalse(self.db.grants)
            self.assert_serialized_unattributed(self.host.messages)


def burst_case(kind, count):
    async def test(self):
        await self.scenario(kind, count)
    test.__name__ = "test_" + kind + "_burst_" + str(count)
    return test


for _kind in ("speech_queued", "additional_question_running", "cancel_queued", "cancel_running",
              "replacement_after_result", "cancel_delivery_queued", "already_delivered_duplicates"):
    for _count in (1, 2, 3):
        _test = burst_case(_kind, _count)
        setattr(TaskInterruptionBurstTests, _test.__name__, _test)


if __name__ == "__main__":
    unittest.main()
