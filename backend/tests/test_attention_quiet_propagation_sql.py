"""Isolated SQL and controlled PCM writer; no provider or native/hardware calls."""
import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from functools import wraps
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg

from app.attention.quiet import OWNER_LOCK, OwnerQuietRepository, QuietCommand, quiet_delivery_decision
from app.attention.repository import AttentionRepository
from app.attention.service import AttentionService
from app.attention.tts import SpeechFrame
from app.attention.voice import ExpertVoiceDelivery
from app.migrations.runner import apply_migrations, load_migrations, quote_schema
from app.realtime.socket_sender import WebSocketEventSender
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskCreate, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tools.invocation import ToolInvocationContext
from evals.second_brain.fixtures import DatabaseFixture
from test_notification_ledger_database import cleanup_fixture_resources


def qualification(method):
    @wraps(method)
    async def run(self):
        try:
            await method(self)
        except Exception as error:
            self.report.update(status="failed", error_type=type(error).__name__,
                               sqlstate=getattr(error, "sqlstate", None))
            raise
        else:
            self.report["status"] = "passed"
        finally:
            self.save_report()
    return run


class ControlledSpeech:
    def __init__(self, *, partial=False):
        self.partial = partial
        self.submitted = asyncio.Event()
        self.first_written = asyncio.Event()
        self.release_tail = asyncio.Event()
        self.requests = []
        self.closed = 0

    def prepare(self, text):
        return SimpleNamespace(text=text, digest=sha256(text.encode()).hexdigest(),
            model="fixture-tts", voice="fixture-voice", instructions="Synthetic PCM only")

    @asynccontextmanager
    async def open_stream(self, request, *, authorize_submission, is_current):
        assert await authorize_submission(request)
        self.requests.append(request)
        self.submitted.set()

        async def frames():
            assert await is_current()
            yield SpeechFrame(b"\x00\x00" * 240, 0)
            # Resuming the generator proves the first guarded write committed.
            self.first_written.set()
            if self.partial:
                await self.release_tail.wait()
            assert await is_current()
            yield SpeechFrame(b"\x00\x00" * 240, 240)

        try:
            yield SimpleNamespace(frames=frames, samples_yielded=480, first_bytes_ms=1.0,
                                  provider_request_id="synthetic-no-provider")
        finally:
            self.closed += 1


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class QuietPropagationSQLTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.fixture = DatabaseFixture(dsn, directory, test_name=self.id())
        self.report_path = directory / f"quiet-propagation-{self.fixture.run_id}.json"
        self.report = {"format": "jarvis.quiet_propagation.sql.v1", "test": self.id(),
            "run_id": self.fixture.run_id, "status": "unreached", "cleanup": None,
            "provider_calls": 0, "hardware_evidence": False}
        self.connection = self.pool = self.service = None
        self.deliveries = []
        self.gates = []
        self.addAsyncCleanup(self.cleanup)
        self.save_report()
        try:
            self.connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
            schema = await self.fixture.create_schema(self.connection)
            # Deployment contract: migrate before creating runtime pools.
            await apply_migrations(self.connection, schema=schema, fresh=True,
                migrations=tuple(m for m in load_migrations() if m.version <= 42))
            path_sql = f"set search_path to {quote_schema(schema)},pg_catalog,extensions"
            await self.connection.execute(path_sql)

            async def setup(connection):
                await connection.execute(path_sql)
                await connection.execute("set lock_timeout='8s'; set statement_timeout='45s'")

            self.pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4, timeout=10,
                                                command_timeout=60, setup=setup)
            self.owner, self.other = self.fixture.owners
            self.session_id = await self.connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.owner)
            self.tasks = TaskRepository(self.pool, TaskLimits(lease_seconds=300))
            self.contexts = ContextRepository(self.tasks)
            self.quiet = OwnerQuietRepository(self.pool)
            self.attention = AttentionRepository(self.tasks, lease_seconds=300)
            self.service = AttentionService(self.attention)
        except Exception as error:
            self.report.update(setup_status="failed", setup_error_type=type(error).__name__,
                               sqlstate=getattr(error, "sqlstate", None))
            self.save_report()
            raise AssertionError("Fixture prerequisite failed; delivery checks unreached: "
                                 + type(error).__name__) from None

    def save_report(self):
        descriptor = os.open(self.report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(self.report, handle, indent=2)
            handle.write("\n")

    async def cleanup(self):
        try:
            for gate in self.gates:
                gate.set()
            for delivery in self.deliveries:
                if delivery.session._app_sender._lock.locked():
                    # All assertions have ended; never leave a controlled writer blocked.
                    delivery.session._app_sender._lock.release()
                if delivery.task is not None and not delivery.task.done():
                    await asyncio.wait_for(delivery.stop("fixture_cleanup"), 5)
                    await asyncio.wait_for(delivery.task, 10)
            if self.service is not None:
                await self.service.close()
        finally:
            await cleanup_fixture_resources(self.fixture, self.report, self.save_report,
                                            pool=self.pool, connection=self.connection)
            print("QUIET_PROPAGATION_SQL=" + json.dumps({"report": str(self.report_path),
                "test": self.report["test"], "status": self.report["status"],
                "cleanup": self.report["cleanup"]}))

    async def source(self, text, *, previous=None):
        return await self.contexts.append(self.owner, ContextIngress(session_id=self.session_id,
            source_namespace="quiet_propagation_fixture", source_event_id=(
                previous.source_event_id if previous is not None else uuid4().hex),
            receipt_id=uuid4().hex, expected_revision=previous.revision if previous is not None else None,
            kind="user_transcript", trust="user_input", finality="revised" if previous else "final",
            payload={"text": text}))

    async def change(self, action="activate", *, previous=None, **kwargs):
        record = await self.source("Resume interruptions." if action == "resume" else
            "Please do not disturb; keep working. Correction " + str(previous.revision if previous else 0),
            previous=previous)
        revision = (await self.quiet.get(self.owner)).revision
        authority = ToolInvocationContext(provider_call_id="fixture", request_event_id=record.id,
                                           request_revision=record.revision)
        command = QuietCommand(action=action, expected_revision=revision, **kwargs)
        result = await self.quiet.save(self.owner, self.session_id, authority, command)
        retry = await self.quiet.save(self.owner, self.session_id, authority, command)
        self.assertTrue(retry["request_replayed"])
        self.assertEqual(result["accepted_revision"], retry["accepted_revision"])
        return record

    async def complete_work(self):
        task = await self.tasks.create(self.owner, TaskCreate(request_id=uuid4(),
            title="Synthetic quiet propagation work", requested_outcome="Read-only synthetic result.",
            session_id=self.session_id), model="fixture-model")
        claim = await self.tasks.claim_next("quiet-propagation-worker", ("frontier",), owner_id=self.owner)
        self.assertEqual(claim.task.id, task.id)
        context = TaskRunner(self.tasks, {}).context(claim)
        outcome = await HermesTaskResult(status="succeeded", answer="Synthetic result remains available.",
            speech=[{"text": "Synthetic result remains available."}], evidence=[],
            completed_operation_refs=[], uncertainties=[]).verified(context, SimpleNamespace())
        self.assertTrue(await self.tasks.finish(claim, outcome))
        intent = await self.connection.fetchval(
            "select id from jarvis_attention_intents where owner_id=$1 and task_id=$2", self.owner, task.id)
        self.assertIsNotNone(intent)
        return task, intent

    async def task_preserved(self, task, intent):
        self.assertEqual((await self.tasks.get(self.owner, task.id)).status, TaskStatus.SUCCEEDED)
        detail = await self.attention.detail(self.owner, intent)
        self.assertIsNotNone(detail["expert_result"])

    async def claim(self):
        claim = await self.attention.claim_next(self.owner, self.session_id, "controlled-voice")
        self.assertIsNotNone(claim)
        return claim

    async def delivery(self, claim, *, partial=False, start_gate=None):
        events = []
        lock_free = []

        async def send(text):
            event = json.loads(text)
            if event["type"] == "audio.output.clear":
                # Fencing must not hold an owner transaction across transport.
                async with self.pool.acquire() as connection, connection.transaction():
                    lock_free.append(await connection.fetchval(OWNER_LOCK.replace(
                        "pg_advisory_xact_lock", "pg_try_advisory_xact_lock"), self.owner))
            events.append(event)

        session = SimpleNamespace(_closed=asyncio.Event(), _context_revoked=False,
            _context_capture=None, _app_sender=WebSocketEventSender(SimpleNamespace(send_text=send)),
            remember_attention_item=lambda *_: None, record_attention_speech=AsyncMock())
        session._mark_closing = lambda reason: session._closed.set()

        async def release(value):
            if session._attention_delivery is value:
                session._attention_delivery = None

        session.release_attention = release
        speech = ControlledSpeech(partial=partial)
        result = ExpertVoiceDelivery(self.attention, speech, claim, session, receipt_seconds=.01)
        session._attention_delivery = result
        result.fixture_events, result.fixture_lock_free = events, lock_free

        async def run():
            if start_gate is not None:
                await start_gate.wait()
            await result.run()

        if start_gate is not None:
            self.gates.append(start_gate)
        self.gates.append(speech.release_tail)
        self.deliveries.append(result)
        self.service._deliveries.add(result)
        result.task = asyncio.create_task(run())
        result.task.add_done_callback(lambda _: self.service._deliveries.discard(result))
        return result, speech

    async def direct_answer_while_quiet(self):
        before = await self.quiet.get(self.owner)
        self.assertTrue(before.suppresses_unsolicited)
        await self.source("What is the available result?")
        decision = await quiet_delivery_decision(None, self.owner, kind="solicited_reply", channel="voice")
        self.assertTrue(decision.allowed)
        socket = SimpleNamespace(send_text=AsyncMock())
        writer = WebSocketEventSender(socket)
        self.assertTrue(await writer.send_text('{"type":"fixture.direct_answer"}',
                                              should_send=lambda: decision.allowed))
        socket.send_text.assert_awaited_once()
        after = await OwnerQuietRepository(self.pool).get(self.owner)
        self.assertEqual(before.revision, after.revision)
        self.assertTrue(after.suppresses_unsolicited)

    async def summary_once(self, *, expires=False):
        if expires:
            deadline = await self.connection.fetchval("select clock_timestamp() + interval '3 seconds'")
            await self.change(expires_at=deadline, timezone="UTC")
        else:
            await self.change()
        task, intent = await self.complete_work()
        self.assertIsNone(await self.attention.claim_next(self.owner, self.session_id, "muted"))
        await self.direct_answer_while_quiet()
        if expires:
            now = await self.connection.fetchval("select clock_timestamp()")
            await asyncio.sleep(max(0, (deadline-now).total_seconds()) + .05)
            self.assertEqual((await self.quiet.get(self.owner)).effective_state, "expired")
        else:
            await self.change("resume")
        claim = await self.claim()
        self.assertEqual(claim.source_namespace, "quiet_resume_summary")
        delivery, speech = await self.delivery(claim)
        await asyncio.wait_for(delivery.task, 15)
        self.assertGreater(len(speech.requests), 0)
        self.assertTrue(any(e["type"] == "audio.output.delta" for e in delivery.fixture_events))
        detail = await self.attention.detail(self.owner, claim.intent_id)
        self.assertEqual(detail["attempts"][0]["phase"], "unknown")
        self.assertIsNone(detail["attempts"][0]["played_samples"])
        # Explicit repeated resume and a reconstructed claim client cannot replay.
        for _ in range(3):
            await self.change("resume")
            self.assertEqual(await self.service.enforce_owner_quiet(self.owner), 0)
            reconstructed = AttentionRepository(self.tasks, lease_seconds=300)
            self.assertIsNone(await reconstructed.claim_next(self.owner, self.session_id, "reconnect"))
        count = await self.connection.fetchval("select count(*) from jarvis_attention_attempts "
            "where owner_id=$1 and intent_id=$2", self.owner, claim.intent_id)
        self.assertEqual(count, 1)
        await self.task_preserved(task, intent)

    @qualification
    async def test_quiet_before_claim_direct_answer_and_explicit_resume_summary_once(self):
        await self.summary_once()

    @qualification
    async def test_exact_expiry_reconstruction_summary_once(self):
        await self.summary_once(expires=True)

    @qualification
    async def test_queued_unsubmitted_delivery_is_retired_without_cancelling_work(self):
        task, intent = await self.complete_work()
        claim = await self.claim()
        gate = asyncio.Event()
        delivery, speech = await self.delivery(claim, start_gate=gate)
        await self.change()
        self.assertEqual(await self.service.enforce_owner_quiet(self.owner), 1)
        self.assertEqual(await self.service.enforce_owner_quiet(self.owner), 0)
        gate.set()
        await asyncio.wait_for(delivery.task, 15)
        self.assertEqual(speech.requests, [])
        self.assertEqual(delivery.fixture_events, [])
        attempt = (await self.attention.detail(self.owner, intent))["attempts"][0]
        self.assertEqual(attempt["phase"], "failed")
        self.assertIsNone(attempt["submitted_at"])
        self.assertEqual(attempt["reason"], "owner_quiet_quiet")
        await self.task_preserved(task, intent)
        await self.direct_answer_while_quiet()

    @qualification
    async def test_already_submitted_pcm_queued_at_writer_rechecks_without_poll(self):
        task, intent = await self.complete_work()
        claim = await self.claim()
        gate = asyncio.Event()
        delivery, speech = await self.delivery(claim, start_gate=gate)
        await delivery.session._app_sender._lock.acquire()
        gate.set()
        await asyncio.wait_for(speech.submitted.wait(), 10)
        await self.change()
        # Deliberately do not call propagation: existing final writer gate suffices.
        delivery.session._app_sender._lock.release()
        await asyncio.wait_for(delivery.task, 15)
        self.assertFalse(any(e["type"] == "audio.output.delta" for e in delivery.fixture_events))
        attempt = (await self.attention.detail(self.owner, intent))["attempts"][0]
        self.assertEqual(attempt["phase"], "unknown")
        self.assertIsNotNone(attempt["submitted_at"])
        self.assertEqual(attempt["forwarded_samples"], 0)
        self.assertIsNone(attempt["played_samples"])
        self.assertEqual(attempt["reason"], "owner_quiet_quiet")
        self.assertEqual(delivery.fixture_lock_free, [True])
        await self.task_preserved(task, intent)

    async def partial_cutoffs(self, repeats):
        task, intent = await self.complete_work()
        claim = await self.claim()
        delivery, speech = await self.delivery(claim, partial=True)
        await asyncio.wait_for(speech.first_written.wait(), 10)
        # Simulated client telemetry, not evidence a person heard any words.
        await delivery.receipt({"type": "playback.progress", "item_id": delivery.item_id(0),
            "response_id": delivery.response_id, "played_samples": 120, "content_index": 0}, completed=False)
        record = None
        for _ in range(repeats):
            record = await self.change(previous=record)
            await self.service._maintenance_cycle(self.owner)
            self.assertTrue((await self.quiet.get(self.owner)).suppresses_unsolicited)
        await asyncio.wait_for(delivery.task, 15)
        speech.release_tail.set()
        self.assertEqual(len([e for e in delivery.fixture_events if e["type"] == "audio.output.delta"]), 1)
        self.assertEqual(len([e for e in delivery.fixture_events if e["type"] == "audio.output.clear"]), 1)
        self.assertEqual(delivery.fixture_lock_free, [True])
        self.assertFalse(delivery.session._closed.is_set())
        self.assertEqual(speech.closed, 1)
        attempt = (await self.attention.detail(self.owner, intent))["attempts"][0]
        self.assertEqual(attempt["phase"], "unknown")
        self.assertEqual(attempt["reason"], "owner_quiet_quiet")
        self.assertEqual(attempt["forwarded_samples"], 240)
        self.assertEqual(attempt["played_samples"], 120)
        await self.task_preserved(task, intent)
        # Distinct authorized work still completes while quiet.
        other_task, other_intent = await self.complete_work()
        await self.task_preserved(other_task, other_intent)
        await self.direct_answer_while_quiet()
        reconnect = AttentionRepository(self.tasks, lease_seconds=300)
        self.assertIsNone(await reconnect.claim_next(self.owner, self.session_id, "reconnect-still-muted"))
        self.assertEqual(await self.service.enforce_owner_quiet(self.owner), 0)

    @qualification
    async def test_partial_output_one_quiet_command(self):
        await self.partial_cutoffs(1)

    @qualification
    async def test_partial_output_two_corrected_quiet_commands(self):
        await self.partial_cutoffs(2)

    @qualification
    async def test_partial_output_three_corrected_quiet_commands(self):
        await self.partial_cutoffs(3)
