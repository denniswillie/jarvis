"""Registered handler -> real executor/capture/SQL -> controlled output fencing."""
import asyncio
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.attention.quiet import quiet_delivery_decision
from app.attention.service import AttentionService, initialize_attention_runtime
from app.experience.repository import ToolExecutionRepository
from app.preferences.repository import PreferenceRepository
from app.preferences.tools import PreferenceToolModule
from app.realtime.socket_sender import WebSocketEventSender
from app.tasks.context_capture import SessionContextCapture
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry
import test_attention_quiet_propagation_sql as propagation
from test_attention_quiet_tool_entry import attention_only_flags


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class QuietToolEntrySQLTests(unittest.IsolatedAsyncioTestCase):
    # Reuse manifested fixtures and controlled PCM, not their test methods.
    cleanup = propagation.QuietPropagationSQLTests.cleanup
    save_report = propagation.QuietPropagationSQLTests.save_report
    complete_work = propagation.QuietPropagationSQLTests.complete_work
    task_preserved = propagation.QuietPropagationSQLTests.task_preserved
    claim = propagation.QuietPropagationSQLTests.claim
    delivery = propagation.QuietPropagationSQLTests.delivery

    async def asyncSetUp(self):
        await propagation.QuietPropagationSQLTests.asyncSetUp(self)
        self.report["format"] = "jarvis.quiet_tool_entry.sql.v1"
        self.module = PreferenceToolModule(PreferenceRepository(self.pool))
        self.registry = create_tool_registry(modules=(self.module,), context=ToolBuildContext(
            settings=None, owner_id=self.owner, session_id=self.session_id))
        runtime = SimpleNamespace(settings=SimpleNamespace(SUPABASE_DB_URL="fixture", OPENAI_API_KEY="",
            JARVIS_OWNER_ID=self.owner), db_pool=self.pool, tool_modules=(self.module,),
            background_task_service=SimpleNamespace(repository=self.tasks),
            realtime_session_manager=SimpleNamespace(), attention_service=None)
        await self.service.close()
        # Actual runtime binding and schema readiness; no workers/providers start.
        with patch("app.attention.service.get_second_brain_settings", return_value=attention_only_flags()), \
             patch.object(AttentionService, "start_maintenance") as maintenance, \
             patch.object(AttentionService, "start_voice", side_effect=AssertionError("No provider worker")), \
             patch("app.reminders.duty_runtime.initialize_responsibilities", new_callable=AsyncMock,
                   return_value=None):
            await initialize_attention_runtime(runtime)
        self.assertIsNone(runtime.attention_issue, "Runtime binding prerequisite failed; delivery unreached")
        self.service = runtime.attention_service
        self.attention = self.service.repository
        maintenance.assert_called_once_with(owner=self.owner)
        self.capture = SessionContextCapture(self.contexts, owner_id=self.owner, session_id=self.session_id,
            provider="openai", audio_session_id="fixture-" + uuid4().hex, write_timeout=10)
        self.addAsyncCleanup(self.capture.close)
        self.execution_repository = ToolExecutionRepository(self.pool)
        self.executor = JarvisToolExecutor(self.registry, execution_repository=self.execution_repository,
            owner_id=self.owner, session_id=self.session_id, context_capture=self.capture)

    def bind(self, input_id, call_id, *, text=None, final=True):
        self.capture.input_admitted(input_id)
        if final:
            self.capture.input_final(input_id, text or "Please do not disturb; keep working.")
        self.capture.bind_call(call_id, input_id)

    async def execute(self, action, revision, call):
        return await self.executor.execute("set_attention_quiet",
            {"action": action, "expected_revision": revision}, provider_call_id=call)

    async def assert_quiet_rows(self, count):
        self.assertEqual(await self.connection.fetchval("select count(*) from jarvis_capability_preferences "
            "where owner_id=$1 and capability='owner_attention_quiet'", self.owner), count)

    @propagation.qualification
    async def test_finalized_entry_fences_partial_output_retries_corrections_and_resume_once(self):
        task, intent = await self.complete_work()
        delivery, speech = await self.delivery(await self.claim(), partial=True)
        await asyncio.wait_for(speech.first_written.wait(), 10)
        self.bind("focus", "focus-1")
        result = await self.execute("activate", 0, "focus-1")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["delivery_propagation"]["stop_requests"], 1)
        self.assertFalse(result["delivery_propagation"]["playback_stop_confirmed"])
        await asyncio.wait_for(delivery.task, 15)
        self.assertEqual(delivery.fixture_lock_free, [True])
        self.assertEqual(len([e for e in delivery.fixture_events if e["type"] == "audio.output.clear"]), 1)
        attempt = (await self.attention.detail(self.owner, intent))["attempts"][0]
        self.assertEqual(attempt["phase"], "unknown")
        self.assertEqual(attempt["forwarded_samples"], 240)
        self.assertIsNone(attempt["played_samples"])
        await self.task_preserved(task, intent)
        episode = result["quiet"]["episode"]["episode_id"]

        # Same durable provider call returns its original receipt, not a new write.
        again = await self.execute("activate", 0, "focus-1")
        self.assertEqual(again, result)
        for call in ("focus-retry-2", "focus-retry-3"):
            self.capture.bind_call(call, "focus")
            retry = await self.execute("activate", 0, call)
            self.assertTrue(retry["request_replayed"])
            self.assertEqual(retry["delivery_propagation"]["stop_requests"], 0)
        await self.assert_quiet_rows(1)

        for revision in (2, 3):
            call = f"focus-correction-{revision}"
            self.bind("focus", call, text=f"Correction {revision}: keep do-not-disturb active until I resume.")
            corrected = await self.execute("activate", revision - 1, call)
            self.assertEqual(corrected["status"], "success")
            self.assertEqual(corrected["accepted_revision"], revision)
            self.assertEqual(corrected["quiet"]["episode"]["episode_id"], episode)
            self.assertTrue((await self.quiet.get(self.owner)).suppresses_unsolicited)
        await self.assert_quiet_rows(3)

        self.bind("question", "question-call", text="What result is available?")
        read = await self.executor.execute("get_attention_quiet", {}, provider_call_id="question-call")
        self.assertEqual(read["quiet"]["revision"], 3)
        self.assertEqual(read["quiet"]["effective_state"], "quiet")
        decision = await quiet_delivery_decision(None, self.owner, kind="solicited_reply", channel="voice")
        self.assertTrue(decision.allowed)
        socket = SimpleNamespace(send_text=AsyncMock())
        self.assertTrue(await WebSocketEventSender(socket).send_text('{"type":"fixture.direct_answer"}',
            should_send=lambda: decision.allowed))
        socket.send_text.assert_awaited_once()
        self.assertTrue((await self.quiet.get(self.owner)).suppresses_unsolicited)
        pending_task, pending_intent = await self.complete_work()
        self.assertIsNone(await self.attention.claim_next(self.owner, self.session_id, "still-quiet"))

        self.bind("resume", "resume-1", text="Resume interruptions now.")
        resumed = await self.execute("resume", 3, "resume-1")
        self.assertEqual(resumed["status"], "success")
        self.assertEqual(resumed["quiet"]["effective_state"], "resumed")
        claim = await self.claim()
        self.assertEqual(claim.source_namespace, "quiet_resume_summary")
        summary, _ = await self.delivery(claim)
        await asyncio.wait_for(summary.task, 15)
        self.assertTrue(any(e["type"] == "audio.output.delta" for e in summary.fixture_events))
        for call in ("resume-1", "resume-retry-2", "resume-retry-3"):
            self.capture.bind_call(call, "resume")
            repeated = await self.execute("resume", 3, call)
            self.assertEqual(repeated["quiet"]["episode"]["episode_id"], episode)
            self.assertIsNone(await self.attention.claim_next(self.owner, self.session_id, "reconnect"))
        await self.assert_quiet_rows(4)
        self.assertEqual(await self.connection.fetchval("select count(*) from jarvis_attention_attempts "
            "where owner_id=$1 and intent_id=$2", self.owner, claim.intent_id), 1)
        # A historical successful activate retry cannot resurrect quiet after resume.
        self.assertEqual((await self.execute("activate", 0, "focus-1"))["accepted_revision"], 1)
        self.assertEqual((await self.quiet.get(self.owner)).effective_state, "resumed")
        await self.task_preserved(pending_task, pending_intent)

    @propagation.qualification
    async def test_provisional_stale_revision_and_cross_owner_requests_fail_closed(self):
        self.bind("pending", "pending-call", final=False)
        denied = await self.execute("activate", 0, "pending-call")
        self.assertEqual(denied["error"]["code"], "request_evidence_required")
        await self.assert_quiet_rows(0)
        self.bind("pending", "final-call")
        self.assertEqual((await self.execute("activate", 0, "pending-call"))["status"], "failed")
        accepted = await self.execute("activate", 0, "final-call")
        self.assertEqual(accepted["status"], "success")
        stale = await self.capture.request_for_call("final-call")
        self.bind("pending", "revised-call", text="Correction: remain quiet until I explicitly resume.")
        self.assertTrue(await self.capture.flush())
        with patch.object(self.capture, "request_for_call", new_callable=AsyncMock, return_value=stale):
            denied = await self.execute("resume", 1, "stale-source-call")
        self.assertEqual(denied["error"]["code"], "request_evidence_unavailable")
        self.assertEqual((await self.execute("activate", 1, "revised-call"))["status"], "success")
        self.bind("fresh", "stale-policy-call")
        denied = await self.execute("resume", 0, "stale-policy-call")
        self.assertEqual(denied["error"]["code"], "revision_conflict")

        # Provider IDs are immutable operation receipts, not authority to rewrite history.
        reused = await self.execute("resume", 2, "final-call")
        self.assertEqual(reused["status"], "failed")
        await self.assert_quiet_rows(2)
        other_session = await self.connection.fetchval(
            "insert into jarvis_sessions(owner_id) values($1) returning id", self.other)
        registry = create_tool_registry(modules=(self.module,), context=ToolBuildContext(
            settings=None, owner_id=self.other, session_id=other_session))
        foreign = JarvisToolExecutor(registry, execution_repository=self.execution_repository,
            owner_id=self.other, session_id=other_session, context_capture=self.capture)
        self.capture.bind_call("cross-owner", "fresh")
        denied = await foreign.execute("set_attention_quiet", {"action": "activate", "expected_revision": 0},
                                       provider_call_id="cross-owner")
        self.assertEqual(denied["error"]["code"], "request_evidence_unavailable")
        self.assertEqual((await self.quiet.get(self.other)).revision, 0)
        self.assertTrue((await self.quiet.get(self.owner)).suppresses_unsolicited)

    @propagation.qualification
    async def test_post_commit_fence_failure_retains_commit_and_new_call_repairs_without_new_episode(self):
        task, intent = await self.complete_work()
        delivery, speech = await self.delivery(await self.claim(), partial=True)
        await asyncio.wait_for(speech.first_written.wait(), 10)
        self.bind("focus", "failed-fence")
        with patch.object(self.service, "enforce_owner_quiet", new_callable=AsyncMock,
                          side_effect=RuntimeError("controlled fence unavailable")):
            result = await self.execute("activate", 0, "failed-fence")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["delivery_propagation"]["state"], "unconfirmed")
        self.assertFalse(delivery.task.done())
        self.assertTrue((await self.quiet.get(self.owner)).suppresses_unsolicited)
        stored = await self.execution_repository.get_executions_by_call_ids(owner_id=self.owner,
            session_id=self.session_id, provider_call_ids=["failed-fence"])
        self.assertEqual(stored[0].status, "completed")
        self.assertEqual((await self.execute("activate", 0, "failed-fence")), result)
        self.capture.bind_call("repair-fence", "focus")
        repaired = await self.execute("activate", 0, "repair-fence")
        self.assertTrue(repaired["request_replayed"])
        self.assertEqual(repaired["delivery_propagation"]["stop_requests"], 1)
        self.assertEqual(repaired["quiet"]["episode"], result["quiet"]["episode"])
        await asyncio.wait_for(delivery.task, 15)
        self.assertEqual(delivery.fixture_lock_free, [True])
        self.assertEqual(len([e for e in delivery.fixture_events if e["type"] == "audio.output.clear"]), 1)
        await self.assert_quiet_rows(1)
        await self.task_preserved(task, intent)
