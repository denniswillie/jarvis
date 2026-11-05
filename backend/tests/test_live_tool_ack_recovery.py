"""Receipt-backed voice recovery never replays a domain operation."""
import asyncio
from datetime import UTC, datetime
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from app.assistant.context import AssistantContext
from app.assistant.context_providers import merge_behavior_instructions
from app.realtime.continuity import recover_voice_context
from app.realtime.session import RealtimeAudioSession
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextRecord
from app.tasks.context_history import pending_tool_acknowledgment
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError
from app.tools.executor import JarvisToolExecutor
from tests.test_live_response_source_fence import LiveResponseSourceDatabaseTests


def record(sequence, kind, content, **changes):
    values = dict(id=uuid4(), owner_id="fixture", session_id=uuid4(), sequence=sequence,
        origin_sequence=sequence, source_namespace="voice:old", source_event_id=str(sequence),
        revision=1, supersedes_event_id=None, kind=kind,
        trust={"user_transcript":"user_input", "tool_outcome":"tool_receipt",
               "assistant_output":"assistant_output"}.get(kind, "derived"),
        finality="final", source_time=datetime.now(UTC), recorded_at=datetime.now(UTC),
        payload_id=uuid4(), availability="available", content_digest=None, content=content, is_current=True)
    return ContextRecord(**(values | changes))


def history():
    return [
        record(1, "user_transcript", {"input_id":"input", "text":"Correct the firmware record."}),
        record(2, "tool_outcome", {"input_id":"input", "tool_name":"correct_memory_source", "mutates_state":True,
            "result":{"status":"success", "revision":2, "content_omitted":True}}),
        record(3, "context_gap", {"reason":"source_context_rebuild",
            "client_reported_audio_cleared":True, "unfinished_input_at_reset":False}),
    ]


def context():
    return AssistantContext(current_utc_time=datetime.now(UTC), session=None,
        behavior=merge_behavior_instructions(persistent="", session_override=""),
        recent_messages=[], active_reminders=[], preferences=[], recent_visual_interrupts=[])


class PendingToolAcknowledgmentTests(unittest.TestCase):
    def test_exact_successful_receipt_and_request_survive_bounded_recovery(self):
        records = history()
        expected = {"reset_event_id":str(records[-1].id), "request_event_id":str(records[0].id),
                    "tool_event_ids":[str(records[1].id)]}
        self.assertEqual(pending_tool_acknowledgment(records), expected)
        rebuilt = recover_voice_context(context(), {"total":3, "records":list(reversed(records))})
        self.assertEqual(rebuilt.voice_continuity["pending_tool_acknowledgment"], expected)
        self.assertEqual(set(rebuilt.voice_continuity_source_ids), {r.payload_id for r in records})

    def test_missing_or_removed_evidence_cannot_resume(self):
        for index in range(3):
            records = history()
            records[index] = records[index].model_copy(update={"availability":"deleted", "content":None})
            with self.subTest(index=index):
                self.assertIsNone(pending_tool_acknowledgment(records))
        records = history()
        records[0] = records[0].model_copy(update={"finality":"unavailable"})
        self.assertIsNone(pending_tool_acknowledgment(records))

    def test_wrong_call_scope_pending_input_or_unconfirmed_clear_cannot_resume(self):
        alternatives = [
            (1, {"source_namespace":"voice:other"}),
            (1, {"content":{"input_id":"other", "result":{"status":"success"}}}),
            (1, {"content":{"input_id":"input", "result":{"status":"failed"}}}),
            (2, {"content":{"reason":"source_context_rebuild", "client_reported_audio_cleared":False,
                           "unfinished_input_at_reset":False}}),
            (2, {"content":{"reason":"source_context_rebuild", "client_reported_audio_cleared":True,
                           "unfinished_input_at_reset":True}}),
        ]
        for index, changes in alternatives:
            records = history()
            records[index] = records[index].model_copy(update=changes)
            with self.subTest(index=index, changes=changes):
                self.assertIsNone(pending_tool_acknowledgment(records))

    def test_any_existing_output_new_request_or_attempt_prevents_automatic_repetition(self):
        for state in ("partial", "completed", "interrupted"):
            records = history()
            records[-1] = records[-1].model_copy(update={"sequence":4, "origin_sequence":4})
            records.insert(2, record(3, "assistant_output", {"text":"Earlier words.", "completion_state":state}))
            self.assertIsNone(pending_tool_acknowledgment(records))
        for kind, content in [
            ("user_transcript", {"input_id":"new", "text":"A different question."}),
            ("context_gap", {"reason":"tool_acknowledgment_attempted"}),
        ]:
            self.assertIsNone(pending_tool_acknowledgment(history() + [record(4, kind, content)]))

    def test_character_omission_does_not_turn_incomplete_context_into_recovery(self):
        rebuilt = recover_voice_context(context(), {"total":3, "records":history()}, max_characters=1)
        self.assertNotIn("pending_tool_acknowledgment", rebuilt.voice_continuity)

    def test_read_only_or_older_unclassified_receipt_cannot_trigger_mutation_acknowledgment(self):
        for value in (None, False, 1):
            records = history()
            records[1] = records[1].model_copy(update={"content": records[1].content | {"mutates_state":value}})
            self.assertIsNone(pending_tool_acknowledgment(records))
        records = history()
        records[1] = records[1].model_copy(update={"sequence":3, "origin_sequence":3})
        records[2] = records[2].model_copy(update={"sequence":4, "origin_sequence":4})
        inspection = record(2, "tool_outcome", {"input_id":"input", "mutates_state":False,
                            "result":{"status":"success", "state":"active"}})
        records.insert(1, inspection)
        self.assertEqual(pending_tool_acknowledgment(records)["tool_event_ids"], [str(records[2].id)])

    def test_removed_search_result_is_a_gap_not_reconstructed_content(self):
        records = history()
        old = record(2, "tool_outcome", None, availability="unavailable")
        records[1] = records[1].model_copy(update={"sequence":3, "origin_sequence":3})
        records[2] = records[2].model_copy(update={"sequence":4, "origin_sequence":4})
        records.insert(1, old)
        rebuilt = recover_voice_context(context(), {"total":4, "records":records})
        self.assertIn("pending_tool_acknowledgment", rebuilt.voice_continuity)
        self.assertNotIn(old.payload_id, rebuilt.voice_continuity_source_ids)


class RecoveryAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repository = ContextRepository(SimpleNamespace(pool=None))
        self.repository.claim_tool_acknowledgment = AsyncMock(return_value="Acknowledge the existing receipt only.")
        self.capture = SessionContextCapture(self.repository, owner_id="fixture", session_id=uuid4(),
            provider="openai", audio_session_id="new")
        self.voice = RealtimeAudioSession(websocket=SimpleNamespace(send_text=AsyncMock(), close=AsyncMock()),
            context=SimpleNamespace(owner_id="fixture", jarvis_session_id=self.capture.session_id, audio_session_id="new"),
            settings=JarvisSettings(_env_file=None), repository=SimpleNamespace(),
            tool_modules=(), context_providers=(), context_capture=self.capture)
        self.voice._assistant_context = recover_voice_context(context(), {"total":3, "records":history()})
        self.voice._request_normal_response = AsyncMock(return_value=True)
        self.addAsyncCleanup(self.voice.close, "fixture_complete")

    async def test_recovery_uses_normal_admission_but_has_no_mutation_authority(self):
        await self.voice._recover_tool_acknowledgment()
        self.voice._request_normal_response.assert_awaited_once()
        arguments = self.voice._request_normal_response.await_args.kwargs
        self.assertEqual(arguments["turn_generation"], 1)
        self.assertEqual(arguments["metadata"], {"jarvis_context_recovery":"tool_acknowledgment"})
        self.assertEqual(self.capture._inputs, {})
        self.assertIsNone(await self.capture.request_for_call("an-unbound-recovery-call"))

    async def test_new_user_turn_wins_before_and_during_durable_claim(self):
        self.voice._turn_generation = 1
        await self.voice._recover_tool_acknowledgment()
        self.repository.claim_tool_acknowledgment.assert_not_awaited()
        self.voice._turn_generation = 0
        async def racing_claim(*args, **kwargs):
            self.voice._turn_generation = 1
            return "Do not speak over the new user."
        self.repository.claim_tool_acknowledgment.side_effect = racing_claim
        await self.voice._recover_tool_acknowledgment()
        self.voice._request_normal_response.assert_not_awaited()

    async def test_empty_recovery_has_no_claim_and_refused_claim_cannot_generate(self):
        self.repository.claim_tool_acknowledgment.return_value = None
        await self.voice._recover_tool_acknowledgment()
        self.voice._request_normal_response.assert_not_awaited()
        self.repository.claim_tool_acknowledgment.reset_mock()
        self.voice._assistant_context = context()
        await self.voice._recover_tool_acknowledgment()
        self.repository.claim_tool_acknowledgment.assert_not_awaited()

    async def test_actual_tool_catalog_not_result_prose_classifies_mutations(self):
        events = []
        async def append(owner, event):
            events.append(event)
        capture = SessionContextCapture(SimpleNamespace(append=append), owner_id="fixture",
            session_id=uuid4(), provider="openai", audio_session_id="classification")
        try:
            for mutates in (False, True):
                tool = SimpleNamespace(mutates_state=mutates, audit_result=None,
                    handler=AsyncMock(return_value={"status":"success", "mutates_state":not mutates}))
                executor = JarvisToolExecutor(SimpleNamespace(get=lambda name: tool), context_capture=capture)
                await executor.execute("fixture", {}, provider_call_id=str(mutates))
            await capture.flush()
            self.assertEqual([event.payload["mutates_state"] for event in events], [False, True])
            self.assertEqual([event.payload["result"]["mutates_state"] for event in events], [True, False])
        finally:
            await capture.close()


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Requires explicitly manifested isolated database fixtures")
class RecoveryClaimDatabaseTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = LiveResponseSourceDatabaseTests.asyncSetUp
    voice = LiveResponseSourceDatabaseTests.voice

    async def seed(self):
        voice, _, _ = self.voice()
        capture = voice._context_capture
        capture.input_final("input", "Correct this synthetic memory.")
        capture.bind_call("call", "input")
        capture.tool_outcome("call", "correct_memory_source",
            {"status":"success", "revision":2, "content_omitted":True}, mutates_state=True)
        capture.context_reset(str(uuid4()), client_cleared=True, input_was_pending=False)
        await capture.close()
        repo = capture.repository
        snapshot = await repo.voice_history(self.f.owner, self.session.id)
        expected = pending_tool_acknowledgment(snapshot["records"])
        self.assertIsNotNone(expected)
        return repo, expected

    async def test_two_connections_admit_one_durable_attempt_without_replaying_the_tool(self):
        repo, expected = await self.seed()
        attempts = await asyncio.gather(*(repo.claim_tool_acknowledgment(
            self.f.owner, self.session.id, expected, audio_session_id=name) for name in ("new-a", "new-b")))
        self.assertEqual(sum(result is not None for result in attempts), 1)
        self.assertIsNone(await repo.claim_tool_acknowledgment(
            self.f.owner, self.session.id, expected, audio_session_id="third"))
        snapshot = await repo.voice_history(self.f.owner, self.session.id)
        self.assertEqual(sum(row.kind == "tool_outcome" for row in snapshot["records"]), 1)
        attempts = [row for row in snapshot["records"] if (row.content or {}).get("reason") == "tool_acknowledgment_attempted"]
        self.assertEqual(len(attempts), 1)
        self.assertNotIn("heard", attempts[0].content)
        self.assertIsNone(pending_tool_acknowledgment(snapshot["records"]))

    async def test_revoked_receipt_and_foreign_owner_cannot_recover(self):
        repo, expected = await self.seed()
        with self.assertRaises(TaskError):
            await repo.claim_tool_acknowledgment(self.f.other, self.session.id, expected, audio_session_id="foreign")
        async with self.f.pool.acquire() as connection:
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=(select payload_id from jarvis_context_events where owner_id=$1 and id=$2)""",
                self.f.owner, UUID(expected["tool_event_ids"][0]))
        self.assertIsNone(await repo.claim_tool_acknowledgment(
            self.f.owner, self.session.id, expected, audio_session_id="new"))

    async def test_a_later_user_request_cancels_old_recovery_eligibility(self):
        repo, expected = await self.seed()
        capture = SessionContextCapture(repo, owner_id=self.f.owner, session_id=self.session.id,
            provider="openai", audio_session_id="new")
        capture.input_final("later", "Leave it; a different question now.")
        await capture.close()
        self.assertIsNone(await repo.claim_tool_acknowledgment(
            self.f.owner, self.session.id, expected, audio_session_id="new"))
