"""Late source ordering through the real journal and owned PostgreSQL fixtures."""
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.experience.evidence import EventEvidenceDescriptor, reduce_event_aggregate
from app.experience.repository import ExperienceRepository, ToolExecutionRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.realtime.gemini.persistence import FinalTranscript, GeminiConversationJournal
from app.sessions.repository import SessionRepository
from evals.second_brain.source_fixture import source_fixture


class EvidenceOrderingTests(unittest.TestCase):
    def test_older_evidence_adds_coverage_without_overwriting_latest_state(self):
        now, latest = datetime.now(UTC), uuid4()
        event = SimpleNamespace(started_at=now - timedelta(seconds=3),
            state={"last_evidence_at": now.isoformat(), "last_evidence_type": "observation",
                   "last_evidence_ref_id": str(latest), "evidence_counts": {"observation": 1}},
            modalities=("visual",), confidence=.8, importance=.5,
            scene="new scene", activity="new activity", title="new title", summary="new summary")
        older = EventEvidenceDescriptor("message", uuid4(), now - timedelta(seconds=1),
            "audio", "older detail", scene="old scene", activity="old activity")
        result = reduce_event_aggregate(event, older)
        self.assertEqual((result.scene, result.activity, result.title, result.summary),
                         ("new scene", "new activity", "new title", "new summary"))
        self.assertEqual(result.state["last_evidence_at"], now.isoformat())
        self.assertEqual(result.state["last_evidence_ref_id"], str(latest))
        self.assertEqual(result.state["late_evidence_count"], 1)
        self.assertEqual(result.state["evidence_counts"], {"observation": 1, "message": 1})
        self.assertIn("older detail", result.state["recent_summaries"])
        self.assertEqual(result.modalities, ("audio", "visual"))

    def test_first_attachment_to_empty_reserved_event_establishes_state(self):
        now = datetime.now(UTC)
        event = SimpleNamespace(started_at=now, state={}, modalities=(),
            confidence=0, importance=0, scene=None, activity=None, title="", summary="")
        evidence = EventEvidenceDescriptor("message", uuid4(), now - timedelta(seconds=1),
            "audio", "The only actual source.", scene="original scene", activity="original activity")
        result = reduce_event_aggregate(event, evidence)
        self.assertEqual(result.summary, evidence.summary)
        self.assertEqual(result.scene, evidence.scene)
        self.assertEqual(result.state["last_evidence_at"], evidence.occurred_at.isoformat())
        self.assertEqual(result.state["last_evidence_ref_id"], str(evidence.evidence_ref_id))
        self.assertEqual(result.state["evidence_counts"], {"message": 1})


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Explicit isolated DB fixture configuration required")
class ExperienceOrderingDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        async with self.fixture.pool.acquire() as connection:
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.fixture.owner)
        self.repository = SessionRepository(self.fixture.pool)
        self.events = ExperienceRepository(self.fixture.pool)
        self.tools = ToolExecutionRepository(self.fixture.pool)
        self.experience = ExperienceEventService(repository=self.events,
            segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)))
        self.addAsyncCleanup(self.experience.close)
        self.original = datetime.now(UTC) - timedelta(seconds=20)
        self.audio_id = "ordering-" + uuid4().hex
        self.errors = []

    async def tool(self, completed_at):
        call = uuid4().hex
        await self.tools.start_execution(owner_id=self.fixture.owner, session_id=self.session,
            provider_call_id=call, tool_name="fixture_read", arguments={}, started_at=completed_at)
        execution = await self.tools.finalize_execution(owner_id=self.fixture.owner, session_id=self.session,
            provider_call_id=call, status="completed", completed_at=completed_at, result={"status": "success"})
        attached = await self.experience.attach_tool_execution(execution)
        self.assertTrue(attached.attached)
        return attached.event

    async def journal(self, occurred_at, *, item="turn:1:user"):
        async def failure(code):
            self.errors.append(code)
        journal = GeminiConversationJournal(repository=self.repository, experience_service=self.experience,
            owner_id=self.fixture.owner, session_id=self.session, audio_session_id=self.audio_id, on_failure=failure)
        journal.submit(FinalTranscript(item, "user", "Synthetic original request: do not run code.",
                                      occurred_at, "fixture-input"))
        await journal.drain(timeout_seconds=15)
        return journal

    async def test_older_transcript_attaches_once_and_keeps_original_time(self):
        newer_time = self.original + timedelta(seconds=1)
        event = await self.tool(newer_time)
        journal = await self.journal(self.original)
        self.assertEqual(journal.diagnostics()["states"], {"attached": 1})
        self.assertEqual(self.errors, [])
        current = await self.events.get_open_event(owner_id=self.fixture.owner, session_id=self.session)
        self.assertEqual(current.id, event.id)
        self.assertEqual(current.started_at, self.original)
        self.assertEqual(current.state["last_evidence_at"], newer_time.isoformat())
        self.assertEqual(current.summary, event.summary)
        self.assertEqual(current.state["evidence_counts"], {"tool_execution": 1, "message": 1})
        replay = await self.journal(self.original)
        self.assertEqual(replay.diagnostics()["states"], {"attached": 1})
        async with self.fixture.pool.acquire() as connection:
            messages = await connection.fetch("select * from jarvis_messages where owner_id=$1", self.fixture.owner)
            evidence = await connection.fetch("select * from experience_event_evidence where event_id=$1", event.id)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["created_at"], self.original)
        self.assertEqual(len(evidence), 2)
        self.assertTrue(await self.repository.gemini_message_has_experience_evidence(
            owner_id=self.fixture.owner, session_id=self.session, message_id=messages[0]["id"]))
        self.assertFalse(await self.repository.gemini_message_has_experience_evidence(
            owner_id=self.fixture.other, session_id=self.session, message_id=messages[0]["id"]))

    async def test_initial_attachment_to_reserved_empty_event_retains_content(self):
        reserved = await self.events.get_or_create_open_event(owner_id=self.fixture.owner,
            session_id=self.session, event_kind="interaction",
            started_at=self.original + timedelta(seconds=1), modalities=())
        journal = await self.journal(self.original)
        self.assertEqual(journal.diagnostics()["states"], {"attached": 1})
        current = await self.events.get_open_event(owner_id=self.fixture.owner, session_id=self.session)
        self.assertEqual(current.id, reserved.id)
        self.assertEqual(current.started_at, self.original)
        self.assertEqual(current.summary, "Synthetic original request: do not run code.")
        self.assertEqual(current.state["last_evidence_at"], self.original.isoformat())
        self.assertEqual(current.state["evidence_counts"], {"message": 1})

    async def test_late_message_after_closed_boundary_can_extend_only_new_open_event(self):
        previous = await self.tool(self.original)
        await self.experience._close_event(previous, ended_at=self.original, reason="fixture_boundary")
        newer = await self.tool(self.original + timedelta(seconds=2))
        journal = await self.journal(self.original + timedelta(seconds=1))
        self.assertEqual(journal.diagnostics()["states"], {"attached": 1})
        current = await self.events.get_open_event(owner_id=self.fixture.owner, session_id=self.session)
        self.assertEqual(current.id, newer.id)
        self.assertEqual(current.started_at, self.original + timedelta(seconds=1))
        closed = await self.events.get_event(owner_id=self.fixture.owner, session_id=self.session, event_id=previous.id)
        self.assertEqual(closed.ended_at, self.original)
        self.assertEqual(closed.state["evidence_counts"], {"tool_execution": 1})

    async def test_finalized_boundary_is_not_reopened_or_claimed_retained(self):
        previous = await self.tool(self.original)
        await self.experience._close_event(previous, ended_at=self.original, reason="fixture_boundary")
        newer = await self.tool(self.original + timedelta(seconds=2))
        journal = await self.journal(self.original)
        self.assertEqual(journal.diagnostics()["states"], {"failed": 1})
        self.assertEqual(self.errors, ["gemini_memory_persistence_failed"])
        self.assertEqual(journal.diagnostics()["failures"][0]["attachment_reason"], "finalized_boundary")
        current = await self.events.get_open_event(owner_id=self.fixture.owner, session_id=self.session)
        self.assertEqual((current.id, current.started_at), (newer.id, newer.started_at))
        async with self.fixture.pool.acquire() as connection:
            count = await connection.fetchval("select count(*) from experience_event_evidence where evidence_type='message'")
        self.assertEqual(count, 0)
