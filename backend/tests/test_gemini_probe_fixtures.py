from datetime import UTC, datetime
import unittest
from uuid import uuid4

from evals.gemini_live.session_probe import ProbeExperienceService, ProbeMessageRepository
from app.realtime.gemini.persistence import FinalTranscript, GeminiConversationJournal


class GeminiProbeFixtureTests(unittest.IsolatedAsyncioTestCase):
    async def test_transport_fixture_supports_the_real_journal_without_database_claims(self):
        repository = ProbeMessageRepository()
        session_id = uuid4()
        failures = []

        async def failed(code):
            failures.append(code)

        journal = GeminiConversationJournal(
            repository=repository, experience_service=ProbeExperienceService(repository),
            owner_id="fixture", session_id=session_id, audio_session_id="run", on_failure=failed,
        )
        item = FinalTranscript(logical_item_id="first", role="user", content="hello",
                               occurred_at=datetime.now(UTC), provider_item_id="input-first")
        journal.submit(item)
        journal.submit(item)
        await journal.drain(timeout_seconds=1)
        self.assertEqual(len(repository.messages), 1)
        self.assertEqual(len(repository.attached), 1)
        self.assertFalse(journal.diagnostics()["degraded"])
        self.assertEqual(failures, [])

    async def test_transport_fixture_rejects_changed_text_for_the_same_identity(self):
        repository = ProbeMessageRepository()
        fields = dict(owner_id="fixture", session_id=uuid4(), audio_session_id="run",
                      logical_item_id="first", role="user", content="hello")
        await repository.create_or_reuse_gemini_message(**fields)
        with self.assertRaises(ValueError):
            await repository.create_or_reuse_gemini_message(**{**fields, "content": "changed"})
