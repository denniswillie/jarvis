import asyncio
import os
import unittest

from app.sessions.repository import SessionRepository
from tests import test_attention_voice as voice_tests
from tests.test_session_message_transaction import NoNestedPool


@unittest.skipUnless(os.environ.get("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.environ.get("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Requires explicitly manifested isolated database fixtures")
class ExpertHistoryTransactionTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = voice_tests.AttentionVoiceDatabaseTests.asyncSetUp
    complete = voice_tests.AttentionVoiceDatabaseTests.complete

    async def test_guarded_delivery_persists_history_without_another_pool_slot(self):
        await self.complete()
        # The actual guard still uses the manifested database. History may only
        # use that transaction; acquiring any separate slot fails deterministically.
        self.voice.repository = SessionRepository(NoNestedPool())
        delivery = await self.voice.try_attention(self.inbox, voice_tests.SyntheticSpeech(self.tts),
            worker_id="fixture-history", receipt_seconds=10)
        self.assertIsNotNone(delivery)
        await asyncio.wait_for(self.generated.wait(), 15)
        for index, item in delivery._items.items():
            await delivery.receipt(dict(type="playback.completed", item_id=delivery.item_id(index),
                response_id=delivery.response_id, played_samples=item["samples"]), completed=True)
        await asyncio.wait_for(delivery.task, 5)
        self.assertTrue(delivery._played.is_set())
        async with self.f.pool.acquire() as connection:
            count = await connection.fetchval("""select count(*) from jarvis_messages
                where owner_id=$1 and session_id=$2 and metadata->>'response_kind'='expert_tts'""",
                self.f.owner, self.session)
        self.assertEqual(count, 2)
