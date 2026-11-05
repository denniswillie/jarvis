import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.tasks.conversation import WorkMessage
from app.tasks.hermes_work import HermesWorkSession
from app.tasks.models import ClaimLost, TaskError
from app.tasks.runner import TaskRunner


class HermesWorkTests(unittest.IsolatedAsyncioTestCase):
    def setup_work(self, messages=()):
        context = SimpleNamespace(claim=object(), task=SimpleNamespace(owner_id="fixture"))
        mailbox = SimpleNamespace(read_inputs=AsyncMock(return_value=list(messages)), publish_progress=AsyncMock())
        return HermesWorkSession(context, mailbox, poll_seconds=0.01), mailbox

    async def test_initial_history_preserves_originals_and_source_roots(self):
        message = WorkMessage(4, uuid4(), {"text": "  use the original wording\n"})
        work, _ = self.setup_work([message])
        initial = await work.prepare()
        self.assertEqual(initial["messages"], [{"sequence": 4, "input": message.content}])
        self.assertEqual(work.cursor, 4)
        self.assertEqual(work.roots, {message.payload_id})

    async def test_progress_callback_never_waits_on_database(self):
        work, mailbox = self.setup_work()
        await work.offer_progress("First status")
        await work.offer_progress("Newest status")
        mailbox.publish_progress.assert_not_awaited()
        self.assertEqual(work.pending_progress, "Newest status")
        self.assertTrue(work.wake.is_set())

    async def test_three_followups_share_run_without_cancellation(self):
        messages = [WorkMessage(i, uuid4(), {"text": f"Correction {i}"}) for i in range(1, 4)]
        work, mailbox = self.setup_work(messages)
        mailbox.read_inputs.side_effect = [messages, []]
        sent = []
        complete = asyncio.Event()
        async def steer(text):
            sent.append(json.loads(text))
            if len(sent) == 3:
                complete.set()
        run = SimpleNamespace(steer=steer)
        task = asyncio.create_task(work.pump(run))
        try:
            await asyncio.wait_for(complete.wait(), 1)
            self.assertEqual(work.cursor, 3)
            self.assertEqual([item["work_followup"]["text"] for item in sent],
                             ["Correction 1", "Correction 2", "Correction 3"])
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_process_end_leaves_late_input_for_completion_guard(self):
        message = WorkMessage(7, uuid4(), {"text": "Late correction"})
        work, _ = self.setup_work([message])
        run = SimpleNamespace(steer=AsyncMock(side_effect=TaskError("hermes_input_run_ended", "Ended")))
        await work.pump(run)
        self.assertEqual(work.cursor, 0)

    async def test_withdrawn_context_stops_next_model_request(self):
        work, _ = self.setup_work()
        work.roots.add(uuid4())
        with self.assertRaises(ClaimLost):
            await work.guard(SimpleNamespace(fetch=AsyncMock(return_value=[])))

    async def test_new_input_during_guard_does_not_revoke_previous_context(self):
        work, _ = self.setup_work()
        original, arriving = uuid4(), uuid4()
        work.roots.add(original)
        async def fetch(_query, _owner, identities):
            self.assertEqual(identities, [original])
            work.roots.add(arriving)
            return [{"id": original, "availability": "available"}]
        await work.guard(SimpleNamespace(fetch=fetch))
        self.assertEqual(work.roots, {original, arriving})

    async def test_existing_domain_completion_keeps_priority(self):
        domain = SimpleNamespace(finish_outcome=AsyncMock(return_value=True))
        provider = SimpleNamespace(prepare=domain, finish_outcome=AsyncMock(return_value=True))
        self.assertTrue(await TaskRunner._finish_owned_outcome(provider, object(), object()))
        provider.finish_outcome.assert_not_awaited()
        domain.finish_outcome.return_value = False
        self.assertTrue(await TaskRunner._finish_owned_outcome(provider, object(), object()))
        provider.finish_outcome.assert_awaited_once()
