"""Native adapter mechanics only: no model, database or sandbox calls."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from app.tasks.hermes import _NativeRun
from app.tasks.hermes_child import handle_control, run_conversation
from app.tasks.models import TaskError


class NativeControlTests(unittest.TestCase):
    def setUp(self):
        self.agent = SimpleNamespace(steer=Mock(return_value=True), interrupt=Mock())

    def test_followups_use_same_agent_without_interrupting_work(self):
        for text in ('Use the blue label.', 'Actually use green.', 'How is it going?'):
            self.assertTrue(handle_control(self.agent, {'operation': 'steer', 'text': text}))
        self.assertEqual([call.args[0] for call in self.agent.steer.call_args_list],
                         ['Use the blue label.', 'Actually use green.', 'How is it going?'])
        self.agent.interrupt.assert_not_called()

    def test_cancel_remains_separate_from_steering(self):
        self.assertFalse(handle_control(self.agent, {'operation': 'cancel'}))
        self.agent.interrupt.assert_called_once()
        self.agent.steer.assert_not_called()

    def test_malformed_and_oversized_messages_do_not_reach_agent(self):
        for message in (None, {'operation': 'unknown'}, {'operation': 'steer', 'text': ' '},
                        {'operation': 'steer', 'text': 'x' * 64001},
                        {'operation': 'steer', 'text': 'Hi', 'untrusted_extra': True}):
            with self.subTest(kind=type(message).__name__), self.assertRaises(ValueError):
                handle_control(self.agent, message)
        self.agent.steer.assert_not_called()

    def test_existing_native_history_interface_is_preserved(self):
        agent = SimpleNamespace(run_conversation=Mock(return_value={'completed': True}))
        history = [{'role': 'user', 'content': 'Earlier original request.'}]
        config = {'manifest': {'task': 'current'}, 'system_prompt': 'system',
                  'task_id': 'work', 'conversation_history': history}
        run_conversation(agent, config)
        self.assertEqual(agent.run_conversation.call_args.kwargs['conversation_history'], history)


class NativeHostTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self):
        run = object.__new__(_NativeRun)
        run.run_id = 'test-run'
        run.configuration = SimpleNamespace(max_iterations=4)
        run.marker = {}
        run.gateway = SimpleNamespace(diagnostics=[])
        run.write_marker = Mock()
        run.on_progress = AsyncMock()
        run._ending = False
        output, errors = asyncio.StreamReader(), asyncio.StreamReader()
        errors.feed_eof()
        run.process = SimpleNamespace(returncode=None, stdout=output, stderr=errors,
            stdin=SimpleNamespace(write=Mock(), drain=AsyncMock()), wait=AsyncMock())

        async def wait():
            run.process.returncode = 0

        run.process.wait.side_effect = wait
        return run

    def feed(self, run, event):
        run.process.stdout.feed_data((json.dumps({'run_id': run.run_id, **event}) + '\n').encode())

    async def test_progress_and_final_result_share_existing_reader(self):
        run = self.fixture()
        self.feed(run, {'type': 'ready'})
        self.feed(run, {'type': 'progress', 'text': 'I am checking the existing file.'})
        self.feed(run, {'type': 'result', 'completed': True, 'final_response': 'actual answer'})
        run.process.stdout.feed_eof()
        self.assertEqual((await run.result())['final_response'], 'actual answer')
        run.on_progress.assert_awaited_once_with('I am checking the existing file.')
        with self.assertRaises(TaskError):
            await run.steer('A follow-up after the process finished')

    async def test_steering_preserves_original_text_without_restart_or_receipt_machine(self):
        run = self.fixture()
        text = '  Keep my original words and spacing.\nNew details.  '
        await run.steer(text)
        payload = json.loads(run.process.stdin.write.call_args.args[0])
        self.assertEqual(payload, {'operation': 'steer', 'text': text})
        run.process.stdin.drain.assert_awaited_once()

    async def test_ended_worker_and_invalid_input_are_rejected(self):
        run = self.fixture()
        with self.assertRaises(TaskError):
            await run.steer(' ')
        run._ending = True
        with self.assertRaises(TaskError):
            await run.steer('New request')
        run.process.stdin.write.assert_not_called()

    async def test_unknown_events_are_not_presented_as_progress(self):
        run = self.fixture()
        self.feed(run, {'type': 'private_reasoning', 'text': 'Not an assistant message'})
        with self.assertRaises(TaskError):
            await run.result()
        run.on_progress.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
