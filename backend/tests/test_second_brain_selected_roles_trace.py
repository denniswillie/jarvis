"""Selected-role acceptance verdicts must require actual source/tool provenance."""
from copy import deepcopy
import unittest
from uuid import uuid4

from evals.second_brain.selected_roles_trace import verify_results


class SelectedRoleVerdictTests(unittest.TestCase):
    def setUp(self):
        self.source, self.event = uuid4(), uuid4()
        self.memory = {"answer": "You reported AZURE 12345. Calibration remains unfinished.",
            "evidence": [{"source_id": str(self.source)}, {"event_id": str(self.event)}],
            "tool_receipts": [{"name": "search_memory", "status": "success"}],
            "authorship": {"kind": "expert"}, "speech": [{"text": "Calibration is unfinished."}]}
        self.code = {"answer": "mid = (low + high) // 2", "authorship": {"kind": "expert"},
                     "speech": [{"text": "Add low and high, then divide by two."}]}

    def check(self, memory=None, code=None):
        verify_results(self.memory if memory is None else memory, self.code if code is None else code,
                       label="AZURE 12345", source_id=self.source, event_id=self.event)

    def test_complete_evidence_passes(self):
        self.check()

    def test_missing_or_wrong_provenance_fails(self):
        for key, value in (("evidence", [{"source_id": str(uuid4())}]),
                           ("tool_receipts", []), ("speech", []),
                           ("authorship", {"kind": "runtime"}),
                           ("answer", "The work is complete.")):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                self.check(deepcopy(self.memory) | {key: value})
        with self.assertRaises(AssertionError):
            self.check(code=deepcopy(self.code) | {"answer": "mid = low + high // 2"})


class SelectedRoleCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_bank_cleanup_precedes_client_close(self):
        from contextlib import asynccontextmanager
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, Mock, patch
        from evals.second_brain.selected_roles_trace import cleanup_memory

        connection = SimpleNamespace(fetch=AsyncMock(side_effect=[
            [{"bank_name": "fixture-only-bank"}], []]), fetchval=AsyncMock(return_value=0))

        class Pool:
            @asynccontextmanager
            async def acquire(self):
                yield connection

        closed = False

        async def close():
            nonlocal closed
            closed = True

        async def call(method, path, **kwargs):
            self.assertFalse(closed, "Native cleanup used a closed client")
            if method == "GET":
                return {"banks": []}

        client = SimpleNamespace(_call=AsyncMock(side_effect=call), _bank=lambda bank: "/banks/" + bank)
        runtime = SimpleNamespace(client=client, close=AsyncMock(side_effect=close), start=Mock(),
                                  sources=SimpleNamespace(delete_source=AsyncMock()))
        fixture = SimpleNamespace(pool=Pool(), owner="fixture-owner", area=uuid4())
        with TemporaryDirectory() as directory, patch(
                "app.knowledge.runtime.KnowledgeRuntime.create", AsyncMock(return_value=runtime)):
            await cleanup_memory(fixture, Path(directory), "synthetic-key")
        self.assertTrue(closed)
        self.assertEqual(client._call.call_count, 2)


class SelectedRoleResultReadTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_owner_scoped_task_repository_and_exact_payload(self):
        from contextlib import asynccontextmanager
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from evals.second_brain.selected_roles_trace import read_task_results

        identity, payload = uuid4(), uuid4()
        connection = SimpleNamespace(fetchval=AsyncMock(return_value='{"answer":"synthetic"}'))

        class Pool:
            @asynccontextmanager
            async def acquire(self):
                yield connection

        tasks = SimpleNamespace(pool=Pool(), get=AsyncMock(return_value=SimpleNamespace(result_payload_id=payload)))
        self.assertEqual(await read_task_results(tasks, "fixture-owner", [str(identity)]),
                         [{"answer": "synthetic"}])
        tasks.get.assert_awaited_once_with("fixture-owner", identity)
        self.assertEqual(connection.fetchval.await_args.args[1:], ("fixture-owner", payload))
