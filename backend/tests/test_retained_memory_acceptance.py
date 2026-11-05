"""Offline-only qualification of the one-shot native memory receipt harness."""

from argparse import Namespace
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/retained_memory_acceptance.py"
SPEC = importlib.util.spec_from_file_location("retained_memory_acceptance", SCRIPT)
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def completed_receipt():
    return {
        "admission_id": "admission-1", "status": "terminal", "outcome": "completed",
        "result": {
            "final_response": "The recalled source says voice-smoke-03.txt contains 391.",
            "reasoning": "PRIVATE_REASONING_SENTINEL",
            "messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "terminal"}}]},
                {"role": "user", "content": harness.PROMPT},
                {"role": "assistant", "tool_calls": [{"function": {
                    "name": "hindsight_recall", "arguments": '{"query":"voice-smoke-03.txt"}'}}]},
                {"role": "tool", "content": "RAW_TOOL_CONTENT_SENTINEL"},
            ],
        },
    }


class FakeClient:
    def __init__(self, receipts):
        self.receipts = copy.deepcopy(receipts)
        self.requests = []
        self.submissions = []
        self.snapshots = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def snapshot(self, native):
        self.snapshots.append(native)
        return {"session_id": native}

    async def submit(self, native, input_id, text):
        self.submissions.append((native, input_id, text))
        return {"admission_id": "admission-1", "status": "queued"}

    async def request(self, method, **params):
        self.requests.append((method, params))
        return self.receipts.pop(0)


class RetainedMemoryAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, receipts, *, admission_id=None):
        client = FakeClient(receipts)
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(profile=Path(directory), output=Path(directory),
                             native=harness.NATIVE, master=harness.MASTER,
                             admission_id=admission_id, seconds=1)
            report = {}
            with patch("app.hermes_sessions.client.HermesGatewayClient", return_value=client), \
                    patch.object(harness.asyncio, "sleep", new_callable=AsyncMock):
                await harness.live(args, report)
        return client, report

    async def test_receipt_requests_result_and_submits_only_once(self):
        client, report = await self.exercise([
            {"status": "queued"}, {"status": "started"}, completed_receipt(),
        ])
        self.assertEqual(len(client.submissions), 1)
        self.assertEqual(len(client.requests), 3)
        for method, params in client.requests:
            self.assertEqual(method, "prompt.receipt")
            self.assertIs(params["include_result"], True)
            self.assertEqual(params["session_id"], harness.NATIVE)
            self.assertEqual(params["admission_id"], "admission-1")
        self.assertEqual(report["status"], "terminal")
        self.assertEqual(report["outcome"], "completed")
        self.assertTrue(report["recall_tool_called_once_and_expected_answer"])

    async def test_existing_admission_reconciles_without_resubmission(self):
        client, report = await self.exercise([completed_receipt()], admission_id="admission-1")
        self.assertEqual(client.submissions, [])
        self.assertEqual(len(client.requests), 1)
        self.assertIs(client.requests[0][1]["include_result"], True)
        self.assertTrue(report["terminal"])
        self.assertFalse(report["recall_tool_called_once_and_expected_answer"])

    async def test_terminal_failure_stops_without_retry(self):
        client, report = await self.exercise([
            {"admission_id": "admission-1", "status": "terminal", "outcome": "failed"},
        ])
        self.assertEqual(len(client.submissions), 1)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(report["outcome"], "failed")
        self.assertFalse(report["recall_tool_called_once_and_expected_answer"])

    def test_projection_is_current_turn_only_and_does_not_expose_private_content(self):
        report = harness.project(completed_receipt(), current_prompt=harness.PROMPT)
        self.assertEqual(report["tool_names"], ["hindsight_recall"])
        self.assertIn("391", report["final_response"])
        self.assertNotIn("PRIVATE_REASONING_SENTINEL", json.dumps(report))
        self.assertNotIn("RAW_TOOL_CONTENT_SENTINEL", json.dumps(report))
        self.assertNotIn("messages", report)
        self.assertNotIn("reasoning", report)

    def test_completed_outcome_without_terminal_status_does_not_qualify(self):
        receipt = completed_receipt()
        receipt["status"] = "started"
        report = harness.project(receipt, current_prompt=harness.PROMPT)
        self.assertFalse(report["terminal"])
        self.assertFalse(report["recall_tool_called_once_and_expected_answer"])


if __name__ == "__main__":
    unittest.main()
