"""Negative controls for the native GPT acceptance trace audit."""
from types import SimpleNamespace
import unittest

from evals.glasses_reliability.gpt_turn_probe import TraceCapture


class NativeTurnTraceTests(unittest.TestCase):
    def fixture(self):
        trace = TraceCapture("fixture")
        trace.records = [
            {"event": "input_commit_received", "turn_generation": 1, "monotonic_s": 10, "process_id": "p"},
            {"event": "response_admitted", "request_id": "request", "turn_generation": 1, "process_id": "p"},
            {"event": "provider_response_started", "request_id": "request", "response_id": "response", "process_id": "p"},
            {"event": "first_audio_forwarded", "response_id": "response", "monotonic_s": 12.5, "process_id": "p"},
        ]
        wire = SimpleNamespace(
            writes=[{"type": "response.create", "request_id": "request"}],
            response_generations={"response": "1"}, first_audio={"response": 12},
        )
        return trace, wire

    def test_complete_trace_separates_provider_audio_from_forwarding(self):
        trace, wire = self.fixture()
        result = trace.audit(wire)
        self.assertEqual(result["unexplained_requests"], [])
        self.assertEqual(result["untraced_native_requests"], [])
        self.assertEqual(result["timing"][0]["commit_to_first_provider_audio_ms"], 2000)
        self.assertEqual(result["timing"][0]["commit_to_first_forwarded_audio_ms"], 2500)

    def test_missing_admission_is_not_hidden_by_an_empty_request_set(self):
        trace, wire = self.fixture()
        trace.records = [row for row in trace.records if row["event"] != "response_admitted"]
        self.assertEqual(trace.audit(wire)["untraced_native_requests"], ["request"])

    def test_missing_provider_start_leaves_an_unexplained_admission(self):
        trace, wire = self.fixture()
        trace.records = [row for row in trace.records if row["event"] != "provider_response_started"]
        self.assertEqual(trace.audit(wire)["unexplained_requests"], ["request"])

    def test_never_submitted_terminal_outcome_explains_a_prewrite_failure(self):
        trace, wire = self.fixture()
        trace.records = [row for row in trace.records if row["event"] != "provider_response_started"]
        trace.records.append({"event": "response_not_submitted", "request_id": "request", "process_id": "p"})
        wire.writes = []
        self.assertEqual(trace.audit(wire)["unexplained_requests"], [])

    def test_explicit_rejection_explains_only_its_matching_admission(self):
        trace, wire = self.fixture()
        trace.records = [row for row in trace.records if row["event"] != "provider_response_started"]
        trace.records.append({"event": "response_rejected", "request_id": "unrelated", "process_id": "p"})
        self.assertEqual(trace.audit(wire)["unexplained_requests"], ["request"])
        trace.records.append({"event": "response_rejected", "request_id": "request", "process_id": "p"})
        self.assertEqual(trace.audit(wire)["unexplained_requests"], [])
        self.assertEqual(trace.audit(wire)["rejected_requests"], 1)
