"""Production routing evidence excludes user and provider payloads."""
import json
import unittest
from unittest.mock import Mock

from app.realtime.live_router import LunaLiveRouter


class RouterDiagnosticTests(unittest.TestCase):
    def test_logs_only_stage_metadata_and_preserves_local_observer(self):
        router = object.__new__(LunaLiveRouter)
        router.observe = Mock()
        with self.assertLogs("app.realtime.live_router", level="INFO") as logs:
            router.event("router_request", delegation_id="test-delegation", step=2,
                         request={"input": "private-dialogue"},
                         response={"output": "private-response"},
                         arguments={"query": "private-query"})
        record = json.loads(logs.output[0].split("live_router_diagnostic=", 1)[1])
        self.assertEqual(record["stage"], "router_request")
        self.assertEqual(record["step"], 2)
        self.assertEqual(record["delegation_id"], "test-delegation")
        self.assertNotIn("private-", logs.output[0])
        self.assertEqual(router.observe.call_args.args[0]["request"],
                         {"input": "private-dialogue"})

    def test_failure_type_is_available_without_exception_contents(self):
        router = object.__new__(LunaLiveRouter)
        router.observe = Mock()
        with self.assertLogs("app.realtime.live_router", level="INFO") as logs:
            router.event("router_failed", error_type="TimeoutError")
        self.assertIn("TimeoutError", logs.output[0])
