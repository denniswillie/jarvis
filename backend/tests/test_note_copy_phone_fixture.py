"""Offline checks of qualification evidence; no SQL, socket or native run."""
from copy import deepcopy
import json
import unittest

from evals.second_brain.note_copy_phone_fixture import RequestTrace, evaluate_trace


class CopyEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.first = "aaaaaaaa-1111-1111-1111-111111111111"
        self.second = "bbbbbbbb-2222-2222-2222-222222222222"
        self.manifest = {"initial_session_id": self.first, "request": {
            "request_id": "request", "file_id": "file", "version_id": "v1",
            "byte_count": 123, "content_sha256": "a" * 64}}
        self.receipt = {"state": "saved_local", "local_receipt": {
            key: value for key, value in self.manifest["request"].items() if key != "request_id"}}
        self.receipt["local_receipt"]["destination"] = "app_storage"
        def event(path, **extra):
            return dict(phase="native", path=path, status=200, authenticated=True, session_id=self.first, **extra)
        self.trace = [
            event("/sessions/start", response_session_id=self.first),
            event("/workspace/requests/request/ack", ack_state="claimed"),
            event("/files/file/content", response_bytes=123, content_sha256="a" * 64),
            event("/workspace/requests/request/ack", ack_state="saved_local"),
            event("/sessions/" + self.first + "/close"),
            event("/sessions/start", response_session_id=self.second),
            event("/client-context"),
        ]
        self.trace[-1]["session_id"] = self.second

    def evaluate(self, **kwargs):
        return evaluate_trace(self.manifest, kwargs.get("trace", self.trace), kwargs.get("receipt", self.receipt),
                              kwargs.get("network", ["offline", "online"]))

    def test_complete_server_evidence_and_idempotent_ack(self):
        self.assertTrue(all(self.evaluate().values()))
        self.trace.insert(4, deepcopy(self.trace[3]))
        self.assertTrue(all(self.evaluate().values()))

    def test_preflight_success_does_not_replace_native_download(self):
        self.trace[2]["phase"] = "preflight"
        self.assertFalse(self.evaluate()["one_exact_content_read"])

    def test_redownload_corrupt_copy_wrong_session_and_wrong_auth_are_rejected(self):
        for field, value in (("response_bytes", 124), ("content_sha256", "b" * 64),
                             ("session_id", self.second), ("authenticated", False)):
            with self.subTest(field=field):
                trace = deepcopy(self.trace)
                trace[2][field] = value
                self.assertFalse(self.evaluate(trace=trace)["one_exact_content_read"])
        self.assertFalse(self.evaluate(trace=self.trace + [self.trace[2]])["one_exact_content_read"])

    def test_current_version_receipt_does_not_prove_exact_old_copy(self):
        receipt = deepcopy(self.receipt)
        receipt["local_receipt"]["version_id"] = "v2"
        self.assertFalse(self.evaluate(receipt=receipt)["persisted_exact_receipt"])

    def test_unverified_or_unclosed_session_cannot_qualify_restart(self):
        for index in (4, 5, 6):
            with self.subTest(omitted=index):
                trace = self.trace[:index] + self.trace[index + 1:]
                self.assertFalse(self.evaluate(trace=trace)["new_session_reverified"])

    def test_ack_without_claim_or_without_real_network_cycle_fails(self):
        self.assertFalse(self.evaluate(trace=self.trace[:1] + self.trace[2:])["claimed_then_saved"])
        self.assertFalse(self.evaluate(network=[])["real_offline_then_online"])

    def test_mixed_case_UUIDs_identify_the_same_session_without_mutating_trace(self):
        trace = deepcopy(self.trace)
        for item in trace:
            item['session_id'] = item['session_id'].upper()
            if 'response_session_id' in item:
                item['response_session_id'] = item['response_session_id'].upper()
        trace[4]['path'] = '/sessions/' + self.first.upper() + '/close'
        before = deepcopy(trace)
        self.assertTrue(all(self.evaluate(trace=trace).values()))
        self.assertEqual(trace, before)

    def test_invalid_or_foreign_UUIDs_do_not_match_or_invent_a_new_session(self):
        for value in (None, 'not-a-uuid', self.second.upper()):
            trace = deepcopy(self.trace);trace[1]['session_id'] = value
            self.assertFalse(self.evaluate(trace=trace)['claimed_then_saved'])
        trace = deepcopy(self.trace)
        trace[5]['response_session_id'] = self.first.upper()
        trace[6]['session_id'] = self.first.upper()
        self.assertFalse(self.evaluate(trace=trace)['new_session_reverified'])
        self.manifest['initial_session_id'] = 'invalid'
        self.assertFalse(self.evaluate()['claimed_then_saved'])
        self.assertFalse(self.evaluate()['one_exact_content_read'])


class TraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_preserved_and_credentials_and_response_body_not_retained(self):
        state = {"phase": "native", "requests": []}
        saved = []
        received = []
        output = []
        body = b'{"session_id":"first","state":"saved_local","unlogged_secret":"sensitive"}'
        async def receive(): return {"type": "http.request", "body": body, "more_body": False}
        async def send(value): output.append(value)
        async def app(scope, receive, send):
            received.append(await receive())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b'{"request":', "more_body": True})
            await send({"type": "http.response.body", "body": b'{"private":"body"}}', "more_body": False})
        trace = RequestTrace(app, state=state, persist=lambda: saved.append(True), token="fixture-secret")
        await trace({"type": "http", "method": "POST", "path": "/workspace/requests/request/ack",
                     "query_string": b"unlogged=value", "headers": [(b"authorization", b"Bearer fixture-secret")]}, receive, send)
        self.assertEqual(received[0]["body"], body)
        self.assertEqual(len(output), 3)
        self.assertEqual(saved, [True])
        self.assertEqual(state["requests"][0]["ack_state"], "saved_local")
        self.assertEqual(state["requests"][0]["session_id"], "first")
        for excluded in ("fixture-secret", "sensitive", "private", "unlogged"):
            self.assertNotIn(excluded, json.dumps(state))


if __name__ == "__main__":
    unittest.main()
