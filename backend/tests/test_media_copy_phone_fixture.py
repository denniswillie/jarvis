"""Pure manifest/trace qualification checks; no SQL, listener, probe or native run."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from uuid import uuid4

from evals.second_brain.media_copy_phone_fixture import (
    EXPECTED_SHA256, EXPECTED_SIZE, MediaRequestTrace, application, evaluate_trace, read_stage, validate_manifest)


def fixture():
    now = datetime.now(UTC)
    identity, version, session = str(uuid4()), "exact-opaque-v1", str(uuid4())
    base = dict(artifact_kind="file", artifact_id=identity, version_id=version, content_ready=True,
                state="pending", created_at=now.isoformat(), expires_at=(now + timedelta(minutes=5)).isoformat())
    opened = base | {"request_id": str(uuid4()), "action": "open"}
    saved = base | dict(request_id=str(uuid4()), action="download", file_id=identity, byte_count=EXPECTED_SIZE,
                        content_sha256=EXPECTED_SHA256, transfer_policy="mp4_output_v1", transfer_limit_bytes=20_000_000)
    return dict(schema_version=1, run_id=str(uuid4()).upper(), base_url="http://127.0.0.1:34567", credential_path="/private/fixture-token",
        expected_owner_id="fixture-owner", expected_backend_namespace="fixture-backend", initial_session_id=session,
        open_request=opened, download_request=saved, filename="synthetic-generated.mp4", content_type="video/mp4")


class MediaManifestTests(unittest.TestCase):
    def test_valid_paired_manifest_and_phone_schema(self):
        manifest = fixture()
        self.assertEqual(validate_manifest(manifest), manifest)
        schema_path = Path(__file__).resolve().parents[2] / ".jarvis-dev/phone-tests/connected-media-20261102/fixture.schema.json"
        schema = json.loads(schema_path.read_text())
        self.assertEqual(set(schema["required"]), set(manifest))
        download = schema["$defs"]["download_request"]["properties"]
        self.assertEqual(download["byte_count"]["const"], EXPECTED_SIZE)
        self.assertEqual(download["content_sha256"]["const"], EXPECTED_SHA256)

    def test_mismatched_pair_save_fields_escalation_and_wrong_pin_rejected(self):
        for name, changes in (("open_request", {"artifact_id": str(uuid4())}),
                ("open_request", {"file_id": str(uuid4())}), ("open_request", {"state": "displayed"}),
                ("download_request", {"byte_count": EXPECTED_SIZE - 1}),
                ("download_request", {"content_sha256": "0" * 64}),
                ("download_request", {"transfer_policy": "mp4_capture_v1"}),
                ("download_request", {"transfer_limit_bytes": 64 * 1024**2}),
                ("download_request", {"file_id": str(uuid4())})):
            value = fixture(); value[name].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_manifest(value)
        value = fixture(); value["open_request"]["request_id"] = value["download_request"]["request_id"]
        with self.assertRaises(ValueError): validate_manifest(value)

    def test_remote_origins_embedded_credentials_unknown_fields_and_wrong_run_case_rejected(self):
        for changes in ({"base_url": "https://example.com"}, {"base_url": "http://secret@127.0.0.1"},
                        {"base_url": "http://127.0.0.1?secret=token"}, {"credential_path": "relative"},
                        {"bearer_token": "must-not-appear"}, {"schema_version": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_manifest(fixture() | changes)
        value = fixture(); value["run_id"] = value["run_id"].lower()
        with self.assertRaises(ValueError): validate_manifest(value)

    def test_stage_is_bounded_and_bound_to_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path, run = Path(directory) / "stage.json", str(uuid4()).upper()
            self.assertEqual(read_stage(path, run)["phase"], "startup")
            stage = dict(run_id=run, phase="save", session_id=str(uuid4()).upper(), instruction="not retained")
            path.write_text(json.dumps(stage))
            self.assertEqual(set(read_stage(path, run)), {"phase", "session_id"})
            for changes in ({"run_id": str(uuid4())}, {"phase": "invented"}, {"session_id": "bad"}):
                path.write_text(json.dumps(stage | changes))
                with self.assertRaises(ValueError): read_stage(path, run)
            path.write_bytes(b"x" * 16_385)
            with self.assertRaises(ValueError): read_stage(path, run)


class MediaEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.manifest = fixture()
        opened, saved = self.manifest["open_request"], self.manifest["download_request"]
        self.local_receipt = {key: saved[key] for key in ("file_id", "version_id", "byte_count", "content_sha256")}
        self.local_receipt["destination"] = "app_storage"
        self.receipts = {"open": opened | {"state": "displayed"},
                         "download": saved | {"state": "saved_local", "local_receipt": self.local_receipt}}
        self.trace = []
        self.initial = self.manifest["initial_session_id"]
        self.fresh = str(uuid4())

        def event(path, *, stage="verify-account", method="GET", **fields):
            start = len(self.trace) * 2
            result = dict(sequence=len(self.trace), request_started_order=start, response_completed_order=start + 1,
                request_started_ns=start * 1000, response_completed_ns=(start + 1) * 1000,
                phase="native", path=path, stage=stage, method=method,
                status=200, authenticated=True, response_complete=True, session_id=self.initial, **fields)
            self.trace.append(result)
            return result

        def context(session):
            row = event("/client-context", context={"owner_id": self.manifest["expected_owner_id"],
                "backend_namespace": self.manifest["expected_backend_namespace"], "session_id": session})
            row["session_id"] = session

        def ack(target, state, stage):
            return event("/workspace/requests/" + target["request_id"] + "/ack", method="POST", stage=stage,
                request_id=target["request_id"], ack_state=state, response_ack_state=state,
                local_receipt=self.local_receipt if state == "saved_local" else None,
                response_local_receipt=self.local_receipt if state == "saved_local" else None)

        context(self.initial)
        event("/sessions/start", method="POST", response_session_id=self.initial)
        event("/files/" + saved["file_id"], stage="open-video", version_id=saved["version_id"])
        for target, terminal, stage in ((opened, "displayed", "open-video"), (saved, "saved_local", "save")):
            ack(target, "claimed", stage)
            event("/files/" + saved["file_id"] + "/content", stage=stage, version_id=saved["version_id"],
                response_version_id=saved["version_id"], response_bytes=EXPECTED_SIZE,
                content_sha256=EXPECTED_SHA256, observed_sha256=EXPECTED_SHA256)
            ack(target, terminal, stage)
        event("/sessions/" + self.initial + "/close", method="POST", stage="restore-online")
        event("/sessions/start", method="POST", stage="verify-after-reauthentication", response_session_id=self.fresh)
        context(self.fresh)

    def evaluate(self, trace=None, receipts=None, network=None):
        return evaluate_trace(self.manifest, self.trace if trace is None else trace,
            self.receipts if receipts is None else receipts, ["offline", "online"] if network is None else network)

    def test_complete_server_evidence_is_independent_of_operator_native_claim(self):
        self.assertTrue(all(self.evaluate().values()))
        repeated = deepcopy(self.trace)
        repeated.append(deepcopy(repeated[8]) | {"sequence": 8.5})  # Idempotent saved ACK before offline.
        self.assertTrue(all(self.evaluate(trace=repeated).values()))

    def test_preflight_cannot_supply_native_bytes_and_extra_failed_reads_fail(self):
        trace = deepcopy(self.trace); trace[4]["phase"] = "preflight"
        self.assertFalse(self.evaluate(trace=trace)["exactly_two_native_content_reads"])
        trace = deepcopy(self.trace); trace.append(trace[4] | {"sequence": 99, "status": 404})
        self.assertFalse(self.evaluate(trace=trace)["exactly_two_native_content_reads"])

    def test_wrong_hash_size_version_session_auth_or_partial_body_fails(self):
        for field, value in (("observed_sha256", "0" * 64), ("content_sha256", "0" * 64),
                ("response_bytes", EXPECTED_SIZE - 1), ("version_id", "other"), ("response_version_id", "other"),
                ("session_id", self.fresh), ("authenticated", False), ("response_complete", False)):
            trace = deepcopy(self.trace); trace[4][field] = value
            with self.subTest(field=field):
                self.assertFalse(self.evaluate(trace=trace)["exactly_two_native_content_reads"])

    def test_open_and_save_receipts_do_not_substitute_for_each_other(self):
        records = deepcopy(self.receipts); records["open"]["state"] = "saved_local"
        self.assertFalse(self.evaluate(receipts=records)["open_persisted_exact_outcome"])
        records = deepcopy(self.receipts); records["download"]["local_receipt"]["version_id"] = "other"
        self.assertFalse(self.evaluate(receipts=records)["download_persisted_exact_outcome"])
        trace = deepcopy(self.trace); trace[5]["ack_state"] = trace[5]["response_ack_state"] = "saved_local"
        self.assertFalse(self.evaluate(trace=trace)["open_claimed_then_terminal"])
        trace = deepcopy(self.trace); trace[4]["sequence"] = 2.5
        self.assertFalse(self.evaluate(trace=trace)["open_read_between_claim_and_receipt"])

    def test_reopen_new_download_or_ack_and_audio_perception_fail(self):
        for index, stage in ((4, "open-offline"), (8, "open-after-reauthentication")):
            trace = deepcopy(self.trace); trace[index]["stage"] = stage
            self.assertFalse(self.evaluate(trace=trace)["no_copy_or_ack_on_reopen"])
        for path, method in (("/realtime/session", "POST"), ("/visual/frames", "GET"), ("/tasks", "POST")):
            trace = deepcopy(self.trace); trace.append(trace[0] | {"sequence": 99, "path": path, "method": method})
            self.assertFalse(self.evaluate(trace=trace)["no_audio_perception_or_other_mutation"])

    def test_overlapping_claim_or_content_response_cannot_pass_start_order_alone(self):
        for index, next_index in ((3, 4), (4, 5), (6, 7), (7, 8)):
            with self.subTest(index=index):
                trace = deepcopy(self.trace)
                trace[index]["response_completed_order"] = trace[next_index]["request_started_order"] + 1
                trace[index]["response_completed_ns"] = trace[next_index]["request_started_ns"] + 1
                name = "open" if index < 6 else "download"
                self.assertFalse(self.evaluate(trace=trace)[name + "_read_between_claim_and_receipt"])

    def test_real_close_new_session_and_owner_namespace_verification_required(self):
        for index in (1, 9, 10, 11):
            self.assertFalse(self.evaluate(trace=self.trace[:index] + self.trace[index + 1:])["initial_reused_then_new_verified_session"])
        trace = deepcopy(self.trace); trace[-1]["context"]["owner_id"] = "foreign"
        self.assertFalse(self.evaluate(trace=trace)["initial_reused_then_new_verified_session"])
        self.assertFalse(self.evaluate(network=[])["real_offline_then_online"])


class MediaTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_observed_bytes_are_hashed_and_metadata_never_retains_credentials_or_bodies(self):
        with tempfile.TemporaryDirectory() as directory:
            stage_path = Path(directory) / "stage.json"
            manifest = fixture()
            stage_path.write_text(json.dumps({"run_id": manifest["run_id"], "phase": "open-video", "session_id": "none"}))
            state = {"phase": "native", "requests": []}
            body = b"private movie fixture bytes"
            sent = []

            async def app(scope, receive, send):
                await send({"type": "http.response.start", "status": 200, "headers": [
                    (b"x-content-sha256", b"declared"), (b"x-jarvis-version-id", b"exact")]})
                await send({"type": "http.response.body", "body": body[:5], "more_body": True})
                await send({"type": "http.response.body", "body": body[5:], "more_body": False})

            async def receive(): return {"type": "http.request", "body": b""}
            async def send(message): sent.append(message)
            trace = MediaRequestTrace(app, state=state, persist=lambda: None, token="fixture-secret",
                stage_path=stage_path, run_id=manifest["run_id"])
            await trace({"type": "http", "method": "GET", "path": "/files/" + str(uuid4()) + "/content",
                "headers": [(b"authorization", b"Bearer fixture-secret")],
                "query_string": ("session_id=" + manifest["initial_session_id"] + "&version_id=exact&secret=not-retained").encode()}, receive, send)
            result = state["requests"][0]
            self.assertEqual(result["observed_sha256"], hashlib.sha256(body).hexdigest())
            self.assertEqual(result["response_bytes"], len(body))
            self.assertTrue(result["response_complete"])
            self.assertEqual(result["request_started_order"], 0)
            self.assertEqual(result["response_completed_order"], 1)
            self.assertLessEqual(result["request_started_ns"], result["response_completed_ns"])
            self.assertEqual(result["stage"], "open-video")
            self.assertEqual(b"".join(item.get("body", b"") for item in sent), body)
            for excluded in ("fixture-secret", "private movie", "not-retained"):
                self.assertNotIn(excluded, json.dumps(state))

    async def test_ack_trace_normalizes_ids_and_captures_only_receipt_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = fixture(); target = manifest["download_request"]
            local = {key: target[key] for key in ("file_id", "version_id", "byte_count", "content_sha256")}
            local["destination"] = "app_storage"
            incoming = {"session_id": manifest["initial_session_id"].upper(), "state": "saved_local", "local_receipt": local,
                        "unlogged": "sensitive input"}
            outgoing = {"request": {"state": "saved_local", "local_receipt": local}, "private": "sensitive response"}
            state = {"phase": "preflight", "requests": []}

            async def receive(): return {"type": "http.request", "body": json.dumps(incoming).encode()}
            async def send(message): pass
            async def app(scope, receive, send):
                await receive()
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": json.dumps(outgoing).encode()})
            trace = MediaRequestTrace(app, state=state, persist=lambda: None, token="fixture-secret",
                stage_path=Path(directory) / "unused", run_id=manifest["run_id"])
            await trace({"type": "http", "method": "POST", "path": "/workspace/requests/" + target["request_id"].upper() + "/ack",
                "query_string": b"", "headers": [(b"authorization", b"Bearer fixture-secret")]}, receive, send)
            result = state["requests"][0]
            self.assertEqual(result["request_id"], target["request_id"])
            self.assertEqual(result["session_id"], manifest["initial_session_id"])
            self.assertEqual(result["local_receipt"], result["response_local_receipt"])
            self.assertEqual(result["phase"], "preflight")
            self.assertNotIn("sensitive", json.dumps(state))

    async def test_shared_application_has_exactly_one_media_trace_writer(self):
        services = SimpleNamespace(settings=SimpleNamespace(JARVIS_STATIC_BEARER_TOKEN="fixture"))
        app = application(services, {"phase": "preflight", "requests": []}, lambda: None,
                          stage_path=Path("/unused"), run_id=str(uuid4()).upper())
        self.assertEqual([middleware.cls for middleware in app.user_middleware], [MediaRequestTrace])


if __name__ == "__main__":
    unittest.main()
