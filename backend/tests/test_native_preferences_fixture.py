"""Offline fixture-support checks, NOT SQL/native/provider chain qualification."""
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import httpx

from evals.responsive_voice.frontend_host import ReplayHostScope
from evals.responsive_voice.native_preferences_fixture import (
    FORMAT, OPT_IN, PHASES, Driver, FixtureError, PreferencesTrace, attach_preferences_trace,
    commit_rows, created_session_row, decode, prepare, private_directory, profile, progress, provider_evidence,
    push_row, put_once, read_json, resources, rows, save,
)


class PreferencesFixtureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.root.chmod(0o700)
        self.run = uuid4().hex
        self.owner = f"direct-image-eval-second-brain-{self.run}-a"
        self.scope = ReplayHostScope(run_id=self.run, owner_id=self.owner,
            database_schema=f"jarvis_sb_eval_{self.run}_0", area_id=uuid4(),
            memory_bank_id="jarvis_" + uuid4().hex, expires_at=datetime.now(UTC) + timedelta(minutes=5))
        self.manifest_path = self.root / f"frontend-host-resources-{self.run}.json"
        self.host = {"format": "jarvis.frontend_host.resources.v1", "run_id": self.run,
            "owner_id": self.owner, "database_schema": self.scope.database_schema,
            "area_id": str(self.scope.area_id), "memory_bank_id": self.scope.memory_bank_id,
            "scope_file": str(self.root / f"frontend-host-scope-{self.run}.json"),
            "fixture_manifest": str(self.root / f"second-brain-{self.run}.json"),
            "control_file": str(self.root / f"frontend-host-control-{self.run}.json"),
            "bearer_token_file": str(self.root / f"frontend-host-bearer-{self.run}.token"),
            "status": "prepared", "capability_profile": "native_realtime_media",
            "native_options": {"voice_route": "realtime", "workload": "none"}}
        save(Path(self.host["scope_file"]), self.scope.model_dump(mode="json"), new=True)
        save(self.manifest_path, self.host, new=True)
        self.wire = private_directory(self.root / "wire", create=True)
        self.session, self.audio = str(uuid4()), str(uuid4())

    def committed(self, revision=1, text="Synthetic preference"):
        return {"owner_id": self.owner, "revision": revision, "behavior_instructions": text,
                "created_at": "2026-11-02T12:00:00+00:00", "updated_at": "2026-11-02T12:00:01+00:00"}

    def test_prepare_matches_exact_phone_manifest_and_private_ordered_stage_contract(self):
        path = prepare(self.manifest_path, self.root / "controller", self.wire)
        plan = read_json(path)
        manifest = read_json(plan["ui_manifest"])
        self.assertEqual(plan["phases"], list(PHASES))
        self.assertEqual(manifest["run_id"], str(uuid4().__class__(self.run)).upper())
        self.assertEqual(set(manifest), {"schema_version", "run_id", "stage_path", "not_before", "draft",
                                        "pushed_while_dirty", "pushed_while_clean"})
        self.assertEqual(Path(manifest["stage_path"]).parent.name, "phone-preferences-" + manifest["run_id"])
        self.assertEqual(Path(plan["ui_manifest"]).stat().st_mode & 0o077, 0)
        self.assertFalse(Path(manifest["stage_path"]).exists())
        self.assertEqual(read_json(self.wire / OPT_IN)["owner_id"], self.owner)
        self.assertEqual(len({manifest[key] for key in ("draft", "pushed_while_dirty", "pushed_while_clean")}), 3)

    def test_scope_rejects_personal_owner_foreign_paths_and_unserved_host(self):
        with self.assertRaises(FixtureError):
            resources(self.manifest_path, serving=True)
        for update in ({"owner_id": "personal-owner"}, {"control_file": str(self.root / "another-control.json")},
                       {"native_options": {"voice_route": "live", "workload": "none"}}):
            save(self.manifest_path, self.host | update)
            with self.subTest(update=update), self.assertRaises(FixtureError):
                resources(self.manifest_path)

    def test_json_is_bounded_finite_and_unambiguous(self):
        for raw in ('{"x":1,"x":2}', '{"at":NaN}'):
            with self.assertRaises(FixtureError):
                decode(raw)

    def test_stale_wrong_run_and_wrong_phase_progress_cannot_advance(self):
        now = time.time()
        manifest = {"run_id": self.run, "not_before": now - 2}
        good = {"run_id": self.run, "phase": "edit-draft", "at": now}
        self.assertEqual(progress(good, manifest, "edit-draft", now - 1), good)
        for change in ({"run_id": uuid4().hex}, {"phase": "verify-saved"}, {"at": now - 3}, {"at": float("inf")}):
            with self.subTest(change=change), self.assertRaises(FixtureError):
                progress(good | change, manifest, "edit-draft", now - 1)

    async def test_driver_cannot_skip_operator_stage(self):
        manifest = {"stage_path": str(self.root / "stage.json"), "run_id": self.run, "not_before": time.time()}
        driver = Driver({}, manifest, object(), self.session, self.audio)
        with self.assertRaises(FixtureError):
            await driver.stage("save-draft")
        self.assertFalse((self.root / "stage.json").exists())

    def test_observer_hook_is_noop_without_explicit_private_opt_in(self):
        app = object()
        with patch.dict(os.environ, {"JARVIS_EVAL_REALTIME_WIRE_DIR": str(self.wire)}):
            self.assertIs(attach_preferences_trace(app, self.scope), app)
            save(self.wire / OPT_IN, {"format": FORMAT, "run_id": uuid4().hex}, new=True)
            with self.assertRaises(FixtureError):
                attach_preferences_trace(app, self.scope)

    async def test_observer_forwards_actual_http_and_websocket_messages_unchanged(self):
        committed = self.committed()
        outbound = []
        async def app(scope, receive, send):
            if scope["type"] == "http":
                request = await receive()
                self.assertEqual(decode(request["body"])["expected_revision"], 0)
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": json.dumps(committed).encode()})
            else:
                await send({"type": "websocket.accept"})
                event = dict(type="behavior_profile.committed", event_id=f"behavior-profile:{self.owner}:1",
                    owner_id=self.owner, session_id=self.session, audio_session_id=self.audio, revision=1,
                    behavior_instructions=committed["behavior_instructions"], committed_at=committed["updated_at"])
                await send({"type": "websocket.send", "text": json.dumps(event)})
        trace = PreferencesTrace(app, self.scope, self.wire)
        self.addCleanup(trace.close)
        async def receive():
            return {"type": "http.request", "body": json.dumps({"expected_revision": 0,
                "behavior_instructions": committed["behavior_instructions"], "authorization": "NEVER_RETAIN"}).encode()}
        async def send(message):
            outbound.append(message)
        await trace({"type": "http", "path": "/assistant/profile", "method": "PUT"}, receive, send)
        await trace({"type": "websocket", "path": "/realtime/audio/ws",
                     "query_string": ("session_id=" + self.session).encode()}, receive, send)
        captured = rows(self.wire, "preferences-chain-*.jsonl")
        self.assertEqual(len(outbound), 4)
        self.assertIsNotNone(commit_rows(captured, self.owner, 1, committed["behavior_instructions"], 0))
        self.assertIsNotNone(push_row(captured, committed, self.session, self.audio))
        self.assertNotIn("NEVER_RETAIN", trace.path.read_text())
        self.assertEqual(trace.path.stat().st_mode & 0o077, 0)

    async def test_failed_app_send_is_not_a_transport_delivery_receipt(self):
        async def app(scope, receive, send):
            await send({"type": "websocket.send", "text": json.dumps({"type": "behavior_profile.committed"})})
        async def send(_):
            raise ConnectionError("fixture")
        trace = PreferencesTrace(app, self.scope, self.wire)
        self.addCleanup(trace.close)
        with self.assertRaises(ConnectionError):
            await trace({"type": "websocket", "path": "/realtime/audio/ws"}, None, send)
        self.assertFalse(any(row["stage"] == "profile_app_send_returned" for row in rows(self.wire, "preferences-chain-*.jsonl")))

    async def test_commit_is_one_real_http_put_without_retry_and_no_callback(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=self.committed())
        async with httpx.AsyncClient(base_url="http://fixture", transport=httpx.MockTransport(handler)) as client:
            result = await put_once(client, self.owner, 0, "Synthetic preference")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/assistant/profile")
        self.assertEqual(decode(requests[0].content), {"expected_revision": 0, "behavior_instructions": "Synthetic preference"})
        self.assertEqual(result["revision"], 1)

    async def test_conflict_is_terminal_not_an_automatic_retry(self):
        attempts = []
        def handler(request):
            attempts.append(request)
            return httpx.Response(409, json={"error": {"code": "behavior_profile_revision_conflict"}})
        async with httpx.AsyncClient(base_url="http://fixture", transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(FixtureError):
                await put_once(client, self.owner, 0, "Synthetic preference")
        self.assertEqual(len(attempts), 1)

    def test_native_save_requires_actual_expected_revision_and_one_committed_effect(self):
        committed = self.committed(2, "My draft")
        row = {"stage": "profile_http_response_returned", "method": "PUT", "http_status": 200,
               "request": {"behavior_instructions": "My draft", "expected_revision": 1}, "response": committed}
        self.assertIsNone(commit_rows([], self.owner, 2, "My draft", 1))
        self.assertEqual(commit_rows([row], self.owner, 2, "My draft", 1), row)
        with self.assertRaises(FixtureError):
            commit_rows([row, row], self.owner, 2, "My draft", 1)
        with self.assertRaises(FixtureError):
            commit_rows([row], self.owner, 2, "My draft", 0)

    def test_wrong_owner_audio_session_and_commit_timestamp_are_not_valid_pushes(self):
        committed = self.committed()
        payload = dict(type="behavior_profile.committed", event_id=f"behavior-profile:{self.owner}:1",
            owner_id=self.owner, session_id=self.session, audio_session_id=self.audio, revision=1,
            behavior_instructions=committed["behavior_instructions"], committed_at=committed["updated_at"])
        for change in ({"owner_id": "other"}, {"audio_session_id": str(uuid4())},
                       {"committed_at": "2026-11-02T12:00:02+00:00"}):
            with self.subTest(change=change), self.assertRaises(FixtureError):
                push_row([{"stage": "profile_app_send_returned", "query_session_ids": [self.session],
                           "payload": payload | change}], committed, self.session, self.audio)

    def test_provider_write_and_ack_never_claim_model_instruction_consumption(self):
        committed = self.committed()
        anchor = self.provider_anchor(committed)
        instructions = "Base instructions\n" + committed["behavior_instructions"]
        records = [{"stage": "socket_connected", "connection_id": "one"},
            {"stage": "send_attempt", "sequence": 2, "connection_id": "one", "monotonic_s": 101,
             "utc": "2026-11-02T12:00:02+00:00",
             "payload": {"type": "session.update", "session": {"instructions": instructions}}}]
        self.assertFalse(provider_evidence(records, committed, app_send=anchor)["configuration_acknowledged"])
        records.append({"stage": "send_returned", "sequence": 3, "attempt_sequence": 2, "connection_id": "one"})
        written = provider_evidence(records, committed, app_send=anchor)
        self.assertEqual(written["send_returned_sequences"], [3])
        self.assertFalse(written["configuration_acknowledged"])
        records.append({"stage": "receive", "sequence": 4, "connection_id": "one",
            "payload": {"type": "session.updated", "session": {"instructions": instructions}}})
        acknowledged = provider_evidence(records, committed, app_send=anchor)
        self.assertTrue(acknowledged["configuration_acknowledged"])
        self.assertEqual(acknowledged["instruction_consumption"], "not_proven_no_model_response_requested")

    def test_stale_provider_update_or_other_connection_does_not_confirm_new_commit(self):
        committed = self.committed()
        records = [{"stage": "socket_connected", "connection_id": "one"},
            {"stage": "send_attempt", "sequence": 2, "connection_id": "one", "monotonic_s": 99,
             "utc": "2026-11-02T12:00:02+00:00",
             "payload": {"type": "session.update", "session": {"instructions": "Base\nSynthetic preference"}}},
            {"stage": "receive", "sequence": 3, "connection_id": "other",
             "payload": {"type": "session.updated", "session": {"instructions": "Base\nSynthetic preference"}}}]
        anchor = self.provider_anchor(committed)
        self.assertFalse(provider_evidence(records, committed, app_send=anchor)["configuration_acknowledged"])
        records[-1]["connection_id"] = "one"
        self.assertFalse(provider_evidence(records, committed, app_send=anchor)["configuration_acknowledged"])
        records[1]["monotonic_s"] = 101
        records[-1]["connection_id"] = "other"
        self.assertFalse(provider_evidence(records, committed, app_send=anchor)["configuration_acknowledged"])

    def provider_anchor(self, committed):
        return {"stage": "profile_app_send_returned", "sequence": 8, "monotonic_s": 100,
            "payload": {"owner_id": committed["owner_id"], "revision": committed["revision"],
                "behavior_instructions": committed["behavior_instructions"],
                "committed_at": committed["updated_at"]}}

    def test_database_clock_ahead_does_not_hide_real_provider_acknowledgment(self):
        committed = self.committed()
        instructions = "Base\n" + committed["behavior_instructions"]
        records = [{"stage": "socket_connected", "connection_id": "one"},
            {"stage": "send_attempt", "sequence": 2, "connection_id": "one", "monotonic_s": 100.001,
             "utc": "2026-11-02T12:00:00.984+00:00",
             "payload": {"type": "session.update", "session": {"instructions": instructions}}},
            {"stage": "send_returned", "sequence": 3, "attempt_sequence": 2, "connection_id": "one"},
            {"stage": "receive", "sequence": 4, "connection_id": "one",
             "payload": {"type": "session.updated", "session": {"instructions": instructions}}}]
        result = provider_evidence(records, committed, app_send=self.provider_anchor(committed))
        self.assertTrue(result["configuration_acknowledged"])
        self.assertEqual(result["send_attempt_sequences"], [2])
        self.assertEqual(result["ordering_boundary"], "same_process_app_profile_send")

    def test_missing_or_foreign_commit_anchor_fails_closed(self):
        committed = self.committed()
        anchor = self.provider_anchor(committed)
        for value in (None, anchor | {"stage": "profile_http_response_returned"},
                      anchor | {"payload": anchor["payload"] | {"owner_id": "other"}},
                      anchor | {"payload": anchor["payload"] | {"revision": 2}},
                      anchor | {"payload": anchor["payload"] | {"committed_at": "other"}}):
            with self.subTest(anchor=value), self.assertRaises(FixtureError):
                provider_evidence([], committed, app_send=value)

    def test_missing_or_invalid_monotonic_anchor_fails_closed(self):
        committed = self.committed()
        for value in (None, True, -1, float("nan"), float("inf"), "100"):
            with self.subTest(value=value), self.assertRaises(FixtureError):
                provider_evidence([], committed,
                    app_send=self.provider_anchor(committed) | {"monotonic_s": value})

    def test_initial_profile_must_be_empty_not_silently_cleared(self):
        with self.assertRaises(FixtureError):
            profile(self.committed(), self.owner, 0, "")

    def test_existing_attempt_or_artifact_cannot_be_overwritten(self):
        path = self.root / "controller-result.json"
        save(path, {"status": "uncertain"}, new=True)
        with self.assertRaises(FileExistsError):
            save(path, {"status": "retry"}, new=True)
        self.assertEqual(read_json(path)["status"], "uncertain")

    def created(self, **changes):
        payload = dict(type="session.created", session_id=self.session, audio_session_id=self.audio,
                       conversation_mode="lean_v1", audio_transport="webrtc")
        return dict(stage="session_created_app_send_returned", query_session_ids=[self.session],
                    payload=payload | changes)

    def test_actual_mode_requires_exact_session_audio_and_webrtc(self):
        good = self.created()
        self.assertEqual(created_session_row([good], self.session, self.audio), good)
        for changes in ({"session_id": str(uuid4())}, {"audio_session_id": str(uuid4())}):
            with self.subTest(changes=changes):
                self.assertIsNone(created_session_row([self.created(**changes)], self.session, self.audio))
        for changes in ({"conversation_mode": "legacy"}, {"conversation_mode": None},
                        {"audio_transport": "websocket"}):
            with self.subTest(changes=changes), self.assertRaises(FixtureError):
                created_session_row([self.created(**changes)], self.session, self.audio)
        self.assertIsNone(created_session_row([dict(stage="configuration", required_conversation_mode="lean_v1")],
                                              self.session, self.audio))
        self.assertIsNone(created_session_row([good | {"query_session_ids": [str(uuid4())]}], self.session, self.audio))

    async def test_session_created_observer_preserves_message_and_filters_unrelated_fields(self):
        event = self.created()["payload"] | {"sdp": "NEVER_RETAIN"}
        message = {"type": "websocket.send", "text": json.dumps(event)}
        sent = []
        async def app(scope, receive, send):
            await send(message)
        async def send(value):
            sent.append(value)
        trace = PreferencesTrace(app, self.scope, self.wire)
        self.addCleanup(trace.close)
        await trace({"type": "websocket", "path": "/realtime/audio/ws",
                     "query_string": ("session_id=" + self.session).encode()}, None, send)
        self.assertIs(sent[0], message)
        actual = created_session_row(rows(self.wire, "preferences-chain-*.jsonl"), self.session, self.audio)
        self.assertEqual(actual["payload"], self.created()["payload"])
        self.assertNotIn("NEVER_RETAIN", trace.path.read_text())

    async def test_failed_session_created_send_has_no_readiness_receipt(self):
        async def app(scope, receive, send):
            await send({"type": "websocket.send", "text": json.dumps(self.created()["payload"])})
        async def send(_):
            raise ConnectionError("synthetic transport failure")
        trace = PreferencesTrace(app, self.scope, self.wire)
        self.addCleanup(trace.close)
        with self.assertRaises(ConnectionError):
            await trace({"type": "websocket", "path": "/realtime/audio/ws",
                         "query_string": ("session_id=" + self.session).encode()}, None, send)
        self.assertIsNone(created_session_row(rows(self.wire, "preferences-chain-*.jsonl"), self.session, self.audio))

    async def test_wrong_actual_mode_stops_before_any_ui_stage_or_put(self):
        methods, phases = [], []
        def handler(request):
            methods.append(request.method)
            value = (dict(run_id=self.run, owner_id=self.owner, isolation_enforced=True)
                     if request.url.path.endswith("/scope") else
                     dict(owner_id=self.owner, revision=0, behavior_instructions=""))
            return httpx.Response(200, json=value)
        async with httpx.AsyncClient(base_url="http://fixture.invalid", transport=httpx.MockTransport(handler)) as client:
            driver = Driver(dict(run_id=self.run, owner_id=self.owner),
                            {"stage_path": str(self.root / "stage.json")}, client, self.session, self.audio)
            driver.records = lambda: [dict(stage="audio_websocket_accepted", query_session_ids=[self.session]),
                                      self.created(conversation_mode="legacy")]
            async def stage(phase):
                phases.append(phase)
            driver.stage = stage
            with self.assertRaisesRegex(FixtureError, "actual_conversation_mode_mismatch"):
                await driver.run()
        self.assertEqual(methods, ["GET", "GET"])
        self.assertEqual(phases, [])
        self.assertIsNone(driver.session_created)

    async def test_actual_lean_receipt_is_retained_before_first_ui_stage(self):
        def handler(request):
            value = (dict(run_id=self.run, owner_id=self.owner, isolation_enforced=True)
                     if request.url.path.endswith("/scope") else
                     dict(owner_id=self.owner, revision=0, behavior_instructions=""))
            return httpx.Response(200, json=value)
        async with httpx.AsyncClient(base_url="http://fixture.invalid", transport=httpx.MockTransport(handler)) as client:
            driver = Driver(dict(run_id=self.run, owner_id=self.owner),
                            {"stage_path": str(self.root / "stage.json")}, client, self.session, self.audio)
            receipt = self.created()
            driver.records = lambda: [dict(stage="audio_websocket_accepted", query_session_ids=[self.session]), receipt]
            async def stage(phase):
                self.assertEqual(phase, "open-preferences")
                self.assertIs(driver.session_created, receipt)
                raise FixtureError("stop_before_ui")
            driver.stage = stage
            with self.assertRaisesRegex(FixtureError, "stop_before_ui"):
                await driver.run()
