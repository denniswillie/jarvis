"""Source-only preparation for the existing real native Preferences journey.

No app/server/provider launcher, callback injection, speech, SQL, or second
runtime lives here. Core prepares/serves the existing native frontend host and
owns the native session. Phone's unregistered attach-only UI operator consumes
the manifest produced by prepare().

ONE REQUIRED CORE HARNESS HOOK, not installed by this file:
  In frontend_host_qualification.child_server, after AdmissionTrace(...):
      trace = attach_preferences_trace(trace, scope)
  Import this function there. It is a strict no-op without this fixture's
  opt-in file in the already-forwarded JARVIS_EVAL_REALTIME_WIRE_DIR.

The normal host's default auto_cleanup must remain enabled. The controller
requests its existing owned stop file and awaits its actual cleanup receipt.
UI progress proves controls only. Provider send_returned proves a socket write;
matching session.updated proves acknowledged configuration, not that a future
model response follows it. Actual URLSession receive/AudioVM/verified-owner
evidence and a real XCTest pass with zero skips remain separate native evidence.
"""
from __future__ import annotations

import argparse
import asyncio
import atexit
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

import httpx

from evals.responsive_voice.frontend_host import ReplayHostScope


FORMAT = "jarvis.native-preferences.fixture.v1"
OPT_IN = "preferences-chain-opt-in.json"
PHASES = ("open-preferences", "edit-draft", "review-update", "save-draft",
          "verify-saved", "verify-clean-push", "finished-awaiting-review")
JOURNEY_SECONDS = 180
CLEANUP_SECONDS = 120
MAX_JSON = 256 * 1024
MAX_TRACE = 64 * 1024 * 1024
REQUIRED_CONVERSATION_MODE = "lean_v1"


class FixtureError(Exception):
    pass


def require(condition, code):
    if not condition:
        raise FixtureError(code)


def stamp():
    return datetime.now(UTC).isoformat()


def instant(value):
    require(isinstance(value, str), "timestamp_missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise FixtureError("timestamp_invalid") from None
    require(parsed.tzinfo is not None, "timestamp_without_offset")
    return parsed.timestamp()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    def constant(_):
        raise FixtureError("nonfinite_json")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def read_json(path):
    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size <= MAX_JSON, "unsafe_or_large_json")
    return decode(path.read_bytes())


def private_directory(path, *, create=False):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink(), "private_absolute_directory_required")
    if create:
        path.mkdir(mode=0o700)
    require(path.is_dir() and path.stat().st_mode & 0o077 == 0, "directory_not_private")
    return path


def save(path, value, *, new=False):
    path = Path(path)
    private_directory(path.parent)
    require(not path.is_symlink(), "symlink_output_forbidden")
    target = path if new else path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(value, stream, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        stream.flush()
    if not new:
        os.replace(target, path)


def resources(path, *, serving=False):
    path = Path(path).absolute()
    value = read_json(path)
    require(value.get("format") == "jarvis.frontend_host.resources.v1", "wrong_host_manifest")
    run = UUID(value["run_id"]).hex
    require(value["run_id"] == run and path.name == f"frontend-host-resources-{run}.json", "wrong_fixture_run")
    for key, name in (("scope_file", f"frontend-host-scope-{run}.json"),
                      ("fixture_manifest", f"second-brain-{run}.json"),
                      ("control_file", f"frontend-host-control-{run}.json"),
                      ("bearer_token_file", f"frontend-host-bearer-{run}.token")):
        require(Path(value[key]).absolute() == path.parent / name, "foreign_owned_path")
    scope = ReplayHostScope.model_validate(read_json(value["scope_file"]))
    require(all(str(getattr(scope, key)) == str(value[key]) for key in
                ("run_id", "owner_id", "database_schema", "area_id", "memory_bank_id")), "scope_mismatch")
    require(scope.expires_at.timestamp() > time.time(), "fixture_expired")
    options = value.get("native_options", {})
    require(value.get("capability_profile") == "native_realtime_media"
            and options.get("voice_route") == "realtime" and options.get("workload") == "none",
            "idle_native_realtime_host_required")
    if serving:
        require(value.get("status") == "serving" and value.get("normal_lifespan_qualified") is True,
                "normal_host_not_serving")
        url = urlsplit(value.get("backend_base_url", ""))
        require(url.scheme == "http" and url.hostname == "127.0.0.1" and url.port
                and not url.username and not url.password and url.path in {"", "/"}, "nonlocal_host")
    return value, scope


def prepare(resource_manifest, directory, wire_directory):
    value, scope = resources(resource_manifest)
    root = private_directory(Path(directory).absolute(), create=True)
    wire = private_directory(Path(wire_directory).absolute())
    run = str(UUID(scope.run_id)).upper()
    phone = private_directory(root / ("phone-preferences-" + run), create=True)
    marker = uuid4().hex[:8]
    manifest = {"schema_version": 1, "run_id": run, "stage_path": str(phone / "stage.json"),
        "not_before": time.time(), "draft": f"Use concise numbered steps. Preference fixture {marker}.",
        "pushed_while_dirty": f"Use brief paragraphs and metric units. Preference fixture {marker}.",
        "pushed_while_clean": f"Use concise sentences and Celsius. Preference fixture {marker}."}
    save(phone / "manifest.json", manifest, new=True)
    opt_in = {"format": FORMAT, "run_id": scope.run_id, "owner_id": scope.owner_id,
              "expires_at": scope.expires_at.isoformat()}
    save(wire / OPT_IN, opt_in, new=True)
    plan = {"format": FORMAT, "resource_manifest": str(Path(resource_manifest).absolute()),
        "run_id": scope.run_id, "owner_id": scope.owner_id, "ui_manifest": str(phone / "manifest.json"),
        "wire_directory": str(wire), "created_at": stamp(), "journey_seconds": JOURNEY_SECONDS,
        "cleanup_seconds": CLEANUP_SECONDS, "phases": list(PHASES),
        "required_host_hook": "trace = attach_preferences_trace(trace, scope)",
        "native_receipt_limits": "UI progress is not URLSession receipt, AudioVM/owner gate proof, or an XCTest verdict."}
    save(root / "controller-plan.json", plan, new=True)
    return root / "controller-plan.json"


class PreferencesTrace:
    """Passive observer around the SAME AdmissionTrace/app; messages unchanged."""
    def __init__(self, app, scope, root):
        self.app, self.scope = app, scope
        root = private_directory(Path(root))
        self.path = root / ("preferences-chain-" + uuid4().hex + ".jsonl")
        self.stream = os.fdopen(os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")
        self.sequence, self.bytes, self.incomplete = 0, 0, False
        self.record("capture_started", boundary="actual_asgi_http_and_app_websocket_not_phone_receive")
        atexit.register(self.close)

    def __getattr__(self, name):
        return getattr(self.app, name)

    def record(self, stage, **fields):
        if self.incomplete or self.stream.closed:
            return
        self.sequence += 1
        row = dict(stage=stage, sequence=self.sequence, utc=stamp(), monotonic_s=time.monotonic(),
                   run_id=self.scope.run_id, owner_id=self.scope.owner_id, **fields)
        encoded = json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n"
        if self.bytes + len(encoded.encode()) > 8 * 1024 * 1024:
            encoded = json.dumps(dict(stage="capture_incomplete", reason="byte_limit", sequence=self.sequence)) + "\n"
            self.incomplete = True
        try:
            self.stream.write(encoded)
            self.stream.flush()
            self.bytes += len(encoded.encode())
        except OSError:
            self.incomplete = True

    def close(self):
        if not self.stream.closed:
            self.record("capture_closed", complete=not self.incomplete)
            self.stream.close()

    async def __call__(self, scope, receive, send):
        profile = scope["type"] == "http" and scope.get("path") == "/assistant/profile"
        audio = scope["type"] == "websocket" and scope.get("path") == "/realtime/audio/ws"
        identity, request_body, response_body = uuid4().hex, bytearray(), bytearray()
        status = None
        session = parse_qs(scope.get("query_string", b"").decode("ascii", errors="replace")).get("session_id", [])

        def body(raw):
            try:
                value = decode(raw)
                if not isinstance(value, dict):
                    return {"invalid_body": True}
                # No headers, credentials, SDP or unrelated request fields.
                return {key: value[key] for key in ("owner_id", "revision", "behavior_instructions",
                    "expected_revision", "created_at", "updated_at") if key in value}
            except (ValueError, FixtureError, UnicodeError):
                return {"invalid_body": True}

        async def observed_receive():
            message = await receive()
            if profile and message["type"] == "http.request":
                request_body.extend(message.get("body", b""))
                if len(request_body) > MAX_JSON:
                    request_body[:] = b"null"
                    self.record("capture_incomplete", reason="http_request_limit")
            return message

        async def observed_send(message):
            nonlocal status
            if profile and message["type"] == "http.response.start":
                status = message["status"]
            if profile and message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))
            try:
                await send(message)
            except BaseException:
                self.record("app_send_failed", request_id=identity)
                raise
            if profile and message["type"] == "http.response.body" and not message.get("more_body", False):
                self.record("profile_http_response_returned", request_id=identity, method=scope["method"],
                    http_status=status, request=body(request_body),
                    response=body(response_body) if len(response_body) <= MAX_JSON else {"invalid_body": True})
            if audio and message["type"] == "websocket.accept":
                self.record("audio_websocket_accepted", query_session_ids=session)
            if audio and message["type"] == "websocket.send" and isinstance(message.get("text"), str):
                try:
                    value = decode(message["text"])
                except (ValueError, FixtureError):
                    return
                if isinstance(value, dict) and value.get("type") == "behavior_profile.committed":
                    self.record("profile_app_send_returned", query_session_ids=session,
                        payload={key: value[key] for key in ("type", "event_id", "owner_id", "session_id",
                            "audio_session_id", "revision", "behavior_instructions", "committed_at") if key in value})
                if isinstance(value, dict) and value.get("type") == "session.created":
                    self.record("session_created_app_send_returned", query_session_ids=session,
                        payload={key: value[key] for key in ("type", "session_id", "audio_session_id",
                            "conversation_mode", "audio_transport", "client_kind", "server_time") if key in value})
        await self.app(scope, observed_receive if profile else receive, observed_send)


def attach_preferences_trace(app, scope):
    root = os.environ.get("JARVIS_EVAL_REALTIME_WIRE_DIR")
    if not root or not (Path(root) / OPT_IN).exists():
        return app
    value = read_json(Path(root) / OPT_IN)
    require(value == {"format": FORMAT, "run_id": scope.run_id, "owner_id": scope.owner_id,
                      "expires_at": scope.expires_at.isoformat()}, "preferences_trace_scope_mismatch")
    return PreferencesTrace(app, scope, root)


def rows(root, pattern):
    result, total = [], 0
    paths = sorted(Path(root).glob(pattern))
    require(len(paths) <= 8, "too_many_trace_files")
    for path in paths:
        require(not path.is_symlink(), "trace_symlink")
        total += path.stat().st_size
        require(total <= MAX_TRACE, "trace_read_limit")
        raw = path.read_bytes()
        # A live writer may be halfway through its last line. Never call that
        # partial line an observation; completed runs retain capture_closed.
        for number, line in enumerate(raw.splitlines(keepends=True), 1):
            if not line.endswith(b"\n"):
                continue
            row = decode(line)
            require(isinstance(row, dict), "invalid_trace_row")
            result.append(dict(row, source=str(path), source_line=number))
    require(not any(row.get("stage") == "capture_incomplete" for row in result), "trace_incomplete")
    return result


def created_session_row(records, session_id, audio_session_id):
    """Actual app-route evidence, never a launch flag or requested mode.

    A matching legacy/unknown mode fails before any fixture mutation. A foreign
    session cannot satisfy readiness. This receipt is not proof of phone receipt
    or of the model following subsequent preference instructions.
    """
    matched = None
    for row in records:
        payload = row.get("payload", {})
        if (row.get("stage") != "session_created_app_send_returned"
                or row.get("query_session_ids") != [session_id]
                or not isinstance(payload, dict) or payload.get("type") != "session.created"
                or payload.get("session_id") != session_id or payload.get("audio_session_id") != audio_session_id):
            continue
        require(payload.get("conversation_mode") == REQUIRED_CONVERSATION_MODE, "actual_conversation_mode_mismatch")
        require(payload.get("audio_transport") == "webrtc", "actual_audio_transport_mismatch")
        matched = row
    return matched


def profile(value, owner, revision, text):
    require(isinstance(value, dict) and value.get("owner_id") == owner
            and type(value.get("revision")) is int and value["revision"] == revision
            and value.get("behavior_instructions") == text, "authoritative_profile_mismatch")
    if revision:
        instant(value.get("updated_at"))
    return value


def progress(value, manifest, phase, posted_at):
    require(value.get("run_id") == manifest["run_id"] and value.get("phase") == phase,
            "ui_progress_scope_mismatch")
    at = value.get("at")
    require(type(at) in (int, float) and math.isfinite(at)
            and max(manifest["not_before"], posted_at) <= at <= time.time() + 2, "stale_ui_progress")
    return value


def commit_rows(records, owner, revision, text, expected_revision):
    matched = [row for row in records if row.get("stage") == "profile_http_response_returned"
        and row.get("method") == "PUT" and row.get("request", {}).get("behavior_instructions") == text]
    require(len(matched) <= 1, "repeated_profile_put_no_automatic_retry")
    if not matched:
        return None
    row = matched[0]
    require(row.get("http_status") == 200 and row["request"].get("expected_revision") == expected_revision,
            "native_save_failed_or_wrong_cas")
    profile(row["response"], owner, revision, text)
    return row


def push_row(records, committed, session_id, audio_session_id):
    for row in records:
        event = row.get("payload", {})
        if row.get("stage") != "profile_app_send_returned" or event.get("revision") != committed["revision"]:
            continue
        require(event.get("owner_id") == committed["owner_id"] and event.get("session_id") == session_id
                and event.get("audio_session_id") == audio_session_id and row.get("query_session_ids") == [session_id]
                and event.get("behavior_instructions") == committed["behavior_instructions"]
                and event.get("event_id") == f"behavior-profile:{committed['owner_id']}:{committed['revision']}"
                and instant(event.get("committed_at")) == instant(committed["updated_at"]), "commit_push_identity_mismatch")
        return row
    return None


def provider_evidence(records, committed, *, app_send):
    # The database's updated_at and the host's UTC clock are not synchronized.
    # This passive app-send receipt and the provider wire use the SAME server
    # process's monotonic clock. Keep the committed identity/content checks;
    # never accept an earlier matching configuration merely because text matches.
    require(isinstance(app_send, dict)
            and app_send.get("stage") == "profile_app_send_returned",
            "provider_ordering_anchor_missing")
    payload = app_send.get("payload", {})
    require(payload.get("owner_id") == committed["owner_id"]
            and payload.get("revision") == committed["revision"]
            and payload.get("behavior_instructions") == committed["behavior_instructions"]
            and payload.get("committed_at") == committed["updated_at"],
            "provider_ordering_anchor_mismatch")
    def monotonic(row):
        value = row.get("monotonic_s")
        require(type(value) in {int, float} and math.isfinite(value) and value >= 0,
                "provider_monotonic_time_missing")
        return value
    after = monotonic(app_send)
    connected = {row.get("connection_id") for row in records if row.get("stage") == "socket_connected"}
    require(len(connected) == 1 and None not in connected, "single_provider_connection_required")
    connection = next(iter(connected))
    attempts = [row for row in records if row.get("stage") == "send_attempt"
        and row.get("connection_id") == connection and row.get("payload", {}).get("type") == "session.update"
        and isinstance(row["payload"].get("session", {}).get("instructions"), str)
        and row["payload"]["session"]["instructions"].endswith("\n" + committed["behavior_instructions"])
        and monotonic(row) >= after]
    returned, acknowledged = [], []
    for attempt in attempts:
        for row in records:
            if row.get("connection_id") != connection:
                continue
            if row.get("stage") == "send_returned" and row.get("attempt_sequence") == attempt["sequence"]:
                returned.append(row["sequence"])
            if (row.get("stage") == "receive" and row.get("sequence", 0) > attempt["sequence"]
                    and row.get("payload", {}).get("type") == "session.updated"
                    and row["payload"].get("session", {}).get("instructions") == attempt["payload"]["session"]["instructions"]):
                acknowledged.append(row["sequence"])
    return {"revision": committed["revision"], "connection_id": connection,
        "ordering_boundary": "same_process_app_profile_send",
        "ordering_anchor_sequence": app_send.get("sequence"),
        "send_attempt_sequences": [row["sequence"] for row in attempts],
        "send_returned_sequences": returned, "session_updated_sequences": acknowledged,
        "configuration_acknowledged": bool(acknowledged),
        "instruction_consumption": "not_proven_no_model_response_requested"}


async def put_once(client, owner, expected_revision, text):
    response = await client.put("/assistant/profile", json={"expected_revision": expected_revision,
                                                            "behavior_instructions": text})
    require(response.status_code == 200, "profile_put_not_confirmed_do_not_retry")
    return profile(response.json(), owner, expected_revision + 1, text)


class Driver:
    def __init__(self, plan, manifest, client, session_id, audio_session_id):
        self.plan, self.manifest, self.client = plan, manifest, client
        self.session_id, self.audio_session_id = str(UUID(session_id)), str(UUID(audio_session_id))
        self.phone = Path(manifest["stage_path"]).parent
        self.deadline = time.monotonic() + JOURNEY_SECONDS
        self.events, self.commits, self.pushes = [], [], []
        self.session_created = None

    def records(self):
        value = rows(self.plan["wire_directory"], "preferences-chain-*.jsonl")
        require(all(row.get("run_id") == self.plan["run_id"] and row.get("owner_id") == self.plan["owner_id"]
                    for row in value), "foreign_receipt_trace")
        return value

    async def wait(self, read, *, seconds=20):
        end = min(self.deadline, time.monotonic() + seconds)
        while time.monotonic() < end:
            result = read()
            if result is not None:
                return result
            await asyncio.sleep(.1)
        raise FixtureError("bounded_observation_timeout")

    async def stage(self, phase):
        require(phase == PHASES[len(self.events)], "stage_order_violation")
        posted = time.time()
        save(Path(self.manifest["stage_path"]), {"run_id": self.manifest["run_id"], "phase": phase, "at": posted})
        if phase != PHASES[-1]:
            def ready():
                path = self.phone / "ui-progress.json"
                if not path.exists():
                    return None
                value = read_json(path)
                require(value.get("run_id") == self.manifest["run_id"], "foreign_ui_progress")
                if value.get("phase") != phase:
                    require(value.get("phase") in PHASES[:len(self.events)], "future_ui_progress")
                    return None
                return progress(value, self.manifest, phase, posted)
            receipt = await self.wait(ready)
        else:
            def done():
                path = self.phone / "native-ui-driver.json"
                if not path.exists():
                    return None
                value = read_json(path)
                require(value.get("run_id") == self.manifest["run_id"] and value.get("completed_ui_stages") is True,
                        "ui_stages_not_completed")
                observed = [event.get("stage") for event in value.get("events", []) if event.get("action") == "observed"]
                require(observed == list(PHASES), "ui_stage_evidence_missing")
                return value
            receipt = await self.wait(done)
        self.events.append({"phase": phase, "posted_at": posted, "receipt": receipt})

    async def committed(self, value, expected_revision):
        recorded = await self.wait(lambda: commit_rows(self.records(), self.plan["owner_id"],
            value["revision"], value["behavior_instructions"], expected_revision))
        profile(recorded["response"], self.plan["owner_id"], value["revision"], value["behavior_instructions"])
        require(recorded["response"]["updated_at"] == value["updated_at"], "commit_timestamp_mismatch")
        pushed = await self.wait(lambda: push_row(self.records(), value, self.session_id, self.audio_session_id))
        self.commits.append(value)
        self.pushes.append(pushed)

    async def run(self):
        public = await self.client.get(f"/eval/replay/{self.plan['run_id']}/scope")
        require(public.status_code == 200 and public.json().get("isolation_enforced") is True
                and public.json().get("run_id") == self.plan["run_id"]
                and public.json().get("owner_id") == self.plan["owner_id"], "real_host_scope_unconfirmed")
        initial = await self.client.get("/assistant/profile")
        require(initial.status_code == 200, "initial_profile_read_failed")
        profile(initial.json(), self.plan["owner_id"], 0, "")
        # Both observations come from the actual route, not requested settings.
        # Retain one five-second readiness budget and perform no writes first.
        def ready():
            records = self.records()
            if not any(row.get("stage") == "audio_websocket_accepted"
                       and row.get("query_session_ids") == [self.session_id] for row in records):
                return None
            return created_session_row(records, self.session_id, self.audio_session_id)
        self.session_created = await self.wait(ready, seconds=5)
        await self.stage("open-preferences")
        await self.stage("edit-draft")
        dirty = await put_once(self.client, self.plan["owner_id"], 0, self.manifest["pushed_while_dirty"])
        await self.committed(dirty, 0)
        await self.stage("review-update")
        await self.stage("save-draft")
        saved = await self.wait(lambda: commit_rows(self.records(), self.plan["owner_id"], 2, self.manifest["draft"], 1))
        current = await self.client.get("/assistant/profile")
        require(current.status_code == 200, "native_save_readback_failed")
        profile(current.json(), self.plan["owner_id"], 2, self.manifest["draft"])
        require(current.json()["updated_at"] == saved["response"]["updated_at"], "native_save_receipt_mismatch")
        await self.committed(current.json(), 1)
        await self.stage("verify-saved")
        clean = await put_once(self.client, self.plan["owner_id"], 2, self.manifest["pushed_while_clean"])
        await self.committed(clean, 2)
        await self.stage("verify-clean-push")
        await self.stage("finished-awaiting-review")


async def stop_existing_host(plan):
    value = read_json(plan["resource_manifest"])
    require(value["run_id"] == plan["run_id"] and value["owner_id"] == plan["owner_id"], "cleanup_scope_changed")
    command = {"run_id": plan["run_id"], "action": "stop"}
    path = Path(value["control_file"])
    expected = Path(plan["resource_manifest"]).parent / f"frontend-host-control-{plan['run_id']}.json"
    require(path == expected, "foreign_cleanup_control")
    if path.exists():
        require(read_json(path) == command, "different_host_control")
    else:
        save(path, command, new=True)
    end = time.monotonic() + CLEANUP_SECONDS
    while time.monotonic() < end:
        value = read_json(plan["resource_manifest"])
        if value.get("status") in {"cleaned", "cleanup_incomplete"}:
            require(value["status"] == "cleaned", "existing_host_cleanup_incomplete")
            receipt = read_json(value["cleanup_report"])
            require(receipt.get("status") == "passed" and receipt.get("run_id") == plan["run_id"], "cleanup_receipt_unconfirmed")
            return {"status": "passed", "receipt": value["cleanup_report"], "cleanup": receipt["cleanup"]}
        await asyncio.sleep(.2)
    raise FixtureError("existing_host_cleanup_not_confirmed")


async def drive(plan_path, session_id, audio_session_id):
    plan_path = Path(plan_path).absolute()
    plan = read_json(plan_path)
    require(plan.get("format") == FORMAT and plan.get("phases") == list(PHASES), "wrong_controller_plan")
    value, scope = resources(plan["resource_manifest"], serving=True)
    require((scope.run_id, scope.owner_id) == (plan["run_id"], plan["owner_id"]), "controller_scope_changed")
    manifest = read_json(plan["ui_manifest"])
    require(manifest["run_id"] == str(UUID(scope.run_id)).upper(), "ui_run_mismatch")
    result_path = plan_path.parent / "controller-result.json"
    result = {"format": FORMAT, "run_id": scope.run_id, "owner_id": scope.owner_id,
        "status": "running", "started_at": stamp(), "session_id": session_id, "audio_session_id": audio_session_id,
        "native_transport_receive": "requires_separate_actual_URLSession_AudioVM_verified_owner_receipts",
        "xctest_verdict": "requires_actual_pass_zero_skips_not_ui_json_alone", "cleanup": {}}
    save(result_path, result, new=True)  # Burns the one attempt even on uncertain PUT.
    driver = None
    try:
        token_path = Path(value["bearer_token_file"])
        require(not token_path.is_symlink() and token_path.stat().st_mode & 0o077 == 0
                and token_path.stat().st_size <= 4096, "unsafe_token_file")
        token = token_path.read_text().strip()
        require(bool(token) and "\n" not in token, "invalid_fixture_token")
        async with httpx.AsyncClient(base_url=value["backend_base_url"], timeout=10, follow_redirects=False,
                trust_env=False, headers={"Authorization": "Bearer " + token, "X-Jarvis-Evaluation-Run": scope.run_id}) as client:
            driver = Driver(plan, manifest, client, session_id, audio_session_id)
            async with asyncio.timeout(JOURNEY_SECONDS):
                await driver.run()
        result["status"] = "collected_not_native_or_model_qualified"
    except Exception as error:
        result.update(status="failed", failure_code=str(error) if isinstance(error, FixtureError) else type(error).__name__)
    finally:
        if driver is not None:
            result.update(stages=driver.events, commits=driver.commits, app_socket_sends=driver.pushes,
                          actual_session_created=driver.session_created)
        save(result_path, result)
        try:
            result["cleanup"] = await stop_existing_host(plan)
        except Exception as error:
            result.update(status="failed", cleanup={"status": "unconfirmed", "error_type": type(error).__name__})
        try:
            wire = rows(plan["wire_directory"], "wire-*.jsonl")
            app_sends = {row["payload"]["revision"]: row for row in result.get("app_socket_sends", [])}
            result["provider_instruction_evidence"] = [provider_evidence(wire, item,
                app_send=app_sends.get(item["revision"])) for item in result.get("commits", [])]
            result["provider_trace_closed"] = any(row.get("stage") == "capture_closed" and row.get("complete") is True for row in wire)
        except Exception as error:
            result["provider_instruction_evidence"] = {"state": "unconfirmed", "error_type": type(error).__name__}
        result["finished_at"] = stamp()
        save(result_path, result)
    return result_path, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("prepare")
    stage.add_argument("--resource-manifest", type=Path, required=True)
    stage.add_argument("--directory", type=Path, required=True)
    stage.add_argument("--wire-directory", type=Path, required=True)
    run = sub.add_parser("drive")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--session-id", required=True)
    run.add_argument("--audio-session-id", required=True)
    run.add_argument("--execute-once", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            path = prepare(args.resource_manifest, args.directory, args.wire_directory)
            print(json.dumps({"status": "prepared_source_contract", "plan": str(path)}))
            return 0
        require(args.execute_once, "explicit_execute_once_required")
        path, result = asyncio.run(drive(args.plan, args.session_id, args.audio_session_id))
        print(json.dumps({"status": result["status"], "report": str(path)}))
        return int(result["status"] == "failed")
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "code": str(error) if isinstance(error, FixtureError) else "fixture_support_error"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
