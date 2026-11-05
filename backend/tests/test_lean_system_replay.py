"""Offline only: no speech synthesis, browser, SQL, provider or service calls."""
import ast
import asyncio
import base64
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from evals.responsive_voice import lean_system_replay as replay


class LeanSystemReplayTests(unittest.TestCase):
    def test_control_events_are_one_complete_json_value_per_line(self):
        events = [{"event": {"type": "first", "text": "line one\nline two"}},
                  {"event": {"type": "second", "nested": [1, {"ok": True}]}}]
        wire = b"".join(replay.encode_event(value) for value in events)
        self.assertTrue(wire.endswith(b"\n"))
        lines = wire.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual([json.loads(line) for line in lines], events)

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.contract = self.root / "contract.md"
        self.contract.write_text("## Coordinator first-case proposal\n" + "\n".join(
            f'|{i}|\u201cSpoken input {i}.\u201d|External oracle {i}.|' for i in range(1, 5)))

    def fake_speech(self, output, utterances):
        output.mkdir(mode=0o700)
        clips = []
        for identity, text in utterances:
            path = output / (identity + ".wav")
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(replay.RATE)
                audio.writeframes(b"\x00\x00\xff\x7f\x00\x00")
            path.chmod(0o600)
            clips.append({"id": identity, "text": text, "relative_path": path.name,
                "sha256": replay.digest(path.read_bytes()), "sample_count": 3,
                "first_active_sample": 1, "last_active_sample_exclusive": 2})
        replay.save(output / "speech.json", {"format": "jarvis.speech_assets.v1", "clips": clips})

    def prepared(self):
        root = self.root / "run"
        replay.prepare(root, model="offline-model", voice="offline-voice", contract=self.contract,
                       speech_builder=self.fake_speech)
        return root

    def test_actual_coordinator_has_four_frozen_inputs_and_oracles(self):
        rows = replay.read_contract()["utterances"]
        self.assertEqual([x["id"] for x in rows], list(replay.IDS))
        self.assertEqual(rows[1]["text"], "In that note, change the reps for set one to eight.")
        self.assertIn("reps unknown", rows[0]["text"])
        self.assertIn("before that correction", rows[3]["text"])
        self.assertTrue(all(x["oracle"] for x in rows))

    def test_missing_or_duplicate_utterance_is_rejected(self):
        for text in ("## Coordinator first-case proposal\n", self.contract.read_text() + '\n|4|\u201cExtra\u201d|Oracle|'):
            self.contract.write_text(text)
            with self.assertRaises(replay.ReplayError):
                replay.read_contract(self.contract)

    def test_preparation_freezes_private_assets_before_execution(self):
        root = self.prepared()
        manifest, clips = replay.validate_bundle(root)
        self.assertEqual(len(clips), 4)
        self.assertFalse((root / "execution-attempt.json").exists())
        self.assertEqual(manifest["limits"]["system_seconds"], 120)
        self.assertEqual((root / "manifest.json").stat().st_mode & 0o777, 0o600)
        self.assertEqual(root.stat().st_mode & 0o777, 0o700)

    def test_preparation_never_reuses_output(self):
        root = self.prepared()
        with self.assertRaises(FileExistsError):
            replay.prepare(root, model="offline", voice="offline", contract=self.contract,
                           speech_builder=lambda *_: self.fail("must not synthesize"))

    def test_changed_audio_fails_before_claim(self):
        root = self.prepared()
        with (root / "speech/create.wav").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(replay.ReplayError, "speech_hash_mismatch"):
            replay.validate_bundle(root)

    def test_changed_coordinator_contract_fails(self):
        root = self.prepared()
        self.contract.write_text(self.contract.read_text() + "\nChanged oracle revision")
        with self.assertRaisesRegex(replay.ReplayError, "coordinator_contract_changed"):
            replay.validate_bundle(root)

    def test_budget_cannot_be_widened_in_manifest(self):
        root = self.prepared()
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["limits"]["system_seconds"] = 121
        replay.save(root / "manifest.json", manifest)
        with self.assertRaisesRegex(replay.ReplayError, "frozen_run_contract_changed"):
            replay.validate_bundle(root)

    def test_attempt_is_consumed_even_without_a_success(self):
        root = self.prepared()
        replay.claim(root)
        with self.assertRaises(FileExistsError):
            replay.claim(root)

    def test_explicit_execute_is_required_before_any_setup(self):
        with self.assertRaisesRegex(replay.ReplayError, "execute_once_required"):
            asyncio.run(replay.execute(self.root / "absent", execute_once=False,
                hindsight_url="unused", api_key_file=self.root / "unused"))

    def test_only_authenticated_loopback_native_route_is_constructed(self):
        url = replay.audio_url("http://127.0.0.1:12345", "00000000-0000-0000-0000-000000000001")
        self.assertIn("/realtime/audio/ws?", url)
        self.assertIn("conversation_mode=lean_v1", url)
        self.assertIn("audio_transport=webrtc", url)
        self.assertNotIn("voice_route=", url)
        for origin in ("http://example.com", "http://user@127.0.0.1:123", "https://127.0.0.1:123"):
            with self.assertRaises(replay.ReplayError):
                replay.audio_url(origin, "00000000-0000-0000-0000-000000000001")

    def complete_state(self):
        return {"audio_frame": 15000, "turns": [{"input_ended": True, "last_input_sample": 100,
            "input_item_ids": ["actual-input"], "response_ids": ["actual-response"], "playback_generation": 1,
            "first_output_sample": 200, "last_output_sample": 2000}],
            "responses": {"actual-response": {"status": "completed", "buffer_started": True, "buffer_stopped": True,
                "created_frame": 150, "output_observed": True, "has_audio_output": True,
                "creation_order": 1, "playback_generation": 1,
                "has_function_calls": False, "function_call_ids": []}},
            "playback": {"generation": 1, "latest_response_order": 1, "interrupted": False, "uncertain": False,
                "last_start": {"generation": 1, "response_order": 1, "event_sequence": 2},
                "drain": {"generation": 1, "response_order": 1, "event_sequence": 4, "response_id": "actual-response"}}}

    def test_response_done_alone_never_advances_speech(self):
        state = self.complete_state()
        self.assertTrue(replay.completed_turn(state, 0))
        for key in ("input_ended", "last_input_sample", "input_item_ids", "first_output_sample", "last_output_sample"):
            candidate = self.complete_state()
            candidate["turns"][0][key] = None
            self.assertFalse(replay.completed_turn(candidate, 0), key)

    def test_tool_only_pending_or_unstopped_output_never_advances(self):
        for key, value in (("status", "in_progress"), ("status", "cancelled"), ("has_audio_output", False)):
            state = self.complete_state()
            state["responses"]["actual-response"][key] = value
            self.assertFalse(replay.completed_turn(state, 0))
        state = self.complete_state()
        state["playback"]["drain"] = None
        self.assertFalse(replay.completed_turn(state, 0))
        state = self.complete_state()
        state["audio_frame"] = 2100
        self.assertFalse(replay.completed_turn(state, 0))

    def test_ack_with_function_call_waits_for_actual_terminal_spoken_continuation(self):
        state = self.complete_state()
        ack = state["responses"]["actual-response"]
        ack.update(has_function_calls=True, function_call_ids=["raw-provider-call"])
        self.assertFalse(replay.completed_turn(state, 0))
        answer = {"status": "in_progress", "created_frame": 3000,
                  "output_observed": False, "has_function_calls": False, "function_call_ids": [],
                  "creation_order": 2, "playback_generation": 1}
        state["responses"]["actual-continuation"] = answer
        state["turns"][0]["response_ids"].append("actual-continuation")
        state["playback"]["latest_response_order"] = 2
        self.assertFalse(replay.completed_turn(state, 0))
        answer.update(status="completed", output_observed=True, has_audio_output=True, buffer_stopped=True)
        state["playback"]["drain"] = {"generation": 1, "response_order": 2,
            "event_sequence": 7, "response_id": "actual-continuation"}
        # The old ACK's decoded samples cannot stand in for continuation audio.
        self.assertFalse(replay.completed_turn(state, 0))
        state["turns"][0]["last_output_sample"] = 4500
        state["audio_frame"] = 18000
        self.assertTrue(replay.completed_turn(state, 0))
        self.assertEqual(ack["function_call_ids"], ["raw-provider-call"])

    def test_raw_paired_flags_do_not_substitute_for_shared_drain(self):
        state = self.complete_state()
        del state["responses"]["actual-response"]["buffer_started"]
        del state["responses"]["actual-response"]["buffer_stopped"]
        self.assertTrue(replay.completed_turn(state, 0))
        state["playback"]["drain"] = None
        self.assertFalse(replay.completed_turn(state, 0))

    def test_unknown_or_out_of_generation_drain_is_rejected(self):
        for update in ({"response_id": "unknown"}, {"generation": 0}, {"response_order": 2},
                       {"event_sequence": 1}):
            state = self.complete_state()
            state["playback"]["drain"].update(update)
            self.assertFalse(replay.completed_turn(state, 0), update)

    def test_cleared_cancelled_or_newer_generation_cannot_finish_old_turn(self):
        for update in ({"interrupted": True}, {"uncertain": True}, {"generation": 2},
                       {"latest_response_order": 2}):
            state = self.complete_state()
            state["playback"].update(update)
            self.assertFalse(replay.completed_turn(state, 0), update)

    def test_every_chain_generation_must_be_completed(self):
        state = self.complete_state()
        state["turns"][0]["response_ids"].insert(0, "pending")
        state["responses"]["pending"] = {"status": "in_progress", "output_observed": False}
        self.assertFalse(replay.completed_turn(state, 0))

    def test_missing_output_or_function_presence_without_id_cannot_finish_turn(self):
        for update in ({"output_observed": False}, {"has_function_calls": True},
                       {"function_call_ids": ["actual-call"]}):
            state = self.complete_state()
            state["responses"]["actual-response"].update(update)
            self.assertFalse(replay.completed_turn(state, 0))

    def test_each_tool_bearing_continuation_still_requires_a_terminal_answer(self):
        state = self.complete_state()
        state["responses"]["actual-response"].update(has_function_calls=True, function_call_ids=["first-call"])
        state["turns"][0]["response_ids"].append("next-response")
        state["responses"]["next-response"] = {
            **self.complete_state()["responses"]["actual-response"],
            "has_function_calls": True, "function_call_ids": ["next-call"]}
        self.assertFalse(replay.completed_turn(state, 0))

    def complete_report(self):
        return {"sequence_completed": True, "decoded_signal": 100, "canonical_status": "captured",
            "cleanup": {"status": "passed"}, "host_shutdown": {"status": "passed"},
            "wiretrace": {"incomplete": False}, "browser_cleanup": {
                "media": {"peer": "closed", "context": "closed", "physical_gain": 0},
                "chrome_returncode": -15, "loopback_server_stopped": True, "process_group_gone": True}}

    def test_confirmed_browser_cleanup_allows_collection_without_semantic_pass(self):
        report = self.complete_report()
        self.assertTrue(replay.browser_cleanup_confirmed(report["browser_cleanup"]))
        self.assertTrue(replay.collection_complete(report))
        self.assertNotIn("acceptance", report)

    def test_returned_browser_cleanup_failure_rejects_collection_without_exception(self):
        for key, value in (("media_error", "timeout"), ("socket_close_error", "timeout"),
                           ("loopback_server_stopped", False), ("process_group_gone", False),
                           ("chrome_returncode", None)):
            with self.subTest(key=key):
                report = self.complete_report()
                report["browser_cleanup"][key] = value
                self.assertFalse(replay.collection_complete(report))
                self.assertEqual(report["browser_cleanup"][key], value)
                self.assertNotIn("browser_cleanup_error", report)

    def test_missing_or_unconfirmed_media_closure_rejects_collection(self):
        for media in (None, {}, {"peer": "connected", "context": "closed"},
                      {"peer": "closed", "context": "running"}):
            report = self.complete_report()
            report["browser_cleanup"]["media"] = media
            self.assertFalse(replay.collection_complete(report))
        report = self.complete_report()
        del report["browser_cleanup"]
        self.assertFalse(replay.collection_complete(report))

    def test_media_budget_failure_is_terminal(self):
        state = self.complete_state()
        state["fatal"] = "response_budget_stop"
        with self.assertRaisesRegex(replay.ReplayError, "response_budget_stop"):
            replay.completed_turn(state, 0)

    def test_bounded_pcm_is_written_as_actual_wav_and_hashed(self):
        raw = b"\x00\x00\x01\x00\xff\x7f"
        async def chunk(start, count):
            return base64.b64encode(raw[start * 2:(start + count) * 2]).decode()
        result = asyncio.run(replay.record_output(self.root, {"captured_samples": 3}, chunk))
        self.assertEqual(result["pcm_sha256"], hashlib.sha256(raw).hexdigest())
        with wave.open(str(self.root / "output.wav"), "rb") as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, 48000))
            self.assertEqual(audio.readframes(3), raw)

    def test_pcm_bounds_and_missing_bytes_fail(self):
        async def empty(*_):
            return ""
        with self.assertRaisesRegex(replay.ReplayError, "output_capture_bound"):
            asyncio.run(replay.record_output(self.root, {"captured_samples": replay.LIMITS["capture_samples"] + 1}, empty))
        with self.assertRaisesRegex(replay.ReplayError, "output_chunk_size_mismatch"):
            asyncio.run(replay.record_output(self.root, {"captured_samples": 2}, empty))

    def test_private_reader_rejects_public_and_symlink_files(self):
        path = self.root / "value"
        path.write_bytes(b"private")
        path.chmod(0o644)
        with self.assertRaisesRegex(replay.ReplayError, "private_file_required"):
            replay.private_read(path, 100)
        link = self.root / "link"
        link.symlink_to(path)
        with self.assertRaises(OSError):
            replay.private_read(link, 100)

    def test_host_lean_opt_in_is_native_profile_only(self):
        # Execute only this pure environment builder, not the host/import graph.
        path = Path(replay.__file__).with_name("frontend_host_qualification.py")
        tree = ast.parse(path.read_text())
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "child_environment")
        namespace = {"__file__": str(path), "os": os, "Path": Path, "json": json,
            "datetime": datetime, "UTC": UTC, "NO_MEDIA": "no_media", "NATIVE_MEDIA": "native_realtime_media",
            "NO_WORKLOAD": "none", "REASONING_REMINDER": "reasoning_reminder", "VISUAL_MEMORY": "visual_memory",
            "require": replay.require, "native_host_options": lambda *a, **kw: {"native_pool_max_size": 5}}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        settings = SimpleNamespace(LEAN_VOICE_ENABLED=True, JARVIS_REALTIME_MAX_SESSION_SECONDS=120,
                                   model_dump=lambda **kw: {"OPENAI_REALTIME_MODEL": "offline"})
        scope = SimpleNamespace(owner_id="fixture", area_id="area", expires_at=datetime.now(UTC) + timedelta(seconds=300))
        with patch.dict(os.environ, {"SSL_CERT_FILE": "/unused/offline.pem", "LEAN_VOICE_ENABLED": "true"}, clear=True):
            args = dict(dsn="unused", scope=scope, token="private", origin="http://127.0.0.1:1",
                        hindsight_url="unused", api_key_file="unused", configured=settings)
            self.assertEqual(namespace["child_environment"](**args, profile="native_realtime_media")["LEAN_VOICE_ENABLED"], "true")
            self.assertNotIn("LEAN_VOICE_ENABLED", namespace["child_environment"](**args, profile="no_media"))
            settings.LEAN_VOICE_ENABLED = False
            self.assertEqual(namespace["child_environment"](**args, profile="native_realtime_media")["LEAN_VOICE_ENABLED"], "false")


class SharedBufferHTMLTests(unittest.TestCase):
    """Execute the actual pure HTML reducer in Node, without browser/media I/O."""

    @classmethod
    def setUpClass(cls):
        cls.node = shutil.which("node")
        if cls.node is None:
            raise unittest.SkipTest("Node is required for offline HTML reducer checks")
        html = Path(replay.__file__).with_suffix(".html").read_text()
        cls.source = html.split("// BEGIN SHARED_BUFFER_TRACKER\n", 1)[1].split("// END SHARED_BUFFER_TRACKER", 1)[0]

    def track(self, actions, *, limit=256):
        # This process has no app imports, network, timers, providers or browser.
        program = self.source + """
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const tracker = createSharedBufferTracker(input.limit), snapshots = [];
for (const action of input.actions) {
  if (action === 'begin') tracker.beginTurn(); else tracker.observe(action);
  snapshots.push(JSON.parse(JSON.stringify(tracker.snapshot())));
}
process.stdout.write(JSON.stringify({snapshots, bindings:Object.fromEntries(
  input.actions.filter(x => x.type === 'response.created').map(x => [x.response.id, tracker.response(x.response.id)]))}));
"""
        result = subprocess.run([self.node, "-e", program], input=json.dumps({"actions": actions, "limit": limit}),
            capture_output=True, text=True, check=True, timeout=3)
        return json.loads(result.stdout)

    @staticmethod
    def created(identity):
        return {"type": "response.created", "response": {"id": identity}}

    @staticmethod
    def done(identity, status="completed"):
        return {"type": "response.done", "response": {"id": identity, "status": status}}

    @staticmethod
    def buffer(kind, identity):
        return {"type": "output_audio_buffer." + kind, "response_id": identity}

    def completed_state(self, tracked, ids):
        state = LeanSystemReplayTests.complete_state(self)
        state["playback"] = tracked["snapshots"][-1]
        state["turns"][0].update(response_ids=ids, playback_generation=state["playback"]["generation"])
        template = state["responses"]["actual-response"]
        state["responses"] = {identity: {**template, "creation_order": tracked["bindings"][identity]["response_order"],
            "playback_generation": tracked["bindings"][identity]["generation"],
            "has_function_calls": index < len(ids) - 1,
            "function_call_ids": ["actual-call-" + identity] if index < len(ids) - 1 else []}
            for index, identity in enumerate(ids)}
        return state

    def test_captured_r1_pair_r2_start_r3_stop_without_start(self):
        tracked = self.track(["begin", self.created("r1"), self.buffer("started", "r1"), self.done("r1"),
            self.created("r2"), self.buffer("stopped", "r1"), self.buffer("started", "r2"), self.done("r2"),
            self.created("r3"), self.done("r3"), self.buffer("stopped", "r3")])
        self.assertEqual(tracked["snapshots"][-1]["drain"]["response_order"], 3)
        self.assertEqual(tracked["snapshots"][-1]["last_start"]["response_id"], "r2")
        state = self.completed_state(tracked, ["r1", "r2", "r3"])
        state["responses"]["r3"].pop("buffer_started")
        state["responses"]["r2"].pop("buffer_stopped")
        self.assertTrue(replay.completed_turn(state, 0))

    def test_previous_stuck_prefix_is_covered_by_later_drain(self):
        tracked = self.track(["begin", self.created("r1"), self.buffer("started", "r1"), self.done("r1"),
            self.created("r2"), self.buffer("started", "r2"), self.done("r2"), self.buffer("stopped", "r2")])
        self.assertTrue(replay.completed_turn(self.completed_state(tracked, ["r1", "r2"]), 0))

    def test_delayed_old_stop_cannot_clear_newer_start(self):
        tracked = self.track(["begin", self.created("r1"), self.buffer("started", "r1"), self.done("r1"),
            self.created("r2"), self.buffer("started", "r2"), self.done("r2"), self.buffer("stopped", "r1")])
        self.assertIsNone(tracked["snapshots"][-1]["drain"])
        self.assertFalse(replay.completed_turn(self.completed_state(tracked, ["r1", "r2"]), 0))

    def test_new_start_invalidates_same_response_previous_drain(self):
        tracked = self.track(["begin", self.created("r1"), self.buffer("started", "r1"), self.done("r1"),
            self.buffer("stopped", "r1"), self.buffer("started", "r1")])
        self.assertIsNone(tracked["snapshots"][-1]["drain"])

    def test_unknown_stop_does_not_clear_playback(self):
        tracked = self.track(["begin", self.created("r1"), self.buffer("started", "r1"), self.done("r1"),
            self.buffer("stopped", "unknown")])
        self.assertIsNone(tracked["snapshots"][-1]["drain"])

    def test_done_without_drain_does_not_complete(self):
        tracked = self.track(["begin", self.created("r1"), self.buffer("started", "r1"), self.done("r1")])
        self.assertFalse(replay.completed_turn(self.completed_state(tracked, ["r1"]), 0))

    def test_stop_before_done_retains_valid_drain(self):
        tracked = self.track(["begin", self.created("r1"), self.buffer("started", "r1"),
            self.buffer("stopped", "r1"), self.done("r1")])
        self.assertTrue(replay.completed_turn(self.completed_state(tracked, ["r1"]), 0))

    def test_cancel_or_clear_never_normal_success(self):
        for terminal in (self.done("r1", "cancelled"), self.buffer("cleared", "r1")):
            tracked = self.track(["begin", self.created("r1"), self.buffer("started", "r1"),
                self.done("r1"), terminal, self.buffer("stopped", "r1")])
            self.assertTrue(tracked["snapshots"][-1]["interrupted"])
            self.assertFalse(replay.completed_turn(self.completed_state(tracked, ["r1"]), 0))

    def test_new_speech_interrupts_existing_chain_without_success(self):
        tracked = self.track(["begin", self.created("r1"), self.buffer("started", "r1"), self.done("r1"),
            {"type": "input_audio_buffer.speech_started"}, self.buffer("stopped", "r1")])
        self.assertTrue(tracked["snapshots"][-1]["interrupted"])

    def test_old_generation_events_do_not_clear_or_poison_new_turn(self):
        actions = ["begin", self.created("old"), self.buffer("started", "old"), self.done("old", "cancelled"),
            "begin", self.created("new"), self.buffer("started", "new"), self.done("new"),
            self.buffer("stopped", "old"), self.buffer("cleared", "old"), self.done("old", "cancelled")]
        tracked = self.track(actions + [self.buffer("stopped", "new")])
        self.assertIsNone(tracked["snapshots"][-2]["drain"])
        self.assertFalse(tracked["snapshots"][-1]["interrupted"])
        self.assertTrue(replay.completed_turn(self.completed_state(tracked, ["new"]), 0))

    def test_stale_stop_cannot_roll_back_current_drain(self):
        tracked = self.track(["begin", self.created("r1"), self.done("r1"), self.created("r2"), self.done("r2"),
            self.buffer("stopped", "r2"), self.buffer("stopped", "r1")])
        self.assertEqual(tracked["snapshots"][-1]["drain"]["response_id"], "r2")

    def test_response_history_is_bounded(self):
        tracked = self.track(["begin", self.created("r1"), self.created("r2"), self.created("r3")], limit=2)
        self.assertEqual(tracked["snapshots"][-1]["retained_responses"], 2)
        self.assertEqual(tracked["snapshots"][-1]["error"], "shared_buffer_response_identity_or_bound")


if __name__ == "__main__":
    unittest.main()
