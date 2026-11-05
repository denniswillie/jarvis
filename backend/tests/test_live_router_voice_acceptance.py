"""Offline seam/evidence tests only: no provider, SQL, browser, or server."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
import wave


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import live_router_voice_acceptance as r
import live_luna_voice_runtime as runtime


def speech_builder(root, script):
    """Explicit offline-only PCM fixture, never accepted as live qualification."""
    import hashlib
    from evals.responsive_voice.speech_assets import inspect_pcm
    root.mkdir(mode=0o700)
    clips = []
    for identity, text in script:
        path = root / (identity + ".wav")
        with wave.open(str(path), "wb") as output:
            output.setparams((1, 2, r.h.RATE, 0, "NONE", "not compressed"))
            output.writeframes(struct.pack("<h", 1000) * 4800)
        clips.append(dict(id=identity, text=text, relative_path=path.name,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **inspect_pcm(path)))
    return {"clips": clips}


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"

    def prepare(self, mode="lookup", **kwargs):
        return r.prepare(self.root, mode=mode, speech_builder=speech_builder, **kwargs)

    def test_modes_use_original_bundle_but_only_note_has_commit_gate(self):
        for mode in r.QUESTIONS:
            root = self.root / mode
            value = r.prepare(root, mode=mode, speech_builder=speech_builder)
            config, limits, clips = r.scenario(root)
            self.assertEqual(config["mode"], mode)
            self.assertIs(config["task_result_gate"], mode == "work")
            self.assertEqual(clips[0]["text"], r.QUESTIONS[mode])
            self.assertEqual(limits.scenario_seconds, 180)
            self.assertFalse(value["paid_execution"])

    def test_readonly_rejects_hooks_before_creating_artifacts(self):
        with self.assertRaisesRegex(r.h.HarnessError, "hooks_require_work_mode"):
            self.prepare(busy_followup="Actually use the red shelf.")
        self.assertFalse(self.root.exists())

    def test_original_note_hooks_retained(self):
        self.prepare("work", busy_followup="Use the red shelf.", continuation="Add the cup too.")
        _, _, clips = r.scenario(self.root)
        self.assertEqual([clip["id"] for clip in clips], ["work", "busy_followup", "continuation"])

    def test_manifest_mismatch_and_tampered_pcm_rejected(self):
        self.prepare()
        config = r.h.read_json(self.root / "router-scenario.json")
        config["question"] = "different"
        r.h.private_write(self.root / "router-scenario.json", config, exclusive=False)
        with self.assertRaisesRegex(r.h.HarnessError, "router_scenario_bundle_mismatch"):
            r.scenario(self.root)
        with (self.root / "speech/work.wav").open("ab") as output:
            output.write(b"changed")
        with self.assertRaisesRegex(r.h.HarnessError, "speech_hash_mismatch"):
            r.scenario(self.root)

    def test_observer_preserves_exact_multiline_input_output_and_is_private(self):
        self.prepare()
        trace = r.AcceptanceTrace(self.root, r.h.Limits())
        observer = runtime.RouterObserver(self.root, r.h.Limits(), trace)
        self.addCleanup(trace.close)
        self.addCleanup(observer.close)
        event = dict(stage="request", delegation_id="d", payload={"model": "gpt-6-luna",
            "reasoning": {"effort": "none"}, "input": [{"role": "user", "content": "first\nsecond"}]})
        observer(event)
        response = dict(stage="response", payload={"output": [{"type": "message", "content": "No active tasks.\nNext line"}]})
        observer(response)
        observer(dict(api_key="private-key", sdp="private-sdp"))
        rows = [json.loads(line) for line in (self.root / "router-events.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["event"], event)
        self.assertEqual(rows[1]["event"], response)
        rendered = (self.root / "router-timeline.txt").read_text()
        self.assertIn("L0002 | second", rendered)
        self.assertNotIn("private-key", rendered)
        self.assertNotIn("private-sdp", rendered)
        self.assertEqual(trace.router_observations, 3)
        self.assertEqual((self.root / "router-events.jsonl").stat().st_mode & 0o077, 0)

    def test_observer_rejects_non_json_and_trace_overflow(self):
        self.prepare()
        observer = runtime.RouterObserver(self.root, r.h.Limits(max_trace_bytes=100))
        self.addCleanup(observer.close)
        with self.assertRaises(TypeError):
            observer({"response": object()})
        with self.assertRaisesRegex(r.h.HarnessError, "router_trace_budget_exceeded"):
            observer({"input": "x" * 1000})

    def test_readonly_effects_and_chat_delegation_fail_closed(self):
        baseline = {table: [] for table in r.h.TABLES}
        trace = SimpleNamespace(worker_requests=0, native_delegation_events=[], router_observations=0)
        r.check_no_effects(baseline, baseline, trace, "chat")
        for table in r.h.TABLES:
            changed = {**baseline, table: [{"id": "unexpected"}]}
            with self.subTest(table=table), self.assertRaises(r.h.HarnessError):
                r.check_no_effects(changed, baseline, trace, "lookup")
        trace.router_observations = 1
        with self.assertRaisesRegex(r.h.HarnessError, "unexpected_chat_delegation"):
            r.check_no_effects(baseline, baseline, trace, "chat")
        trace.worker_requests = 1
        with self.assertRaisesRegex(r.h.HarnessError, "unexpected_hermes_model_request"):
            r.check_no_effects(baseline, baseline, trace, "lookup")

    def test_lookup_ack_before_result_and_no_pcm_do_not_collect_evidence(self):
        trace = SimpleNamespace(router_observations=1, last_commentary=3., transcript=[
            {"monotonic_s": 2., "event": {"type": "session.output_transcript.delta", "delta": "Checking."}}])
        speech = dict(host_after_monotonic_s=1., end_sample=4800)
        state = dict(lastSignal=9600, audio_sample=96000)
        self.assertFalse(r.spoken_evidence_ready(trace, state, speech, "lookup"))
        trace.transcript.append({"monotonic_s": 4., "event": {
            "type": "session.output_transcript.delta", "delta": "There are no active tasks."}})
        self.assertTrue(r.spoken_evidence_ready(trace, state, speech, "lookup"))
        self.assertFalse(r.spoken_evidence_ready(trace, {**state, "lastSignal": None}, speech, "lookup"))

    def report(self, mode):
        self.prepare(mode)
        value = dict(router_format=r.FORMAT, mode=mode, evidence_collected=True,
            cleanup=dict(fixture_cleaned=True, session_and_worker_joined=True, provider_shutdown_confirmed=True,
                         application_context_closed=True, browser={"media": {"closed": True},
                             "process_group_gone": True, "loopback_server_stopped": True, "chrome_returncode": -15}),
            inputs=[{"last_active_sample": 4800}], worker_model_requests=0,
            router_observations=0 if mode == "chat" else 2, native_delegation_events=[],
            transcript=[{"event": {"type": "session.output_transcript.delta", "delta": "Useful answer."}}])
        r.h.private_write(self.root / "result.json", value)
        with wave.open(str(self.root / "output.wav"), "wb") as output:
            output.setparams((1, 2, r.h.RATE, 0, "NONE", "not compressed"))
            output.writeframes(struct.pack("<h", 1000) * 96000)
        return value

    def test_lookup_requires_actual_read_and_useful_answer_review_not_note_commit(self):
        self.report("lookup")
        args = dict(useful_audio_sample=52800, spoken_result_correct=True, transcript_correct=True,
                    router_trace_correct=True)
        with self.assertRaisesRegex(r.h.HarnessError, "explicit_live_state_read_review_required"):
            r.review(self.root, **args)
        result = r.review(self.root, **args, live_state_read_correct=True)
        self.assertEqual(result["useful_audio_latency_ms"], 1000)
        self.assertEqual(result["task_result_gate"], "not_applicable")
        self.assertFalse(result["physical_hearing"])

    def test_chat_review_needs_transcript_review_but_no_router_or_note_attestation(self):
        self.report("chat")
        with self.assertRaisesRegex(r.h.HarnessError, "explicit_useful_answer_review_required"):
            r.review(self.root, useful_audio_sample=52800, spoken_result_correct=True)
        result = r.review(self.root, useful_audio_sample=52800, spoken_result_correct=True, transcript_correct=True)
        self.assertEqual(result["semantic_acceptance"], "reviewer_confirmed")

    def test_review_rejects_unconfirmed_provider_shutdown_and_empty_transcript(self):
        value = self.report("lookup")
        args = dict(useful_audio_sample=52800, spoken_result_correct=True, transcript_correct=True,
                    router_trace_correct=True, live_state_read_correct=True)
        value["cleanup"]["provider_shutdown_confirmed"] = False
        r.h.private_write(self.root / "result.json", value, exclusive=False)
        with self.assertRaisesRegex(r.h.HarnessError, "cleanup_incomplete"):
            r.review(self.root, **args)
        value["cleanup"]["provider_shutdown_confirmed"] = True
        value["transcript"] = []
        r.h.private_write(self.root / "result.json", value, exclusive=False)
        with self.assertRaisesRegex(r.h.HarnessError, "output_transcript_missing"):
            r.review(self.root, **args)

    def test_review_requires_real_browser_media_process_and_server_cleanup_receipts(self):
        value = self.report("chat")
        args = dict(useful_audio_sample=52800, spoken_result_correct=True, transcript_correct=True)
        complete = value["cleanup"]["browser"]
        incomplete = [
            {"closed": True},
            {key: child for key, child in complete.items() if key != "media"},
            {**complete, "media": {"closed": False}},
            {key: child for key, child in complete.items() if key != "process_group_gone"},
            {**complete, "process_group_gone": False},
            {key: child for key, child in complete.items() if key != "loopback_server_stopped"},
            {**complete, "loopback_server_stopped": False},
            {**complete, "media_error": "offline_failure"},
            {**complete, "socket_close_error": "offline_failure"},
        ]
        for browser in incomplete:
            with self.subTest(browser=browser):
                value["cleanup"]["browser"] = browser
                r.h.private_write(self.root / "result.json", value, exclusive=False)
                with self.assertRaisesRegex(r.h.HarnessError, "cleanup_incomplete"):
                    r.review(self.root, **args)
        value["cleanup"]["browser"] = complete
        r.h.private_write(self.root / "result.json", value, exclusive=False)
        result = r.review(self.root, **args)
        self.assertEqual(result["semantic_acceptance"], "reviewer_confirmed")


class AsyncOfflineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"

    async def test_paid_guard_precedes_all_live_construction(self):
        with patch.object(r.h, "run", AsyncMock()) as run:
            with self.assertRaisesRegex(r.h.HarnessError, "explicit_paid_execution_required"):
                await r.run(self.root, settings=None)
        run.assert_not_awaited()

    async def test_modes_reuse_original_lifecycle_and_note_driver_without_edits(self):
        original_drive, original_trace = r.h.drive, r.h.Trace
        async def offline_run(root, **kwargs):
            self.assertEqual(kwargs["factory"], r.FACTORY)
            self.assertTrue(kwargs["execute_paid"])
            trace = r.h.Trace(root, r.h.Limits())
            if self.mode == "work":
                self.assertIs(r.h.drive, original_drive)
                trace.record("luna_router_observe", payload={"stage": "request"})
            else:
                self.assertIs(r.h.drive.func, r.drive_readonly)
            trace.close()
            return {"evidence_collected": False}
        for self.mode in ("work", "chat", "lookup"):
            root = self.root / self.mode
            r.prepare(root, mode=self.mode, speech_builder=speech_builder)
            with patch.object(r.h, "run", side_effect=offline_run):
                report = await r.run(root, settings=object(), execute_paid=True)
            self.assertEqual(report["mode"], self.mode)
            self.assertIs(r.h.drive, original_drive)
            self.assertIs(r.h.Trace, original_trace)
        self.assertEqual(report["failure"]["code"], "router_observations_missing")

    def runtime_modules(self, *, fail=False):
        bridge = object()
        app = SimpleNamespace(session=SimpleNamespace(work=bridge, delegation_handler=bridge), runner=object())
        client = SimpleNamespace(close=AsyncMock())
        sdk = ModuleType("openai")
        sdk.AsyncOpenAI = Mock(return_value=client)
        core = ModuleType("app.realtime.live_router")
        core.LunaLiveRouter = Mock(return_value=SimpleNamespace())
        base = ModuleType("live_hermes_voice_runtime")
        closed = []
        @asynccontextmanager
        async def original(**kwargs):
            if fail:
                raise RuntimeError("offline construction failure")
            try:
                yield app
            finally:
                closed.append(True)
        base.application = original
        return {"openai": sdk, "app.realtime.live_router": core, "live_hermes_voice_runtime": base}, app, client, bridge, closed

    def runtime_arguments(self):
        self.root.mkdir(mode=0o700)
        return dict(fixture=object(), session_id="session", websocket=SimpleNamespace(), offer="offline-unused",
                    client_capabilities={}, limits=r.h.Limits(), artifact_dir=self.root,
                    settings=SimpleNamespace(OPENAI_API_KEY="offline-settings-key"))

    async def test_runtime_attaches_router_keeps_bridge_and_closes_actual_client_seam(self):
        modules, app, client, bridge, closed = self.runtime_modules()
        with patch.dict(sys.modules, modules):
            async with runtime.application(**self.runtime_arguments()) as actual:
                self.assertIs(actual, app)
                self.assertIs(app.session.work, bridge)
                self.assertIs(app.session.delegation_handler, modules["app.realtime.live_router"].LunaLiveRouter.return_value)
                args = modules["app.realtime.live_router"].LunaLiveRouter.call_args
                self.assertEqual(args.args, (app.session, client))
                self.assertTrue(callable(args.kwargs["observe"]))
                modules["openai"].AsyncOpenAI.assert_called_once_with(
                    api_key="offline-settings-key", max_retries=0, timeout=8.0)
                client.close.assert_not_awaited()
        client.close.assert_awaited_once()
        self.assertEqual(closed, [True])
        receipt = (self.root / "luna-runtime-composition.json").read_text()
        self.assertNotIn("offline-settings-key", receipt)
        self.assertFalse(json.loads(receipt)["serving_activation"])

    async def test_runtime_closes_client_on_factory_failure_and_body_failure(self):
        modules, _, client, _, _ = self.runtime_modules(fail=True)
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "offline construction failure"):
            async with runtime.application(**self.runtime_arguments()):
                self.fail("must not yield")
        client.close.assert_awaited_once()
        self.root = Path(self.temp.name) / "body"
        modules, _, client, _, closed = self.runtime_modules()
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "body failure"):
            async with runtime.application(**self.runtime_arguments()):
                raise RuntimeError("body failure")
        client.close.assert_awaited_once()
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
