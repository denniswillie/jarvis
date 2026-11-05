"""Offline tests for the independent acceptance sidecar. No providers/SQL/Chrome."""
import asyncio
from contextlib import asynccontextmanager
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
from uuid import uuid4
import unittest
from unittest.mock import AsyncMock, patch
import wave


PATH = Path(__file__).resolve().parents[2] / "scripts/live_hermes_voice_acceptance.py"
SPEC = importlib.util.spec_from_file_location("live_hermes_voice_acceptance", PATH)
h = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = h
SPEC.loader.exec_module(h)


def speech_builder(root, script):
    from evals.responsive_voice.speech_assets import inspect_pcm
    import hashlib
    root.mkdir(mode=0o700)
    clips = []
    for identity, text in script:
        path = root / (identity + ".wav")
        with wave.open(str(path), "wb") as output:
            output.setparams((1, 2, h.RATE, 0, "NONE", "not compressed"))
            output.writeframes(struct.pack("<h", 1000) * 4800)
        clips.append(dict(id=identity, text=text, relative_path=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **inspect_pcm(path)))
    return {"clips": clips}


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"

    def prepare(self, **kwargs):
        return h.prepare(self.root, speech_builder=speech_builder, **kwargs)

    def test_prepare_and_validate_are_local_and_keep_hooks_out_of_audio_protocol(self):
        result = self.prepare(busy_followup="Actually, use the red shelf.", continuation="Add the matching cup too.")
        _, limits, clips = h.validate_bundle(self.root)
        self.assertEqual([clip["id"] for clip in clips], ["work", "busy_followup", "continuation"])
        self.assertFalse(result["paid_execution"])
        self.assertEqual(limits.max_attempts, 2)
        self.assertEqual(self.root.stat().st_mode & 0o077, 0)
        self.assertNotIn("expected", result["speech"])

    def test_mutated_wav_is_rejected_before_any_execution(self):
        self.prepare()
        with (self.root / "speech/work.wav").open("ab") as output:
            output.write(b"changed")
        with self.assertRaisesRegex(h.HarnessError, "speech_hash_mismatch"):
            h.validate_bundle(self.root)

    def test_path_traversal_and_duplicate_hooks_are_rejected(self):
        self.prepare()
        manifest = h.read_json(self.root / "manifest.json")
        manifest["speech"]["clips"][0]["relative_path"] = "../other.wav"
        h.private_write(self.root / "manifest.json", manifest, exclusive=False)
        with self.assertRaisesRegex(h.HarnessError, "clip_script_mismatch"):
            h.validate_bundle(self.root)

    def test_limits_cannot_grow_or_accept_boolean(self):
        for limits in (h.Limits(scenario_seconds=181), h.Limits(tool_steps=True), h.Limits(max_attempts=0)):
            with self.subTest(limits=limits), self.assertRaises(h.HarnessError):
                limits.validate()

    def test_trace_keeps_every_prompt_line_but_never_sdp_or_keys(self):
        self.prepare()
        trace = h.Trace(self.root, h.Limits())
        trace.record("creation_send_attempt", payload={"session":{"instructions":"first\nsecond", "api_key":"private-key"},
                                                       "transport":{"sdp":"private-sdp"}})
        trace.close()
        raw = (self.root / "events.jsonl").read_text()
        text = (self.root / "timeline.txt").read_text()
        self.assertIn("first\\nsecond", raw)
        self.assertIn("instructions:L0001 | first", text)
        self.assertIn("instructions:L0002 | second", text)
        self.assertNotIn("private-key", raw + text)
        self.assertNotIn("private-sdp", raw + text)

    def test_note_prose_and_ack_do_not_count_as_a_commit(self):
        facts = h.state_facts({"jarvis_background_tasks":[{"id":"task", "status":"succeeded", "answer":"Saved!"}],
                               "jarvis_note_operations":[{"operation_id":"bad", "outcome":"unknown"}]})
        self.assertTrue(facts["idle"])
        self.assertEqual(facts["completed_results"], [])

    def test_real_running_boundary_not_task_title_drives_busy_hook(self):
        self.assertFalse(h.state_facts({"jarvis_task_attempts":[{"status":"running", "provider_state":"pending"}]})["running"])
        self.assertTrue(h.state_facts({"jarvis_task_attempts":[{"status":"running", "provider_state":"live"}]})["running"])

    def test_useful_latency_requires_review_not_any_nonzero_output(self):
        self.prepare()
        report = {"evidence_collected":True, "inputs":[{"last_active_sample":4800}],
                  "cleanup":{"fixture_cleaned":True, "session_and_worker_joined":True}}
        h.private_write(self.root / "result.json", report)
        with wave.open(str(self.root / "output.wav"), "wb") as output:
            output.setparams((1, 2, h.RATE, 0, "NONE", "not compressed"))
            output.writeframes(struct.pack("<h", 1000) * 96000)
        with self.assertRaisesRegex(h.HarnessError, "explicit_semantic_review_required"):
            h.review(self.root, useful_audio_sample=52800, task_correct=True, spoken_result_correct=False)
        result = h.review(self.root, useful_audio_sample=52800, task_correct=True, spoken_result_correct=True)
        self.assertEqual(result["useful_audio_latency_ms"], 1000)
        self.assertIn("no hearing claim", result["boundary"])

    def test_review_rejects_unfinished_fixture_cleanup(self):
        self.prepare()
        h.private_write(self.root / "result.json", {"evidence_collected":True, "cleanup":{"fixture_cleaned":False}})
        with self.assertRaisesRegex(h.HarnessError, "cleanup_incomplete"):
            h.review(self.root, useful_audio_sample=1, task_correct=True, spoken_result_correct=True)


class AsyncOfflineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        h.prepare(self.root, speech_builder=speech_builder)
        self.trace = h.Trace(self.root, h.Limits())
        self.addCleanup(self.trace.close)

    async def test_paid_opt_in_checked_before_factory_or_database(self):
        factory = AsyncMock()
        with self.assertRaisesRegex(h.HarnessError, "explicit_paid_execution_required"):
            await h.run(self.root, factory="must.not.import:factory", settings=None, fixture_factory=factory)
        factory.assert_not_called()

    async def test_socket_observer_forwards_original_events_once(self):
        raw = json.dumps({"type":"session.commentary.append", "content":"Saved note\nexact second line", "delegation_id":"d"})
        socket = SimpleNamespace(send=AsyncMock(), recv=AsyncMock(return_value=raw), close=AsyncMock())
        observed = h.ObservedSocket(socket, self.trace)
        await observed.send(raw)
        self.assertEqual(await observed.recv(), raw)
        socket.send.assert_awaited_once_with(raw)
        self.assertIsNotNone(self.trace.last_commentary)
        self.assertIsNone(self.trace.last_caption)

    async def test_peer_acks_media_only_after_real_browser_operation(self):
        browser = SimpleNamespace(cdp=SimpleNamespace(evaluate=AsyncMock(return_value={"provider_started":True,"output_gated":True})))
        peer = h.BrowserPeer(browser, self.trace)
        event = dict(type="live.media.prepare", answer_sdp="v=0 private", route={"id":"r"},
                     audio_session_id="a", command_id="c", provider_session_id="p")
        await peer.send_json(event)
        receipt = json.loads(await peer.receive_text())
        self.assertEqual(receipt["phase"], "prepare")
        self.assertNotIn("answer_sdp", receipt)
        browser.cdp.evaluate.assert_awaited_once()
        browser.cdp.evaluate.return_value = {"provider_started":False,"output_gated":True}
        with self.assertRaisesRegex(h.HarnessError, "actual_media_operation_failed"):
            await peer.send_json(event)
        self.assertTrue(peer.queue.empty())

    async def test_worker_observer_forwards_real_model_events_and_counts(self):
        async def events(payload):
            yield SimpleNamespace(data={"type":"response.output_text.delta","delta":"Saved."})
        upstream = SimpleNamespace(count=AsyncMock(return_value=42), events=events, aclose=AsyncMock())
        observed = h.ObservedUpstream(upstream, self.trace)
        payload = {"model":"explicit-worker", "input":[{"role":"user","content":"original message"}]}
        self.assertEqual(await observed.count(payload), 42)
        result = [event async for event in observed.events(payload)]
        self.assertEqual(result[0].data["delta"], "Saved.")
        self.assertEqual(self.trace.worker_requests, 1)
        await observed.aclose()
        upstream.aclose.assert_awaited_once()

    async def test_canonical_export_only_reads_fixed_tables_and_exact_owner(self):
        queries = []
        class Connection:
            async def fetch(self, sql, owner, limit):
                queries.append((sql, owner, limit))
                return [{"row":json.dumps({"owner_id":owner})}]
        class Pool:
            @asynccontextmanager
            async def acquire(self):
                yield Connection()
        data = await h.canonical_snapshot(SimpleNamespace(pool=Pool(),owner="fixture-owner"), h.Limits())
        self.assertEqual(set(data), set(h.TABLES))
        self.assertTrue(all(owner == "fixture-owner" and "where owner_id=$1" in sql for sql,owner,_ in queries))

    async def test_live_close_error_does_not_skip_worker_shutdown(self):
        session = SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("private failure")), shutdown_confirmed=False)
        runner = SimpleNamespace(stop=AsyncMock(), _active=set())
        job = asyncio.create_task(asyncio.sleep(100))
        receipt = await h.stop_application(h.HarnessApplication(session, runner), SimpleNamespace(_runs={}), [job], h.Limits())
        runner.stop.assert_awaited_once()
        self.assertTrue(job.done())
        self.assertTrue(receipt["session_and_worker_joined"])
        self.assertFalse(receipt["provider_shutdown_confirmed"])
        self.assertEqual(receipt["errors"], ["RuntimeError"])

    async def test_unconfirmed_native_cleanup_is_not_marked_safe(self):
        session = SimpleNamespace(close=AsyncMock(), shutdown_confirmed=True)
        runner = SimpleNamespace(stop=AsyncMock(), _active=set())
        native = SimpleNamespace(stop=AsyncMock(side_effect=OSError("private")))
        receipt = await h.stop_application(h.HarnessApplication(session, runner), SimpleNamespace(_runs={"run":native}), [], h.Limits())
        self.assertFalse(receipt["session_and_worker_joined"])
        native.stop.assert_awaited_once()
        self.assertEqual(receipt["errors"], ["OSError"])

    async def test_fixture_manifest_is_retained_when_worker_shutdown_unknown(self):
        fixture = SimpleNamespace(owners=("one", "two"), create_schema=AsyncMock(return_value="fixture_schema"),
                                  cleanup=AsyncMock(), manifest={}, persist=lambda:None)
        connection = SimpleNamespace(close=AsyncMock(), execute=AsyncMock(), fetchval=AsyncMock(return_value="area"))
        class Pool:
            @asynccontextmanager
            async def acquire(self):
                yield connection
        pool = Pool()
        with patch("evals.second_brain.fixtures.DatabaseFixture", return_value=fixture), \
             patch("asyncpg.connect", AsyncMock(return_value=connection)), \
             patch("asyncpg.create_pool", AsyncMock(return_value=pool)), \
             patch("app.migrations.runner.apply_migrations", AsyncMock()), \
             patch("evals.second_brain.source_fixture.close_fixture_pool", AsyncMock()) as close:
            async with h.isolated_fixture("not-a-real-dsn", self.root, test_name="offline", limits=h.Limits(),
                                          cleanup_gate={"safe":False}):
                pass
        close.assert_awaited_once_with(pool)
        fixture.cleanup.assert_not_called()
        self.assertEqual(fixture.manifest["status"], "cleanup_deferred")


class ProductionFactoryOfflineTests(unittest.IsolatedAsyncioTestCase):
    """Real construction/services; only SQL reads/readiness are external fakes."""
    async def asyncSetUp(self):
        runtime_path = PATH.with_name("live_hermes_voice_runtime.py")
        spec = importlib.util.spec_from_file_location("live_hermes_voice_runtime", runtime_path)
        self.runtime = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.runtime
        spec.loader.exec_module(self.runtime)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root.chmod(0o700)
        from app.knowledge.sources import SourceRepository
        from app.settings import JarvisSettings
        self.pool = SimpleNamespace(close=AsyncMock())
        self.fixture = SimpleNamespace(pool=self.pool, sources=SourceRepository(self.pool),
            owner="direct-image-eval-second-brain-" + uuid4().hex + "-a", area=uuid4())
        self.session_id = uuid4()
        self.settings = JarvisSettings(_env_file=None, OPENAI_API_KEY="offline-unused-key",
            SUPABASE_DB_URL="postgresql://unused", OPENAI_LIVE_ENABLED="true", RESPONSIVE_VOICE_ENABLED="true")
        flags = SimpleNamespace(frontier_model="gpt-5.6-terra", frontier_reasoning_effort="low", hermes_python=sys.executable)
        self.patches = [
            patch.object(self.runtime, "SecondBrainSettings", return_value=flags),
            patch.dict(os.environ, {"JARVIS_HERMES_PYTHON":sys.executable}),
            patch.object(self.runtime.SessionRepository, "get_session", AsyncMock(return_value=SimpleNamespace(
                id=self.session_id, status="active", behavior_instructions="", metadata={}))),
            patch.object(self.runtime.TaskWorkerRegistry, "advertise", AsyncMock()),
            patch.object(self.runtime.TaskWorkerRegistry, "withdraw", AsyncMock()),
            patch("app.realtime.live_session.LiveBackendTransport", side_effect=AssertionError("provider forbidden")),
            patch.object(self.runtime.HermesTaskProvider, "execute", side_effect=AssertionError("native execution forbidden")),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def arguments(self):
        return dict(fixture=self.fixture, session_id=self.session_id,
            websocket=SimpleNamespace(close=AsyncMock(), send_json=AsyncMock()),
            offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n", client_capabilities=dict(h.CAPABILITIES),
            limits=h.Limits(), artifact_dir=self.root, settings=self.settings)

    async def test_actual_normal_factory_has_scoped_hermes_capture_and_live_work(self):
        from app.realtime.live_work import LiveWorkBridge
        from app.tasks.context_capture import SessionContextCapture
        real_factory = self.runtime.create_live_session
        with patch.object(self.runtime, "create_live_session", wraps=real_factory) as factory:
            async with self.runtime.application(**self.arguments()) as app:
                h.check_application(app, self.fixture, self.session_id, h.Limits())
                self.assertIsInstance(app.session.delegation_handler, LiveWorkBridge)
                self.assertIs(app.session.work, app.session.delegation_handler)
                self.assertIsInstance(app.session._context_capture, SessionContextCapture)
                self.assertIs(app.session._context_capture.repository.pool, self.pool)
                self.assertIsNone(app.services.knowledge_runtime)
                self.assertIsNone(app.services.recording_admission)
                self.assertIsNone(app.runner.providers["frontier"].prepare.sandbox_provider)
                self.assertEqual(app.runner.providers["frontier"].model, "gpt-5.6-terra")
                self.assertEqual(app.runner.providers["frontier"].configuration.work_root, self.root / "hermes")
                names = {row["name"] for row in app.session.work.commands.catalog()}
                self.assertTrue({"read_live_state", "read_preferences", "reply_to_user"} <= names)
                self.assertNotIn("live_recording", names)
                self.assertNotIn("view_sandbox", names)
                status = await app.services.realtime_session_manager.get_session_diagnostics(self.session_id)
                self.assertTrue(status["realtime_connected"])
                self.assertFalse(app.session._ran or app.runner._active)
                self.runtime.TaskWorkerRegistry.advertise.assert_awaited_once()
            factory.assert_awaited_once()
            self.assertNotIn("delegation_handler", factory.call_args.kwargs)
        self.pool.close.assert_not_awaited()
        self.runtime.TaskWorkerRegistry.withdraw.assert_awaited_once_with(self.fixture.owner, app.runner.worker_id)
        self.assertTrue(app.services._closed)
        self.assertTrue(app.services.live_work_runtime.closed)
        self.assertTrue(app.services.behavior_profile_updates.closed)
        receipt = json.loads((self.root / "runtime-composition.json").read_text())
        self.assertEqual(receipt["delegation_handler"], "LiveWorkBridge")
        self.assertNotIn("offline-unused-key", json.dumps(receipt))
        self.assertFalse(receipt["paid_execution_qualified"])

    async def test_no_selected_active_session_refuses_construction_before_readiness(self):
        with patch.object(self.runtime.SessionRepository, "get_session", AsyncMock(return_value=None)):
            with self.assertRaisesRegex(RuntimeError, "acceptance_active_owned_session_required"):
                async with self.runtime.application(**self.arguments()):
                    self.fail("unexpected constructed application")
        self.runtime.TaskWorkerRegistry.advertise.assert_not_awaited()
        self.pool.close.assert_not_awaited()

    async def test_configured_model_and_limits_do_not_mutate_serving_settings(self):
        original = self.settings.model_dump()
        frontier, budgets, native = self.runtime.configuration(self.settings, h.Limits(), self.root)
        self.assertEqual(frontier.model, "gpt-5.6-terra")
        self.assertEqual(frontier.reasoning_effort, "low")
        self.assertEqual((budgets.owner_concurrency, budgets.global_concurrency), (1, 1))
        self.assertEqual((budgets.active_seconds, budgets.tool_steps, budgets.max_attempts), (120, 20, 2))
        self.assertEqual(self.settings.model_dump(), original)
        self.assertEqual(native.python, Path(sys.executable))

    async def test_missing_explicit_native_interpreter_never_falls_back(self):
        with patch.dict(os.environ, {"JARVIS_HERMES_PYTHON":str(self.root / "missing-python")}):
            with self.assertRaisesRegex(RuntimeError, "acceptance_native_interpreter_unavailable"):
                self.runtime.configuration(self.settings, h.Limits(), self.root)


if __name__ == "__main__":
    unittest.main()
