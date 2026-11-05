"""Offline evidence guards only; no provider, model, SQL, native worker or codec."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.runtime.feature_config import FrontierConfiguration
from app.sandbox.tools import SandboxToolBridge
from app.tasks.hermes import HermesRuntimeConfiguration
from app.tasks.model_gateway import ModelEvent
from app.workspace.media_recipe import RetainedRef, TemplatePin
from evals.second_brain import managed_media_english as case
from evals.second_brain.managed_media_qualification import recipe_for


class EnglishMediaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="media-english-offline-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.probe = SimpleNamespace(root=self.root, record={}, persist=lambda: None)
        self.configuration = case.EnglishQualification(
            FrontierConfiguration(model=case.MODEL, reasoning_effort=case.REASONING_EFFORT, api_key="never-call-secret"),
            HermesRuntimeConfiguration(python=Path(sys.executable).resolve(), work_root=self.root / "hermes",
                                       host_id="offline-fixture", max_iterations=12))

    def trace(self):
        value = case.PrivateTrace(self.root / (uuid4().hex + ".jsonl"))
        self.addCleanup(value.close)
        return value

    def test_explicit_model_native_bounds_and_fresh_root(self):
        self.configuration.validate(self.root)
        self.assertNotIn("never-call-secret", json.dumps(self.configuration.public_metadata()))
        self.assertEqual(self.configuration.public_metadata()["model"], case.MODEL)
        with self.assertRaisesRegex(ValueError, "approved development model"):
            replace(self.configuration, frontier=self.configuration.frontier.model_copy(update={"model": "different"})).validate(self.root)
        for changes in ({"max_iterations": 13}, {"tool_timeout_seconds": 121},
                        {"stop_seconds": 2}, {"host_id": ""}, {"work_root": self.root / "other"}):
            with self.assertRaises(ValueError):
                replace(self.configuration, runtime=replace(self.configuration.runtime, **changes)).validate(self.root)
        self.configuration.runtime.work_root.mkdir()
        with self.assertRaises(ValueError):
            self.configuration.validate(self.root)

    def test_case_budgets_are_independent_from_scripted_case(self):
        from evals.second_brain import managed_media_qualification as original
        self.assertEqual((original.MAIN_SECONDS, original.PARENT_SECONDS), (210, 300))
        self.assertEqual((case.MAIN_SECONDS, case.PARENT_SECONDS), (300, 390))
        limits = case.task_limits()
        self.assertEqual((limits.max_attempts, limits.tool_steps, limits.task_input_tokens,
                          limits.task_output_tokens, limits.request_output_tokens), (1, 24, 100000, 8000, 4000))

    async def test_upstream_trace_preserves_events_and_exact_request_and_ids(self):
        trace = self.trace()
        events = [ModelEvent({"type": "response.created", "response": {"id": "actual-response-id"}}, b"exact-wire"),
                  ModelEvent({"type": "response.output_text.delta", "delta": "ok"}, b"second-wire")]
        payload = {"model": "fixture", "input": [{"role": "user", "content": case.INSTRUCTION}]}
        async def stream(request):
            self.assertIs(request, payload)
            for event in events:
                yield event
        real = SimpleNamespace(count=AsyncMock(return_value=51), events=stream, aclose=AsyncMock())
        wrapped = case.TracedUpstream(real, trace, self.probe)
        self.assertEqual(await wrapped.count(payload), 51)
        actual = [event async for event in wrapped.events(payload)]
        self.assertIs(actual[0], events[0])
        self.assertEqual(actual[0].wire, b"exact-wire")
        await wrapped.aclose()
        real.aclose.assert_awaited_once()
        records = [json.loads(line) for line in trace.path.read_text().splitlines()]
        self.assertEqual([r["sequence"] for r in records], list(range(1, len(records) + 1)))
        self.assertEqual(records[0]["payload"], payload)
        self.assertEqual(records[2]["identity"], records[3]["identity"])
        self.assertEqual(records[3]["payload"]["response"]["id"], "actual-response-id")
        self.assertTrue(all(not r["truncated"] and r["at"] and r["elapsed_ns"] >= 0 for r in records))
        self.assertEqual(trace.path.stat().st_mode & 0o777, 0o600)
        self.assertTrue(self.probe.record["model_called"])

    async def test_errors_never_log_raw_exception_messages(self):
        trace = self.trace()
        upstream = SimpleNamespace(count=AsyncMock(side_effect=ValueError("Bearer private-key-and-url")))
        with self.assertRaises(ValueError):
            await case.TracedUpstream(upstream, trace, self.probe).count({"model": "fixture"})
        raw = trace.path.read_text()
        self.assertNotIn("private-key", raw)
        self.assertIn("ValueError", raw)

    def test_trace_overflow_is_explicit_failure_not_truncated_success(self):
        trace = self.trace()
        trace.MAX_RECORD = 250
        with self.assertRaisesRegex(ValueError, "trace exceeded"):
            trace.append("tool_result", identity="actual-call-id", payload={"large": "x" * 1000})
        record = json.loads(trace.path.read_text())
        self.assertTrue(trace.truncated)
        self.assertTrue(record["truncated"])
        self.assertEqual(record["identity"], "actual-call-id")
        self.assertEqual(len(record["payload"]["omitted_sha256"]), 64)
        self.assertNotIn("x" * 100, trace.path.read_text())

    async def test_context_wrapper_preserves_actual_bridge_and_owner_finish(self):
        trace = self.trace()
        bridge = object.__new__(SandboxToolBridge)
        bridge.media = object()
        result = {"status": "success", "media_operation": {"state": "published", "output": {"fixture": True}}}
        original = AsyncMock(return_value=result)
        bridge.invoke = original
        factory = AsyncMock(return_value=SimpleNamespace(tools=bridge))
        factory.capabilities = ("fixture-capability",)
        factory.finish_outcome = AsyncMock(return_value=False)
        traced = case.TracedContext(factory, trace, self.probe)
        prepared = await traced(object())
        self.assertIs(prepared.tools, bridge)
        args = {"operation_key": "model-chosen"}
        self.assertIs(await bridge.invoke("media_publish", args, provider_call_id="actual-tool-id"), result)
        original.assert_awaited_once_with("media_publish", args, provider_call_id="actual-tool-id")
        self.assertEqual(traced.capabilities, factory.capabilities)
        self.assertFalse(await traced.finish_outcome("context", "outcome"))
        factory.finish_outcome.assert_awaited_once_with("context", "outcome")
        records = [json.loads(line) for line in trace.path.read_text().splitlines()]
        self.assertEqual([r["identity"] for r in records], ["actual-tool-id", "actual-tool-id"])
        self.assertEqual(records[0]["payload"]["arguments"], args)
        self.assertEqual(self.probe.record["english_publications"], [result["media_operation"]])

    def test_recipe_semantics_require_exact_input_cuts_caption_audio_and_fit(self):
        ref = RetainedRef(file_id=uuid4(), version_id="fixture-version", content_sha256="a" * 64)
        template = TemplatePin(id="jarvis.clip.v1", version="fixture", bundle_sha256="b" * 64)
        recipe = recipe_for(ref, template)
        case.verify_recipe(recipe, ref, template)
        # Model-chosen identifiers are allowed; content/source drift is not.
        case.verify_recipe(recipe.model_copy(update={"edit_id": uuid4()}), ref, template)
        for changed in (
            recipe.model_copy(update={"audio": "mute"}),
            recipe.model_copy(update={"fit": "cover"}),
            recipe.model_copy(update={"segments": tuple(reversed(recipe.segments))}),
            recipe.model_copy(update={"captions": []}),
            recipe.model_copy(update={"sources": []}),
        ):
            with self.assertRaisesRegex(ValueError, "English request"):
                case.verify_recipe(changed, ref, template)
        with self.assertRaises(ValueError):
            case.verify_recipe(recipe, ref.model_copy(update={"version_id": "wrong-version"}), template)

    def test_native_cleanup_requires_exact_attempt_and_absent_home(self):
        root = self.configuration.runtime.work_root / uuid4().hex
        root.mkdir(parents=True)
        self.probe.record.update(model_called=True, task_id=str(uuid4()), attempt_id=str(uuid4()))
        value = {key: self.probe.record[key] for key in ("task_id", "attempt_id")}
        value.update(phase="exited", private_home_cleanup={"state": "complete"}, pid=None)
        path = root / "run.json"
        path.write_text(json.dumps(value))
        case.verify_native_cleanup(self.configuration, self.probe)
        (root / "home").mkdir()
        with self.assertRaisesRegex(ValueError, "unconfirmed"):
            case.verify_native_cleanup(self.configuration, self.probe)
        (root / "home").rmdir()
        value["attempt_id"] = str(uuid4())
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            case.verify_native_cleanup(self.configuration, self.probe)

    def test_no_native_receipt_cannot_qualify_actual_model_call(self):
        self.probe.record["model_called"] = True
        with self.assertRaisesRegex(ValueError, "unique native"):
            case.verify_native_cleanup(self.configuration, self.probe)

    async def test_pending_executor_is_cancelled_and_observed(self):
        provider = object.__new__(case.ObservedHermesProvider)
        stopped = asyncio.Event()
        async def pending():
            try:
                await asyncio.sleep(60)
            finally:
                stopped.set()
        provider.active = asyncio.create_task(pending())
        await asyncio.sleep(0)
        await provider.observe_stopped()
        self.assertTrue(stopped.is_set())
        self.assertTrue(provider.active.done())

    async def test_executor_trace_retains_sanitized_project_frames_not_error_text(self):
        trace = self.trace()
        provider = object.__new__(case.ObservedHermesProvider)
        provider.prepare = SimpleNamespace(trace=trace)
        context = SimpleNamespace(claim=SimpleNamespace(attempt_id=uuid4()))
        error = ValueError("secret-token raw-SQL private-host")
        with patch("app.tasks.hermes.HermesTaskProvider.execute", AsyncMock(side_effect=error)):
            with self.assertRaises(ValueError) as raised:
                await provider.execute(context)
        self.assertIs(raised.exception, error)
        record = json.loads(trace.path.read_text())
        self.assertEqual(record["kind"], "executor_error")
        self.assertEqual(record["identity"], str(context.claim.attempt_id))
        self.assertEqual(record["payload"]["error_type"], "ValueError")
        self.assertTrue(any(frame["file"] == "evals/second_brain/managed_media_english.py"
                            and frame["function"] == "execute" for frame in record["payload"]["frames"]))
        self.assertLessEqual(len(record["payload"]["frames"]), 8)
        for private in ("secret-token", "raw-SQL", "private-host", str(self.root)):
            self.assertNotIn(private, trace.path.read_text())


if __name__ == "__main__":
    unittest.main()
