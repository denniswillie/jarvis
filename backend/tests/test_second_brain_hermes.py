import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from pydantic import ValidationError

from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider
from app.tasks.hermes_result import HermesTaskResult, WORKER_SYSTEM_PROMPT
from app.tasks.models import TaskError, TaskStatus
from app.tasks.model_gateway import TaskModelGateway
from test_second_brain_model_gateway import context as gateway_context, Upstream


def context():
    ctx = gateway_context()
    ctx.task.id = uuid4()
    ctx.task.revision = 1
    ctx.claim.task_revision = 1
    ctx.claim.attempt_id = uuid4()
    return ctx


class HermesResultTests(unittest.IsolatedAsyncioTestCase):
    def result(self, **changes):
        return HermesTaskResult.model_validate({
            "status":"succeeded","answer":"An evidence-backed answer.","evidence":[],
            "completed_operation_refs":[],"uncertainties":[],
            "speech":[{"text":changes.get("answer", "An evidence-backed answer.")}],**changes,
        })

    async def test_completion_claims_require_accepted_durable_receipts(self):
        ctx = context()
        ctx.tasks.pool = None
        bridge = SimpleNamespace(verified_operation=AsyncMock(
            side_effect=TaskError("task_result_operation_unconfirmed","Not accepted")))
        with self.assertRaises(TaskError):
            await self.result(completed_operation_refs=["op_fixture"]).verified(ctx,bridge)
        bridge.verified_operation.assert_awaited_once_with("op_fixture")

    async def test_changed_or_missing_source_cannot_support_a_result(self):
        ctx = context()
        citation = {"event_id":uuid4(),"revision":1}
        for source in (
            SimpleNamespace(availability="deleted",is_current=True,revision=1),
            SimpleNamespace(availability="available",is_current=False,revision=1),
            SimpleNamespace(availability="available",is_current=True,revision=2),
        ):
            with self.subTest(source=source), patch(
                "app.tasks.hermes_result.ContextRepository",
                return_value=SimpleNamespace(source_for_claim=AsyncMock(return_value=source)),
            ):
                with self.assertRaises(TaskError):
                    await self.result(evidence=[citation]).verified(ctx,SimpleNamespace())

    async def test_waiting_for_input_is_not_reported_as_success(self):
        ctx = context()
        ctx.tasks.pool = None
        output = await self.result(status="waiting_for_input",answer="Please show the actual image.",
                                   uncertainties=["No original pixels are available."]).verified(ctx,SimpleNamespace())
        self.assertEqual(output.status,TaskStatus.WAITING)
        self.assertIn("No original",output.payload["uncertainties"][0])

    async def test_duplicate_completion_reference_is_rejected(self):
        with self.assertRaises(TaskError):
            await self.result(completed_operation_refs=["op_same","op_same"]).verified(context(),SimpleNamespace())

    def test_native_result_schema_requires_all_declared_fields(self):
        schema = HermesTaskResult.model_json_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]),set(schema["properties"]))
        with self.assertRaises(ValidationError):
            HermesTaskResult.model_validate({"status":"succeeded","answer":"Trust me."})

    def test_paused_schema_is_a_real_stop_not_an_unfinished_progress_update(self):
        schema = HermesTaskResult.model_json_schema()["properties"]["status"]
        self.assertEqual(set(schema["enum"]), {"succeeded", "waiting_for_input", "paused", "failed"})
        self.assertIn("established blocker", schema["description"])
        self.assertIn("not progress", schema["description"])
        self.assertIn("commands that can still be observed", schema["description"])

    def test_worker_distinguishes_read_only_observation_from_repeating_effects(self):
        self.assertIn("final structured result ends this worker run", WORKER_SYSTEM_PROMPT)
        self.assertIn("Checking the existing operation's status", WORKER_SYSTEM_PROMPT)
        self.assertIn("resubmission: use those read-only steps", WORKER_SYSTEM_PROMPT)
        self.assertIn("do not invent an expired deadline", WORKER_SYSTEM_PROMPT)

    async def test_a_genuinely_paused_result_remains_representable_without_automatic_replay(self):
        ctx = context()
        ctx.tasks.pool = None
        output = await self.result(status="paused", answer="The application outcome remains unknown.",
            uncertainties=["Read-only reconciliation is unavailable."]).verified(ctx, SimpleNamespace())
        self.assertEqual(output.status, TaskStatus.PAUSED)
        self.assertEqual(output.payload["verified_operations"], [])

    def test_native_paths_must_be_explicit_absolute_operator_configuration(self):
        with self.assertRaises(ValueError):
            HermesRuntimeConfiguration(Path("python"),Path("/tmp/private-worker"))
        with self.assertRaises(ValueError):
            HermesRuntimeConfiguration(Path("/bin/python"),Path("relative-work"))

    def test_native_compression_configuration_is_bounded_and_default_preserving(self):
        base = {"python":Path("/bin/python"),"work_root":Path("/tmp/private-worker")}
        default = HermesRuntimeConfiguration(**base)
        self.assertIsNone(default.context_tokens)
        self.assertEqual(default.compression_tail_messages,20)
        for changed in ({"context_tokens":True},{"context_tokens":65535},{"context_tokens":2000001},
                        {"compression_tail_messages":True},{"compression_tail_messages":2},
                        {"compression_tail_messages":101}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                HermesRuntimeConfiguration(**base,**changed)
        configured = HermesRuntimeConfiguration(**base,context_tokens=65536,compression_tail_messages=3)
        self.assertEqual(configured.context_tokens,65536)

    async def test_gateway_validates_images_inside_native_multimodal_tool_outputs(self):
        url = "data:image/jpeg;base64,c3ludGhldGlj"
        validate = AsyncMock()
        gateway = TaskModelGateway(context(),Upstream(),images={url:validate})
        body = {"model":"fixture-model","input":[{"type":"function_call_output","call_id":"call_fixture",
                "output":[{"type":"input_text","text":"Original fixture pixels follow."},
                          {"type":"input_image","image_url":url}]}]}
        [event async for event in gateway.stream(body)]
        self.assertEqual(validate.await_count,3)


class HermesTransportContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_required_task_output_format_is_injected_after_native_preflight(self):
        ctx, upstream = context(), Upstream()
        schema = {"format":{"type":"json_schema","name":"task","strict":True,"schema":{"type":"object"}}}
        gateway = TaskModelGateway(ctx,upstream,response_text=schema)
        [value async for value in gateway.stream({"model":"fixture-model","input":"Synthetic"})]
        self.assertEqual(upstream.count_calls[0]["text"],schema)
        self.assertEqual(upstream.requests[0]["text"],schema)
        auxiliary = Upstream()
        gateway = TaskModelGateway(context(),auxiliary,response_text=schema)
        [value async for value in gateway.stream({"model":"fixture-model","input":"Summarize synthetic history."},auxiliary=True)]
        self.assertNotIn("text",auxiliary.requests[0])
        self.assertEqual(ctx.checkpoint.await_args.args[0]["purpose"],"task")
        self.assertEqual(gateway.context.checkpoint.await_args.args[0]["purpose"],"compression")
        gateway.context.reserve.assert_awaited_once()
        gateway.context.settle.assert_awaited_once()

    async def test_internal_deltas_do_not_each_take_a_database_transaction(self):
        ctx, upstream = context(), Upstream()
        from test_second_brain_model_gateway import frame
        upstream.sequence = [upstream.sequence[0], *[
            frame({"type":"response.output_text.delta","delta":"x"}) for _ in range(100)
        ], upstream.sequence[-1]]
        gateway = TaskModelGateway(ctx,upstream)
        [value async for value in gateway.stream({"model":"fixture-model","input":"Synthetic"})]
        self.assertLess(ctx.tasks.context_problem.await_count,10)
        ctx.settle.assert_awaited_once()
