import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx

from app.tasks.model_gateway import ModelEvent, ModelGatewayError, OpenAIResponsesHTTP, TaskModelGateway
from app.tasks.models import ClaimLost


def frame(value):
    return ModelEvent(value, ("data: " + json.dumps(value) + "\n\n").encode())


class Upstream:
    def __init__(self):
        self.count_calls, self.requests = [], []
        self.input_tokens = 11
        self.sequence = [
            frame({"type":"response.created","response":{"id":"resp_fixture","status":"in_progress"}}),
            frame({"type":"response.output_text.delta","delta":"Use parentheses."}),
            frame({"type":"response.completed","response":{"id":"resp_fixture","status":"completed",
                "usage":{"input_tokens":11,"output_tokens":7}}}),
        ]
        self.before_count = self.before_events = None

    async def count(self, payload):
        self.count_calls.append(payload)
        if self.before_count:
            await self.before_count()
        return self.input_tokens

    async def events(self, payload):
        self.requests.append(payload)
        if self.before_events:
            await self.before_events()
        for event in self.sequence:
            yield event


def context():
    task = SimpleNamespace(owner_id="synthetic-owner",model="fixture-model",
                           budget=SimpleNamespace(request_output_tokens=100))
    @asynccontextmanager
    async def transaction(owner):
        assert owner == task.owner_id
        yield object()
    tasks = SimpleNamespace(transaction=transaction,lock_claim=AsyncMock(return_value=(task,None)),
                            context_problem=AsyncMock(return_value=False))
    async def reserve(*args,**kwargs):
        return {"id":uuid4(), "output_tokens":kwargs["output_tokens"]}
    return SimpleNamespace(
        task=task,claim=SimpleNamespace(task=task),tasks=tasks,
        reserve=AsyncMock(side_effect=reserve),dispatch=AsyncMock(),settle=AsyncMock(),checkpoint=AsyncMock(),
    )


class ModelGatewayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context, self.upstream = context(), Upstream()
        self.tool = {"name":"read_task_context_source","description":"Read exact source",
                     "parameters":{"type":"object","properties":{"id":{"type":"string"}}}}
        self.gateway = TaskModelGateway(self.context,self.upstream,tools=(self.tool,))
        self.payload = {"model":"fixture-model","input":"Check the visible midpoint.","max_output_tokens":100}

    async def collect(self,payload=None,gateway=None):
        return b"".join([chunk async for chunk in (gateway or self.gateway).stream(payload or self.payload)])

    async def test_counts_before_reserving_dispatches_once_and_settles_actual_usage(self):
        result = await self.collect()
        self.assertIn(b"Use parentheses",result)
        self.context.reserve.assert_awaited_once()
        self.assertEqual(self.context.reserve.await_args.kwargs,
                         {"input_tokens":11,"output_tokens":100,"output_is_ceiling":True})
        self.context.dispatch.assert_awaited_once()
        self.assertEqual(self.context.settle.await_args.kwargs,
                         {"input_tokens":11,"output_tokens":7,"tool_steps":0,"provider_terminal":True})
        self.assertEqual(self.context.checkpoint.await_args.kwargs["provider_handle"],"openai_response:resp_fixture")
        self.assertNotIn("input",self.context.checkpoint.await_args.args[0])
        body = self.upstream.requests[0]
        self.assertFalse(body["store"])
        self.assertFalse(body["background"])
        self.assertTrue(body["stream"])
        self.assertEqual(body["truncation"],"disabled")

    async def test_only_atomically_allocated_output_bound_reaches_the_provider(self):
        self.context.reserve.side_effect = None
        self.context.reserve.return_value = {"id":uuid4(), "output_tokens":41}
        await self.collect()
        self.assertEqual(self.upstream.requests[0]["max_output_tokens"], 41)
        self.assertEqual(self.payload["max_output_tokens"], 100)

    async def test_invalid_output_allocation_cannot_be_dispatched(self):
        self.context.reserve.side_effect = None
        for allocation in (0, 101, True, "41"):
            self.context.reserve.return_value = {"id":uuid4(), "output_tokens":allocation}
            with self.subTest(allocation=allocation), self.assertRaises(ModelGatewayError):
                await self.collect()
        self.context.dispatch.assert_not_awaited()
        self.assertFalse(self.upstream.requests)

    async def test_valid_next_request_has_new_reservation_and_no_implicit_replay(self):
        await self.collect()
        await self.collect()
        keys = [call.args[0] for call in self.context.reserve.await_args_list]
        self.assertEqual(len(set(keys)),2)
        self.assertEqual(len(self.upstream.requests),2)

    async def test_forbids_provider_state_remote_media_and_ungranted_native_tools(self):
        invalid = [
            {"model":"another-model"},{"store":True},{"background":True},{"previous_response_id":"resp_other"},
            {"conversation":"another-user"},{"truncation":"auto"},{"max_output_tokens":101},
            {"max_output_tokens":True},{"tools":[{"type":"web_search"}]},
            {"tools":[{"type":"function","name":"send_email","parameters":{}}]},
            {"tools":[{"type":"function",**self.tool,"parameters":[]}]},
            {"input":[{"role":"user","content":[{"type":"input_image","image_url":"https://example.com/private.jpg"}]}]},
            {"input":[{"role":"user","content":[{"type":"input_file","file_id":"file_foreign"}]}]},
            {"input":[{"type":"function_call_output","call_id":"call_one","output":[
                {"type":"input_image","image_url":"https://example.com/another.jpg"}]}]},
        ]
        for change in invalid:
            with self.subTest(change=change):
                with self.assertRaises(ModelGatewayError):
                    await self.collect(self.payload | change)
        self.assertFalse(self.upstream.count_calls)
        self.context.dispatch.assert_not_awaited()

    async def test_only_registered_function_schema_reaches_provider(self):
        await self.collect(self.payload | {"tools":[{"type":"function",**self.tool,"strict":False}]})
        self.assertEqual(self.upstream.requests[0]["tools"][0]["name"],self.tool["name"])

    async def test_native_schema_changes_cannot_replace_registered_authority(self):
        incoming = {"type":"function",**self.tool,"parameters":{"type":"object","properties":{}},
                    "description":"Untrusted changed instructions","strict":True}
        await self.collect(self.payload | {"tools":[incoming]})
        for request in (self.upstream.count_calls[0],self.upstream.requests[0]):
            self.assertEqual(request["tools"][0]["parameters"],self.tool["parameters"])
            self.assertEqual(request["tools"][0]["description"],self.tool["description"])
            self.assertFalse(request["tools"][0]["strict"])
        self.assertEqual(incoming["description"],"Untrusted changed instructions")

    async def test_registered_original_image_is_checked_before_network_and_terminal(self):
        url = "data:image/png;base64,c3ludGhldGlj"
        revalidate = AsyncMock()
        gateway = TaskModelGateway(self.context,self.upstream,images={url:revalidate})
        body = self.payload | {"input":[{"role":"user","content":[{"type":"input_image","image_url":url}]}]}
        await self.collect(body,gateway)
        self.assertEqual(revalidate.await_count,3)
        self.assertEqual(self.upstream.count_calls[0]["input"],self.upstream.requests[0]["input"])

    async def test_image_revoked_during_count_does_not_reach_inference(self):
        url = "data:image/png;base64,c3ludGhldGlj"
        gateway = TaskModelGateway(self.context,self.upstream,images={url:AsyncMock(side_effect=[None,ClaimLost()])})
        with self.assertRaises(ClaimLost):
            await self.collect(self.payload | {"input":[{"role":"user","content":[{"type":"input_image","image_url":url}]}]},gateway)
        self.context.dispatch.assert_not_awaited()
        self.assertFalse(self.upstream.requests)

    async def test_large_authorized_image_uses_transport_not_database_limit(self):
        from app.tasks.models import json_text
        url = "data:image/png;base64," + "a" * 350000
        revalidate = AsyncMock()
        gateway = TaskModelGateway(self.context, self.upstream, images={url: revalidate}, max_body_bytes=1000000)
        body = self.payload | {"input": [{"type": "function_call_output", "call_id": "call_large",
            "output": [{"type": "input_image", "image_url": url}]}]}
        with self.assertRaises(ValueError):
            json_text(body)  # The durable-storage limit is deliberately unchanged.
        await self.collect(body, gateway)
        self.assertEqual(revalidate.await_count, 3)
        self.assertEqual(self.upstream.requests[0]["input"], body["input"])
        checkpoint = self.context.checkpoint.await_args.args[0]
        self.assertEqual(len(checkpoint["request_digest"]), 64)
        self.assertNotIn(url, json.dumps(checkpoint))

    async def test_transport_limit_and_nonfinite_numbers_still_fail_before_egress(self):
        url = "data:image/png;base64," + "a" * 3000
        gateway = TaskModelGateway(self.context, self.upstream, images={url: AsyncMock()}, max_body_bytes=2000)
        for body in [self.payload | {"input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": url}]}]},
                self.payload | {"temperature": float("nan")}]:
            with self.assertRaises(ModelGatewayError):
                await self.collect(body, gateway)
        self.assertFalse(self.upstream.count_calls)

    async def test_registered_schema_expansion_cannot_bypass_final_transport_bound(self):
        tool = self.tool | {"description": "x" * 3000}
        gateway = TaskModelGateway(self.context, self.upstream, tools=(tool,), max_body_bytes=2000)
        body = self.payload | {"tools": [{"type": "function", **self.tool}]}
        with self.assertRaises(ModelGatewayError):
            await self.collect(body, gateway)
        self.assertFalse(self.upstream.count_calls)

    async def test_claim_lost_before_count_prevents_data_egress(self):
        self.context.tasks.context_problem.return_value = "deleted_source"
        with self.assertRaises(ClaimLost):
            await self.collect()
        self.assertFalse(self.upstream.count_calls)

    async def test_count_round_trip_cannot_bypass_a_task_correction(self):
        async def changed():
            self.context.tasks.context_problem.return_value = "corrected_source"
        self.upstream.before_count = changed
        with self.assertRaises(ClaimLost):
            await self.collect()
        self.assertFalse(self.upstream.requests)
        self.context.dispatch.assert_not_awaited()

    async def test_unknown_usage_and_truncated_stream_keep_full_reservation_and_block_reissue(self):
        for sequence in (self.upstream.sequence[:-1],[
            self.upstream.sequence[0],
            frame({"type":"response.completed","response":{"id":"resp_fixture","status":"completed","usage":None}}),
        ]):
            with self.subTest(sequence=sequence):
                upstream, ctx = Upstream(), context()
                upstream.sequence = sequence
                gateway = TaskModelGateway(ctx,upstream)
                with self.assertRaises(ModelGatewayError):
                    await self.collect(gateway=gateway)
                with self.assertRaises(ModelGatewayError):
                    await self.collect(gateway=gateway)
                self.assertEqual(len(upstream.requests),1)
                ctx.settle.assert_not_awaited()

    async def test_mismatched_or_nonterminal_receipt_cannot_settle_usage(self):
        for response in ({"id":"resp_another","status":"completed"},{"id":"resp_fixture","status":"in_progress"}):
            with self.subTest(response=response):
                upstream, ctx = Upstream(), context()
                upstream.sequence[-1] = frame({"type":"response.completed","response":response | {
                    "usage":{"input_tokens":11,"output_tokens":7}}})
                with self.assertRaises(ModelGatewayError):
                    await self.collect(gateway=TaskModelGateway(ctx,upstream))
                ctx.settle.assert_not_awaited()

    async def test_actual_overrun_is_accounted_but_prevents_another_request(self):
        self.upstream.input_tokens = 10
        with self.assertRaisesRegex(ModelGatewayError,"exceeded_reservation"):
            await self.collect()
        self.assertEqual(self.context.settle.await_args.kwargs["input_tokens"],11)
        with self.assertRaises(ModelGatewayError):
            await self.collect()
        self.assertEqual(len(self.upstream.requests),1)

    async def test_cancelled_io_is_not_a_remote_cancellation_receipt(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def pause():
            entered.set()
            await release.wait()
        self.upstream.before_events = pause
        work = asyncio.create_task(self.collect())
        await entered.wait()
        self.gateway.close()
        with self.assertRaises(asyncio.CancelledError):
            await work
        with self.assertRaises(ModelGatewayError):
            await self.collect()
        self.context.settle.assert_not_awaited()
        self.assertEqual(len(self.upstream.requests),1)

    async def test_parallel_request_cannot_skip_serial_per_attempt_admission(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def pause():
            entered.set()
            await release.wait()
        self.upstream.before_count = pause
        work = asyncio.create_task(self.collect())
        try:
            await entered.wait()
            with self.assertRaisesRegex(ModelGatewayError,"already_active"):
                await self.collect()
            release.set()
            await work
        finally:
            release.set()
            await asyncio.gather(work,return_exceptions=True)
        self.assertEqual(len(self.upstream.count_calls),1)

    async def test_request_limit_also_bounds_failed_preinference_attempts(self):
        gateway = TaskModelGateway(self.context,self.upstream,max_requests=1)
        with self.assertRaises(ModelGatewayError):
            await self.collect(self.payload | {"model":"not-allowed"},gateway)
        with self.assertRaisesRegex(ModelGatewayError,"limit_reached"):
            await self.collect(gateway=gateway)
        self.assertFalse(self.upstream.requests)

    async def test_terminal_usage_can_arrive_after_claim_loss_without_publishing_stale_answer(self):
        ctx, upstream = self.context,self.upstream
        class LateReceipt(Upstream):
            async def events(self,payload):
                yield self.sequence[0]
                ctx.tasks.context_problem.return_value = "newer_revision"
                yield self.sequence[-1]
        with self.assertRaises(ClaimLost):
            await self.collect(gateway=TaskModelGateway(ctx,LateReceipt()))
        ctx.settle.assert_awaited_once()

    async def test_http_boundary_requires_exact_ephemeral_token_and_exposes_no_other_routes(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.gateway.application()),base_url="http://fixture") as client:
            denied = await client.post("/v1/responses",json=self.payload)
            self.assertEqual(denied.status_code,401)
            headers = {"Authorization":"Bearer " + self.gateway.token}
            self.assertEqual((await client.post("/v1/chat/completions",headers=headers,json=self.payload)).status_code,404)
            allowed = await client.post("/v1/responses",headers=headers,json=self.payload)
            self.assertEqual(allowed.status_code,200)
            self.assertIn("response.completed",allowed.text)
        self.assertEqual(len(self.upstream.requests),1)

    async def test_http_errors_never_leak_provider_secrets(self):
        self.upstream.before_count = AsyncMock(side_effect=RuntimeError("secret-provider-key"))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.gateway.application()),base_url="http://fixture") as client:
            response = await client.post("/v1/responses",headers={"Authorization":"Bearer "+self.gateway.token},json=self.payload)
        self.assertNotIn("secret-provider-key",response.text)
        self.assertIn("model_transport_failed",response.text)


    async def test_stopping_before_terminal_receipt_closes_upstream_immediately(self):
        closed = asyncio.Event()
        class ClosingUpstream(Upstream):
            async def events(self,payload):
                try:
                    for value in self.sequence:
                        yield value
                finally:
                    closed.set()
        gateway = TaskModelGateway(self.context,ClosingUpstream())
        stream = gateway.stream(self.payload)
        await anext(stream)
        await stream.aclose()
        self.assertTrue(closed.is_set())
        self.context.settle.assert_not_awaited()
        with self.assertRaises(ModelGatewayError):
            await self.collect(gateway=gateway)


class OpenAIHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_provider_route_exact_counter_payload_and_stream_contract(self):
        requests = []
        def dispatch(request):
            requests.append(request)
            if request.url.path.endswith("/input_tokens"):
                return httpx.Response(200,json={"object":"response.input_tokens","input_tokens":11})
            return httpx.Response(200,content=b": heartbeat\n\n" + b"".join(x.wire for x in Upstream().sequence),
                                  headers={"Content-Type":"text/event-stream"})
        upstream = OpenAIResponsesHTTP("synthetic-secret",transport=httpx.MockTransport(dispatch))
        try:
            gateway = TaskModelGateway(context(),upstream)
            result = b"".join([frame async for frame in gateway.stream({"model":"fixture-model","input":"Synthetic"})])
        finally:
            await upstream.aclose()
        self.assertIn(b"response.completed",result)
        self.assertEqual([x.url.path for x in requests],["/v1/responses/input_tokens","/v1/responses"])
        self.assertTrue(all(x.url.host=="api.openai.com" for x in requests))
        self.assertTrue(all(x.headers["authorization"]=="Bearer synthetic-secret" for x in requests))
        self.assertNotIn("max_output_tokens",json.loads(requests[0].content))

    async def test_counter_and_stream_fail_closed_on_invalid_upstream_frames(self):
        bodies = [b"data: broken\n\n",b"data: []\n\n",b'data: {"type":"response.created"}']
        for body in bodies:
            with self.subTest(body=body):
                upstream = OpenAIResponsesHTTP("synthetic-secret",transport=httpx.MockTransport(
                    lambda request: httpx.Response(200,content=body)))
                try:
                    with self.assertRaises(ModelGatewayError):
                        await upstream.count({"model":"fixture-model","input":"x"})
                    with self.assertRaises(ModelGatewayError):
                        [event async for event in upstream.events({})]
                finally:
                    await upstream.aclose()
