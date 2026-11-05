"""Task-bound model transport, not an agent loop.

Hermes has no upstream credential. Its configured Responses endpoint reaches
this gateway. Authorization and budget failures cannot be swallowed by native
observer callbacks. The parent retains the ledger and exact source resolvers.
"""
from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass
import hmac
import hashlib
import json
from math import isfinite
import re
import secrets
from typing import Protocol
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.images import ImageUnavailableError
from app.memory.jobs.models import normalize_job_json_object
from app.tasks.models import BudgetExceeded, ClaimLost, TaskError, json_text
from app.tasks.runner import WorkerContext


class ModelGatewayError(Exception):
    def __init__(self, code: str, *, http_status=None, parameter=None):
        self.code = code
        self.http_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        # Keep a structural API field path, never a provider message or value.
        self.parameter = parameter if (isinstance(parameter, str) and len(parameter) <= 200
            and re.fullmatch(r"(?:model|input|tools|text|reasoning|max_output_tokens|instructions|truncation)"
                             r"(?:\[[0-9]+\]|\.[A-Za-z_][A-Za-z_0-9]*){0,12}", parameter)) else None
        super().__init__(code)


@dataclass(frozen=True)
class ModelEvent:
    data: dict
    wire: bytes


class ResponsesUpstream(Protocol):
    async def count(self, payload: dict) -> int: ...
    def events(self, payload: dict) -> AsyncIterator[ModelEvent]: ...


class OpenAIResponsesHTTP:
    """Fixed upstream, parent-owned credentials, no SDK retries or redirects."""

    def __init__(self, api_key: str, *, timeout_seconds=180, transport=None):
        if not api_key or not 1 <= timeout_seconds <= 600:
            raise ValueError("A provider credential and bounded timeout are required")
        self.client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            headers={"Authorization": "Bearer " + api_key},
            timeout=timeout_seconds, follow_redirects=False, trust_env=False,
            transport=transport,
        )

    async def aclose(self):
        await self.client.aclose()

    @staticmethod
    def rejection(code, status, body):
        parameter = None
        if len(body) <= 64000:
            try:
                data = json.loads(body)
                error = data.get("error") if isinstance(data, dict) else None
                if isinstance(error, dict):
                    parameter = error.get("param")
            except (ValueError, TypeError):
                pass
        return ModelGatewayError(code, http_status=status, parameter=parameter)

    async def count(self, payload):
        # This is the provider's multimodal counter, not characters / tokens.
        fields = ("model", "input", "instructions", "tools", "tool_choice", "parallel_tool_calls",
                  "reasoning", "text", "truncation")
        response = await self.client.post("responses/input_tokens", json={
            key: payload[key] for key in fields if key in payload
        })
        if response.status_code != 200 or len(response.content) > 64000:
            raise self.rejection("model_input_count_unavailable", response.status_code, response.content)
        try:
            result = response.json()
            value = result["input_tokens"]
        except (ValueError, KeyError, TypeError) as error:
            raise ModelGatewayError("model_input_count_invalid") from error
        if result.get("object") != "response.input_tokens" or type(value) is not int or not 0 <= value <= 1000000000:
            raise ModelGatewayError("model_input_count_invalid")
        return value

    async def events(self, payload):
        async with self.client.stream("POST", "responses", json=payload) as response:
            if response.status_code != 200:
                # The provider might have accepted work despite a failed HTTP
                # exchange. Do not refund or automatically repeat the request.
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > 64000:
                        body.clear()
                        break
                    body.extend(chunk)
                raise self.rejection("model_provider_exchange_failed", response.status_code, body)
            lines, size = [], 0
            async for line in response.aiter_lines():
                size += len(line.encode("utf-8")) + 1
                if size > 4000000:
                    raise ModelGatewayError("model_provider_event_too_large")
                if line:
                    lines.append(line)
                    continue
                if not lines:
                    continue
                data = "\n".join(value[5:].lstrip(" ") for value in lines if value.startswith("data:"))
                wire = ("\n".join(lines) + "\n\n").encode()
                lines, size = [], 0
                if not data or data == "[DONE]":
                    continue
                try:
                    value = json.loads(data)
                except ValueError as error:
                    raise ModelGatewayError("model_provider_event_invalid") from error
                if not isinstance(value, dict):
                    raise ModelGatewayError("model_provider_event_invalid")
                yield ModelEvent(value, wire)
            if lines:
                raise ModelGatewayError("model_provider_stream_truncated")


class TaskModelGateway:
    """One authenticated gateway per exact task attempt.

    Only a trusted worker registers schemas and image revalidators. A model
    request cannot select another owner/model, provider state, built-in tool,
    remote URL, or file. All requests, including native retries and auxiliary
    requests, must pass through this admission boundary.
    """

    def __init__(self, context: WorkerContext, upstream: ResponsesUpstream, *, tools=(),
                 images: dict[str, Callable[[], Awaitable[None]]] | None = None,
                 run_id: str | None = None, max_requests=128, max_body_bytes=24000000,
                 max_stream_bytes=16000000, managed_runtime=False,
                 response_text=None, fence_poll_seconds=0.5, context_guard=None):
        for value, lower, upper in ((max_requests, 1, 1000), (max_body_bytes, 2000, 64000000),
                                    (max_stream_bytes, 2000, 64000000)):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("Invalid gateway bound")
        if isinstance(fence_poll_seconds,bool) or not isfinite(fence_poll_seconds) or not 0 < fence_poll_seconds <= 60:
            raise ValueError("Invalid streamed-output fence interval")
        self.fence_poll_seconds = fence_poll_seconds
        self.response_text = deepcopy(response_text)
        self.context_guard = context_guard
        self.context, self.upstream = context, upstream
        self.tools = {item["name"]: deepcopy(item) for item in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Duplicate gateway tool schema")
        self.images = dict(images or {})
        if any(not key.startswith("data:image/") or not callable(value) for key, value in self.images.items()):
            raise ValueError("Images require original inline bytes and a trusted revalidator")
        self.run_id = run_id or "hermes:" + uuid4().hex
        self.max_requests, self.max_body_bytes, self.max_stream_bytes = max_requests, max_body_bytes, max_stream_bytes
        self.managed_runtime = managed_runtime
        self.image_metadata = {}
        self._tool_calls = set()
        self.tool_receipts = []
        self.diagnostics = []
        self.token = secrets.token_urlsafe(48)
        self.auxiliary_token = secrets.token_urlsafe(48)
        self._lock = asyncio.Lock()
        self._requests = 0
        self._closed = False
        self._uncertain = False
        self._active: asyncio.Task | None = None

    def record_failure(self, stage, error):
        # Operator-only metadata for the owned run receipt. No request, response
        # body, exception message, tool arguments or credentials are retained.
        diagnostic = {"stage": stage, "error_type": type(error).__name__}
        if isinstance(error, (ModelGatewayError, TaskError, ClaimLost, BudgetExceeded)):
            diagnostic["code"] = getattr(error, "code", "model_request_stopped")
        if isinstance(error, ModelGatewayError):
            diagnostic.update(http_status=error.http_status, parameter=error.parameter)
        self.diagnostics.append(diagnostic)
        del self.diagnostics[:-8]

    def close(self):
        """Revoke admission and local I/O, not claim that remote work stopped."""
        self._closed = True
        if self._active is not None and self._active is not asyncio.current_task():
            self._active.cancel()
        for call in tuple(self._tool_calls):
            if call is not asyncio.current_task():
                call.cancel()

    async def check(self):
        if self._closed or self._uncertain:
            raise ModelGatewayError("model_gateway_closed_or_uncertain")
        tasks, claim = self.context.tasks, self.context.claim
        async with tasks.transaction(claim.task.owner_id) as connection:
            task, attempt = await tasks.lock_claim(connection, claim)
            if self.managed_runtime and attempt["provider_handle"] != self.run_id:
                raise ClaimLost()
            if await tasks.context_problem(connection, task):
                raise ClaimLost()
            if self.context_guard is not None:
                await self.context_guard(connection)

    def _request_json(self, body):
        # Multimodal transport has its own explicit bound. The default 256 KiB
        # database/job payload limit must not reject authorized image history.
        # Registered JSON Schemas also nest more deeply than persisted job data.
        # This transport-only bound does not change any memory/tool payload limit.
        normalized = normalize_job_json_object(body, label="model transport",
            max_bytes=self.max_body_bytes, max_depth=32)
        return json.dumps(normalized, sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    def prepare(self, body, *, auxiliary=False):
        try:
            if not isinstance(body, dict) or len(self._request_json(body).encode()) > self.max_body_bytes:
                raise ValueError()
            payload = deepcopy(body)
            if auxiliary and payload.get("tools"):
                raise ValueError()
            if not auxiliary and self.response_text is not None:
                payload["text"] = deepcopy(self.response_text)
            allowed = {"model", "input", "instructions", "tools", "tool_choice", "parallel_tool_calls",
                       "max_output_tokens", "reasoning", "text", "include", "store", "stream",
                       "temperature", "top_p", "background", "truncation", "prompt_cache_key"}
            if set(payload) - allowed or payload.get("model") != self.context.task.model:
                raise ValueError()
            if payload.get("store", False) is not False or payload.get("background", False) is not False:
                raise ValueError()
            if payload.get("truncation", "disabled") != "disabled":
                raise ValueError()
            output = payload.get("max_output_tokens", self.context.task.budget.request_output_tokens)
            if type(output) is not int or not 1 <= output <= self.context.task.budget.request_output_tokens:
                raise ValueError()
            payload.update(max_output_tokens=output, stream=True, store=False, background=False, truncation="disabled")
            definitions = payload.get("tools", [])
            if not isinstance(definitions, list):
                raise ValueError()
            seen = set()
            for tool in definitions:
                if not isinstance(tool, dict) or tool.get("type") != "function":
                    raise ValueError()
                name = tool.get("name")
                expected = self.tools.get(name)
                if (expected is None or name in seen or not isinstance(tool.get("parameters"), dict)
                        or set(tool) - {"type", "name", "description", "parameters", "strict"}):
                    raise ValueError()
                seen.add(name)
                # Hermes normalizes JSON Schema null unions for its own catalog.
                # Native schema/description text is never authority: forward the
                # exact registered Jarvis definition, not the rewritten copy.
                tool.update(parameters=deepcopy(expected["parameters"]),
                            description=expected["description"], strict=False)
            used_images = set()
            def parts(content):
                if not isinstance(content, list):
                    raise ValueError()
                for part in content:
                    if not isinstance(part, dict):
                        raise ValueError()
                    if part.get("type") == "input_image":
                        url = part.get("image_url")
                        if not isinstance(url, str) or url not in self.images or set(part) - {"type","image_url","detail"}:
                            raise ValueError()
                        used_images.add(url)
                    elif part.get("type") not in {"input_text", "output_text", "refusal"}:
                        raise ValueError()
            items = payload.get("input")
            if isinstance(items, str):
                items = [{"role": "user", "content": items}]
            if not isinstance(items, list):
                raise ValueError()
            for item in items:
                kind = item.get("type", "message") if isinstance(item, dict) else None
                if kind == "message":
                    if item.get("role") not in {"user", "assistant", "system", "developer"}:
                        raise ValueError()
                    content = item.get("content")
                    if isinstance(content, str):
                        continue
                    parts(content)
                elif kind not in {"function_call", "function_call_output", "reasoning"}:
                    raise ValueError()
                elif kind == "function_call_output" and not isinstance(item.get("output"), str):
                    parts(item.get("output"))
            # Registered schemas and response formatting can enlarge the input.
            # Validate the final envelope before counting or submitting it too.
            self._request_json(payload)
            return payload, used_images
        except (ValueError, TypeError, KeyError) as error:
            raise ModelGatewayError("model_request_not_allowed") from error

    async def revalidate(self, used_images):
        await self.check()
        for url in used_images:
            await self.images[url]()

    async def stream(self, body, *, auxiliary=False):
        if self._lock.locked():
            raise ModelGatewayError("model_request_already_active")
        async with self._lock:
            self._active = asyncio.current_task()
            submitted = False
            settled = False
            try:
                await self.check()
                if self._requests >= self.max_requests:
                    raise ModelGatewayError("model_request_limit_reached")
                self._requests += 1
                payload, used_images = self.prepare(body, auxiliary=auxiliary)
                await self.revalidate(used_images)
                count = await self.upstream.count(payload)
                if type(count) is not int or not 0 <= count <= 1000000000:
                    raise ModelGatewayError("model_input_count_invalid")
                key = "model-request:" + uuid4().hex
                reservation = await self.context.reserve(
                    key, input_tokens=count, output_tokens=payload["max_output_tokens"],
                    output_is_ceiling=True,
                )
                # Reserve only what is still available after settled and unknown
                # usage, under the ledger's owner lock. Never release uncertain
                # charges or send the native runtime's larger original ceiling.
                allocated = reservation["output_tokens"]
                if type(allocated) is not int or not 1 <= allocated <= payload["max_output_tokens"]:
                    raise ModelGatewayError("model_output_reservation_invalid")
                payload["max_output_tokens"] = allocated
                # The count call is also data egress. Recheck after its round trip
                # so a deletion or task correction cannot authorize inference.
                await self.revalidate(used_images)
                if self.managed_runtime:
                    await self.context.budgets.submit(self.context.claim,reservation["id"],
                                                       parent_provider_handle=self.run_id)
                else:
                    await self.context.dispatch(reservation["id"])
                submitted = True
                size, response_id = 0, None
                loop = asyncio.get_running_loop()
                next_fence_check = 0
                async with aclosing(self.upstream.events(payload)) as upstream_events:
                    async for event in upstream_events:
                        size += len(event.wire)
                        if size > self.max_stream_bytes:
                            raise ModelGatewayError("model_provider_stream_too_large")
                        kind = event.data.get("type")
                        value = event.data.get("response")
                        if kind == "response.created":
                            await self.check()
                            response_id = self._response_id(value)
                            await self.context.checkpoint({
                                "format": "jarvis.hermes.model_request.v1", "native_run": self.run_id,
                                "operation_key": key,
                                "request_digest": hashlib.sha256(self._request_json(payload).encode()).hexdigest(),
                                "reservation_id": str(reservation["id"]), "response_id": response_id,
                                "phase": "response_created", "purpose": "compression" if auxiliary else "task",
                                "images": [self.image_metadata.get(url, {"registered":True}) for url in sorted(used_images)],
                            }, provider_handle=None if self.managed_runtime else "openai_response:" + response_id)
                            next_fence_check = loop.time() + self.fence_poll_seconds
                        if kind in {"response.completed", "response.incomplete", "response.failed"}:
                            terminal_id = self._response_id(value)
                            if (response_id is None or terminal_id != response_id
                                    or value.get("status") != kind.removeprefix("response.")):
                                raise ModelGatewayError("model_provider_response_mismatch")
                            usage = value.get("usage")
                            if (not isinstance(usage, dict) or any(
                                type(usage.get(field)) is not int or not 0 <= usage[field] <= 1000000000
                                for field in ("input_tokens", "output_tokens")
                            )):
                                raise ModelGatewayError("model_usage_unknown")
                            await self.context.settle(
                                reservation["id"], input_tokens=usage["input_tokens"],
                                output_tokens=usage["output_tokens"], tool_steps=0, provider_terminal=not self.managed_runtime,
                            )
                            settled = True
                            if usage["input_tokens"] > count or usage["output_tokens"] > payload["max_output_tokens"]:
                                self._closed = True
                                raise ModelGatewayError("model_usage_exceeded_reservation")
                            await self.revalidate(used_images)
                            yield event.wire
                            return
                        if self._closed or self._uncertain or loop.time() >= next_fence_check:
                            await self.check()
                            next_fence_check = loop.time() + self.fence_poll_seconds
                        yield event.wire
                raise ModelGatewayError("model_provider_stream_incomplete")
            finally:
                # A cancelled local request is not a provider cancellation
                # receipt. Keep its reservation charged and disallow reissue.
                if submitted and not settled:
                    self._uncertain = True
                self._active = None

    @staticmethod
    def _response_id(value):
        identifier = value.get("id") if isinstance(value, dict) else None
        if not isinstance(identifier, str) or not 1 <= len(identifier) <= 512 or "\x00" in identifier:
            raise ModelGatewayError("model_provider_response_invalid")
        return identifier

    async def register_image(self, resolved):
        if (resolved.reference.owner_id != self.context.task.owner_id or resolved.revalidate is None
                or not resolved.image_bytes or len(resolved.image_bytes)>12000000):
            raise ModelGatewayError("task_image_not_allowed")
        await resolved.revalidate()
        await self.check()
        url = "data:" + resolved.reference.content_type + ";base64," + base64.b64encode(resolved.image_bytes).decode("ascii")
        if url not in self.images and sum(len(key) for key in self.images)+len(url)>20000000:
            raise ModelGatewayError("task_image_context_limit")
        previous = self.images.get(url)
        async def validate():
            if previous is not None:
                await previous()
            await resolved.revalidate()
        self.images[url] = validate
        self.image_metadata[url] = resolved.reference.to_metadata()
        return url

    def application(self, *, tool_bridge=None):
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        if tool_bridge is not None:
            if tool_bridge.context.claim != self.context.claim:
                raise ValueError("The native tool transport requires this exact task attempt")

            @app.post("/v1/task/tools/{name}")
            async def scoped_tool(name: str, request: Request):
                supplied = request.headers.get("authorization","").encode()
                if not hmac.compare_digest(supplied,("Bearer "+self.token).encode()):
                    return JSONResponse({"status":"failed","code":"unauthorized"},status_code=401)
                call = asyncio.current_task()
                self._tool_calls.add(call)
                try:
                    chunks, length = [], 0
                    async for chunk in request.stream():
                        length += len(chunk)
                        if length>128000:
                            return JSONResponse({"status":"failed","code":"request_too_large"},status_code=413)
                        chunks.append(chunk)
                    body = json.loads(b"".join(chunks))
                    if not isinstance(body,dict) or set(body)!={"arguments","call_id"}:
                        raise ValueError("Invalid scoped tool envelope")
                    await self.check()
                    result = await tool_bridge.invoke(name,body["arguments"],provider_call_id=body["call_id"])
                    self.tool_receipts.append({"name":name,"call_id":body["call_id"],"status":result.get("status"),
                                               "operation":result.get("task_operation")})
                    tool = tool_bridge.registry.get(name)
                    if (tool and tool.image_resolver is not None and result.get("status")=="success"
                            and result.get("image_status")=="selected"):
                        resolved = await tool_bridge.resolve_image(name,result)
                        url = await self.register_image(resolved)
                        summary = json_text(result | {"image_status":"supplied_to_native_runtime"})
                        return {"_multimodal":True,"text_summary":summary,"content":[
                            {"type":"text","text":summary},
                            {"type":"image_url","image_url":{"url":url,"detail":"high"}},
                        ]}
                    return result
                except (ValueError,TypeError):
                    return JSONResponse({"status":"failed","code":"invalid_tool_input"},status_code=400)
                except (ClaimLost,TaskError,ModelGatewayError,ImageUnavailableError) as error:
                    return JSONResponse({"status":"failed","code":error.code,
                                         "meaning":"The operation or image is not confirmed; do not repeat effects."},status_code=409)
                except Exception:
                    return JSONResponse({"status":"failed","code":"task_tool_transport_failed",
                                         "meaning":"The operation is not confirmed."},status_code=500)
                finally:
                    self._tool_calls.discard(call)

        @app.post("/v1/responses")
        async def responses(request: Request):
            supplied = request.headers.get("authorization", "")
            primary = hmac.compare_digest(supplied.encode(), ("Bearer " + self.token).encode())
            auxiliary = hmac.compare_digest(supplied.encode(), ("Bearer " + self.auxiliary_token).encode())
            if not primary and not auxiliary:
                return JSONResponse({"error": {"code": "unauthorized"}}, status_code=401)
            try:
                chunks, length = [], 0
                async for chunk in request.stream():
                    length += len(chunk)
                    if length > self.max_body_bytes:
                        return JSONResponse({"error": {"code": "request_too_large"}}, status_code=413)
                    chunks.append(chunk)
                body = json.loads(b"".join(chunks))
                self.prepare(body, auxiliary=auxiliary)
            except (ValueError, ModelGatewayError) as error:
                self.record_failure("request_validation", error)
                return JSONResponse({"error": {"code": "model_request_not_allowed"}}, status_code=400)

            async def events():
                try:
                    async for frame in self.stream(body, auxiliary=auxiliary):
                        yield frame
                except (ClaimLost, BudgetExceeded, TaskError, ModelGatewayError) as error:
                    self.record_failure("model_stream", error)
                    code = getattr(error, "code", "model_request_stopped")
                    yield ("event: error\ndata: " + json.dumps({
                        "type": "error", "code": code,
                        "message": "The task model request stopped. Do not infer completion or retry effects.",
                    }) + "\n\n").encode()
                except Exception as error:
                    self.record_failure("model_stream", error)
                    yield b'event: error\ndata: {"type":"error","code":"model_transport_failed","message":"The provider outcome is not confirmed."}\n\n'

            return StreamingResponse(events(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-store"})

        return app
