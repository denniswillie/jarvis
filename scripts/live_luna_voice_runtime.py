"""Inert SINGLE-PASS client Luna sidecar, never native Responses delegation.

Wrap the existing production fixture composition without replacing session.work.
Core supplies app.realtime.live_router.LunaLiveRouter(session, client, observe=None).
The router owns exactly one responses.create(model="gpt-6-luna",
reasoning={"effort": "none"}, ...) per delegation, not a tool/model loop.

Suggested synchronous observe(event_dict) contract: emit the complete request
kwargs before create, the complete response.model_dump(mode="json") afterward,
and actual action/read results, each tagged with stage and delegation_id. No
specific event spelling is required here. Never supply headers/credentials or
private reasoning. JSONL and the readable timeline preserve all ordinary model
input/output, with credential/SDP fields elided, not summarized or truncated.
Missing observations are NOT proof of a successful single-pass route.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time

import live_hermes_voice_acceptance as harness


class RouterObserver:
    """Synchronous, private, byte-bounded sink; never changes router results."""

    def __init__(self, root, limits, trace=None):
        root = harness.private_root(root)
        self.limit, self.bytes, self.sequence = limits.max_trace_bytes, 0, 0
        self.trace = trace
        self.stream = self.text = None
        try:
            for attr, name in (("stream", "router-events.jsonl"), ("text", "router-timeline.txt")):
                fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                setattr(self, attr, os.fdopen(fd, "w", encoding="utf-8"))
        except BaseException:
            self.close()
            raise

    def __call__(self, event):
        from evals.responsive_voice.model_wire_trace import readable
        harness.require(isinstance(event, dict), "router_observation_must_be_dict")
        self.sequence += 1
        row = dict(sequence=self.sequence, utc=datetime.now(UTC).isoformat(),
                   monotonic_s=time.monotonic(), event=harness.scrub(event))
        # No default=str: a non-JSON SDK object must not silently lose fields.
        encoded = json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"
        rendered = "\n".join(readable(row)) + "\n\n"
        self.bytes += len(encoded.encode()) + len(rendered.encode())
        harness.require(self.bytes <= self.limit, "router_trace_budget_exceeded")
        self.stream.write(encoded)
        self.stream.flush()
        self.text.write(rendered)
        self.text.flush()
        if self.trace is not None:
            self.trace.record("luna_router_observe", payload=row["event"], source="luna")

    def close(self):
        for stream in (self.stream, self.text):
            if stream is not None:
                stream.close()


@asynccontextmanager
async def application(*, fixture, session_id, websocket, offer, client_capabilities,
                      limits, artifact_dir, settings):
    """Construct only; the original acceptance harness owns run/stop/SQL cleanup."""
    from openai import AsyncOpenAI
    from app.realtime.live_router import LunaLiveRouter
    from live_hermes_voice_runtime import application as hermes_application

    observer = RouterObserver(artifact_dir, limits, getattr(websocket, "trace", None))
    client = None
    try:
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, max_retries=0, timeout=8.0)
        async with hermes_application(fixture=fixture, session_id=session_id, websocket=websocket,
                offer=offer, client_capabilities=client_capabilities, limits=limits,
                artifact_dir=artifact_dir, settings=settings) as app:
            harness.require(app.session.work is not None and app.session.delegation_handler is app.session.work,
                            "existing_hermes_bridge_required")
            app.session.delegation_handler = LunaLiveRouter(app.session, client, observe=observer)
            harness.private_write(Path(artifact_dir) / "luna-runtime-composition.json", dict(
                format="jarvis.live_luna_voice_runtime.v1", delegation_type="client",
                delegation_handler=type(app.session.delegation_handler).__name__,
                work_bridge=type(app.session.work).__name__, expected_model="gpt-6-luna",
                expected_reasoning_effort="none", expected_requests_per_delegation=1,
                client_max_retries=0, client_timeout_seconds=8.0,
                serving_activation=False, paid_execution_qualified=False,
                observer_contract="synchronous observe(event_dict); complete request, response, action results"))
            yield app
    finally:
        try:
            if client is not None:
                await client.close()
        finally:
            observer.close()
