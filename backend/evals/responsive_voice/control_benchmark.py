"""Controlled P2 backend timing, never network, provider, or hearing evidence.

Run with PYTHONPATH=backend and --output /absolute/report.json. The actual
Realtime receive loops, handler, ordered transport and tool executor run against
the existing isolated fixtures. No service, database or model is contacted.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta
import json
import logging
import math
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

TESTS = Path(__file__).resolve().parents[2] / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import test_realtime_image_tools as session_fixtures
import test_live_state_runtime as state_fixtures
from app.live_state.projection import select_voice_context
from app.live_state.service import LiveStateService


WORKLOADS = ("slow_tool", "slow_result_persistence", "slow_provider_write")


def distribution(values):
    valid = sorted(value for value in values
                   if isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value) and value >= 0)
    def quantile(fraction):
        return valid[max(0, math.ceil(len(valid) * fraction) - 1)] if valid else None
    return dict(count=len(valid), missing=len(values) - len(valid), p50_ms=quantile(.50),
                p95_ms=quantile(.95), maximum_ms=max(valid) if valid else None)


class Probe(logging.Handler):
    """Observe real redacted telemetry; enqueue and dispatch share monotonic()."""
    def __init__(self):
        super().__init__(logging.INFO)
        self.pending = {}

    def marker(self, origin, event_type):
        identity = "p2_" + uuid4().hex
        self.pending[identity] = dict(origin=origin, event_type=event_type,
            enqueued=time.monotonic(), started=asyncio.Event(), finished=asyncio.Event(),
            start=None, finish=None)
        return identity

    def emit(self, record):
        message = record.getMessage()
        if not message.startswith("realtime_trace "):
            return
        row = json.loads(message.removeprefix("realtime_trace "))
        item = self.pending.get(row.get("event_id") or row.get("client_event_id"))
        if item is None:
            return
        if row["event"] == "control_handler_started":
            item["start"] = row
            item["started"].set()
        elif row["event"] == "control_handler_finished":
            item["finish"] = row
            item["finished"].set()

    def sample(self, identity):
        item = self.pending[identity]
        start, finish = item["start"], item["finish"]
        handler_start = (start["dequeued_monotonic_s"] + start["decode_dispatch_ms"] / 1000
                         if start is not None else None)
        return dict(event_id=identity, origin=item["origin"], event_type=item["event_type"],
            enqueue_to_dispatch_ms=(max(0, handler_start - item["enqueued"]) * 1000
                                    if handler_start is not None else None),
            dequeue_to_dispatch_ms=start["decode_dispatch_ms"] if start else None,
            handler_ms=finish["duration_ms"] if finish else None,
            status=finish["status"] if finish else "unfinished_under_load",
            boundary="controlled_application_enqueue_not_network_ingress")


async def control_sample(workload, *, hold_ms, deadline_ms):
    fixture = session_fixtures.RealtimeImageToolTests()
    probe = Probe()
    logger = logging.getLogger("app.realtime.telemetry")
    old_level, old_propagate = logger.level, logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(probe)
    entered, release = asyncio.Event(), asyncio.Event()
    client_task = None
    initialized = False
    try:
        await fixture.asyncSetUp()
        initialized = True
        # Replace only the fixture receive pump, not production handling.
        fixture.receiver.cancel()
        await asyncio.gather(fixture.receiver, return_exceptions=True)
        async def recv():
            payload = await fixture.wire.incoming.get()
            fixture.wire.incoming.task_done()
            return json.dumps(payload)
        fixture.wire.recv = recv
        fixture.receiver = asyncio.create_task(fixture.session._openai_to_client_loop(),
                                               name="p2-real-provider-reader")
        await fixture.start()
        session = fixture.session
        # This gate selects the actual responsive shared handler/executor. It
        # does not pretend to qualify a native route's negotiation or media.
        session._voice_route_runtime = SimpleNamespace(
            before_session_close=AsyncMock(return_value=True), after_session_close=AsyncMock(return_value=True))
        clients = asyncio.Queue()
        async def client_messages():
            while True:
                payload = await clients.get()
                try:
                    yield json.dumps(payload)
                finally:
                    clients.task_done()
        session.websocket.iter_text = client_messages
        client_task = asyncio.create_task(session._client_to_openai_loop(), name="p2-real-client-reader")

        if workload == "slow_tool":
            async def work(_arguments):
                entered.set()
                await release.wait()
                return {"status": "success", "value": "synthetic operation completed"}
            fixture.ordinary.side_effect = work
        elif workload == "slow_result_persistence":
            finalize = fixture.executions.finalize_execution
            async def blocked_finalize(**kwargs):
                entered.set()
                await release.wait()
                return await finalize(**kwargs)
            fixture.executions.finalize_execution = blocked_finalize
        elif workload == "slow_provider_write":
            send = fixture.wire.send
            async def blocked_send(raw):
                event = json.loads(raw)
                if event.get("item", {}).get("type") == "function_call_output":
                    entered.set()
                    await release.wait()
                await send(raw)
            fixture.wire.send = blocked_send
        else:
            raise ValueError("Unknown controlled workload")

        fixture.call("synthetic-tool", name="ordinary_example", arguments={})
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.sleep(hold_ms / 1000)
        markers = []
        def provider_marker(event_type, **payload):
            identity = probe.marker("provider_control", event_type)
            markers.append(identity)
            fixture.wire.incoming.put_nowait(dict(type=event_type, event_id=identity, **payload))
            return identity
        provider_marker("input_audio_buffer.committed", item_id="synthetic-input")
        for _ in range(2):
            identity = probe.marker("client_control", "response.cancel")
            markers.append(identity)
            clients.put_nowait(dict(type="response.cancel", client_event_id=identity))
        try:
            await asyncio.wait_for(probe.pending[markers[1]]["started"].wait(), deadline_ms / 1000)
        except TimeoutError:
            pass
        # A slow client control write must not prevent provider completion from
        # being consumed by the independent provider reader.
        provider_marker("response.done", response={"id": "response-1", "status": "completed"})
        try:
            await asyncio.wait_for(asyncio.gather(*(probe.pending[key]["finished"].wait()
                for key in markers)), deadline_ms / 1000)
        except TimeoutError:
            pass
        # Freeze evidence BEFORE releasing slow work. Missing events are failures,
        # never zeros or measurements taken after artificially draining the load.
        samples = [probe.sample(identity) for identity in markers]
        blocked = not release.is_set()
        release.set()
        await fixture.until(lambda: not session._ordinary_tool_tasks)
        await asyncio.wait_for(clients.join(), 2)
        return dict(workload=workload, held_before_probe_ms=hold_ms,
            external_work_still_held_at_measurement=blocked, controls=samples,
            canonical_execution_count=len(fixture.executions.records),
            canonical_finalization_count=len(fixture.executions.finalized),
            tool_calls=fixture.ordinary.await_count)
    finally:
        release.set()
        if client_task is not None:
            client_task.cancel()
            await asyncio.gather(client_task, return_exceptions=True)
        try:
            if initialized:
                await fixture.asyncTearDown()
        finally:
            logger.removeHandler(probe)
            logger.setLevel(old_level)
            logger.propagate = old_propagate


async def snapshot_samples(count):
    fixture = state_fixtures.RuntimeFixtures()
    fixture.setUp()
    service = LiveStateService(fixture.scope, clock=lambda: fixture.now)
    await service.start()
    try:
        service.register_producer("fixture", fixture.epoch)
        service.submit(fixture.watermark(1))
        for index in range(8):
            service.submit(fixture.upsert(index + 2, fixture.entry(entry_id=f"observation-{index}",
                summary=f"The display reads {42 + index}.",
                fresh_until=fixture.now + timedelta(seconds=5), expires_at=fixture.now + timedelta(seconds=10))))
        await service.wait_idle()
        service.mark_ready()
        select_voice_context(service.snapshot(), now=fixture.now)  # warm pure-code imports
        timings = []
        with patch("builtins.open", side_effect=AssertionError("projection attempted file I/O")) as files, \
             patch("socket.socket.connect", side_effect=AssertionError("projection attempted network I/O")) as network:
            for _ in range(count):
                start = time.perf_counter_ns()
                selection = select_voice_context(service.snapshot(), now=fixture.now)
                timings.append((time.perf_counter_ns() - start) / 1_000_000)
                if not selection.selected_entry_ids:
                    raise AssertionError("Timing an empty projection is not this workload")
            files.assert_not_called()
            network.assert_not_called()
        return dict(timings_ms=timings, summary=distribution(timings), entries=8,
            selected_entries=len(selection.selected_entry_ids),
            boundary="in_process_immutable_snapshot_and_pure_projection",
            intercepted_file_and_network_calls=0)
    finally:
        await service.close()


async def run_benchmark(*, samples=20, hold_ms=50, deadline_ms=250):
    if (type(samples) is not int or not 1 <= samples <= 100
            or not 0 < hold_ms <= 1000 or not 0 < deadline_ms <= 2000):
        raise ValueError("Benchmark bounds exceeded")
    cases = []
    for workload in WORKLOADS:
        runs = [await control_sample(workload, hold_ms=hold_ms, deadline_ms=deadline_ms)
                for _ in range(samples)]
        controls = [control for run in runs for control in run["controls"]]
        cases.append(dict(workload=workload, samples=runs,
            dispatch=distribution([item["enqueue_to_dispatch_ms"] for item in controls]),
            provider_dispatch=distribution([item["enqueue_to_dispatch_ms"] for item in controls
                                            if item["origin"] == "provider_control"]),
            client_dispatch=distribution([item["enqueue_to_dispatch_ms"] for item in controls
                                          if item["origin"] == "client_control"]),
            handler=distribution([item["handler_ms"] for item in controls]),
            unfinished_or_failed=sum(item["status"] != "completed" for item in controls),
            duplicate_or_missing_operations=sum(run["canonical_execution_count"] != 1
                or run["canonical_finalization_count"] != 1 or run["tool_calls"] != 1 for run in runs)))
    snapshots = await snapshot_samples(max(20, samples * 10))
    qualified_population = samples >= 20 and hold_ms >= 25
    dispatch_ok = all(case["dispatch"]["missing"] == 0
        and case["dispatch"]["p95_ms"] < 25 and case["unfinished_or_failed"] == 0
        and case["duplicate_or_missing_operations"] == 0 for case in cases)
    snapshot_ok = snapshots["summary"]["missing"] == 0 and snapshots["summary"]["p95_ms"] < 5
    return dict(schema=1, scope="controlled_shared_Realtime_backend_handlers_not_native_media",
        clock_domains={"control": "same-process time.monotonic", "projection": "time.perf_counter_ns"},
        host=dict(system=platform.system(), machine=platform.machine(), python=platform.python_version(),
                  asyncio_debug=asyncio.get_running_loop().get_debug()),
        exclusions=["real network", "Supabase", "model reasoning", "phone", "glasses", "human hearing"],
        logging="real redacted telemetry with in-memory handler, no external log sink",
        qualified_population=qualified_population, workloads=cases, snapshots=snapshots,
        gates=dict(snapshot_p95_under_5ms=snapshot_ok, control_p95_under_25ms_and_no_stalls=dispatch_ok,
                   passed=qualified_population and snapshot_ok and dispatch_ok))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or not 20 <= args.samples <= 100:
        parser.error("Use an absolute output path and 20-100 samples per workload")
    try:
        report = asyncio.run(run_benchmark(samples=args.samples))
    except Exception as error:
        report = dict(schema=1, failure=type(error).__name__, gates={"passed": False},
                      scope="offline_control_benchmark_incomplete")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"report": str(args.output), "gates": report["gates"],
                      "failure": report.get("failure")}, sort_keys=True))
    return 0 if report["gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
