"""Live comparison of interrupted memory writes. Never imported by production."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from app.realtime.openai_client import OpenAIRealtimeConnection, build_session_update_event
from evals.gemini_live.client import LiveProbe
from evals.gemini_live.interruption import (
    STAGES, Barrier, DatabaseFixtures, Lifecycle, Trace, instrument_registry,
)
from evals.gemini_live.operation_context import OperationContext


INSTRUCTIONS = (
    "You are Jarvis in a disposable memory test. Answer briefly, in spoken English. "
    "Use remember for an explicit request to save a fact. Use only facts and tool outcomes "
    "actually provided. An interruption of speech does not undo an already admitted save. "
    "Respect the newest user request. Do not claim that a save succeeded without evidence."
)
SAVE = "Remember this fact: my test project's name is Amber Finch."
INTERRUPT = (
    "Quick interruption: let any save already underway finish, but do not start or retry "
    "a save. What is four plus five?"
)
STATUS = "Did that save complete? Do not save or retry anything. Tell me its status briefly."
FOLLOWUP_SAVE = "Also remember this separate fact: my test project's color is teal."
HANDOFFS = ("native", "operation_context")


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


class Provider:
    """Small transport adapter, not a replacement Jarvis realtime session."""

    def __init__(self, name, keys, registry, executor, barrier, trace, *, context="", operation_context=None):
        self.name, self.keys, self.registry = name, keys, registry
        self.trace, self.barrier = trace, barrier
        self.lifecycle = Lifecycle(registry, executor, self.tool_output, barrier, trace)
        self.calls, self.tasks, self.delivered = {}, [], set()
        self.cancelled = set()
        self.progress = asyncio.Event()
        self.post_result_audio = asyncio.Event()
        self.phase = "save"
        self.phases = {}
        self.active_response = None
        self.response_phases = {}
        self.reader = None
        self.context = context
        self.operation_context = operation_context
        self.cancel_requests = set()
        self.wire = None
        self.output_time = None
        self._gemini_output_phase = self.phase
        self._gemini_pending_phase = None
        self._gemini_turn_interrupted = False
        self._gemini_turn_has_audio = False

    def phase_result(self, name=None):
        return self.phases.setdefault(name or self.phase, {
            "transcript": "", "audio": bytearray(), "completed": False,
        })

    async def connect(self):
        instructions = INSTRUCTIONS + self.context
        if self.operation_context is not None:
            instructions += self.operation_context.instructions
        if self.name == "gemini":
            self.wire = LiveProbe(key=self.keys["gemini"], model=self.keys.get("gemini_model", "gemini-3.1-flash-live-preview"),
                                  instructions=instructions, manual_vad=False,
                                  bridge=SimpleNamespace(registry=self.registry))
            await self.wire.connect()
        else:
            settings = SimpleNamespace(
                OPENAI_API_KEY=self.keys["openai"], OPENAI_REALTIME_MODEL=self.keys["openai_model"],
                OPENAI_REALTIME_VOICE="alloy", OPENAI_REALTIME_TRANSCRIPTION_MODEL="",
                JARVIS_AUDIO_FORMAT="pcm16", JARVIS_AUDIO_SAMPLE_RATE=24000,
            )
            self.wire = OpenAIRealtimeConnection(settings=settings, owner_id="disposable-interruption-evaluation")
            await self.wire.connect()
            await self.wire.send_event(build_session_update_event(
                settings=settings, instructions=instructions, tools=self.registry.openai_realtime_tools(),
            ))
            async with asyncio.timeout(20):
                while True:
                    event = await self.wire.receive_event()
                    if event.get("type") == "error":
                        raise RuntimeError("OpenAI session setup failed")
                    if event.get("type") == "session.updated":
                        break
        self.reader = asyncio.create_task(self.receive())

    def check(self):
        for task in [self.reader, *self.tasks]:
            if task is not None and task.done():
                if task.cancelled():
                    raise RuntimeError("Unexpected evaluation task cancellation")
                error = task.exception()
                if error is not None:
                    raise error

    async def until(self, predicate, timeout=30):
        async with asyncio.timeout(timeout):
            while True:
                self.progress.clear()
                self.check()
                if predicate():
                    return
                # No injected timing gap at the mutation boundary. This timeout
                # only wakes up to propagate receiver/task failures promptly.
                try:
                    await asyncio.wait_for(self.progress.wait(), .1)
                except TimeoutError:
                    pass

    async def start_response(self, generation):
        await self.until(lambda: self.active_response is None)
        if generation != self.lifecycle.generation:
            return
        await self.wire.send_event({"type": "response.create", "response": {
            "metadata": {"evaluation_phase": self.phase},
        }})
        # Reserving until response.created prevents two concurrent followups.
        self.active_response = "pending"

    async def text(self, text, phase, *, interrupt=False):
        if self.name == "gemini":
            if self._gemini_pending_phase is not None:
                raise RuntimeError("The interrupted Gemini turn has not ended")
            previous = self.phase_result(self._gemini_output_phase)
            if interrupt and not previous["completed"] and (previous["audio"] or previous["transcript"]):
                # Sending a new input does not advance the server's output
                # stream. Queued audio and the old turnComplete still belong
                # to the old turn, including when interrupted arrives alone.
                self._gemini_pending_phase = phase
                self.trace.add("gemini_turn_handoff_pending",
                               from_phase=self._gemini_output_phase, to_phase=phase)
            else:
                self._gemini_output_phase = phase
                self._gemini_turn_interrupted = False
                self._gemini_turn_has_audio = False
        if interrupt:
            self.lifecycle.interrupt()
        self.phase = phase
        self.phase_result()
        self.trace.add("user_text", phase=phase, text=text)
        wire_text = text
        if interrupt and self.operation_context is not None:
            started = time.monotonic()
            receipts = await self.operation_context.capture(self)
            wire_text = json.dumps({"application_operation_context": receipts,
                                    "user_message": text}, ensure_ascii=True, default=str)
            self.trace.add("operation_context_injected", phase=phase, receipts=receipts,
                           build_seconds=time.monotonic() - started)
        if self.name == "gemini":
            await self.wire.send({"realtimeInput": {"text": wire_text}})
        else:
            if self.active_response is not None:
                event_id = "eval_cancel_" + uuid4().hex
                self.cancel_requests.add(event_id)
                payload = {"type": "response.cancel", "event_id": event_id}
                if self.active_response != "pending":
                    payload["response_id"] = self.active_response
                await self.wire.send_event(payload)
                await self.until(lambda: self.active_response is None)
            await self.wire.send_event({"type": "conversation.item.create", "item": {
                "type": "message", "role": "user", "content": [{"type": "input_text", "text": wire_text}],
            }})
            await self.start_response(self.lifecycle.generation)

    def dispatch(self, call, source):
        if len(self.tasks) >= 8:
            raise RuntimeError("Evaluation tool-call limit reached")
        previous = self.calls.get(call["id"])
        if previous is not None and previous != call:
            raise ValueError("Provider reused a call ID with different arguments")
        self.calls[call["id"]] = call
        self.lifecycle.source(source)

        async def execute():
            try:
                eligible = await self.lifecycle.dispatch(call, source)
                if self.name == "openai" and eligible and self.lifecycle.session._tool_followup is not None:
                    await self.start_response(self.lifecycle.generation)
            finally:
                self.progress.set()
        task = asyncio.create_task(execute())
        self.tasks.append(task)

    async def tool_output(self, event):
        item = event["item"]
        call_id = item["call_id"]
        result = json.loads(item["output"])
        if self.name == "gemini":
            # Send the real outcome even after speech was interrupted. This is
            # an observed delivery attempt, NOT a fabricated provider ACK.
            await self.wire.send({"toolResponse": {"functionResponses": [{
                "id": call_id, "name": self.calls[call_id]["name"], "response": result,
            }]}})
        else:
            await self.wire.send_event(event)
        self.delivered.add(call_id)
        self.output_time = time.monotonic()
        self.trace.add("tool_output_sent", call_id=call_id, cancelled=call_id in self.cancelled,
                       result=result, receipt_confirmed=False)
        self.progress.set()
        return True

    def receive_gemini(self, event):
        server = event.get("serverContent", {})
        phase = self._gemini_output_phase
        self.trace.add("provider_event", phase=phase, input_phase=self.phase,
                       keys=list(event), server_keys=list(server))
        for call in event.get("toolCall", {}).get("functionCalls", []):
            self.dispatch(call, "gemini-client-turn-" + str(self.lifecycle.generation))
        ids = event.get("toolCallCancellation", {}).get("ids", [])
        self.cancelled.update(ids)
        if ids:
            self.trace.add("provider_tool_cancellation", ids=ids)
        if server.get("interrupted"):
            self._gemini_turn_interrupted = True
            self.trace.add("provider_interrupted", phase=phase)
        result = self.phase_result(phase)
        text = server.get("outputTranscription", {}).get("text", "")
        if text:
            result["transcript"] += text
            self.trace.add("transcript", phase=phase, text=text)
        for part in server.get("modelTurn", {}).get("parts", []):
            inline = part.get("inlineData", {})
            if str(inline.get("mimeType", "")).startswith("audio/") and inline.get("data"):
                data = base64.b64decode(inline["data"])
                self._gemini_turn_has_audio |= bool(data)
                self.audio(data, phase)
        if server.get("turnComplete"):
            completed = self._gemini_turn_has_audio and not self._gemini_turn_interrupted
            if completed:
                result["completed"] = True
            self.trace.add("gemini_turn_ended", phase=phase,
                           interrupted=self._gemini_turn_interrupted,
                           accepted_completion=completed)
            self._gemini_turn_interrupted = False
            self._gemini_turn_has_audio = False
            if self._gemini_pending_phase is not None:
                self._gemini_output_phase = self._gemini_pending_phase
                self._gemini_pending_phase = None
                self.trace.add("gemini_turn_handoff_completed",
                               from_phase=phase, to_phase=self._gemini_output_phase)
        self.progress.set()

    async def receive(self):
        try:
            while True:
                event = await (self.wire.receive() if self.name == "gemini" else self.wire.receive_event())
                if self.name == "gemini":
                    self.receive_gemini(event)
                else:
                    kind = event.get("type", "unknown")
                    response_id = event.get("response_id")
                    phase = self.response_phases.get(response_id, self.phase)
                    self.trace.add("provider_event", phase=phase, type=kind, response_id=response_id)
                    if kind == "error":
                        error = event.get("error", {})
                        if error.get("event_id") in self.cancel_requests and error.get("code") == "response_cancel_not_active":
                            self.trace.add("cancel_completion_race")
                        else:
                            self.trace.add("provider_error", code=error.get("code"), message=error.get("message"))
                            raise RuntimeError("OpenAI provider error; see scoped trace")
                    elif kind == "response.created":
                        response = event["response"]
                        self.active_response = response["id"]
                        self.response_phases[response["id"]] = response.get("metadata", {}).get("evaluation_phase", self.phase)
                        self.lifecycle.source(response["id"])
                    elif kind == "response.output_item.done" and event.get("item", {}).get("type") == "function_call":
                        item = event["item"]
                        self.dispatch({"id": item["call_id"], "name": item["name"],
                                       "args": json.loads(item.get("arguments", "{}"))}, response_id)
                    elif kind == "conversation.item.created" and event.get("item", {}).get("type") == "function_call_output":
                        self.trace.add("tool_output_provider_ack", call_id=event["item"]["call_id"])
                    elif kind in {"response.output_audio.delta", "response.audio.delta"}:
                        self.audio(base64.b64decode(event["delta"]), phase)
                    elif kind in {"response.output_audio_transcript.delta", "response.audio_transcript.delta"}:
                        self.phase_result(phase)["transcript"] += event.get("delta", "")
                        self.trace.add("transcript", phase=phase, text=event.get("delta", ""))
                    elif kind == "response.done":
                        response = event["response"]
                        if self.active_response in {response["id"], "pending"}:
                            self.active_response = None
                        done_phase = self.response_phases.get(response["id"], self.phase)
                        result = self.phase_result(done_phase)
                        if response.get("status") == "completed" and result["audio"]:
                            result["completed"] = True
                        if response.get("status") == "failed":
                            self.trace.add("response_failed", details=response.get("status_details"))
                            raise RuntimeError("OpenAI response failed")
                self.progress.set()
        finally:
            self.progress.set()

    def audio(self, data, phase):
        self.phase_result(phase)["audio"].extend(data)
        if self.delivered and phase == "save":
            self.post_result_audio.set()
        self.trace.add("audio_received", phase=phase, byte_count=len(data))

    async def settle(self, phase):
        await self.until(lambda: self.phase_result(phase)["completed"]
                         and all(task.done() for task in self.tasks), timeout=35)

    async def close(self):
        self.barrier.release.set()
        self.lifecycle.session._closed.set()
        if self.tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), 25)
            except TimeoutError:
                self.trace.add("tool_drain_timeout")
        if self.reader is not None:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        if self.wire is not None:
            await self.wire.close()

    def save(self, directory, prefix):
        results = {}
        for phase, result in self.phases.items():
            path = directory / f"{prefix}-{phase}.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(24000)
                output.writeframes(bytes(result["audio"]))
            results[phase] = {"transcript": result["transcript"], "completed": result["completed"],
                              "audio_bytes": len(result["audio"]), "wav": path.name}
        return results


async def live_case(fixtures, keys, provider_name, stage, directory, *, handoff="native", followup_save=False):
    trace = Trace()
    barrier = Barrier(stage, trace)
    owner, session_id = await fixtures.create(directory.name)
    registry = instrument_registry(fixtures.registry(owner, session_id), barrier, trace)
    operation_context = OperationContext(fixtures, owner, session_id) if handoff == "operation_context" else None
    if operation_context is not None:
        operation_context.register(registry)
    executor = fixtures.executor(registry, owner, session_id)
    provider = Provider(provider_name, keys, registry, executor, barrier, trace, operation_context=operation_context)
    report = {"provider": provider_name, "handoff": handoff, "stage": stage, "owner": owner, "session_id": str(session_id),
              "followup_save_requested": followup_save,
              "expected_memory_count": 0 if stage == "before_admission" else 1}
    try:
        await fixtures.message(owner, session_id, SAVE)
        await provider.connect()
        await provider.text(SAVE, "save")
        if stage == "after_receipt":
            await provider.until(provider.post_result_audio.is_set)
            trace.add("barrier_reached", stage=stage,
                      evidence="model audio after tool output; not physical playback or a Gemini ACK")
        else:
            await provider.until(barrier.reached.is_set)
        report["at_interruption"] = await fixtures.snapshot(owner, session_id)
        await provider.text(INTERRUPT, "interruption", interrupt=True)
        barrier.release.set()
        await provider.settle("interruption")
        await provider.text(STATUS, "status")
        await provider.settle("status")
        report["after_status"] = await fixtures.snapshot(owner, session_id)
        if followup_save:
            await fixtures.message(owner, session_id, FOLLOWUP_SAVE)
            await provider.text(FOLLOWUP_SAVE, "followup_save")
            await provider.settle("followup_save")
            report["after_followup"] = await fixtures.snapshot(owner, session_id)
        report["execution"] = "completed"
    except Exception as error:
        report["execution"] = "failed"
        report["error_type"] = type(error).__name__
        trace.add("case_failed", error_type=type(error).__name__)
    finally:
        await provider.close()
        report["phases"] = provider.save(directory, "live")
        report["cancelled_call_ids"] = sorted(provider.cancelled)
        report["native_outcome_attempts"] = sorted(provider.delivered)
        report["after_interruption"] = await fixtures.snapshot(owner, session_id)
        write_json(directory / "native-trace.json", trace.events)

    # Rehydrate from REAL committed execution outcomes, not the fixture's belief
    # about a write. This separately tests a proposed reconnection integration;
    # it must never be reported as native cancelled-tool-result delivery support.
    recovery_trace = Trace()
    recovery_barrier = Barrier(None, recovery_trace)
    snapshot = report["after_interruption"]
    recovery_snapshot = operation_context.describe_snapshot(snapshot) if operation_context is not None else snapshot
    context = ("\nApplication recovery context from durable storage, not a new user request:\n"
               + json.dumps(recovery_snapshot, default=str)
               + "\nThis is the complete saved state for this isolated session. Do not repeat mutations.")
    recovery = Provider(provider_name, keys, registry, executor, recovery_barrier, recovery_trace,
                        context=context, operation_context=operation_context)
    try:
        await recovery.connect()
        await recovery.text(STATUS, "recovery")
        await recovery.settle("recovery")
        report["recovery_execution"] = "completed"
    except Exception as error:
        report["recovery_execution"] = "failed"
        report["recovery_error_type"] = type(error).__name__
    finally:
        await recovery.close()
        report["recovery_phases"] = recovery.save(directory, "recovery")
        write_json(directory / "recovery-trace.json", recovery_trace.events)
    report["final_storage"] = await fixtures.snapshot(owner, session_id)
    rows = report["final_storage"]["memories"]
    executions = report["final_storage"]["executions"]
    original_count = len(report["after_status"]["memories"]) if "after_status" in report else None
    followup_count = len(report["after_followup"]["memories"]) if "after_followup" in report else None
    report["expected_final_memory_count"] = report["expected_memory_count"] + int(followup_save)
    report["storage_checks"] = {
        "expected_memory_count": len(rows) == report["expected_final_memory_count"],
        "original_save_count": original_count == report["expected_memory_count"],
        "followup_save_count": not followup_save or (original_count is not None and followup_count == original_count + 1),
        "memory_count": len(rows),
        "all_execution_outcomes_terminal": all(row["status"] in {"completed", "failed"} for row in executions),
        "successful_execution_count": sum(row["status"] == "completed" for row in executions),
        "status_lookups": sum(row["tool_name"] == "get_operation_status" for row in executions),
        "status_lookups_succeeded": all(row["status"] == "completed" for row in executions
                                        if row["tool_name"] == "get_operation_status"),
    }
    report["semantic_grade"] = "manual_review_required"
    write_json(directory / "report.json", report)
    return report


async def durable_controls(fixtures):
    reports = []
    previous_status_ref = None
    for reuse in (True, False):
        owner, session_id = await fixtures.create("same_id_replay" if reuse else "new_id_control")
        await fixtures.message(owner, session_id, SAVE)
        registry = fixtures.registry(owner, session_id)
        arguments = {"memory_type": "fact", "content": "My test project's name is Amber Finch."}
        outcomes = []
        for index in range(2):
            # Fresh executor each time, same durable repository/session.
            outcomes.append(await fixtures.executor(registry, owner, session_id).execute(
                "remember", arguments, provider_call_id="control-" + str(0 if reuse else index),
            ))
        snapshot = await fixtures.snapshot(owner, session_id)
        expected = 1 if reuse else 2
        status_tool = OperationContext(fixtures, owner, session_id)
        receipts = status_tool.describe_snapshot(snapshot)["operations"]
        status_ref = receipts[0]["status_ref"] if receipts else "op_unavailable"
        status = await status_tool.read_status({"status_ref": status_ref})
        status_passed = (status.get("status") == "success"
                         and len(status["operations"]) == 1
                         and status["operations"][0]["status_ref"] == status_ref
                         and status["operations"][0]["state"] == "completed")
        restored = await OperationContext(fixtures, owner, session_id).read_status({"status_ref": status_ref})
        restored_passed = (restored.get("status") == "success"
                           and restored["operations"][0]["status_ref"] == status_ref)
        foreign_passed = True
        if previous_status_ref is not None:
            foreign = await status_tool.read_status({"status_ref": previous_status_ref})
            foreign_passed = foreign.get("error", {}).get("code") == "operation_not_found"
        previous_status_ref = status_ref
        reports.append({"name": "same_id_replay" if reuse else "new_id_control",
                        "expected_memory_count": expected, "snapshot": snapshot,
                        "operation_status_readback": status_passed,
                        "status_reference_rehydration": restored_passed,
                        "other_owner_operation_hidden": foreign_passed,
                        "passed": len(snapshot["memories"]) == expected
                        and len(snapshot["executions"]) == expected
                        and status_passed and restored_passed and foreign_passed
                        and all(outcome.get("status") == "success" for outcome in outcomes)})
    return reports


async def run(args, keys):
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = DatabaseFixtures(keys["database"], args.output)
    report = {"scope": "real non-image lifecycle, durable executor, real memory service and database commits",
              "models": {"gemini": keys.get("gemini_model", "gemini-3.1-flash-live-preview"),
                         "openai": keys["openai_model"]},
              "limitations": ["Text-driven interruption, not acoustic VAD or headset playback",
                              "Gemini tool origins are client-assigned turns, not provider response IDs",
                              "Fresh-session recovery context is a proposed adapter behavior",
                              "Operation-context mode adds real receipts and a read-only status tool, not exactly-once enforcement",
                              "No embeddings, image analysis, profile/search, or production deployment",
                              "Semantic status and answer correctness require transcript/audio review"],
              "cases": []}
    try:
        await fixtures.connect()
        report["durable_controls"] = await durable_controls(fixtures)
        write_json(args.output / "report.json", report)
        if not all(item["passed"] for item in report["durable_controls"]):
            raise RuntimeError("Durable fixture controls failed; no live comparison is valid")
        for repeat in range(args.repeats):
            for stage in args.stages:
                for provider_name in args.providers:
                    handoffs = getattr(args, "handoffs", ["native"])
                    # Alternate A/B order rather than always warming one mode first.
                    for handoff in (handoffs if repeat % 2 == 0 else list(reversed(handoffs))):
                        label = (f"{provider_name}-{stage}-{repeat + 1}" if handoffs == ["native"]
                                 else f"{provider_name}-{handoff}-{stage}-{repeat + 1}")
                        directory = args.output / label
                        directory.mkdir()
                        print(json.dumps({"starting": label}), flush=True)
                        case = await live_case(fixtures, keys, provider_name, stage, directory,
                                               handoff=handoff, followup_save=getattr(args, "followup_save", False))
                        report["cases"].append(case)
                        write_json(args.output / "report.json", report)
                        print(json.dumps({"finished": label, "execution": case["execution"],
                                          "storage_checks": case["storage_checks"]}), flush=True)
    finally:
        try:
            await fixtures.cleanup()
        finally:
            report["cleanup"] = fixtures.cleanup_result
            write_json(args.output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys-stdin", required=True, action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--providers", nargs="+", choices=("gemini", "openai"), default=["gemini", "openai"])
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--handoffs", nargs="+", choices=HANDOFFS, default=["native"])
    parser.add_argument("--followup-save", action="store_true", help="Check a genuinely new save after the interruption")
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=2)
    args = parser.parse_args()
    keys = json.load(sys.stdin)
    for key in ("database", "openai", "gemini", "openai_model"):
        if not isinstance(keys.get(key), str) or not keys[key]:
            parser.error("Missing required credential/model input: " + key)
    logging.basicConfig(level=logging.WARNING)
    result = asyncio.run(run(args, keys))
    if any(case["execution"] != "completed" or case["recovery_execution"] != "completed"
           or not case["storage_checks"]["expected_memory_count"]
           or not case["storage_checks"]["original_save_count"]
           or not case["storage_checks"]["followup_save_count"]
           or not case["storage_checks"]["status_lookups_succeeded"]
           or not case["storage_checks"]["all_execution_outcomes_terminal"] for case in result["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
