#!/usr/bin/env python3
"""Isolated Luna client-router voice acceptance; prepare/review are offline.

Run is explicitly paid and Core-owned: real WebRTC, Live, Luna, canonical SQL,
and (work mode only) Hermes. It reuses the UNMODIFIED Hermes harness lifecycle,
fixture, audio, transport trace and teardown. No server or serving activation.
The two scoped patches below instrument this single-process CLI, not app code;
do not run multiple harnesses concurrently in one interpreter.

Modes: lookup (actual Live State read, no Hermes work), chat (no delegation),
note (the generic work/optional busy-followup/continuation scenario verbatim).
The legacy speech bundle calls its first clip "work" in every mode; this is
ONLY an asset identity, never a task-result gate for lookup/chat.

Evidence collection is not semantic acceptance. Review output.wav, transcript,
canonical receipts and BOTH timelines. For lookup/work, --router-trace-correct
attests exactly one gpt-6-luna/effort-none request per delegation, complete
request/response observations and the intended actual action, not just an ACK.
Lookup additionally requires review of the actual Live State read/result.
No live path offers a fake model, fake state, or injected transcript/result.
"""
from __future__ import annotations

import argparse
import asyncio
from array import array
from functools import partial
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import wave

import live_hermes_voice_acceptance as h


FORMAT = "jarvis.live_router_voice_acceptance.v1"
FACTORY = "live_luna_voice_runtime:application"
QUESTIONS = {
    "lookup": "Please check Live State and tell me whether I have any active background tasks right now.",
    "chat": "Hi Jarvis. Tell me a short joke about a teapot.",
    "work": h.DEFAULT_QUESTION,
}


def prepare(root, *, mode, question=None, busy_followup=None, continuation=None, speech_builder=None):
    h.require(mode in QUESTIONS, "unknown_scenario")
    h.require(mode == "work" or not (busy_followup or continuation), "hooks_require_work_mode")
    question = QUESTIONS[mode] if question is None else question
    manifest = h.prepare(root, question=question, busy_followup=busy_followup,
                         continuation=continuation, speech_builder=speech_builder)
    scenario = dict(format=FORMAT, mode=mode, question=question, factory=FACTORY,
                    task_result_gate=mode == "work", paid_execution=False)
    h.private_write(Path(root) / "router-scenario.json", scenario)
    return {**scenario, "speech": manifest["speech"]}


def scenario(root):
    value = h.read_json(h.private_root(root) / "router-scenario.json", 65536)
    h.require(value.get("format") == FORMAT and value.get("mode") in QUESTIONS, "invalid_router_scenario")
    manifest, limits, clips = h.validate_bundle(root)
    h.require(value.get("factory") == FACTORY and value.get("question") == manifest["script"][0][1],
              "router_scenario_bundle_mismatch")
    h.require(value.get("task_result_gate") is (value["mode"] == "work"), "wrong_task_gate")
    h.require(value["mode"] == "work" or len(clips) == 1, "hooks_require_work_mode")
    return value, limits, clips


class AcceptanceTrace(h.Trace):
    def __init__(self, root, limits):
        super().__init__(root, limits)
        self.router_observations = 0
        self.native_delegation_events = []
        self.transcript = []

    def record(self, stage, *, raw=None, **fields):
        sequence = super().record(stage, raw=raw, **fields)
        payload = json.loads(raw) if raw is not None else fields.get("payload")
        if stage == "luna_router_observe":
            self.router_observations += 1
        if isinstance(payload, dict) and stage in {"receive", "client_data_channel"}:
            kind = payload.get("type", "")
            if "delegation" in kind:
                self.native_delegation_events.append(dict(sequence=sequence, event=h.scrub(payload)))
            if "transcript" in kind:
                self.transcript.append(dict(sequence=sequence, monotonic_s=time.monotonic(), event=h.scrub(payload)))
        return sequence


def check_no_effects(snapshot, baseline, trace, mode):
    h.require(not any(snapshot.get(table) for table in h.TABLES ),
              "unexpected_canonical_work_effect")
    h.require(snapshot == baseline, "readonly_canonical_state_changed")
    h.require(trace.worker_requests == 0, "unexpected_hermes_model_request")
    if mode == "chat":
        h.require(not trace.native_delegation_events and trace.router_observations == 0,
                  "unexpected_chat_delegation")


def output_text(event):
    if "output_transcript" not in event.get("type", ""):
        return ""
    value = event.get("delta", event.get("text", event.get("transcript", "")))
    if isinstance(value, dict):
        value = value.get("text", "")
    return value.strip() if isinstance(value, str) else ""


def spoken_evidence_ready(trace, state, speech, mode):
    """Collection gate only; a plausible caption/PCM is not a useful answer."""
    boundary = speech["host_after_monotonic_s"]
    if mode == "lookup":
        if not trace.router_observations or trace.last_commentary is None or trace.last_commentary <= boundary:
            return False
        boundary = trace.last_commentary
    captions = [row for row in trace.transcript if row["monotonic_s"] > boundary and output_text(row["event"])]
    signal = state.get("lastSignal")
    return bool(captions and signal is not None and signal > speech["end_sample"]
                and state["audio_sample"] - signal > h.RATE)


async def drive_readonly(app, fixture, peer, trace, root, clips, limits, report, *, mode):
    h.require(mode in {"lookup", "chat"} and len(clips) == 1, "invalid_readonly_scenario")
    started = time.monotonic()
    report.update(mode=mode, task_result_gate="not_applicable", scenario_started_monotonic_s=started,
                  semantic_acceptance="requires_audio_transcript_and_router_review",
                  useful_audio_latency_ms=None, physical_hearing=False,
                  router_single_pass="requires_exact_request_response_review",
                  live_state_read="requires_actual_read_result_review" if mode == "lookup" else "not_applicable")
    baseline = await h.canonical_snapshot(fixture, limits)
    trace.record("canonical_observation", payload=baseline)
    speech = None
    while time.monotonic() < started + limits.scenario_seconds:
        h.require(app.session.failure is None, "live_session_failed")
        state = await peer.js("acceptance.snapshot()")
        h.require(not state.get("fatal"), "browser_failed")
        for event in await peer.js("acceptance.drain()"):
            trace.record("client_data_channel", payload=event)
        snapshot = await h.canonical_snapshot(fixture, limits)
        check_no_effects(snapshot, baseline, trace, mode)
        if speech is None and app.session.ready:
            clip = clips[0]
            before = time.monotonic()
            speech = await peer.js("acceptance.play(" + json.dumps(clip["id"]) + ")")
            speech.update(last_active_sample=speech["start_sample"] + clip["last_active_sample_exclusive"],
                          host_before_monotonic_s=before, host_after_monotonic_s=time.monotonic())
            report["inputs"].append(speech)
            trace.record("speech_scheduled", payload=speech, source_sha256=clip["sha256"])
        if speech is not None and spoken_evidence_ready(trace, state, speech, mode):
            report["evidence_collected"] = True
            break
        await asyncio.sleep(.2)
    report["canonical"] = await h.canonical_snapshot(fixture, limits)
    trace.record("canonical_observation", payload=report["canonical"])
    check_no_effects(report["canonical"], baseline, trace, mode)
    report["audio"] = await h.collect_audio(peer, root, limits)
    report["hooks"] = {clips[0]["id"]: "injected" if speech else "not_observed_within_budget"}


async def run(root, *, settings, execute_paid=False):
    # This guard precedes imports of the runtime, DB, browser and SDK clients.
    h.require(execute_paid is True, "explicit_paid_execution_required")
    config, _, _ = scenario(root)
    mode = config["mode"]
    traces = []

    def trace_factory(*args, **kwargs):
        trace = AcceptanceTrace(*args, **kwargs)
        traces.append(trace)
        return trace

    original_drive = h.drive
    driver = original_drive if mode == "work" else partial(drive_readonly, mode=mode)
    with patch.object(h, "Trace", trace_factory), patch.object(h, "drive", driver):
        report = await h.run(root, factory=FACTORY, settings=settings, execute_paid=True)
    trace = traces[0]
    report.update(router_format=FORMAT, mode=mode, router_observations=trace.router_observations,
                  native_delegation_events=trace.native_delegation_events, transcript=trace.transcript,
                  router_single_pass="requires_exact_request_response_review",
                  task_result_gate=mode == "work")
    # Include late events observed during normal harness shutdown, not just the
    # polling snapshot. A failed run retains its original failure receipt.
    if not report.get("failure"):
        try:
            if mode == "chat":
                h.require(trace.router_observations == 0 and not trace.native_delegation_events,
                          "unexpected_chat_delegation")
            else:
                h.require(trace.router_observations > 0, "router_observations_missing")
            if mode != "work":
                h.require(trace.worker_requests == 0, "unexpected_hermes_model_request")
        except h.HarnessError as error:
            report["failure"] = {"type": type(error).__name__, "code": str(error)}
            report["evidence_collected"] = False
    h.private_write(h.private_root(root) / "result.json", report, exclusive=False)
    return report


def review(root, *, useful_audio_sample, spoken_result_correct=False, transcript_correct=False,
           router_trace_correct=False, live_state_read_correct=False, task_correct=False):
    config, _, _ = scenario(root)
    mode = config["mode"]
    root = h.private_root(root)
    report = h.read_json(root / "result.json")
    h.require(report.get("mode") == mode and report.get("router_format") == FORMAT, "wrong_result_scenario")
    h.require(report.get("evidence_collected") is True and not report.get("failure"), "run_incomplete")
    cleanup = report.get("cleanup", {})
    browser = cleanup.get("browser", {})
    h.require(cleanup.get("fixture_cleaned") is True and cleanup.get("session_and_worker_joined") is True
              and cleanup.get("provider_shutdown_confirmed") is True
              and cleanup.get("application_context_closed") is True
              and browser.get("media", {}).get("closed") is True
              and browser.get("process_group_gone") is True
              and browser.get("loopback_server_stopped") is True
              and not any(key in browser for key in ("media_error", "socket_close_error")), "cleanup_incomplete")
    h.require(spoken_result_correct is True and transcript_correct is True, "explicit_useful_answer_review_required")
    h.require(mode == "chat" or router_trace_correct is True, "explicit_single_pass_router_review_required")
    h.require(mode != "lookup" or live_state_read_correct is True, "explicit_live_state_read_review_required")
    if mode == "work":
        result = h.review(root, useful_audio_sample=useful_audio_sample,
                          task_correct=task_correct, spoken_result_correct=spoken_result_correct)
    else:
        h.require(report.get("worker_model_requests") == 0, "unexpected_hermes_model_request")
        h.require(mode != "chat" or report.get("router_observations") == 0
                  and not report.get("native_delegation_events"), "unexpected_chat_delegation")
        h.require(mode != "lookup" or report.get("router_observations", 0) > 0, "router_observations_missing")
        h.require(any(output_text(row["event"]) for row in report.get("transcript", [])), "output_transcript_missing")
        h.require(type(useful_audio_sample) is int, "invalid_useful_audio_sample")
        end = report["inputs"][0]["last_active_sample"]
        h.require(useful_audio_sample > end, "useful_audio_precedes_request_end")
        with wave.open(str(root / "output.wav"), "rb") as audio:
            h.require(audio.getparams()[:3] == (1, 2, h.RATE) and useful_audio_sample < audio.getnframes(),
                      "useful_audio_outside_capture")
            audio.setpos(useful_audio_sample)
            samples = array("h", audio.readframes(h.RATE // 10))
            if sys.byteorder != "little":
                samples.byteswap()
            h.require(any(abs(sample) >= 128 for sample in samples), "useful_audio_window_has_no_signal")
        result = dict(format=FORMAT, mode=mode, semantic_acceptance="reviewer_confirmed",
                      useful_audio_sample=useful_audio_sample, useful_audio_latency_ms=(useful_audio_sample-end)*1000/h.RATE,
                      task_result_gate="not_applicable", physical_hearing=False,
                      boundary="browser PCM clock; energy end, not word alignment; no hearing claim")
        h.private_write(root / "review.json", result)
    h.private_write(root / "router-review.json", dict(format=FORMAT, mode=mode,
        transcript_correct=True, spoken_result_correct=True,
        router_trace_correct=router_trace_correct if mode != "chat" else "not_applicable",
        live_state_read_correct=live_state_read_correct if mode == "lookup" else "not_applicable"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("root", type=Path)
    p.add_argument("--mode", choices=QUESTIONS, required=True)
    p.add_argument("--question")
    p.add_argument("--busy-followup")
    p.add_argument("--continuation")
    p = sub.add_parser("run")
    p.add_argument("root", type=Path)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--execute-paid", action="store_true")
    p = sub.add_parser("review")
    p.add_argument("root", type=Path)
    p.add_argument("--useful-audio-sample", type=int, required=True)
    for name in ("spoken-result-correct", "transcript-correct", "router-trace-correct", "live-state-read-correct", "task-correct"):
        p.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.root, mode=args.mode, question=args.question,
                             busy_followup=args.busy_followup, continuation=args.continuation)
        elif args.command == "review":
            result = review(args.root, **{key: value for key, value in vars(args).items() if key not in {"root", "command"}})
        else:
            h.require(args.execute_paid, "explicit_paid_execution_required")
            from app.settings import JarvisSettings
            settings = JarvisSettings(_env_file=args.env_file)
            result = asyncio.run(run(args.root, settings=settings, execute_paid=True))
        print(json.dumps(dict(format=FORMAT, mode=result.get("mode"), command=args.command,
                              evidence_collected=result.get("evidence_collected"), failure=result.get("failure"),
                              semantic_acceptance=result.get("semantic_acceptance", "not_evaluated"))))
        return int(bool(result.get("failure") or args.command == "run" and not result.get("evidence_collected")))
    except Exception as error:
        print(json.dumps(dict(failure_type=type(error).__name__,
                              code=str(error) if isinstance(error, h.HarnessError) else "harness_failed")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
