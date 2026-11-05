"""Matched native Gemini continuity checks with isolated real conversation storage.

Only synthetic speech, the non-playing phone sink, and a controlled optional
preparation wait are fixtures. Fresh-connection history and audio coordination
remain production code. Lexical checks require an independent answer review.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import suppress
import json
import logging
import os
from pathlib import Path
import re
from uuid import uuid4

from app.assistant.context_providers import CurrentTimeContextProvider, RecentMessagesContextProvider, SessionContextProvider
from app.experience.repository import ExperienceRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.realtime.gemini.client import GeminiLiveConnection
from app.realtime.gemini.session import GeminiAudioSession
from app.realtime.models import RealtimeSessionContext
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from evals.gemini_live.interruption import DatabaseFixtures
from evals.gemini_live.transport_probe import read_pcm, write_wav
from evals.glasses_reliability.search_tool_probe import NativePhone
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


PROMPTS = {
    "step": (
        "We are writing binary search in Python using arr, mid, and target. "
        "We have already handled equality with return mid. "
        "The next branch is when target is greater than arr at mid. "
        "These branches are indented eight spaces inside our loop. "
        "Tell me that elif condition, then explain that comparison in two detailed sentences. "
        "Do not move on to the body or another branch yet."
    ),
    "bridge": "One moment.",
    "spell": "Now spell that next condition character by character, including the indentation and punctuation.",
}
CASES = ("ordinary", "interrupted_live", "interrupted_reset", "completed_reset")


def compact(text):
    return re.sub(r"[^a-z]", "", text.lower())


def assess_answer(text):
    # Fixture oracle only. Never used to route or interpret production speech.
    words = compact(text)
    condition = compact(text.replace(">=", " greater than or equal to ").replace(">", " greater than "))
    condition = condition.partition("elif")[2].partition("colon")[0]
    positions = [condition.find(word) for word in ("target", "greaterthan", "arr", "mid")]
    return {
        "current_elif_step": all(word in words for word in ("elif", "target", "arr", "mid")),
        "correct_comparison": (
            all(position >= 0 for position in positions) and positions == sorted(positions)
            and "orequal" not in condition and "lessthan" not in condition
        ),
        "explicit_character_spelling": all(
            re.search(r"\b" + r"[\s,.-]+".join(word) + r"\b", text, re.I)
            for word in ("elif", "target", "arr", "mid")
        ),
        "no_added_parenthesis": "parenthes" not in words,
        "explicit_indentation_count": (
            bool(re.search(r"\b(?:eight|8)\s+spaces\b", text, re.I))
            or words.partition("elif")[0].count("space") == 8
        ),
        "explicit_bracket_directions": all(
            re.search(r"\b(?:" + names + r")\s+(?:square\s+)?bracket\b", text, re.I)
            for names in ("open|opening|left", "close|closed|closing|right")
        ),
        "spoken_colon": bool(re.search(r"\bcolon\b", text, re.I)),
    }


def classify_checks(case, checks):
    """Keep the planned continuity gate separate from the provider-quality grade.

    The combined result remains failed when either fails. A quality failure is
    evidence for the final provider decision, never a green coding-capability claim.
    """
    continuity = [
        "current_elif_step", "answer_audio_complete", "input_pcm_not_replayed",
        "no_stale_end_marker", "expected_connection_count", "no_provider_error", "resets_explained",
    ]
    if case.startswith("interrupted"):
        continuity.append("interruption_during_active_output")
    if case.endswith("_reset"):
        continuity.append("bridge_input_retained_or_explicit_gap")
        if case.startswith("interrupted"):
            continuity.append("interrupted_step_retained_as_uncertain")
    quality = [
        "current_elif_step", "correct_comparison", "explicit_character_spelling",
        "no_added_parenthesis", "explicit_indentation_count", "explicit_bracket_directions", "spoken_colon",
    ]
    continuity_passed = all(checks.get(key, False) for key in continuity)
    dictation_quality_passed = all(checks.get(key, False) for key in quality)
    return {
        "continuity_passed": continuity_passed,
        "dictation_quality_passed": dictation_quality_passed,
        "passed": continuity_passed and dictation_quality_passed,
    }


def turn_timings(diagnostics):
    rows = []
    for turn in diagnostics["turns"]:
        commit = turn.get("commit_received_at")
        rows.append({
            "segment_id": turn["segment_id"],
            "connection_generation": turn["connection_generation"],
            "input_status": turn["input_status"],
            **{
                key: round((turn[field] - commit) * 1000, 1)
                if commit is not None and turn.get(field) is not None else None
                for key, field in (
                    ("commit_to_first_provider_audio_ms", "first_provider_audio_at"),
                    ("commit_to_first_forwarded_audio_ms", "first_forwarded_audio_at"),
                )
            },
        })
    return rows


def incomplete_records(history):
    records = []
    for item in history:
        for part in item.get("parts", []):
            try:
                value = json.loads(part.get("text", ""))
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict) and value.get("kind") == "jarvis_incomplete_exchange":
                records.append(value)
    return records


async def run_case(name, *, fixtures, speech, output):
    owner, session_id = await fixtures.create("continuity-" + name)
    repository = SessionRepository(fixtures.pool)
    experience = ExperienceEventService(
        repository=ExperienceRepository(fixtures.pool),
        segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)),
    )
    phone, connections = NativePhone(), []
    preparation_entered = asyncio.Event()
    session = running = None
    supplied_pcm = 0
    result = {"case": name, "passed": False, "continuity_passed": False,
              "dictation_quality_passed": False, "checks": {},
              "device_playback": "not_tested", "answer_review": "required"}

    class ObservedConnection(GeminiLiveConnection):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.history, self.input_bytes, self.end_markers = [], 0, 0
            connections.append(self)

        async def connect(self, **kwargs):
            self.history = kwargs.get("history", [])
            return await super().connect(**kwargs)

        async def send_event(self, event):
            await super().send_event(event)
            realtime = event.get("realtimeInput", {})
            if "audio" in realtime:
                self.input_bytes += len(base64.b64decode(realtime["audio"]["data"]))
            self.end_markers += int("activityEnd" in realtime)

    async def until(predicate, timeout=45):
        async with asyncio.timeout(timeout):
            while not predicate():
                if phone.closed or (running is not None and running.done()):
                    raise RuntimeError("Native session ended before the expected boundary")
                await asyncio.sleep(.01)

    async def speak(segment):
        nonlocal supplied_pcm
        pcm = speech[segment]
        supplied_pcm += len(pcm)
        for offset in range(0, len(pcm), 4800):
            chunk = pcm[offset:offset + 4800]
            await phone.send("audio.input.append", segment_id=segment,
                             audio_base64=base64.b64encode(chunk).decode())
            await asyncio.sleep(len(chunk) / 48000)
        await phone.send("audio.input.commit", segment_id=segment)
        await phone.send("response.create")

    try:
        async with asyncio.timeout(180):
            settings = JarvisSettings(_env_file=None, VISUAL_REALTIME_ENABLED=False)
            session = GeminiAudioSession(
                websocket=phone, context=RealtimeSessionContext(
                    audio_session_id=str(uuid4()), owner_id=owner, jarvis_session_id=session_id,
                    client_kind="phone_audio", session=await repository.get_session(owner, session_id),
                ),
                settings=settings, repository=repository, connection_factory=ObservedConnection,
                context_providers=(CurrentTimeContextProvider(), SessionContextProvider(), RecentMessagesContextProvider(12)),
                experience_event_service=experience,
            )
            prepare = session._prepare_turn

            async def optional_preparation_barrier(turn):
                if turn.segment_id == "bridge":
                    preparation_entered.set()
                    await asyncio.Future()
                return await prepare(turn)

            session._prepare_turn = optional_preparation_barrier
            running = asyncio.create_task(session.run())
            await until(lambda: any(e["type"] == "session.created" for e in phone.events))
            await speak("step")
            interrupted = name.startswith("interrupted")
            if interrupted:
                await until(lambda: (
                    session._turns["step"].input_finalized
                    and session._turns["step"].output.forwarded_audio_bytes > 0
                    and all(word in compact(session._turns["step"].output.transcript) for word in ("elif", "mid"))
                ))
                result["checks"]["interruption_during_active_output"] = not session._turns["step"].output.completed
                await phone.send("response.cancel")
            else:
                await until(lambda: session._turns["step"].output.completed and session._turns["step"].input_finalized)
            first = session._turns["step"]
            result["first_step"] = first.output.transcript
            reset = name.endswith("_reset")
            if reset:
                await speak("bridge")
                await until(preparation_entered.is_set, timeout=5)
            await speak("spell")
            await until(lambda: session._turns.get("spell") is not None
                        and session._turns["spell"].output.completed and session._turns["spell"].input_finalized)
            final = session._turns["spell"].output
            result["answer"] = final.transcript
            result["checks"].update(assess_answer(final.transcript))
            result["checks"].update({
                "answer_audio_complete": (
                    final.completed and not final.interrupted and final.forwarded_audio_bytes > 0
                    and final.forwarded_audio_bytes == final.provider_audio_bytes
                ),
                "input_pcm_not_replayed": sum(c.input_bytes for c in connections) == supplied_pcm,
                "no_stale_end_marker": sum(c.end_markers for c in connections) == 2,
                "expected_connection_count": len(connections) == (2 if reset else 1),
                "no_provider_error": not any(e["type"] == "error" for e in phone.events),
            })
            result["reconnections"] = [
                {key: event.get(key) for key in ("reason", "connection_generation", "resumed")}
                for event in phone.events if event["type"] == "audio.provider.reconnected"
            ]
            result["checks"]["resets_explained"] = (
                len(result["reconnections"]) == int(reset)
                and all(row["reason"] == "superseded_before_response" for row in result["reconnections"])
            )
            if reset:
                history = connections[-1].history
                records = incomplete_records(history)
                bridge = session._turns["bridge"]
                result["restored_history"] = history
                result["checks"]["bridge_input_retained_or_explicit_gap"] = (
                    bridge.input_transcript in json.dumps(history) if bridge.input_finalized else
                    any(row.get("turn_number") == bridge.number and row.get("input_committed")
                        and row.get("input_status") == "unavailable_after_reset" for row in records)
                )
                if interrupted:
                    result["checks"]["interrupted_step_retained_as_uncertain"] = any(
                        row.get("turn_number") == first.number
                        and row.get("assistant_output", {}).get("heard_text_status") == "unknown"
                        and "elif" in compact(row.get("assistant_output", {}).get("context_transcript", ""))
                        for row in records
                    )
            result["timing"] = turn_timings(session.diagnostics())
            result.update(classify_checks(name, result["checks"]))
    except Exception as error:
        result["error_type"] = type(error).__name__
    finally:
        try:
            try:
                if session is not None:
                    await session.close("continuity_probe_complete")
            finally:
                try:
                    if running is not None:
                        await asyncio.wait_for(running, 20)
                finally:
                    await experience.close()
            if session is not None:
                diagnostics = session.diagnostics()
                result["diagnostics"] = diagnostics
                if diagnostics["owned_tasks"] or diagnostics["domain_tasks"] or diagnostics["conversation_persistence"]["degraded"]:
                    result["passed"] = False
                    result["continuity_passed"] = False
        finally:
            result["cleanup"] = await cleanup_fixture_owners(fixtures.dsn, {owner})
            for index, pcm in enumerate(phone.audio.values(), 1):
                write_wav(output / f"{name}-forwarded-{index}.wav", bytes(pcm))
            (output / (name + ".json")).write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({key: result.get(key) for key in
                      ("case", "passed", "continuity_passed", "dictation_quality_passed",
                       "checks", "answer", "first_step", "timing", "error_type")}
                     | {"remaining_rows": result["cleanup"]["remaining_rows"]}), flush=True)
    return result


async def run(args):
    speech = {name: read_pcm(args.speech_dir / (name + ".wav")) for name in PROMPTS}
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = DatabaseFixtures(os.environ["JARVIS_TEST_DB_URL"], args.output)
    results = []
    try:
        await fixtures.connect()
        for name in args.cases.split(","):
            results.append(await run_case(name, fixtures=fixtures, speech=speech, output=args.output))
    finally:
        await fixtures.cleanup()
        (args.output / "matrix.json").write_text(json.dumps({
            "results": [{key: row.get(key) for key in (
                "case", "passed", "continuity_passed", "dictation_quality_passed", "checks", "timing", "answer_review",
            )} for row in results],
            "cleanup": fixtures.cleanup_result,
        }, indent=2, default=str) + "\n")
    return 0 if len(results) == len(args.cases.split(",")) and all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speech-dir", type=Path, required=True)
    parser.add_argument("--cases", default=",".join(CASES))
    args = parser.parse_args()
    if any(name not in CASES for name in args.cases.split(",")):
        parser.error("Unknown continuity case")
    logging.disable(logging.CRITICAL)
    raise SystemExit(asyncio.run(run(args)))
