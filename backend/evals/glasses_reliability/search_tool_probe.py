"""Paid native search-tool checks through both production audio adapters.

Synthetic speech, focus-only context, and a non-playing phone sink are fixtures.
Tool definitions, execution, search state, transcripts, and experience attachment
use real Jarvis services and isolated committed database records.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import suppress
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

from app.assistant.context_providers import (
    CurrentTimeContextProvider, RecentMessagesContextProvider, SessionContextProvider,
)
from app.experience.repository import ExperienceRepository, ToolExecutionRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.memory.prospective.context import ProspectiveContextProvider
from app.memory.prospective.models import ProspectiveContext
from app.memory.prospective.repository import ProspectiveRepository
from app.memory.prospective.service import ProspectiveMemoryService
from app.memory.prospective.tools import ProspectiveToolModule
from app.realtime.gemini.session import GeminiAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.realtime.session_manager import RealtimeSessionManager
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from evals.gemini_live.interruption import DatabaseFixtures
from evals.gemini_live.session_probe import ProbePhone
from evals.gemini_live.transport_probe import read_pcm, write_wav
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


PROMPTS = {
    "lost": "Help me find my silver keys with the black fob.",
    "monitor": "Keep watching the doorway and tell me every time someone comes in. I want continuous monitoring, not just the first time.",
    "stop": "Stop looking for the merch box. I do not want to search for it anymore.",
    "found": "I have found the merch box myself. We do not need to keep looking.",
    "switch": "Let us stop looking for the merch box and find my silver keys with the black fob instead.",
}


class NativePhone(ProbePhone):
    """Same fixture sink for GPT's encoded events and Gemini's JSON events."""

    async def send_text(self, encoded):
        await self.send_json(json.loads(encoded))


class FocusOnlyContext:
    def __init__(self, repository):
        self.repository = repository

    async def build(self, *, owner_id, generated_at=None):
        return ProspectiveContext(
            visual_focus=await self.repository.get_active_visual_focus(owner_id=owner_id),
            conditional_reminders=(), goals=(), commitments=(), generated_at=generated_at or datetime.now(UTC),
        )


def assess(case, *, before, after, executions):
    names = [row["tool_name"] for row in executions]
    if case in {"stop", "found"}:
        return {
            "real_clear_handler_called": "clear_visual_focus" in names,
            "no_active_focus": after is None,
        }
    expected_mode = "monitor" if case == "monitor" else "find_once"
    checks = {
        "real_set_handler_called": "set_visual_focus" in names,
        "active_focus_exists": after is not None,
        "correct_mode": after is not None and after.mode == expected_mode,
    }
    if case in {"lost", "switch"}:
        # This is a fixture oracle, never application intent parsing. Keep the
        # full target/description in the report for independent semantic review.
        checks["keys_target"] = after is not None and "key" in after.target.casefold()
    if case == "switch":
        checks["replaced_old_generation"] = (
            before is not None and after is not None and after.id != before.id
        )
    return checks


async def run_case(*, provider, case, fixtures, settings, speech_dir, output):
    label = provider + "-" + case
    owner, session_id = await fixtures.create("search-tools-" + label)
    sessions = SessionRepository(fixtures.pool)
    prospective = ProspectiveRepository(fixtures.pool)
    manager = RealtimeSessionManager()
    service = ProspectiveMemoryService(repository=prospective,
                                       realtime_session_manager=manager, settings=settings)
    experience = ExperienceEventService(
        repository=ExperienceRepository(fixtures.pool),
        segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)),
    )
    phone = NativePhone()
    audio_session_id = str(uuid4())
    session = running = None
    result = {"case": case, "provider": provider, "passed": False,
              "device_playback": "not_tested", "prompt": PROMPTS[case]}
    before = None
    try:
        async with asyncio.timeout(100):
            if case in {"stop", "found", "switch"}:
                before = await service.set_visual_focus(
                    owner_id=owner, target="merch box", description="Eras Tour merchandise box",
                    purpose="Tell me when you find it", mode="find_once", expires_in_seconds=3600,
                )
                await fixtures.message(owner, session_id, "Help me find my Eras Tour merch box.")
            cls = GeminiAudioSession if provider == "gemini" else RealtimeAudioSession
            session = cls(
                websocket=phone,
                context=RealtimeSessionContext(
                    audio_session_id=audio_session_id, owner_id=owner,
                    jarvis_session_id=session_id, client_kind="phone_audio",
                    session=await sessions.get_session(owner, session_id),
                ),
                settings=settings, repository=sessions,
                tool_modules=(ProspectiveToolModule(service=service),),
                context_providers=(
                    CurrentTimeContextProvider(), SessionContextProvider(),
                    RecentMessagesContextProvider(12),
                    ProspectiveContextProvider(FocusOnlyContext(prospective)),
                ),
                tool_execution_repository=ToolExecutionRepository(fixtures.pool),
                experience_event_service=experience,
            )
            await manager.register(audio_session_id=audio_session_id, owner_id=owner,
                                   jarvis_session_id=session_id, session=session)
            running = asyncio.create_task(session.run())

            async def until(predicate):
                while not predicate():
                    if running.done() or phone.closed:
                        raise RuntimeError("Audio adapter ended before the expected boundary")
                    await asyncio.sleep(.01)

            await until(lambda: any(e["type"] == "session.created" for e in phone.events))
            pcm = read_pcm(speech_dir / (case + ".wav"))
            start = len(phone.events)
            for offset in range(0, len(pcm), 4800):
                chunk = pcm[offset:offset + 4800]
                await phone.send("audio.input.append", segment_id=case,
                                 audio_base64=base64.b64encode(chunk).decode())
                await asyncio.sleep(len(chunk) / 48000)
            await phone.send("audio.input.commit", segment_id=case)
            await phone.send("response.create")
            if provider == "gemini":
                await until(lambda: case in session._turns and session._turns[case].output.completed)
                await until(lambda: not session._domain_tasks)
            else:
                def finished():
                    events = phone.events[start:]
                    audio_ids = {e.get("response_id") for e in events if e["type"] == "audio.output.delta"}
                    return (not session._normal_turn_pending and any(
                        e["type"] == "response.done" and e.get("response_id") in audio_ids
                        for e in events
                    ))
                await until(finished)
            after = await prospective.get_active_visual_focus(owner_id=owner)
            async with fixtures.pool.acquire() as connection:
                executions = [dict(row) for row in await connection.fetch(
                    "select tool_name, arguments, status from jarvis_tool_executions "
                    "where owner_id=$1 and session_id=$2 order by started_at, id",
                    owner, session_id,
                )]
                old_cleared = (
                    await connection.fetchval(
                        "select cleared_at is not null from jarvis_visual_focuses where owner_id=$1 and id=$2",
                        owner, before.id,
                    ) if before is not None else None
                )
            checks = assess(case, before=before, after=after, executions=executions)
            if before is not None:
                checks["old_focus_cleared"] = old_cleared is True
            result.update(checks=checks, passed=all(checks.values()), executions=executions,
                          focus=after.as_prompt_payload() if after is not None else None)
    except Exception as error:
        result["error_type"] = type(error).__name__
    finally:
        # Stop producers before owner cleanup, even when setup or a native turn
        # fails. The outer fixture manifest remains a recoverable cleanup record.
        try:
            if session is not None:
                await session.close("search_tool_probe_finished")
            if running is not None:
                with suppress(asyncio.CancelledError):
                    await running
            await experience.close()
            await manager.unregister(audio_session_id=audio_session_id, owner_id=owner,
                                     jarvis_session_id=session_id)
            result["events"] = [
                e if e["type"] != "error" else {"type": "error", "code": e.get("code")}
                for e in phone.events if e["type"] in {"transcript.done", "error"}
            ]
            for index, pcm in enumerate(phone.audio.values()):
                write_wav(output / (label + "-answer-" + str(index) + ".wav"), bytes(pcm))
        finally:
            result["cleanup"] = await cleanup_fixture_owners(fixtures.dsn, {owner})
            (output / (label + ".json")).write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({
        "provider": provider, "case": case, "passed": result["passed"],
        "checks": result.get("checks"), "error_type": result.get("error_type"),
        "focus": result.get("focus"), "events": result.get("events"),
        "cleanup_remaining": result["cleanup"]["remaining_rows"],
    }, default=str), flush=True)
    return result


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = DatabaseFixtures(os.environ["JARVIS_TEST_DB_URL"], args.output)
    settings = JarvisSettings(_env_file=None)
    providers = ["openai", "gemini"] if args.provider == "both" else [args.provider]
    results = []
    try:
        await fixtures.connect()
        for provider in providers:
            for case in args.cases.split(","):
                results.append(await run_case(
                    provider=provider, case=case, fixtures=fixtures, settings=settings,
                    speech_dir=args.speech_dir, output=args.output,
                ))
    finally:
        await fixtures.cleanup()
        summary = {
            "results": [{"provider": r["provider"], "case": r["case"],
                         "passed": r["passed"], "checks": r.get("checks")} for r in results],
            "configured_models": {
                key: value for key, value in settings.model_dump().items()
                if key.endswith("_MODEL") and ("REALTIME" in key or "LIVE" in key)
            },
            "cleanup": fixtures.cleanup_result,
        }
        (args.output / "matrix.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return 0 if results and all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speech-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("both", "openai", "gemini"), default="both")
    parser.add_argument("--cases", default=",".join(PROMPTS))
    args = parser.parse_args()
    if any(case not in PROMPTS for case in args.cases.split(",")):
        parser.error("Unknown case")
    logging.disable(logging.CRITICAL)
    raise SystemExit(asyncio.run(run(args)))
