"""Paid native proactive-speech checks with disposable, tracked database owners.

Reminder outcomes use the real repository. Prospective outcomes observe the
existing callback contract; no physical playback or prospective scheduler is faked.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import defaultdict
from contextlib import suppress
from datetime import UTC, datetime, timedelta
import hashlib
import json
import logging
from pathlib import Path
import sys
from uuid import uuid4

from app.assistant.context_providers import CurrentTimeContextProvider, RecentMessagesContextProvider, SessionContextProvider
from app.experience.repository import ExperienceRepository, ToolExecutionRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.realtime.gemini.client import GeminiLiveConnection
from app.realtime.gemini.events import decode_pcm16
from app.realtime.gemini.session import GeminiAudioSession
from app.realtime.models import RealtimeSessionContext
from app.reminders.repository import ReminderRepository
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tools.module import FunctionToolModule
from evals.gemini_live.interruption import DatabaseFixtures
from evals.gemini_live.session_probe import ProbePhone
from evals.gemini_live.transport_probe import read_pcm, redact_native, write_wav
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


CASES = ("reminder-completed", "visual-completed", "prospective-completed",
         "prospective-interrupted", "reminder-interrupted", "reminder-interrupts-assistant")
SHORT_MESSAGE = "Synthetic reminder: your test phrase is copper orchard."
FOLLOWUP_SENTENCE = "The pineapple is on the table."
LONG_MESSAGE = ("This is a synthetic appointment reminder. Before leaving, check that you have your keys, "
                "your phone, and your notebook. The notebook contains the questions you wanted to ask. "
                "You planned to discuss the project schedule, the next review, and the remaining tasks. "
                "You can also bring a bottle of water and check the weather before choosing a jacket. "
                "There is no need to rush. This message is only a test of an interruptible proactive reminder.")


async def run_case(*, label, fixtures, keys, speech_dir, output):
    owner, session_id = await fixtures.create("proactive-" + label)
    repository, reminders = SessionRepository(fixtures.pool), ReminderRepository(fixtures.pool)
    experience = ExperienceEventService(
        repository=ExperienceRepository(fixtures.pool),
        segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)),
    )
    registry = fixtures.registry(owner, session_id)
    phone, trace, connections = ProbePhone(), [], []
    native_audio = defaultdict(bytearray)
    outcomes = []

    class TracedConnection(GeminiLiveConnection):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            connections.append(self)
            self.generation, self.native_turn = len(connections), 1

        async def receive_event(self):
            event = await super().receive_event()
            trace.append({"generation": self.generation, "native_turn": self.native_turn,
                          "event": redact_native(event)})
            content = event.get("serverContent", {})
            for part in content.get("modelTurn", {}).get("parts", []):
                inline = part.get("inlineData", {})
                if inline.get("mimeType", "").startswith("audio/"):
                    native_audio[(self.generation, self.native_turn)].extend(
                        decode_pcm16(inline.get("data"), max_bytes=8 * 1024 * 1024))
            if content.get("turnComplete"):
                self.native_turn += 1
            return event

    session = GeminiAudioSession(
        websocket=phone, context=RealtimeSessionContext(
            audio_session_id=str(uuid4()), owner_id=owner, jarvis_session_id=session_id,
            client_kind="phone_audio", session=await repository.get_session(owner, session_id),
        ), settings=JarvisSettings(_env_file=None, GEMINI_API_KEY=keys["gemini"]),
        repository=repository, reminder_repository=reminders, connection_factory=TracedConnection,
        tool_modules=(FunctionToolModule("real_memory_fixture", lambda _: registry.list()),),
        context_providers=(CurrentTimeContextProvider(), SessionContextProvider(), RecentMessagesContextProvider(12)),
        tool_execution_repository=ToolExecutionRepository(fixtures.pool), experience_event_service=experience,
    )
    running = asyncio.create_task(session.run())
    result = {"case": label, "passed": False, "device_playback": "not_tested",
              "prospective_scheduler": "not_exercised; native callback contract observed"}
    reminder_event = None

    async def until(predicate):
        async with asyncio.timeout(40):
            while not predicate():
                if running.done() or phone.closed:
                    raise RuntimeError("Native proactive run ended before the expected boundary")
                await asyncio.sleep(.005)

    async def speak(name, segment):
        pcm = read_pcm(speech_dir / (name + ".wav"))
        for offset in range(0, len(pcm), 4800):
            chunk = pcm[offset:offset + 4800]
            await phone.send("audio.input.append", segment_id=segment,
                             audio_base64=base64.b64encode(chunk).decode())
            await asyncio.sleep(len(chunk) / 48000)
        await phone.send("audio.input.commit", segment_id=segment)
        await phone.send("response.create")

    async def outcome(status, response_id):
        outcomes.append({"status": status, "response_id": response_id})

    try:
        async with asyncio.timeout(150):
            await phone.wait_for("session.created")
            interrupted = label in {"prospective-interrupted", "reminder-interrupted"}
            message = LONG_MESSAGE if interrupted else SHORT_MESSAGE
            if label.startswith("reminder"):
                reminder = await reminders.create_reminder(
                    owner_id=owner, session_id=session_id, title="Synthetic proactive fixture",
                    details="Isolated native audio acceptance", remind_at=datetime.now(UTC) + timedelta(days=30),
                    condition=None, metadata={"synthetic_fixture": True},
                )
                token = uuid4()
                # Future-due owner-scoped fixture cannot be claimed by the live
                # scheduler. Seed only its claim, then use the real finalizer.
                async with fixtures.pool.acquire() as conn:
                    await conn.execute("update jarvis_reminders set trigger_claim_token=$3, trigger_claimed_at=now() "
                                       "where owner_id=$1 and id=$2", owner, reminder.id, token)
                reminder_event = await reminders.finalize_trigger(
                    owner_id=owner, reminder_id=reminder.id, claim_token=token,
                    session_id=session_id, message=message,
                )
                if label == "reminder-interrupts-assistant":
                    await speak("long", "ordinary")
                    await until(lambda: bool(phone.audio))
                    assert not session._turns["ordinary"].output.completed
                await session.speak_reminder(message=message, reminder_id=reminder.id,
                                             reminder_event_id=reminder_event.id)
                if interrupted:
                    await until(lambda: bool(phone.audio))
                    await speak("barge_sentence", "barge")
                    await until(lambda: "barge" in session._turns and session._turns["barge"].output.completed)
                    await phone.wait_for("reminder.interrupted")
                    assert not any(e["type"] == "reminder.done" for e in phone.events)
                else:
                    await phone.wait_for("reminder.done")
                events = await reminders.list_events(owner_id=owner, limit=10)
                event = next(item for item in events if item.id == reminder_event.id)
                assert event.event_type == ("interrupted" if interrupted else "delivered_to_session")
                assert event.provider_response_id
                result["reminder_event"] = event.model_dump(mode="json")
                if label == "reminder-interrupts-assistant":
                    assert session._turns["ordinary"].output.interrupted
            elif label == "visual-completed":
                session.notify_visual_frame_available()
                accepted = await session.speak_visual_interrupt(
                    owner_id=owner, jarvis_session_id=session_id, source_received_at=datetime.now(UTC),
                    message=SHORT_MESSAGE, interrupt_id=uuid4(),
                )
                assert accepted
                await phone.wait_for("visual.interrupt.delivered")
            else:
                accepted = await session.speak_prospective_interrupt(
                    message=message, event_id=uuid4(), subject_type="reminder", subject_id=uuid4(),
                    on_outcome=outcome,
                )
                assert accepted
                if interrupted:
                    await until(lambda: bool(phone.audio))
                    assert not outcomes
                    await speak("barge_sentence", "barge")
                    await until(lambda: "barge" in session._turns and session._turns["barge"].output.completed)
                else:
                    await until(lambda: bool(outcomes))
                assert len(outcomes) == 1
                assert outcomes[0]["status"] == ("interrupted" if interrupted else "delivered")
            if interrupted:
                actual = session._turns["barge"].output.transcript
                assert "".join(c.lower() for c in actual if c.isalnum()) == "".join(
                    c.lower() for c in FOLLOWUP_SENTENCE if c.isalnum())
            else:
                completed = next(t for t in session._turns.values() if t.proactive is not None)
                # Exact synthetic fixture expectation, not production output
                # processing: an old answer appended after the message fails.
                normalized = lambda value: "".join(c.lower() for c in value if c.isalnum())
                assert normalized(completed.output.transcript) == normalized(SHORT_MESSAGE)
            snapshot = await fixtures.snapshot(owner, session_id)
            assert not snapshot["memories"]
            assert not any(call.admitted for call in session.operations.calls.values())
            result["memory_count"] = len(snapshot["memories"])
            result["passed"] = True
    except Exception as error:
        result["error_type"] = type(error).__name__
    finally:
        await session.close("proactive_probe_finished")
        with suppress(asyncio.CancelledError):
            await running
        await experience.close()
        diagnostics = session.diagnostics()
        comparisons = []
        for turn in session._turns.values():
            if turn.output.completed and not turn.output.interrupted and turn.output.forwarded_audio_bytes:
                pcm = bytes(phone.audio[turn.output.item_id])
                exact = any(pcm == bytes(native) for native in native_audio.values())
                comparisons.append({"response_id": turn.output.response_id, "bytes": len(pcm),
                                    "sha256": hashlib.sha256(pcm).hexdigest(), "native_pcm_exact": exact})
        if (not comparisons or not all(c["native_pcm_exact"] for c in comparisons)
                or diagnostics["domain_tasks"] or diagnostics["owned_tasks"]
                or diagnostics["proactive"]["degraded"]
                or diagnostics["conversation_persistence"]["degraded"]):
            result["passed"] = False
        result.update({"diagnostics": diagnostics, "pcm_comparisons": comparisons, "outcomes": outcomes,
                       "native_trace": trace,
                       "app_events": [e for e in phone.events if e["type"] != "audio.output.delta"]})
        for (generation, native_turn), pcm in native_audio.items():
            write_wav(output / f"{label}-native-g{generation}-t{native_turn}.wav", bytes(pcm))
        for index, pcm in enumerate(phone.audio.values(), 1):
            write_wav(output / f"{label}-app-{index}.wav", bytes(pcm))
        result["cleanup"] = await cleanup_fixture_owners(keys["database"], {owner})
        (output / f"{label}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({"case": label, "passed": result["passed"], "error_type": result.get("error_type"),
                      "outcomes": outcomes, "pcm_comparisons": comparisons,
                      "cleanup_remaining": result["cleanup"]["remaining_rows"],
                      "answers": [e for e in phone.events if e["type"] in {"transcript.done", "error"}]}), flush=True)
    return result


async def run(args, keys):
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = DatabaseFixtures(keys["database"], args.output)
    results = []
    try:
        await fixtures.connect()
        for label in CASES:
            result = await run_case(label=label, fixtures=fixtures, keys=keys,
                                    speech_dir=args.speech_dir, output=args.output)
            results.append({"case": label, "passed": result["passed"]})
    finally:
        await fixtures.cleanup()
        (args.output / "matrix.json").write_text(json.dumps({"results": results, "cleanup": fixtures.cleanup_result}, indent=2) + "\n")
    return 0 if len(results) == len(CASES) and all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speech-dir", type=Path, required=True)
    logging.disable(logging.CRITICAL)
    raise SystemExit(asyncio.run(run(parser.parse_args(), json.load(sys.stdin))))
