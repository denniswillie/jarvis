"""Opt-in, paid Stage 2 native-audio probe using production transport/translation.

No database, router, domain tool, or production configuration is touched. The sink
is a WAV writer, not a phone: forwarded bytes here are adapter-output evidence,
not proof of device playback. Independent audio review is still required.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import suppress
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from uuid import uuid4
import wave

from app.realtime.gemini.client import GeminiLiveConnection, GeminiTransportError
from app.realtime.gemini.config import GeminiLiveConfig
from app.realtime.gemini.events import GeminiOutputTurn, decode_pcm16
from app.realtime.gemini.instructions import build_gemini_instructions


def read_pcm(path: Path, rate: int = 24000) -> bytes:
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, rate):
            raise ValueError("Probe WAV must match the existing 24 kHz mono PCM16 phone format")
        return audio.readframes(audio.getnframes())


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(pcm)


def redact_native(value):
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if key in {"data", "newHandle", "handle"} else redact_native(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_native(item) for item in value]
    return value


async def run_case(*, key: str, model: str, speech: bytes, case: str, output: Path) -> dict:
    started = time.monotonic()
    connection = GeminiLiveConnection(config=GeminiLiveConfig(api_key=key, model=model))
    turn = GeminiOutputTurn(str(uuid4()), 1, 1)
    trace = []
    native_audio = bytearray()
    forwarded_audio = bytearray()
    input_transcript = []
    completion = asyncio.Event()
    first_audio_at = None
    activity_end_at = None
    terminal_count = 0
    extra_audio_parts = 0
    marker = "copper lantern"
    context = json.dumps({
        "kind": "jarvis_application_context", "current_test_marker": marker,
        "current_visual_evidence": "unavailable",
        "instruction": "Use this context to answer the spoken question; do not acknowledge it separately.",
    })

    def record(kind, **fields):
        trace.append({"at_seconds": round(time.monotonic() - started, 6), "kind": kind, **fields})

    async def receive():
        nonlocal first_audio_at, terminal_count, extra_audio_parts
        while True:
            event = await connection.receive_event()
            record("provider", event=redact_native(event))
            content = event.get("serverContent", {})
            if content.get("inputTranscription"):
                input_transcript.append(content["inputTranscription"].get("text", ""))
            for part in content.get("modelTurn", {}).get("parts", []):
                inline = part.get("inlineData", {})
                if inline.get("mimeType", "").startswith("audio/"):
                    pcm = decode_pcm16(inline.get("data"), max_bytes=8 * 1024 * 1024)
                    native_audio.extend(pcm)
                    if first_audio_at is None:
                        first_audio_at = time.monotonic()
                    if turn.completed:
                        extra_audio_parts += 1
                    record("provider_pcm", bytes=len(pcm), sha256=hashlib.sha256(pcm).hexdigest())
            for translated in turn.consume(event):
                if translated["type"] == "audio.output.delta":
                    pcm = decode_pcm16(translated["audio_base64"], max_bytes=8 * 1024 * 1024)
                    forwarded_audio.extend(pcm)
                    turn.record_forwarded(translated)
                    record("adapter_pcm", bytes=len(pcm), item_id=translated["item_id"])
                else:
                    record("adapter", event=translated)
            if content.get("turnComplete"):
                terminal_count += 1
                completion.set()

    receiver = None
    error_code = None
    try:
        async with asyncio.timeout(65):
            await connection.connect(instructions=build_gemini_instructions())
            record("setup_ready")
            receiver = asyncio.create_task(receive())
            await connection.send_event({"realtimeInput": {"activityStart": {}}})
            record("activity_start")
            if case == "marker_early":
                await connection.send_event({"realtimeInput": {"text": context}})
                record("context_written")
            chunk_size = 4800  # 100 ms at the frozen 24 kHz mono PCM16 input rate.
            for offset in range(0, len(speech), chunk_size):
                chunk = speech[offset:offset + chunk_size]
                await connection.send_event({"realtimeInput": {"audio": {
                    "data": base64.b64encode(chunk).decode(), "mimeType": "audio/pcm;rate=24000",
                }}})
                await asyncio.sleep(len(chunk) / 48000)
            record("last_input_audio_written", bytes=len(speech))
            if case in {"marker_end", "marker_pause"}:
                await connection.send_event({"realtimeInput": {"text": context}})
                record("context_written")
                if case == "marker_pause":
                    await asyncio.sleep(.35)
            activity_end_at = time.monotonic()
            await connection.send_event({"realtimeInput": {"activityEnd": {}}})
            record("activity_end_written")
            wait_complete = asyncio.create_task(completion.wait())
            try:
                finished, _ = await asyncio.wait({receiver, wait_complete}, return_when=asyncio.FIRST_COMPLETED)
                if receiver in finished:
                    await receiver
                await wait_complete
                # Observe, rather than assume, that context did not create a second answer.
                await asyncio.sleep(1.0)
                if receiver.done():
                    await receiver
            finally:
                wait_complete.cancel()
                with suppress(asyncio.CancelledError):
                    await wait_complete
    except Exception as error:
        error_code = error.code if isinstance(error, GeminiTransportError) else type(error).__name__
        record("error", code=error_code)
    finally:
        if receiver is not None:
            receiver.cancel()
            with suppress(asyncio.CancelledError, GeminiTransportError):
                await receiver
        await connection.close()

    write_wav(output.with_suffix(".provider.wav"), bytes(native_audio))
    write_wav(output.with_suffix(".adapter.wav"), bytes(forwarded_audio))
    result = {
        "case": case, "model": model, "input_sample_rate": 24000,
        "input_bytes": len(speech), "input_sha256": hashlib.sha256(speech).hexdigest(),
        "error_code": error_code, "input_transcript": "".join(input_transcript),
        "output_transcript": turn.transcript, "diagnostics": turn.diagnostics(),
        "terminal_count": terminal_count, "extra_audio_parts": extra_audio_parts,
        "first_audio_after_activity_end_seconds": (
            round(first_audio_at - activity_end_at, 6)
            if first_audio_at is not None and activity_end_at is not None else None
        ),
        "context_marker_in_transcript": marker in turn.transcript.casefold() if case.startswith("marker") else None,
        "adapter_pcm_matches_provider": bytes(native_audio) == bytes(forwarded_audio),
        "spoken_completeness": "not_assessed_requires_independent_audio_review",
        "device_playback": "not_tested",
        "trace": trace,
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


async def main(args, keys):
    args.output.mkdir(parents=True, exist_ok=False)
    speech = {name: read_pcm(args.speech_dir / f"{name}.wav") for name in ("marker", "explanation")}
    results = []
    for case in args.cases.split(","):
        if case not in {"marker_early", "marker_end", "marker_pause", "explanation"}:
            raise ValueError("Unknown transport probe case")
        for index in range(args.repeats):
            name = f"{case}-{index + 1}"
            print(f"Running {name}", flush=True)
            result = await run_case(
                key=keys["gemini"], model=args.model, speech=speech["explanation" if case == "explanation" else "marker"],
                case=case, output=args.output / name,
            )
            summary = {k: v for k, v in result.items() if k != "trace"}
            results.append(summary)
            print(json.dumps(summary), flush=True)
            (args.output / "report.json").write_text(json.dumps({
                "created_at": datetime.now(UTC).isoformat(),
                "scope": "production transport and output translator; not a complete production session or device test",
                "database_access": False, "production_configuration_changed": False,
                "results": results,
            }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemini-3.1-flash-live-preview")
    parser.add_argument("--cases", default="marker_early,marker_end,marker_pause,explanation")
    parser.add_argument("--repeats", type=int, default=1)
    arguments = parser.parse_args()
    asyncio.run(main(arguments, json.load(sys.stdin)))
