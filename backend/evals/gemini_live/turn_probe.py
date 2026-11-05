"""Opt-in native interruption ordering probe; no domain tools or database access."""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import suppress
import json
from pathlib import Path
import sys
import time

from app.realtime.gemini.client import GeminiLiveConnection, GeminiTransportError
from app.realtime.gemini.config import GeminiLiveConfig
from app.realtime.gemini.events import GeminiOutputTurn, decode_pcm16
from app.realtime.gemini.instructions import build_gemini_instructions
from evals.gemini_live.transport_probe import read_pcm, redact_native, write_wav


async def probe(*, key: str, speech_dir: Path, output: Path, supersede: bool, fence: bool = False) -> dict:
    connection = GeminiLiveConnection(config=GeminiLiveConfig(api_key=key))
    started = time.monotonic()
    case = "supersede_fenced" if fence else ("supersede_preparation" if supersede else "barge_in")
    turns = [GeminiOutputTurn(case, 1, i) for i in ((2,) if supersede and not fence else (1, 2))]
    raw_audio = [bytearray() for _ in turns]
    translated_audio = [bytearray() for _ in turns]
    trace = []
    first_audio = asyncio.Event()
    complete = asyncio.Event()
    current = 0
    error_code = None

    def record(kind, **fields):
        trace.append({"at_seconds": round(time.monotonic() - started, 6), "kind": kind, **fields})

    async def receive():
        nonlocal current
        while True:
            event = await connection.receive_event()
            record("provider", event=redact_native(event))
            content = event.get("serverContent", {})
            if current >= len(turns):
                continue
            turn = turns[current]
            for part in content.get("modelTurn", {}).get("parts", []):
                inline = part.get("inlineData", {})
                if inline.get("mimeType", "").startswith("audio/"):
                    raw_audio[current].extend(decode_pcm16(inline.get("data"), max_bytes=8 * 1024 * 1024))
                    first_audio.set()
            for translated in turn.consume(event):
                if translated["type"] == "audio.output.delta":
                    translated_audio[current].extend(decode_pcm16(translated["audio_base64"], max_bytes=8 * 1024 * 1024))
                    turn.record_forwarded(translated)
                else:
                    record("adapter", event=translated)
            if content.get("turnComplete"):
                current += 1
                if current == len(turns):
                    complete.set()

    async def send_audio(name):
        pcm = read_pcm(speech_dir / f"{name}.wav")
        for offset in range(0, len(pcm), 4800):
            chunk = pcm[offset:offset + 4800]
            await connection.send_event({"realtimeInput": {"audio": {
                "data": base64.b64encode(chunk).decode(), "mimeType": "audio/pcm;rate=24000",
            }}})
            await asyncio.sleep(len(chunk) / 48000)

    receiver = None
    try:
        async with asyncio.timeout(65):
            await connection.connect(instructions=build_gemini_instructions())
            receiver = asyncio.create_task(receive())
            await connection.send_event({"realtimeInput": {"activityStart": {}}})
            record("first_activity_start")
            await send_audio("explanation")
            if not supersede or fence:
                await connection.send_event({"realtimeInput": {"activityEnd": {}}})
                record("first_activity_end")
                if not supersede:
                    await first_audio.wait()
                    await asyncio.sleep(.35)
                for event in turns[0].interrupt(source="client"):
                    record("adapter", event=event)
            await connection.send_event({"realtimeInput": {"activityStart": {}}})
            record("second_activity_start", superseded_before_release=supersede)
            await send_audio("marker")
            await connection.send_event({"realtimeInput": {"text": json.dumps({
                "kind": "jarvis_application_context",
                "current_test_marker": "silver meadow",
                "previous_request": "cancelled; no operation was admitted",
                "instruction": "Answer only the latest spoken question, not the cancelled explanation.",
            })}})
            await connection.send_event({"realtimeInput": {"activityEnd": {}}})
            record("second_activity_end")
            done = asyncio.create_task(complete.wait())
            try:
                finished, _ = await asyncio.wait({done, receiver}, return_when=asyncio.FIRST_COMPLETED)
                if receiver in finished:
                    await receiver
                await done
                await asyncio.sleep(.5)
            finally:
                done.cancel()
                with suppress(asyncio.CancelledError):
                    await done
    except Exception as error:
        error_code = error.code if isinstance(error, GeminiTransportError) else type(error).__name__
    finally:
        if receiver:
            receiver.cancel()
            with suppress(asyncio.CancelledError, GeminiTransportError):
                await receiver
        await connection.close()
    results = []
    for index, turn in enumerate(turns):
        name = f"{case}-{turn.turn_number}"
        write_wav(output / f"{name}.provider.wav", bytes(raw_audio[index]))
        write_wav(output / f"{name}.adapter.wav", bytes(translated_audio[index]))
        results.append({**turn.diagnostics(), "transcript": turn.transcript})
    result = {"case": case, "error_code": error_code, "turns": results, "trace": trace}
    (output / f"{case}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
    return result


async def main(args, keys):
    args.output.mkdir(parents=True, exist_ok=False)
    for case in args.cases.split(","):
        if case not in {"barge_in", "supersede_preparation", "supersede_fenced"}:
            raise ValueError("Unknown native turn probe")
        await probe(
            key=keys["gemini"], speech_dir=args.speech_dir, output=args.output,
            supersede=case != "barge_in", fence=case == "supersede_fenced",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="barge_in,supersede_preparation,supersede_fenced")
    args = parser.parse_args()
    asyncio.run(main(args, json.load(sys.stdin)))
