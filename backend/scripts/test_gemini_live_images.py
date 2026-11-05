"""Probe Gemini Live vision without changing Jarvis or writing to its database.

Run from backend:
    python scripts/test_gemini_live_images.py image1.jpg image2.jpg > results.json

Uses GEMINI_API_KEY from the environment or existing JarvisSettings.
--stdin-images accepts [{"name": "...", "data": "<original JPEG base64>"}].
Audio is requested natively; its transcript and first-chunk latency are recorded.
This tests image understanding, not microphone input, playback, or barge-in.
"""
import argparse
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import websockets

MODEL = "gemini-3.1-flash-live-preview"
ENDPOINT = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
PROMPT = (
    "Are you able to read the code on the laptop screen in this image? "
    "First transcribe exactly what is visible, preserving the code as written "
    "rather than silently correcting it. Then explain any errors you can identify "
    "and how to fix them. If a character or detail is unclear, say so rather "
    "than guessing. Keep the answer focused on the visible code."
)


def api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        from app.settings import JarvisSettings
        key = JarvisSettings().GEMINI_API_KEY
    if hasattr(key, "get_secret_value"):
        key = key.get_secret_value()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    return key


async def probe(frame, key, args):
    raw = base64.b64decode(frame["data"], validate=True)
    result = {
        "image": frame["name"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "image_bytes": len(raw),
        "status": "incomplete",
        "transcript": "",
        "text_parts": [],
        "audio_bytes": 0,
        "first_audio_seconds": None,
        "usage": [],
    }
    started = time.monotonic()
    sent = None
    try:
        async with asyncio.timeout(args.timeout):
            async with websockets.connect(
                ENDPOINT,
                additional_headers={"x-goog-api-key": key},
                open_timeout=15,
                close_timeout=5,
                max_size=8 * 1024 * 1024,
            ) as ws:
                await ws.send(json.dumps({"setup": {
                    "model": "models/" + args.model,
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "maxOutputTokens": 4096,
                    },
                    "outputAudioTranscription": {},
                    "historyConfig": {"initialHistoryInClientContent": True},
                }}))
                while True:
                    event = json.loads(await ws.recv())
                    if "error" in event:
                        raise RuntimeError(json.dumps(event["error"]))
                    if "setupComplete" in event:
                        break
                result["setup_seconds"] = round(time.monotonic() - started, 3)
                await ws.send(json.dumps({"clientContent": {
                    "turns": [{"role": "user", "parts": [{"inlineData": {
                        "mimeType": "image/jpeg", "data": frame["data"],
                    }}]}],
                    "turnComplete": True,
                }}))
                sent = time.monotonic()
                await ws.send(json.dumps({"realtimeInput": {"text": args.prompt}}))
                while True:
                    event = json.loads(await ws.recv())
                    if "error" in event:
                        raise RuntimeError(json.dumps(event["error"]))
                    if "usageMetadata" in event:
                        result["usage"].append(event["usageMetadata"])
                    content = event.get("serverContent", {})
                    transcript = content.get("outputTranscription", {}).get("text", "")
                    result["transcript"] += transcript
                    for part in content.get("modelTurn", {}).get("parts", []):
                        if part.get("text") and not part.get("thought"):
                            result["text_parts"].append(part["text"])
                        inline = part.get("inlineData", {})
                        if inline.get("mimeType", "").startswith("audio/"):
                            audio = base64.b64decode(inline["data"])
                            result["audio_bytes"] += len(audio)
                            if result["first_audio_seconds"] is None:
                                result["first_audio_seconds"] = round(
                                    time.monotonic() - sent, 3
                                )
                    if content.get("interrupted"):
                        result["status"] = "interrupted"
                        break
                    if content.get("turnComplete"):
                        result["status"] = "complete"
                        break
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc).replace(key, "[REDACTED]")
    result["visual_input_tokens"] = max((
        sum(detail.get("tokenCount", 0) for detail in usage.get("promptTokensDetails", [])
            if detail.get("modality") in {"IMAGE", "VIDEO"})
        for usage in result["usage"]
    ), default=0)
    if result["status"] == "complete" and not result["visual_input_tokens"]:
        result["status"] = "unconfirmed_image_input"
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--stdin-images", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    if args.stdin_images:
        frames = json.load(sys.stdin)
    else:
        frames = [
            {"name": str(path), "data": base64.b64encode(path.read_bytes()).decode()}
            for path in args.images
        ]
    if not frames:
        parser.error("Provide JPEG paths or --stdin-images")
    key = api_key()
    results = []
    for frame in frames:
        results.append(await probe(frame, key, args))
    print(json.dumps({
        "model": args.model,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": args.prompt,
        "configuration": {
            "response_modality": "AUDIO",
            "output_audio_transcription": True,
            "thinking": "model default",
            "fresh_session_per_image": True,
            "input": "unmodified JPEG in initial history, then realtime text prompt",
        },
        "results": results,
    }, indent=2))
    return 0 if all(r["status"] == "complete" for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
