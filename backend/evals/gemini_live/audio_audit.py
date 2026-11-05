"""Independently transcribe explicitly selected probe WAVs, without transcript hints.

Paid and opt-in. The existing OpenAI key is supplied as JSON on stdin. No Gemini
transcript is sent to the recognizer, and no production audio path is changed.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import wave

from openai import OpenAI


def inspect_wav(path: Path) -> dict:
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 24000):
            raise ValueError("Expected a native 24 kHz mono PCM16 probe recording")
        frames = audio.getnframes()
        if frames <= 0 or frames * 2 > 24 * 1024 * 1024:
            raise ValueError("Probe recording is empty or exceeds the upload budget")
        pcm = audio.readframes(frames)
    return {
        "audio_file": path.name, "pcm_bytes": len(pcm),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "duration_seconds": frames / 24000,
    }


def audit_files(*, client, files: list[Path], output: Path, model: str) -> dict:
    if output.exists():
        raise ValueError("Refusing to overwrite existing audio audit")
    report = {
        "created_at": datetime.now(UTC).isoformat(), "recognizer_model": model,
        "scope": "Independent ASR of native provider PCM, not device-playback evidence",
        "prompt_or_expected_transcript_supplied": False, "results": [],
    }
    for path in files:
        result = {"audio_file": path.name}
        try:
            result.update(inspect_wav(path))
            with path.open("rb") as audio:
                transcription = client.audio.transcriptions.create(
                    model=model, file=audio,
                )
            result.update(transcript=transcription.text, status="transcribed")
        except Exception as error:
            result.update(status="error", error_type=type(error).__name__)
        report["results"].append(result)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o-transcribe")
    args = parser.parse_args()
    key = json.load(sys.stdin)["openai"]
    with OpenAI(api_key=key, timeout=60, max_retries=0) as api:
        report = audit_files(client=api, files=args.files, output=args.output, model=args.model)
    sys.exit(0 if all(item["status"] == "transcribed" for item in report["results"]) else 1)
