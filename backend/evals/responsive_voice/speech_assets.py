"""Prepare private, reusable speech fixtures without microphone or provider calls.

Reuse the existing local macOS speech helper. PCM energy boundaries support
synthetic-input latency accounting; they are not semantic word alignments.
"""
from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
import re
import sys
import wave

from evals.responsive_voice.synthetic_webrtc import make_speech
from evals.responsive_voice.video_assets import sha256_file


DEFAULT_UTTERANCES = (
    ("current_view", "What is in front of me?"),
    ("current_activity", "What am I doing right now?"),
    ("remember_placement", "Please remember where I put this object."),
    ("historical_recall", "Where did I last leave that object?"),
    ("goal_correction", "Actually, stop looking for that. Focus on the tool in my hand."),
    ("quiet_mode", "I need to focus now. Please don't disturb me until I tell you."),
    ("quiet_direct_question", "What am I holding right now?"),
    ("resume_updates", "You can interrupt me again. Tell me only what is still relevant."),
    ("cancel_task", "Cancel that background task, but keep listening."),
    ("evidence_question", "Did you actually see me turn that off?"),
)


def inspect_pcm(path: Path, *, threshold=128):
    if type(threshold) is not int or not 1 <= threshold <= 32767:
        raise ValueError("speech_energy_threshold_invalid")
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (
                1, 2, 48000, "NONE"):
            raise ValueError("speech_fixture_requires_pcm16_mono_48000")
        count = source.getnframes()
        if not 0 < count <= 15 * 48000:
            raise ValueError("speech_fixture_duration_invalid")
        raw = source.readframes(count)
    if len(raw) != count * 2:
        raise ValueError("speech_fixture_truncated")
    samples = array("h", raw)
    if sys.byteorder != "little":
        samples.byteswap()
    first = next((i for i, value in enumerate(samples) if abs(value) >= threshold), None)
    last = next((count - i for i, value in enumerate(reversed(samples)) if abs(value) >= threshold), None)
    if first is None:
        raise ValueError("speech_fixture_has_no_signal")
    return {"sample_rate_hz": 48000, "channels": 1, "sample_width_bytes": 2,
        "sample_count": count, "duration_ms": count * 1000 / 48000,
        "first_active_sample": first, "last_active_sample_exclusive": last,
        "energy_threshold_abs_pcm16": threshold,
        "input_active_end_ms": last * 1000 / 48000,
        "trailing_below_threshold_ms": (count - last) * 1000 / 48000,
        "boundary_kind": "pcm_energy_threshold_not_semantic_alignment"}


def prepare_speech_bundle(output: Path, utterances=DEFAULT_UTTERANCES):
    utterances = tuple(utterances)
    if not 1 <= len(utterances) <= 16:
        raise ValueError("speech_fixture_clip_budget_invalid")
    identities = set()
    for identity, text in utterances:
        if (not isinstance(identity, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", identity)
                or identity in identities):
            raise ValueError("speech_fixture_identity_invalid")
        if not isinstance(text, str) or not text.strip() or len(text) > 800:
            raise ValueError("speech_fixture_text_invalid")
        identities.add(identity)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    clips, duration_ms = [], 0
    for identity, text in utterances:
        # Existing helper writes only files, never plays speech through speakers.
        path, _ = make_speech(output, question=text, stem=identity)
        audio = inspect_pcm(path)
        duration_ms += audio["duration_ms"]
        if duration_ms > 180000:
            raise ValueError("speech_fixture_total_duration_exceeded")
        clips.append({"id": identity, "text": text, "relative_path": path.name,
                      "sha256": sha256_file(path), **audio})
    manifest = {"format": "jarvis.speech_assets.v1", "generator": "macos_say",
        "voice": "Samantha", "rate_words_per_minute": 180, "clips": clips,
        "total_duration_ms": duration_ms, "provider_calls": 0,
        "microphone_used": False, "frontend_replay_executed": False,
        "intelligibility_qualified": False,
        "note": "Private synthetic fixtures; no default-speaker, semantic-alignment or hearing claim."}
    (output / "speech.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--script", type=Path,
        help="Optional JSON array of objects containing id and text; no expected answers.")
    args = parser.parse_args()
    utterances = DEFAULT_UTTERANCES
    if args.script is not None:
        if args.script.stat().st_size > 32000:
            raise ValueError("speech_script_too_large")
        rows = json.loads(args.script.read_text(encoding="utf-8"))
        utterances = tuple((row["id"], row["text"]) for row in rows)
    manifest = prepare_speech_bundle(args.output, utterances)
    print(json.dumps({"clips": len(manifest["clips"]), "total_duration_ms": manifest["total_duration_ms"],
                      "provider_calls": 0, "frontend_replay_executed": False}))


if __name__ == "__main__":
    main()
