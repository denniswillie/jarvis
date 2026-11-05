"""Retained-photo uncertainty check through the existing real audio evaluator.

The source photograph and matching speech stay private. This does not claim to
reproduce historical audio-model image selection or physical glasses playback.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
from uuid import UUID

from app.settings import JarvisSettings
from app.visual_memory.validation import jpeg_dimensions
from evals.gemini_live.interruption import DatabaseFixtures
from evals.gemini_live.transport_probe import read_pcm
from evals.glasses_reliability.quality_cases import QualityCase
from evals.glasses_reliability.quality_probe import run_case, summarize


CASE = QualityCase(
    name="retained_branded_possible_match", code="",
    prompts=("In this view, do you see anything that could be my Taylor Swift Eras Tour merch box? Please flag possible matches even if you are not certain.",),
    expected=("Identify the Eras Tour-branded item as a possible match, while preserving uncertainty about its three-dimensional form.",
              "A useful next view may show the item's side or thickness; a flat-looking view alone cannot exclude a box."),
    forbidden=("Confidently reject a plausible branded match because it looks like a poster.",
               "Invent certainty about the item's shape, or promise an unregistered continuing search."),
    family="uncertainty", stratum="retained_glasses_photo_controlled_replay",
)


def case_definition():
    return json.loads(json.dumps(asdict(CASE)))


def load_photo_assets(media):
    manifest = json.loads((media / "manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("case") != case_definition():
        raise ValueError("Retained-photo case definition changed after predeclaration")
    if manifest.get("original_audio_model_input") != "unproven":
        raise ValueError("Historical audio-model image selection is not established")
    if not isinstance(manifest.get("ground_truth"), str) or not manifest["ground_truth"].strip():
        raise ValueError("Independently inspected visual ground truth is required")
    provenance = manifest["provenance"]
    UUID(provenance["frame_id"])
    UUID(provenance["session_id"])
    captured = datetime.fromisoformat(provenance["captured_at"])
    if captured.tzinfo is None:
        raise ValueError("Original capture time must include a timezone")
    image = (media / manifest["image"]["file"]).read_bytes()
    speech_path = media / manifest["speech"]["file"]
    speech_bytes = speech_path.read_bytes()
    if sha256(image).hexdigest() != manifest["image"]["sha256"] or sha256(speech_bytes).hexdigest() != manifest["speech"]["wav_sha256"]:
        raise ValueError("Private media changed after predeclaration")
    size = jpeg_dimensions(image)
    if max(size) > 1024 or len(image) > 512000:
        raise ValueError("Retained photo exceeds the existing camera input contract")
    pcm = read_pcm(speech_path)
    if not pcm:
        raise ValueError("A matched spoken input is required")
    assets = {CASE.name: {"jpeg": image, "size": size, "speech": [pcm], "bridge": b""}}
    return manifest, assets, {"image_sha256": sha256(image).hexdigest(), "input_pcm_sha256": sha256(pcm).hexdigest()}


async def run(args):
    manifest, assets, hashes = load_photo_assets(args.media)
    args.output.mkdir(parents=True, exist_ok=False)
    settings = JarvisSettings(_env_file=None)
    order = [(repeat, provider) for repeat in range(1, args.repeats + 1)
             for provider in (("openai", "gemini") if repeat % 2 else ("gemini", "openai"))]
    (args.output / "run-manifest.json").write_text(json.dumps({
        "source": manifest, "source_revision": args.source_revision, "input_hashes": hashes,
        "created_at": datetime.now(UTC).isoformat(), "order": order,
        "models": {"openai": settings.OPENAI_REALTIME_MODEL, "gemini": settings.GEMINI_LIVE_MODEL},
        "schedule": "Identical retained pixels every 500 ms; identical 24 kHz PCM paced at 100 ms chunks.",
        "scope": "Real sessions, image tools, native connections and isolated DB owners; no background analyzer, object upload, continuing search registration or device playback.",
        "ground_truth_scope": "Branded possible-match uncertainty only; no exact transcription or OCR score for unreadable code in the same photo.",
    }, indent=2) + "\n")
    fixtures = DatabaseFixtures(os.environ["JARVIS_TEST_DB_URL"], args.output)
    results = []
    try:
        await fixtures.connect()
        for repeat, provider in order:
            results.append(await run_case(CASE, repeat, provider, fixtures=fixtures, settings=settings,
                                          assets=assets, output=args.output))
            (args.output / "matrix.json").write_text(json.dumps(summarize(results, expected_runs=len(order)), indent=2) + "\n")
    finally:
        await fixtures.cleanup()
        (args.output / "matrix.json").write_text(json.dumps({**summarize(results, expected_runs=len(order)),
                                                           "cleanup": fixtures.cleanup_result}, indent=2) + "\n")
    return 0 if summarize(results, expected_runs=len(order))["infrastructure_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3:
        parser.error("Use one to three repeats; fewer than three is only a pilot")
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("app.realtime.telemetry").setLevel(logging.INFO)
    logging.getLogger("app.realtime.telemetry").propagate = False
    raise SystemExit(asyncio.run(run(args)))
