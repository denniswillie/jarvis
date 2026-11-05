"""Opt-in native comparison of legacy UUID and request-local visual references.

Only calls the configured Gemini visual model. No database/storage clients,
tools, notifications, or production owner are created. Images stay private.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time
from uuid import uuid4

from app.model_providers.gemini_fast_visual import GeminiFastVisualModel
from app.model_providers.protocols import VisualModelImage
from app.perception.contracts import FastPerceptionFrame, FastPerceptionOutputError, FastPerceptionRequest, parse_fast_perception_output
from app.perception.fast_analyzer import FastPerceptionAnalyzer
from app.perception.prompts import build_fast_perception_prompt, frame_attachment_label
from app.perception.references import PerceptionReferenceError, PerceptionReferences
from app.perception.result_validation import parse_referenced_perception_output
from app.settings import JarvisSettings
from app.visual_memory.gemini_client import GeminiVisualAnalyzerClient, gemini_visual_config_from_settings


class CaptureModel:
    """Captures one native result for offline validation, not another inference."""
    def __init__(self, model):
        self.delegate = model
        self.provider, self.model = model.provider, model.model
        self.raw = None
        self.failed = False

    async def generate_temporal_json(self, **kwargs):
        try:
            self.raw = await self.delegate.generate_temporal_json(**kwargs)
        except Exception:
            self.failed = True
            raise
        return self.raw


def inspect_result(raw, request, *, short_handles=True):
    try:
        result = parse_referenced_perception_output(raw, request=request,
                                                   references=PerceptionReferences(request, short_handles=short_handles))
        return {"rejected_candidates": [asdict(item) for item in result.rejected],
                "reference_rejections": sum(item.reason.startswith("unknown_") for item in result.rejected),
                "accepted_matches": len(result.output.prospective_matches), "core_usable": True}
    except PerceptionReferenceError as error:
        return {"rejected_candidates": [], "reference_rejections": 1, "core_usable": False,
                "core_rejection": str(error), "accepted_matches": 0}
    except FastPerceptionOutputError:
        return {"rejected_candidates": [], "reference_rejections": 0, "core_usable": False,
                "core_rejection": "invalid_core", "accepted_matches": 0}


async def trial(model, request, variant):
    capture = CaptureModel(model)
    started = time.monotonic()
    outcome = "accepted"
    try:
        if variant == "short_references":
            await FastPerceptionAnalyzer(capture).analyze(request)
        else:
            raw = await capture.generate_temporal_json(
                prompt=build_fast_perception_prompt(request),
                images=tuple(VisualModelImage(frame_attachment_label(index, frame.frame_id), frame.image_bytes, frame.content_type)
                             for index, frame in enumerate(request.frames)),
            )
            parse_fast_perception_output(raw, request=request)
    except FastPerceptionOutputError:
        outcome = "contract_rejected"
    except Exception:
        # Native transport/model failure is not a successful validation trial.
        outcome = "provider_failed" if capture.failed else "harness_failed"
    elapsed = (time.monotonic() - started) * 1000
    result = {"variant": variant, "outcome": outcome, "analysis_ms": round(elapsed, 2),
              "model_returned": capture.raw is not None}
    if capture.raw is not None:
        result.update(inspect_result(capture.raw, request, short_handles=variant == "short_references"))
        if outcome != "accepted":
            result["accepted_matches"] = 0
    return result


def aggregate(rows):
    result = {}
    for variant in ("legacy_uuid", "short_references"):
        selected = [row for row in rows if row["variant"] == variant]
        returned = [row for row in selected if row.get("model_returned") and row["outcome"] != "harness_failed"]
        times = sorted(row["analysis_ms"] for row in returned)
        result[variant] = {
            "attempts": len(selected), "outcomes": dict(Counter(row["outcome"] for row in selected)),
            "responses_with_reference_errors": sum(row.get("reference_rejections", 0) > 0 for row in returned),
            "reference_error_rate": sum(row.get("reference_rejections", 0) > 0 for row in returned) / len(returned) if returned else None,
            "rejected_optional_candidates": sum(len(row.get("rejected_candidates", [])) for row in returned),
            "median_analysis_ms": round(statistics.median(times), 2) if times else None,
            "maximum_analysis_ms": max(times) if times else None,
        }
    return result


async def run(args):
    config = gemini_visual_config_from_settings(JarvisSettings(_env_file=None))
    if config is None:
        raise SystemExit("The Gemini visual model is not configured.")
    manifest = json.loads(args.manifest.read_text())
    images = []
    for item in manifest["images"]:
        image = Path(item["path"]).read_bytes()
        images.append((item, image, hashlib.sha256(image).hexdigest()))
    run_id = uuid4()
    client = GeminiVisualAnalyzerClient(config)
    model = GeminiFastVisualModel(client)
    rows = []
    try:
        for repeat in range(args.runs):
            for item, image, digest in images:
                captured = datetime.fromisoformat(item["captured_at"])
                request = FastPerceptionRequest(
                    owner_id=f"reference-eval-{run_id}", session_id=uuid4(), window_id=uuid4(),
                    frames=(FastPerceptionFrame(uuid4(), item["sequence"], captured, "glasses", "image/jpeg", image),),
                    previous_working_state={}, existing_entity_ids=frozenset(), recent_audio_transcripts=(),
                    behavior_instructions="Report possible matches honestly. Do not invent objects or substitute unrelated alerts.",
                    active_focus={"id": str(uuid4()), "target_description": "Find a toilet roll in the room.", "mode": "find_once"},
                    active_reminders=({"id": str(uuid4()), "title": "Look for house keys", "condition": "House keys are clearly visible."},),
                    recent_interrupts=(), requested_at=captured,
                )
                variants = ("legacy_uuid", "short_references") if repeat % 2 == 0 else ("short_references", "legacy_uuid")
                for variant in variants:
                    result = await trial(model, request, variant)
                    rows.append({**result, "repeat": repeat, "sequence": item["sequence"], "sha256": digest})
                    document = {"model": model.model, "run_id": str(run_id), "rows": rows, "aggregate": aggregate(rows),
                                "evidence": manifest.get("evidence"),
                                "limitations": "Small non-seeded matched trial. Both variants share current scene guidance; legacy uses UUIDs and all-or-nothing parsing. No hardware latency or OCR-quality claim."}
                    args.output.write_text(json.dumps(document, indent=2) + "\n")
                    print(json.dumps({"trial": len(rows), "variant": variant, "sequence": item["sequence"],
                                      "outcome": result["outcome"], "analysis_ms": result["analysis_ms"]}), flush=True)
    finally:
        await client.close()
    print(json.dumps(aggregate(rows), indent=2))
    if any(row["outcome"] in {"provider_failed", "harness_failed"} for row in rows):
        raise SystemExit("Native comparison incomplete; inspect the recorded outcome categories.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.runs <= 10:
        parser.error("--runs must be between 1 and 10")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
