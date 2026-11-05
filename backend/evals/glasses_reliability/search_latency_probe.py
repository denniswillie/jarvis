"""Paid native camera-to-alert comparison, with real durable search completion.

Load a predeclared private manifest. Exact source images stay outside Git; their
original capture times remain provenance, while the controlled replay is retimed.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from hashlib import sha256
import json
import logging
import math
import os
from pathlib import Path
import statistics
import time

from app.settings import JarvisSettings
from app.visual_memory.validation import jpeg_dimensions
from evals.gemini_live.interruption import DatabaseFixtures
from evals.gemini_live.transport_probe import write_wav
from evals.glasses_reliability.gpt_turn_probe import Phone, TraceCapture
from evals.glasses_reliability.quality_probe import answer_observation_deadline
from evals.glasses_reliability.search_runtime import native_search_runtime, read_durable_state, submit_frame
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


def load_sequence(path):
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != 1 or manifest.get("stratum") not in {
        "retained_glasses_photo_controlled_replay", "synthetic_photograph_controlled_replay",
    }:
        raise ValueError("Declare a supported image provenance; captions are not object fixtures")
    target = manifest.get("target", {})
    if not all(isinstance(target.get(key), str) and target[key].strip() for key in ("name", "description", "purpose")):
        raise ValueError("An explicit target and intended behavior are required")
    images = {}
    for role in ("negative", "positive"):
        entry = manifest["images"][role]
        image_path = Path(entry["path"])
        image = (image_path if image_path.is_absolute() else path.parent / image_path).read_bytes()
        if sha256(image).hexdigest() != entry["sha256"]:
            raise ValueError("Image does not match its predeclared hash")
        width, height = jpeg_dimensions(image)
        if max(width, height) > 1024 or len(image) > 512000:
            raise ValueError("Replay image exceeds the existing capture contract")
        images[role] = image
    if images["negative"] == images["positive"]:
        raise ValueError("Positive and negative must not be the same pixels")
    if not isinstance(manifest.get("ground_truth"), str) or not manifest["ground_truth"].strip():
        raise ValueError("Declare independently inspected target visibility before inference")
    if manifest.get("original_audio_model_input") != "unproven":
        raise ValueError("This harness cannot prove historical audio-model image selection")
    return manifest, images


def schedule():
    # Same cadence and visibility transition for each provider, before inference.
    return [(float(offset), "negative" if offset < 4 else "positive") for offset in range(13)]


def runtime_pending(runtime):
    diagnostics = getattr(runtime.session, "diagnostics", None)
    proactive = diagnostics().get("proactive", {}) if callable(diagnostics) else {}
    return bool(runtime.service._pending_deliveries or runtime.native_turns.pending
                or proactive.get("pending") or proactive.get("owned_tasks"))


def observation_timings(captured, analyses, persisted):
    """Each observation is timed against its own primary capture, not upload."""
    frames = {frame["id"]: frame for frame in captured}
    rows = []
    for phase, records, id_key, time_key in (
        ("native_valid_output", analyses, "primary_frame_id", "valid_output_at"),
        ("persisted_observation", persisted, "primary_frame_id", "persisted_at"),
    ):
        for record in records:
            frame = frames.get(record.get(id_key))
            if frame is None or time_key not in record:
                continue
            elapsed = record[time_key] - frame["at"]
            rows.append({"boundary": phase, "frame_id": frame["id"],
                         "eligible_target": frame["eligible_target"],
                         "capture_to_boundary_seconds": elapsed if elapsed >= 0 else None})
    return rows


def valid_target_evidence(delivery, captured):
    evidence = delivery.get("evidence")
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except (TypeError, ValueError):
            return False
    if not isinstance(evidence, dict):
        return False
    references = evidence.get("supporting_frame_ids")
    if not isinstance(references, list) or not references or not all(isinstance(item, str) for item in references):
        return False
    known = {frame["id"] for frame in captured}
    positive = {frame["id"] for frame in captured if frame["eligible_target"]}
    return set(references) <= known and bool(set(references) & positive)


def assess(snapshot, *, owner, session_id, focus_id, captured, timeline):
    deliveries = snapshot["deliveries"]
    matched = [row for row in deliveries if str(row["visual_focus_id"]) == str(focus_id)
               and row["event_type"] == "delivered_to_session"]
    response_ids = {row["provider_response_id"] for row in matched if row["provider_response_id"]}
    audio = [event for event in timeline if event["type"] == "audio.output.delta"]
    eligible = [frame["at"] for frame in captured if frame["eligible_target"]]
    first_eligible = min(eligible) if eligible else None
    target_audio = [event["at"] for event in audio if event.get("response_id") in response_ids]
    first_forwarded = min(target_audio) if target_audio else None
    focus = snapshot["focus"]
    checks = {
        "one_tracked_delivery": len(deliveries) == len(matched) == 1,
        "delivery_has_target_evidence": bool(matched) and all(valid_target_evidence(row, captured) for row in matched),
        "completed_not_cancelled": bool(focus and focus["completed_at"] is not None and focus["cleared_at"] is None),
        "exact_focus": bool(focus and str(focus["id"]) == str(focus_id) and focus["mode"] == "find_once"),
        "one_intended_audio_response": bool(response_ids) and {event.get("response_id") for event in audio} == response_ids,
        "no_audio_before_target": first_eligible is not None and all(event["at"] >= first_eligible for event in audio),
        "frames_and_deliveries_scoped": all(str(row["owner_id"]) == owner and str(row["session_id"]) == str(session_id)
                                             for row in snapshot["frames"] + deliveries),
        "all_captured_frames_persisted": bool(captured) and len(snapshot["frames"]) == len(captured)
            and len({row["id"] for row in captured}) == len(captured)
            and {str(row["id"]) for row in snapshot["frames"]} == {row["id"] for row in captured},
        "real_observations_and_working_state": bool(snapshot["observations"])
            and any(isinstance(row.get("revision"), int) and row["revision"] > 0 for row in snapshot["working_memory"]),
        "no_audio_tool_mutations": not any(row["mutates_state"] for row in snapshot["executions"]),
    }
    latency = (first_forwarded - first_eligible) if first_forwarded is not None and first_eligible is not None else None
    return {"checks": checks, "infrastructure_passed": all(checks.values()),
            "capture_to_first_forwarded_seconds": latency if latency is not None and latency >= 0 else None,
            "first_eligible_at": first_eligible, "first_target_forwarded_at": first_forwarded}


def aggregate(rows, *, expected_pairs):
    if type(expected_pairs) is not int or not 1 <= expected_pairs <= 10:
        raise ValueError("Expected pairs must be an integer from one to ten")
    identities = set()
    for row in rows:
        if row.get("provider") not in {"openai", "gemini"} or type(row.get("repeat")) is not int or not 1 <= row["repeat"] <= expected_pairs:
            raise ValueError("Unexpected provider or repeat identity")
        identity = (row["provider"], row["repeat"])
        if identity in identities:
            raise ValueError("Duplicate provider/repeat cannot represent an independent trial")
        identities.add(identity)
    report = {}
    for provider in ("openai", "gemini"):
        selected = [row for row in rows if row["provider"] == provider]
        measured_rows = [row for row in selected
                         if type(row.get("capture_to_first_forwarded_seconds")) in (int, float)
                         and math.isfinite(row["capture_to_first_forwarded_seconds"])
                         and row["capture_to_first_forwarded_seconds"] >= 0]
        measured = [row["capture_to_first_forwarded_seconds"] for row in measured_rows]
        ordered = sorted(measured)
        median = statistics.median(ordered) if ordered else None
        complete = len(selected) == expected_pairs and len(ordered) == expected_pairs
        infrastructure_passed = complete and all(row["infrastructure_passed"] for row in selected)
        report[provider] = {"expected_runs": expected_pairs, "recorded_runs": len(selected),
            "measured_runs": len(ordered), "missing_measurements": len(selected) - len(ordered),
            "median_seconds": median,
            "p95_nearest_rank_seconds": ordered[math.ceil(.95 * len(ordered)) - 1] if ordered else None,
            "over_ten_seconds": [{"repeat": row["repeat"], "seconds": row["capture_to_first_forwarded_seconds"]}
                                 for row in measured_rows if row["capture_to_first_forwarded_seconds"] > 10],
            "infrastructure_passed": infrastructure_passed,
            "five_second_median_gate": infrastructure_passed and median is not None and median <= 5,
            "ten_pair_acceptance_eligible": expected_pairs >= 10 and complete,
        }
    return report


async def run_case(provider, repeat, *, fixtures, settings, manifest, images, output):
    label = f"search-{repeat}-{provider}"
    row = {"provider": provider, "repeat": repeat, "infrastructure_passed": False,
           "capture_to_first_forwarded_seconds": None, "device_playback": "not_tested",
           "stratum": manifest["stratum"], "remote_storage_objects_created": 0}
    owner = None
    runtime = trace = None
    phone = Phone()
    try:
        owner, session_id = await fixtures.create(label)
        async with asyncio.timeout(210), native_search_runtime(fixtures=fixtures, owner=owner, session_id=session_id,
                provider=provider, settings=settings, phone=phone, target=manifest["target"]) as runtime:
            trace = TraceCapture(runtime.session.context.audio_session_id)
            logging.getLogger("app.realtime.telemetry").addHandler(trace)
            started = time.monotonic()
            row["replay_started_at"] = datetime.now(UTC).isoformat()
            for offset, role in schedule():
                await asyncio.sleep(max(0, started + offset - time.monotonic()))
                await submit_frame(runtime, images[role], eligible_target=role == "positive", phase="primary", scheduled_offset=offset)
            await runtime.pipeline.drain()

            async def settled():
                wait_started = time.monotonic()
                await asyncio.sleep(0)
                while runtime_pending(runtime):
                    if runtime.running.done() or phone.closed:
                        raise RuntimeError("Native session ended with a pending alert")
                    first = next((event["at"] for event in phone.timeline if event["type"] == "audio.output.delta"), None)
                    deadline = answer_observation_deadline(wait_started, first_audio_at=first,
                                                          pcm_bytes=sum(len(pcm) for pcm in phone.audio.values()))
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Pending native alert exceeded observation budget")
                    await asyncio.sleep(.025)

            await settled()
            first = await read_durable_state(fixtures, owner=owner, session_id=session_id, focus_id=runtime.focus.id)
            row["before_late_view"] = first
            if first["focus"] and first["focus"]["completed_at"] is not None:
                # Cross the production ten-second duplicate-suppression window.
                await asyncio.sleep(settings.VISUAL_INTERRUPT_DEDUPE_SECONDS + 2)
                late_start = time.monotonic()
                for offset in (0, 1, 2):
                    await asyncio.sleep(max(0, late_start + offset - time.monotonic()))
                    await submit_frame(runtime, images["positive"], eligible_target=True, phase="late_repeat", scheduled_offset=offset)
                await runtime.pipeline.drain()
                await settled()
                row["late_repeat_exercised"] = True
            else:
                row["late_repeat_exercised"] = False
            row["native_turn_journal"] = runtime.native_turns.diagnostics()
            row["runtime_settled"] = not runtime_pending(runtime)
        # The services have stopped producing records before independent readback.
        snapshot = await read_durable_state(fixtures, owner=owner, session_id=session_id, focus_id=runtime.focus.id)
        row["durable_state"] = snapshot
        row.update(assess(snapshot, owner=owner, session_id=session_id, focus_id=runtime.focus.id,
                          captured=runtime.captured, timeline=phone.timeline))
        row["checks"]["late_repeat_exercised"] = row["late_repeat_exercised"]
        row["checks"]["no_phone_error"] = not any(event["type"] == "error" for event in phone.events)
        row["checks"]["trace_complete"] = trace is not None and not trace.overflowed
        row["checks"]["all_native_alerts_settled"] = row["runtime_settled"] and row["native_turn_journal"]["balanced"]
        row["checks"]["reference_audit_complete"] = all("reference_audit_error" not in call for call in runtime.analyzer.calls)
        row["infrastructure_passed"] = all(row["checks"].values())
    except Exception as error:
        row["error_type"] = type(error).__name__
        row["infrastructure_passed"] = False
    finally:
        if trace is not None:
            logging.getLogger("app.realtime.telemetry").removeHandler(trace)
            row["traces"] = trace.records
        if runtime is not None:
            row.update(captured_frames=runtime.captured, analyses=runtime.analyzer.calls,
                       persisted_observations=runtime.perception.observations, frame_diagnostics=runtime.diagnostics,
                       written_images=runtime.written_images, native_events=runtime.native_events,
                       native_usage=runtime.native_usage)
            row["observation_timings"] = observation_timings(runtime.captured, runtime.analyzer.calls,
                                                             runtime.perception.observations)
        row["phone_timeline"] = phone.timeline
        row["phone_events"] = [e for e in phone.events if e["type"] != "audio.output.delta"]
        row["audio_files"] = []
        for index, (item, pcm) in enumerate(phone.audio.items(), 1):
            name = f"{label}-{index}.wav"
            write_wav(output / name, bytes(pcm))
            row["audio_files"].append({"file": name, "item_id": item, "pcm_sha256": sha256(pcm).hexdigest()})
        if owner is not None:
            row["cleanup"] = await cleanup_fixture_owners(fixtures.dsn, {owner})
            row["infrastructure_passed"] = row["infrastructure_passed"] and row["cleanup"]["remaining_rows"] == 0
        (output / (label + ".json")).write_text(json.dumps(row, indent=2, default=str) + "\n")
    print(json.dumps({key: row.get(key) for key in ("provider", "repeat", "infrastructure_passed",
                                                  "capture_to_first_forwarded_seconds", "checks", "error_type")}), flush=True)
    return row


async def run(args):
    manifest, images = load_sequence(args.manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    settings = JarvisSettings(_env_file=None)
    (args.output / "run-manifest.json").write_text(json.dumps({"source": manifest, "schedule": schedule(),
        "repeats": args.repeats, "source_revision": args.source_revision,
        "late_repeat_delay_after_completion_seconds": settings.VISUAL_INTERRUPT_DEDUPE_SECONDS + 2,
        "models": {"openai": settings.OPENAI_REALTIME_MODEL, "gemini": settings.GEMINI_LIVE_MODEL,
                   "visual": settings.GEMINI_VISUAL_MODEL},
        "scope": "Real inference, validation, DB, working memory, search routing and audio adapters; controlled capture, no object upload or device playback."}, indent=2) + "\n")
    fixtures = DatabaseFixtures(os.environ["JARVIS_TEST_DB_URL"], args.output)
    results = []
    try:
        await fixtures.connect()
        for repeat in range(1, args.repeats + 1):
            for provider in (("openai", "gemini") if repeat % 2 else ("gemini", "openai")):
                results.append(await run_case(provider, repeat, fixtures=fixtures, settings=settings,
                                             manifest=manifest, images=images, output=args.output))
                (args.output / "matrix.json").write_text(json.dumps(aggregate(results, expected_pairs=args.repeats), indent=2) + "\n")
    finally:
        await fixtures.cleanup()
        (args.output / "matrix.json").write_text(json.dumps({"providers": aggregate(results, expected_pairs=args.repeats),
            "cleanup": fixtures.cleanup_result}, indent=2) + "\n")
    return 0 if len(results) == args.repeats * 2 and all(row["infrastructure_passed"] for row in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("Use 1-10 paired runs; fewer than ten is only a pilot")
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("app.realtime.telemetry").setLevel(logging.INFO)
    logging.getLogger("app.realtime.telemetry").propagate = False
    raise SystemExit(asyncio.run(run(args)))
