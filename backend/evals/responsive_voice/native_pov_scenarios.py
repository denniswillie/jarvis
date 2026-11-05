"""Offline frozen POV wrappers for the existing zero-cut prepared baseline transport.

No execution, credentials, release decisions, provider calls, or context seeding.
Only selected immutable media and v1 launch fields enter runtime/. Evaluator joins
and source annotations are siblings, never runtime inputs. Original permissions
are provenance, not reinterpreted or modified by this compiler.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_CEILING
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import wave

from evals.responsive_voice import native_interruption_scenarios as base


VARIANTS = {"c01-a": "u01", "c01-b": "u02"}
EXPANDED_VARIANTS = {
    "c02-a": dict(case_id="c02", offset=35, duration=95, workload="none", prelude=("u03",),
                  events=((40000, "u04"),), frame_count=16, media_end_ms=32000),
    "c02-b": dict(case_id="c02", offset=35, duration=95, workload="none", prelude=("u03",),
                  events=((40000, "u04"),), frame_count=16, media_end_ms=32000, withhold_from_ms=18000),
    "c05-a": dict(case_id="c05", offset=35, duration=165, post_media_capture_seconds=0,
                  workload="reasoning-reminder", prelude=("u10",),
                  events=((0, "u09"), (8000, "u11"), (96000, "u12"), (110000, "u13")),
                  frame_count=15, media_end_ms=29984),
    "c05-b": dict(case_id="c05", offset=5, duration=50, post_media_capture_seconds=0,
                  workload="reasoning-reminder", prelude=(),
                  events=((0, "u09"), (8000, "u11")), frame_count=15, media_end_ms=29984),
    "c01-c": dict(case_id="c01", offset=5, duration=45, workload="reasoning-reminder", prelude=(),
                  events=(), frame_count=16, media_end_ms=32000),
}
QUIET_SCRIPT = "I need to focus now. Please do not disturb me until I tell you."
RESUME_SCRIPT = "You can interrupt me again. Tell me only what is still relevant."
TIMER_SCRIPT = "In ten seconds, remind me to check whether this cleaning is finished."
M = 5
QUESTION_T = 23
RUN_SECONDS = 45
SPEECH_FIELDS = (
    "id", "relative_path", "sha256", "sample_rate_hz", "channels",
    "sample_width_bytes", "sample_count", "first_active_sample",
    "last_active_sample_exclusive", "energy_threshold_abs_pcm16",
)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _pairs(items):
    result = {}
    for key, value in items:
        base.need(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def _load(root, relative, receipts):
    path = base.local_path(root, relative)
    raw = base.bounded_read(path, 512 * 1024)
    receipts.append({"path": str(path), "sha256": base.digest(raw), "bytes": len(raw)})
    return json.loads(raw, object_pairs_hook=_pairs)


def _one(items, key, value):
    matches = [item for item in items if item.get(key) == value]
    base.need(len(matches) == 1, "missing_or_duplicate_selected_record")
    return matches[0]


def _source(pack, variants):
    receipts = []
    assets = _load(pack, "inputs/c01/assets.json", receipts)
    timelines = _load(pack, "inputs/timelines.json", receipts)
    speech = _load(pack, "inputs/speech/speech.json", receipts)
    oracle = _load(pack, "evaluator/expectations.json", receipts)
    timeline = _one(timelines["cases"], "case_id", "c01")
    expectation = _one(oracle["cases"], "case_id", "c01")
    base.need(assets.get("format") == "jarvis.video_assets.v1"
              and assets.get("source_audio_included") is False, "unsupported_visual_source")
    base.need(speech.get("format") == "jarvis.speech_assets.v1", "unsupported_speech_bundle")
    inputs = assets["inputs"]
    base.need(inputs.get("initial_context") == [] and inputs.get("events") == [],
              "c01_requires_unseeded_visual_input")
    base.need(timeline["assets"] == "c01/assets.json" and timeline["media_end_ms"] == 32000
              and timeline["scenario_end_ms"] == 40000, "unsupported_c01_timeline")
    frames, pts, indices = inputs["frames"], assets["source_pts_seconds"], assets["source_frame_indices"]
    base.need(len(frames) == len(pts) == len(indices) == 16, "c01_requires_exact_frame_selection")
    first_pts = Decimal(str(assets["source_first_pts_seconds"]))
    base.need(first_pts.is_finite(), "invalid_source_pts")
    image_bytes, frame_joins, runtime_frames = {}, [], []
    for index, (frame, source_pts, source_index) in enumerate(zip(frames, pts, indices)):
        point = Decimal(str(source_pts))
        base.need(point.is_finite() and point >= first_pts, "invalid_source_pts")
        source_ms = int(((point - first_pts) * 1000).to_integral_value(rounding=ROUND_CEILING))
        base.need(type(frame["offset_ms"]) is int and frame["offset_ms"] == source_ms == index * 2000,
                  "source_frame_timing_changed_or_future")
        base.need(type(source_index) is int and source_index >= 0
                  and (index == 0 or source_index > indices[index - 1]), "invalid_source_frame_index")
        path = base.local_path(pack / "inputs/c01", frame["relative_path"])
        raw = base.bounded_read(path, 4 * 1024 * 1024)
        base.need(base.digest(raw) == frame["sha256"], "frame_hash_mismatch")
        base.need(raw.startswith(b"\xff\xd8") and raw.endswith(b"\xff\xd9"), "invalid_jpeg_signature")
        relative = f"frames/{index + 1:06d}.jpg"
        at = M + source_ms / 1000
        image_bytes[relative] = raw
        runtime_frames.append({"at_seconds": at, "relative_path": relative, "sha256": frame["sha256"]})
        frame_joins.append({
            "native_frame_ordinal": index, "runtime_relative_path": relative,
            "source_path": str(path), "sha256": frame["sha256"], "bytes": len(raw),
            "source_frame_index": source_index, "source_pts_seconds": source_pts,
            "source_offset_ms": source_ms, "native_not_before_seconds": at,
            "staged_relative_path": f"frames/frame-{index}.jpg",
        })
    base.need(sum(map(len, image_bytes.values())) <= 64 * 1024 * 1024, "frame_budget_exceeded")
    selected = {}
    for variant in variants:
        clip_id = VARIANTS[variant]
        timeline_variant = _one(timeline["variants"], "variant_id", variant)
        base.need(timeline_variant["prior_context_utterance_ids"] == []
                  and timeline_variant["events"] == [{"occurrence_id": variant + "-01",
                       "at_ms": 18000, "utterance_id": clip_id}], "unsupported_c01_variant_timeline")
        meta = _one(speech["clips"], "id", clip_id)
        base.need(meta.get("original_environment_audio") is False, "directed_synthetic_speech_required")
        path = base.local_path(pack / "inputs/speech", meta["relative_path"])
        raw = base.bounded_read(path, 60 * base.RATE * 2 + 65536)
        base.need(base.digest(raw) == meta["sha256"], "speech_hash_mismatch")
        base.need(meta["sample_rate_hz"] == base.RATE and meta["channels"] == 1
                  and meta["sample_width_bytes"] == 2, "unsupported_speech_format")
        with wave.open(io.BytesIO(raw), "rb") as wav:
            base.need((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype())
                      == (base.RATE, 1, 2, "NONE") and wav.getnframes() == meta["sample_count"],
                      "wav_format_or_count_mismatch")
            pcm = wav.readframes(wav.getnframes())
        base.need(len(pcm) == 2 * meta["sample_count"] and 0 < meta["sample_count"]
                  and QUESTION_T + meta["sample_count"] / base.RATE + 5 <= RUN_SECONDS,
                  "speech_cannot_fit_deadline")
        selected[variant] = {
            "meta": {key: meta[key] for key in SPEECH_FIELDS}, "wav": raw,
            "source_path": str(path), "pcm_sha256": base.digest(pcm),
            "oracle": _one(expectation["variants"], "id", variant),
        }
    return assets, expectation, receipts, image_bytes, runtime_frames, frame_joins, selected


def _stage(runtime, expected_hash, *, workload):
    """Only the existing pure staging function, never run_once/main/host startup."""
    from evals.responsive_voice.native_replay_run import stage_prepared_media

    with tempfile.TemporaryDirectory(prefix="jarvis-pov-offline-stage-") as temporary:
        options = SimpleNamespace(
            input_manifest=runtime / "manifest.json", media_root=runtime,
            speech_manifest=runtime / "speech/speech.json", artifact_dir=Path(temporary),
            task_aware=False, voice_route="realtime", cuts=0,
            input_mode="prepared-scenario", scenario="baseline", workload=workload,
            result_release_policy="all_sdk_inputs",
        )
        plan, records = stage_prepared_media(options, expected_hash)
        return plan, len(records)


class _Writer:
    def __init__(self, root):
        self.root, self.records = root, []
        root.mkdir(mode=0o700, parents=False, exist_ok=False)

    def put(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("xb") as handle:
            path.chmod(0o600)
            handle.write(data)
        self.records.append({"relative_path": relative, "bytes": len(data), "sha256": base.digest(data)})

    def json(self, relative, value):
        self.put(relative, encoded(value))


def _prepare_c01(pack_root, output, *, variants=tuple(VARIANTS), validate_staging=False):
    """Create fresh roots; only --validate-staging imports the offline stager."""
    pack, output = Path(pack_root).resolve(), Path(output).absolute()
    variants = tuple(variants)
    base.need(bool(variants) and len(set(variants)) == len(variants)
              and all(variant in VARIANTS for variant in variants), "unsupported_or_duplicate_variant")
    source = _source(pack, variants)
    assets, expectation, receipts, images, frames, frame_joins, selected = source
    writer = _Writer(output)
    cases = []
    for variant in variants:
        item = selected[variant]
        meta = dict(item["meta"], relative_path=VARIANTS[variant] + ".wav")
        speech = {"format": "jarvis.speech_assets.v1", "clips": [meta]}
        manifest = {
            "version": 1, "run_id": "0" * 32, "backend_base_url": "http://127.0.0.1:1",
            "owner_id": "00000000-0000-0000-0000-000000000000",
            "area_id": "00000000-0000-0000-0000-000000000000",
            "memory_bank_id": "compiler-placeholder-not-a-live-bank", "voice_route": "realtime",
            "bearer_token_file": "token.txt", "duration_seconds": RUN_SECONDS,
            "speech_manifest": "speech/speech.json", "frames": frames,
            "utterances": [{"clip_id": VARIANTS[variant],
                            "trigger": {"kind": "at_time", "at_seconds": QUESTION_T}}],
        }
        manifest_raw = encoded(manifest)
        manifest_hash = base.digest(manifest_raw)
        prefix = variant + "/runtime/"
        for relative, raw in images.items():
            writer.put(prefix + relative, raw)
        writer.put(prefix + "speech/" + meta["relative_path"], item["wav"])
        writer.json(prefix + "speech/speech.json", speech)
        writer.put(prefix + "manifest.json", manifest_raw)
        joins = {
            "format": "jarvis.native_pov_evaluator_join.v1", "not_model_input": True,
            "case_id": "c01", "variant_id": variant, "source": expectation["source"],
            "source_sha256": assets["source_sha256"], "source_records": receipts,
            "compiler_input_manifest_sha256": manifest_hash,
            "clock": {"M_seconds": M, "question_seconds": QUESTION_T, "run_seconds": RUN_SECONDS,
                      "source_window_end_native_seconds": 37,
                      "relation": "native_release = M + ceil((source_pts - source_first_pts) * 1000) / 1000",
                      "source_first_pts_seconds": assets["source_first_pts_seconds"],
                      "actual_media_epoch_id": None, "actual_run_id": None},
            "frames": frame_joins,
            "utterance": {"sidecar_occurrence_id": variant + "-01", "native_input_ordinal": 0,
                          "runtime_occurrence_pattern": "<actual_run_id>/utterance/0",
                          "clip_id": meta["id"], "source_at_ms": 18000,
                          "native_trigger": manifest["utterances"][0]["trigger"],
                          "source_wav_path": item["source_path"], "wav_sha256": meta["sha256"],
                          "pcm_sha256": item["pcm_sha256"], "sample_count": meta["sample_count"],
                          "first_active_sample": meta["first_active_sample"],
                          "last_active_sample_exclusive": meta["last_active_sample_exclusive"],
                          "sample_rate_hz": base.RATE, "synthetic_directed_speech": True,
                          "environment_audio_included": False},
            "context_payload": [], "prelude": [],
            "runtime_evidence_required": [
                "Bind actual run/resource manifest, scope and media_epoch_id; fresh scope for each variant.",
                "Join original SHA to staging SHA and actual frame admission timing; re-encoded outgoing JPEG need not have original SHA.",
                "Join ordinal/clip/PCM hash to callback offsets, SDK acceptance and received transcription, not scripted transcript injection.",
                "Use decoded PCM/sample map, control events and canonical rows for actual response evidence; mark useful-answer onset manually.",
                "No frame may precede its mapped release; answers may use later frames only if actually admitted before the answer.",
            ],
            "limits": [
                "T=23 frame and question can race; source PTS 18 is not guaranteed admitted at the question onset.",
                "No camera-stop trigger at T=37; the last released source frame may remain selected through T=45, not fresh visual evidence.",
                "Staging success is not native delivery, inference, semantic correctness or audio acceptance.",
            ],
        }
        writer.json(variant + "/evaluator/joins.json", joins)
        writer.json(variant + "/evaluator/expectations.json", {
            "not_model_input": True, "case_id": "c01", "variant_id": variant,
            "source_evidence": expectation["evidence"], "original_variant": item["oracle"],
            "native_useful_answer_window_seconds": [23, 37] if variant == "c01-a" else None,
            "native_question_seconds": 23, "native_source_window_end_seconds": 37,
            "note": "Original oracle unchanged. Native times are M=5 translations; c01-b has no separately authored answer window.",
        })
        writer.json(variant + "/RELEASE_TEMPLATE.json", {
            "status": "TEMPLATE_NOT_RELEASED_NOT_CONSUMABLE", "schema_version": 1,
            "authorization_id": None, "allow_one_launch": False, "exclusive_simulator_app": False,
            "expires_at": None, "simulator_id": None, "bundle_id": None,
            "scenario": "baseline", "input_mode": "prepared-scenario", "cuts": 0,
            "input_manifest_sha256": manifest_hash, "workload": "none",
            "voice_route": "realtime", "task_aware": False, "public_demo": False,
            "historical_attempts": [],
            "semantic_variant_not_runner_scenario": variant,
            "candidate_limits_not_independently_enforced_by_this_template": {
                "attempts_for_this_variant": 1, "concurrency_across_cases": 1,
                "media_seconds": 45, "baseline_host_lease_seconds": 285,
                "automatic_retries": 0,
            },
            "payload_scope": {
                "frames": [{"sha256": frame["sha256"], "at_seconds": frame["at_seconds"]} for frame in frames],
                "wav_sha256": meta["sha256"], "pcm_sha256": item["pcm_sha256"],
                "sample_count": meta["sample_count"], "context": [],
                "providers_for_main_to_bind": ["Google Gemini API visual path", "OpenAI Realtime audio path"],
                "frontier_workload": "none",
            },
            "main_must_author": [
                "A separate actual one-attempt authorization with current expiry, simulator/app and historical-attempt decision required by validate_risk.",
                "Current release/provider/model/account bindings; do not treat this template or compiler placeholder IDs as a release.",
                "Fresh isolated host scope and token through existing runner; compiler input bytes stay unchanged.",
            ],
        })
        status, staged_count = "not_requested", 0
        if validate_staging:
            plan, staged_count = _stage(output / variant / "runtime", manifest_hash,
                                        workload="none")
            base.need(plan["utterances"] == manifest["utterances"] and len(plan["frames"]) == 16
                      and [(f["at_seconds"], f["sha256"]) for f in plan["frames"]]
                      == [(f["at_seconds"], f["sha256"]) for f in frames]
                      and staged_count == 17, "offline_staging_mapping_mismatch")
            status = "passed"
        cases.append({"variant_id": variant, "runtime_root": str(output / variant / "runtime"),
                      "input_manifest_sha256": manifest_hash, "selected_wav": meta["id"],
                      "frame_count": 16, "jpeg_bytes": sum(map(len, images.values())),
                      "wav_bytes": len(item["wav"]), "wav_seconds": meta["sample_count"] / base.RATE,
                      "offline_prepared_staging": status, "temporary_staged_media_files": staged_count})
    report = {
        "format": "jarvis.native_pov_preparation.v1", "cases": cases,
        "native_runs": 0, "provider_calls": 0, "sql_calls": 0, "releases_authored": 0,
        "new_distinct_source_media": 0, "download_bytes": 0,
        "copied_jpeg_files": 16 * len(cases), "copied_wav_files": len(cases),
        "copied_media_bytes": sum(c["jpeg_bytes"] + c["wav_bytes"] for c in cases),
        "distinct_source_window_seconds": 32, "per_case_run_seconds": 45,
        "directed_speech_seconds": sum(c["wav_seconds"] for c in cases),
        "temporary_staging_copies_retained": False,
    }
    writer.json("OFFLINE_VALIDATION.json", report)
    writer.put("README.md", _readme(pack, output, cases).encode())
    writer.put("DIARY_ENTRY.md", (
        "# Private diary: c01 native preparation\n\n"
        "Prepared only c01-a/current-view and c01-b/occluded-count uncertainty using unchanged approved source bytes. "
        "M=5, T=23 question, 45-second run, prepared-scenario baseline, cuts=0, workload=none, non-task-aware.\n\n"
        + "Offline existing prepared-staging results: "
        + ", ".join(c["variant_id"] + "=" + c["offline_prepared_staging"] for c in cases)
        + ". Temporary staging copies removed by TemporaryDirectory. No native/provider/SQL execution or release.\n\n"
        + f"Retained wrapper copies: {report['copied_jpeg_files']} JPEGs + {report['copied_wav_files']} WAVs, "
        + f"{report['copied_media_bytes']} bytes; one distinct 32-second source window; "
        + f"{report['directed_speech_seconds']:.9f} seconds selected synthetic speech. No downloads/new footage.\n\n"
        + "Only the new compiler/test files and this fresh private output are owned changes. FILES.json lists all private files. "
        "Focused unit-test execution is reported separately by the caller, not inferred from staging. "
        "Main authors actual release/current bindings and consolidates this entry. Old grants, review receipts and plans unchanged.\n\n"
        + "Remaining broader gaps: audible prelude/order barriers, zero-speech quiet cases, historical/multisession memory transitions, "
        "long quiet windows beyond the 120-second native cap, and proactive/responsibility task support. "
        "This baseline transport does not solve or test those gaps.\n"
    ).encode())
    writer.json("FILES.json", {"files": list(writer.records), "inventory_self": "FILES.json",
                              "note": "Self listed by path only to avoid a circular hash; no existing file changed."})
    return dict(report, output=str(output), private_files_created=len(writer.records),
                private_bytes_created=sum(row["bytes"] for row in writer.records))


def _readme(pack, output, cases):
    roots = "\n".join(f"- `{c['variant_id']}`: `{c['runtime_root']}`; manifest SHA256 `{c['input_manifest_sha256']}`."
                      for c in cases)
    return f"""# Prepared c01 native wrappers (not released or executed)

{roots}

Each runtime root contains only manifest.json, speech/speech.json, one selected WAV and 16 unchanged JPEGs.
Question T=23; frame releases T=5,7,...35; run45. Source interval remains 2 seconds, source PTS 0..30.
The selected source window is 32 seconds; two variants are not two distinct source clips.
WAVs are synthetic directed user speech, not original environment audio. No context/prelude is injected.
All answer requirements, source PTS, source records and evaluator joins are outside runtime/.
Compiler binding fields are placeholders only; token.txt is absent. No actual authorization is present.

## Reproduce offline (from /Users/denniswillie/jarvis/backend)

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/denniswillie/jarvis/.jarvis-dev/p0-venv/bin/python -B -m unittest discover -s tests -p test_native_pov_scenarios.py -v
PYTHONDONTWRITEBYTECODE=1 /Users/denniswillie/jarvis/.jarvis-dev/p0-venv/bin/python -B -m evals.responsive_voice.native_pov_scenarios \\
  --pack-root {pack} --output /ABSOLUTE/FRESH/OUTPUT --validate-staging
```

The latter calls only existing stage_prepared_media in a disposable directory. It does not call run_once,
validate_risk, host preparation, simulator tools, SQL or providers. OFFLINE_VALIDATION.json records actual staging only.

## Main's later native invocation inputs

Existing runner module: evals.responsive_voice.native_replay_run. Use these exact transport flags per case:
`--input-mode prepared-scenario --scenario baseline --cuts 0 --workload none --voice-route realtime`.
Supply `--input-manifest <runtime>/manifest.json --media-root <runtime> --speech-manifest <runtime>/speech/speech.json`.
Do not use --task-aware, --result-hold-seconds or reasoning-reminder. The 45-second duration lives in the manifest.
No baseline/current_view clip-name assumption is used: this is prepared-scenario transport, not the older hardcoded baseline input mode.

Main supplies a NEW actual --risk-authorization, current simulator/app, fresh artifact/ledger paths,
canonical-export command and current host/Hindsight configuration through the existing runner. Do not pass RELEASE_TEMPLATE.json
as authorization: allow_one_launch=false and missing IDs/expiry/history intentionally prevent consumption.
Main must satisfy the runner's current historical-attempt gate, not invent a decision here. Existing user authorization and
completed privacy review are not reopened. Existing immutable 15-frame grant and DRAFT_GRANT.json remain untouched.
The new release binds this compiler manifest hash, selected raw PCM/WAV/image hashes and current provider settings.
One attempt per case, sequential; no automatic retries or public demo. Proposed host lease285 is run45+baseline overhead240,
not a claim that this template independently enforces a total process wall-clock limit.

## Evaluation joins and limitations

Each case's evaluator/joins.json maps source path/PTS/index/hash to compiler and staged frame path/time, plus
WAV/container/raw PCM hashes, native input ordinal0 and actual-run occurrence pattern. Main attaches actual resource manifest,
scope, media epoch, capture/canonical receipts and provider route/model evidence after execution. Start with fresh independent
scope for each variant; no prelude, remembered facts, seeded answer or cross-case conversation reuse.

c01-a: describe observed pepper cutting, not all preparation complete. Native useful-answer window23..37; later responses must
be last-seen/uncertain and marked late. c01-b: do not invent an exact count of fully hidden pieces; visible pieces do not establish
that count. Its original oracle has no independently authored useful-answer deadline; retain that distinction.
T=23 frame/question race needs actual admission evidence. At T=37 there is no camera-stop event; source PTS30 may remain selected
until T=45. Repeated capture is not new source evidence. Native re-encoding changes outgoing JPEG hashes; do not equate these with
original source hashes. Callback acceptance is not ASR correctness, rendered/audio capture is not semantic acceptance, and decoded
output has no automatic useful-answer-onset label. Do not claim tests of those properties from successful staging.

Larger unsupported work remains explicit: prelude delivery/settlement, zero-input silence/quiet, chronological historical placement
and multisession memory transitions, >120-second quiet windows, proactive obligations and changed-goal task barriers.

## Definitive existing sources (unchanged)

- `{pack}/inputs/c01/assets.json`
- `{pack}/inputs/timelines.json`
- `{pack}/inputs/speech/speech.json`, selected u01.wav/u02.wav only
- `{pack}/evaluator/expectations.json`
- `{pack}/review-20261031-02/NATIVE_MAPPING.md`
- `/Users/denniswillie/jarvis/backend/evals/responsive_voice/native_replay_run.py`: stage_prepared_media, validate_risk, run_once
- `/Users/denniswillie/jarvis/backend/evals/responsive_voice/native_interruption_scenarios.py`: load_bundles

FILES.json inventories exactly the fresh private files; old media/permissions/review receipts and shared code/plans are not modified.
"""


def _expanded_clip(pack, raw):
    """Select one existing WAV; authored text stays outside runtime metadata."""
    base.need(raw.get("original_environment_audio") is False, "directed_synthetic_speech_required")
    path = base.local_path(pack / "inputs/speech", raw["relative_path"])
    wav_bytes = base.bounded_read(path, 60 * base.RATE * 2 + 65536)
    base.need(base.digest(wav_bytes) == raw["sha256"], "speech_hash_mismatch")
    base.need(raw["sample_rate_hz"] == base.RATE and raw["channels"] == 1
              and raw["sample_width_bytes"] == 2, "unsupported_speech_format")
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        base.need((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype())
                  == (base.RATE, 1, 2, "NONE") and wav.getnframes() == raw["sample_count"],
                  "wav_format_or_count_mismatch")
        pcm = wav.readframes(wav.getnframes())
    base.need(0 < raw["sample_count"] <= 60 * base.RATE and len(pcm) == raw["sample_count"] * 2,
              "invalid_speech_sample_count")
    return dict(meta={key: raw[key] for key in SPEECH_FIELDS}, wav=wav_bytes,
                pcm_sha256=base.digest(pcm), source_path=str(path), text=raw.get("text"))


def _expanded_inputs(pack, variants):
    receipts, assets, images, clips = [], {}, {}, {}
    timelines = _load(pack, "inputs/timelines.json", receipts)
    speech = _load(pack, "inputs/speech/speech.json", receipts)
    oracle = _load(pack, "evaluator/expectations.json", receipts)
    base.need(speech.get("format") == "jarvis.speech_assets.v1", "unsupported_speech_bundle")
    bundles = []
    for variant in variants:
        spec = EXPANDED_VARIANTS.get(variant)
        if spec is None:
            spec = dict(case_id="c01", offset=5, duration=45, workload="none", prelude=(),
                        events=((18000, VARIANTS[variant]),), frame_count=16, media_end_ms=32000)
        case_id = spec["case_id"]
        timeline = _one(timelines["cases"], "case_id", case_id)
        selected = _one(timeline["variants"], "variant_id", variant)
        expected_events = [{"occurrence_id": f"{variant}-{index + 1:02d}", "at_ms": at, "utterance_id": clip}
                           for index, (at, clip) in enumerate(spec["events"])]
        base.need(selected["prior_context_utterance_ids"] == list(spec["prelude"])
                  and selected["events"] == expected_events and not selected.get("control_events"),
                  "unsupported_frozen_variant_timeline")
        admission = ({"only_release_offsets_less_than_ms": spec["withhold_from_ms"]}
                     if "withhold_from_ms" in spec else None)
        base.need(selected.get("frame_admission") == admission, "unsupported_frame_admission")
        end_ms = selected.get("scenario_end_ms", timeline["scenario_end_ms"])
        base.need(timeline["assets"] == case_id + "/assets.json"
                  and timeline["media_end_ms"] == spec["media_end_ms"]
                  and spec["offset"] + end_ms / 1000 == spec["duration"]
                  and 0 < spec["duration"] <= 180, "unsupported_frozen_source_clock")
        if case_id not in assets:
            assets[case_id] = _load(pack, f"inputs/{case_id}/assets.json", receipts)
        source = assets[case_id]
        base.need(source.get("format") == "jarvis.video_assets.v1" and source.get("source_audio_included") is False,
                  "unsupported_visual_source")
        source_inputs = source["inputs"]
        base.need(source_inputs.get("initial_context") == [] and source_inputs.get("events") == [],
                  "unseeded_visual_input_required")
        frames, points, indices = source_inputs["frames"], source["source_pts_seconds"], source["source_frame_indices"]
        base.need(len(frames) == len(points) == len(indices) == spec["frame_count"], "frozen_frame_selection_changed")
        origin = Decimal(str(source["source_first_pts_seconds"]))
        base.need(origin.is_finite(), "invalid_source_pts")
        selected_images, frame_plan, frame_joins = {}, [], []
        previous_ms, previous_index = -1, -1
        for original_ordinal, (frame, point, source_index) in enumerate(zip(frames, points, indices)):
            point_decimal = Decimal(str(point))
            base.need(point_decimal.is_finite() and point_decimal >= origin, "invalid_source_pts")
            release_ms = int(((point_decimal - origin) * 1000).to_integral_value(rounding=ROUND_CEILING))
            base.need(type(frame["offset_ms"]) is int and frame["offset_ms"] == release_ms
                      and previous_ms < release_ms < spec["media_end_ms"], "source_frame_timing_changed_or_future")
            base.need(type(source_index) is int and source_index > previous_index, "invalid_source_frame_index")
            previous_ms, previous_index = release_ms, source_index
            if "withhold_from_ms" in spec and release_ms >= spec["withhold_from_ms"]:
                continue
            source_path = base.local_path(pack / "inputs" / case_id, frame["relative_path"])
            key = str(source_path)
            if key not in images:
                images[key] = base.bounded_read(source_path, 4 * 1024 * 1024)
            raw = images[key]
            base.need(base.digest(raw) == frame["sha256"], "frame_hash_mismatch")
            base.need(raw.startswith(b"\xff\xd8") and raw.endswith(b"\xff\xd9"), "invalid_jpeg_signature")
            relative = f"frames/{original_ordinal + 1:06d}.jpg"
            native_ms = spec["offset"] * 1000 + release_ms
            frame_plan.append(dict(at_seconds=native_ms / 1000, relative_path=relative, sha256=frame["sha256"]))
            frame_joins.append(dict(native_frame_ordinal=len(frame_plan) - 1, source_frame_ordinal=original_ordinal,
                source_frame_index=source_index, source_pts_seconds=point, source_offset_ms=release_ms,
                native_not_before_ms=native_ms, source_path=str(source_path), sha256=frame["sha256"],
                bytes=len(raw), runtime_relative_path=relative, staged_relative_path=f"frames/frame-{len(frame_plan)-1}.jpg"))
            selected_images[relative] = raw
        base.need(0 < len(frame_plan) <= 240 and sum(map(len, selected_images.values())) <= 64 * 1024 * 1024,
                  "invalid_frame_budget")
        occurrences = [dict(role="prelude", clip_id=clip, at_seconds=5,
                            source_occurrence_id=f"{variant}/prelude/{clip}", source_at_ms=None)
                       for clip in spec["prelude"]]
        occurrences += [dict(role="scripted_user", clip_id=event["utterance_id"],
                             at_seconds=spec["offset"] + event["at_ms"] // 1000,
                             source_occurrence_id=event["occurrence_id"], source_at_ms=event["at_ms"])
                        for event in selected["events"]]
        base.need(all(event["at_ms"] % 1000 == 0 for event in selected["events"]), "unsupported_subsecond_utterance")
        selected_clips = {}
        for occurrence in occurrences:
            clip_id = occurrence["clip_id"]
            if clip_id not in clips:
                clips[clip_id] = _expanded_clip(pack, _one(speech["clips"], "id", clip_id))
            selected_clips[clip_id] = clips[clip_id]
        base.need(sum(clips[o["clip_id"]]["meta"]["sample_count"] for o in occurrences) <= 60 * base.RATE,
                  "input_budget_exceeded")
        case_oracle = _one(oracle["cases"], "case_id", case_id)
        bundles.append(dict(variant=variant, spec=spec, source=source, timeline=selected,
                            case_oracle=case_oracle, oracle=_one(case_oracle["variants"], "id", variant),
                            images=selected_images, frames=frame_plan, frame_joins=frame_joins,
                            clips=selected_clips, occurrences=occurrences))
    return bundles, receipts


def _qualification(bundle):
    variant, spec, clips = bundle["variant"], bundle["spec"], bundle["clips"]
    active = spec["workload"] == "reasoning-reminder"
    prelude = dict(mode="none", correlated_setup_proven=False, decoded_drain_proven=False)
    if spec["prelude"]:
        nominal_end = 5 + sum(clips[key]["meta"]["sample_count"] / base.RATE for key in spec["prelude"])
        base.need(nominal_end < spec["offset"], "prelude_pcm_exceeds_reservation")
        prelude.update(mode="fixed_reservation_diagnostic_only", input_start_seconds=5,
            nominal_pcm_end_seconds=nominal_end, reservation_end_seconds=spec["offset"],
            maximum_setup_seconds=30, nominal_padding_seconds=spec["offset"] - nominal_end,
            acceptance_requirement="Canonical setup acceptance and attributable setup-response decoded drain must precede media zero; elapsed padding is not proof.",
            failure_rule="Missing/late/unobservable setup means failed or unsupported setup. Never shift the frozen source clock or seed expected context.")
    quiet = dict(applicable=variant == "c05-a", script_check="not_applicable", actual_quiet_acceptance_proven=False)
    if variant == "c05-a":
        matches = clips["u10"]["text"] == QUIET_SCRIPT and clips["u13"]["text"] == RESUME_SCRIPT
        quiet.update(script_check="matches_until_explicit_resume" if matches else "mismatch_or_unrecognized",
            original_quiet_script=clips["u10"]["text"], original_resume_script=clips["u13"]["text"],
            instruction_has_fixed_expiry=False if matches else None,
            measured_native_window_seconds=[35, 145], measured_seconds=110, minimum_seconds=90,
            required_reconciliation_cycles=2, drain_guard_seconds=5,
            direct_question_windows_seconds=[[43, 60], [131, 143]],
            explanation=("The frozen script is quiet until explicit resume, not a 90-second timed mute. Earlier prelude acceptance does not itself expire it. Scope/revision and continued quiet still need actual evidence."
                         if matches else "The script does not match the frozen until-resume wording. Fixed padding does not establish its quiet duration; no semantic qualification is claimed."),
            mismatch_rule="Keep the original script, schedule and expectations unchanged; flag mismatch, do not certify the quiet interval or rewrite the oracle.")
    timer = dict(applicable=variant in {"c05-a", "c05-b"})
    if timer["applicable"]:
        timer.update(script_matches_frozen_ten_seconds=clips["u09"]["text"] == TIMER_SCRIPT,
            due_basis="actual accepted canonical scheduling time + 10 seconds", nominal_due_seconds=None,
            acceptance_proven=False, positive_control="c05-b", quiet_case="c05-a")
    return dict(not_model_input=True, execution="not_executed", prelude=prelude, quiet=quiet, timer=timer,
        required_actual_profile=dict(workload=spec["workload"], record_actual_profile=True,
            frontier_enabled=True if active else None, attention_voice_enabled=True if active else None,
            reminder_scheduler_enabled=True if active else None, worker_readiness_required=active,
            silence_is_sufficient=False, disabled_machinery_silence_can_pass=False,
            record_output_channel_eligibility=["speech", "UI", "push"],
            disabled_output_channel_verdict="unmeasured, not successfully suppressed",
            paired_quiet_controls_require_same_actual_profile=variant in {"c05-a", "c05-b"}),
        scope=dict(fresh_owner_area_bank_and_session=True, baseline_memory_reuse_allowed=False,
                   initial_context=[], multisession=False, no_duty_control=variant == "c01-c"),
        remaining_evidence=["Actual input SDK acceptance and correlated ASR", "Actual canonical frame admission and source-age evidence",
            "Correlated response/decoded PCM and useful-answer annotation where a reply is required",
            "Actual scheduler, attention and output eligibility receipts; disabled machinery cannot establish successful silence"],
        no_new_framework="All stimuli use existing at_time triggers; no setup-state, decoded-drain, camera-stop or multisession framework is implemented.")


def _translated_windows(oracle, offset):
    result = {}
    for key in ("question_ms", "response_deadline_ms"):
        if key in oracle:
            result[key.replace("_ms", "_seconds")] = offset + oracle[key] / 1000
    for key in ("camera_unavailable_ms", "quiet_window_ms", "silence_window_ms", "useful_answer_window_ms"):
        if key in oracle:
            result[key.replace("_ms", "_seconds")] = [offset + value / 1000 for value in oracle[key]]
    if "direct_question_windows_ms" in oracle:
        result["direct_question_windows_seconds"] = [[offset + value / 1000 for value in window]
                                                     for window in oracle["direct_question_windows_ms"]]
    return result


def _prepare_expanded(pack_root, output, variants, validate_staging):
    pack, output = Path(pack_root).resolve(), Path(output).absolute()
    bundles, receipts = _expanded_inputs(pack, variants)
    writer, cases = _Writer(output), []
    for bundle in bundles:
        variant, spec = bundle["variant"], bundle["spec"]
        occurrences, speech_clips = [], []
        for ordinal, item in enumerate(bundle["occurrences"]):
            clip = bundle["clips"][item["clip_id"]]
            occurrences.append(dict(item, native_input_ordinal=ordinal,
                runtime_occurrence_pattern=f"<actual_run_id>/utterance/{ordinal}",
                wav_sha256=clip["meta"]["sha256"], pcm_sha256=clip["pcm_sha256"],
                sample_count=clip["meta"]["sample_count"], source_wav_path=clip["source_path"]))
        for identity, clip in bundle["clips"].items():
            speech_clips.append(dict(clip["meta"], relative_path=identity + ".wav"))
        manifest = dict(version=1, run_id="0" * 32, backend_base_url="http://127.0.0.1:1",
            owner_id="00000000-0000-0000-0000-000000000000", area_id="00000000-0000-0000-0000-000000000000",
            memory_bank_id="compiler-placeholder-not-a-live-bank", voice_route="realtime", bearer_token_file="token.txt",
            duration_seconds=spec["duration"], speech_manifest="speech/speech.json", frames=bundle["frames"],
            utterances=[dict(clip_id=item["clip_id"], trigger=dict(kind="at_time", at_seconds=item["at_seconds"]))
                        for item in occurrences])
        if "post_media_capture_seconds" in spec:
            manifest["post_media_capture_seconds"] = spec["post_media_capture_seconds"]
        manifest_raw = encoded(manifest)
        manifest_hash = base.digest(manifest_raw)
        prefix = variant + "/runtime/"
        for relative, raw in bundle["images"].items():
            writer.put(prefix + relative, raw)
        for identity, clip in bundle["clips"].items():
            writer.put(prefix + "speech/" + identity + ".wav", clip["wav"])
        writer.json(prefix + "speech/speech.json", dict(format="jarvis.speech_assets.v1", clips=speech_clips))
        writer.put(prefix + "manifest.json", manifest_raw)
        qualification = _qualification(bundle)
        writer.json(variant + "/evaluator/qualification.json", qualification)
        writer.json(variant + "/evaluator/expectations.json", dict(not_model_input=True,
            case_id=spec["case_id"], variant_id=variant, original_variant=bundle["oracle"],
            source_case={key: value for key, value in bundle["case_oracle"].items() if key != "variants"},
            native_windows_seconds=_translated_windows(bundle["oracle"], spec["offset"])))
        writer.json(variant + "/evaluator/joins.json", dict(not_model_input=True,
            format="jarvis.native_pov_evaluator_join.v2", variant_id=variant, case_id=spec["case_id"],
            compiler_input_manifest_sha256=manifest_hash, source_sha256=bundle["source"]["source_sha256"],
            source_records=receipts, source_first_pts_seconds=bundle["source"]["source_first_pts_seconds"],
            M_seconds=spec["offset"], source_media_end_ms=spec["media_end_ms"],
            source_window_end_native_seconds=spec["offset"] + spec["media_end_ms"] / 1000,
            duration_seconds=spec["duration"], actual_media_epoch_id=None, actual_run_id=None,
            frames=bundle["frame_joins"], occurrences=occurrences, original_timeline_variant=bundle["timeline"],
            context_payload=[], original_environment_audio=False,
            withheld_at_or_after_source_ms=spec.get("withhold_from_ms"),
            join_limits=["Frame/utterance scheduling is not proof of native admission, ASR, correlated setup or decoded drain.",
                "Source offsets are preserved exactly, rounded upward only as in the frozen asset manifest; source time never pauses for responses.",
                "No new camera sample is scheduled after the selected last frame. Cached or repeated last pixels must retain original source age.",
                "c02 is same-session historical recall, not fresh-session or durable-memory qualification; each condition needs a separate scope.",
                "Native re-encoding requires actual source-to-outgoing image hash lineage; evaluator labels are never runtime inputs."]))
        lease = spec["duration"] + 240 + (120 if spec["workload"] == "reasoning-reminder" else 0)
        writer.json(variant + "/RELEASE_TEMPLATE.json", dict(status="TEMPLATE_NOT_RELEASED_NOT_CONSUMABLE",
            schema_version=1, authorization_id=None, allow_one_launch=False, exclusive_simulator_app=False,
            expires_at=None, simulator_id=None, bundle_id=None, historical_attempts=[],
            scenario="baseline", input_mode="prepared-scenario", cuts=0, task_aware=False,
            voice_route="realtime", workload=spec["workload"], input_manifest_sha256=manifest_hash,
            frontier_model=None, frontier_reasoning_effort=None, public_demo=False,
            semantic_variant_not_runner_scenario=variant,
            candidate_limits_not_independently_enforced_by_this_template=dict(attempts_for_this_variant=1,
                concurrency_across_cases=1, automatic_retries=0, media_seconds=spec["duration"],
                host_lease_seconds=lease, capture_wait_seconds=spec["duration"] + 30),
            payload_scope=dict(frames=[dict(sha256=f["sha256"], at_seconds=f["at_seconds"]) for f in bundle["frames"]],
                speech=[dict(id=key, wav_sha256=clip["meta"]["sha256"], pcm_sha256=clip["pcm_sha256"],
                             sample_count=clip["meta"]["sample_count"]) for key, clip in bundle["clips"].items()],
                context=[], providers_for_main_to_bind=["Google Gemini API visual path", "OpenAI Realtime audio path"]
                    + (["Main-selected text frontier worker"] if spec["workload"] == "reasoning-reminder" else [])),
            required_actual_profile=qualification["required_actual_profile"],
            evaluator_qualification_file="evaluator/qualification.json",
            main_must_author=["Fresh actual authorization, expiry/current simulator/app and historical-attempt decision.",
                "Actual provider/model/account and nonpersonal native prompt/tool-schema bindings; no placeholder is a release.",
                "Actual scheduler/attention/worker profile and output-route eligibility; disabled-machinery silence cannot pass.",
                "Fixed prelude reservation is diagnostic only until correlated canonical setup and decoded drain are proven."]))
        staging, staged_count = "not_requested", 0
        if validate_staging:
            plan, staged_count = _stage(output / variant / "runtime", manifest_hash,
                                        workload=spec["workload"])
            base.need(plan["utterances"] == manifest["utterances"]
                and plan.get("post_media_capture_seconds") == manifest.get("post_media_capture_seconds")
                and [(f["at_seconds"], f["sha256"]) for f in plan["frames"]]
                    == [(f["at_seconds"], f["sha256"]) for f in bundle["frames"]]
                and staged_count == len(bundle["frames"]) + len(bundle["clips"]), "offline_staging_mapping_mismatch")
            staging = "passed"
        cases.append(dict(variant_id=variant, case_id=spec["case_id"], runtime_root=str(output / variant / "runtime"),
            input_manifest=str(output / variant / "runtime/manifest.json"), input_manifest_sha256=manifest_hash,
            speech_manifest=str(output / variant / "runtime/speech/speech.json"),
            release_template=str(output / variant / "RELEASE_TEMPLATE.json"),
            evaluator_qualification=str(output / variant / "evaluator/qualification.json"),
            frame_count=len(bundle["frames"]), selected_wavs=list(bundle["clips"]),
            jpeg_bytes=sum(map(len, bundle["images"].values())), wav_bytes=sum(len(c["wav"]) for c in bundle["clips"].values()),
            speech_seconds=sum(c["meta"]["sample_count"] / base.RATE for c in bundle["clips"].values()),
            M_seconds=spec["offset"], duration_seconds=spec["duration"], workload=spec["workload"], lease_seconds=lease,
            offline_prepared_staging=staging, quiet_script_check=qualification["quiet"]["script_check"]))
    report = dict(format="jarvis.native_pov_expanded_preparation.v1", cases=cases,
        native_runs=0, provider_calls=0, sql_calls=0, releases_authored=0, download_bytes=0,
        new_distinct_source_media=0, copied_jpeg_files=sum(c["frame_count"] for c in cases),
        copied_wav_files=sum(len(c["selected_wavs"]) for c in cases),
        copied_media_bytes=sum(c["jpeg_bytes"] + c["wav_bytes"] for c in cases),
        directed_speech_seconds=sum(c["speech_seconds"] for c in cases),
        temporary_staging_copies_retained=False, source_data_modified=False)
    writer.json("OFFLINE_VALIDATION.json", report)
    writer.put("README.md", ("# Expanded fixed-clock POV diagnostics, not released\n\n"
        + "\n".join(f"- {c['variant_id']}: {c['input_manifest']}; SHA256 {c['input_manifest_sha256']}; "
                    f"speech {c['speech_manifest']}; template {c['release_template']}" for c in cases)
        + "\n\nAll use existing prepared-scenario baseline, cuts0, realtime, task_aware=false. "
        "Workload and lease differ per case; see OFFLINE_VALIDATION.json and each non-consumable RELEASE_TEMPLATE.json. "
        "Main authors actual bindings/releases. No provider/native/SQL execution occurred.\n\n"
        "M35 cases play the existing prelude WAV at T5; the fixed30-second reservation is only diagnostic. "
        "Correlated canonical setup acceptance and decoded acknowledgement drain before T35 remain required. "
        "No multisession, setup-state, playback-drain or camera-stop framework is added.\n\n"
        "c05-a uses until-explicit-resume wording, not a timed quiet duration. Measured quiet is T35..145, "
        "direct questions T43 and T131, resume T145, end165. The timer's ten seconds begin at actual accepted scheduling, "
        "not nominal T35. If script semantics mismatch, retain the originals and mark qualification unsupported. "
        "c05-b is the no-quiet positive control (T5 timer, T13 question, end50).\n\n"
        "c01-c has no utterances/WAVs. Candidate reasoning-reminder profiles for all three silence/timer controls "
        "require actual enabled scheduler/attention and worker readiness. Disabled machinery or output channels "
        "cannot establish successful suppression. Match c05-a/b profiles; record two reconciliation opportunities "
        "and the five-second guard. Do not replace frozen all-channel expectations with silence-only success.\n\n"
        "Runtime directories contain only neutral v1 manifests, selected WAV metadata and unchanged image/WAV bytes. "
        "Scripts, source PTS, setup conditions and oracle text remain evaluator-only. Existing c01-a/b default compilation "
        "is retained by the legacy path. FILES.json is the creation inventory for this prepared subtree.\n").encode())
    writer.json("FILES.json", dict(files=list(writer.records), inventory_self="FILES.json"))
    return dict(report, output=str(output), private_files_created=len(writer.records),
                private_bytes_created=sum(row["bytes"] for row in writer.records))


def prepare(pack_root, output, *, variants=tuple(VARIANTS), validate_staging=False):
    variants = tuple(variants)
    allowed = set(VARIANTS) | set(EXPANDED_VARIANTS)
    base.need(bool(variants) and len(set(variants)) == len(variants)
              and all(variant in allowed for variant in variants), "unsupported_or_duplicate_variant")
    if all(variant in VARIANTS for variant in variants):
        return _prepare_c01(pack_root, output, variants=variants, validate_staging=validate_staging)
    return _prepare_expanded(pack_root, output, variants, validate_staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="fresh directory; parent must exist")
    parser.add_argument("--variant", action="append", choices=tuple(VARIANTS) + tuple(EXPANDED_VARIANTS))
    parser.add_argument("--validate-staging", action="store_true", help="existing pure prepared stager only")
    args = parser.parse_args(argv)
    result = prepare(args.pack_root, args.output, variants=args.variant or tuple(VARIANTS),
                     validate_staging=args.validate_staging)
    print(json.dumps(result, sort_keys=True))


NATIVE_IMAGE_TRANSFORM_SNAPSHOT = {
    "basis": "reviewed_source_snapshot_not_build_or_execution_evidence",
    "sources": [
        {"path": "ios/Jarvis/Jarvis/Services/JarvisReplayCaptureSource.swift",
         "sha256": "e14d4034385a92ed7443dca09fdd1d50c48399824d57c900eb57496813d1aa40",
         "lines": [18, 78, 84]},
        {"path": "ios/Jarvis/Jarvis/Services/GlassesFrameCaptureService.swift",
         "sha256": "210e974a21b5445ceab67cc42dba311fbd9f2ede0cd514561e46eaf0a72071d9",
         "lines": [110, 138, 166]},
        {"path": "ios/Jarvis/Jarvis/Services/JarvisVisualIngestClient.swift",
         "sha256": "af54a9f6bc7654f809ec618241f0fbb3543ec18cf2b33daa3c8be7965ac533ae",
         "lines": [111, 125]},
    ],
    "decoder": "UIImage(data:)",
    "adapter": "GlassesFrameAdapter default initializer",
    "max_dimension": 768,
    "size_rule": "min(1,768/max(UIImage.size)); positive dimensions rounded to nearest, ties away from zero; minimum 1",
    "renderer": "UIGraphicsImageRenderer; image.draw(in: output rectangle)",
    "renderer_format": "UIGraphicsImageRendererFormat.default() with scale=1; other defaults not pinned",
    "jpeg_compression_quality": 0.7,
    "output": "CapturedVisualFrame.jpegData; sent as image_base64",
    "executed_here": False,
    "native_build_binding": None,
    "native_input_output_hash_join_proven": False,
    "limitation": "No UIKit encoding is reproduced. Source configuration, dimensions or temporal order cannot prove native JPEG bytes, coalescing, delivery or provider use.",
}


def write_frame_lineage_receipt(*, pack_root, variant, runtime, output,
                                expected_manifest_sha256):
    """Write an external, exclusive receipt using the real offline staging path.

    This is deliberately opt-in for already prepared inputs: neither prepare's
    outputs nor any scenario/authorization bytes change. Native re-encoding is
    not emulated or claimed. No evaluator text or annotations enter staging.
    """
    import copy
    import hashlib
    import json
    import math
    from decimal import Decimal, ROUND_CEILING
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace

    from . import native_interruption_scenarios as staging_assets
    from . import native_replay_run as runner

    def need(condition, reason):
        if not condition:
            raise ValueError(reason)

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    starts = {"c01-a": 5, "c01-b": 5, "c01-c": 5, "c02-a": 35,
              "c02-b": 35, "c05-a": 35, "c05-b": 5}
    need(variant in starts, "unsupported_lineage_variant")
    pack_root, runtime, output = Path(pack_root).resolve(), Path(runtime).resolve(), Path(output).resolve()
    need(not output.is_relative_to(runtime) and not output.is_relative_to(pack_root / "inputs"),
         "lineage_must_stay_outside_model_inputs")
    need(not output.exists(), "lineage_output_exists")
    asset_root = pack_root / "inputs" / variant.split("-")[0]
    asset_path = asset_root / "assets.json"
    asset_raw = staging_assets.bounded_read(asset_path, 512 * 1024)
    assets = json.loads(asset_raw)
    source_frames = assets["inputs"]["frames"]
    indices, pts = assets["source_frame_indices"], assets["source_pts_seconds"]
    need(len(source_frames) == len(indices) == len(pts), "lineage_source_count_mismatch")
    selected = [(i, item) for i, item in enumerate(source_frames)
                if variant != "c02-b" or item["offset_ms"] < 18000]
    manifest_path = runtime / "manifest.json"
    manifest_raw = staging_assets.bounded_read(manifest_path, 512 * 1024)
    need(digest(manifest_raw) == expected_manifest_sha256, "lineage_manifest_hash_mismatch")
    manifest = json.loads(manifest_raw)
    need(len(manifest["frames"]) == len(selected), "lineage_selected_count_mismatch")
    original_rows = []
    for ordinal, ((source_ordinal, source), prepared) in enumerate(zip(selected, manifest["frames"])):
        offset = source["offset_ms"]
        source_offset = ((Decimal(str(pts[source_ordinal])) - Decimal(str(assets["source_first_pts_seconds"]))) * 1000)
        need(type(offset) is int and offset == int(source_offset.to_integral_value(rounding=ROUND_CEILING)),
             "lineage_source_clock_mismatch")
        need(Decimal(str(prepared["at_seconds"])) * 1000 == starts[variant] * 1000 + offset,
             "lineage_prepared_clock_mismatch")
        source_path = staging_assets.local_path(asset_root, source["relative_path"])
        original = staging_assets.bounded_read(source_path, 4 * 1024 * 1024)
        source_hash = digest(original)
        need(source_hash == source["sha256"] == prepared["sha256"], "lineage_source_hash_mismatch")
        prepared_path = staging_assets.local_path(runtime, prepared["relative_path"])
        prepared_data = staging_assets.bounded_read(prepared_path, 4 * 1024 * 1024)
        need(prepared_data == original, "lineage_prepared_bytes_mismatch")
        width, height = source["width"], source["height"]
        need(type(width) is int and type(height) is int and width > 0 and height > 0,
             "lineage_invalid_declared_dimensions")
        scale = min(1, 768 / max(width, height))
        original_rows.append({
            "prepared_ordinal": ordinal,
            "source": {"path": str(source_path), "sha256": source_hash, "bytes": len(original),
                       "asset_frame_ordinal": source_ordinal, "source_frame_index": indices[source_ordinal],
                       "source_pts_seconds": pts[source_ordinal], "release_offset_ms": offset,
                       "declared_dimensions": {"width": width, "height": height},
                       "dimension_basis": "retained_asset_manifest_not_UIImage_decode"},
            "prepared": {"relative_path": prepared["relative_path"], "sha256": digest(prepared_data),
                         "bytes": len(prepared_data), "at_seconds": prepared["at_seconds"]},
            "native": {"status": "unobserved_not_joined", "output_sha256": None,
                       "frame_sequence": None, "canonical_frame_id": None,
                       "conditional_output_dimensions": {
                           "width": max(1, math.floor(width * scale + 0.5)),
                           "height": max(1, math.floor(height * scale + 0.5))},
                       "dimension_condition": "Only if UIImage.size matches recorded dimensions and the reviewed default adapter is used; not an observed native result"},
        })
    # Invoke the existing prepared-media boundary, not a second image pipeline.
    with TemporaryDirectory(prefix="jarvis-pov-lineage-") as temporary:
        options = SimpleNamespace(input_manifest=manifest_path, media_root=runtime,
            speech_manifest=staging_assets.local_path(runtime, manifest["speech_manifest"]),
            artifact_dir=Path(temporary), voice_route="realtime", cuts=0, task_aware=False,
            input_mode="prepared-scenario", scenario="baseline",
            workload=EXPANDED_VARIANTS[variant]["workload"] if variant in EXPANDED_VARIANTS else "none",
            result_release_policy="all_sdk_inputs")
        plan, _ = runner.stage_prepared_media(options, expected_manifest_sha256)
        need(len(plan["frames"]) == len(original_rows), "lineage_staged_count_mismatch")
        for row, staged in zip(original_rows, plan["frames"]):
            data = staging_assets.bounded_read(staging_assets.local_path(Path(temporary) / "inputs",
                                                                        staged["relative_path"]), 4 * 1024 * 1024)
            need(digest(data) == staged["sha256"] == row["source"]["sha256"]
                 and len(data) == row["source"]["bytes"], "lineage_staged_bytes_mismatch")
            need(staged["at_seconds"] == row["prepared"]["at_seconds"], "lineage_staged_clock_mismatch")
            row["staged"] = {"relative_path": staged["relative_path"], "sha256": digest(data),
                             "bytes": len(data), "at_seconds": staged["at_seconds"]}
    receipt = {
        "format": "jarvis.native_pov_frame_lineage.v1", "variant": variant,
        "status": "verified_original_jpeg_to_offline_staged_jpeg_only",
        "runtime_manifest": {"path": str(manifest_path), "sha256": digest(manifest_raw)},
        "source_asset_manifest": {"path": str(asset_path), "sha256": digest(asset_raw),
                                  "source_video_sha256_recorded": assets["source_sha256"],
                                  "source_video_decode_reperformed": False},
        "preparation": {"implementation": "evals.responsive_voice.native_replay_run.stage_prepared_media",
                        "transform": "byte_identity_copy", "resize_parameters": None,
                        "resize_applied": False, "jpeg_reencoded": False,
                        "temporary_stage_removed": True, "frame_count": len(original_rows),
                        "jpeg_bytes": sum(row["source"]["bytes"] for row in original_rows)},
        "native_transform": copy.deepcopy(NATIVE_IMAGE_TRANSFORM_SNAPSHOT),
        "frames": original_rows,
        "scope": "Offline source/staging receipt only; no native launch, delivery, inference, authorization release or semantic pass. Original video hash/PTS are retained extraction declarations, not a re-extraction proof.",
    }
    payload = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with output.open("xb") as target:
        output.chmod(0o600)
        target.write(payload)
    return receipt


if __name__ == "__main__":
    main()
