"""Offline native replay input compiler, not an executor or semantic judge.

Only immutable audio/image bytes and the existing v1 Swift launch fields enter
manifest.json/speech.json. Typed expectations remain in expectations.json. The
caller stages its isolated host token in token.txt separately. No token is read,
copied, generated or printed; nothing here launches an app or contacts a server.
"""
from __future__ import annotations

import argparse
from array import array
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
from uuid import UUID
import wave

from evals.responsive_voice.native_capture_evidence import analyze_capture


ROOT = Path(__file__).resolve().parents[3]
AUTHORITY = ROOT / "backend/evals/responsive_voice/replay_cases/E2E_EXPECTATIONS.md"
DEFAULT_BUNDLES = {
    "baseline": ROOT / ".jarvis-dev/native-replay-media-20261031.C36v0x/speech/speech.json",
    "conversation": ROOT / ".jarvis-dev/interruption-speech-20261031.Mx1TB1/conversation/speech.json",
    "state": ROOT / ".jarvis-dev/interruption-speech-20261031.Mx1TB1/state/speech.json",
    "setup": ROOT / ".jarvis-dev/interruption-setup-speech-20261031.sdzzLU/speech/speech.json",
}
RATE = 48000
TASK_CASES = frozenset({"pause_continue", "parallel_keep_working", "explicit_cancel", "ambiguous_stop",
                        "late_cancel", "quiet_direct_resume"})
SETUP_CASES = TASK_CASES | {"memory_correction_deletion"}
CASES = {
    "baseline": ("E2E-VIS-001",),
    "pause_continue": ("E2E-INT-BURST-001",),
    "parallel_keep_working": ("E2E-INT-BURST-002", "E2E-WORK-001"),
    "correction_chain": ("E2E-INT-BURST-003", "E2E-WORK-002"),
    "explicit_cancel": ("E2E-INT-BURST-004", "E2E-WORK-003"),
    "ambiguous_stop": ("E2E-INT-BURST-004",),
    "quiet_direct_resume": ("E2E-INT-BURST-007", "E2E-DND-001", "E2E-DND-002", "E2E-DND-003"),
    "memory_correction_deletion": ("E2E-INT-BURST-009", "E2E-MEM-002", "E2E-MEM-004"),
    "late_cancel": ("E2E-INT-BURST-005", "E2E-WORK-003"),
}


class ScenarioError(ValueError):
    """Fixed diagnostic code; never include source JSON or credentials."""


def need(condition, code):
    if not condition:
        raise ScenarioError(code)


def finite(value, code, minimum=0, maximum=120):
    need(type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum, code)
    return float(value)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bounded_read(path, limit):
    path = Path(path)
    need(path.stat().st_size <= limit, "asset_size_limit")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    need(len(data) <= limit, "asset_size_limit")
    return data


def local_path(root, value):
    need(isinstance(value, str) and value and not Path(value).is_absolute()
         and ":" not in value and "\\" not in value, "invalid_asset_path")
    root = Path(root).resolve()
    result = (root / value).resolve()
    need(result != root and result.is_relative_to(root), "asset_path_escape")
    return result


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    try:
        return json.loads(bounded_read(path, 512 * 1024), object_pairs_hook=pairs)
    except (ValueError, UnicodeError, RecursionError):
        raise ScenarioError("invalid_json") from None


@dataclass(frozen=True)
class Clip:
    bundle: str
    source_id: str
    wav: bytes
    sha256: str
    pcm_sha256: str
    sample_count: int
    first: int | None
    end: int | None
    threshold: int


def load_bundles(paths):
    """Validate real local WAVs; discard script text and unrelated metadata."""
    result = {}
    for bundle in ("baseline", "conversation", "state", *(("setup",) if paths.get("setup") is not None else ())):
        manifest = Path(paths[bundle])
        value = read_json(manifest)
        need(isinstance(value, dict) and value.get("format") == "jarvis.speech_assets.v1", "unsupported_speech_bundle")
        clips = value.get("clips")
        need(isinstance(clips, list) and len(clips) <= 100, "invalid_speech_clips")
        total = 0
        for raw in clips:
            need(isinstance(raw, dict), "invalid_speech_clip")
            identity = raw.get("id")
            need(isinstance(identity, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity), "invalid_clip_identity")
            key = bundle + "/" + identity
            need(key not in result, "duplicate_clip_identity")
            count, threshold = raw.get("sample_count"), raw.get("energy_threshold_abs_pcm16")
            need(type(count) is int and 0 < count <= 60 * RATE, "invalid_speech_sample_count")
            need(type(threshold) is int and 1 <= threshold <= 32767, "invalid_speech_threshold")
            need(raw.get("sample_rate_hz") == RATE and raw.get("channels") == 1
                 and raw.get("sample_width_bytes") == 2, "unsupported_speech_format")
            data = bounded_read(local_path(manifest.parent, raw.get("relative_path")), 60 * RATE * 2 + 65536)
            total += len(data)
            need(total <= 64 * 1024 * 1024, "speech_bundle_size_limit")
            need(digest(data) == raw.get("sha256"), "speech_hash_mismatch")
            try:
                with wave.open(io.BytesIO(data), "rb") as wav:
                    need(wav.getframerate() == RATE and wav.getnchannels() == 1 and wav.getsampwidth() == 2
                         and wav.getcomptype() == "NONE" and wav.getnframes() == count, "wav_format_or_count_mismatch")
                    pcm = wav.readframes(count)
            except (wave.Error, EOFError):
                raise ScenarioError("invalid_wav") from None
            need(len(pcm) == count * 2, "wav_truncated")
            samples = array("h", pcm)
            if sys.byteorder != "little":
                samples.byteswap()
            first = next((i for i, sample in enumerate(samples) if abs(sample) >= threshold), None)
            last = next((i for i in range(count - 1, -1, -1) if abs(samples[i]) >= threshold), None)
            end = last + 1 if last is not None else None
            need(first == raw.get("first_active_sample") and end == raw.get("last_active_sample_exclusive"),
                 "speech_activity_metadata_mismatch")
            result[key] = Clip(bundle, identity, data, digest(data), digest(pcm), count, first, end, threshold)
    return result


@dataclass(frozen=True)
class Expectation:
    assertion_id: str
    kind: str
    predicate: str
    evidence_bundles: tuple[str, ...]
    verdict: str = "unmeasured"


@dataclass(frozen=True)
class PrerequisiteExpectation:
    """Evaluator requirements, not invented backend receipt/wire fields."""
    prerequisite_id: str
    kind: str
    setup_occurrence_id: str
    must_precede_occurrence_id: str
    predicate: str
    required_canonical_evidence: tuple[str, ...]
    evidence_bundles: tuple[str, ...]
    runtime_barrier_supported: bool = False
    verdict: str = "unmeasured"


@dataclass
class CompiledScenario:
    manifest: dict
    speech: dict
    expectations: dict
    assets: dict[str, bytes]


def compile_scenario(case, *, cuts, binding, frame_plan, frame_root, clips, authority=AUTHORITY,
                     duration=90, start=5, timeout=10, followup_gap=1,
                     output_trigger="energy_cursor", response_delay=0.3, max_pcm_age=0.05):
    """Compile stimulus-only diagnostic variants of the frozen E2E cases.

    Native output-start after_cursor is a floor, not dynamic provider identity.
    Floors increase and post-run accounting requires strictly consumed cursors.
    Opt-in response triggers require an admitted control identity and fresh
    decoded PCM after actual preceding SDK input completion. They do not prove
    which request the response answers or which RTP response supplied a sample.
    Fixed-time followups use worst-case bounded predecessor start plus full clip
    duration. They are NOT playback-drain or canonical-state waits. Native input
    readiness/overlap checks remain authoritative if callbacks stall.
    """
    need(case in CASES, "unknown_scenario")
    need(type(cuts) is int and (cuts == 0 if case == "baseline" else 1 <= cuts <= 3), "invalid_cut_count")
    duration = finite(duration, "invalid_duration", 1)
    start = finite(start, "invalid_start", 0, duration)
    timeout = finite(timeout, "invalid_trigger_timeout", 0.01, duration)
    gap = finite(followup_gap, "invalid_followup_gap", 0.1, 10)
    need(output_trigger in {"energy_cursor", "response_correlated"}, "unsupported_output_trigger")
    correlated = output_trigger == "response_correlated"
    if correlated:
        need(cuts > 0, "response_trigger_requires_cut_variant")
        response_delay = finite(response_delay, "invalid_response_delay", 0, timeout)
        max_pcm_age = finite(max_pcm_age, "invalid_pcm_age", 0.001, timeout)
        need(response_delay < timeout, "response_delay_exhausts_timeout")
    need(isinstance(binding, dict) and set(binding) == {
        "run_id", "backend_base_url", "owner_id", "area_id", "memory_bank_id", "voice_route"}, "invalid_binding_fields")
    run = binding["run_id"]
    need(isinstance(run, str) and re.fullmatch(r"[a-f0-9]{32}", run), "invalid_run_id")
    need(binding["voice_route"] in {"realtime", "live"}, "unsupported_voice_route")
    need(not correlated or binding["voice_route"] == "realtime", "response_trigger_unsupported_for_route")
    for key in ("owner_id", "area_id", "memory_bank_id"):
        need(isinstance(binding[key], str) and 0 < len(binding[key]) <= 256, "invalid_scope_identity")
    try:
        url = urlsplit(binding["backend_base_url"])
        need(url.scheme in {"http", "https"} and url.hostname in {"localhost", "127.0.0.1", "::1"}
             and not url.username and not url.password and not url.query and not url.fragment, "nonlocal_or_credentialed_host")
    except (ValueError, TypeError):
        raise ScenarioError("invalid_host_url") from None
    authority_bytes = bounded_read(authority, 128 * 1024)
    authority_text = authority_bytes.decode("utf-8")
    need(all(identity in authority_text for identity in CASES[case]), "authority_case_missing")
    assets, frames, total_frames, last_time = {}, [], 0, -1.0
    need(isinstance(frame_plan, list) and 0 < len(frame_plan) <= 240, "invalid_frames")
    for index, frame in enumerate(frame_plan):
        need(isinstance(frame, dict), "invalid_frame")
        at = finite(frame.get("at_seconds"), "invalid_frame_time", 0, duration)
        need(last_time < at < duration, "nonmonotonic_or_late_frame")
        data = bounded_read(local_path(frame_root, frame.get("relative_path")), 4 * 1024 * 1024)
        need(data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9"), "invalid_jpeg_signature")
        need(digest(data) == frame.get("sha256"), "frame_hash_mismatch")
        total_frames += len(data)
        need(total_frames <= 64 * 1024 * 1024, "frame_budget_exceeded")
        path = f"assets/frame_{index:04d}.jpg"
        assets[path] = data
        frames.append(dict(at_seconds=at, relative_path=path, sha256=digest(data)))
        last_time = at

    speech, selected, utterances, occurrences, prerequisites = [], {}, [], [], []
    latest_start, previous_duration, floor, total_samples = start, 0.0, 0, 0

    def add(key, intent, *, interrupt=False, silence_allowed=False, not_before=None):
        nonlocal latest_start, previous_duration, floor, total_samples
        need(key in clips, "required_speech_clip_missing")
        clip = clips[key]
        clip_duration = clip.sample_count / RATE
        if key not in selected:
            identity = f"clip_{len(selected):03d}"
            selected[key] = identity
            path = f"assets/{identity}.wav"
            assets[path] = clip.wav
            speech.append(dict(id=identity, relative_path=path, sha256=clip.sha256, sample_rate_hz=RATE,
                               channels=1, sample_width_bytes=2, sample_count=clip.sample_count,
                               first_active_sample=clip.first, last_active_sample_exclusive=clip.end,
                               energy_threshold_abs_pcm16=clip.threshold))
        identity = selected[key]
        if interrupt:
            if correlated:
                need(bool(utterances), "response_trigger_missing_preceding_input")
                trigger = dict(kind="response_output_start", after_input_ordinal=len(utterances) - 1,
                               delay_seconds=response_delay, timeout_seconds=timeout,
                               max_pcm_age_seconds=max_pcm_age)
                # This wait starts after actual successful completion of the
                # preceding clip, not its nominal start or first active sample.
                latest_start += previous_duration + timeout
            else:
                # Positive delay prevents reuse of the event that triggered the
                # preceding step. Cover its full nominal input duration as well.
                delay = max(0.3, previous_duration + 0.1)
                need(delay < timeout, "timeout_cannot_cover_preceding_input")
                trigger = dict(kind="output_start", after_cursor=floor, delay_seconds=delay, timeout_seconds=timeout)
                floor += 1
                latest_start += timeout
        else:
            if utterances:
                latest_start += previous_duration + gap
            if not_before is not None:
                latest_start = max(latest_start, not_before)
            trigger = dict(kind="at_time", at_seconds=latest_start)
        total_samples += clip.sample_count
        need(total_samples <= 60 * RATE and len(utterances) < 20, "input_budget_exceeded")
        need(latest_start + clip_duration + 5 <= duration, "scenario_cannot_fit_deadline")
        ordinal = len(utterances)
        occurrence_id = f"{run}/utterance/{ordinal}"
        utterances.append(dict(clip_id=identity, trigger=trigger))
        occurrences.append(dict(ordinal=ordinal, occurrence_id=occurrence_id, trigger_id=occurrence_id + "/trigger",
            clip_id=identity, source_clip=key, pcm16_sha256=clip.pcm_sha256, sample_count=clip.sample_count,
            intent=intent, planned_cut=interrupt, silence_after_input_allowed=silence_allowed,
            trigger=trigger.copy(), latest_nominal_start_seconds=latest_start,
            input_duration_seconds=clip_duration, execution_verdict="not_executed"))
        previous_duration = clip_duration

    def prerequisite(identity, kind, predicate, fields, bundles):
        prerequisites.append(PrerequisiteExpectation(identity, kind, occurrences[-1]["occurrence_id"],
            f"{run}/utterance/{len(occurrences)}", predicate, tuple(fields), tuple(bundles)))

    add("baseline/current_view", "current_view_question")
    if case in TASK_CASES:
        add("setup/background_task_setup", "request_one_bounded_read_only_background_task")
        prerequisite("task_created", "canonical_task_creation",
            "one_workbench_plan_task_accepted_from_setup_before_dependent_question_or_cut; spoken_ack_is_not_creation",
            ("owner_and_session_scope", "accepted_request_and_setup_utterance", "task_id_and_revision", "accepted_at",
             "read_only_allowed_effects", "actual_task_state_before_each_dependent_utterance"), ("E1", "E2", "E7"))
        # This is a real spoken probe, not a substitute for the prerequisite
        # receipt or a way to infer state from its answer. It also separates
        # the long setup PCM from the first bounded output-start wait.
        add("state/operation_status", "query_setup_task_status_without_assuming_admission")
    if case in {"pause_continue", "ambiguous_stop", "parallel_keep_working"}:
        interrupt_clip, intent, follow_clip, follow_intent = {
            "pause_continue": ("speech_only_pause", "pause_speech_only", "continue_answer", "continue_answer"),
            "ambiguous_stop": ("ambiguous_stop", "ambiguous_stop_not_implicit_task_cancel", "continue_answer", "continue_answer"),
            "parallel_keep_working": ("parallel_question", "parallel_question", "keep_working", "keep_independent_work"),
        }[case]
        for _ in range(cuts):
            add("conversation/" + interrupt_clip, intent, interrupt=True, silence_allowed=case != "parallel_keep_working")
            # A pause is allowed to produce silence: continuation never waits
            # for a reply to "hold on" or treats silence as provider failure.
            add("conversation/" + follow_clip, follow_intent)
    elif case == "correction_chain":
        for key in ("correction_left", "correction_right", "correction_label")[:cuts]:
            add("conversation/" + key, "replace_current_goal", interrupt=True)
    elif case == "explicit_cancel":
        for index in range(cuts):
            add("conversation/" + ("explicit_cancel" if index == 0 else "repeat_cancel"),
                "cancel_same_task_idempotently", interrupt=True)
        add("state/operation_status", "query_actual_operation_status")
    elif case == "quiet_direct_resume":
        add("setup/reminder_setup", "teach_one_shot_thirty_second_reminder")
        reminder_due_nominal = latest_start + previous_duration + 30
        prerequisite("reminder_created", "canonical_reminder_creation",
            "one_active_one_shot_stretch_reminder_accepted_before_quiet_request; due_time_and_quiet_order_observed_not_assumed",
            ("owner_and_session_scope", "setup_utterance_and_creation_receipt", "reminder_id_and_revision",
             "canonical_due_time", "active_status", "quiet_acceptance_before_due", "all_channel_delivery_or_suppression"),
            ("E1", "E2", "E4", "E6"))
        add("state/quiet_until_resume", "enter_persistent_quiet", interrupt=True)
        add("state/quiet_direct_question", "direct_reply_preserves_quiet")
        for _ in range(cuts - 1):
            add("conversation/speech_only_pause", "pause_speech_preserve_quiet", interrupt=True, silence_allowed=True)
            add("state/quiet_direct_question", "direct_reply_preserves_quiet")
        # Reserve an opportunity for the spoken 30-second reminder. This
        # elapsed-time placement is neither canonical due evidence nor a
        # quiet-state barrier; acceptance remains inconclusive without both.
        add("state/resume_summary", "explicit_resume_rechecks_relevance", not_before=reminder_due_nominal + 5)
    elif case == "memory_correction_deletion":
        add("setup/memory_setup", "remember_explicit_user_report_not_visual_evidence")
        prerequisite("memory_created", "canonical_memory_creation",
            "setup_user_report_source_accepted_before_correction_with_observed_publication_state; not_seeded_expected_answer",
            ("owner_session_area_and_bank_scope", "setup_utterance_source_id_and_version", "canonical_acceptance_order",
             "publication_receipt_or_pending_state", "correction_and_deletion_source_dependencies"), ("E1", "E2", "E3", "E7"))
        add("state/correct_notebook", "correct_source_not_observed_move", interrupt=True)
        add("state/recall_notebook", "recall_corrected_source")
        add("state/forget_notebook", "delete_selected_source", interrupt=cuts >= 2)
        add("state/recall_notebook", "recall_without_deleted_detail")
        if cuts == 3:
            add("state/forget_notebook", "repeat_delete_without_resurrection", interrupt=True)
            add("state/recall_notebook", "recall_without_deleted_detail")
    elif case == "late_cancel":
        for _ in range(cuts):
            add("state/operation_status", "query_actual_operation_status")
            add("state/cancel_if_pending", "cancel_or_report_committed_or_unknown_effect", interrupt=True)

    checks = [Expectation("inputs", "native_evidence", "every_planned_occurrence_has_complete_sdk_accepted_active_input", ("E0", "E5")),
              Expectation("truth", "answer_fidelity", "answers_use_available_evidence_and_preserve_uncertainty", ("E1", "E2", "E5"))]
    barriers = ["reviewed_frame_windows_and_rights", "actual_isolated_host_scope", "external_answer_and_audio_review"]
    if cuts:
        checks += [Expectation("cutoff", "output_evidence", "old_answer_was_active_and_cutoff_has_explicit_sample_annotations", ("E0", "E5")),
                   Expectation("owners", "canonical_state", "one_speaker_no_stale_continuation_or_unintended_task_cancellation", ("E1", "E2", "E4", "E7"))]
        barriers += (["response_control_trigger_and_local_fence_evidence", "semantic_output_attribution"]
                     if correlated else ["correlated_output_active_barrier"])
        barriers.append("playback_drain_barrier")
    if case in TASK_CASES:
        checks.append(Expectation("task_effects", "task_state", "same_authorized_task_no_duplicate_effects_truthful_cancel_and_unknown_status", ("E1", "E2", "E4")))
        checks.append(Expectation("task_phase", "task_state",
            "task_running_at_required_cuts_or_case_not_reached; committed_or_unknown_effects_only_in_explicit_race_variant",
            ("E1", "E2", "E4")))
        barriers += ["canonical_task_creation_and_running_state", "canonical_task_dispatch_commit_barriers"]
    if case == "correction_chain":
        checks.append(Expectation("goal_revision", "goal_state", "only_current_goal_revision_can_resolve_current_intent", ("E1", "E2", "E4")))
        barriers.append("canonical_goal_revision_and_delayed_result_barriers")
    if case == "quiet_direct_resume":
        checks.append(Expectation("quiet", "quiet_policy", "direct_replies_preserve_quiet_resume_rechecks_and_coalesces", ("E1", "E2", "E4", "E6")))
        barriers += ["canonical_reminder_creation_and_due_order", "canonical_quiet_acceptance_barrier",
                     "unsolicited_event_and_all_channel_silence_observation"]
    if case == "memory_correction_deletion":
        checks.append(Expectation("memory", "memory_sources", "corrections_supersede_deletions_fence_no_resurrection_independent_sources_survive", ("E1", "E2", "E3", "E4", "E7")))
        barriers += ["publication_correction_deletion_barriers", "pending_retrieval_fault_controls",
                     "fresh_session_restart_and_days_to_months_clock", "independent_memory_positive_control"]
    manifest = dict(version=1, **binding, bearer_token_file="token.txt", duration_seconds=duration,
                    speech_manifest="speech.json", frames=frames, utterances=utterances)
    variant = ("/native_response_control_with_spoken_setup" if case in SETUP_CASES else "/native_response_control_stimulus") if correlated else (
        "/native_onset_with_spoken_setup" if case in SETUP_CASES else "/native_onset_stimulus")
    expectations = dict(schema_version=1, expectations_version=1,
        case_revision=(2 if case in SETUP_CASES else 1) + int(correlated),
        case_ids=list(CASES[case]), variant=case + variant, authority_sha256=digest(authority_bytes),
        authority="backend/evals/responsive_voice/replay_cases/E2E_EXPECTATIONS.md", run_id=run,
        evidence_class="offline_compilation_not_frontend_execution", planned_cuts=cuts, actual_cuts=None,
        occurrences=occurrences, assertions=[asdict(check) for check in checks], qualification="unmeasured",
        prerequisites=[asdict(check) for check in prerequisites],
        prerequisite_evidence_contract="typed evaluator requirements must map to actual canonical records; neither speech, ASR, fixed waits nor assistant words satisfy creation",
        setup_safety="Workbench Plan is one bounded reasoning-only task with no browsing/files/messages/purchases/external actions; reminder and memory writes belong only to isolated evaluation owners",
        unresolved_acceptance_barriers=barriers, aggregate_percentile_qualification=False,
        manifest_sha256=digest(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()),
        frame_progression="original supplied timestamps; no pause, repeated tail, or future frames added",
        scheduling_limit="fixed followups are bounded nominal input scheduling, not playback/state completion; native overlap checks may fail closed",
        output_trigger=output_trigger,
        cursor_limit=("not used by response_output_start" if correlated else
            "local energy-onset cursor, rearmed on input enqueue; may be continuing old speech, never provider response identity"),
        response_correlation_limit="authorized control-gate interval only; request causality, per-packet RTP attribution, render drain and hearing remain unmeasured",
        actual_cut_limit="requires explicit stop evidence and external old-answer activity/sample attribution; completed input or a control binding alone is insufficient",
        setup_limit="token.txt must be staged separately; host enforces isolation, compiler does not attest it",
        unsupported_runtime_controls=["after_input_complete", "after_state", "playback_drained",
            *( ["per_packet_response_attribution", "request_response_causality"] if correlated else ["provider_response_correlated_output"] ),
            "reconnect", "domain_clock_jump"],
        synthetic_input_seconds=total_samples / RATE)
    return CompiledScenario(manifest, {"format": "jarvis.speech_assets.v1", "clips": speech}, expectations, assets)


def account_execution(compiled, capture, pcm, *, task_evidence=None):
    """Account all occurrences using the existing reader, never count actual cuts.

    The native cursor identifies a local energy onset, not an active old answer.
    This function checks consumption and actual SDK input. The reader can also
    validate opt-in response control bindings and local-stop records. These do
    not establish semantic request causality or per-packet response attribution.
    """
    expected = compiled.expectations
    need(UUID(capture.get("run_id")) == UUID(expected["run_id"]), "capture_run_mismatch")
    reader_options = {"task_evidence": task_evidence} if task_evidence is not None else {}
    distinct = expected.get("distinct_provider_turns", False)
    if distinct:
        reader_options["replay_manifest_bytes"] = json.dumps(compiled.manifest).encode()
    evidence = analyze_capture(capture, pcm, **reader_options)
    rows = {item["ordinal"]: item for item in evidence["occurrences"]}
    raw = {item["ordinal"]: item for item in capture["utterances"]}
    need(len(rows) <= len(expected["occurrences"]), "unplanned_occurrence")
    starts = capture.get("output_starts", [])
    need(isinstance(starts, list) and len(starts) <= 20, "invalid_output_start_history")
    last_at, last_cursor = -1.0, 0
    for start in starts:
        need(isinstance(start, dict) and type(start.get("local_cursor")) is int
             and start["local_cursor"] > last_cursor, "nonmonotonic_output_cursor")
        at = finite(start.get("uptime"), "invalid_output_start_time", 0, 1e12)
        need(at >= last_at, "nonmonotonic_output_start_time")
        last_cursor, last_at = start["local_cursor"], at
    accounted, consumed, completed_cuts, completed_response_inputs = [], 0, 0, 0
    local_stops = []
    for plan in expected["occurrences"]:
        index = plan["ordinal"]
        row, actual = rows.get(index), raw.get(index)
        result = dict(occurrence_id=plan["occurrence_id"], planned_cut=plan["planned_cut"],
                      input_verdict="not_recorded", cursor_verdict="not_applicable", consumed_cursor=None,
                      response_trigger_verdict="not_applicable")
        if row is not None:
            need(row["occurrence_id"] == plan["occurrence_id"] and actual.get("clip_id") == plan["clip_id"]
                 and actual.get("trigger") == plan["trigger"] and actual.get("sample_count") == plan["sample_count"]
                 and actual.get("pcm16_sha256") == plan["pcm16_sha256"], "capture_occurrence_binding_mismatch")
            result["input_verdict"] = row["input_verdict"]
        if distinct and plan["trigger"]["kind"] == "after_provider_input_finalized":
            result["provider_input_finalization_verdict"] = (
                row.get("provider_input_finalization_verdict", "missing") if row else "missing")
        if plan["planned_cut"]:
            if plan["trigger"]["kind"] == "response_output_start":
                # Only the evidence reader validates receipt/sample/gate links.
                # Never promote a raw client flag into an accepted measurement.
                verdict = row.get("response_trigger_verdict", "unmeasured") if row else "missing"
                result["response_trigger_verdict"] = verdict
                if row and row["input_verdict"] == "sdk_completed_active_input" and verdict == "validated_control_binding":
                    completed_response_inputs += 1
                    if row.get("response_control_stop") is not None:
                        local_stops.append(row["response_control_stop"])
                accounted.append(result)
                continue
            result["cursor_verdict"] = "input_not_completed"
            if row is not None and row["input_verdict"] == "sdk_completed_active_input":
                trigger_at = actual.get("triggered_uptime_seconds")
                wait_at = raw.get(index - 1, {}).get("triggered_uptime_seconds")
                result["cursor_verdict"] = "matching_fresh_cursor_unproven"
                if type(trigger_at) in (int, float) and type(wait_at) in (int, float):
                    available = [item for item in starts if item["uptime"] <= trigger_at]
                    candidate = available[-1] if available else None
                    trigger = plan["trigger"]
                    if (candidate and candidate["local_cursor"] > max(consumed, trigger["after_cursor"])
                            and candidate["uptime"] >= wait_at
                            and candidate["uptime"] + trigger["delay_seconds"] <= trigger_at + 1e-6
                            and trigger_at - wait_at < trigger["timeout_seconds"]):
                        consumed = candidate["local_cursor"]
                        result.update(consumed_cursor=consumed, cursor_verdict="fresh_cursor_not_active_answer_proof")
                        completed_cuts += 1
        accounted.append(result)
    all_inputs = all(item["input_verdict"] == "sdk_completed_active_input" for item in accounted)
    if distinct:
        all_inputs = all_inputs and all(item.get("provider_input_finalization_verdict",
            "validated_provider_transaction") == "validated_provider_transaction" for item in accounted)
    by_occurrence = {item["occurrence_id"]: item for item in accounted}
    prerequisite_results = []
    for prerequisite in expected.get("prerequisites", []):
        setup = by_occurrence[prerequisite["setup_occurrence_id"]]
        dependent = by_occurrence[prerequisite["must_precede_occurrence_id"]]
        reached = all(item["input_verdict"] == "sdk_completed_active_input" for item in (setup, dependent))
        prerequisite_results.append(dict(prerequisite_id=prerequisite["prerequisite_id"], kind=prerequisite["kind"],
            setup_occurrence_id=setup["occurrence_id"], setup_input_verdict=setup["input_verdict"],
            must_precede_occurrence_id=dependent["occurrence_id"], dependent_input_verdict=dependent["input_verdict"],
            verdict="inconclusive" if reached else "not_reached", canonical_creation_confirmed=False,
            canonical_receipt=None, runtime_barrier_supported=False,
            reason="canonical_creation_and_order_not_observed" if reached else "setup_or_dependent_input_not_completed"))
    semantic_execution = ("not_reached" if any(item["verdict"] == "not_reached" for item in prerequisite_results)
                          else "inconclusive" if prerequisite_results else "unmeasured")
    return dict(schema_version=1, run_id=expected["run_id"], occurrences=accounted,
        planned_cuts=expected["planned_cuts"], completed_cursor_triggered_inputs=completed_cuts, actual_cuts=None,
        completed_response_triggered_inputs=completed_response_inputs,
        response_control_stop_measurements=local_stops,
        actual_cut_count_verdict="unmeasured_requires_external_active_old_answer_annotations",
        input_execution="complete" if all_inputs and completed_cuts + completed_response_inputs == expected["planned_cuts"] else "incomplete",
        semantic_execution=semantic_execution, prerequisites=prerequisite_results,
        timing_blockers=evidence["timing_blockers"], qualification="unmeasured", semantic_assertions_evaluated=False,
        provider_response_attribution="unavailable", response_semantic_causality="unknown",
        hearing_measured=False, aggregate_percentile_qualification=False)


_compile_legacy_scenario = compile_scenario


def _compile_task_cancellation(*, cuts, **kwargs):
    """One explicit cancellation, not speech pause or fabricated task success."""
    need(type(cuts) is int and cuts == 1, "task_cancel_requires_one_explicit_cut")
    need(kwargs["binding"]["voice_route"] == "realtime", "task_cancel_requires_realtime")
    duration = finite(kwargs.get("duration", 120), "invalid_duration", 1, 120)
    compiled = _compile_legacy_scenario("explicit_cancel", cuts=1, **dict(kwargs,
        duration=duration, start=7, timeout=10, followup_gap=1, output_trigger="response_correlated",
        response_delay=.3, max_pcm_age=.05))
    manifest, expected = compiled.manifest, compiled.expectations
    manifest["utterances"] = manifest["utterances"][1:]
    expected["occurrences"] = expected["occurrences"][1:]
    need("conversation/post_cancel_question" in kwargs["clips"], "post_cancel_question_clip_required")
    clip = kwargs["clips"]["conversation/post_cancel_question"]
    identity = f"clip_{len(compiled.speech['clips']):03d}"
    path = f"assets/{identity}.wav"
    compiled.assets[path] = clip.wav
    compiled.speech["clips"].append(dict(id=identity, relative_path=path, sha256=clip.sha256,
        sample_rate_hz=RATE, channels=1, sample_width_bytes=2, sample_count=clip.sample_count,
        first_active_sample=clip.first, last_active_sample_exclusive=clip.end,
        energy_threshold_abs_pcm16=clip.threshold))
    manifest["utterances"][-1] = dict(clip_id=identity, trigger={})
    expected["occurrences"][-1] = dict(clip_id=identity, source_clip="conversation/post_cancel_question",
        pcm16_sha256=clip.pcm_sha256, sample_count=clip.sample_count,
        intent="canonical_status_and_unrelated_question_without_task_restart", planned_cut=False,
        silence_after_input_allowed=False, input_duration_seconds=clip.sample_count / RATE,
        execution_verdict="not_executed")
    manifest["task_barrier"] = dict(version=1, scope_file="task-scope.json", request_input_ordinal=0)
    old_setup, old_status = (row["occurrence_id"] for row in expected["occurrences"][:2])
    latest, previous_duration = 7.0, 0.0
    for ordinal, (item, row) in enumerate(zip(manifest["utterances"], expected["occurrences"])):
        if ordinal == 0:
            trigger = dict(kind="at_time", at_seconds=7.0)
        elif ordinal == 1:
            trigger = dict(kind="after_task_claim", after_input_ordinal=0, timeout_seconds=30.0)
        elif ordinal == 2:
            trigger = dict(kind="response_output_start", after_input_ordinal=1, delay_seconds=.3,
                timeout_seconds=10.0, max_pcm_age_seconds=.05, require_task_claim=True)
        else:
            trigger = dict(kind="after_provider_input_finalized", after_input_ordinal=ordinal - 1,
                timeout_seconds=10.0)
        if ordinal:
            latest += previous_duration + trigger["timeout_seconds"]
        occurrence = f"{expected['run_id']}/utterance/{ordinal}"
        row.update(ordinal=ordinal, occurrence_id=occurrence, trigger_id=occurrence + "/trigger",
            trigger=dict(trigger), latest_nominal_start_seconds=latest)
        item["trigger"] = trigger
        previous_duration = row["input_duration_seconds"]
        need(latest + previous_duration + 5 <= duration, "task_trigger_deadlines_exceed_run_duration")
    remapped = {old_setup: expected["occurrences"][0]["occurrence_id"],
                old_status: expected["occurrences"][1]["occurrence_id"]}
    for prerequisite in expected["prerequisites"]:
        prerequisite.update(setup_occurrence_id=remapped[prerequisite["setup_occurrence_id"]],
            must_precede_occurrence_id=remapped[prerequisite["must_precede_occurrence_id"]],
            runtime_barrier_supported=True, verdict="unmeasured")
    used = {item["clip_id"] for item in manifest["utterances"]}
    compiled.speech["clips"] = [row for row in compiled.speech["clips"] if row["id"] in used]
    wavs = {row["relative_path"] for row in compiled.speech["clips"]}
    compiled.assets = {name: data for name, data in compiled.assets.items() if not name.endswith(".wav") or name in wavs}
    input_seconds = sum(row["input_duration_seconds"] for row in expected["occurrences"])
    need(input_seconds <= 60, "input_budget_exceeded")
    expected.update(case_revision=9, variant="explicit_cancel/native_claim_distinct_status_and_conversation",
        case_ids=[*expected["case_ids"], "E2E-WORK-003"],
        task_aware=True, distinct_provider_turns=True, task_scope_staged=False,
        task_scope_status="missing_until_actual_worker_readiness",
        required_runner_staging=["task-scope.json", "../controls/session-binding.json",
            "../controls/task-barrier.json", "../controls/stimulus-completion.json"],
        task_barrier_contract="NATIVE_TASK_REPLAY_CONTRACT.md", synthetic_input_seconds=input_seconds,
        canonical_input_binding="unknown", provider_input_finalization_basis="replay_exclusive_provider_transaction",
        scheduling_limit="Post-cancel probes wait for preceding SDK completion AND the same reserved provider item finalization; not tool completion.",
        setup_limit="Actual worker scope and active claim are mandatory before cancellation; post-cancel probes never require the cancelled claim.",
        task_terminal_expectation="canonical_cancel_or_honest_completion_before_cancel_race_not_spoken_ack",
        result_control_expectation="cancelled_held_attempt_must_not_be_released; release_is_not_required_after_cancel",
        cancellation_ordinal=2, status_probe_ordinal=3, conversation_probe_ordinal=3,
        status_and_conversation_probe_share_user_turn=True,
        late_callback_injection_supported=False,
        distinct_turn_limit="Provider transaction identity is not canonical intent or tool completion proof.")
    expected["assertions"] += [asdict(Expectation(identity, kind, predicate, evidence)) for identity, kind, predicate, evidence in (
        ("explicit_task_cancel", "canonical_task_state", "exact_task_cancel_source_revision_fence_and_terminal_or_pending_state_observed; no_pause_as_cancel", ("E1", "E2", "E4")),
        ("obsolete_result", "result_delivery", "no_post_cancel_obsolete_result_speech_or_duplicate_task; already_completed_race_reported_honestly", ("E1", "E2", "E4", "E6")),
        ("conversation_after_cancel", "answer_fidelity", "status_uses_current_canonical_source_and_unrelated_question_answered_without_task_restart", ("E1", "E2", "E5")),
    )]
    expected["manifest_sha256"] = digest(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    return compiled


def compile_scenario(case, *, cuts, task_aware=False, distinct_provider_turns=False, **kwargs):
    """Keep legacy bytes unchanged; opt in to the frozen task-claim sequence.

    The prepared manifest is deliberately not executable until the runner
    supplies task-scope.json from actual worker readiness. This compiler has
    no worker identity, canonical claim, or authority to manufacture either.
    """
    need(type(task_aware) is bool, "invalid_task_aware_option")
    if task_aware and case == "explicit_cancel":
        need(distinct_provider_turns is True, "task_cancel_requires_distinct_provider_inputs")
        return _compile_task_cancellation(cuts=cuts, **kwargs)
    need(type(distinct_provider_turns) is bool, "invalid_distinct_provider_turns_option")
    need(not distinct_provider_turns or task_aware, "distinct_provider_turns_requires_task_aware")
    if not task_aware:
        return _compile_legacy_scenario(case, cuts=cuts, **kwargs)
    need(case == "pause_continue" and type(cuts) is int and 1 <= cuts <= 3,
         "task_aware_requires_pause_continue_one_two_three")
    need(kwargs["binding"]["voice_route"] == "realtime", "task_aware_requires_realtime")
    requested_duration = kwargs.get("duration", 120)
    extended_three_cut = (distinct_provider_turns and cuts == 3
                          and type(requested_duration) in (int, float) and requested_duration == 150)
    duration = finite(requested_duration, "invalid_duration", minimum=1,
                      maximum=150 if extended_three_cut else 120)
    # Reuse the existing speech/image validation and expectation authority,
    # not a second asset loader. The template's first question is never sent.
    template = dict(kwargs, duration=120, start=7, timeout=10, followup_gap=1,
                    output_trigger="response_correlated", response_delay=0.3, max_pcm_age=0.05)
    compiled = _compile_legacy_scenario(case, cuts=cuts, **template)
    manifest, expected = compiled.manifest, compiled.expectations
    manifest["duration_seconds"] = duration
    need(all(frame["at_seconds"] < duration for frame in manifest["frames"]), "frame_after_task_run_end")
    manifest["utterances"] = manifest["utterances"][1:]
    expected["occurrences"] = expected["occurrences"][1:]
    manifest["task_barrier"] = dict(version=1, scope_file="task-scope.json", request_input_ordinal=0)
    remapped, latest, previous_duration = {}, 7.0, 0.0
    for ordinal, (utterance, occurrence) in enumerate(zip(manifest["utterances"], expected["occurrences"])):
        old_identity = occurrence["occurrence_id"]
        identity = f"{expected['run_id']}/utterance/{ordinal}"
        remapped[old_identity] = identity
        if ordinal == 0:
            trigger = dict(kind="at_time", at_seconds=7.0)
        elif ordinal == 1:
            trigger = dict(kind="after_task_claim", after_input_ordinal=0, timeout_seconds=30.0)
            latest += previous_duration + 30.0
        elif occurrence["planned_cut"]:
            trigger = dict(kind="response_output_start", after_input_ordinal=ordinal - 1,
                           delay_seconds=0.3, timeout_seconds=10.0, max_pcm_age_seconds=0.05,
                           require_task_claim=True)
            latest += previous_duration + 10.0
        else:
            if distinct_provider_turns:
                trigger = dict(kind="after_provider_input_finalized", after_input_ordinal=ordinal - 1,
                               timeout_seconds=10.0)
                latest += previous_duration + trigger["timeout_seconds"]
            else:
                trigger = dict(kind="after_input_complete", after_input_ordinal=ordinal - 1,
                               delay_seconds=1.0, timeout_seconds=10.0)
                latest += previous_duration + trigger["delay_seconds"]
        utterance["trigger"] = trigger
        occurrence.update(ordinal=ordinal, occurrence_id=identity, trigger_id=identity + "/trigger",
                          trigger=dict(trigger), latest_nominal_start_seconds=latest)
        previous_duration = occurrence["input_duration_seconds"]
        need(latest + previous_duration + 5 <= duration, "task_trigger_deadlines_exceed_run_duration")
    for prerequisite in expected["prerequisites"]:
        prerequisite["setup_occurrence_id"] = remapped[prerequisite["setup_occurrence_id"]]
        prerequisite["must_precede_occurrence_id"] = remapped[prerequisite["must_precede_occurrence_id"]]
        prerequisite["runtime_barrier_supported"] = True
        prerequisite["verdict"] = "unmeasured"
    used = {item["clip_id"] for item in manifest["utterances"]}
    compiled.speech["clips"] = [clip for clip in compiled.speech["clips"] if clip["id"] in used]
    # Drop the unplayed current-view WAV, while retaining every visual frame.
    used_wavs = {clip["relative_path"] for clip in compiled.speech["clips"]}
    assets = {path: data for path, data in compiled.assets.items()
              if not path.endswith(".wav") or path in used_wavs}
    expected.update(case_revision=4, variant="pause_continue/native_task_claim_response_control",
        task_aware=True, task_scope_staged=False, task_scope_status="missing_until_actual_worker_readiness",
        required_runner_staging=["task-scope.json", "../controls/session-binding.json",
                                 "../controls/task-barrier.json", "../controls/stimulus-completion.json"],
        task_barrier_contract="NATIVE_TASK_REPLAY_CONTRACT.md",
        scheduling_limit="Claim waits and followups use actual SDK completion; frames keep their original clock.",
        setup_limit="Runner must stage actual worker scope; spoken setup and SDK completion are not canonical admission.",
        synthetic_input_seconds=sum(item["input_duration_seconds"] for item in expected["occurrences"]))
    expected["unsupported_runtime_controls"] = [name for name in expected["unsupported_runtime_controls"]
        if name != "after_input_complete"]
    if distinct_provider_turns:
        expected.update(case_revision=6 if extended_three_cut else 5,
            variant="pause_continue/native_task_claim_distinct_provider_transactions" +
                ("_scene150" if extended_three_cut else ""),
            distinct_provider_turns=True,
            provider_input_finalization_basis="replay_exclusive_provider_transaction",
            canonical_input_binding="unknown",
            scheduling_limit="Continue requires complete successful SDK input AND the same reserved provider item finalized; frames keep their original clock.",
            distinct_turn_limit="Local provider transaction evidence only; canonical/semantic pause, cuts and task continuity need separate qualification.")
        if extended_three_cut:
            expected.update(declared_task_scene_seconds=150, scene_extension_scope="three_cuts_only")
    expected["manifest_sha256"] = digest(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    return CompiledScenario(manifest, compiled.speech, expected, assets)


def write_inputs(compiled, output):
    output = Path(output)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    (output / "assets").mkdir()
    for path, data in compiled.assets.items():
        with (output / path).open("xb") as handle:
            handle.write(data)
    for name, value in (("manifest.json", compiled.manifest), ("speech.json", compiled.speech), ("expectations.json", compiled.expectations)):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--cuts", type=int)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--frames", required=True, help="JSON list; frame paths relative to this JSON file")
    parser.add_argument("--output", required=True, help="fresh directory; never overwrite previous evidence")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--task-aware", action="store_true", help="task-claim pause/continue or explicit cancel; runner must stage actual worker scope")
    parser.add_argument("--distinct-provider-turns", action="store_true",
                        help="explicit v5 task fixture; continue waits for SDK completion AND reserved provider input finalization")
    parser.add_argument("--start-seconds", type=float, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=10)
    parser.add_argument("--output-trigger", choices=("energy_cursor", "response_correlated"), default="energy_cursor")
    parser.add_argument("--response-delay-seconds", type=float, default=0.3)
    parser.add_argument("--max-pcm-age-seconds", type=float, default=0.05)
    for bundle, default in DEFAULT_BUNDLES.items():
        parser.add_argument("--" + bundle + "-speech", default=str(default))
    args = parser.parse_args(argv)
    try:
        clips = load_bundles({key: getattr(args, key + "_speech") for key in DEFAULT_BUNDLES
                              if key != "setup" or args.case in SETUP_CASES})
        compiled = compile_scenario(args.case, cuts=args.cuts if args.cuts is not None else (0 if args.case == "baseline" else 1),
            binding=read_json(args.binding), frame_plan=read_json(args.frames), frame_root=Path(args.frames).parent,
            clips=clips, duration=args.duration if args.duration is not None else (120 if args.task_aware else 90),
            task_aware=args.task_aware, distinct_provider_turns=args.distinct_provider_turns,
            start=args.start_seconds, timeout=args.timeout_seconds,
            output_trigger=args.output_trigger, response_delay=args.response_delay_seconds,
            max_pcm_age=args.max_pcm_age_seconds)
        write_inputs(compiled, args.output)
    except (ScenarioError, OSError, ValueError) as error:
        print(json.dumps({"status": "compile_failed", "error_code": str(error) if isinstance(error, ScenarioError) else "input_or_output_unavailable"}))
        return 2
    print(json.dumps({"status": "inputs_prepared_not_executed", "planned_cuts": compiled.expectations["planned_cuts"],
                      "actual_cuts": None, "qualification": "unmeasured", "token_staged": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
