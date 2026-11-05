"""Offline evidence reader for Jarvis native capture.json schema 2.

No replay, model calls, semantic judging, writes or percentile qualification.
Run from backend: python3 -m evals.responsive_voice.native_capture_evidence
CAPTURE_JSON [--pcm decoded.pcm] [--annotations annotations.json]

External annotation document (all sample offsets are zero-based mono samples):
  {"schema_version": 1, "capture_run_id": "<capture run UUID>",
   "media_epoch_id": "<capture media UUID>", "pcm_sha256": "<whole PCM SHA256>",
   "annotations": [{"annotation_id": "review-1/useful-0",
     "occurrence_id": "<run hex>/utterance/0", "kind": "useful_onset",
     "sample_offset": 123, "local_output_epoch": 2,
     "provenance": {"method": "human_audio_review", "review_id": "review-1",
                    "reference": "private review note, item 0"}}]}
Other kind: old_answer_tail. Other method: external_content_review. These are
external assertions, not verified semantics. Multiple labels of one kind for an
occurrence are ambiguous; this reader never chooses the earliest or newest.

SDK callback envelopes are monotonic seconds. Samples within an envelope have
NOMINAL timing start + offset/48000, not physical playback/hearing timestamps.
The supported native window is anchor-30 through anchor+150 seconds; this
matches the bounded native runner, not arbitrary recordings. Coherent units
cannot be proven solely from self-reported metadata; consistency is checked.
"""
from __future__ import annotations

import argparse
from array import array
from bisect import bisect_right
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from uuid import UUID


RATE = 48_000
MAX_SCENE_DURATION = 180
# Bounded startup storage is separate from the unchanged scene clock ceiling.
MAX_PCM_BYTES = (30 + MAX_SCENE_DURATION) * RATE * 2
MAX_DECODED_BLOCKS = (30 + MAX_SCENE_DURATION) * 100
MAX_JSON_BYTES = 16 * 1024 * 1024
EPSILON = 1e-6
KINDS = {"useful_onset", "old_answer_tail"}


class EvidenceError(ValueError):
    """A fixed, content-free error code, never a raw artifact excerpt."""


def require(condition, code):
    if not condition:
        raise EvidenceError(code)


def integer(value, code, minimum=0, maximum=None):
    require(type(value) is int and value >= minimum and (maximum is None or value <= maximum), code)
    return value


def number(value, code):
    require(type(value) in (int, float) and math.isfinite(value), code)
    return float(value)


def boolean(value, code):
    require(type(value) is bool, code)
    return value


def string(value, code, limit=1024):
    require(isinstance(value, str) and 0 < len(value) <= limit, code)
    return value


def uuid(value, code):
    try:
        return UUID(string(value, code, 36))
    except (ValueError, AttributeError):
        raise EvidenceError(code) from None


def sequence(value, code, limit):
    require(isinstance(value, list) and len(value) <= limit, code)
    return value


def mapping(value, code):
    require(isinstance(value, dict), code)
    return value


def _clock(capture):
    value = capture.get("clock_anchor")
    if value is None:
        return None, ["clock_anchor_missing"]
    value = mapping(value, "invalid_clock_anchor")
    before = number(value.get("monotonic_before_seconds"), "invalid_monotonic_anchor")
    after = number(value.get("monotonic_after_seconds"), "invalid_monotonic_anchor")
    reference = number(value.get("monotonic_reference_seconds"), "invalid_monotonic_anchor")
    bracket = number(value.get("sampling_bracket_seconds"), "invalid_anchor_bracket")
    utc = number(value.get("utc_unix_seconds"), "invalid_utc_anchor")
    require(0 <= before <= after and abs(reference - (before + after) / 2) <= EPSILON
            and abs(bracket - (after - before)) <= EPSILON, "inconsistent_clock_anchor")
    try:
        stamp = datetime.fromisoformat(value["utc_iso8601"].replace("Z", "+00:00"))
        require(stamp.tzinfo is not None and abs(stamp.timestamp() - utc) <= 0.0011, "inconsistent_utc_anchor")
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        raise EvidenceError("invalid_utc_anchor") from None
    return reference, (["clock_anchor_sampling_uncertain"] if bracket > 1 else [])


def _time(value, reference, code, optional=False):
    if optional and value is None:
        return None
    result = number(value, code)
    require(result >= 0, code)
    if reference is not None:
        require(reference - 30 - EPSILON <= result <= reference + MAX_SCENE_DURATION + 30 + EPSILON,
                "timestamp_outside_supported_monotonic_seconds_domain")
    return result


def _prepare(capture, pcm):
    mapping(capture, "capture_not_object")
    require(type(capture.get("schema_version")) is int and capture["schema_version"] == 2, "unsupported_capture_schema")
    require(capture.get("provenance") == "frontend_replay"
            and capture.get("boundary") == "custom_native_adm_pcm_callback", "unsupported_capture_boundary")
    require(capture.get("physical_playback_measured") is False
            and capture.get("useful_answer_onset") is None, "unsupported_capture_claim")
    run = uuid(capture.get("run_id"), "invalid_run_id")
    media = uuid(capture.get("media_epoch_id"), "invalid_media_epoch")
    require(isinstance(pcm, bytes) and len(pcm) <= MAX_PCM_BYTES and len(pcm) % 2 == 0, "invalid_pcm_length")
    fmt = mapping(capture.get("decoded_pcm"), "missing_pcm_description")
    require(fmt.get("encoding") == "signed_pcm16_little_endian"
            and integer(fmt.get("sample_rate_hz"), "invalid_pcm_rate") == RATE
            and integer(fmt.get("channels"), "invalid_pcm_channels") == 1, "unsupported_pcm_format")
    require(integer(fmt.get("byte_count"), "invalid_pcm_byte_count") == len(pcm)
            and integer(fmt.get("sample_count"), "invalid_pcm_sample_count") == len(pcm) // 2, "pcm_description_length_mismatch")
    require(integer(capture.get("input_threshold_abs_pcm16"), "invalid_input_threshold") == 1
            and integer(capture.get("decoded_output_threshold_abs_pcm16"), "invalid_output_threshold") == 17,
            "unsupported_activity_thresholds")
    reference, blockers = _clock(capture)
    for key in ("truncated", "decoded_mapping_truncated", "input_mapping_truncated", "output_gate_history_truncated"):
        if boolean(capture.get(key), "missing_or_invalid_" + key):
            blockers.append(key)
    failure = capture.get("callback_failure")
    if failure is not None:
        integer(failure, "invalid_callback_failure", -(2**31), 2**31 - 1)
        require(failure != 0, "zero_callback_failure_is_not_an_error")
        blockers.append("sdk_callback_failure")
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    blocks, offsets, next_offset, previous_end = [], [], 0, None
    for raw in sequence(capture.get("decoded_sample_map"), "invalid_decoded_sample_map", MAX_DECODED_BLOCKS):
        raw = mapping(raw, "invalid_decoded_block")
        offset = integer(raw.get("sample_offset"), "invalid_decoded_offset")
        count = integer(raw.get("sample_count"), "invalid_decoded_count", 1, 480)
        require(offset == next_offset and offset + count <= len(samples), "decoded_map_not_contiguous_with_pcm")
        start = _time(raw.get("callback_start_uptime_seconds"), reference, "invalid_decoded_start")
        end = _time(raw.get("callback_end_uptime_seconds"), reference, "invalid_decoded_end", True)
        require(end is None or end >= start, "reversed_decoded_callback")
        require(previous_end is None or start + EPSILON >= previous_end, "nonmonotonic_decoded_callbacks")
        previous_end = end if end is not None else start
        epoch = integer(raw.get("local_output_epoch"), "invalid_decoded_epoch", 1)
        first = next((i for i in range(count) if abs(samples[offset + i]) >= 17), None)
        blocks.append(dict(offset=offset, count=count, start=start, end=end, epoch=epoch, first=first))
        offsets.append(offset)
        next_offset += count
    require(next_offset == len(samples), "unmapped_decoded_pcm")
    gates, opened, closed, epoch, enabled, last_gate_time = [], {}, {}, 0, False, None
    for index, raw in enumerate(sequence(capture.get("output_gates"), "invalid_output_gate_history", 512)):
        raw = mapping(raw, "invalid_output_gate")
        require(raw.get("event_id") == f"{media}/output-gate/{index}", "invalid_output_gate_identity")
        now_enabled = boolean(raw.get("enabled"), "invalid_output_gate_enabled")
        changed = boolean(raw.get("changed"), "invalid_output_gate_changed")
        require(changed == (now_enabled != enabled), "inconsistent_output_gate_transition")
        if now_enabled and changed:
            epoch += 1
        require(integer(raw.get("local_output_epoch"), "invalid_output_gate_epoch") == epoch, "inconsistent_output_gate_epoch")
        time = _time(raw.get("requested_uptime_seconds"), reference, "invalid_output_gate_time", True)
        require(time is None or last_gate_time is None or time + EPSILON >= last_gate_time, "nonmonotonic_output_gates")
        if time is not None:
            last_gate_time = time
        if now_enabled and changed:
            opened[epoch] = time
        elif changed:
            closed[epoch] = time
        gates.append((time, epoch, changed))
        enabled = now_enabled
    for block in blocks:
        reasons = []
        if block["epoch"] not in opened or opened[block["epoch"]] is None:
            reasons.append("local_epoch_history_missing")
        elif block["start"] + EPSILON < opened[block["epoch"]]:
            reasons.append("local_epoch_boundary_ambiguous")
        if block["end"] is None:
            reasons.append("decoded_callback_end_missing")
        elif any(changed and time is not None and block["start"] < time <= block["end"] for time, _, changed in gates):
            reasons.append("local_epoch_boundary_ambiguous")
        if block["epoch"] in closed:
            closing = closed[block["epoch"]]
            if closing is None:
                reasons.append("local_epoch_history_missing")
            elif block["end"] is not None and block["end"] >= closing:
                # Native admission checks the gate/epoch after getPlayoutData
                # returns, under the same lock as gate changes. A callback that
                # has not returned before closure cannot establish old-epoch
                # admission. Its nominal sample tail is NOT this boundary: an
                # admitted block may extend past closing at offset / RATE.
                reasons.append("local_epoch_closed_before_admission_or_order_ambiguous")
        block["issues"] = sorted(set(reasons))
    return dict(capture=capture, run=run, media=media, samples=samples, blocks=blocks, offsets=offsets,
                reference=reference, blockers=sorted(set(blockers)), pcm_sha256=hashlib.sha256(pcm).hexdigest())


def _locate(context, offset):
    integer(offset, "annotation_sample_offset_invalid")
    require(offset < len(context["samples"]), "annotation_sample_offset_out_of_range")
    block = context["blocks"][bisect_right(context["offsets"], offset) - 1]
    return dict(sample_offset=offset, local_output_epoch=block["epoch"],
                nominal_uptime_seconds=block["start"] + (offset - block["offset"]) / RATE,
                sample_abs_pcm16=abs(context["samples"][offset]), mapping_issues=block["issues"])


def _first_energy(context, after=None):
    for block in context["blocks"]:
        if block["first"] is None:
            continue
        relative = block["first"]
        if after is not None and block["start"] + relative / RATE < after:
            relative = max(relative, math.ceil((after - block["start"]) * RATE))
            relative = next((i for i in range(relative, block["count"])
                             if abs(context["samples"][block["offset"] + i]) >= 17), None)
        if relative is not None and relative < block["count"]:
            return _locate(context, block["offset"] + relative)
    return None


def _input(context, item, ordinal):
    mapping(item, "invalid_utterance")
    require(integer(item.get("ordinal"), "invalid_ordinal") == ordinal, "noncontiguous_utterance_ordinals")
    identity = f"{context['run'].hex}/utterance/{ordinal}"
    require(item.get("occurrence_id") == identity and item.get("trigger_id") == identity + "/trigger", "utterance_identity_mismatch")
    trigger = mapping(item.get("trigger"), "invalid_trigger")
    require(trigger.get("kind") in {"at_time", "output_start", "playback_drained", "response_output_start",
                                    "after_task_claim", "after_input_complete", "after_manual_response",
                                    "after_provider_input_finalized"}, "invalid_trigger_kind")
    if trigger["kind"] == "after_manual_response":
        from evals.responsive_voice.native_memory_readiness_control import memory_gate_rows
        try:
            memory_gate_rows({"utterances": [{}] * ordinal + [{"trigger": trigger}]})
        except (ValueError, TypeError, KeyError):
            raise EvidenceError("invalid_manual_response_trigger") from None
    count = integer(item.get("sample_count"), "invalid_input_count", 1, 60 * RATE)
    attempted = integer(item.get("attempted_clip_samples"), "invalid_attempted_count", 0, count)
    accepted = integer(item.get("sdk_accepted_clip_samples"), "invalid_accepted_count", 0, attempted)
    status = item.get("status")
    require(status in {"scheduled", "enqueued", "sdk_input_in_progress", "sdk_input_complete", "failed", "cancelled", "not_started"}, "invalid_input_status")
    zero = boolean(item.get("all_samples_zero"), "invalid_silent_input_flag")
    threshold = integer(item.get("activity_threshold_abs_pcm16"), "invalid_occurrence_threshold", 1, 32767)
    require(item.get("activity_comparison") == ">=", "invalid_activity_comparison")
    require(item.get("activity_threshold_origin") in {"native_default", "speech_manifest"}, "invalid_activity_origin")
    if item["activity_threshold_origin"] == "native_default":
        require(threshold == 1, "invalid_default_activity_threshold")
    times = {}
    for key in ("trigger_armed_uptime_seconds", "triggered_uptime_seconds", "enqueue_requested_uptime_seconds",
                "first_sdk_callback_start_uptime_seconds", "last_sdk_callback_end_uptime_seconds",
                "first_successful_activity_uptime_seconds", "last_successful_activity_uptime_seconds", "stop_requested_uptime_seconds"):
        times[key] = _time(item.get(key), context["reference"], "invalid_utterance_timestamp", True)
    callbacks, offset, successes, prior_end = [], 0, 0, None
    for raw in sequence(item.get("callbacks"), "invalid_input_callbacks", 6020):
        raw = mapping(raw, "invalid_input_callback")
        start_offset = integer(raw.get("source_sample_offset"), "invalid_input_offset")
        amount = integer(raw.get("clip_sample_count"), "invalid_input_block_count", 1, 480)
        require(integer(raw.get("transport_frame_count"), "invalid_transport_count") == 480,
                "unsupported_input_transport_block")
        require(start_offset == offset and offset + amount <= count, "input_map_not_contiguous")
        start = _time(raw.get("callback_start_uptime_seconds"), context["reference"], "invalid_input_start")
        end = _time(raw.get("callback_end_uptime_seconds"), context["reference"], "invalid_input_end")
        require(end >= start and (prior_end is None or start + EPSILON >= prior_end), "nonmonotonic_input_callbacks")
        code = integer(raw.get("sdk_status"), "invalid_input_sdk_status", -(2**31), 2**31 - 1)
        callbacks.append(dict(offset=offset, count=amount, start=start, end=end, status=code))
        offset += amount
        successes += amount if code == 0 else 0
        prior_end = end
    incomplete_map = context["capture"]["input_mapping_truncated"]
    if not incomplete_map:
        require(offset == attempted and successes == accepted, "input_callback_counters_mismatch")
        require((callbacks and times["first_sdk_callback_start_uptime_seconds"] == callbacks[0]["start"]
                 and times["last_sdk_callback_end_uptime_seconds"] == callbacks[-1]["end"])
                or (not callbacks and times["first_sdk_callback_start_uptime_seconds"] is None
                    and times["last_sdk_callback_end_uptime_seconds"] is None), "input_callback_envelope_mismatch")
    else:
        require(offset <= attempted and successes <= accepted, "invalid_truncated_input_counters")
    first, last = times["first_successful_activity_uptime_seconds"], times["last_successful_activity_uptime_seconds"]
    require((first is None) == (last is None), "partial_activity_timestamps")
    if first is not None:
        require(not zero and first <= last, "inconsistent_activity_timestamps")
        if not incomplete_map:
            for activity in (first, last):
                require(any(block["status"] == 0 and block["start"] - EPSILON <= activity
                            <= block["start"] + (block["count"] - 1) / RATE + EPSILON for block in callbacks),
                        "activity_outside_sdk_accepted_samples")
    source_first, source_end = item.get("source_first_active_sample"), item.get("source_last_active_sample_exclusive")
    require((source_first is None) == (source_end is None), "partial_source_activity_indices")
    if source_first is not None:
        integer(source_first, "invalid_source_activity_start", 0, count - 1)
        integer(source_end, "invalid_source_activity_end", source_first + 1, count)
    if status == "sdk_input_complete" and not incomplete_map:
        require(accepted == count and attempted == count and all(block["status"] == 0 for block in callbacks),
                "completed_input_not_supported_by_callbacks")
        if item["activity_threshold_origin"] == "speech_manifest":
            require((source_first is None) == (first is None), "source_activity_mismatch")
            if source_first is not None:
                for sample, value in ((source_first, first), (source_end - 1, last)):
                    block = next(block for block in callbacks if block["offset"] <= sample < block["offset"] + block["count"])
                    require(abs(value - (block["start"] + (sample - block["offset"]) / RATE)) <= EPSILON,
                            "activity_units_or_sample_mapping_mismatch")
    if incomplete_map:
        verdict = "input_mapping_unverifiable"
    elif status != "sdk_input_complete":
        verdict = "not_started" if status in {"scheduled", "not_started"} else "incomplete_or_failed_input"
    elif first is None:
        verdict = "silent_input" if zero else "no_input_activity_at_threshold"
    else:
        verdict = "sdk_completed_active_input"
    return dict(occurrence_id=identity, ordinal=ordinal, clip_id=item.get("clip_id"), trigger_id=item["trigger_id"],
                recorded_status=status, input_verdict=verdict, input_activity_threshold_abs_pcm16=threshold,
                speech_onset_uptime_seconds=first if verdict == "sdk_completed_active_input" else None,
                input_end_exclusive_uptime_seconds=last + 1 / RATE if verdict == "sdk_completed_active_input" else None)


def _response_identifier(value, code, limit=160):
    value = string(value, code, limit)
    require(value.isascii() and re.fullmatch(r"[A-Za-z0-9_.:/-]+", value) is not None, code)
    return value


def _response_identity(value):
    value = mapping(value, "invalid_response_identity")
    require(set(value) == {"connection_id", "response_id", "item_id", "local_output_epoch"},
            "invalid_response_identity_fields")
    return dict(connection_id=str(uuid(value.get("connection_id"), "invalid_response_connection")),
                response_id=_response_identifier(value.get("response_id"), "invalid_response_id"),
                item_id=_response_identifier(value.get("item_id"), "invalid_response_item_id"),
                local_output_epoch=integer(value.get("local_output_epoch"), "invalid_response_epoch", 1))


def _response_key(identity):
    return tuple(identity[key] for key in ("connection_id", "response_id", "item_id", "local_output_epoch"))


def _response_point(context, raw, identity, block_identities):
    raw = mapping(raw, "invalid_response_sample_point")
    location = _locate(context, raw.get("sample_offset"))
    require(integer(raw.get("local_output_epoch"), "invalid_response_point_epoch", 1)
            == location["local_output_epoch"] == identity["local_output_epoch"], "response_sample_epoch_mismatch")
    nominal = _time(raw.get("nominal_uptime_seconds"), context["reference"], "invalid_response_point_time")
    require(abs(nominal - location["nominal_uptime_seconds"]) <= EPSILON, "response_sample_time_mismatch")
    require(location["sample_abs_pcm16"] >= 17, "response_sample_not_active_pcm")
    index = bisect_right(context["offsets"], location["sample_offset"]) - 1
    assigned = block_identities[index]
    require(assigned is None or assigned == identity, "response_sample_identity_mismatch")
    issues = list(location["mapping_issues"])
    if assigned is None:
        issues.append("response_sample_identity_missing")
    return location, context["blocks"][index], issues


def _response_evidence(context, raw_inputs, occurrences):
    """Validate local authorized-gate bindings, never RTP identity or actual cuts.

    The optional extension is independent of external semantic annotations. SDK
    completion precedes trigger eligibility; it establishes no utterance-to-
    response causality. Missing proof is not a measured zero.
    """
    capture = context["capture"]
    result = dict(schema_version=1, verdict="not_recorded", correlation_basis=None,
        request_causality="unknown", semantic_causality="unknown", actual_cuts=None,
        provider_response_attribution="unavailable", hearing_measured=False, usefulness_measured=False,
        control_correlated_stop_count=None, validated_control_correlated_stops=[],
        trigger_diagnostics=[], stop_diagnostics=[], blockers=[])
    for raw, occurrence in zip(raw_inputs, occurrences):
        occurrence["response_trigger_verdict"] = (
            "missing" if raw["trigger"]["kind"] == "response_output_start" else "not_applicable")
        occurrence["response_control_stop"] = None
    if "response_trigger_schema_version" not in capture:
        require(not any(raw.get("response_trigger") is not None for raw in raw_inputs),
                "response_extension_version_missing")
        return result
    require(type(capture["response_trigger_schema_version"]) is int
            and capture["response_trigger_schema_version"] == 1, "unsupported_response_trigger_schema")
    require(capture.get("response_correlation_basis") == "authorized_gate_interval"
            and capture.get("response_semantic_causality") == "unknown", "unsupported_response_correlation_claim")
    result["correlation_basis"] = "authorized_gate_interval"
    blockers = list(context["blockers"])
    for name in ("response_evidence_truncated", "response_evidence_invalid"):
        if boolean(capture.get(name), "invalid_" + name):
            blockers.append(name)
    result["uncorrelated_decoded_blocks"] = integer(capture.get("uncorrelated_decoded_blocks"),
        "invalid_uncorrelated_decoded_blocks", 0, 1_000_000)
    if _first_energy(context) is None:
        blockers.append("no_retained_active_decoded_pcm")

    block_identities, groups, epoch_owners = [], {}, {}
    for raw, block in zip(capture["decoded_sample_map"], context["blocks"]):
        identity = _response_identity(raw["response_identity"]) if raw.get("response_identity") is not None else None
        block_identities.append(identity)
        if identity is None:
            if block["first"] is not None:
                blockers.append("uncorrelated_retained_active_pcm")
            continue
        require(identity["local_output_epoch"] == block["epoch"], "response_block_epoch_mismatch")
        require(epoch_owners.setdefault(block["epoch"], identity) == identity, "mixed_response_identity_within_epoch")
        last = next((i for i in reversed(range(block["count"]))
                     if abs(context["samples"][block["offset"] + i]) >= 17), None)
        groups.setdefault(_response_key(identity), []).append((block, last))
    require(result["uncorrelated_decoded_blocks"] >= sum(value is None for value in block_identities),
            "uncorrelated_decoded_counter_inconsistent")
    gates, opened, closed_gates = {}, {}, {}
    for raw in capture["output_gates"]:
        identity = _response_identity(raw["response_identity"]) if raw.get("response_identity") is not None else None
        require(identity is None or identity["local_output_epoch"] == raw["local_output_epoch"],
                "response_gate_epoch_mismatch")
        event = dict(raw, response_identity=identity)
        link = event.get("local_interruption_event_id")
        if link is not None:
            _response_identifier(link, "invalid_gate_interruption_reference", 256)
        gates[event["event_id"]] = event
        if event["enabled"] and event["changed"]:
            opened[event["local_output_epoch"]] = event
        elif event["changed"]:
            closed_gates[event["local_output_epoch"]] = event
    interruptions, previous_interruption_time = {}, None
    for raw in sequence(capture.get("local_interruptions"), "invalid_local_interruptions", 512):
        raw = mapping(raw, "invalid_local_interruption")
        event_id = _response_identifier(raw.get("event_id"), "invalid_local_interruption_id", 256)
        require(event_id not in interruptions, "duplicate_local_interruption")
        require(raw.get("cause") in {"client_interrupt_method", "route_output_fence_close"},
                "unsupported_local_interruption_cause")
        stamp = _time(raw.get("requested_uptime_seconds"), context["reference"], "invalid_local_interruption_time")
        require(previous_interruption_time is None or stamp + EPSILON >= previous_interruption_time,
                "nonmonotonic_local_interruptions")
        previous_interruption_time = stamp
        interruptions[event_id] = dict(event_id=event_id, identity=_response_identity(raw.get("identity")),
            cause=raw["cause"], requested_uptime_seconds=stamp)

    bindings, consumed_epoch = {}, 0
    for raw, occurrence in zip(raw_inputs, occurrences):
        config, proof = raw["trigger"], raw.get("response_trigger")
        if config["kind"] != "response_output_start":
            require(proof is None, "response_proof_on_unrelated_trigger")
            continue
        ordinal = occurrence["ordinal"]
        previous = integer(config.get("after_input_ordinal"), "invalid_response_preceding_ordinal", 0, ordinal - 1)
        delay = number(config.get("delay_seconds"), "invalid_response_trigger_delay")
        timeout = number(config.get("timeout_seconds"), "invalid_response_trigger_timeout")
        max_age = number(config.get("max_pcm_age_seconds"), "invalid_response_pcm_age_limit")
        require(0 <= delay <= 120 and 0 < timeout <= 120 and 0 < max_age <= 120, "invalid_response_trigger_bounds")
        if proof is None:
            result["trigger_diagnostics"].append(dict(occurrence_id=occurrence["occurrence_id"],
                verdict="missing", reasons=["response_trigger_missing"]))
            continue
        proof = mapping(proof, "invalid_response_trigger")
        identity = _response_identity(proof.get("identity"))
        require(proof.get("semantic_causality") == "unknown", "unsupported_trigger_semantic_claim")
        require(integer(proof.get("after_input_ordinal"), "invalid_response_proof_ordinal") == previous,
                "response_preceding_ordinal_mismatch")
        require(identity["local_output_epoch"] > consumed_epoch, "response_trigger_epoch_reused_or_reversed")
        consumed_epoch = identity["local_output_epoch"]
        issues = list(blockers)
        if config.get("require_task_claim", False) and occurrence.get("task_trigger_verdict") != "validated_control_metadata":
            issues.append("task_claim_at_enqueue_unqualified")
        times = {key: _time(proof.get(key), context["reference"], "invalid_response_trigger_time",
                            optional=key == "enqueue_revalidated_uptime_seconds") for key in (
            "input_complete_uptime_seconds", "response_admitted_uptime_seconds",
            "latest_decoded_callback_end_uptime_seconds", "selected_uptime_seconds", "enqueue_revalidated_uptime_seconds")}
        selected, revalidated = times["selected_uptime_seconds"], times["enqueue_revalidated_uptime_seconds"]
        completed, admitted = times["input_complete_uptime_seconds"], times["response_admitted_uptime_seconds"]
        latest_end = times["latest_decoded_callback_end_uptime_seconds"]
        require(completed <= selected + EPSILON and admitted <= selected + EPSILON, "response_trigger_time_order")
        if occurrences[previous]["input_verdict"] != "sdk_completed_active_input":
            issues.append("preceding_input_not_completed_active")
        prior_end = raw_inputs[previous].get("last_sdk_callback_end_uptime_seconds")
        if prior_end is None:
            issues.append("preceding_sdk_completion_missing")
        else:
            require(abs(completed - prior_end) <= EPSILON, "response_sdk_completion_mismatch")
        first, first_block, first_issues = _response_point(context, proof.get("first_retained_active_sample"), identity, block_identities)
        latest, latest_block, latest_issues = _response_point(context, proof.get("latest_retained_active_sample"), identity, block_identities)
        issues += first_issues + latest_issues
        require(first["sample_offset"] <= latest["sample_offset"], "response_active_sample_order")
        if not boolean(proof.get("enqueue_revalidated"), "invalid_response_enqueue_revalidated") or revalidated is None:
            issues.append("response_enqueue_not_revalidated")
        if proof.get("failure_code") is not None:
            _response_identifier(proof["failure_code"], "invalid_response_trigger_failure", 128)
            issues.append("response_trigger_failure_recorded")
        if config.get("response_id") is not None:
            if _response_identifier(config["response_id"], "invalid_exact_response_id") != identity["response_id"]:
                issues.append("exact_response_filter_mismatch")
        if config.get("connection_id") is not None:
            if str(uuid(config["connection_id"], "invalid_exact_connection_id")) != identity["connection_id"]:
                issues.append("exact_connection_filter_mismatch")
        opening = opened.get(identity["local_output_epoch"])
        if opening is None or opening["response_identity"] is None:
            issues.append("response_gate_identity_missing")
        else:
            require(opening["response_identity"] == identity, "response_open_gate_identity_mismatch")
        closing = closed_gates.get(identity["local_output_epoch"])
        opening_time = opening.get("requested_uptime_seconds") if opening is not None else None
        closing_time = closing.get("requested_uptime_seconds") if closing is not None else None
        if opening is not None and opening_time is None:
            issues.append("response_gate_open_time_missing")
        if closing is not None and closing_time is None:
            issues.append("response_gate_close_time_missing")
        for boundary, stamp in (("selection", selected), ("enqueue", revalidated)):
            if stamp is None:
                continue  # Missing enqueue proof is already unmeasured above.
            # A retained, recently active block is not proof that its gate is
            # still open. Equal/indistinguishable closure times are ambiguous;
            # this does not constrain a pre-close block's nominal sample tail.
            if ((opening_time is not None and stamp + EPSILON < opening_time)
                    or (closing_time is not None and stamp + EPSILON >= closing_time)):
                issues.append("response_gate_not_open_at_" + boundary)
        age = number(proof.get("pcm_age_seconds"), "invalid_response_pcm_age")
        require(age >= 0, "invalid_response_pcm_age")
        eligible_times = [selected] + ([revalidated] if revalidated is not None else [])
        require(any(abs(age - (time - latest_end)) <= EPSILON for time in eligible_times), "response_pcm_age_mismatch")
        group = groups.get(_response_key(identity), [])
        snapshot_found = False
        for boundary in eligible_times:
            eligible = [(block, last) for block, last in group if block["end"] is not None and block["end"] <= boundary + EPSILON]
            active = [(block, last) for block, last in eligible if last is not None]
            if (active and abs(max(block["end"] for block, _ in eligible) - latest_end) <= EPSILON
                    and first["sample_offset"] == active[0][0]["offset"] + active[0][0]["first"]
                    and latest["sample_offset"] == active[-1][0]["offset"] + active[-1][1]):
                snapshot_found = True
        if not snapshot_found:
            issues.append("response_active_snapshot_unconfirmed")
        if revalidated is not None:
            require(selected <= revalidated + EPSILON and latest_end <= revalidated + EPSILON,
                    "response_enqueue_time_order")
            if revalidated > completed + timeout + EPSILON or selected > completed + timeout + EPSILON:
                issues.append("response_trigger_deadline_exceeded")
            enqueue = raw.get("enqueue_requested_uptime_seconds")
            if enqueue is None or enqueue + EPSILON < revalidated or enqueue > completed + timeout + EPSILON:
                issues.append("response_enqueue_boundary_unconfirmed")
            if revalidated + EPSILON < completed + delay:
                issues.append("response_trigger_delay_not_elapsed")
            active_end = latest_block["end"]
            if (active_end is None or revalidated - active_end > max_age + EPSILON
                    or revalidated - latest_end > max_age + EPSILON):
                issues.append("response_pcm_stale_at_enqueue")
        verdict = "unmeasured" if issues else "validated_control_binding"
        occurrence["response_trigger_verdict"] = verdict
        bindings[occurrence["occurrence_id"]] = dict(identity=identity, proof=proof, times=times, issues=issues)
        result["trigger_diagnostics"].append(dict(occurrence_id=occurrence["occurrence_id"],
            verdict=verdict, identity=identity, reasons=sorted(set(issues))))

    by_occurrence = {item["occurrence_id"]: item for item in occurrences}
    used_occurrences, used_interruptions, used_gates = set(), set(), set()
    for raw in sequence(capture.get("response_stop_measurements"), "invalid_response_stop_measurements", 20):
        raw = mapping(raw, "invalid_response_stop_measurement")
        occurrence_id = raw.get("occurrence_id")
        require(isinstance(occurrence_id, str) and occurrence_id in by_occurrence, "response_stop_occurrence_unknown")
        require(occurrence_id not in used_occurrences, "duplicate_response_stop_occurrence")
        used_occurrences.add(occurrence_id)
        identity = _response_identity(raw.get("identity"))
        interrupt_id = _response_identifier(raw.get("interruption_event_id"), "invalid_stop_interruption_id", 256)
        gate_id = _response_identifier(raw.get("output_gate_event_id"), "invalid_stop_gate_id", 256)
        require(interrupt_id not in used_interruptions and gate_id not in used_gates, "response_stop_event_reused")
        used_interruptions.add(interrupt_id)
        used_gates.add(gate_id)
        occurrence, binding = by_occurrence[occurrence_id], bindings.get(occurrence_id)
        issues = list(blockers)
        if occurrence["response_trigger_verdict"] != "validated_control_binding":
            issues.append("response_trigger_not_validated")
        if occurrence["input_verdict"] != "sdk_completed_active_input":
            issues.append("interruption_input_not_completed_active")
        if binding is not None:
            require(identity == binding["identity"], "response_stop_trigger_identity_mismatch")
        interruption, gate = interruptions.get(interrupt_id), gates.get(gate_id)
        if interruption is None:
            issues.append("local_interruption_missing")
        else:
            require(interruption["identity"] == identity, "response_stop_interruption_identity_mismatch")
        if gate is None or gate.get("response_identity") is None:
            issues.append("response_close_gate_identity_missing")
        else:
            require(gate["response_identity"] == identity, "response_stop_gate_identity_mismatch")
            if gate["enabled"] or not gate["changed"] or gate.get("local_interruption_event_id") != interrupt_id:
                issues.append("explicit_interruption_gate_close_missing")
        onset = _time(raw.get("input_first_successful_activity_uptime_seconds"), context["reference"], "invalid_stop_input_time")
        requested = _time(raw.get("local_interruption_requested_uptime_seconds"), context["reference"], "invalid_stop_request_time")
        closed = _time(raw.get("local_gate_closed_uptime_seconds"), context["reference"], "invalid_stop_gate_time")
        require(requested <= closed + EPSILON, "response_stop_time_order")
        if occurrence["speech_onset_uptime_seconds"] is not None:
            require(abs(onset - occurrence["speech_onset_uptime_seconds"]) <= EPSILON, "response_stop_input_time_mismatch")
        if interruption is not None:
            require(abs(requested - interruption["requested_uptime_seconds"]) <= EPSILON, "response_stop_request_time_mismatch")
        if gate is not None and gate.get("requested_uptime_seconds") is not None:
            require(abs(closed - gate["requested_uptime_seconds"]) <= EPSILON, "response_stop_gate_time_mismatch")
        else:
            issues.append("response_close_time_missing")
        if binding is not None and binding["times"]["enqueue_revalidated_uptime_seconds"] is not None:
            require(requested + EPSILON >= binding["times"]["enqueue_revalidated_uptime_seconds"],
                    "response_interruption_before_selected_enqueue")
        for key, expected in (("input_to_local_interruption_seconds", requested - onset),
                              ("input_to_local_gate_close_seconds", closed - onset)):
            require(abs(number(raw.get(key), "invalid_response_stop_interval") - expected) <= EPSILON,
                    "response_stop_interval_mismatch")
        group = groups.get(_response_key(identity), [])
        active = [(block, last) for block, last in group if last is not None]
        for block, _ in group:
            issues += block["issues"]
        if not active:
            issues.append("response_stop_active_pcm_missing")
        tail = raw.get("last_retained_active_minus_input_seconds")
        if tail is not None:
            number(tail, "invalid_response_stop_tail")
            if active:
                block, last = active[-1]
                require(abs(tail - (block["start"] + last / RATE - onset)) <= EPSILON,
                        "response_stop_tail_mismatch")
        if issues:
            result["stop_diagnostics"].append(dict(occurrence_id=occurrence_id, verdict="unmeasured",
                                                  reasons=sorted(set(issues))))
        else:
            validated = dict(raw, identity=identity, verdict="validated_control_correlated_stop",
                cause=interruption["cause"], correlation_basis="authorized_gate_interval",
                semantic_causality="unknown", provider_response_attribution="unavailable",
                hearing_measured=False, usefulness_measured=False)
            occurrence["response_control_stop"] = validated
            result["validated_control_correlated_stops"].append(validated)
    result["blockers"] = sorted(set(blockers))
    unqualified = bool(blockers or any(row["verdict"] != "validated_control_binding" for row in result["trigger_diagnostics"])
                       or result["stop_diagnostics"])
    result["verdict"] = "unqualified_evidence" if unqualified else "validated_control_evidence"
    if not unqualified:
        result["control_correlated_stop_count"] = len(result["validated_control_correlated_stops"])
    return result


def _annotations(context, document, identities):
    result = {identity: {kind: [] for kind in KINDS} for identity in identities}
    if document is None:
        return result
    mapping(document, "annotations_not_object")
    require(type(document.get("schema_version")) is int and document["schema_version"] == 1, "unsupported_annotation_schema")
    require(uuid(document.get("capture_run_id"), "invalid_annotation_run") == context["run"]
            and uuid(document.get("media_epoch_id"), "invalid_annotation_media") == context["media"]
            and document.get("pcm_sha256") == context["pcm_sha256"], "annotations_capture_identity_mismatch")
    seen = set()
    for entry in sequence(document.get("annotations"), "invalid_annotations", 80):
        entry = mapping(entry, "invalid_annotation")
        identity, kind = entry.get("occurrence_id"), entry.get("kind")
        require(identity in result and kind in KINDS, "annotation_occurrence_or_kind_unknown")
        annotation = {"annotation_id": entry.get("annotation_id")}
        try:
            aid = string(entry.get("annotation_id"), "invalid_annotation_id", 256)
            require(aid not in seen, "duplicate_annotation_id")
            seen.add(aid)
            provenance = mapping(entry.get("provenance"), "annotation_provenance_missing")
            require(provenance.get("method") in {"human_audio_review", "external_content_review"}, "annotation_provenance_method_unsupported")
            for field in ("review_id", "reference"):
                string(provenance.get(field), "annotation_provenance_incomplete")
            location = _locate(context, entry.get("sample_offset"))
            epoch = integer(entry.get("local_output_epoch"), "annotation_epoch_missing", 1)
            require(epoch == location["local_output_epoch"], "annotation_sample_epoch_mismatch")
            annotation.update(location=location, provenance={key: provenance[key] for key in ("method", "review_id", "reference")})
        except EvidenceError as error:
            annotation["error_code"] = str(error)
        result[identity][kind].append(annotation)
    return result


def _metric(context, occurrence, kind, entries):
    if not entries:
        return {"verdict": "not_annotated", "delta_seconds": None}
    if len(entries) != 1:
        return {"verdict": "ambiguous_annotations", "delta_seconds": None,
                "annotation_ids": [entry["annotation_id"] for entry in entries]}
    entry = entries[0]
    if "error_code" in entry:
        return dict(verdict="invalid_annotation", delta_seconds=None, **entry)
    blocked = list(context["blockers"]) + entry["location"]["mapping_issues"]
    if occurrence["input_verdict"] != "sdk_completed_active_input":
        blocked.append(occurrence["input_verdict"])
    if blocked:
        return dict(verdict="timing_unqualified", delta_seconds=None, reasons=sorted(set(blocked)), **entry)
    base = occurrence["speech_onset_uptime_seconds"] if kind == "old_answer_tail" else occurrence["input_end_exclusive_uptime_seconds"]
    delta = entry["location"]["nominal_uptime_seconds"] - base
    return dict(verdict="externally_annotated_nominal_timing", delta_seconds=delta,
                negative_interval=delta < 0, semantic_label_verified=False, **entry)


def _task_json(data, limit, code):
    require(isinstance(data, bytes) and 0 < len(data) <= limit, code)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_task_control_key")
            result[key] = value
        return result
    try:
        result = json.loads(data, object_pairs_hook=pairs,
                            parse_constant=lambda _: (_ for _ in ()).throw(EvidenceError("nonfinite_task_control")))
    except (UnicodeError, ValueError) as error:
        if isinstance(error, EvidenceError):
            raise
        raise EvidenceError(code) from None
    return mapping(result, code)


def _task_trigger_semantics(value, ordinal):
    """Frozen task-mode trigger meaning, without rewriting retained evidence.

    Swift may serialize its two optional response filters as null while the
    manifest omits them. No other defaults, extra fields or values are erased.
    """
    trigger = mapping(value, "invalid_task_trigger")
    kind = trigger.get("kind")
    fields = {
        "at_time": {"kind", "at_seconds"},
        "after_task_claim": {"kind", "after_input_ordinal", "timeout_seconds"},
        "after_input_complete": {"kind", "after_input_ordinal", "delay_seconds", "timeout_seconds"},
        "after_provider_input_finalized": {"kind", "after_input_ordinal", "timeout_seconds"},
        "response_output_start": {"kind", "after_input_ordinal", "delay_seconds", "timeout_seconds", "max_pcm_age_seconds"},
    }
    require(isinstance(kind, str) and kind in fields, "invalid_task_trigger_kind")
    optional = {"connection_id", "response_id", "require_task_claim"} if kind == "response_output_start" else set()
    require(fields[kind] <= set(trigger) and set(trigger) <= fields[kind] | optional, "invalid_task_trigger_fields")
    result = dict(trigger)
    if kind == "at_time":
        at = number(trigger["at_seconds"], "invalid_task_trigger_at_time")
        require(0 <= at <= MAX_SCENE_DURATION, "invalid_task_trigger_bounds")
        return result
    integer(trigger["after_input_ordinal"], "invalid_task_preceding_ordinal", 0, ordinal - 1)
    if kind == "after_provider_input_finalized":
        require(trigger["after_input_ordinal"] == ordinal - 1, "provider_input_predecessor_not_adjacent")
    timeout = number(trigger["timeout_seconds"], "invalid_task_trigger_timeout")
    require(0 < timeout <= 120, "invalid_task_trigger_bounds")
    if "delay_seconds" in trigger:
        delay = number(trigger["delay_seconds"], "invalid_task_trigger_delay")
        require(0 <= delay <= timeout, "invalid_task_trigger_bounds")
    if kind == "response_output_start":
        age = number(trigger["max_pcm_age_seconds"], "invalid_response_pcm_age_limit")
        require(0 < age <= 120, "invalid_response_pcm_age_limit")
        if "require_task_claim" in trigger:
            boolean(trigger["require_task_claim"], "invalid_require_task_claim")
        for key in ("connection_id", "response_id"):
            if trigger.get(key) is None:
                result.pop(key, None)
            elif key == "connection_id":
                uuid(trigger[key], "invalid_exact_connection_id")
            else:
                _response_identifier(trigger[key], "invalid_exact_response_id")
    return result


def _task_controls(context, inputs, occurrences, documents):
    """Validate frozen control artifacts, not invented capture.json fields.

    documents is an offline reader API, not a native wire envelope. Its bytes
    must be retained originals: manifest_bytes, scope_bytes,
    session_binding_bytes, stimulus_completion_bytes, barrier_receipts[].
    Absence/partial startup remains unmeasured. These self-reported artifacts
    do not replace the postrun canonical export or prove a semantic cut.
    """
    result = dict(verdict="not_recorded", blockers=[], validated_claim_inputs=0,
                  task_continuity="unknown", actual_cuts=None, hearing_measured=False,
                  canonical_postrun_required=True, evidence_basis="retained_control_artifacts_not_independent_canonical_audit")
    dependent = []
    for ordinal, (raw, row) in enumerate(zip(inputs, occurrences)):
        trigger = raw["trigger"]
        kind = trigger["kind"]
        require("require_task_claim" not in trigger or kind == "response_output_start",
                "task_claim_flag_on_unrelated_trigger")
        flag = boolean(trigger.get("require_task_claim", False), "invalid_require_task_claim")
        needs_claim = kind == "after_task_claim" or flag
        row.update(task_trigger_verdict="missing" if needs_claim else "not_applicable",
                   task_claim_sequence_at_enqueue=None)
        if needs_claim:
            dependent.append(ordinal)
        if kind not in {"after_task_claim", "after_input_complete"}:
            continue
        allowed = {"kind", "after_input_ordinal", "timeout_seconds"}
        if kind == "after_input_complete":
            allowed.add("delay_seconds")
        require(set(trigger) == allowed, "invalid_task_trigger_fields")
        previous = integer(trigger.get("after_input_ordinal"), "invalid_task_preceding_ordinal", 0, ordinal - 1)
        timeout = number(trigger.get("timeout_seconds"), "invalid_task_trigger_timeout")
        delay = number(trigger.get("delay_seconds", 0), "invalid_task_trigger_delay")
        require(0 < timeout <= 120 and 0 <= delay <= timeout, "invalid_task_trigger_bounds")
        end = inputs[previous].get("last_sdk_callback_end_uptime_seconds")
        enqueue = raw.get("enqueue_requested_uptime_seconds")
        ready = occurrences[previous]["input_verdict"] == "sdk_completed_active_input" and end is not None
        if ready and enqueue is not None:
            require(end + delay <= enqueue + EPSILON and enqueue <= end + timeout + EPSILON,
                    "task_trigger_outside_sdk_completion_window")
        if kind == "after_input_complete":
            row["task_trigger_verdict"] = ("validated_input_completion" if ready and enqueue is not None
                                          and row["input_verdict"] == "sdk_completed_active_input" else "unmeasured")
        elif not ready or enqueue is None:
            row["task_trigger_verdict"] = "unmeasured"
    if documents is None:
        if dependent:
            result.update(verdict="unmeasured", blockers=["task_control_artifacts_missing"])
        return result
    documents = mapping(documents, "invalid_task_documents")
    require(set(documents) <= {"manifest_bytes", "scope_bytes", "session_binding_bytes",
                              "stimulus_completion_bytes", "barrier_receipts"}, "unknown_task_document")
    required = ("manifest_bytes", "scope_bytes", "session_binding_bytes")
    if any(documents.get(key) is None for key in required):
        result.update(verdict="unmeasured", blockers=["task_scope_or_session_binding_missing"])
        return result
    manifest = _task_json(documents["manifest_bytes"], 262144, "invalid_task_manifest")
    scope = _task_json(documents["scope_bytes"], 8192, "invalid_task_scope")
    binding = _task_json(documents["session_binding_bytes"], 8192, "invalid_task_session_binding")
    scope_keys = {"format", "run_id", "barrier_id", "owner_id", "database_schema", "worker_id",
                  "runtime_host_id", "model", "host_expires_at"}
    require(set(scope) == scope_keys and scope["format"] == "jarvis.native_task_scope.v1", "invalid_task_scope_fields")
    require(type(manifest.get("version")) is int and manifest["version"] == 1
            and type(manifest.get("task_barrier", {}).get("version")) is int and manifest.get("task_barrier") ==
            dict(version=1, scope_file="task-scope.json", request_input_ordinal=0), "invalid_task_manifest_barrier")
    require(uuid(manifest.get("run_id"), "invalid_task_manifest_run") == context["run"]
            and uuid(scope.get("run_id"), "invalid_task_scope_run") == context["run"]
            and manifest.get("owner_id") == scope["owner_id"]
            and manifest.get("voice_route") == "realtime", "task_manifest_scope_mismatch")
    plans = sequence(manifest.get("utterances"), "invalid_task_manifest_inputs", 20)
    require(len(plans) == len(inputs) and len(inputs) >= 2, "task_manifest_input_count_mismatch")
    require(any(item["trigger"]["kind"] == "after_task_claim" for item in inputs), "task_claim_trigger_missing")
    for ordinal, (plan, raw) in enumerate(zip(plans, inputs)):
        require(isinstance(plan, dict) and plan.get("clip_id") == raw["clip_id"], "task_manifest_input_mismatch")
        require(_task_trigger_semantics(plan.get("trigger"), ordinal) ==
                _task_trigger_semantics(raw["trigger"], ordinal), "task_manifest_input_mismatch")
    common = {"run_id", "barrier_id", "owner_id", "session_id", "manifest_file_sha256", "scope_file_sha256", "recorded_at"}
    from datetime import datetime, timezone
    def utc(value):
        try:
            parsed = datetime.fromisoformat(string(value, "invalid_task_utc").replace("Z", "+00:00"))
            require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0, "task_time_not_utc")
            return parsed
        except (ValueError, OverflowError):
            raise EvidenceError("invalid_task_utc") from None
    def validate_control(doc, format_name, extra):
        require(set(doc) == common | {"format"} | extra and doc["format"] == format_name,
                "invalid_task_control_fields")
        for key in ("run_id", "barrier_id"):
            require(uuid(doc[key], "invalid_task_control_uuid") == uuid(scope[key], "invalid_task_scope_uuid"),
                    "task_control_scope_mismatch")
        require(doc["owner_id"] == scope["owner_id"], "task_control_owner_mismatch")
        uuid(doc["session_id"], "invalid_task_control_session")
        require(doc["manifest_file_sha256"] == hashlib.sha256(documents["manifest_bytes"]).hexdigest()
                and doc["scope_file_sha256"] == hashlib.sha256(documents["scope_bytes"]).hexdigest(),
                "task_control_file_hash_mismatch")
        require(utc(doc["recorded_at"]) <= utc(scope["host_expires_at"]), "task_control_after_host_expiry")
    validate_control(binding, "jarvis.native_task_session_binding.v1", set())
    # Reuse the observer's exact schema and identity validation without pool,
    # observer construction, SQL, or any external calls.
    from evals.responsive_voice.native_task_barrier import FixtureScope, decode_receipt
    try:
        expected_scope = FixtureScope.model_validate_json(json.dumps(dict(
            {key: value for key, value in scope.items() if key != "format"}, session_id=binding["session_id"])))
    except ValueError:
        raise EvidenceError("invalid_task_scope") from None
    receipts, last_sequence, last_observed, admission, claim_identity = {}, 0, None, None, None
    for data in sequence(documents.get("barrier_receipts", []), "invalid_task_receipts", 4096):
        require(isinstance(data, bytes) and len(data) <= 16384, "task_receipt_over_budget")
        try:
            receipt = decode_receipt(data)
        except ValueError:
            raise EvidenceError("invalid_task_receipt") from None
        require(receipt.scope == expected_scope, "task_receipt_scope_mismatch")
        require(receipt.sequence > last_sequence and (last_observed is None or receipt.observed_at >= last_observed),
                "task_receipt_sequence_reversed_or_replayed")
        last_sequence, last_observed = receipt.sequence, receipt.observed_at
        require(admission is None or receipt.admission == admission, "task_admission_replaced")
        admission = receipt.admission
        if receipt.status == "eligible":
            claim = receipt.claim
            identity = (claim.task_id, claim.task_revision, claim.attempt_id, claim.attempt_number, claim.fence,
                        claim.worker_id, claim.runtime_host_id)
            require(claim_identity is None or identity == claim_identity, "task_claim_replaced")
            claim_identity = identity
        receipts[receipt.sequence] = receipt
    if documents.get("stimulus_completion_bytes") is None:
        result.update(verdict="unmeasured", blockers=["task_stimulus_completion_missing"])
        return result
    stimulus = _task_json(documents["stimulus_completion_bytes"], 8192, "invalid_task_stimulus_completion")
    validate_control(stimulus, "jarvis.native_task_stimulus_completion.v1", {"inputs"})
    require(uuid(stimulus["session_id"], "invalid_task_control_session") ==
            uuid(binding["session_id"], "invalid_task_control_session"), "task_control_session_mismatch")
    require(utc(stimulus["recorded_at"]) >= utc(binding["recorded_at"]), "task_control_recording_order")
    recorded = sequence(stimulus.get("inputs"), "invalid_task_stimulus_inputs", 20)
    require(len(recorded) == len(inputs), "task_stimulus_input_coverage")
    rows, blocks = {}, list(context["blockers"])
    for item in recorded:
        item = mapping(item, "invalid_task_stimulus_input")
        require(set(item) == {"input_ordinal", "status", "sample_count", "accepted_samples",
                "first_sdk_callback_monotonic", "last_sdk_callback_end_monotonic", "task_claim_sequence_at_enqueue"},
                "invalid_task_stimulus_input_fields")
        ordinal = integer(item["input_ordinal"], "invalid_task_stimulus_ordinal", 0, len(inputs) - 1)
        require(ordinal not in rows, "duplicate_task_stimulus_input")
        raw, row = inputs[ordinal], occurrences[ordinal]
        require(item["status"] == "sdk_input_complete" and raw["status"] == "sdk_input_complete"
                and raw.get("failure_code") is None, "task_stimulus_input_not_complete")
        require(integer(item["sample_count"], "invalid_task_stimulus_samples", 1) == raw["sample_count"]
                and integer(item["accepted_samples"], "invalid_task_accepted_samples", 1) == raw["sample_count"]
                and raw["sdk_accepted_clip_samples"] == raw["sample_count"], "task_stimulus_samples_mismatch")
        for external, native in (("first_sdk_callback_monotonic", "first_sdk_callback_start_uptime_seconds"),
                                 ("last_sdk_callback_end_monotonic", "last_sdk_callback_end_uptime_seconds")):
            stamp = _time(item[external], context["reference"], "invalid_task_sdk_monotonic_seconds")
            require(raw.get(native) is not None and abs(stamp - raw[native]) <= EPSILON, "task_stimulus_sdk_time_mismatch")
        selected = item["task_claim_sequence_at_enqueue"]
        if selected is not None:
            integer(selected, "invalid_task_claim_sequence", 1)
            require(selected in receipts, "task_selected_receipt_missing")
        require(ordinal not in dependent or selected is not None, "task_claim_sequence_missing")
        row["task_claim_sequence_at_enqueue"] = selected
        rows[ordinal] = item
        if row["input_verdict"] != "sdk_completed_active_input":
            blocks.append("task_input_not_completed_active")
        if ordinal not in dependent:
            continue
        receipt = receipts.get(selected)
        if receipt is None or receipt.status != "eligible":
            blocks.append("task_enqueue_eligible_receipt_missing")
            continue
        enqueue = raw.get("enqueue_requested_uptime_seconds")
        anchor = context["capture"].get("clock_anchor")
        if enqueue is None or context["reference"] is None or anchor is None:
            blocks.append("task_enqueue_clock_missing")
            continue
        center = anchor["utc_unix_seconds"] + enqueue - context["reference"]
        spread = anchor["sampling_bracket_seconds"] / 2
        if (center - spread < receipt.observed_at.timestamp()
                or center + spread >= receipt.expires_at.timestamp()):
            blocks.append("task_claim_freshness_unconfirmed")
            continue
        if any(other.sequence > selected and other.status == "blocked"
               and other.observed_at.timestamp() <= center + spread for other in receipts.values()):
            blocks.append("task_claim_invalidated_by_blocked_receipt")
            continue
        if row["input_verdict"] == "sdk_completed_active_input" and not context["blockers"]:
            row["task_trigger_verdict"] = "validated_control_metadata"
            result["validated_claim_inputs"] += 1
    result.update(verdict="unmeasured" if blocks else "validated_control_metadata", blockers=sorted(set(blocks)),
                  clock_limit="Self-reported paired anchor; no independent UTC calibration or remote causality.")
    return result


def _capture_observations(context):
    """Additive recorder claims, never stronger item/cut/playback evidence."""
    capture = context["capture"]

    def association(value, epoch=None):
        if value is None:
            return None
        value = mapping(value, "invalid_response_association")
        require(set(value) == {"connection_id", "response_id", "local_output_epoch"},
                "invalid_response_association_fields")
        result = dict(connection_id=str(uuid(value.get("connection_id"), "invalid_association_connection")),
                      response_id=_response_identifier(value.get("response_id"), "invalid_association_response"),
                      local_output_epoch=integer(value.get("local_output_epoch"), "invalid_association_epoch", 1))
        require(epoch is None or result["local_output_epoch"] == epoch, "response_association_epoch_mismatch")
        return result

    response = {"status": "not_recorded", "packet_attribution_established": False,
                "item_or_phase_attribution_established": False, "strong_trigger_authority_added": False}
    if "response_association_basis" in capture or "response_uncorrelated_decoded_blocks" in capture:
        basis = "authorized_response_and_epoch_at_callback_entry_and_return_not_packet_item_or_phase_attribution"
        require(capture.get("response_association_basis") == basis, "unsupported_response_association_basis")
        associated = item_unknown = 0
        for kind in ("decoded_sample_map", "output_gates"):
            for raw in capture[kind]:
                current = association(raw.get("response_association"), raw["local_output_epoch"])
                full = raw.get("response_identity")
                if full is not None:
                    full = _response_identity(full)
                    require(current == {key: full[key] for key in ("connection_id", "response_id", "local_output_epoch")},
                            "response_association_item_identity_mismatch")
                if kind == "decoded_sample_map" and current is not None:
                    associated += 1
                    item_unknown += full is None
        missing = integer(capture.get("response_uncorrelated_decoded_blocks"),
                          "invalid_response_unassociated_count", 0, MAX_DECODED_BLOCKS)
        require(missing == len(context["blocks"]) - associated, "response_association_counter_mismatch")
        response.update(status="recorded_control_associations", basis=basis,
                        associated_blocks=associated, item_unknown_associated_blocks=item_unknown,
                        unassociated_blocks=missing,
                        meaning="Recorder control association only; existing item-level checks and semantic limits are unchanged.")

    tail = {"status": "not_recorded", "decoded_quiet_independently_verified": False,
            "full_answer_delivery_established": False, "physical_playback_measured": False}
    if "post_media_capture" in capture:
        raw = mapping(capture["post_media_capture"], "invalid_post_media_capture")
        require(type(raw.get("schema_version")) is int and raw["schema_version"] == 1,
                "unsupported_post_media_capture_schema")
        status = raw.get("status")
        require(status in {"not_started", "capturing", "disabled", "decoded_quiet_observed", "timeout", "closed_early"},
                "invalid_post_media_capture_status")
        require(raw.get("decoded_drain_confirmed") is False and raw.get("physical_playback_measured") is False,
                "unsupported_post_media_playback_claim")
        timing = {key: _time(raw.get(key), context["reference"], "invalid_post_media_time", True)
                  for key in ("media_end_scheduled_uptime_seconds", "media_end_observed_uptime_seconds",
                              "capture_deadline_uptime_seconds", "finished_uptime_seconds",
                              "generation_done_uptime_seconds", "natural_buffer_stop_uptime_seconds",
                              "authorized_quiet_start_uptime_seconds", "authorized_quiet_callback_end_uptime_seconds")}
        scheduled, deadline = timing["media_end_scheduled_uptime_seconds"], timing["capture_deadline_uptime_seconds"]
        require((scheduled is None) == (deadline is None), "partial_post_media_deadline")
        if scheduled is not None:
            require(0 <= deadline - scheduled <= 20 + EPSILON, "post_media_tail_out_of_bounds")
        for start, end in (("media_end_scheduled_uptime_seconds", "media_end_observed_uptime_seconds"),
                           ("media_end_observed_uptime_seconds", "finished_uptime_seconds"),
                           ("authorized_quiet_start_uptime_seconds", "authorized_quiet_callback_end_uptime_seconds")):
            require(timing[start] is None or timing[end] is None or timing[start] <= timing[end] + EPSILON,
                    "reversed_post_media_interval")
        inputs_complete = boolean(raw.get("required_inputs_complete"), "invalid_post_media_input_completion")
        quiet_blocks = integer(raw.get("authorized_quiet_blocks"), "invalid_post_media_quiet_blocks", 0, MAX_DECODED_BLOCKS)
        quiet_seconds = number(raw.get("required_quiet_seconds"), "invalid_post_media_quiet_duration")
        gap = number(raw.get("maximum_callback_gap_seconds"), "invalid_post_media_callback_gap")
        require(abs(quiet_seconds - 0.5) <= EPSILON and abs(gap - 0.03) <= EPSILON,
                "unsupported_post_media_quiet_policy")
        owner = association(raw.get("response_association"))
        if status == "decoded_quiet_observed":
            require(inputs_complete and owner is not None and quiet_blocks >= 50
                    and all(value is not None for value in timing.values()), "incomplete_post_media_quiet_claim")
            require(timing["authorized_quiet_callback_end_uptime_seconds"] - timing["authorized_quiet_start_uptime_seconds"]
                    + EPSILON >= quiet_seconds, "short_post_media_quiet_claim")
        tail.update(status="recorded_native_observation", recorded_status=status, **timing,
                    required_inputs_complete=inputs_complete, response_association=owner,
                    authorized_quiet_blocks=quiet_blocks, required_quiet_seconds=quiet_seconds,
                    maximum_callback_gap_seconds=gap,
                    meaning="Native quiet/stop claims are not independently rederived here. No complete-answer, drain or hearing verdict; original response deadlines still apply.")
    return {"response_association": response, "post_media_capture": tail}


_PROVIDER_INPUT_STEPS = (
    "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed", "conversation.item.input_audio_transcription.completed",
)
_PROVIDER_INPUT_INDEX_FIELDS = (
    "speech_started_control_event_index", "speech_stopped_control_event_index",
    "committed_control_event_index", "transcription_completed_control_event_index",
)
_PROVIDER_INPUT_TIMES = (
    "transaction_reserved_uptime_seconds", "input_complete_uptime_seconds",
    "scene_deadline_uptime_seconds", "deadline_uptime_seconds", "enqueue_revalidated_uptime_seconds",
)
_PROVIDER_INPUT_COUNTS = ("clean_control_event_count", "validation_control_event_count",
    "session_created_control_event_index", *_PROVIDER_INPUT_INDEX_FIELDS)
_PROVIDER_INPUT_FAILURES = frozenset("provider_input_" + suffix for suffix in (
    "evidence_limit", "invalid_event", "event_conflict", "session_replaced", "identity_missing_or_changed",
    "item_reused_or_limit", "lifecycle_conflict", "unknown_item", "transcription_failed", "connection_reset",
    "clean_boundary_missing", "clean_boundary_changed", "evidence_unavailable", "sdk_incomplete",
    "ambiguous_items", "before_sdk_production", "competing_sdk_input", "not_reached",
)) | {"trigger_outputWaitExpired", "trigger_providerInputFinalizationUnavailable",
      "enqueue_providerInputFinalizationUnavailable", "run_closed_before_injection"}


def _provider_input_trace(context, proof, previous, enqueue):
    """Reconstruct one bounded clean-prefix transaction, not semantic causality."""
    controls = sequence(context["capture"].get("control_events"), "invalid_provider_control_events", 512)
    clean, cutoff = proof["clean_control_event_count"], proof["validation_control_event_count"]
    created = proof["session_created_control_event_index"]
    require(0 <= created < clean < cutoff <= len(controls), "invalid_provider_control_prefix")
    session = mapping(controls[created], "invalid_provider_session_event")
    require(session.get("stage") == "provider_data_channel_handled"
            and isinstance(session.get("fields"), dict)
            and session["fields"].get("event_type") == "session.created", "provider_session_start_missing")
    reserve = proof["transaction_reserved_uptime_seconds"]
    revalidated = proof["enqueue_revalidated_uptime_seconds"]
    first_sdk = previous["first_sdk_callback_start_uptime_seconds"]
    connection, selected = proof["connection_id"], proof["provider_item_id"]
    items, seen_events, current_indices = {}, {}, None
    last_received = last_handled = None
    for index, raw in enumerate(controls[:cutoff]):
        raw = mapping(raw, "invalid_provider_control_event")
        handled = _time(raw.get("handled_monotonic_s"), context["reference"], "invalid_provider_control_time")
        received = _time(raw.get("received_monotonic_s"), context["reference"], "invalid_provider_control_time", True)
        require(handled <= (reserve if index < clean else revalidated) + EPSILON,
                "provider_control_prefix_time_mismatch")
        if index < created:
            continue
        if raw.get("stage") == "connection_reset":
            raise EvidenceError("provider_input_connection_reset")
        if raw.get("stage") != "provider_data_channel_handled":
            continue
        fields = mapping(raw.get("fields"), "invalid_provider_control_fields")
        event = fields.get("event_type")
        require(isinstance(event, str), "invalid_provider_control_type")
        relevant = (event in {"session.created", "session.closed", "replay.provider_input.invalid"}
                    or event.startswith(("input_audio_buffer.", "conversation.item.input_audio_transcription.")))
        if not relevant:
            continue
        require(fields.get("boundary") == "ordered_current_provider_channel"
                and str(uuid(fields.get("connection_id"), "invalid_provider_connection")) == connection,
                "provider_input_identity_missing_or_changed")
        require(received is not None and handled is not None and received <= handled + EPSILON,
                "invalid_provider_control_order")
        require((last_received is None or received + EPSILON >= last_received)
                and (last_handled is None or handled + EPSILON >= last_handled), "provider_control_reordered")
        last_received, last_handled = received, handled
        event_id = _response_identifier(fields.get("provider_event_id"), "invalid_provider_event_id")
        item_id = fields.get("item_id")
        if event != "session.created":
            item_id = _response_identifier(item_id, "invalid_provider_input_item")
        signature = (event, item_id, connection)
        if event_id in seen_events:
            require(seen_events[event_id] == signature, "provider_input_event_conflict")
            continue
        seen_events[event_id] = signature
        if index == created:
            require(event == "session.created" and item_id is None, "provider_session_start_missing")
            continue
        require(event != "session.created", "provider_input_session_replaced")
        require(event in _PROVIDER_INPUT_STEPS, "provider_input_invalid_event")
        step = _PROVIDER_INPUT_STEPS.index(event)
        if index == clean:
            require(all(len(value) == 4 for value in items.values()), "provider_input_clean_boundary_missing")
        if step == 0:
            require(item_id not in items and len(items) < 20, "provider_input_item_reused_or_limit")
            items[item_id] = []
        require(item_id in items and len(items[item_id]) == step, "provider_input_lifecycle_conflict")
        if index >= clean:
            require(item_id == selected and received + EPSILON >= max(reserve, first_sdk),
                    "provider_input_old_or_competing_item")
            require(all(key == selected or len(value) == 4 for key, value in items.items()),
                    "provider_input_clean_boundary_missing")
            if current_indices is None:
                require(step == 0, "provider_input_before_clean_boundary")
                current_indices = items[item_id]
        else:
            require(item_id != selected, "provider_input_item_reused_or_limit")
        items[item_id].append(index)
    require(current_indices == [proof[key] for key in _PROVIDER_INPUT_INDEX_FIELDS]
            and len(current_indices or []) == 4, "provider_input_finalization_not_proven")
    require(revalidated <= enqueue + EPSILON, "provider_input_revalidation_after_enqueue")


def _provider_input_finalizations(context, inputs, occurrences, manifest_bytes):
    applicable = [i for i, raw in enumerate(inputs) if raw["trigger"]["kind"] == "after_provider_input_finalized"]
    for index, raw in enumerate(inputs):
        require(index in applicable or raw.get("provider_input_finalization") is None,
                "provider_input_proof_on_unrelated_trigger")
    if not applicable:
        return None
    manifest = None
    if manifest_bytes is not None:
        manifest = _task_json(manifest_bytes, 262144, "invalid_provider_input_manifest")
        require(type(manifest.get("version")) is int and manifest["version"] == 1
                and uuid(manifest.get("run_id"), "invalid_provider_manifest_run") == context["run"]
                and manifest.get("voice_route") == "realtime" and manifest.get("capture_mode", "auto") == "auto",
                "provider_input_requires_automatic_realtime")
        duration = number(manifest.get("duration_seconds"), "invalid_provider_scene_duration")
        plans = sequence(manifest.get("utterances"), "invalid_provider_manifest_inputs", 20)
        # Only the explicitly declared three-cut task plan extends the scene.
        # This changes no trigger timeout or evidence/authority requirement.
        extended_three_cut = False
        if duration == 150 and len(plans) == 8:
            triggers = [_task_trigger_semantics(mapping(plan, "invalid_provider_manifest_input").get("trigger"), i)
                        for i, plan in enumerate(plans)]
            barrier = manifest.get("task_barrier")
            extended_three_cut = (
                isinstance(barrier, dict)
                and type(barrier.get("version")) is int
                and type(barrier.get("request_input_ordinal")) is int
                and barrier == {"version": 1, "scope_file": "task-scope.json", "request_input_ordinal": 0}
                and [trigger["kind"] for trigger in triggers] ==
                    ["at_time", "after_task_claim"] +
                    ["response_output_start", "after_provider_input_finalized"] * 3
                and all(trigger["after_input_ordinal"] == i - 1 for i, trigger in enumerate(triggers) if i)
                and all(triggers[i].get("require_task_claim") is True for i in (2, 4, 6)))
        require(0 < duration <= 120 or extended_three_cut, "invalid_provider_scene_duration")
        require(len(plans) == len(inputs), "provider_manifest_input_count_mismatch")
        for ordinal, (plan, raw) in enumerate(zip(plans, inputs)):
            require(isinstance(plan, dict) and plan.get("clip_id") == raw.get("clip_id")
                    and _task_trigger_semantics(plan.get("trigger"), ordinal) ==
                    _task_trigger_semantics(raw["trigger"], ordinal), "provider_manifest_input_mismatch")
    validated = 0
    for ordinal in applicable:
        raw, row = inputs[ordinal], occurrences[ordinal]
        trigger = _task_trigger_semantics(raw["trigger"], ordinal)
        row.update(provider_input_finalization_verdict="missing", provider_input_finalization_blockers=[],
                   canonical_input_binding="unknown")
        proof = raw.get("provider_input_finalization")
        if proof is None:
            continue
        proof = dict(mapping(proof, "invalid_provider_input_finalization"))
        keys = {"schema_version", "after_input_ordinal", "correlation_basis", "canonical_input_binding",
                "status", "failure_code", "connection_id", "provider_item_id", *_PROVIDER_INPUT_TIMES, *_PROVIDER_INPUT_COUNTS}
        require(set(proof) == keys and type(proof["schema_version"]) is int and proof["schema_version"] == 1
                and proof["correlation_basis"] == "replay_exclusive_provider_transaction"
                and proof["canonical_input_binding"] == "unknown", "invalid_provider_input_proof_fields")
        require(integer(proof["after_input_ordinal"], "invalid_provider_input_ordinal") == ordinal - 1,
                "provider_input_predecessor_not_adjacent")
        status = proof["status"]
        require(status in {"validated_provider_transaction", "not_reached"}, "invalid_provider_input_status")
        failure = proof["failure_code"]
        require(failure is None or isinstance(failure, str) and failure in _PROVIDER_INPUT_FAILURES,
                "invalid_provider_input_failure_code")
        for key in _PROVIDER_INPUT_TIMES:
            _time(proof[key], context["reference"], "invalid_provider_input_time", True)
        for key in _PROVIDER_INPUT_COUNTS:
            if proof[key] is not None:
                integer(proof[key], "invalid_provider_control_index", 0, 512)
        if proof["connection_id"] is not None:
            proof["connection_id"] = str(uuid(proof["connection_id"], "invalid_provider_connection"))
        if proof["provider_item_id"] is not None:
            _response_identifier(proof["provider_item_id"], "invalid_provider_input_item")
        if status == "not_reached":
            require(raw.get("enqueue_requested_uptime_seconds") is None and raw.get("sdk_accepted_clip_samples") == 0,
                    "provider_input_unproved_enqueue")
            row["provider_input_finalization_verdict"] = "not_reached"
            row["provider_input_finalization_blockers"] = [failure or "provider_input_not_reached"]
            continue
        require(failure is None and all(proof[key] is not None for key in keys - {"failure_code"}),
                "incomplete_provider_input_proof")
        blockers = list(context["blockers"])
        if manifest is None:
            blockers.append("provider_input_manifest_missing")
        if context["reference"] is None:
            blockers.append("clock_anchor_missing")
        for key in ("control_events_truncated", "control_event_timing_invalid"):
            if key not in context["capture"]:
                blockers.append("provider_input_control_flags_missing")
            elif boolean(context["capture"][key], "invalid_" + key):
                blockers.append(key)
        if context["capture"].get("control_events") is None:
            blockers.append("provider_input_controls_missing")
        previous = inputs[ordinal - 1]
        if (previous["status"] != "sdk_input_complete" or previous.get("failure_code") is not None
                or previous["sdk_accepted_clip_samples"] != previous["sample_count"]):
            blockers.append("provider_input_sdk_incomplete")
        enqueue = raw.get("enqueue_requested_uptime_seconds")
        if enqueue is None:
            blockers.append("provider_input_enqueue_missing")
        if blockers:
            row.update(provider_input_finalization_verdict="unmeasured",
                       provider_input_finalization_blockers=sorted(set(blockers)))
            continue
        sdk_end, sdk_start = previous["last_sdk_callback_end_uptime_seconds"], previous["first_sdk_callback_start_uptime_seconds"]
        reserve, revalidated = proof["transaction_reserved_uptime_seconds"], proof["enqueue_revalidated_uptime_seconds"]
        require(sdk_end is not None and sdk_start is not None and previous.get("enqueue_requested_uptime_seconds") is not None
                and reserve <= previous["enqueue_requested_uptime_seconds"] + EPSILON
                and previous["enqueue_requested_uptime_seconds"] <= sdk_start + EPSILON
                and sdk_end <= revalidated + EPSILON and revalidated <= enqueue + EPSILON,
                "provider_input_sdk_completion_or_reservation_order")
        require(all(item.get("last_sdk_callback_end_uptime_seconds") is not None
                    and item["last_sdk_callback_end_uptime_seconds"] <= reserve + EPSILON
                    and item["status"] == "sdk_input_complete" for item in inputs[:ordinal - 1]),
                "provider_input_queue_not_exclusive")
        require(abs(proof["input_complete_uptime_seconds"] - sdk_end) <= EPSILON
                and abs(proof["scene_deadline_uptime_seconds"] - (context["reference"] + duration)) <= EPSILON,
                "provider_input_deadline_anchor_mismatch")
        deadline = min(proof["scene_deadline_uptime_seconds"], sdk_end + trigger["timeout_seconds"])
        require(abs(proof["deadline_uptime_seconds"] - deadline) <= EPSILON
                and revalidated < deadline and enqueue < deadline, "provider_input_deadline_expired_or_changed")
        _provider_input_trace(context, proof, previous, enqueue)
        row["provider_input_finalization_verdict"] = "validated_provider_transaction"
        validated += 1
    return dict(validated_transactions=validated, required_transactions=len(applicable),
                canonical_input_binding="unknown", semantic_pause_established=False,
                meaning="Recorded exclusive provider transaction plus SDK completion; not canonical user-source binding, response attribution, useful speech or hearing.")


def analyze_capture(capture, pcm, annotations=None, *, task_evidence=None, replay_manifest_bytes=None):
    """Return evidence verdicts without modifying any input or judging content."""
    context = _prepare(capture, pcm)
    raw = sequence(capture.get("utterances"), "invalid_utterances", 20)
    callback_lists = [sequence(mapping(item, "invalid_utterance").get("callbacks"),
                               "invalid_input_callbacks", 6020) for item in raw]
    require(sum(len(callbacks) for callbacks in callback_lists) <= 6020, "input_callback_map_over_budget")
    occurrences = [_input(context, item, index) for index, item in enumerate(raw)]
    previous_input_end = None
    for callbacks in callback_lists:
        if not callbacks:
            continue
        # These are locally recorded SDK envelopes, already validated by
        # _input. Empty occurrences cannot reset the ordering constraint. This
        # asserts no provider receipt, recognition or response relationship.
        require(previous_input_end is None or callbacks[0]["callback_start_uptime_seconds"] + EPSILON >= previous_input_end,
                "nonmonotonic_cross_occurrence_input_callbacks")
        previous_input_end = callbacks[-1]["callback_end_uptime_seconds"]
    task_controls = _task_controls(context, raw, occurrences, task_evidence)
    task_manifest = task_evidence.get("manifest_bytes") if task_evidence else None
    if replay_manifest_bytes is not None and task_manifest is not None:
        require(_task_json(replay_manifest_bytes, 262144, "invalid_provider_input_manifest") ==
                _task_json(task_manifest, 262144, "invalid_task_manifest"), "provider_task_manifest_mismatch")
    provider_inputs = _provider_input_finalizations(context, raw, occurrences,
        replay_manifest_bytes if replay_manifest_bytes is not None else task_manifest)
    response_evidence = _response_evidence(context, raw, occurrences)
    associations = _annotations(context, annotations, {item["occurrence_id"] for item in occurrences})
    for item in occurrences:
        end = item["input_end_exclusive_uptime_seconds"]
        energy = _first_energy(context, end) if end is not None else None
        item["first_decoded_non_silent_after_input_end"] = {
            "sample": energy, "response_onset_established": False,
            "warning": "May be continuing old output; neither response attribution nor usefulness is inferred."
        }
        labels = associations[item["occurrence_id"]]
        item["speech_onset_to_annotated_old_tail"] = _metric(context, item, "old_answer_tail", labels["old_answer_tail"])
        item["input_end_to_annotated_useful_onset"] = _metric(context, item, "useful_onset", labels["useful_onset"])
    result = {
        "schema_version": 1, "capture_run_id": str(context["run"]), "media_epoch_id": str(context["media"]),
        "pcm_sha256": context["pcm_sha256"], "capture_verdict": "limited_evidence" if context["blockers"] else "structurally_valid_evidence",
        "timing_blockers": context["blockers"], "occurrences": occurrences,
        "response_control_evidence": response_evidence, "task_control_evidence": task_controls, "actual_cuts": None,
        "capture_observations": _capture_observations(context),
        "first_decoded_onset": {"sample": _first_energy(context), "definition": "first retained abs>=17 sample",
                                  "response_attribution": None, "useful_speech_established": False},
        "timing_basis": "monotonic seconds, nominal SDK callback-relative sample placement; not hearing",
        "clock_validation_limit": "Self-reported domain/units consistency, not independent clock calibration or UTC drift correction.",
        "provider_response_attribution": "unavailable", "hearing_measured": False, "answer_quality_judged": False,
        "aggregate_percentile_qualification": {"supported": False,
            "reason": "Comparable groups and an explicit required sample-count policy are not established; this reader never qualifies percentiles."},
    }
    if provider_inputs is not None:
        result["provider_input_finalization_evidence"] = provider_inputs
    return result


def _read(path, limit):
    path = Path(path)
    require(path.stat().st_size <= limit, "artifact_size_limit")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    require(len(data) <= limit, "artifact_size_limit")
    return data


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def nonfinite(_):
        raise EvidenceError("nonfinite_json_number")

    try:
        return json.loads(_read(path, MAX_JSON_BYTES), object_pairs_hook=pairs, parse_constant=nonfinite)
    except (json.JSONDecodeError, UnicodeError):
        raise EvidenceError("invalid_json") from None


def read_capture(capture_path, pcm_path=None, annotations_path=None, *, task_evidence=None, replay_manifest_bytes=None):
    capture_path = Path(capture_path)
    return analyze_capture(_json(capture_path), _read(pcm_path or capture_path.parent / "decoded.pcm", MAX_PCM_BYTES),
                           _json(annotations_path) if annotations_path else None, task_evidence=task_evidence,
                           replay_manifest_bytes=replay_manifest_bytes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--pcm")
    parser.add_argument("--annotations")
    parser.add_argument("--task-manifest")
    parser.add_argument("--replay-manifest", help="retained manifest for opt-in provider-input transaction/scene validation")
    parser.add_argument("--task-scope")
    parser.add_argument("--task-session-binding")
    parser.add_argument("--task-stimulus-completion")
    parser.add_argument("--task-receipt", action="append", default=[], help="retained barrier receipt, in observed sequence order")
    args = parser.parse_args(argv)
    try:
        task_paths = {"manifest_bytes": args.task_manifest, "scope_bytes": args.task_scope,
                      "session_binding_bytes": args.task_session_binding, "stimulus_completion_bytes": args.task_stimulus_completion}
        documents = None
        if any(task_paths.values()) or args.task_receipt:
            documents = {key: _read(path, 262144 if key == "manifest_bytes" else 8192)
                         for key, path in task_paths.items() if path}
            require(len(args.task_receipt) <= 4096, "task_receipt_count_over_budget")
            documents["barrier_receipts"] = [_read(path, 16384) for path in args.task_receipt]
        result = read_capture(args.capture, args.pcm, args.annotations, task_evidence=documents,
                              replay_manifest_bytes=_read(args.replay_manifest, 262144) if args.replay_manifest else None)
    except EvidenceError as error:
        print(json.dumps({"verdict": "invalid_artifact", "error_code": str(error)}))
        return 2
    except OSError:
        print(json.dumps({"verdict": "unreadable_artifact", "error_code": "artifact_read_failed"}))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
