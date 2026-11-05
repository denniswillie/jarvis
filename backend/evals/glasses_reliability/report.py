"""Summarize trace JSONL without treating absent timings as zero.

Usage: python -m evals.glasses_reliability.report < private-trace.jsonl
Accepts logger-prefixed backend traces. Never prints source lines or payloads.
"""
from collections import Counter
import json
import math
import sys

from app.realtime.telemetry import safe_fields


def response_timings(samples):
    """Only subtract clocks from the same process and correlated response."""
    writes, starts, audio = {}, {}, {}
    for sample in samples:
        scope = (sample["process_id"], sample.get("audio_session_id"))
        event = sample["event"]
        if event == "response_write_finished" and sample.get("status") == "written":
            for name in ("request_id", "response_id"):
                if sample.get(name):
                    writes.setdefault((*scope, name, sample[name]), sample)
        if sample.get("response_id"):
            key = (*scope, sample["response_id"])
            if event == "provider_response_started":
                starts.setdefault(key, sample)
            elif event == "first_audio_forwarded":
                audio.setdefault(key, sample)

    def elapsed(start, end):
        if start is None or end is None:
            return None
        difference = (end["monotonic_s"] - start["monotonic_s"]) * 1000
        return round(difference, 3) if difference >= 0 else None

    results = []
    for key in dict.fromkeys([*starts, *audio]):
        start, first = starts.get(key), audio.get(key)
        written = writes.get((*key[:2], "response_id", key[2]))
        if written is None and start is not None and start.get("request_id"):
            written = writes.get((*key[:2], "request_id", start["request_id"]))
        results.append({"process_id": key[0], "audio_session_id": key[1], "response_id": key[2],
                        "response_write_to_first_forwarded_ms": elapsed(written, first),
                        "provider_start_to_first_forwarded_ms": elapsed(start, first)})
    return results


def summarize(lines):
    counts = Counter()
    first_audio = set()
    processes = set()
    malformed = 0
    samples = []
    for line in lines:
        if "realtime_trace " not in line:
            continue
        try:
            record = json.loads(line.split("realtime_trace ", 1)[1])
            if not isinstance(record, dict) or record.get("schema") != 1:
                raise ValueError("unsupported trace")
            event = record["event"]
            if "event_type" not in safe_fields({"event_type": event}):
                raise ValueError("missing event")
            process = safe_fields({"item_id": record.get("process_id")}).get("item_id")
            safe = safe_fields(record)
            counts[event] += 1
            if process is not None:
                processes.add(process)
            if event == "first_audio_forwarded":
                first_audio.add((safe.get("audio_session_id"), safe.get("response_id"), safe.get("item_id")))
            clock = record.get("monotonic_s")
            if process is not None and type(clock) in (int, float) and 0 <= clock <= 1e15 and math.isfinite(clock):
                samples.append({**safe, "event": event, "process_id": process, "monotonic_s": clock})
        except (ValueError, KeyError, TypeError):
            malformed += 1
    return {"events": dict(counts), "distinct_audio_items_forwarded": len(first_audio),
            "processes": len(processes), "malformed_traces": malformed,
            "responses": response_timings(samples),
            "hardware_first_audible_ms": None,
            "timing_limit": "Backend writes do not measure phone or glasses playback; never subtract clocks across processes."}


if __name__ == "__main__":
    print(json.dumps(summarize(sys.stdin), indent=2))
