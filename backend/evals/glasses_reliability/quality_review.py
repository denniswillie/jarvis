"""Validate artifact-bound manual judgments; never infer quality from keywords."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from evals.gemini_live.transport_probe import read_pcm
from evals.glasses_reliability.quality_cases import BY_NAME, definition_manifest
from evals.glasses_reliability.quality_probe import select_matrix


JUDGMENTS = {"pass", "fail", "unverified"}


def cleanup_is_zero(record):
    remaining = record.get("remaining_rows")
    return type(remaining) is int and remaining == 0


def identity(row):
    if type(row.get("repeat")) is not int:
        raise ValueError("Repeat must be an integer, not a coerced or boolean identity")
    return row.get("case"), row["repeat"], row.get("provider")


def indexed(rows, expected):
    result = {}
    for row in rows:
        key = identity(row)
        if key not in expected or key in result:
            raise ValueError("Unexpected or duplicated case/repeat/provider identity")
        result[key] = row
    if set(result) != expected:
        raise ValueError("Missing trials or manual reviews cannot disappear from the denominator")
    return result


def final_audio_files(artifact):
    turns = artifact.get("turns", [])
    responses = set(turns[-1].get("response_ids", [])) if turns else set()
    items = {event.get("item_id") for event in artifact.get("phone_timeline", [])
             if event.get("type") == "audio.output.delta" and event.get("response_id") in responses}
    return [file for file in artifact.get("audio_files", []) if file.get("item_id") in items]


def score_verified(records):
    """Aggregate only judgments whose identities, artifacts and evidence were checked."""
    report = {}
    for provider in ("openai", "gemini"):
        rows = [row for row in records if row["provider"] == provider]
        code = [row for row in rows if BY_NAME[row["case"]].family == "code"]
        continuity = [row for row in rows if BY_NAME[row["case"]].family == "continuity"]
        uncertainty = [row for row in rows if BY_NAME[row["case"]].family == "uncertainty"]
        approvals = sum(row["unsupported_confident_approval"] for row in code if BY_NAME[row["case"]].approval_trap)
        useful = sum(row["judgment"] == "pass" for row in code)
        code_gate = bool(code) and useful / len(code) >= .8 and approvals == 0
        continuity_gate = bool(continuity) and all(row["judgment"] == row["audio_judgment"] == "pass" for row in continuity)
        uncertainty_gate = bool(uncertainty) and all(row["judgment"] == "pass" for row in uncertainty)
        infrastructure = bool(rows) and all(row["infrastructure_passed"] for row in rows)
        report[provider] = {
            "reviewed_runs": len(rows), "infrastructure_passed": infrastructure,
            "clear_code_runs": len(code), "useful_correct_clear_code_runs": useful,
            "useful_correct_clear_code_fraction": useful / len(code) if code else None,
            "unsupported_confident_approvals_on_traps": approvals, "clear_code_gate": code_gate,
            "continuity_runs": len(continuity), "continuity_gate": continuity_gate,
            "uncertainty_runs": len(uncertainty), "uncertainty_gate": uncertainty_gate,
            "unverified_judgments": sum(row["judgment"] == "unverified" or row["audio_judgment"] == "unverified" for row in rows),
            "provider_quality_acceptance": infrastructure and code_gate and continuity_gate and uncertainty_gate,
        }
    return report


def review_matrix(matrix_dir, review_path, audio_review_path):
    manifest = json.loads((matrix_dir / "run-manifest.json").read_text())
    expected_manifest = definition_manifest()
    if any(manifest.get(key) != expected_manifest[key] for key in ("schema", "cases", "schedule", "scoring", "limits")):
        raise ValueError("Native definitions differ from the frozen quality rubric")
    if (manifest.get("selected_cases") != list(BY_NAME) or type(manifest.get("repeats")) is not int
            or manifest["repeats"] != 3 or manifest.get("providers") != ["openai", "gemini"]):
        raise ValueError("Final review requires the complete declared three-repeat matrix")
    expected = {(case.name, repeat, provider) for case, repeat, provider in select_matrix(list(BY_NAME), 3)}
    matrix = json.loads((matrix_dir / "matrix.json").read_text())
    if matrix.get("expected_runs") != len(expected) or matrix.get("recorded_runs") != len(expected):
        raise ValueError("Native matrix is incomplete")
    native = indexed(matrix["results"], expected)
    reviews = json.loads(review_path.read_text())
    if reviews.get("schema") != 1:
        raise ValueError("Unsupported manual-review format")
    grades = indexed(reviews["reviews"], expected)
    audio_bytes = audio_review_path.read_bytes()
    if reviews.get("audio_review_sha256") != sha256(audio_bytes).hexdigest():
        raise ValueError("Manual review references a different independent audio artifact")
    audio = json.loads(audio_bytes)
    if audio.get("prompt_or_expected_transcript_supplied") is not False:
        raise ValueError("Independent audio evidence must not be prompted with expected answers")
    transcribed = {row["pcm_sha256"] for row in audio["results"] if row.get("status") == "transcribed"
                   and isinstance(row.get("transcript"), str) and row["transcript"].strip()}
    verified = []
    for key in sorted(expected):
        name, repeat, provider = key
        raw = (matrix_dir / f"{name}-{repeat}-{provider}.json").read_bytes()
        digest = sha256(raw).hexdigest()
        row, grade, artifact = native[key], grades[key], json.loads(raw)
        if digest != row.get("artifact_sha256") or digest != grade.get("artifact_sha256") or identity(artifact) != key:
            raise ValueError("Native output changed or its manual grade belongs to another artifact")
        if grade.get("judgment") not in JUDGMENTS or type(grade.get("unsupported_confident_approval")) is not bool:
            raise ValueError("Every output needs an explicit judgment and approval assessment")
        if not isinstance(grade.get("notes"), str) or not grade["notes"].strip():
            raise ValueError("Explain each judgment against the frozen expected/forbidden criteria")
        if grade["unsupported_confident_approval"] and grade["judgment"] != "fail":
            raise ValueError("An unsupported confident approval cannot receive a passing judgment")
        audio_judgment = grade.get("audio_judgment")
        if BY_NAME[name].family == "continuity":
            if audio_judgment not in JUDGMENTS:
                raise ValueError("Dictation needs a separate independent-audio judgment")
            files = final_audio_files(artifact)
            hashes = []
            for file in files:
                filename = file["file"]
                if Path(filename).name != filename:
                    raise ValueError("Audio artifacts must be files inside this native matrix")
                pcm_hash = sha256(read_pcm(matrix_dir / filename)).hexdigest()
                if pcm_hash != file["pcm_sha256"]:
                    raise ValueError("Saved final-answer audio changed after the native run")
                hashes.append(pcm_hash)
            declared = grade.get("audio_pcm_sha256s", [])
            if not isinstance(declared, list) or len(declared) != len(set(declared)) or set(declared) != set(hashes):
                raise ValueError("Audio review must cover the final response, not an earlier interrupted answer")
            if audio_judgment != "unverified" and (not hashes or not set(hashes) <= transcribed):
                raise ValueError("An audio judgment lacks independent evidence for the final PCM")
        elif audio_judgment != "not_applicable":
            raise ValueError("Separate exact-dictation scoring applies to continuation cases")
        checks = artifact.get("checks", {})
        infrastructure = (row.get("infrastructure_passed") is True and artifact.get("infrastructure_passed") is True
                          and bool(checks) and all(value is True for value in checks.values())
                          and cleanup_is_zero(artifact.get("cleanup", {})))
        verified.append({**grade, "infrastructure_passed": infrastructure})
    cleanup_zero = cleanup_is_zero(matrix.get("cleanup", {}))
    providers = score_verified(verified)
    if not cleanup_zero:
        for provider in providers.values():
            provider["infrastructure_passed"] = provider["provider_quality_acceptance"] = False
    return {"schema": 1, "reviewed_runs": len(verified), "cleanup_zero": cleanup_zero,
            "source_revision": manifest["source_revision"], "providers": providers, "reviews": verified,
            "scope": "Artifact-bound manual semantic judgments and independent ASR; not human listening or device playback."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--audio-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = review_matrix(args.matrix, args.reviews, args.audio_review)
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["providers"], indent=2))
