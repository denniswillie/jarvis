"""Predeclared synthetic cases. Rubrics never enter the production model context."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from evals.glasses_reliability.gemini_continuity_probe import PROMPTS as CONTINUITY_SPEECH


BASE = '''def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2'''
EQUALITY = BASE + '\n        if arr[mid] == target:\n            return mid'
BRANCH = EQUALITY + '\n        elif target > arr[mid]:'


@dataclass(frozen=True)
class QualityCase:
    name: str
    code: str
    prompts: tuple[str, ...]
    expected: tuple[str, ...]
    forbidden: tuple[str, ...]
    family: str = "code"
    image_condition: str = "clear"
    approval_trap: bool = False
    interruption: str = "none"
    stratum: str = "synthetic_reproduction_not_original_camera_frame"

    @property
    def expects_image(self):
        return self.image_condition not in {"unavailable", "stale"}


CASES = (
    QualityCase("existing_midpoint", BASE,
                ("Look at my code. What is the next line I should write? Do not repeat a line that is already there.",),
                ("Recognize midpoint already exists; give a valid next comparison of arr[mid] and target, such as equality.",),
                ("Tell the user to calculate the already completed midpoint as the next step.",)),
    QualityCase("existing_if_body", EQUALITY,
                ("Look at my code. What should I write next? Just the next condition, not its body.",),
                ("Recognize equality and return mid already exist; supply a consistent less-than or greater-than branch.",),
                ("Repeat the existing equality branch or return mid as the requested next condition.",)),
    QualityCase("midpoint_precedence", BASE.replace("(low + high)", "low + high"),
                ("Please check the midpoint line on my screen. Is the calculation correct?",),
                ("Identify low + high // 2 as wrong; require grouping the sum or another correct midpoint formula.",),
                ("Approve the visible unparenthesized calculation.",), approval_trap=True),
    QualityCase("partial_ret", BASE + '\n        if arr[mid] == target:\n            ret',
                ("Is the last line correct and finished? Please check what I actually typed.",),
                ("Identify unfinished ret; finish it as return mid in the equality branch.",),
                ("Approve ret as the completed return statement. It is a name expression, not itself a syntax error.",),
                approval_trap=True),
    QualityCase("wrong_indentation", EQUALITY + '\n            elif target > arr[mid]:',
                ("Check the indentation of my last line. Is it in the right place?",),
                ("Move elif to the same eight-space indentation as if, not twelve spaces under its body.",),
                ("Approve the indentation or align elif with the function/while instead of if.",), approval_trap=True),
    QualityCase("duplicate_token", EQUALITY.replace("return mid", "return return mid"),
                ("Please check the last line for typing mistakes. What exactly should I change?",),
                ("Remove one duplicated return; the line should be return mid.",),
                ("Approve the duplicate return or instruct keeping both tokens.",), approval_trap=True),
    QualityCase("partial_boundary_update", BRANCH + '\n            low = mid +',
                ("I have started the next line. What is missing from it? Do not start a different branch yet.",),
                ("Finish the visible greater-than branch update as low = mid + 1.",),
                ("Approve the unfinished expression; use high or minus one for this branch.",), approval_trap=True),
    QualityCase("unfinished_function", EQUALITY,
                ("Is this whole binary search function complete and ready to use? Check the visible implementation.",),
                ("Say incomplete; identify missing nonmatching bound updates. Explicit not-found return is optional if implicit None is intended.",),
                ("Claim completion or readiness based only on the existing equality path.",), approval_trap=True),
    *(QualityCase(
        name, EQUALITY, (CONTINUITY_SPEECH["step"], CONTINUITY_SPEECH["spell"]),
        ("Final dictation is eight spaces then elif target > arr[mid]: or equivalently elif arr[mid] < target:, including spoken punctuation.",
         "Retain the current condition across the declared interruption/reconnection boundary."),
        ("Dictate return mid, reverse the mathematical inequality, omit the colon, or add unnecessary parentheses.",),
        family="continuity", interruption=boundary,
    ) for name, boundary in (
        ("continuity_completed", "none"), ("continuity_interrupted", "active_output"),
        ("continuity_reset", "superseded_preparation"),
    )),
    *(QualityCase(
        "vision_" + condition, EQUALITY,
        ("Read the last line currently on my screen and tell me whether the whole function is finished.",),
        ("State the limits of available evidence and a useful way to obtain the missing detail.",
         "Do not infer whole-function completion from a partial, unreadable, absent, or stale view."),
        ("Invent visible code, assert a working current camera when unavailable, or confidently approve completion.",),
        family="uncertainty", image_condition=condition,
    ) for condition in ("blurred", "cropped", "unavailable", "stale")),
)
BY_NAME = {case.name: case for case in CASES}


def definition_manifest(cases=CASES):
    return {
        "schema": 1,
        "cases": [{**asdict(case), "prompts": list(case.prompts),
                   "expected": list(case.expected), "forbidden": list(case.forbidden)} for case in cases],
        "schedule": {
            "input_pcm": "24000 Hz mono signed 16-bit; 100 ms chunks paced in real time",
            "camera": "Identical fixture pixels every 500 ms from session-ready until answer; production chooses images.",
            "stale": "A single frame captured 60 seconds before session-ready; no fresh frames.",
            "unavailable": "No frames at any point; active empty stream remains explicit.",
            "interruption": "After first answer audio and elif/mid transcript, before completion; otherwise boundary fails.",
            "reset": "Cancel active first answer; commit One moment into a controlled pending preparation, then supersede it.",
            "order": "Three repeats; alternate provider order each repeat; same assets for both.",
            "terminal_observation": "50 seconds normally; Gemini may wait for assumed PCM playout plus 10 seconds, capped at 120 seconds and a 180-second case budget. This is not measured device playback.",
        },
        "scoring": {
            "minimum_native_repeats_per_provider": 3,
            "unsupported_confident_approvals_allowed_on_traps": 0,
            "minimum_useful_correct_clear_code_fraction": 0.8,
            "answer_review": "Independent artifact-bound manual grading; transport success is not answer-quality acceptance.",
            "missing_or_failed_runs": "Remain in denominator; no quality decision with missing reviews or incomplete matrix.",
            "audio": "Save forwarded WAV; dictation requires independent audio review, not transcript formatting alone.",
        },
        "limits": [
            "No original camera-frame reproduction or physical playback claim.",
            "This suite does not replace native search latency, durable search/memory, or branded-item checks.",
        ],
    }


def render_case(case, *, font):
    """Deterministic test input, not a model-generated or corrected code sample."""
    image = Image.new("RGB", (960, 640), "#fafafa")
    draw = ImageDraw.Draw(image)
    draw.text((28, 18), "scratch.py", font=font, fill="#555555")
    for index, line in enumerate(case.code.splitlines()):
        if font.getlength(line) > 900 or 65 + index * 31 > 600:
            raise ValueError("Fixture code does not fit; refusing silent truncation")
        draw.text((28, 65 + index * 31), line, font=font, fill="#111111")
    if case.image_condition == "blurred":
        image = image.filter(ImageFilter.GaussianBlur(radius=5))
    elif case.image_condition == "cropped":
        # The last statement and lower half are genuinely outside this view.
        image = image.crop((0, 0, 960, 170))
    output = BytesIO()
    image.save(output, format="JPEG", quality=95)
    return output.getvalue(), image.size


def prepare_media(output, font_path):
    output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype(str(font_path), size=22)
    manifest = definition_manifest()
    manifest["font_sha256"] = sha256(font_path.read_bytes()).hexdigest()
    media = {}
    for case in CASES:
        jpeg, size = render_case(case, font=font)
        (output / (case.name + ".jpg")).write_bytes(jpeg)
        media[case.name] = {"image_sha256": sha256(jpeg).hexdigest(), "size": size}
        for index, prompt in enumerate(case.prompts, 1):
            (output / f"{case.name}-{index}.txt").write_text(prompt + "\n")
    (output / "bridge.txt").write_text(CONTINUITY_SPEECH["bridge"] + "\n")
    manifest["media"] = media
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True, help="A declared monospaced TrueType/OpenType font")
    args = parser.parse_args()
    prepare_media(args.output, args.font)
