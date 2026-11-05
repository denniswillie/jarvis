from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from .baseline_runner import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SCENARIOS_PATH,
    run_baseline,
)
from .companion_fixtures import write_companion_manifest
from .metrics import (
    ArchitectureEvaluationReport,
    EvaluationThresholds,
    assess_report,
    compare_to_baseline,
)
from .models import BaselineReport, EvalMode


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_PATH = PACKAGE_DIR / "reports" / "pre_memory_architecture.json"
DEFAULT_OUTPUT_PATH = PACKAGE_DIR / "reports" / "latest_architecture.json"
COMPANION_SCENARIOS_PATH = PACKAGE_DIR / "companion_quality_scenarios.json"


def run_architecture_evaluation(
    *,
    suite_path: Path = DEFAULT_SCENARIOS_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    baseline_path: Path | None = DEFAULT_BASELINE_PATH,
    thresholds: EvaluationThresholds | None = None,
    mode: EvalMode = EvalMode.DETERMINISTIC,
    live_results_path: Path | None = None,
) -> ArchitectureEvaluationReport:
    companion_quality = suite_path.resolve() == COMPANION_SCENARIOS_PATH.resolve()
    limits = thresholds or (
        EvaluationThresholds(maximum_current_view_p95_latency_ms=7_000)
        if companion_quality else EvaluationThresholds()
    )
    # Additional synthetic screens live in a temporary fixture bundle. Neither
    # the original manifest/images nor the frozen legacy report is rewritten.
    with ExitStack() as resources:
        if companion_quality and manifest_path == DEFAULT_MANIFEST_PATH:
            directory = Path(resources.enter_context(TemporaryDirectory(
                prefix="jarvis-companion-fixtures-",
            )))
            manifest_path = write_companion_manifest(directory, manifest_path)
        report = run_baseline(
            suite_path=suite_path,
            manifest_path=manifest_path,
            mode=mode,
            live_results_path=live_results_path,
            simulate_legacy_gaps=False,
            companion_quality=companion_quality,
        )
    report.metadata.update(
        {
            "architecture": "hierarchical_perception_memory",
            "raw_frame_caption_search_enabled": False,
            "failure_injection_mode": "deterministic_test_doubles",
            "validation_mode": mode.value,
            "deterministic_answers_are_fixtures": mode == EvalMode.DETERMINISTIC,
            "reasoning_quality_requires_live_trace": True,
        }
    )
    baseline = (
        BaselineReport.model_validate_json(baseline_path.read_text(encoding="utf-8"))
        if not companion_quality and baseline_path is not None and baseline_path.exists()
        else None
    )
    return ArchitectureEvaluationReport(
        report=report,
        thresholds=limits,
        assessment=assess_report(report, limits),
        baseline_comparison=(
            compare_to_baseline(report, baseline) if baseline is not None else None
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the post-cutover Jarvis perception/memory architecture"
    )
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS_PATH)
    parser.add_argument("--fixture-manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument(
        "--companion-quality", action="store_true",
        help="Run the separate shared-view suite without comparing unrelated legacy scenarios.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Score an explicitly captured live provider trace.",
    )
    parser.add_argument(
        "--live-results",
        type=Path,
        help="LiveObservationBundle JSON captured from a disposable live run.",
    )
    args = parser.parse_args()
    if args.live and args.live_results is None:
        parser.error("--live requires --live-results")
    if not args.live and args.live_results is not None:
        parser.error("--live-results requires --live")

    evaluation = run_architecture_evaluation(
        suite_path=COMPANION_SCENARIOS_PATH if args.companion_quality else args.scenarios,
        manifest_path=args.fixture_manifest,
        baseline_path=None if args.no_baseline or args.companion_quality else args.baseline,
        mode=EvalMode.LIVE if args.live else EvalMode.DETERMINISTIC,
        live_results_path=args.live_results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": evaluation.assessment.passed,
                "baseline_passed": (
                    evaluation.baseline_comparison.passed
                    if evaluation.baseline_comparison is not None
                    else None
                ),
                "scenario_count": evaluation.report.aggregate.scenario_count,
                "passed_count": evaluation.report.aggregate.passed_count,
                "mode": evaluation.report.mode.value,
                "trace_id": evaluation.report.metadata.get("live_trace_id"),
            },
            sort_keys=True,
        )
    )
    if not evaluation.assessment.passed or (
        evaluation.baseline_comparison is not None
        and not evaluation.baseline_comparison.passed
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
