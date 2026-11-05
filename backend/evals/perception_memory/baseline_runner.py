from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import uuid
from pathlib import Path

from .deterministic_pipeline import evaluate_deterministic_scenario
from .models import (
    BaselineReport,
    EvalMode,
    EvalScenario,
    FixtureManifest,
    LiveObservationBundle,
    ProviderUsage,
    ScenarioCategory,
    ScenarioMetrics,
    ScenarioObservation,
    ScenarioResult,
    ScenarioSuite,
)
from .report import build_report


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SCENARIOS_PATH = PACKAGE_DIR / "scenarios.json"
DEFAULT_MANIFEST_PATH = PACKAGE_DIR / "fixtures" / "manifest.json"
DEFAULT_OUTPUT_PATH = PACKAGE_DIR / "reports" / "latest.json"


def load_suite(path: Path) -> ScenarioSuite:
    return ScenarioSuite.model_validate_json(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> FixtureManifest:
    return FixtureManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_live_observations(path: Path) -> tuple[str, dict[str, ScenarioObservation]]:
    bundle = LiveObservationBundle.model_validate_json(path.read_text(encoding="utf-8"))
    observations: dict[str, ScenarioObservation] = {}
    for scenario_id, trace in bundle.scenarios.items():
        provider_events = [event for event in trace.events if event.provider]
        tool_events = [
            event.tool_name
            for event in trace.events
            if event.type == "tool.call.completed" and event.tool_name
        ]
        usage = [
            ProviderUsage(
                provider=event.provider or "unknown",
                model=event.model or "unknown",
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                estimated_cost_usd=event.estimated_cost_usd,
            )
            for event in provider_events
        ]
        observations[scenario_id] = trace.observation.model_copy(
            update={
                "tool_calls": tool_events,
                "provider": provider_events[-1].provider if provider_events else "unknown",
                "provider_usage": usage,
            }
        )
    return bundle.trace_id, observations


def _contains_required_terms(answer: str, terms: list[str]) -> bool:
    normalized = answer.casefold()
    return all(term.casefold() in normalized for term in terms)


def score_scenario(
    scenario: EvalScenario,
    observation: ScenarioObservation,
    *,
    require_response_latency: bool = True,
    require_description_lag: bool = True,
) -> ScenarioResult:
    failures: list[str] = []
    evidence = scenario.expected_evidence
    terms_correct = _contains_required_terms(observation.answer, evidence.required_terms)
    if not terms_correct:
        failures.append("required evidence terms were absent from the answer")

    # Match the entire reply, not just a disclaimer embedded in an otherwise
    # invented answer. Normalize typography, but preserve every spoken word.
    normalized_answer = " ".join(observation.answer.split()).replace("\u2019", "'")
    behavior_correct = not evidence.allowed_answer_patterns or (
        bool(normalized_answer) and any(
            re.fullmatch(pattern, normalized_answer, flags=re.IGNORECASE) is not None
            for pattern in evidence.allowed_answer_patterns
        )
    )
    if not behavior_correct:
        failures.append("answer did not match the expected response behavior")
    length_correct = (
        evidence.max_answer_words is None
        or len(normalized_answer.split()) <= evidence.max_answer_words
    )
    if not length_correct:
        failures.append("answer exceeded the scenario word limit")
    answer_correct = terms_correct and behavior_correct and length_correct

    grounded: bool | None = None
    if evidence.grounding_fixture_ids:
        grounded = set(evidence.grounding_fixture_ids).issubset(
            observation.grounded_fixture_ids
        )
        if not grounded:
            failures.append("answer was not grounded in the expected frame fixtures")

    required_tools_used = set(scenario.required_tool_calls).issubset(observation.tool_calls)
    if not required_tools_used:
        failures.append("required tool call was not observed")
    forbidden_tools_avoided = not set(scenario.forbidden_tool_calls).intersection(
        observation.tool_calls
    )
    if not forbidden_tools_avoided:
        failures.append("forbidden tool call was observed")

    if require_response_latency and observation.response_latency_ms is None:
        failures.append("response latency was not recorded")
    elif (
        observation.response_latency_ms is not None
        and observation.response_latency_ms > scenario.max_response_latency_ms
    ):
        failures.append("response latency exceeded scenario budget")
    if not require_response_latency:
        preparation_ms = observation.response_preparation_latency_ms
        if preparation_ms is None:
            failures.append("response preparation latency was not recorded")
        elif preparation_ms > scenario.max_response_latency_ms:
            failures.append("response preparation exceeded scenario budget")
    if (
        require_description_lag and scenario.input_frame_fixture_ids
        and observation.frame_to_description_lag_ms is None
    ):
        failures.append("frame-to-description lag was not recorded")

    localization_ok = True
    if evidence.max_temporal_localization_error_seconds is not None:
        localization_ok = (
            observation.temporal_localization_error_seconds is not None
            and observation.temporal_localization_error_seconds
            <= evidence.max_temporal_localization_error_seconds
        )
        if not localization_ok:
            failures.append("temporal localization error exceeded scenario budget")

    expected_interrupts = 1 if scenario.proactive_speech_expected else 0
    duplicate_interruptions = max(0, len(observation.interrupts) - expected_interrupts)
    false_interruptions = (
        len(observation.interrupts) if not scenario.proactive_speech_expected else 0
    )
    if scenario.proactive_speech_expected and not observation.interrupts:
        failures.append("expected proactive interruption was absent")
    if duplicate_interruptions:
        failures.append("duplicate proactive interruptions were observed")
    if false_interruptions:
        failures.append("unexpected proactive interruption was observed")

    memory_level_correct = observation.selected_memory_level == scenario.expected_memory_level
    if not memory_level_correct:
        failures.append("observed memory level did not match the scenario requirement")
    provider_calls = len(observation.provider_usage)
    estimated_cost = sum(item.estimated_cost_usd for item in observation.provider_usage)
    metrics = ScenarioMetrics(
        current_view_correct=(
            answer_correct if scenario.category == ScenarioCategory.CURRENT_VIEW else None
        ),
        temporal_relation_correct=(
            answer_correct and localization_ok
            if scenario.category == ScenarioCategory.TEMPORAL
            else None
        ),
        recall_correct=(
            answer_correct if scenario.category == ScenarioCategory.RECALL else None
        ),
        expected_evidence_grounded=grounded,
        required_tools_used=required_tools_used,
        forbidden_tools_avoided=forbidden_tools_avoided,
        temporal_localization_error_seconds=observation.temporal_localization_error_seconds,
        response_latency_ms=observation.response_latency_ms,
        frame_to_description_lag_ms=observation.frame_to_description_lag_ms,
        current_visual_backlog_peak=observation.current_visual_backlog_peak,
        search_candidate_count=observation.search_candidate_count,
        search_returned_description_count=observation.search_returned_description_count,
        memory_level_correct=memory_level_correct,
        session_duration_seconds=observation.session_duration_seconds,
        duplicate_interruptions=duplicate_interruptions,
        false_interruptions=false_interruptions,
        tool_call_count=len(observation.tool_calls),
        provider_call_count=provider_calls,
        estimated_cost_usd=estimated_cost,
    )
    return ScenarioResult(
        scenario_id=scenario.id,
        category=scenario.category,
        passed=not failures,
        failures=failures,
        observation=observation,
        metrics=metrics,
    )


async def _run_deterministic_suite(
    *,
    suite: ScenarioSuite,
    manifest: FixtureManifest,
    manifest_path: Path,
    simulate_legacy_gaps: bool,
    companion_quality: bool = False,
    diagnostics: dict[str, list[dict[str, object]]] | None = None,
) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for scenario in suite.scenarios:
        outcome = await evaluate_deterministic_scenario(
            scenario=scenario,
            manifest=manifest,
            manifest_path=manifest_path,
            simulate_legacy_gaps=simulate_legacy_gaps,
        )
        results.append(score_scenario(
            scenario, outcome.observation,
            require_response_latency=not companion_quality,
            require_description_lag=not companion_quality,
        ))
        if companion_quality and diagnostics is not None:
            diagnostics[scenario.id] = outcome.diagnostic_events
    return results


def run_baseline(
    *,
    suite_path: Path = DEFAULT_SCENARIOS_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    mode: EvalMode = EvalMode.DETERMINISTIC,
    live_results_path: Path | None = None,
    simulate_legacy_gaps: bool = True,
    companion_quality: bool = False,
) -> BaselineReport:
    started = time.monotonic()
    suite = load_suite(suite_path)
    manifest = load_manifest(manifest_path)
    fixtures = {entry.id for entry in manifest.entries}
    for scenario in suite.scenarios:
        missing = set(scenario.input_frame_fixture_ids) - fixtures
        if missing:
            raise ValueError(f"scenario {scenario.id} references missing fixtures: {missing}")

    live_trace_id: str | None = None
    diagnostics: dict[str, list[dict[str, object]]] = {}
    if mode == EvalMode.DETERMINISTIC:
        results = asyncio.run(
            _run_deterministic_suite(
                suite=suite,
                manifest=manifest,
                manifest_path=manifest_path,
                simulate_legacy_gaps=simulate_legacy_gaps,
                companion_quality=companion_quality,
                diagnostics=diagnostics,
            )
        )
    else:
        if live_results_path is None:
            raise ValueError("--live requires --live-results")
        live_trace_id, observations = load_live_observations(live_results_path)
        results = []
        # The frozen baseline keeps its three required tool-selection classes.
        # Shared-view trials instead require real provider evidence for every case.
        live_classes = {} if companion_quality else {
            "current_view": False, "historical_recall": False, "capture": False,
        }
        for scenario in suite.scenarios:
            observation = observations.get(scenario.id)
            if observation is None:
                raise ValueError(f"live trace missing scenario {scenario.id}")
            has_realtime = any(
                usage.provider == "openai_realtime"
                for usage in observation.provider_usage
            )
            if companion_quality and not has_realtime:
                raise ValueError(
                    f"live trace missing OpenAI Realtime events for scenario {scenario.id}"
                )
            if scenario.live_tool_choice_class and not companion_quality:
                name = scenario.live_tool_choice_class
                live_classes[name] = live_classes.get(name, False) or has_realtime
            results.append(score_scenario(
                scenario, observation, require_description_lag=not companion_quality,
            ))
        missing_classes = [name for name, present in live_classes.items() if not present]
        if missing_classes:
            raise ValueError(
                "live trace must contain OpenAI Realtime events for: "
                + ", ".join(missing_classes)
            )

    report = build_report(
        run_id=f"pm-{uuid.uuid4()}",
        mode=mode,
        suite_version=suite.version,
        fixture_manifest_version=manifest.version,
        duration_seconds=time.monotonic() - started,
        results=results,
    )
    if live_trace_id is not None:
        report.metadata["live_trace_id"] = live_trace_id
    if companion_quality:
        report.metadata["scenario_diagnostics"] = diagnostics
        report.metadata["answer_latency_measured"] = mode == EvalMode.LIVE
        report.metadata["preparation_timing_only"] = mode == EvalMode.DETERMINISTIC
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Jarvis perception/memory baseline")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS_PATH)
    parser.add_argument("--fixture-manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--live-results", type=Path)
    args = parser.parse_args()
    report = run_baseline(
        suite_path=args.scenarios,
        manifest_path=args.fixture_manifest,
        mode=EvalMode.LIVE if args.live else EvalMode.DETERMINISTIC,
        live_results_path=args.live_results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
