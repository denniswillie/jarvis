from __future__ import annotations

from datetime import UTC, datetime
from statistics import mean
from typing import Iterable

from .models import (
    AggregateMetrics,
    BaselineReport,
    EvalMode,
    ScenarioResult,
)


def _optional_rate(values: Iterable[bool | None]) -> float | None:
    measured = [value for value in values if value is not None]
    if not measured:
        return None
    return sum(measured) / len(measured)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_report(
    *,
    run_id: str,
    mode: EvalMode,
    suite_version: int,
    fixture_manifest_version: int,
    duration_seconds: float,
    results: list[ScenarioResult],
) -> BaselineReport:
    current_view_latencies = [
        result.metrics.response_latency_ms
        for result in results
        if result.category.value == "current_view"
        and result.metrics.response_latency_ms is not None
    ]
    description_lags = [
        result.metrics.frame_to_description_lag_ms
        for result in results
        if result.metrics.frame_to_description_lag_ms is not None
    ]
    localization_errors = [
        result.metrics.temporal_localization_error_seconds
        for result in results
        if result.metrics.temporal_localization_error_seconds is not None
    ]
    provider_calls: dict[str, int] = {}
    for result in results:
        for usage in result.observation.provider_usage:
            provider_calls[usage.provider] = provider_calls.get(usage.provider, 0) + 1
    evaluated_duration_seconds = sum(
        result.metrics.session_duration_seconds for result in results
    )
    duration_minutes = max(evaluated_duration_seconds / 60, 1 / 60)

    aggregate = AggregateMetrics(
        scenario_count=len(results),
        passed_count=sum(result.passed for result in results),
        current_view_accuracy=_optional_rate(
            result.metrics.current_view_correct for result in results
        ),
        temporal_accuracy=_optional_rate(
            result.metrics.temporal_relation_correct for result in results
        ),
        recall_accuracy=_optional_rate(result.metrics.recall_correct for result in results),
        grounding_rate=_optional_rate(
            result.metrics.expected_evidence_grounded for result in results
        ),
        mean_temporal_localization_error_seconds=(
            mean(localization_errors) if localization_errors else None
        ),
        current_view_p50_response_latency_ms=_percentile(current_view_latencies, 0.50),
        current_view_p95_response_latency_ms=_percentile(current_view_latencies, 0.95),
        frame_analysis_p50_lag_ms=_percentile(description_lags, 0.50),
        frame_analysis_p95_lag_ms=_percentile(description_lags, 0.95),
        mean_frame_to_description_lag_ms=(
            mean(description_lags) if description_lags else None
        ),
        peak_visual_backlog=max(
            (result.metrics.current_visual_backlog_peak for result in results),
            default=0,
        ),
        search_candidate_count=sum(
            result.metrics.search_candidate_count for result in results
        ),
        search_returned_description_count=sum(
            result.metrics.search_returned_description_count for result in results
        ),
        memory_level_accuracy=_optional_rate(
            result.metrics.memory_level_correct for result in results
        ),
        duplicate_interruptions=sum(
            result.metrics.duplicate_interruptions for result in results
        ),
        false_interruptions=sum(result.metrics.false_interruptions for result in results),
        provider_calls_per_minute={
            provider: count / duration_minutes
            for provider, count in sorted(provider_calls.items())
        },
        estimated_cost_usd=sum(result.metrics.estimated_cost_usd for result in results),
    )
    return BaselineReport(
        run_id=run_id,
        mode=mode,
        generated_at=datetime.now(UTC),
        suite_version=suite_version,
        fixture_manifest_version=fixture_manifest_version,
        duration_seconds=duration_seconds,
        aggregate=aggregate,
        scenarios=results,
        metadata={
            "deterministic_runs_mutate_production_state": False,
            "legacy_visual_boundary": "validate_visual_frame_event",
        },
    )
