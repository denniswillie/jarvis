from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .models import BaselineReport


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationThresholds(StrictModel):
    minimum_pass_rate: float = Field(default=1.0, ge=0, le=1)
    minimum_measured_accuracy: float = Field(default=1.0, ge=0, le=1)
    maximum_duplicate_interruptions: int = Field(default=0, ge=0)
    maximum_false_interruptions: int = Field(default=0, ge=0)
    maximum_visual_backlog: int = Field(default=8, ge=0)
    maximum_current_view_p95_latency_ms: float = Field(default=3_000, gt=0)
    maximum_frame_analysis_p95_lag_ms: float = Field(default=1_000, gt=0)


class MetricGate(StrictModel):
    name: str
    passed: bool
    actual: float | int | None
    expected: str


class EvaluationAssessment(StrictModel):
    passed: bool
    gates: list[MetricGate]


class ArchitectureEvaluationReport(StrictModel):
    report: BaselineReport
    thresholds: EvaluationThresholds
    assessment: EvaluationAssessment
    baseline_comparison: EvaluationAssessment | None = None


def _minimum_gate(name: str, actual: float | None, minimum: float) -> MetricGate:
    return MetricGate(
        name=name,
        passed=actual is None or actual >= minimum,
        actual=actual,
        expected=f">= {minimum:g} when measured",
    )


def _maximum_gate(
    name: str,
    actual: float | int | None,
    maximum: float | int,
) -> MetricGate:
    return MetricGate(
        name=name,
        passed=actual is None or actual <= maximum,
        actual=actual,
        expected=f"<= {maximum:g} when measured",
    )


def assess_report(
    report: BaselineReport,
    thresholds: EvaluationThresholds | None = None,
) -> EvaluationAssessment:
    limits = thresholds or EvaluationThresholds()
    aggregate = report.aggregate
    pass_rate = (
        aggregate.passed_count / aggregate.scenario_count
        if aggregate.scenario_count
        else 0.0
    )
    gates = [
        _minimum_gate("scenario_pass_rate", pass_rate, limits.minimum_pass_rate),
        _minimum_gate(
            "current_view_accuracy",
            aggregate.current_view_accuracy,
            limits.minimum_measured_accuracy,
        ),
        _minimum_gate(
            "temporal_accuracy",
            aggregate.temporal_accuracy,
            limits.minimum_measured_accuracy,
        ),
        _minimum_gate(
            "recall_accuracy",
            aggregate.recall_accuracy,
            limits.minimum_measured_accuracy,
        ),
        _minimum_gate(
            "grounding_rate",
            aggregate.grounding_rate,
            limits.minimum_measured_accuracy,
        ),
        _minimum_gate(
            "memory_level_accuracy",
            aggregate.memory_level_accuracy,
            limits.minimum_measured_accuracy,
        ),
        _maximum_gate(
            "duplicate_interruptions",
            aggregate.duplicate_interruptions,
            limits.maximum_duplicate_interruptions,
        ),
        _maximum_gate(
            "false_interruptions",
            aggregate.false_interruptions,
            limits.maximum_false_interruptions,
        ),
        _maximum_gate(
            "peak_visual_backlog",
            aggregate.peak_visual_backlog,
            limits.maximum_visual_backlog,
        ),
        _maximum_gate(
            "current_view_p95_response_latency_ms",
            aggregate.current_view_p95_response_latency_ms,
            limits.maximum_current_view_p95_latency_ms,
        ),
        _maximum_gate(
            "frame_analysis_p95_lag_ms",
            aggregate.frame_analysis_p95_lag_ms,
            limits.maximum_frame_analysis_p95_lag_ms,
        ),
    ]
    return EvaluationAssessment(
        passed=all(gate.passed for gate in gates),
        gates=gates,
    )


def compare_to_baseline(
    current: BaselineReport,
    baseline: BaselineReport,
) -> EvaluationAssessment:
    current_aggregate = current.aggregate
    baseline_aggregate = baseline.aggregate
    gates = [
        MetricGate(
            name="passed_scenarios_not_regressed",
            passed=current_aggregate.passed_count >= baseline_aggregate.passed_count,
            actual=current_aggregate.passed_count,
            expected=f">= {baseline_aggregate.passed_count}",
        )
    ]
    for field_name in (
        "current_view_accuracy",
        "temporal_accuracy",
        "recall_accuracy",
        "grounding_rate",
        "memory_level_accuracy",
    ):
        actual = getattr(current_aggregate, field_name)
        previous = getattr(baseline_aggregate, field_name)
        gates.append(
            MetricGate(
                name=f"{field_name}_not_regressed",
                passed=(
                    previous is None
                    or (actual is not None and actual + 1e-9 >= previous)
                ),
                actual=actual,
                expected=(
                    "measured" if previous is None else f">= {previous:.6g}"
                ),
            )
        )
    for field_name in ("duplicate_interruptions", "false_interruptions"):
        actual = getattr(current_aggregate, field_name)
        previous = getattr(baseline_aggregate, field_name)
        gates.append(
            MetricGate(
                name=f"{field_name}_not_regressed",
                passed=actual <= previous,
                actual=actual,
                expected=f"<= {previous}",
            )
        )
    return EvaluationAssessment(
        passed=all(gate.passed for gate in gates),
        gates=gates,
    )
