from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalMode(StrEnum):
    DETERMINISTIC = "deterministic"
    LIVE = "live"


class ScenarioCategory(StrEnum):
    CURRENT_VIEW = "current_view"
    TEMPORAL = "temporal"
    RECALL = "recall"
    INTERRUPTION = "interruption"
    RESILIENCE = "resilience"
    MEMORY_MUTATION = "memory_mutation"
    MIXED = "mixed"


class MemoryLevel(StrEnum):
    NONE = "none"
    SENSORY = "sensory"
    VISUAL_DESCRIPTION = "visual_description"
    SESSION = "session"
    PROSPECTIVE = "prospective"
    DURABLE_TOOL = "durable_tool"


class ScenarioEventType(StrEnum):
    EXPIRE_RAW_IMAGES = "expire_raw_images"
    CLOSE_SESSION = "close_session"
    OPEN_SESSION = "open_session"


class UserUtterance(StrictModel):
    at_seconds: float = Field(ge=0)
    text: str = Field(min_length=1)


class ScenarioEvent(StrictModel):
    at_seconds: float = Field(ge=0)
    type: ScenarioEventType


class ExpectedEvidence(StrictModel):
    required_terms: list[str] = Field(default_factory=list)
    grounding_fixture_ids: list[str] = Field(default_factory=list)
    max_temporal_localization_error_seconds: float | None = Field(default=None, ge=0)
    # Optional, evaluation-only full-response checks; legacy cases remain unchanged.
    allowed_answer_patterns: list[str] = Field(default_factory=list)
    max_answer_words: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_answer_patterns(self) -> ExpectedEvidence:
        for pattern in self.allowed_answer_patterns:
            try:
                compiled = re.compile(pattern, flags=re.IGNORECASE)
            except re.error as exc:
                raise ValueError("allowed_answer_patterns contains an invalid regex") from exc
            if not pattern.strip() or compiled.fullmatch("") is not None:
                raise ValueError("allowed_answer_patterns must not accept an empty answer")
        return self


class ProviderUsage(StrictModel):
    provider: str
    model: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0)


class ScenarioObservation(StrictModel):
    answer: str = ""
    tool_calls: list[str] = Field(default_factory=list)
    interrupts: list[str] = Field(default_factory=list)
    grounded_fixture_ids: list[str] = Field(default_factory=list)
    frame_descriptions: list[str] = Field(default_factory=list)
    observed_event_time: datetime | None = None
    response_latency_ms: float | None = Field(default=None, ge=0)
    # Deterministic transport timing is not time to a real spoken answer.
    response_preparation_latency_ms: float | None = Field(default=None, ge=0)
    frame_to_description_lag_ms: float | None = Field(default=None, ge=0)
    temporal_localization_error_seconds: float | None = Field(default=None, ge=0)
    current_visual_backlog_peak: int = Field(default=0, ge=0)
    search_candidate_count: int = Field(default=0, ge=0)
    search_returned_description_count: int = Field(default=0, ge=0)
    selected_memory_level: MemoryLevel | None = None
    session_duration_seconds: float = Field(default=0, ge=0)
    session_restart_count: int = Field(default=0, ge=0)
    raw_image_expired: bool = False
    provider: str = "mock"
    provider_usage: list[ProviderUsage] = Field(default_factory=list)


class EvalScenario(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str = Field(min_length=1)
    category: ScenarioCategory
    input_frame_fixture_ids: list[str] = Field(default_factory=list)
    capture_timestamps: list[datetime] = Field(default_factory=list)
    user_utterances: list[UserUtterance] = Field(default_factory=list)
    events: list[ScenarioEvent] = Field(default_factory=list)
    analyzer_descriptions: list[str] = Field(default_factory=list)
    pixel_detail_intentionally_omitted: bool = False
    expected_evidence: ExpectedEvidence
    required_tool_calls: list[str] = Field(default_factory=list)
    forbidden_tool_calls: list[str] = Field(default_factory=list)
    expected_memory_level: MemoryLevel
    max_response_latency_ms: float = Field(gt=0)
    proactive_speech_expected: bool = False
    live_tool_choice_class: str | None = None
    live_test_instructions: str = Field(min_length=1)
    deterministic_observation: ScenarioObservation

    @model_validator(mode="after")
    def validate_frame_timestamps(self) -> EvalScenario:
        if len(self.capture_timestamps) != len(self.input_frame_fixture_ids):
            raise ValueError("capture_timestamps must align one-to-one with input frames")
        if self.capture_timestamps != sorted(self.capture_timestamps):
            raise ValueError("capture_timestamps must be ordered")
        if self.analyzer_descriptions and len(self.analyzer_descriptions) != len(
            self.input_frame_fixture_ids
        ):
            raise ValueError("analyzer_descriptions must align one-to-one with input frames")
        if self.events != sorted(self.events, key=lambda event: event.at_seconds):
            raise ValueError("scenario events must be ordered")
        return self


class ScenarioSuite(StrictModel):
    version: int = Field(ge=1)
    scenarios: list[EvalScenario] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> ScenarioSuite:
        ids = [scenario.id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario ids must be unique")
        return self


class FixtureManifestEntry(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    description: str = Field(min_length=1)


class FixtureManifest(StrictModel):
    version: int = Field(ge=1)
    generator: str
    entries: list[FixtureManifestEntry]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> FixtureManifest:
        ids = [entry.id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("fixture ids must be unique")
        return self


class ScenarioMetrics(StrictModel):
    current_view_correct: bool | None = None
    temporal_relation_correct: bool | None = None
    recall_correct: bool | None = None
    expected_evidence_grounded: bool | None
    required_tools_used: bool
    forbidden_tools_avoided: bool
    temporal_localization_error_seconds: float | None = None
    response_latency_ms: float | None = None
    frame_to_description_lag_ms: float | None = None
    current_visual_backlog_peak: int = Field(default=0, ge=0)
    search_candidate_count: int = Field(default=0, ge=0)
    search_returned_description_count: int = Field(default=0, ge=0)
    memory_level_correct: bool
    session_duration_seconds: float = Field(default=0, ge=0)
    duplicate_interruptions: int = Field(default=0, ge=0)
    false_interruptions: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    provider_call_count: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0)


class ScenarioResult(StrictModel):
    scenario_id: str
    category: ScenarioCategory
    passed: bool
    failures: list[str] = Field(default_factory=list)
    observation: ScenarioObservation
    metrics: ScenarioMetrics


class AggregateMetrics(StrictModel):
    scenario_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    current_view_accuracy: float | None = None
    temporal_accuracy: float | None = None
    recall_accuracy: float | None = None
    grounding_rate: float | None = None
    mean_temporal_localization_error_seconds: float | None = None
    current_view_p50_response_latency_ms: float | None = None
    current_view_p95_response_latency_ms: float | None = None
    frame_analysis_p50_lag_ms: float | None = None
    frame_analysis_p95_lag_ms: float | None = None
    mean_frame_to_description_lag_ms: float | None = None
    peak_visual_backlog: int = Field(default=0, ge=0)
    search_candidate_count: int = Field(default=0, ge=0)
    search_returned_description_count: int = Field(default=0, ge=0)
    memory_level_accuracy: float | None = None
    duplicate_interruptions: int = Field(default=0, ge=0)
    false_interruptions: int = Field(default=0, ge=0)
    provider_calls_per_minute: dict[str, float] = Field(default_factory=dict)
    estimated_cost_usd: float = Field(default=0, ge=0)


class BaselineReport(StrictModel):
    run_id: str
    mode: EvalMode
    generated_at: datetime
    suite_version: int
    fixture_manifest_version: int
    duration_seconds: float = Field(ge=0)
    aggregate: AggregateMetrics
    scenarios: list[ScenarioResult]
    metadata: dict[str, Any] = Field(default_factory=dict)


class LiveTraceEvent(StrictModel):
    type: str = Field(min_length=1)
    occurred_at: datetime
    provider: str | None = None
    model: str | None = None
    tool_name: str | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)


class LiveScenarioTrace(StrictModel):
    observation: ScenarioObservation
    events: list[LiveTraceEvent] = Field(min_length=1)


class LiveObservationBundle(StrictModel):
    version: int = Field(ge=1)
    trace_id: str = Field(min_length=1)
    recorded_at: datetime
    scenarios: dict[str, LiveScenarioTrace]
