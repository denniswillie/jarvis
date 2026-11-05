from __future__ import annotations

from typing import Any, Mapping
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


MAX_REFLECTION_EPISODE_REFERENCES = 30
MAX_REFLECTION_PREFERENCES = 8
MAX_REFLECTION_PROCEDURES = 8
MAX_PROCEDURE_STEPS = 24


class ReflectionValidationError(ValueError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class InferredPreferenceCandidate(_StrictModel):
    key: str = Field(min_length=1, max_length=200)
    value: str = Field(min_length=1, max_length=4_000)
    confidence: float = Field(ge=0, le=1)
    supporting_episode_ids: list[str] = Field(
        min_length=1,
        max_length=MAX_REFLECTION_EPISODE_REFERENCES,
    )


class ProcedureMemoryCandidate(_StrictModel):
    procedure_key: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=240)
    summary: str = Field(min_length=1, max_length=2_000)
    steps: list[str] = Field(min_length=1, max_length=MAX_PROCEDURE_STEPS)
    confidence: float = Field(ge=0, le=1)
    is_explicit: bool
    supporting_episode_ids: list[str] = Field(
        min_length=1,
        max_length=MAX_REFLECTION_EPISODE_REFERENCES,
    )

    @model_validator(mode="after")
    def validate_steps(self) -> ProcedureMemoryCandidate:
        if any(not step.strip() or len(step) > 1_000 for step in self.steps):
            raise ValueError("procedure steps must contain 1-1000 characters")
        return self


class ReflectionResult(_StrictModel):
    title: str = Field(min_length=1, max_length=240)
    summary: str = Field(min_length=1, max_length=2_000)
    content: str = Field(min_length=1, max_length=8_000)
    importance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    supporting_episode_ids: list[str] = Field(
        min_length=1,
        max_length=MAX_REFLECTION_EPISODE_REFERENCES,
    )
    preference_candidates: list[InferredPreferenceCandidate] = Field(
        max_length=MAX_REFLECTION_PREFERENCES
    )
    procedure_candidates: list[ProcedureMemoryCandidate] = Field(
        max_length=MAX_REFLECTION_PROCEDURES
    )


def parse_reflection_result(
    value: str | Mapping[str, Any],
    *,
    allowed_episode_ids: frozenset[UUID],
) -> ReflectionResult:
    try:
        result = (
            ReflectionResult.model_validate_json(value)
            if isinstance(value, str)
            else ReflectionResult.model_validate(value)
        )
    except (ValidationError, ValueError, TypeError) as error:
        raise ReflectionValidationError(
            "reflection output does not match the required contract"
        ) from error

    supporting_ids = _validated_episode_ids(
        result.supporting_episode_ids,
        allowed_episode_ids=allowed_episode_ids,
        label="reflection supporting episodes",
    )
    if not supporting_ids:
        raise ReflectionValidationError("reflection requires supporting episodes")
    return result


def candidate_episode_ids(
    raw_ids: list[str],
    *,
    allowed_episode_ids: frozenset[UUID],
) -> tuple[UUID, ...] | None:
    try:
        return _validated_episode_ids(
            raw_ids,
            allowed_episode_ids=allowed_episode_ids,
            label="candidate supporting episodes",
        )
    except ReflectionValidationError:
        return None


def _validated_episode_ids(
    raw_ids: list[str],
    *,
    allowed_episode_ids: frozenset[UUID],
    label: str,
) -> tuple[UUID, ...]:
    parsed: list[UUID] = []
    seen: set[UUID] = set()
    for raw_id in raw_ids:
        try:
            episode_id = UUID(raw_id)
        except (TypeError, ValueError) as error:
            raise ReflectionValidationError(f"{label} contain an invalid UUID") from error
        if episode_id not in allowed_episode_ids:
            raise ReflectionValidationError(f"{label} are outside the requested range")
        if episode_id not in seen:
            parsed.append(episode_id)
            seen.add(episode_id)
    return tuple(parsed)
