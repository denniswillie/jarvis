"""Compact current understanding, not a second history or reminder authority."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class HumanFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = Field(min_length=1, max_length=600)
    evidence_at: AwareDatetime
    basis: Literal["observed", "user_report", "inferred"]
    sources: tuple[str, ...] = Field(min_length=1, max_length=8)


class HumanActivity(HumanFact):
    status: Literal["active", "interrupted"]


class StatedIntention(HumanFact):
    basis: Literal["user_report"] = "user_report"


class HumanStateContent(BaseModel):
    """Model-authored content. Revision and generation time are not model fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    setting: HumanFact | None = None
    activities: tuple[HumanActivity, ...] = Field(default=(), max_length=8)
    stated_intentions: tuple[StatedIntention, ...] = Field(default=(), max_length=8)
    relevant_details: tuple[HumanFact, ...] = Field(default=(), max_length=12)
    uncertainties: tuple[HumanFact, ...] = Field(default=(), max_length=8)
    # Advisory context only; Voice handles conversational silence.
    availability: Literal["open", "focused", "do_not_disturb"] = "open"


class HumanState(HumanStateContent):
    revision: int = Field(ge=1)
    generated_at: AwareDatetime
    observed_through: AwareDatetime
    incorporated_evidence_ids: tuple[str, ...] = Field(default=(), max_length=64)

    def content(self) -> HumanStateContent:
        return HumanStateContent.model_validate(self.model_dump(include=set(HumanStateContent.model_fields)))

    def context(self, now: datetime, *, stale_after: timedelta = timedelta(minutes=2)) -> dict:
        """Expose age without re-dating evidence or asserting yesterday is today."""
        if now.tzinfo is None or stale_after.total_seconds() <= 0:
            raise ValueError("context requires an aware clock and positive freshness interval")
        result = self.model_dump(mode="json")
        result["prior_context_only"] = True
        result["stale"] = now - self.observed_through >= stale_after
        for name in ("setting", "activities", "stated_intentions", "relevant_details", "uncertainties"):
            facts = (getattr(self, name),) if name == "setting" else getattr(self, name)
            rendered = []
            for fact in facts:
                if fact is not None:
                    rendered.append({**fact.model_dump(mode="json"),
                                     "stale": now - fact.evidence_at >= stale_after})
            result[name] = (rendered[0] if rendered else None) if name == "setting" else rendered
        return result


def stamp_human_state(previous: HumanState | None, content: HumanStateContent, *,
                      observed_through: datetime, generated_at: datetime,
                      evidence_ids: tuple[str, ...]) -> HumanState | None:
    """Server-side stamping. Main must still CAS the expected human revision."""
    if generated_at.tzinfo is None or observed_through.tzinfo is None or generated_at < observed_through:
        raise ValueError("invalid human-state observation/generation time")
    if previous is not None:
        if observed_through < previous.observed_through or content == previous.content():
            return None
    seen = dict.fromkeys((*(previous.incorporated_evidence_ids if previous else ()), *evidence_ids))
    return HumanState(**content.model_dump(), revision=1 if previous is None else previous.revision + 1,
                      generated_at=generated_at, observed_through=observed_through,
                      incorporated_evidence_ids=tuple(seen)[-64:])
