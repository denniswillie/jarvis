from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


MAX_BEHAVIOR_INSTRUCTIONS_LENGTH = 12_000


class BehaviorProfileUpdate(BaseModel):
    """The same compare-and-set request for the app and voice tools."""

    model_config = ConfigDict(extra="forbid", strict=True)

    behavior_instructions: str = Field(max_length=MAX_BEHAVIOR_INSTRUCTIONS_LENGTH)
    expected_revision: int = Field(ge=0, le=9_223_372_036_854_775_806)

    @field_validator("behavior_instructions", mode="before")
    @classmethod
    def trim_instructions(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


@dataclass(frozen=True)
class JarvisBehaviorProfile:
    owner_id: str
    behavior_instructions: str
    created_at: datetime
    updated_at: datetime
    revision: int = 1


def profile_payload(
    profile: JarvisBehaviorProfile | None, *, owner_id: str
) -> dict[str, Any]:
    return {
        "owner_id": owner_id,
        "behavior_instructions": profile.behavior_instructions if profile else "",
        "revision": profile.revision if profile else 0,
        "created_at": profile.created_at.isoformat() if profile else None,
        "updated_at": profile.updated_at.isoformat() if profile else None,
    }


class BehaviorProfileRevisionConflict(Exception):
    """A rejected stale write, never an instruction to retry automatically."""

    def __init__(self, expected_revision: int, current_revision: int) -> None:
        super().__init__("The behavior profile changed. Read it again before editing.")
        self.expected_revision = expected_revision
        self.current_revision = current_revision

    def error_payload(self) -> dict[str, Any]:
        return {
            "code": "behavior_profile_revision_conflict",
            "message": str(self),
            "details": {
                "expected_revision": self.expected_revision,
                "current_revision": self.current_revision,
            },
        }


@dataclass(frozen=True)
class BehaviorProfileCommitted:
    """Durable profile change, not proof of downstream instruction consumption."""

    owner_id: str
    revision: int
    behavior_instructions: str
    committed_at: datetime

    def as_payload(self) -> dict[str, Any]:
        return {
            "type": "behavior_profile.committed",
            "event_id": f"behavior-profile:{self.owner_id}:{self.revision}",
            "owner_id": self.owner_id,
            "revision": self.revision,
            "behavior_instructions": self.behavior_instructions,
            "committed_at": self.committed_at.isoformat(),
        }
