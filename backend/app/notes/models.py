from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class JarvisNote(BaseModel):
    id: UUID
    session_id: UUID | None
    title: str
    body: str
    source: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Runtime-only provenance; never persist these fields inside the original note.
    memory_source: dict[str, Any] | None = Field(default=None, exclude=True)
    task_evidence: dict[str, Any] | None = Field(default=None, exclude=True)
