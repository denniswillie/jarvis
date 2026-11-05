from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class JarvisPreference(BaseModel):
    id: UUID
    key: str
    value: str
    source: str
    confidence: float = 1.0
    evidence_episode_ids: tuple[UUID, ...] = ()
    created_at: datetime
    updated_at: datetime


class InferredPreferenceWriteResult(BaseModel):
    preference: JarvisPreference
    applied: bool
