from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


JarvisFileSource = Literal["assistant", "visual_frame", "upload", "web_fetch"]


class JarvisFile(BaseModel):
    id: UUID
    session_id: UUID | None
    filename: str
    content_type: str
    storage_bucket: str
    storage_path: str
    size_bytes: int | None
    description: str
    source: JarvisFileSource
    source_visual_frame_id: UUID | None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Runtime-only provenance, excluded from the immutable file original.
    memory_source: dict[str, Any] | None = Field(default=None, exclude=True)
    task_evidence: dict[str, Any] | None = Field(default=None, exclude=True)


class StoredFileObject(BaseModel):
    bucket: str
    path: str
    size_bytes: int
