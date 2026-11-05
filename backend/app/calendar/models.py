from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class GoogleOAuthToken(BaseModel):
    owner_id: str
    access_token: str
    refresh_token: str | None
    token_type: str | None
    scopes: list[str] = Field(default_factory=list)
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime


class GoogleOAuthStartResponse(BaseModel):
    authorization_url: str
    expires_in_seconds: int


class GoogleCalendarStatusResponse(BaseModel):
    connected: bool
    expires_at: datetime | None = None
    scopes: list[str] = Field(default_factory=list)
