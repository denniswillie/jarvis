"""Persistent owner preferences, independent of sessions and model delivery."""
from __future__ import annotations

from typing import Any

from app.assistant.models import BehaviorProfileUpdate, profile_payload
from app.assistant.repository import BehaviorProfileRepository


class BehaviorProfileService:
    def __init__(self, repository: BehaviorProfileRepository) -> None:
        self.repository = repository

    async def get(self, *, owner_id: str) -> dict[str, Any]:
        return profile_payload(await self.repository.get(owner_id=owner_id), owner_id=owner_id)

    async def update(self, *, owner_id: str, update: BehaviorProfileUpdate) -> dict[str, Any]:
        profile = await self.repository.set(
            owner_id=owner_id,
            behavior_instructions=update.behavior_instructions,
            expected_revision=update.expected_revision,
        )
        return profile_payload(profile, owner_id=owner_id)
