from __future__ import annotations

import logging
from collections.abc import Callable

import asyncpg

from app.assistant.models import (
    BehaviorProfileCommitted,
    BehaviorProfileRevisionConflict,
    BehaviorProfileUpdate,
    JarvisBehaviorProfile,
)


logger = logging.getLogger(__name__)
BehaviorProfileCommitCallback = Callable[[BehaviorProfileCommitted], None]


def _profile_from_record(record: asyncpg.Record) -> JarvisBehaviorProfile:
    return JarvisBehaviorProfile(
        owner_id=record["owner_id"],
        behavior_instructions=record["behavior_instructions"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        revision=record["revision"],
    )


class BehaviorProfileRepository:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        on_committed: BehaviorProfileCommitCallback | None = None,
    ) -> None:
        self._pool = pool
        # The runtime owns this nonblocking enqueue callback. Reconnect must
        # reload the persisted profile: this seam is not a durable event outbox.
        self._on_committed = on_committed

    async def get(self, *, owner_id: str) -> JarvisBehaviorProfile | None:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                "select owner_id, behavior_instructions, created_at, updated_at, revision "
                "from jarvis_behavior_profiles where owner_id = $1",
                owner_id,
            )
        return _profile_from_record(record) if record else None

    async def set(
        self,
        *,
        owner_id: str,
        behavior_instructions: str,
        expected_revision: int,
    ) -> JarvisBehaviorProfile:
        update = BehaviorProfileUpdate(
            behavior_instructions=behavior_instructions,
            expected_revision=expected_revision,
        )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                if update.expected_revision == 0:
                    record = await conn.fetchrow(
                        """
                        insert into jarvis_behavior_profiles
                            (owner_id, behavior_instructions, revision)
                        values ($1, $2, 1)
                        on conflict (owner_id) do nothing
                        returning owner_id, behavior_instructions, created_at, updated_at, revision
                        """,
                        owner_id,
                        update.behavior_instructions,
                    )
                else:
                    record = await conn.fetchrow(
                        """
                        update jarvis_behavior_profiles
                        set behavior_instructions = $2,
                            revision = revision + 1,
                            updated_at = clock_timestamp()
                        where owner_id = $1 and revision = $3
                        returning owner_id, behavior_instructions, created_at, updated_at, revision
                        """,
                        owner_id,
                        update.behavior_instructions,
                        update.expected_revision,
                    )
                if record is None:
                    revision = await conn.fetchval(
                        "select revision from jarvis_behavior_profiles where owner_id = $1",
                        owner_id,
                    )
                    raise BehaviorProfileRevisionConflict(update.expected_revision, revision or 0)
        # Both transaction completion and connection release precede publication.
        profile = _profile_from_record(record)
        if self._on_committed is not None:
            event = BehaviorProfileCommitted(
                owner_id=profile.owner_id,
                revision=profile.revision,
                behavior_instructions=profile.behavior_instructions,
                committed_at=profile.updated_at,
            )
            try:
                self._on_committed(event)
            except Exception:
                # A failed notification must not turn a committed save into a
                # false failure or encourage blind replay. Do not log contents.
                logger.warning("Behavior profile committed; change notification failed.")
        return profile
