from __future__ import annotations

import json
from contextlib import asynccontextmanager
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg
from app.perception.applicability import effect_connection

from app.perception.models import (
    ObservationFrameLink,
    OwnerWorkingMemoryState,
    SessionWorkingMemoryState,
    VisualObservation,
    WorkingMemoryWriteResult,
    deserialize_entities,
    deserialize_keyframe_candidates,
    deserialize_state_changes,
    deserialize_visible_text,
    deserialize_working_memory_state,
    serialize_entities,
    serialize_keyframe_candidates,
    serialize_state_changes,
    serialize_visible_text,
    serialize_working_memory_state,
)


class PerceptionSessionMissingError(Exception):
    pass


class WorkingMemoryTargetMissingError(Exception):
    pass


class PerceptionRepository:
    def __init__(self, pool: asyncpg.Pool, *, observation_publication=None) -> None:
        self._pool = pool
        self.observation_publication = observation_publication

    @asynccontextmanager
    async def _observation_transaction(self, owner_id):
        if self.observation_publication is not None:
            async with self.observation_publication.transaction(owner_id) as connection:
                yield connection
        else:
            async with self._pool.acquire() as connection:
                async with connection.transaction():
                    yield connection

    async def rank_retrieval_visible_text_events(
        self,
        *,
        owner_id: str,
        session_id: UUID | None,
        start_time: datetime,
        end_time: datetime,
        query: str,
        limit: int,
    ) -> list[UUID]:
        normalized_query = query.strip().lower()
        if not normalized_query:
            return []
        if end_time <= start_time:
            raise ValueError("end_time must be after start_time")
        if isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be positive")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select event.id, max(evidence.occurred_at) as latest_match
                from experience_events event
                join experience_event_evidence evidence
                  on evidence.event_id = event.id
                 and evidence.evidence_type = 'observation'
                join perception_observations observation
                  on observation.id = evidence.evidence_ref_id
                 and observation.owner_id = event.owner_id
                where event.owner_id = $1
                  and ($2::uuid is null or event.session_id = $2)
                  and event.started_at >= $3
                  and event.started_at < $4
                  and position($5 in lower(observation.visible_text::text)) > 0
                group by event.id
                order by latest_match desc, event.id desc
                limit $6
                """,
                owner_id.strip(),
                session_id,
                start_time,
                end_time,
                normalized_query,
                limit,
            )
        return [row["id"] for row in rows]

    async def insert_observation(
        self,
        *,
        observation: VisualObservation,
        frame_links: Sequence[ObservationFrameLink],
        analysis_input=None,
    ) -> VisualObservation:
        links = _validate_frame_links(observation, frame_links)
        async with self._observation_transaction(observation.owner_id) as conn:
            row = await conn.fetchrow(
                """
                insert into perception_observations (
                    id,
                    owner_id,
                    session_id,
                    source,
                    started_at,
                    ended_at,
                    primary_visual_frame_id,
                    summary,
                    scene,
                    activity,
                    scene_confidence,
                    activity_confidence,
                    entities,
                    visible_text,
                    state_changes,
                    memory_tags,
                    keyframe_candidates,
                    interrupt_candidate,
                    starts_new_event,
                    boundary_reason,
                    confidence,
                    analyzer_provider,
                    analyzer_model,
                    analyzer_contract_version,
                    compacted_at,
                    created_at
                )
                select
                    $1, $2, $3, $4, $5, $6, $7, $8,
                    $9, $10, $11, $12, $13::jsonb, $14::jsonb, $15::jsonb,
                    $16::text[], $17::jsonb, $18::jsonb, $19, $20,
                    $21, $22, $23, $24, $25, $26
                from jarvis_sessions
                where id = $3 and owner_id = $2
                returning *
                """,
                observation.id,
                observation.owner_id,
                observation.session_id,
                observation.source,
                observation.started_at,
                observation.ended_at,
                observation.primary_visual_frame_id,
                observation.summary,
                observation.scene,
                observation.activity,
                observation.scene_confidence,
                observation.activity_confidence,
                json.dumps(serialize_entities(observation.entities)),
                json.dumps(serialize_visible_text(observation.visible_text)),
                json.dumps(serialize_state_changes(observation.state_changes)),
                list(observation.memory_tags),
                json.dumps(serialize_keyframe_candidates(observation.keyframe_candidates)),
                json.dumps(dict(observation.interrupt_candidate))
                if observation.interrupt_candidate is not None
                else None,
                observation.starts_new_event,
                observation.boundary_reason,
                observation.confidence,
                observation.analyzer_provider,
                observation.analyzer_model,
                observation.analyzer_contract_version,
                observation.compacted_at,
                observation.created_at,
            )
            if row is None:
                raise PerceptionSessionMissingError()

            if links:
                await conn.executemany(
                    """
                    insert into perception_observation_frames (
                        observation_id,
                        visual_frame_id,
                        ordinal,
                        role
                    )
                    values ($1, $2, $3, $4)
                    """,
                    [
                        (
                            observation.id,
                            link.visual_frame_id,
                            link.ordinal,
                            link.role,
                        )
                        for link in links
                    ],
                )
            if self.observation_publication is not None:
                await self.observation_publication.publish(conn, observation, links, analysis_input=analysis_input)
        return _observation_from_row(row)

    async def list_session_observations(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        limit: int = 100,
    ) -> list[VisualObservation]:
        if limit < 1:
            raise ValueError("limit must be positive")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select *
                from perception_observations
                where owner_id = $1 and session_id = $2
                order by started_at asc, id asc
                limit $3
                """,
                owner_id,
                session_id,
                limit,
            )
        return [_observation_from_row(row) for row in rows]

    async def list_observations_by_ids(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        observation_ids: tuple[UUID, ...],
    ) -> list[VisualObservation]:
        unique_ids = tuple(dict.fromkeys(observation_ids))
        if not unique_ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                select *
                from perception_observations
                where owner_id = $1
                  and session_id = $2
                  and id = any($3::uuid[])
                order by started_at asc, id asc
                """,
                owner_id,
                session_id,
                list(unique_ids),
            )
        if len(rows) != len(unique_ids):
            raise ValueError("one or more consolidation observations are missing")
        return [_observation_from_row(row) for row in rows]

    async def get_session_working_memory(
        self,
        *,
        owner_id: str,
        session_id: UUID,
    ) -> SessionWorkingMemoryState | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                select *
                from jarvis_session_working_memory
                where owner_id = $1 and session_id = $2
                """,
                owner_id,
                session_id,
            )
        return _session_working_memory_from_row(row) if row else None

    async def set_latest_experience_event_id(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
        expires_at: datetime,
    ) -> SessionWorkingMemoryState | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                insert into jarvis_session_working_memory (
                    session_id,
                    owner_id,
                    revision,
                    state,
                    expires_at
                )
                select
                    id,
                    owner_id,
                    1,
                    jsonb_build_object('latest_experience_event_id', $3::uuid::text),
                    $4
                from jarvis_sessions
                where owner_id = $1 and id = $2
                on conflict (session_id) do update
                set revision = jarvis_session_working_memory.revision + 1,
                    state = jsonb_set(
                        jarvis_session_working_memory.state,
                        '{latest_experience_event_id}',
                        to_jsonb($3::uuid::text),
                        true
                    ),
                    expires_at = greatest(
                        jarvis_session_working_memory.expires_at,
                        excluded.expires_at
                    ),
                    updated_at = now()
                returning *
                """,
                owner_id,
                session_id,
                event_id,
                expires_at,
            )
        return _session_working_memory_from_row(row) if row else None

    async def compare_and_swap_session_working_memory(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        expected_revision: int,
        observation_id: UUID,
        state: Mapping[str, Any],
        expires_at: datetime,
        contract_version: int = 1,
        effect_guard=None,
    ) -> WorkingMemoryWriteResult:
        if expected_revision < 0:
            raise ValueError("expected_revision cannot be negative")
        encoded_state = json.dumps(
            serialize_working_memory_state(state, contract_version=contract_version)
        )
        async with effect_connection(self._pool, effect_guard) as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    insert into jarvis_session_working_memory (
                        session_id,
                        owner_id,
                        revision,
                        last_observation_id,
                        last_observation_ended_at,
                        state,
                        expires_at
                    )
                    select
                        sessions.id,
                        sessions.owner_id,
                        1,
                        observations.id,
                        observations.ended_at,
                        $5::jsonb,
                        $6
                    from jarvis_sessions sessions
                    join perception_observations observations
                      on observations.id = $3
                     and observations.owner_id = sessions.owner_id
                     and observations.session_id = sessions.id
                    where sessions.owner_id = $1
                      and sessions.id = $2
                      and (
                          $4 = 0
                          or exists (
                              select 1
                              from jarvis_session_working_memory current_state
                              where current_state.owner_id = sessions.owner_id
                                and current_state.session_id = sessions.id
                          )
                      )
                    on conflict (session_id) do update
                    set
                        revision = jarvis_session_working_memory.revision + 1,
                        last_observation_id = excluded.last_observation_id,
                        last_observation_ended_at = excluded.last_observation_ended_at,
                        state = excluded.state,
                        expires_at = excluded.expires_at,
                        updated_at = now()
                    where jarvis_session_working_memory.owner_id = excluded.owner_id
                      and jarvis_session_working_memory.revision = $4
                      and (
                          jarvis_session_working_memory.last_observation_ended_at is null
                          or excluded.last_observation_ended_at
                             > jarvis_session_working_memory.last_observation_ended_at
                      )
                    returning *
                    """,
                    owner_id,
                    session_id,
                    observation_id,
                    expected_revision,
                    encoded_state,
                    expires_at,
                )
                if row is not None:
                    return WorkingMemoryWriteResult(
                        applied=True,
                        state=_session_working_memory_from_row(row),
                    )

                current = await conn.fetchrow(
                    """
                    select *
                    from jarvis_session_working_memory
                    where owner_id = $1 and session_id = $2
                    """,
                    owner_id,
                    session_id,
                )
                if current is None:
                    raise WorkingMemoryTargetMissingError()
                return WorkingMemoryWriteResult(
                    applied=False,
                    state=_session_working_memory_from_row(current),
                )

    async def get_owner_working_memory(
        self,
        *,
        owner_id: str,
    ) -> OwnerWorkingMemoryState | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                select *
                from jarvis_owner_working_memory
                where owner_id = $1
                """,
                owner_id,
            )
        return _owner_working_memory_from_row(row) if row else None

    async def compare_and_swap_owner_working_memory(
        self,
        *,
        owner_id: str,
        expected_revision: int,
        state: Mapping[str, Any],
        expires_at: datetime | None = None,
        contract_version: int = 1,
    ) -> WorkingMemoryWriteResult:
        if expected_revision < 0:
            raise ValueError("expected_revision cannot be negative")
        encoded_state = json.dumps(
            serialize_working_memory_state(state, contract_version=contract_version)
        )
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    insert into jarvis_owner_working_memory (
                        owner_id,
                        revision,
                        state,
                        expires_at
                    )
                    select $1, 1, $3::jsonb, $4
                    where $2 = 0
                       or exists (
                           select 1
                           from jarvis_owner_working_memory current_state
                           where current_state.owner_id = $1
                       )
                    on conflict (owner_id) do update
                    set
                        revision = jarvis_owner_working_memory.revision + 1,
                        state = excluded.state,
                        expires_at = excluded.expires_at,
                        updated_at = now()
                    where jarvis_owner_working_memory.revision = $2
                    returning *
                    """,
                    owner_id,
                    expected_revision,
                    encoded_state,
                    expires_at,
                )
                if row is not None:
                    return WorkingMemoryWriteResult(
                        applied=True,
                        state=_owner_working_memory_from_row(row),
                    )

                current = await conn.fetchrow(
                    """
                    select *
                    from jarvis_owner_working_memory
                    where owner_id = $1
                    """,
                    owner_id,
                )
                if current is None:
                    raise WorkingMemoryTargetMissingError()
                return WorkingMemoryWriteResult(
                    applied=False,
                    state=_owner_working_memory_from_row(current),
                )


def _validate_frame_links(
    observation: VisualObservation,
    frame_links: Sequence[ObservationFrameLink],
) -> tuple[ObservationFrameLink, ...]:
    links = tuple(sorted(frame_links, key=lambda link: link.ordinal))
    if len({link.ordinal for link in links}) != len(links):
        raise ValueError("observation frame ordinals must be unique")
    if len({link.visual_frame_id for link in links}) != len(links):
        raise ValueError("an observation cannot link the same frame twice")
    primary_links = [link for link in links if link.role == "primary"]
    if observation.primary_visual_frame_id is None:
        if primary_links:
            raise ValueError("primary frame link requires primary_visual_frame_id")
    elif len(primary_links) != 1 or (
        primary_links[0].visual_frame_id != observation.primary_visual_frame_id
    ):
        raise ValueError("primary frame link must match primary_visual_frame_id")
    return links


def _observation_from_row(row: Mapping[str, Any]) -> VisualObservation:
    return VisualObservation(
        id=row["id"],
        owner_id=row["owner_id"],
        session_id=row["session_id"],
        source=row["source"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        primary_visual_frame_id=row["primary_visual_frame_id"],
        summary=row["summary"],
        scene=row["scene"],
        activity=row["activity"],
        scene_confidence=(
            float(row["scene_confidence"])
            if row.get("scene_confidence") is not None
            else None
        ),
        activity_confidence=(
            float(row["activity_confidence"])
            if row.get("activity_confidence") is not None
            else None
        ),
        entities=deserialize_entities(_decode_json(row["entities"])),
        visible_text=deserialize_visible_text(_decode_json(row["visible_text"])),
        state_changes=deserialize_state_changes(_decode_json(row["state_changes"])),
        memory_tags=tuple(row["memory_tags"] or ()),
        keyframe_candidates=deserialize_keyframe_candidates(
            _decode_json(row["keyframe_candidates"])
        ),
        interrupt_candidate=_decode_optional_object(row["interrupt_candidate"]),
        starts_new_event=row["starts_new_event"],
        boundary_reason=row["boundary_reason"],
        confidence=float(row["confidence"]),
        analyzer_provider=row["analyzer_provider"],
        analyzer_model=row["analyzer_model"],
        analyzer_contract_version=row["analyzer_contract_version"],
        compacted_at=row["compacted_at"],
        created_at=row["created_at"],
    )


def _session_working_memory_from_row(row: Mapping[str, Any]) -> SessionWorkingMemoryState:
    contract_version, state = deserialize_working_memory_state(_decode_json(row["state"]))
    return SessionWorkingMemoryState(
        session_id=row["session_id"],
        owner_id=row["owner_id"],
        revision=row["revision"],
        last_observation_id=row["last_observation_id"],
        last_observation_ended_at=row["last_observation_ended_at"],
        state=state,
        expires_at=row["expires_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        contract_version=contract_version,
    )


def _owner_working_memory_from_row(row: Mapping[str, Any]) -> OwnerWorkingMemoryState:
    contract_version, state = deserialize_working_memory_state(_decode_json(row["state"]))
    return OwnerWorkingMemoryState(
        owner_id=row["owner_id"],
        revision=row["revision"],
        state=state,
        expires_at=row["expires_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        contract_version=contract_version,
    )


def _decode_json(value: object) -> object:
    return json.loads(value) if isinstance(value, str) else value


def _decode_optional_object(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    decoded = _decode_json(value)
    if not isinstance(decoded, Mapping):
        raise TypeError("expected a JSON object")
    return dict(decoded)
