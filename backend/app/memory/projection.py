from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

import asyncpg

from app.memory.jobs.handlers import MemoryJobValidationError
from app.memory.jobs.models import MemoryJob
from app.memory.jobs.repository import MemoryJobRepository
from app.memory.prompts import MEMORY_PROFILE_DERIVED_NOTICE, MEMORY_PROFILE_HEADING


REFRESH_MEMORY_PROFILE_JOB_TYPE = "refresh_memory_profile"

_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(api[_ -]?key|access[_ -]?token|bearer|client[_ -]?secret|password|secret)"
    r"\b\s*[:=]\s*\S+"
)
_JWT_PATTERN = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)
_PROVIDER_KEY_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{20,})\b"
)
_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
_LOCAL_PATH_PATTERN = re.compile(r"/(?:Users|home|private|tmp|var)/\S+")
_LABELED_STORAGE_PATH_PATTERN = re.compile(
    r"(?i)\b(storage[_ -]?path|object[_ -]?path|bucket[_ -]?path)"
    r"\b\s*[:=]\s*\S+"
)
_KNOWN_STORAGE_PATH_PATTERN = re.compile(
    r"(?i)(?<![\w.-])(?:jarvis-files|visual-frames|visual_frames|files)/\S+"
)


@dataclass(frozen=True, slots=True)
class MemoryProjectionLimits:
    max_characters: int = 16_000
    section_item_limit: int = 20
    recent_episode_limit: int = 12
    superseded_history_limit: int = 10

    def __post_init__(self) -> None:
        if self.max_characters < 2_000:
            raise ValueError("memory profile max_characters must be at least 2000")
        for label, value in (
            ("section_item_limit", self.section_item_limit),
            ("recent_episode_limit", self.recent_episode_limit),
            ("superseded_history_limit", self.superseded_history_limit),
        ):
            if value < 1:
                raise ValueError(f"memory profile {label} must be positive")


@dataclass(frozen=True, slots=True)
class ProfileMemoryEntry:
    id: UUID
    memory_type: str
    status: str
    title: str
    content: str
    summary: str
    confidence: float
    importance: float
    occurred_at: datetime | None
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ProfilePreferenceEntry:
    id: UUID
    key: str
    value: str
    source: str
    confidence: float
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ProfileEntityEntry:
    id: UUID
    entity_type: str
    canonical_name: str
    description: str
    confidence: float
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ProfileFocusEntry:
    id: UUID
    target: str
    description: str
    purpose: str
    mode: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryProjectionSource:
    owner_id: str
    source_revision: int
    behavior_instructions: str
    focus: ProfileFocusEntry | None
    goals: tuple[ProfileMemoryEntry, ...]
    preferences: tuple[ProfilePreferenceEntry, ...]
    entities: tuple[ProfileEntityEntry, ...]
    facts: tuple[ProfileMemoryEntry, ...]
    episodes: tuple[ProfileMemoryEntry, ...]
    history: tuple[ProfileMemoryEntry, ...]


@dataclass(frozen=True, slots=True)
class MemoryProfile:
    owner_id: str
    markdown: str
    content_hash: str
    generated_at: datetime
    source_revision: int
    projected_revision: int

    @property
    def is_stale(self) -> bool:
        return self.projected_revision < self.source_revision

    def as_payload(self) -> dict[str, object]:
        return {
            "markdown": self.markdown,
            "content_hash": self.content_hash,
            "generated_at": self.generated_at.isoformat(),
            "source_revision": self.source_revision,
            "projected_revision": self.projected_revision,
            "is_stale": self.is_stale,
        }


class MemoryProjectionStore(Protocol):
    async def load_cached(self, *, owner_id: str) -> MemoryProfile | None: ...

    async def load_source(
        self,
        *,
        owner_id: str,
        limits: MemoryProjectionLimits,
    ) -> MemoryProjectionSource: ...

    async def store(
        self,
        profile: MemoryProfile,
        *,
        connection: asyncpg.Connection | None = None,
    ) -> MemoryProfile: ...

    async def list_stale_owner_revisions(
        self,
        *,
        connection: asyncpg.Connection,
        limit: int,
    ) -> list[tuple[str, int]]: ...


class MemoryProjectionRepository:
    """Owner-scoped source reader and cache store for the derived profile."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def load_cached(self, *, owner_id: str) -> MemoryProfile | None:
        normalized_owner = _required_text(owner_id, "owner_id")
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                select projection.*
                from memory_profile_projections projection
                where projection.owner_id = $1 and projection.markdown is not null
                  and not exists (
                      select 1 from memory_items item
                      where item.owner_id = projection.owner_id and item.status = 'active'
                        and (
                            (item.valid_from > projection.generated_at and item.valid_from <= now())
                            or (item.valid_to > projection.generated_at and item.valid_to <= now())
                        )
                  )
                  and not exists (
                      select 1 from jarvis_visual_focuses focus
                      where focus.owner_id = projection.owner_id
                        and focus.expires_at > projection.generated_at and focus.expires_at <= now()
                  )
                """,
                normalized_owner,
            )
        return _profile_from_row(row) if row is not None else None

    async def load_source(
        self,
        *,
        owner_id: str,
        limits: MemoryProjectionLimits,
    ) -> MemoryProjectionSource:
        normalized_owner = _required_text(owner_id, "owner_id")
        async with self._pool.acquire() as connection:
            async with connection.transaction(isolation="repeatable_read"):
                await connection.execute(
                    """
                    insert into memory_profile_projections (owner_id)
                    values ($1)
                    on conflict (owner_id) do nothing
                    """,
                    normalized_owner,
                )
                source_revision = await connection.fetchval(
                    """
                    select source_revision
                    from memory_profile_projections
                    where owner_id = $1
                    """,
                    normalized_owner,
                )
                behavior = await connection.fetchrow(
                    """
                    select behavior_instructions
                    from jarvis_behavior_profiles
                    where owner_id = $1
                    """,
                    normalized_owner,
                )
                focus = await connection.fetchrow(
                    """
                    select id, target, description, purpose, mode, expires_at
                    from jarvis_visual_focuses
                    where owner_id = $1
                      and completed_at is null
                      and cleared_at is null
                      and expires_at > now()
                    order by created_at desc, id desc
                    limit 1
                    """,
                    normalized_owner,
                )
                preferences = await connection.fetch(
                    """
                    select id, key, value, source, confidence, updated_at
                    from jarvis_user_preferences
                    where owner_id = $1
                    order by confidence desc, updated_at desc, id asc
                    limit $2
                    """,
                    normalized_owner,
                    limits.section_item_limit,
                )
                entities = await connection.fetch(
                    """
                    select id, entity_type, canonical_name, description,
                           confidence, updated_at
                    from memory_entities
                    where owner_id = $1
                    order by confidence desc, updated_at desc, id asc
                    limit $2
                    """,
                    normalized_owner,
                    limits.section_item_limit,
                )
                goals = await _fetch_memories(
                    connection,
                    owner_id=normalized_owner,
                    memory_type="goal",
                    statuses=("active",),
                    limit=limits.section_item_limit,
                    order_by_importance=True,
                )
                facts = await _fetch_memories(
                    connection,
                    owner_id=normalized_owner,
                    memory_type="semantic_fact",
                    statuses=("active",),
                    limit=limits.section_item_limit,
                    order_by_importance=True,
                )
                episodes = await _fetch_memories(
                    connection,
                    owner_id=normalized_owner,
                    memory_type="episode",
                    statuses=("active",),
                    limit=limits.recent_episode_limit,
                    order_by_importance=False,
                )
                history = await _fetch_memories(
                    connection,
                    owner_id=normalized_owner,
                    memory_type=None,
                    statuses=("superseded", "retracted"),
                    limit=limits.superseded_history_limit,
                    order_by_importance=False,
                )

        return MemoryProjectionSource(
            owner_id=normalized_owner,
            source_revision=int(source_revision or 0),
            behavior_instructions=(
                str(behavior["behavior_instructions"]) if behavior is not None else ""
            ),
            focus=_focus_from_row(focus) if focus is not None else None,
            goals=tuple(_memory_from_row(row) for row in goals),
            preferences=tuple(_preference_from_row(row) for row in preferences),
            entities=tuple(_entity_from_row(row) for row in entities),
            facts=tuple(_memory_from_row(row) for row in facts),
            episodes=tuple(_memory_from_row(row) for row in episodes),
            history=tuple(_memory_from_row(row) for row in history),
        )

    async def store(
        self,
        profile: MemoryProfile,
        *,
        connection: asyncpg.Connection | None = None,
    ) -> MemoryProfile:
        if connection is not None:
            return await self._store(connection, profile)
        async with self._pool.acquire() as owned_connection:
            async with owned_connection.transaction():
                return await self._store(owned_connection, profile)

    async def list_stale_owner_revisions(
        self,
        *,
        connection: asyncpg.Connection,
        limit: int,
    ) -> list[tuple[str, int]]:
        if limit < 1:
            raise ValueError("memory profile recovery limit must be positive")
        rows = await connection.fetch(
            """
            with source_owners as (
                select owner_id from jarvis_behavior_profiles
                union select owner_id from jarvis_user_preferences
                union select owner_id from jarvis_visual_focuses
                union select owner_id from memory_items
                union select owner_id from memory_entities
                union select owner_id from memory_profile_projections
            )
            select owners.owner_id, coalesce(profile.source_revision, 0) as revision
            from source_owners owners
            left join memory_profile_projections profile
              on profile.owner_id = owners.owner_id
            where profile.owner_id is null
               or profile.markdown is null
               or profile.projected_revision < profile.source_revision
            order by owners.owner_id
            limit $1
            """,
            limit,
        )
        return [(str(row["owner_id"]), int(row["revision"])) for row in rows]

    async def _store(
        self,
        connection: asyncpg.Connection,
        profile: MemoryProfile,
    ) -> MemoryProfile:
        row = await connection.fetchrow(
            """
            insert into memory_profile_projections (
                owner_id,
                source_revision,
                projected_revision,
                markdown,
                content_hash,
                generated_at
            )
            values ($1, $2, $2, $3, $4, $5)
            on conflict (owner_id) do update
            set source_revision = greatest(
                    memory_profile_projections.source_revision,
                    excluded.source_revision
                ),
                projected_revision = excluded.projected_revision,
                markdown = excluded.markdown,
                content_hash = excluded.content_hash,
                generated_at = excluded.generated_at,
                updated_at = now()
            where memory_profile_projections.projected_revision
                  <= excluded.projected_revision
            returning *
            """,
            profile.owner_id,
            profile.source_revision,
            profile.markdown,
            profile.content_hash,
            profile.generated_at,
        )
        if row is None:
            row = await connection.fetchrow(
                "select * from memory_profile_projections where owner_id = $1",
                profile.owner_id,
            )
        if row is None:
            raise RuntimeError("memory profile cache row disappeared during update")
        return _profile_from_row(row)


class MemoryProjectionService:
    """Build and serve a bounded projection without making Markdown authoritative."""

    def __init__(
        self,
        *,
        repository: MemoryProjectionStore,
        limits: MemoryProjectionLimits,
        now: Any = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._limits = limits
        self._now = now

    async def get_profile(self, *, owner_id: str) -> MemoryProfile:
        cached = await self._repository.load_cached(owner_id=owner_id)
        if cached is not None and not cached.is_stale:
            return cached

        stored: MemoryProfile | None = None
        for _ in range(2):
            candidate = await self.build(owner_id=owner_id)
            stored = await self._repository.store(candidate)
            if not stored.is_stale:
                return stored
        if stored is None:
            raise RuntimeError("memory profile generation did not produce a profile")
        return stored

    async def build(self, *, owner_id: str) -> MemoryProfile:
        # A boundary crossed while reading must invalidate this cache on its
        # next use, even if the prepared profile is committed after the boundary.
        generated_at = _utc(self._now())
        source = await self._repository.load_source(
            owner_id=owner_id,
            limits=self._limits,
        )
        markdown = render_memory_profile(
            source,
            generated_at=generated_at,
            max_characters=self._limits.max_characters,
        )
        return MemoryProfile(
            owner_id=source.owner_id,
            markdown=markdown,
            content_hash=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            generated_at=generated_at,
            source_revision=source.source_revision,
            projected_revision=source.source_revision,
        )

    async def commit(
        self,
        profile: MemoryProfile,
        *,
        connection: asyncpg.Connection,
    ) -> MemoryProfile:
        return await self._repository.store(profile, connection=connection)


class RefreshMemoryProfileJobHandler:
    job_type = REFRESH_MEMORY_PROFILE_JOB_TYPE

    def __init__(self, service: MemoryProjectionService) -> None:
        self._service = service

    async def prepare(self, job: MemoryJob) -> MemoryProfile:
        revision = job.payload.get("source_revision")
        if revision is not None and (
            isinstance(revision, bool) or not isinstance(revision, int) or revision < 0
        ):
            raise MemoryJobValidationError(
                "memory profile source_revision must be a non-negative integer"
            )
        return await self._service.build(owner_id=job.owner_id)

    async def commit(
        self,
        job: MemoryJob,
        prepared_result: MemoryProfile,
        connection: asyncpg.Connection,
    ) -> Mapping[str, Any]:
        stored = await self._service.commit(
            prepared_result,
            connection=connection,
        )
        return {
            "owner_id": job.owner_id,
            "source_revision": stored.source_revision,
            "projected_revision": stored.projected_revision,
            "content_hash": stored.content_hash,
            "is_stale": stored.is_stale,
        }


class MemoryProfileRecoveryRule:
    name = "memory_profile_projection"

    def __init__(
        self,
        *,
        projection_repository: MemoryProjectionStore,
        batch_size: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("memory profile recovery batch_size must be positive")
        self._projection_repository = projection_repository
        self._batch_size = batch_size

    async def reconcile(
        self,
        *,
        repository: MemoryJobRepository,
        connection: asyncpg.Connection,
        now: datetime,
    ) -> int:
        stale = await self._projection_repository.list_stale_owner_revisions(
            connection=connection,
            limit=self._batch_size,
        )
        created = 0
        for owner_id, revision in stale:
            result = await repository.enqueue(
                owner_id=owner_id,
                session_id=None,
                job_type=REFRESH_MEMORY_PROFILE_JOB_TYPE,
                dedupe_key=memory_profile_refresh_dedupe_key(owner_id, revision),
                payload={"source_revision": revision},
                priority=-20,
                available_at=now,
                connection=connection,
            )
            created += int(result.created)
        return created


def memory_profile_refresh_dedupe_key(owner_id: str, revision: int) -> str:
    owner = _required_text(owner_id, "owner_id")
    if revision < 0:
        raise ValueError("memory profile revision cannot be negative")
    return f"memory-profile:{owner}:{revision}"


def render_memory_profile(
    source: MemoryProjectionSource,
    *,
    generated_at: datetime,
    max_characters: int,
) -> str:
    sections = [
        MEMORY_PROFILE_HEADING,
        "",
        f"> {MEMORY_PROFILE_DERIVED_NOTICE}",
        f"> Source revision: {source.source_revision}",
        "",
        "## Behavior instructions",
        _paragraph_or_empty(
            source.behavior_instructions,
            empty="_No persistent behavior instructions._",
        ),
        "",
        "## Active goals and focus",
    ]
    focus_and_goals: list[str] = []
    if source.focus is not None:
        focus_and_goals.append(
            "- **Visual focus:** "
            f"{_safe_text(source.focus.target)} — "
            f"{_safe_text(source.focus.description or source.focus.purpose)} "
            f"(mode={_safe_text(source.focus.mode)}, "
            f"expires={_timestamp(source.focus.expires_at)}, id={source.focus.id})"
        )
    focus_and_goals.extend(_memory_line(item) for item in source.goals if item.status == "active")
    sections.extend(focus_and_goals or ["_No active goals or visual focus._"])

    sections.extend(("", "## Stable preferences"))
    sections.extend(
        [
            "- "
            f"**{_safe_text(item.key)}:** {_safe_text(item.value)} "
            f"(confidence={_score(item.confidence)}, source={_safe_text(item.source)}, "
            f"id={item.id}, updated={_timestamp(item.updated_at)})"
            for item in source.preferences
        ]
        or ["_No stable preferences._"]
    )

    sections.extend(("", "## Important people, places, and objects"))
    sections.extend(
        [
            "- "
            f"**{_safe_text(item.canonical_name)}** "
            f"(type={_safe_text(item.entity_type)}, confidence={_score(item.confidence)}, "
            f"id={item.id}, updated={_timestamp(item.updated_at)})"
            + (
                f" — {_safe_text(item.description)}"
                if item.description.strip()
                else ""
            )
            for item in source.entities
        ]
        or ["_No important entities._"]
    )

    sections.extend(("", "## Active facts"))
    sections.extend(
        [_memory_line(item) for item in source.facts if item.status == "active"]
        or ["_No active facts._"]
    )

    sections.extend(("", "## Recent notable episodes"))
    sections.extend(
        [_memory_line(item) for item in source.episodes if item.status == "active"]
        or ["_No notable episodes._"]
    )

    sections.extend(("", "## Superseded or retracted history"))
    sections.extend(
        [
            f"- **[{item.status.upper()}] {_safe_text(item.title)}** — "
            f"{_safe_text(item.summary or item.content)} "
            f"(confidence={_score(item.confidence)}, id={item.id}, "
            f"updated={_timestamp(item.updated_at)})"
            for item in source.history
            if item.status in {"superseded", "retracted"}
        ]
        or ["_No superseded or retracted history._"]
    )

    return _bounded_markdown("\n".join(sections).strip() + "\n", max_characters)


async def _fetch_memories(
    connection: asyncpg.Connection,
    *,
    owner_id: str,
    memory_type: str | None,
    statuses: Sequence[str],
    limit: int,
    order_by_importance: bool,
) -> Sequence[Mapping[str, Any]]:
    ordering = (
        "importance desc, coalesce(occurred_at, created_at) desc, id asc"
        if order_by_importance
        else "coalesce(occurred_at, created_at) desc, importance desc, id asc"
    )
    return await connection.fetch(
        f"""
        select id, memory_type, status, title, content, summary,
               confidence, importance, occurred_at, updated_at
        from memory_items
        where owner_id = $1
          and ($2::text is null or memory_type = $2)
          and status = any($3::text[])
          and (valid_from is null or status <> 'active' or valid_from <= now())
          and (memory_type = 'episode' or valid_to is null or status <> 'active' or valid_to > now())
        order by {ordering}
        limit $4
        """,
        owner_id,
        memory_type,
        list(statuses),
        limit,
    )


def _memory_line(item: ProfileMemoryEntry) -> str:
    occurred = item.occurred_at or item.updated_at
    return (
        f"- **{_safe_text(item.title)}** — "
        f"{_safe_text(item.summary or item.content)} "
        f"(confidence={_score(item.confidence)}, importance={_score(item.importance)}, "
        f"id={item.id}, time={_timestamp(occurred)})"
    )


def _paragraph_or_empty(value: str, *, empty: str) -> str:
    normalized = _safe_text(value)
    return normalized if normalized else empty


def _safe_text(value: object, *, limit: int = 600) -> str:
    text = " ".join(str(value or "").split())
    text = _CREDENTIAL_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    text = _JWT_PATTERN.sub("[REDACTED TOKEN]", text)
    text = _PROVIDER_KEY_PATTERN.sub("[REDACTED KEY]", text)
    text = _URL_PATTERN.sub("[REDACTED URL]", text)
    text = _LOCAL_PATH_PATTERN.sub("[REDACTED PATH]", text)
    text = _LABELED_STORAGE_PATH_PATTERN.sub(
        lambda match: f"{match.group(1)}=[REDACTED PATH]",
        text,
    )
    text = _KNOWN_STORAGE_PATH_PATTERN.sub("[REDACTED PATH]", text)
    if len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _bounded_markdown(markdown: str, max_characters: int) -> str:
    if len(markdown) <= max_characters:
        return markdown
    suffix = "\n\n_Projection truncated to its configured character budget._\n"
    available = max_characters - len(suffix)
    cut = markdown.rfind("\n", 0, available)
    if cut < len(MEMORY_PROFILE_HEADING):
        cut = available
    return markdown[:cut].rstrip() + suffix


def _profile_from_row(row: Mapping[str, Any]) -> MemoryProfile:
    return MemoryProfile(
        owner_id=str(row["owner_id"]),
        markdown=str(row["markdown"]),
        content_hash=str(row["content_hash"]),
        generated_at=_utc(row["generated_at"]),
        source_revision=int(row["source_revision"]),
        projected_revision=int(row["projected_revision"]),
    )


def _memory_from_row(row: Mapping[str, Any]) -> ProfileMemoryEntry:
    return ProfileMemoryEntry(
        id=row["id"],
        memory_type=str(row["memory_type"]),
        status=str(row["status"]),
        title=str(row["title"]),
        content=str(row["content"]),
        summary=str(row["summary"]),
        confidence=float(row["confidence"]),
        importance=float(row["importance"]),
        occurred_at=_utc(row["occurred_at"]) if row["occurred_at"] else None,
        updated_at=_utc(row["updated_at"]),
    )


def _preference_from_row(row: Mapping[str, Any]) -> ProfilePreferenceEntry:
    return ProfilePreferenceEntry(
        id=row["id"],
        key=str(row["key"]),
        value=str(row["value"]),
        source=str(row["source"]),
        confidence=float(row["confidence"]),
        updated_at=_utc(row["updated_at"]),
    )


def _entity_from_row(row: Mapping[str, Any]) -> ProfileEntityEntry:
    return ProfileEntityEntry(
        id=row["id"],
        entity_type=str(row["entity_type"]),
        canonical_name=str(row["canonical_name"]),
        description=str(row["description"]),
        confidence=float(row["confidence"]),
        updated_at=_utc(row["updated_at"]),
    )


def _focus_from_row(row: Mapping[str, Any]) -> ProfileFocusEntry:
    return ProfileFocusEntry(
        id=row["id"],
        target=str(row["target"]),
        description=str(row["description"]),
        purpose=str(row["purpose"]),
        mode=str(row["mode"]),
        expires_at=_utc(row["expires_at"]),
    )


def _score(value: float) -> str:
    return f"{value:.2f}"


def _timestamp(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("memory profile timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _required_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be blank")
    return normalized
