from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID
from unittest.mock import AsyncMock

from app.memory.projection import (
    MemoryProfile,
    MemoryProfileRecoveryRule,
    MemoryProjectionLimits,
    MemoryProjectionRepository,
    MemoryProjectionService,
    MemoryProjectionSource,
    ProfileEntityEntry,
    ProfileFocusEntry,
    ProfileMemoryEntry,
    ProfilePreferenceEntry,
    RefreshMemoryProfileJobHandler,
    memory_profile_refresh_dedupe_key,
    _fetch_memories,
)


NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


class _ProjectionRepository:
    def __init__(self, source: MemoryProjectionSource) -> None:
        self.source = source
        self.cached: MemoryProfile | None = None
        self.load_source_owners: list[str] = []
        self.stale = [(source.owner_id, source.source_revision)]

    async def load_cached(self, *, owner_id: str) -> MemoryProfile | None:
        if self.cached is not None and self.cached.owner_id != owner_id:
            raise AssertionError("owner scope leaked")
        return self.cached

    async def load_source(self, *, owner_id: str, **_: Any) -> MemoryProjectionSource:
        self.load_source_owners.append(owner_id)
        if owner_id != self.source.owner_id:
            raise AssertionError("owner scope leaked")
        return self.source

    async def store(
        self,
        profile: MemoryProfile,
        *,
        connection: object | None = None,
    ) -> MemoryProfile:
        del connection
        self.cached = profile
        return profile

    async def list_stale_owner_revisions(self, **_: Any) -> list[tuple[str, int]]:
        return list(self.stale)


class _JobRepository:
    def __init__(self) -> None:
        self.keys: set[str] = set()

    async def enqueue(self, **kwargs: Any) -> SimpleNamespace:
        key = kwargs["dedupe_key"]
        created = key not in self.keys
        self.keys.add(key)
        return SimpleNamespace(created=created)


class _QueryCaptureConnection:
    def __init__(self) -> None:
        self.query = ""

    async def fetch(self, query: str, *_: Any) -> list[dict[str, Any]]:
        self.query = query
        return []


class MemoryProjectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_source_query_respects_both_validity_boundaries(self) -> None:
        connection = AsyncMock()
        connection.fetch.return_value = []
        await _fetch_memories(
            connection, owner_id="owner-a", memory_type="semantic_fact",
            statuses=("active",), limit=10, order_by_importance=True,
        )
        query = connection.fetch.call_args.args[0]
        self.assertIn("valid_from <= now()", query)
        self.assertIn("valid_to > now()", query)
        self.assertEqual(connection.fetch.call_args.args[1], "owner-a")

    async def test_profile_timestamp_precedes_source_read(self) -> None:
        repository = _ProjectionRepository(_source())
        phases = []
        original = repository.load_source

        async def load_source(**kwargs):
            phases.append("source")
            return await original(**kwargs)

        def now():
            phases.append("timestamp")
            return NOW

        repository.load_source = load_source
        service = MemoryProjectionService(repository=repository, limits=MemoryProjectionLimits(), now=now)
        await service.build(owner_id="owner-a")
        self.assertEqual(phases, ["timestamp", "source"])

    async def test_projection_is_deterministic_bounded_and_redacted(self) -> None:
        repository = _ProjectionRepository(_source())
        first_service = MemoryProjectionService(
            repository=repository,
            limits=MemoryProjectionLimits(max_characters=2_400),
            now=lambda: NOW,
        )
        later_service = MemoryProjectionService(
            repository=repository,
            limits=MemoryProjectionLimits(max_characters=2_400),
            now=lambda: NOW + timedelta(minutes=5),
        )

        first = await first_service.build(owner_id="owner-a")
        second = await later_service.build(owner_id="owner-a")

        self.assertEqual(first.markdown, second.markdown)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertNotEqual(first.generated_at, second.generated_at)
        self.assertLessEqual(len(first.markdown), 2_400)
        self.assertIn("## Active goals and focus", first.markdown)
        self.assertIn("## Stable preferences", first.markdown)
        self.assertIn("[SUPERSEDED] Former city", first.markdown)
        self.assertNotIn("secret-value", first.markdown)
        self.assertNotIn("storage.example", first.markdown)
        self.assertNotIn("jarvis-files/owner-a/private/profile.json", first.markdown)
        self.assertNotIn("Forgotten detail", first.markdown)

    async def test_recovery_includes_projection_only_owners(self) -> None:
        repository = MemoryProjectionRepository(SimpleNamespace())  # type: ignore[arg-type]
        connection = _QueryCaptureConnection()

        result = await repository.list_stale_owner_revisions(
            connection=connection,  # type: ignore[arg-type]
            limit=10,
        )

        self.assertEqual(result, [])
        self.assertIn(
            "union select owner_id from memory_profile_projections",
            connection.query,
        )

    async def test_cached_profile_is_reused_and_deleted_cache_regenerates(self) -> None:
        repository = _ProjectionRepository(_source())
        service = MemoryProjectionService(
            repository=repository,
            limits=MemoryProjectionLimits(),
            now=lambda: NOW,
        )

        first = await service.get_profile(owner_id="owner-a")
        second = await service.get_profile(owner_id="owner-a")
        self.assertIs(first, second)
        self.assertEqual(repository.load_source_owners, ["owner-a"])

        repository.cached = None
        regenerated = await service.get_profile(owner_id="owner-a")
        self.assertEqual(regenerated.content_hash, first.content_hash)
        self.assertEqual(repository.load_source_owners, ["owner-a", "owner-a"])

    async def test_stale_cache_is_rebuilt(self) -> None:
        repository = _ProjectionRepository(_source())
        repository.cached = MemoryProfile(
            owner_id="owner-a",
            markdown="old",
            content_hash="c" * 64,
            generated_at=NOW - timedelta(minutes=5),
            source_revision=7,
            projected_revision=6,
        )
        service = MemoryProjectionService(
            repository=repository,
            limits=MemoryProjectionLimits(),
            now=lambda: NOW,
        )

        result = await service.get_profile(owner_id="owner-a")

        self.assertFalse(result.is_stale)
        self.assertNotEqual(result.markdown, "old")

    async def test_refresh_handler_and_recovery_use_revision_dedupe(self) -> None:
        repository = _ProjectionRepository(_source())
        service = MemoryProjectionService(
            repository=repository,
            limits=MemoryProjectionLimits(),
            now=lambda: NOW,
        )
        handler = RefreshMemoryProfileJobHandler(service)
        job = SimpleNamespace(
            owner_id="owner-a",
            payload={"source_revision": 7},
        )
        prepared = await handler.prepare(job)  # type: ignore[arg-type]
        result = await handler.commit(  # type: ignore[arg-type]
            job,
            prepared,
            SimpleNamespace(),
        )
        self.assertEqual(result["projected_revision"], 7)

        jobs = _JobRepository()
        rule = MemoryProfileRecoveryRule(
            projection_repository=repository,
            batch_size=10,
        )
        first = await rule.reconcile(
            repository=jobs,  # type: ignore[arg-type]
            connection=SimpleNamespace(),  # type: ignore[arg-type]
            now=NOW,
        )
        second = await rule.reconcile(
            repository=jobs,  # type: ignore[arg-type]
            connection=SimpleNamespace(),  # type: ignore[arg-type]
            now=NOW,
        )
        self.assertEqual((first, second), (1, 0))
        self.assertEqual(
            jobs.keys,
            {memory_profile_refresh_dedupe_key("owner-a", 7)},
        )


def _source() -> MemoryProjectionSource:
    active_goal = _memory(
        1,
        memory_type="goal",
        status="active",
        title="Finish the prototype",
        content="Complete the Jarvis prototype.",
    )
    fact = _memory(
        2,
        memory_type="semantic_fact",
        status="active",
        title="Home city",
        content=(
            "The owner lives in Dublin. Stored at "
            "jarvis-files/owner-a/private/profile.json"
        ),
    )
    episode = _memory(
        3,
        memory_type="episode",
        status="active",
        title="Morning walk",
        content="Walked through the park.",
    )
    superseded = _memory(
        4,
        memory_type="semantic_fact",
        status="superseded",
        title="Former city",
        content="The owner lived elsewhere.",
    )
    forgotten = _memory(
        5,
        memory_type="semantic_fact",
        status="forgotten",
        title="Forgotten detail",
        content="This must not appear.",
    )
    return MemoryProjectionSource(
        owner_id="owner-a",
        source_revision=7,
        behavior_instructions="Never reveal password=secret-value.",
        focus=ProfileFocusEntry(
            id=UUID(int=10),
            target="Train platform",
            description="Find the correct platform.",
            purpose="Catch the train.",
            mode="find_once",
            expires_at=NOW + timedelta(hours=1),
        ),
        goals=(active_goal,),
        preferences=(
            ProfilePreferenceEntry(
                id=UUID(int=11),
                key="coffee",
                value="quiet cafes",
                source="explicit",
                confidence=1.0,
                updated_at=NOW,
            ),
        ),
        entities=(
            ProfileEntityEntry(
                id=UUID(int=12),
                entity_type="place",
                canonical_name="Home",
                description="https://storage.example/signed/path",
                confidence=0.95,
                updated_at=NOW,
            ),
        ),
        facts=(fact, forgotten),
        episodes=(episode,),
        history=(superseded, forgotten),
    )


def _memory(
    integer: int,
    *,
    memory_type: str,
    status: str,
    title: str,
    content: str,
) -> ProfileMemoryEntry:
    return ProfileMemoryEntry(
        id=UUID(int=integer),
        memory_type=memory_type,
        status=status,
        title=title,
        content=content,
        summary="",
        confidence=0.9,
        importance=0.8,
        occurred_at=NOW,
        updated_at=NOW,
    )


if __name__ == "__main__":
    unittest.main()
