"""Owned real-database regressions, not native-model or complete pipeline acceptance.

Run only while background memory workers are stopped. Every owner is manifested
before its first write; unittest cleanup closes the pool and uses an independent
connection to remove and verify all fixture-owned rows. The external runner must
also audit cleanup before restoring workers.
"""
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

import asyncpg

from app.memory.entities.repository import EntityRepository
from app.memory.presentation import retrieval_payload
from app.memory.projection import MemoryProjectionLimits, MemoryProjectionRepository, MemoryProjectionService
from app.memory.repository import MemoryRepository
from app.memory.retrieval.candidates import RetrievalCandidateCollector
from app.memory.retrieval.query import MemoryRetrievalQuery
from app.memory.retrieval.service import HierarchicalMemoryRetrievalService
from app.memory.semantic.contracts import SemanticEntityIdentity, SemanticFactProposal
from app.memory.semantic.repository import SemanticMemoryRepository
from app.perception.repository import PerceptionRepository
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners


@unittest.skipUnless(
    os.getenv("JARVIS_MEMORY_TEST_DB_URL") and os.getenv("JARVIS_MEMORY_FIXTURE_MANIFEST_DIR"),
    "An explicit memory-test database and fixture manifest directory are required",
)
class MemoryTemporalRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dsn = os.environ["JARVIS_MEMORY_TEST_DB_URL"]
        self.owner = f"direct-image-eval-memory-temporal-{uuid4()}"
        self.owners = {self.owner}
        self.now = datetime.now(UTC)
        self.base = self.now - timedelta(hours=3)
        directory = Path(os.environ["JARVIS_MEMORY_FIXTURE_MANIFEST_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{self.owner}.json"
        self.manifest = {
            "schema": "jarvis.memory_temporal_repository_fixture.v1",
            "test": self.id(), "owners": sorted(self.owners), "status": "prepared",
            "started_at": self.now.isoformat(), "memory_ids": [], "session_ids": [],
        }
        self._persist_manifest()
        self.addAsyncCleanup(self._cleanup)
        self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=3, timeout=10, command_timeout=20)
        self.addAsyncCleanup(self.pool.close)
        self.memory = MemoryRepository(self.pool)
        self.semantic = SemanticMemoryRepository(self.pool)
        self.entities = {}

    def _persist_manifest(self):
        self.path.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")

    async def _cleanup(self):
        result = await cleanup_fixture_owners(self.dsn, self.owners)
        self.manifest.update(status="cleaned", cleanup=result)
        self._persist_manifest()
        self.assertEqual(result["remaining_rows"], 0)

    async def _context(self, at, *, owner=None):
        owner = owner or self.owner
        if owner not in self.owners:
            self.owners.add(owner)
            self.manifest["owners"] = sorted(self.owners)
            self._persist_manifest()
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                session_id = await connection.fetchval(
                    "insert into jarvis_sessions (owner_id, title) values ($1, 'Memory temporal fixture') returning id",
                    owner,
                )
                event_id = await connection.fetchval(
                    """insert into experience_events (
                        owner_id, session_id, status, event_kind, modalities, started_at, ended_at,
                        boundary_reason, keyframes_finalized_at, consolidation_ready_at
                    ) values ($1, $2, 'closed', 'interaction', array['text'], $3, $3, 'fixture', $3, $3)
                    returning id""", owner, session_id, at,
                )
        self.manifest["session_ids"].append(str(session_id))
        self._persist_manifest()
        episode = await self.memory.promote_event_to_episode(
            owner_id=owner, event_id=event_id, title="Temporal fixture source",
            content="Synthetic repository regression evidence, not a personal event.",
            summary="Synthetic temporal regression", importance=0.8, confidence=0.95,
            source="integration_test", promotion_reason="Owned repository regression",
        )
        self.manifest["memory_ids"].append(str(episode.id))
        self._persist_manifest()
        if owner not in self.entities:
            self.entities[owner] = await EntityRepository(self.pool).create_entity(
                owner_id=owner, entity_type="object", canonical_name="Fixture pouch", confidence=0.99,
            )
        return SimpleNamespace(
            owner_id=owner, session_id=session_id, event_id=event_id,
            episode_memory_id=episode.id, occurred_at=at, result=SimpleNamespace(episode_importance=0.8),
        )

    async def _write(
        self, value, at, *, previous=None, intent=None, confidence=0.95, explicit=True,
        end=None, scope="state", predicate="is", context=None, owner=None,
    ):
        context = context or await self._context(at, owner=owner)
        proposal = SemanticFactProposal(
            subject=SemanticEntityIdentity(name="Fixture pouch", entity_type="object", confidence=0.99),
            predicate=predicate, value=value, confidence=confidence,
            source="user_statement" if explicit else "inferred_observation",
            change_intent=intent or ("transition" if previous is not None else "assert"),
            previous_value=previous, valid_from=at, valid_to=end, evidence=(), temporal_scope=scope,
        )
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                created, updated = await self.semantic._apply_fact_proposal(
                    connection, context=context, subject=self.entities[context.owner_id], proposal=proposal,
                )
        for item in ((created,) if created is not None else ()) + updated:
            if str(item.id) not in self.manifest["memory_ids"]:
                self.manifest["memory_ids"].append(str(item.id))
        self._persist_manifest()
        return created, updated

    async def _search(self, *, historical=False, owner=None):
        perception = PerceptionRepository(self.pool)
        service = HierarchicalMemoryRetrievalService(
            memory_repository=self.memory, perception_repository=perception,
            candidate_collector=RetrievalCandidateCollector(
                memory_repository=self.memory, perception_repository=perception, embedding_model=None,
            ),
        )
        return await service.search(
            owner_id=owner or self.owner,
            request=MemoryRetrievalQuery(
                query="Fixture pouch", start_time=self.base - timedelta(hours=1),
                end_time=self.now + timedelta(hours=2), memory_types=("semantic_fact",),
                include_superseded=historical,
            ), now=self.now,
        )

    async def test_found_replaces_missing_in_real_current_and_historical_retrieval(self):
        old, _ = await self._write("missing", self.base)
        found, _ = await self._write("found", self.base + timedelta(hours=1), previous="missing")
        current = await self._search()
        self.assertEqual({item.id for item in current.results}, {found.id})
        history = {item.id: item for item in (await self._search(historical=True)).results}
        self.assertTrue(history[old.id].is_historical)
        self.assertEqual(history[old.id].valid_to, found.valid_from)
        self.assertEqual(history[found.id].temporal_status, "current")

    async def test_profile_excludes_future_replacement_before_its_effective_time(self):
        missing, _ = await self._write("missing", self.base)
        future, _ = await self._write("found", self.now + timedelta(hours=1), previous="missing")
        service = MemoryProjectionService(
            repository=MemoryProjectionRepository(self.pool), limits=MemoryProjectionLimits(),
        )
        profile = await service.get_profile(owner_id=self.owner)
        self.assertIn(str(missing.id), profile.markdown)
        self.assertNotIn(str(future.id), profile.markdown)

    async def test_ended_episode_remains_in_default_retrieval_and_recent_profile(self):
        context = await self._context(self.base)
        async with self.pool.acquire() as connection:
            await connection.execute(
                "update memory_items set valid_from = $3, valid_to = $4 where owner_id = $1 and id = $2",
                self.owner, context.episode_memory_id, self.base, self.base + timedelta(minutes=1),
            )
        perception = PerceptionRepository(self.pool)
        service = HierarchicalMemoryRetrievalService(
            memory_repository=self.memory, perception_repository=perception,
            candidate_collector=RetrievalCandidateCollector(
                memory_repository=self.memory, perception_repository=perception, embedding_model=None,
            ),
        )
        result = await service.search(owner_id=self.owner, request=MemoryRetrievalQuery(
            query="Temporal fixture", start_time=self.base - timedelta(hours=1),
            end_time=self.now, memory_types=("episode",),
        ))
        self.assertEqual({item.id for item in result.results}, {context.episode_memory_id})
        self.assertEqual(result.results[0].temporal_status, "historical")
        profile = await MemoryProjectionService(
            repository=MemoryProjectionRepository(self.pool), limits=MemoryProjectionLimits(),
        ).get_profile(owner_id=self.owner)
        self.assertIn(str(context.episode_memory_id), profile.markdown)

    async def test_relevant_old_state_is_not_crowded_out_by_recent_unrelated_facts(self):
        missing, _ = await self._write("missing", self.base)
        async with self.pool.acquire() as connection:
            await connection.execute(
                """with entities as (
                    insert into memory_entities (owner_id, entity_type, canonical_name, normalized_name, confidence)
                    select $1, 'object', 'Unrelated notebook ' || n, 'unrelated notebook ' || n,
                           0.95 from generate_series(1, 45) n
                    returning id, canonical_name
                ) insert into memory_items (
                    owner_id, session_id, memory_type, title, content, summary, occurred_at,
                    valid_from, importance, confidence, source, is_explicit, metadata
                ) select $1, $2, 'semantic_fact', canonical_name, canonical_name, canonical_name,
                    $3, $3, 0.8, 0.95, 'user_statement', true,
                    jsonb_build_object('subject_entity_id', id::text, 'predicate', 'is',
                                       'value', 'blue', 'temporal_scope', 'durable')
                  from entities""", self.owner, missing.session_id, self.base + timedelta(hours=1),
            )
        recent = await self.memory.list_semantic_fact_context(owner_id=self.owner, at=self.now, limit=40)
        self.assertFalse(any(item["subject"] == "Fixture pouch" for item in recent))
        relevant = await self.memory.list_semantic_fact_context(
            owner_id=self.owner, at=self.now, limit=40, query_text="I finally found my fixture pouch.",
        )
        self.assertEqual(len(relevant), 40)
        self.assertEqual(relevant[0]["subject"], "Fixture pouch")
        self.assertEqual(relevant[0]["value"], "missing")

    async def test_profile_cache_is_invalidated_by_time_without_revision_change(self):
        found, _ = await self._write("found", self.base + timedelta(hours=1))
        repository = MemoryProjectionRepository(self.pool)
        service = MemoryProjectionService(repository=repository, limits=MemoryProjectionLimits())
        await service.get_profile(owner_id=self.owner)
        # Simulate a cache prepared before the effective boundary. Revisions
        # still match: this must be rejected for time, not a source mutation.
        async with self.pool.acquire() as connection:
            await connection.execute(
                """update memory_profile_projections
                    set markdown = 'stale before state became current', generated_at = $2,
                        projected_revision = source_revision
                    where owner_id = $1""", self.owner, self.base,
            )
        self.assertIsNone(await repository.load_cached(owner_id=self.owner))
        current = await service.get_profile(owner_id=self.owner)
        self.assertIn(str(found.id), current.markdown)
        self.assertNotIn("stale before state became current", current.markdown)

    async def test_late_transition_splits_history_without_resurrecting_an_old_state(self):
        missing, _ = await self._write("missing", self.base)
        newest, _ = await self._write("found", self.base + timedelta(hours=2), previous="missing")
        middle, _ = await self._write("located briefly in a drawer", self.base + timedelta(hours=1), previous="missing")
        self.assertEqual(middle.valid_to, newest.valid_from)
        self.assertEqual({item.id for item in (await self._search()).results}, {newest.id})
        history = {item.id: item for item in (await self._search(historical=True)).results}
        self.assertEqual(history[missing.id].valid_to, middle.valid_from)
        self.assertEqual(history[middle.id].valid_to, newest.valid_from)
        self.assertTrue(history[middle.id].is_historical)

    async def test_older_location_does_not_override_a_newer_explicit_location(self):
        newest, _ = await self._write("shelf", self.base + timedelta(hours=2), predicate="located_at")
        old, _ = await self._write("desk", self.base, predicate="located_at")
        self.assertEqual(old.valid_to, newest.valid_from)
        self.assertEqual({item.id for item in (await self._search()).results}, {newest.id})

    async def test_uncertain_transition_keeps_established_state(self):
        missing, _ = await self._write("missing", self.base)
        result, updated = await self._write(
            "possibly found", self.base + timedelta(hours=1), previous="missing", explicit=False, confidence=0.86,
        )
        self.assertIsNone(result)
        self.assertEqual(updated, ())
        self.assertEqual({item.id for item in (await self._search()).results}, {missing.id})

    async def test_event_snapshot_does_not_reinforce_or_shorten_durable_knowledge(self):
        lasting, _ = await self._write("red", self.base, scope="durable")
        snapshot, _ = await self._write("red", self.base + timedelta(minutes=1), scope="event",
                                        end=self.base + timedelta(minutes=2))
        self.assertNotEqual(lasting.id, snapshot.id)
        self.assertEqual({item.id for item in (await self._search()).results}, {lasting.id})
        history = {item.id: item for item in (await self._search(historical=True)).results}
        self.assertTrue(history[snapshot.id].is_historical)
        self.assertIsNone(history[lasting.id].valid_to)

    async def test_duplicate_confirmation_reinforces_once_and_preserves_explicit_provenance(self):
        first, _ = await self._write("found", self.base, explicit=False)
        duplicate, updated = await self._write("found", self.base + timedelta(minutes=1), explicit=True)
        self.assertIsNone(duplicate)
        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0].id, first.id)
        self.assertTrue(updated[0].is_explicit)
        self.assertEqual(updated[0].source, "user_statement")
        self.assertEqual(updated[0].metadata["source_classification"], "user_statement")
        self.assertEqual({item.id for item in (await self._search()).results}, {first.id})

    async def test_future_transition_does_not_retire_the_current_value_early(self):
        old, _ = await self._write("missing", self.base)
        future, _ = await self._write("found", self.now + timedelta(hours=1), previous="missing")
        self.assertEqual({item.id for item in (await self._search()).results}, {old.id})
        payload = retrieval_payload(await self._search(historical=True))
        future_result = next(item for item in payload["results"] if item["id"] == str(future.id))
        self.assertEqual(future_result["temporal_status"], "future")

    async def test_context_and_retrieval_are_isolated_between_real_owners(self):
        own, _ = await self._write("missing", self.base)
        other_owner = f"direct-image-eval-memory-temporal-{uuid4()}"
        foreign, _ = await self._write("found", self.base, owner=other_owner)
        self.assertEqual({item.id for item in (await self._search()).results}, {own.id})
        self.assertEqual({item.id for item in (await self._search(owner=other_owner)).results}, {foreign.id})
        context = await self.memory.list_semantic_fact_context(owner_id=self.owner, at=self.now, limit=40)
        self.assertEqual([item["value"] for item in context], ["missing"])

    async def test_context_retains_old_and_new_values_for_temporal_reconciliation(self):
        await self._write("missing", self.base)
        await self._write("found", self.base + timedelta(hours=1), previous="missing")
        context = await self.memory.list_semantic_fact_context(owner_id=self.owner, at=self.now, limit=40)
        self.assertEqual({item["value"] for item in context}, {"missing", "found"})
        self.assertEqual(context[0]["value"], "found")
        self.assertEqual(next(item for item in context if item["value"] == "missing")["status"], "superseded")
