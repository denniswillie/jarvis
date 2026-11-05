"""Production memory handlers with a deliberately owner-scoped fixture worker.

Only source evidence and the queue claim are fixture-specific. Generation,
promotion, embedding, reconciliation, reflection, and retrieval are production
implementations. Never start the global worker against a shared database here.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.experience.repository import ExperienceRepository
from app.memory.consolidation.job_handler import ConsolidateExperienceEventJobHandler
from app.memory.consolidation.service import ExperienceConsolidationService
from app.memory.embeddings import ExperienceEventEmbeddingService
from app.memory.embeddings.job_handlers import (
    EmbedExperienceEventJobHandler, EmbedMemoryEntityJobHandler, EmbedMemoryItemJobHandler,
)
from app.memory.episodic.job_handler import PromoteExperienceEventJobHandler
from app.memory.episodic.repository import EpisodicMemoryRepository
from app.memory.episodic.service import EpisodePromotionService
from app.memory.jobs.repository import MemoryJobRepository
from app.memory.projection import (
    MemoryProjectionLimits, MemoryProjectionRepository, MemoryProjectionService,
    RefreshMemoryProfileJobHandler,
)
from app.memory.reflection.job_handler import GenerateReflectionJobHandler
from app.memory.reflection.repository import ReflectionRepository
from app.memory.reflection.service import ReflectionGenerationService, ReflectionTriggerService
from app.memory.repository import MemoryRepository
from app.memory.retrieval.candidates import RetrievalCandidateCollector
from app.memory.retrieval.service import HierarchicalMemoryRetrievalService
from app.memory.semantic.job_handler import ReconcileSemanticMemoryJobHandler
from app.memory.semantic.reconciler import SemanticMemoryReconciler
from app.memory.semantic.repository import SemanticMemoryRepository
from app.perception.models import VisualObservation
from app.perception.repository import PerceptionRepository
from app.preferences.repository import PreferenceRepository
from app.sessions.repository import SessionRepository
from evals.glasses_reliability.memory_grounding_probe import Case


class FixtureLedger:
    """Manifest fresh owners before allowing any database write."""

    def __init__(self, path: Path):
        self.path = path
        self.owner = f"direct-image-eval-memory-pipeline-{uuid4()}"
        self.other_owner = f"direct-image-eval-memory-pipeline-{uuid4()}"
        self.owners = frozenset((self.owner, self.other_owner))
        self.data = {
            "schema": "jarvis.memory_pipeline_fixture.v1", "owners": sorted(self.owners),
            "status": "prepared", "sessions": [], "events": [],
            "started_at": datetime.now(UTC).isoformat(),
        }
        self.persist(exclusive=True)

    def require_owner(self, owner: str) -> None:
        if owner not in self.owners:
            raise ValueError("Owner is not in this run's fresh fixture manifest")

    def persist(self, *, exclusive: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.data, indent=2) + "\n"
        if exclusive:
            # Refuse reuse before any database write can obscure an older run.
            with self.path.open("x", encoding="utf-8") as output:
                output.write(payload)
        else:
            pending = self.path.with_suffix(".pending")
            pending.write_text(payload, encoding="utf-8")
            pending.replace(self.path)


class PipelineRuntime:
    def __init__(self, *, pool, settings, client, embedding_model, ledger: FixtureLedger):
        self.pool = pool
        self.ledger = ledger
        self.jobs = MemoryJobRepository(pool)
        self.sessions = SessionRepository(pool)
        self.memory = MemoryRepository(pool)
        self.perception = PerceptionRepository(pool)
        self.experience = ExperienceRepository(pool, consolidation_job_repository=self.jobs)
        episodic = EpisodicMemoryRepository(
            pool, experience_repository=self.experience, memory_repository=self.memory,
        )
        semantic = SemanticMemoryRepository(
            pool, supersede_min_confidence=settings.MEMORY_SEMANTIC_SUPERSEDE_MIN_CONFIDENCE,
        )
        embedding_service = ExperienceEventEmbeddingService(self.memory, embedding_model)
        reflection = ReflectionRepository(pool)
        reflection_trigger = ReflectionTriggerService(
            pool=pool, repository=reflection, job_repository=self.jobs,
            minimum_episodes=settings.MEMORY_REFLECTION_MIN_NEW_EPISODES,
            maximum_episodes=settings.MEMORY_REFLECTION_MAX_EPISODES,
        )
        self.profile = MemoryProjectionService(
            repository=MemoryProjectionRepository(pool), limits=MemoryProjectionLimits(
                max_characters=settings.MEMORY_PROFILE_MAX_CHARACTERS,
                section_item_limit=settings.MEMORY_PROFILE_SECTION_ITEM_LIMIT,
                recent_episode_limit=settings.MEMORY_PROFILE_RECENT_EPISODE_LIMIT,
                superseded_history_limit=settings.MEMORY_PROFILE_SUPERSEDED_HISTORY_LIMIT,
            ),
        )
        handlers = (
            ConsolidateExperienceEventJobHandler(
                repository=self.experience, service=ExperienceConsolidationService(
                    experience_repository=self.experience, perception_repository=self.perception,
                    client=client, image_loader=None, owner_working_memory_reader=None,
                    active_reminder_reader=None, semantic_context_reader=self.memory,
                    max_keyframes=settings.MEMORY_CONSOLIDATION_MAX_KEYFRAMES,
                    max_transcripts=settings.MEMORY_CONSOLIDATION_MAX_TRANSCRIPTS,
                ), candidate_lease_seconds=settings.EVENT_CANDIDATE_RETENTION_SECONDS,
                promotion_job_repository=self.jobs,
            ),
            PromoteExperienceEventJobHandler(
                repository=episodic, service=EpisodePromotionService(
                    minimum_importance=settings.MEMORY_EPISODE_PROMOTION_MIN_IMPORTANCE,
                    important_keyframe_minimum_importance=settings.MEMORY_IMPORTANT_KEYFRAME_MIN_IMPORTANCE,
                    final_keyframe_limit=settings.EVENT_KEYFRAME_MAX_COUNT,
                ), embedding_job_repository=self.jobs, reflection_trigger_service=reflection_trigger,
                ordinary_keyframe_retention_seconds=settings.EVENT_ORDINARY_KEYFRAME_RETENTION_SECONDS,
                skipped_event_retention_days=settings.MEMORY_SKIPPED_EVENT_RETENTION_DAYS,
            ),
            EmbedExperienceEventJobHandler(
                repository=episodic, service=embedding_service, image_loader=None, job_repository=self.jobs,
            ),
            EmbedMemoryItemJobHandler(repository=episodic, service=embedding_service, job_repository=self.jobs),
            ReconcileSemanticMemoryJobHandler(
                repository=semantic, reconciler=SemanticMemoryReconciler(
                    minimum_inferred_confidence=settings.MEMORY_SEMANTIC_INFERENCE_MIN_CONFIDENCE,
                ), job_repository=self.jobs, embedding_model=embedding_model,
            ),
            EmbedMemoryEntityJobHandler(repository=semantic, model=embedding_model),
            GenerateReflectionJobHandler(
                repository=reflection, service=ReflectionGenerationService(client),
                preference_repository=PreferenceRepository(pool), job_repository=self.jobs,
                minimum_preference_episodes=settings.MEMORY_INFERRED_PREFERENCE_MIN_SUPPORTING_EPISODES,
                minimum_preference_confidence=settings.MEMORY_INFERRED_PREFERENCE_MIN_CONFIDENCE,
                minimum_procedure_episodes=settings.MEMORY_PROCEDURE_MIN_SUPPORTING_EPISODES,
            ),
            RefreshMemoryProfileJobHandler(self.profile),
        )
        self.handlers = {handler.job_type: handler for handler in handlers}
        self.retrieval = HierarchicalMemoryRetrievalService(
            memory_repository=self.memory, perception_repository=self.perception,
            candidate_collector=RetrievalCandidateCollector(
                memory_repository=self.memory, perception_repository=self.perception,
                embedding_model=embedding_model,
            ), candidate_limit=settings.MEMORY_RETRIEVAL_CANDIDATE_LIMIT,
            result_limit=settings.MEMORY_RETRIEVAL_RESULT_LIMIT,
        )

    async def seed_event(self, *, owner: str, case: Case, ended_at: datetime):
        self.ledger.require_owner(owner)
        session = await self.sessions.create_session(owner, f"Synthetic memory fixture: {case.name}")
        self.ledger.data["sessions"].append(str(session.id))
        self.ledger.persist()
        started_at = ended_at - timedelta(minutes=1)
        event = await self.experience.get_or_create_open_event(
            owner_id=owner, session_id=session.id, event_kind="mixed", started_at=started_at,
            modalities=("text", "visual") if case.user_text else ("visual",),
        )
        self.ledger.data["events"].append(str(event.id))
        self.ledger.persist()
        sources = []
        for role, content, offset in (("user", case.user_text, 10), ("assistant", case.assistant_text, 30)):
            if not content:
                continue
            interrupted = role == "assistant" and case.interrupted
            metadata = {
                "experience_eligible": not interrupted,
                "completion_state": "interrupted" if interrupted else "finalized" if role == "user" else "completed",
            }
            if role == "assistant":
                metadata.update(delivery_status="incomplete" if interrupted else "forwarded", playback_status="unconfirmed")
            message = await self.sessions.create_message(
                owner_id=owner, session_id=session.id, role=role,
                modality="audio" if role == "assistant" else "text", content=content, metadata=metadata,
            )
            # Synthetic event time is independent of insertion order, including late evidence.
            async with self.pool.acquire() as connection:
                await connection.execute(
                    "update jarvis_messages set created_at = $3 where owner_id = $1 and id = $2",
                    owner, message.id, started_at + timedelta(seconds=offset),
                )
            if not interrupted:
                await self.experience.attach_evidence(
                    owner_id=owner, session_id=session.id, event_id=event.id,
                    evidence_type="message", evidence_ref_id=message.id, source_sequence=offset,
                )
            sources.append({"role": role, "content": content, "metadata": metadata, "attached": not interrupted})
        observation = await self.perception.insert_observation(
            observation=VisualObservation(
                id=uuid4(), owner_id=owner, session_id=session.id, source="glasses",
                started_at=started_at + timedelta(seconds=20), ended_at=ended_at,
                primary_visual_frame_id=None, summary=case.observation, scene=None, activity=None,
                entities=(), visible_text=(), state_changes=(), memory_tags=(), keyframe_candidates=(),
                interrupt_candidate=None, starts_new_event=False, boundary_reason=None,
                confidence=0.95, analyzer_provider="synthetic_fixture", analyzer_model="not_a_vision_test",
                analyzer_contract_version=1, compacted_at=None, created_at=ended_at,
            ), frame_links=(),
        )
        await self.experience.attach_evidence(
            owner_id=owner, session_id=session.id, event_id=event.id,
            evidence_type="observation", evidence_ref_id=observation.id, source_sequence=20,
        )
        await self.experience.close_event(
            owner_id=owner, session_id=session.id, event_id=event.id, ended_at=ended_at,
            boundary_reason="synthetic_fixture_closed", keyframes_finalized_at=ended_at,
            consolidation_ready_at=ended_at,
        )
        return event, {"messages": sources, "observation": case.observation}

    async def claim_owned_job(self, owner: str):
        self.ledger.require_owner(owner)
        async with self.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                """with candidate as (
                    select id from memory_jobs
                    where owner_id = $1 and status = 'queued'
                      and available_at <= now() and attempt_count < max_attempts
                    order by priority desc, available_at, created_at, id
                    for update skip locked limit 1
                ) update memory_jobs jobs set status = 'claimed',
                    attempt_count = jobs.attempt_count + 1, claimed_at = now(),
                    claimed_by = 'memory-pipeline-fixture', claim_token = gen_random_uuid(), updated_at = now()
                from candidate where jobs.id = candidate.id and jobs.owner_id = $1
                returning jobs.job_type, jobs.dedupe_key""", owner,
            )
        if row is None:
            return None
        return await self.jobs.get_by_dedupe_key(
            owner_id=owner, job_type=row["job_type"], dedupe_key=row["dedupe_key"],
        )

    async def drain(self, *, owner: str, journal: list[dict[str, Any]], persist) -> None:
        for _ in range(100):
            job = await self.claim_owned_job(owner)
            if job is None:
                async with self.pool.acquire() as connection:
                    pending = await connection.fetchval(
                        "select count(*) from memory_jobs where owner_id = $1 and status <> 'completed'", owner,
                    )
                if pending:
                    raise RuntimeError("Fixture has unfinished jobs; no automatic retry or global recovery was run")
                return
            if job.owner_id != owner or job.claim_token is None or job.job_type not in self.handlers:
                raise ValueError("Unexpected fixture claim or unsupported production job type")
            record = {"id": str(job.id), "job_type": job.job_type, "status": "claimed", "payload": dict(job.payload)}
            journal.append(record)
            persist()
            handler = self.handlers[job.job_type]
            prepared = await handler.prepare(job)
            async with self.pool.acquire() as connection, connection.transaction():
                await self.jobs.lock_claim(
                    owner_id=owner, job_id=job.id, claim_token=job.claim_token, connection=connection,
                )
                result = await handler.commit(job, prepared, connection)
                await self.jobs.complete(
                    owner_id=owner, job_id=job.id, claim_token=job.claim_token,
                    completed_at=datetime.now(UTC), result=result, connection=connection,
                )
            record.update(status="completed", result=dict(result))
            persist()
        raise RuntimeError("Fixture job budget exhausted")
