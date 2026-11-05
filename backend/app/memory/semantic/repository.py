from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, cast
from uuid import UUID

import asyncpg

from app.memory.consolidation.contracts import ConsolidationResult
from app.memory.embeddings import PreparedEmbeddingWrite
from app.memory.entities.models import MemoryEntity
from app.memory.forgetting import (
    semantic_fact_identity_fingerprint,
    suppress_automatic_memory_recreation,
)
from app.memory.models import EMBEDDING_DIMENSIONS, MemoryItem, content_hash, normalize_embedding
from app.memory.semantic.contracts import (
    ENTITY_IDENTITY_EMBEDDING_KIND,
    SemanticEntityIdentity,
    SemanticEvidence,
    SemanticFactProposal,
    SemanticReconciliationContext,
    SemanticReconciliationOutcome,
    SemanticReconciliationPlan,
    SemanticRelationProposal,
)
from app.memory.semantic.contradictions import (
    ActiveSemanticFact,
    SemanticContradictionPolicy,
    semantic_fact_conflict_domain,
)
from app.memory.semantic.entity_resolver import ResolvedSemanticEntity, SemanticEntityResolver


_EXCLUSIVE_RELATION_PREDICATES = frozenset({"located_at", "lives_at", "works_at"})


class SemanticMemoryRepository:
    """Transaction-owned persistence for derived semantic facts and entities."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        supersede_min_confidence: float = 0.90,
    ) -> None:
        self._pool = pool
        self._contradiction_policy = SemanticContradictionPolicy(
            supersede_min_confidence=supersede_min_confidence
        )

    async def load_context(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
    ) -> SemanticReconciliationContext:
        async with self._pool.acquire() as connection:
            event = await connection.fetchrow(
                """
                select event.*, episode.id as episode_memory_id
                from experience_events event
                join memory_items episode
                  on episode.id = event.promoted_episode_memory_id
                 and episode.owner_id = event.owner_id
                 and episode.memory_type = 'episode'
                where event.owner_id = $1
                  and event.session_id = $2
                  and event.id = $3
                  and event.status = 'consolidated'
                  and event.promotion_decision = 'promoted'
                  and exists (
                      select 1
                      from memory_item_embeddings embedding
                      where embedding.memory_item_id = episode.id
                  )
                """,
                owner_id,
                session_id,
                event_id,
            )
            if event is None:
                raise ValueError("embedded promoted episode was not found")
            evidence_rows = await connection.fetch(
                """
                select evidence.evidence_type,
                       evidence.evidence_ref_id,
                       evidence.occurred_at,
                       coalesce(memory_evidence.excerpt, '') as excerpt,
                       messages.role as message_role
                from experience_event_evidence evidence
                left join memory_evidence
                  on memory_evidence.memory_item_id = $4
                 and memory_evidence.evidence_type = evidence.evidence_type
                 and memory_evidence.evidence_ref_id = evidence.evidence_ref_id::text
                left join jarvis_messages messages
                  on evidence.evidence_type = 'message'
                 and messages.id = evidence.evidence_ref_id
                 and messages.owner_id = $1
                 and messages.session_id = $2
                where evidence.event_id = $3
                order by evidence.occurred_at asc, evidence.id asc
                """,
                owner_id,
                session_id,
                event_id,
                event["episode_memory_id"],
            )
            tombstone_rows = await connection.fetch(
                """
                select identity_fingerprint
                from memory_forgetting_tombstones
                where owner_id = $1
                  and memory_type = 'semantic_fact'
                  and overridden_at is null
                """,
                owner_id,
            )
        result = ConsolidationResult.model_validate(_json_object(event["consolidation_result"]))
        return SemanticReconciliationContext(
            owner_id=owner_id,
            session_id=session_id,
            event_id=event_id,
            episode_memory_id=event["episode_memory_id"],
            occurred_at=event["ended_at"] or event["started_at"],
            event_started_at=event["started_at"],
            result=result,
            evidence=tuple(
                SemanticEvidence(
                    evidence_type=row["evidence_type"],
                    evidence_ref_id=row["evidence_ref_id"],
                    occurred_at=row["occurred_at"],
                    excerpt=row["excerpt"],
                    message_role=row["message_role"],
                )
                for row in evidence_rows
            ),
            semantic_reconciled_at=event["semantic_reconciled_at"],
            suppressed_memory_fingerprints=frozenset(
                row["identity_fingerprint"] for row in tombstone_rows
            ),
        )

    async def requires_embedding_lookup(
        self,
        *,
        owner_id: str,
        identity: SemanticEntityIdentity,
        provider: str,
        model: str,
    ) -> bool:
        if identity.entity_type == "person" or identity.external_namespace is not None:
            return False
        async with self._pool.acquire() as connection:
            return bool(
                await connection.fetchval(
                    """
                    select not exists (
                        select 1
                        from memory_entities
                        where owner_id = $1
                          and entity_type = $2
                          and (
                              normalized_name = $3
                              or aliases @> array[$3]::text[]
                          )
                    ) and exists (
                        select 1
                        from memory_entities entity
                        join memory_entity_embeddings embedding
                          on embedding.entity_id = entity.id
                        where entity.owner_id = $1
                          and entity.entity_type = $2
                          and embedding.provider = $4
                          and embedding.model = $5
                          and embedding.embedding_kind = $6
                    )
                    """,
                    owner_id,
                    identity.entity_type,
                    identity.normalized_name,
                    provider,
                    model,
                    ENTITY_IDENTITY_EMBEDDING_KIND,
                )
            )

    async def lock_entity_identity(
        self,
        *,
        owner_id: str,
        identity: SemanticEntityIdentity,
        connection: asyncpg.Connection,
    ) -> None:
        await connection.execute(
            "select pg_advisory_xact_lock(hashtext($1), hashtext($2))",
            owner_id,
            f"semantic-entity:{identity.identity_key}",
        )

    async def find_external_identity(
        self,
        *,
        owner_id: str,
        namespace: str,
        external_id: str,
        connection: asyncpg.Connection,
    ) -> MemoryEntity | None:
        row = await connection.fetchrow(
            """
            select * from memory_entities
            where owner_id = $1 and external_namespace = $2 and external_id = $3
            """,
            owner_id,
            namespace,
            external_id,
        )
        return _entity_from_row(row) if row is not None else None

    async def find_exact_entities(
        self,
        *,
        owner_id: str,
        identity: SemanticEntityIdentity,
        connection: asyncpg.Connection,
    ) -> tuple[MemoryEntity, ...]:
        rows = await connection.fetch(
            """
            select * from memory_entities
            where owner_id = $1 and entity_type = $2 and normalized_name = $3
            order by confidence desc, created_at asc, id asc
            limit 2
            """,
            owner_id,
            identity.entity_type,
            identity.normalized_name,
        )
        return tuple(_entity_from_row(row) for row in rows)

    async def find_alias_entities(
        self,
        *,
        owner_id: str,
        identity: SemanticEntityIdentity,
        connection: asyncpg.Connection,
    ) -> tuple[MemoryEntity, ...]:
        rows = await connection.fetch(
            """
            select * from memory_entities
            where owner_id = $1
              and entity_type = $2
              and aliases @> array[$3]::text[]
            order by confidence desc, created_at asc, id asc
            limit 2
            """,
            owner_id,
            identity.entity_type,
            identity.normalized_name,
        )
        return tuple(_entity_from_row(row) for row in rows)

    async def find_embedding_entities(
        self,
        *,
        owner_id: str,
        identity: SemanticEntityIdentity,
        embedding: Sequence[float],
        provider: str,
        model: str,
        minimum_similarity: float,
        connection: asyncpg.Connection,
    ) -> tuple[MemoryEntity, ...]:
        rows = await connection.fetch(
            """
            select entity.*
            from memory_entities entity
            join memory_entity_embeddings embeddings
              on embeddings.entity_id = entity.id
            where entity.owner_id = $1
              and entity.entity_type = $2
              and embeddings.provider = $3
              and embeddings.model = $4
              and embeddings.embedding_kind = $5
              and 1 - (embeddings.embedding <=> $6::vector) >= $7
            order by embeddings.embedding <=> $6::vector, entity.created_at, entity.id
            limit 2
            """,
            owner_id,
            identity.entity_type,
            provider,
            model,
            ENTITY_IDENTITY_EMBEDDING_KIND,
            _vector_literal(normalize_embedding(embedding)),
            minimum_similarity,
        )
        return tuple(_entity_from_row(row) for row in rows)

    async def insert_entity(
        self,
        *,
        owner_id: str,
        identity: SemanticEntityIdentity,
        connection: asyncpg.Connection,
    ) -> MemoryEntity:
        row = await connection.fetchrow(
            """
            insert into memory_entities (
                owner_id, entity_type, canonical_name, normalized_name,
                aliases, confidence, external_namespace, external_id, metadata
            )
            values ($1, $2, $3, $4, $5, $6, $7, $8, '{}'::jsonb)
            returning *
            """,
            owner_id,
            identity.entity_type,
            identity.name,
            identity.normalized_name,
            list(identity.aliases),
            identity.confidence,
            identity.external_namespace,
            identity.external_id,
        )
        if row is None:
            raise RuntimeError("semantic entity insert did not return a row")
        return _entity_from_row(row)

    async def get_entity(self, *, owner_id: str, entity_id: UUID) -> MemoryEntity:
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                "select * from memory_entities where owner_id = $1 and id = $2",
                owner_id,
                entity_id,
            )
        if row is None:
            raise ValueError("semantic entity was not found")
        return _entity_from_row(row)

    async def get_entity_embedding_hash(
        self,
        *,
        owner_id: str,
        entity_id: UUID,
        provider: str,
        model: str,
    ) -> str | None:
        async with self._pool.acquire() as connection:
            return cast(
                str | None,
                await connection.fetchval(
                    """
                    select embedding.content_hash
                    from memory_entity_embeddings embedding
                    join memory_entities entity on entity.id = embedding.entity_id
                    where entity.owner_id = $1
                      and entity.id = $2
                      and embedding.embedding_kind = $3
                      and embedding.provider = $4
                      and embedding.model = $5
                    """,
                    owner_id,
                    entity_id,
                    ENTITY_IDENTITY_EMBEDDING_KIND,
                    provider,
                    model,
                ),
            )

    async def store_entity_embedding(
        self,
        *,
        owner_id: str,
        write: PreparedEmbeddingWrite,
        expected_content_hash: str,
        connection: asyncpg.Connection,
    ) -> bool:
        row = await connection.fetchrow(
            "select * from memory_entities where owner_id = $1 and id = $2 for update",
            owner_id,
            write.subject_id,
        )
        if row is None:
            raise ValueError("entity embedding subject is outside the owner scope")
        from app.memory.semantic.contracts import build_entity_embedding_content

        entity = _entity_from_row(row)
        if content_hash(build_entity_embedding_content(entity)) != expected_content_hash:
            return False
        await connection.execute(
            """
            insert into memory_entity_embeddings (
                entity_id, embedding_kind, provider, model, dimensions,
                embedded_content, content_hash, embedding, embedded_at
            )
            values ($1, $2, $3, $4, $5, $6, $7, $8::vector, now())
            on conflict (entity_id, embedding_kind, provider, model)
            do update set
                dimensions = excluded.dimensions,
                embedded_content = excluded.embedded_content,
                content_hash = excluded.content_hash,
                embedding = excluded.embedding,
                embedded_at = now(),
                updated_at = now()
            where memory_entity_embeddings.content_hash
                is distinct from excluded.content_hash
            """,
            write.subject_id,
            write.embedding_kind,
            write.provider,
            write.model,
            EMBEDDING_DIMENSIONS,
            write.embedded_content,
            content_hash(write.embedded_content),
            _vector_literal(normalize_embedding(write.embedding)),
        )
        return True

    async def apply_plan(
        self,
        *,
        context: SemanticReconciliationContext,
        plan: SemanticReconciliationPlan,
        resolver: SemanticEntityResolver,
        entity_embeddings: Mapping[str, Sequence[float]],
        reconciled_at: datetime,
        connection: asyncpg.Connection,
    ) -> SemanticReconciliationOutcome:
        if not connection.is_in_transaction():
            raise ValueError("semantic reconciliation requires an active transaction")
        if (
            plan.owner_id != context.owner_id
            or plan.event_id != context.event_id
            or plan.episode_memory_id != context.episode_memory_id
        ):
            raise ValueError("semantic plan does not match its source episode")
        event = await connection.fetchrow(
            """
            select * from experience_events
            where owner_id = $1 and session_id = $2 and id = $3
            for update
            """,
            context.owner_id,
            context.session_id,
            context.event_id,
        )
        if (
            event is None
            or event["promotion_decision"] != "promoted"
            or event["promoted_episode_memory_id"] != context.episode_memory_id
        ):
            raise ValueError("semantic episode is outside the owner scope")
        if event["semantic_reconciled_at"] is not None:
            return SemanticReconciliationOutcome(
                created_entities=(),
                created_facts=(),
                updated_facts=(),
                created_relation_count=0,
                already_reconciled=True,
            )

        cache: dict[str, ResolvedSemanticEntity] = {}
        created_entities: dict[UUID, MemoryEntity] = {}
        for identity in plan.entities:
            resolved = await resolver.resolve(
                owner_id=context.owner_id,
                identity=identity,
                connection=connection,
                cache=cache,
                embeddings=entity_embeddings,
            )
            if resolved.created:
                created_entities[resolved.entity.id] = resolved.entity
            await self._attach_mention(
                connection,
                owner_id=context.owner_id,
                entity_id=resolved.entity.id,
                memory_item_id=context.episode_memory_id,
                role="observed",
                confidence=identity.confidence,
            )

        created_facts: list[MemoryItem] = []
        updated_facts: dict[UUID, MemoryItem] = {}
        for proposal in plan.facts:
            subject = await resolver.resolve(
                owner_id=context.owner_id,
                identity=proposal.subject,
                connection=connection,
                cache=cache,
                embeddings=entity_embeddings,
            )
            if subject.created:
                created_entities[subject.entity.id] = subject.entity
            fact, updated = await self._apply_fact_proposal(
                connection,
                context=context,
                subject=subject.entity,
                proposal=proposal,
            )
            if fact is not None:
                created_facts.append(fact)
            updated_facts.update((item.id, item) for item in updated)

        created_relations = 0
        for proposal in plan.relations:
            subject = await resolver.resolve(
                owner_id=context.owner_id,
                identity=proposal.subject,
                connection=connection,
                cache=cache,
                embeddings=entity_embeddings,
            )
            target = await resolver.resolve(
                owner_id=context.owner_id,
                identity=proposal.target,
                connection=connection,
                cache=cache,
                embeddings=entity_embeddings,
            )
            if subject.created:
                created_entities[subject.entity.id] = subject.entity
            if target.created:
                created_entities[target.entity.id] = target.entity
            if await self.apply_relation(
                connection,
                owner_id=context.owner_id,
                supporting_memory_id=context.episode_memory_id,
                subject_entity_id=subject.entity.id,
                object_entity_id=target.entity.id,
                proposal=proposal,
            ):
                created_relations += 1

        await connection.execute(
            """
            update experience_events
            set semantic_reconciled_at = $4, updated_at = now()
            where owner_id = $1 and session_id = $2 and id = $3
            """,
            context.owner_id,
            context.session_id,
            context.event_id,
            reconciled_at,
        )
        return SemanticReconciliationOutcome(
            created_entities=tuple(created_entities.values()),
            created_facts=tuple(created_facts),
            updated_facts=tuple(updated_facts.values()),
            created_relation_count=created_relations,
        )

    async def apply_relation(
        self,
        connection: asyncpg.Connection,
        *,
        owner_id: str,
        supporting_memory_id: UUID,
        subject_entity_id: UUID,
        object_entity_id: UUID,
        proposal: SemanticRelationProposal,
    ) -> bool:
        if subject_entity_id == object_entity_id:
            return False
        references_are_owned = await connection.fetchval(
            """
            select exists (
                select 1 from memory_entities
                where owner_id = $1 and id = $2
            ) and exists (
                select 1 from memory_entities
                where owner_id = $1 and id = $3
            ) and exists (
                select 1 from memory_items
                where owner_id = $1 and id = $4
            )
            """,
            owner_id,
            subject_entity_id,
            object_entity_id,
            supporting_memory_id,
        )
        if not references_are_owned:
            raise ValueError("semantic relation references are outside the owner scope")
        await connection.execute(
            "select pg_advisory_xact_lock(hashtext($1), hashtext($2))",
            owner_id,
            f"semantic-relation:{subject_entity_id}:{proposal.predicate}",
        )
        active = await connection.fetch(
            """
            select * from memory_entity_relations
            where owner_id = $1
              and subject_entity_id = $2
              and predicate = $3
              and status = 'active'
            order by valid_from desc, id
            for update
            """,
            owner_id,
            subject_entity_id,
            proposal.predicate,
        )
        if any(row["object_entity_id"] == object_entity_id for row in active):
            return False
        exclusive = proposal.predicate in _EXCLUSIVE_RELATION_PREDICATES
        if exclusive and any(row["valid_from"] > proposal.valid_from for row in active):
            return False
        status = "active" if proposal.valid_to is None else "retracted"
        relation_id = await connection.fetchval(
            """
            insert into memory_entity_relations (
                owner_id, subject_entity_id, predicate, object_entity_id,
                status, confidence, valid_from, valid_to,
                supporting_memory_id, metadata
            )
            values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
            returning id
            """,
            owner_id,
            subject_entity_id,
            proposal.predicate,
            object_entity_id,
            status,
            proposal.confidence,
            proposal.valid_from,
            proposal.valid_to,
            supporting_memory_id,
            _json_dumps(
                {
                    "source_classification": proposal.source,
                    "evidence": [_evidence_reference(item) for item in proposal.evidence],
                }
            ),
        )
        if relation_id is None:
            raise RuntimeError("semantic relation insert did not return an identity")
        if exclusive and status == "active":
            for previous in active:
                await connection.execute(
                    """
                    update memory_entity_relations
                    set status = 'superseded',
                        valid_to = $3,
                        superseded_by_relation_id = $4,
                        updated_at = now()
                    where owner_id = $1 and id = $2 and status = 'active'
                    """,
                    owner_id,
                    previous["id"],
                    proposal.valid_from,
                    relation_id,
                )
        return True

    async def _apply_fact_proposal(
        self,
        connection: asyncpg.Connection,
        *,
        context: SemanticReconciliationContext,
        subject: MemoryEntity,
        proposal: SemanticFactProposal,
    ) -> tuple[MemoryItem | None, tuple[MemoryItem, ...]]:
        await connection.execute(
            "select pg_advisory_xact_lock(hashtext($1), hashtext($2))",
            context.owner_id,
            (
                f"semantic-fact:{subject.id}:"
                f"{semantic_fact_conflict_domain(proposal.predicate)}"
            ),
        )
        opposite_predicate = (
            "dislikes"
            if proposal.predicate == "likes"
            else "likes" if proposal.predicate == "dislikes" else proposal.predicate
        )
        known_rows = await connection.fetch(
            """
            select * from memory_items
            where owner_id = $1
              and memory_type = 'semantic_fact'
              and status in ('active', 'superseded', 'retracted')
              and metadata->>'subject_entity_id' = $2
              and metadata->>'predicate' = any($3::text[])
              and (
                  ($5 = 'event' and metadata->>'temporal_scope' = 'event'
                   and source_experience_event_id = $6)
                  or ($5 <> 'event' and coalesce(metadata->>'temporal_scope', 'durable') <> 'event'
                      and (valid_to is null or valid_to > $4))
              )
            order by valid_from asc nulls first, created_at asc, id asc
            for update
            """,
            context.owner_id,
            str(subject.id),
            list(dict.fromkeys((proposal.predicate, opposite_predicate))),
            proposal.valid_from,
            proposal.temporal_scope,
            context.event_id,
        )
        known_facts = tuple(
            ActiveSemanticFact(
                id=row["id"],
                predicate=_json_object(row["metadata"])["predicate"],
                value=_json_object(row["metadata"])["value"],
                valid_from=row["valid_from"] or row["occurred_at"] or row["created_at"],
                is_explicit=row["is_explicit"],
                valid_to=row["valid_to"],
                confidence=float(row["confidence"]),
                change_intent=_json_object(row["metadata"]).get("change_intent", "assert"),
                previous_value=_json_object(row["metadata"]).get("previous_value"),
            )
            for row in known_rows
        )
        active_rows = [row for row in known_rows if row["valid_from"] is None or row["valid_from"] <= proposal.valid_from]
        active_ids = {row["id"] for row in active_rows}
        active_facts = tuple(fact for fact in known_facts if fact.id in active_ids)
        proposal = self._contradiction_policy.bound_validity(
            proposal=proposal, known_facts=known_facts,
        )
        decision = self._contradiction_policy.decide(
            proposal=proposal,
            active_facts=active_facts,
        )
        bounded = self._contradiction_policy.bound_validity(
            proposal=proposal, known_facts=known_facts, replaced_ids=decision.supersede_fact_ids,
        )
        if bounded.valid_to != proposal.valid_to:
            proposal = bounded
            decision = self._contradiction_policy.decide(proposal=proposal, active_facts=active_facts)

        existing_by_id = {row["id"]: row for row in active_rows}
        reinforced_fact: MemoryItem | None = None
        if decision.reinforce_fact_id is not None:
            reinforced_row = await connection.fetchrow(
                """
                update memory_items
                set confidence = greatest(confidence, $3),
                    is_explicit = is_explicit or $4,
                    source = case when $4 then 'user_statement' else source end,
                    metadata = case when $4 then
                        metadata || '{"source_classification":"user_statement"}'::jsonb
                        else metadata end,
                    updated_at = now()
                where owner_id = $1 and id = $2 and status in ('active', 'superseded', 'retracted')
                returning *
                """,
                context.owner_id,
                decision.reinforce_fact_id,
                proposal.confidence,
                proposal.is_explicit,
            )
            if reinforced_row is None:
                raise RuntimeError("active semantic fact disappeared during reinforcement")
            reinforced_fact = _memory_from_row(reinforced_row)
            await self._attach_fact_support(
                connection,
                context=context,
                subject=subject,
                fact=reinforced_fact,
                proposal=proposal,
            )

        created_fact = (
            await self._insert_fact(
                connection,
                context=context,
                subject=subject,
                proposal=proposal,
            )
            if decision.create_new
            else None
        )
        if created_fact is not None:
            await self._attach_fact_support(
                connection,
                context=context,
                subject=subject,
                fact=created_fact,
                proposal=proposal,
            )

        replacement_fact = created_fact or reinforced_fact
        updated_ids: list[UUID] = (
            [reinforced_fact.id] if reinforced_fact is not None else []
        )
        for previous_id in decision.supersede_fact_ids:
            if replacement_fact is None or previous_id not in existing_by_id:
                continue
            row = await connection.fetchrow(
                """
                update memory_items
                set status = case when $3 <= now() then 'superseded' else 'active' end,
                    valid_to = $3,
                    superseded_by_memory_id = case when $3 <= now() then $4::uuid else null end,
                    metadata = metadata || $5::jsonb,
                    updated_at = now()
                where owner_id = $1 and id = $2 and status in ('active', 'superseded', 'retracted')
                returning *
                """,
                context.owner_id,
                previous_id,
                proposal.valid_from,
                replacement_fact.id,
                _json_dumps(
                    {
                        "superseded_by_episode_memory_id": str(context.episode_memory_id),
                        "successor_memory_id": str(replacement_fact.id),
                        "transition_effective_at": proposal.valid_from.isoformat(),
                    }
                ),
            )
            if row is not None:
                updated_ids.append(previous_id)
                await self._attach_fact_evidence(
                    connection,
                    owner_id=context.owner_id,
                    fact_id=previous_id,
                    episode_id=context.episode_memory_id,
                    proposal=proposal,
                    occurred_at=context.occurred_at,
                )
                await self._insert_memory_link(
                    connection,
                    owner_id=context.owner_id,
                    source_memory_id=replacement_fact.id,
                    target_memory_id=previous_id,
                    relationship="supersedes",
                    proposal=proposal,
                    supporting_memory_id=context.episode_memory_id,
                    metadata={"reason": decision.reason, "effective_at": proposal.valid_from.isoformat()},
                )

        for previous_id in decision.retract_fact_ids:
            if previous_id not in existing_by_id:
                continue
            row = await connection.fetchrow(
                """
                update memory_items
                set status = case when $3 <= now() then 'retracted' else 'active' end,
                    valid_to = $3,
                    superseded_by_memory_id = null,
                    metadata = metadata || $4::jsonb,
                    updated_at = now()
                where owner_id = $1 and id = $2 and status in ('active', 'superseded', 'retracted')
                returning *
                """,
                context.owner_id,
                previous_id,
                proposal.valid_from,
                _json_dumps(
                    {"retracted_by_episode_memory_id": str(context.episode_memory_id)}
                ),
            )
            if row is not None:
                updated_ids.append(previous_id)
                await self._attach_fact_evidence(
                    connection,
                    owner_id=context.owner_id,
                    fact_id=previous_id,
                    episode_id=context.episode_memory_id,
                    proposal=proposal,
                    occurred_at=context.occurred_at,
                )
                await self._insert_memory_link(
                    connection,
                    owner_id=context.owner_id,
                    source_memory_id=previous_id,
                    target_memory_id=context.episode_memory_id,
                    relationship="retracted_by",
                    proposal=proposal,
                    supporting_memory_id=context.episode_memory_id,
                    metadata={"reason": decision.reason},
                )

        if created_fact is not None:
            for previous_id in decision.contradict_fact_ids:
                if previous_id not in existing_by_id:
                    continue
                await self._insert_memory_link(
                    connection,
                    owner_id=context.owner_id,
                    source_memory_id=created_fact.id,
                    target_memory_id=previous_id,
                    relationship="contradicts",
                    proposal=proposal,
                    supporting_memory_id=context.episode_memory_id,
                    metadata={"reason": decision.reason},
                )

        updated_rows = (
            await connection.fetch(
                """
                select * from memory_items
                where owner_id = $1 and id = any($2::uuid[])
                order by updated_at, id
                """,
                context.owner_id,
                list(dict.fromkeys(updated_ids)),
            )
            if updated_ids
            else ()
        )
        return created_fact, tuple(_memory_from_row(row) for row in updated_rows)

    async def _insert_fact(
        self,
        connection: asyncpg.Connection,
        *,
        context: SemanticReconciliationContext,
        subject: MemoryEntity,
        proposal: SemanticFactProposal,
    ) -> MemoryItem | None:
        resurrection_fingerprint = semantic_fact_identity_fingerprint(
            subject_identity_key=proposal.subject.identity_key,
            predicate=proposal.predicate,
            value=proposal.value,
        )
        if await suppress_automatic_memory_recreation(
            connection,
            owner_id=context.owner_id,
            memory_type="semantic_fact",
            identity_fingerprint=resurrection_fingerprint,
        ) is not None:
            return None
        identity = content_hash(
            _json_dumps(
                {
                    "subject_entity_id": str(subject.id),
                    "predicate": proposal.predicate,
                    "value": " ".join(proposal.value.casefold().split()),
                    "temporal_scope": proposal.temporal_scope,
                }
            )
        )
        title = f"{subject.canonical_name} {proposal.predicate} {proposal.value}"
        row = await connection.fetchrow(
            """
            insert into memory_items (
                owner_id, session_id, memory_type, title, content, summary,
                source_experience_event_id, occurred_at, valid_from, valid_to,
                importance, confidence, source, is_explicit, metadata
            )
            values (
                $1, $2, 'semantic_fact', $3, $3, $3,
                $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb
            )
            returning *
            """,
            context.owner_id,
            context.session_id,
            title,
            context.event_id,
            context.occurred_at,
            proposal.valid_from,
            proposal.valid_to,
            max(context.result.episode_importance, proposal.confidence),
            proposal.confidence,
            proposal.source,
            proposal.is_explicit,
            _json_dumps(
                {
                    "semantic_identity_key": identity,
                    "memory_identity_fingerprint": resurrection_fingerprint,
                    "subject_entity_id": str(subject.id),
                    "predicate": proposal.predicate,
                    "value": proposal.value,
                    "source_classification": proposal.source,
                    "change_intent": proposal.change_intent,
                    "previous_value": proposal.previous_value,
                    "temporal_scope": proposal.temporal_scope,
                    "supporting_episode_memory_id": str(context.episode_memory_id),
                }
            ),
        )
        if row is None:
            raise RuntimeError("semantic fact insert did not return a memory")
        return _memory_from_row(row)

    async def _attach_fact_support(
        self,
        connection: asyncpg.Connection,
        *,
        context: SemanticReconciliationContext,
        subject: MemoryEntity,
        fact: MemoryItem,
        proposal: SemanticFactProposal,
    ) -> None:
        await self._attach_mention(
            connection,
            owner_id=context.owner_id,
            entity_id=subject.id,
            memory_item_id=fact.id,
            role="subject",
            confidence=proposal.confidence,
        )
        await self._attach_fact_evidence(
            connection,
            owner_id=context.owner_id,
            fact_id=fact.id,
            episode_id=context.episode_memory_id,
            proposal=proposal,
            occurred_at=context.occurred_at,
        )
        await self._insert_memory_link(
            connection,
            owner_id=context.owner_id,
            source_memory_id=fact.id,
            target_memory_id=context.episode_memory_id,
            relationship="derived_from",
            proposal=proposal,
            supporting_memory_id=context.episode_memory_id,
            metadata={},
        )

    async def _insert_memory_link(
        self,
        connection: asyncpg.Connection,
        *,
        owner_id: str,
        source_memory_id: UUID,
        target_memory_id: UUID,
        relationship: str,
        proposal: SemanticFactProposal,
        supporting_memory_id: UUID,
        metadata: Mapping[str, Any],
    ) -> None:
        await connection.execute(
            """
            insert into memory_links (
                owner_id, source_memory_id, target_memory_id, relationship,
                confidence, valid_from, supporting_memory_id, metadata
            )
            values ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            on conflict (owner_id, source_memory_id, target_memory_id, relationship)
                where valid_to is null
            do nothing
            """,
            owner_id,
            source_memory_id,
            target_memory_id,
            relationship,
            proposal.confidence,
            proposal.valid_from,
            supporting_memory_id,
            _json_dumps(metadata),
        )

    async def _attach_fact_evidence(
        self,
        connection: asyncpg.Connection,
        *,
        owner_id: str,
        fact_id: UUID,
        episode_id: UUID,
        proposal: SemanticFactProposal,
        occurred_at: datetime,
    ) -> None:
        await self._insert_evidence(
            connection,
            owner_id=owner_id,
            memory_item_id=fact_id,
            evidence_type="memory",
            evidence_ref_id=str(episode_id),
            evidence_time=occurred_at,
            excerpt="Supporting promoted episode",
            confidence=proposal.confidence,
        )
        for evidence in proposal.evidence:
            await self._insert_evidence(
                connection,
                owner_id=owner_id,
                memory_item_id=fact_id,
                evidence_type=evidence.evidence_type,
                evidence_ref_id=str(evidence.evidence_ref_id),
                evidence_time=evidence.occurred_at,
                excerpt=evidence.excerpt,
                confidence=proposal.confidence,
            )

    @staticmethod
    async def _insert_evidence(
        connection: asyncpg.Connection,
        *,
        owner_id: str,
        memory_item_id: UUID,
        evidence_type: str,
        evidence_ref_id: str,
        evidence_time: datetime,
        excerpt: str,
        confidence: float,
    ) -> None:
        await connection.execute(
            """
            insert into memory_evidence (
                owner_id, memory_item_id, evidence_type, evidence_ref_id,
                evidence_time, excerpt, confidence, metadata
            )
            values ($1, $2, $3, $4, $5, $6, $7, '{}'::jsonb)
            on conflict (memory_item_id, evidence_type, evidence_ref_id) do nothing
            """,
            owner_id,
            memory_item_id,
            evidence_type,
            evidence_ref_id,
            evidence_time,
            excerpt.strip()[:1_000],
            confidence,
        )

    @staticmethod
    async def _attach_mention(
        connection: asyncpg.Connection,
        *,
        owner_id: str,
        entity_id: UUID,
        memory_item_id: UUID,
        role: str,
        confidence: float,
    ) -> None:
        references_are_owned = await connection.fetchval(
            """
            with owned_references as (
                select 1
                from memory_entities entity
                join memory_items memory
                  on memory.owner_id = entity.owner_id
                 and memory.id = $3
                where entity.owner_id = $1
                  and entity.id = $2
            ), inserted as (
                insert into memory_entity_mentions (
                    owner_id, entity_id, memory_item_id, role, confidence
                )
                select $1, $2, $3, $4, $5
                from owned_references
                on conflict (entity_id, memory_item_id, role) do nothing
                returning 1
            )
            select exists(select 1 from owned_references)
            """,
            owner_id,
            entity_id,
            memory_item_id,
            role,
            confidence,
        )
        if not references_are_owned:
            raise ValueError("semantic mention references are outside the owner scope")


def _entity_from_row(row: Mapping[str, Any]) -> MemoryEntity:
    return MemoryEntity(
        id=row["id"],
        owner_id=row["owner_id"],
        entity_type=row["entity_type"],
        canonical_name=row["canonical_name"],
        normalized_name=row["normalized_name"],
        aliases=tuple(row["aliases"]),
        description=row["description"],
        confidence=float(row["confidence"]),
        external_namespace=row["external_namespace"],
        external_id=row["external_id"],
        metadata=_json_object(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _memory_from_row(row: Mapping[str, Any]) -> MemoryItem:
    return MemoryItem(
        id=row["id"],
        owner_id=row["owner_id"],
        session_id=row["session_id"],
        memory_type=row["memory_type"],
        status=row["status"],
        title=row["title"],
        content=row["content"],
        summary=row["summary"],
        source_experience_event_id=row["source_experience_event_id"],
        occurred_at=row["occurred_at"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        importance=float(row["importance"]),
        confidence=float(row["confidence"]),
        source=row["source"],
        is_explicit=row["is_explicit"],
        superseded_by_memory_id=row["superseded_by_memory_id"],
        metadata=_json_object(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _json_object(value: Any) -> dict[str, Any]:
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, Mapping):
        raise ValueError("semantic memory JSON must be an object")
    return dict(decoded)


def _json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(format(value, ".17g") for value in values) + "]"


def _evidence_reference(evidence: SemanticEvidence) -> dict[str, str]:
    return {
        "evidence_type": evidence.evidence_type,
        "evidence_ref_id": str(evidence.evidence_ref_id),
    }
