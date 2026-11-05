"""Recall candidates select originals; vendor prose never replaces provenance."""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from app.knowledge.hindsight import HindsightError
from app.knowledge.models import SourceError, SourceVersion


@dataclass(frozen=True)
class SourceRecallScope:
    """Trusted source constraints, never vendor-inferred event metadata."""

    session_id: UUID | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None

    def __post_init__(self):
        if self.session_id is not None and not isinstance(self.session_id, UUID):
            raise ValueError("session_id must be a UUID")
        if (self.start_time is None) != (self.end_time is None):
            raise ValueError("start_time and end_time must be provided together")
        if self.start_time is not None:
            if any(value.tzinfo is None or value.utcoffset() is None
                   for value in (self.start_time, self.end_time)):
                raise ValueError("Source time constraints must be timezone-aware")
            if self.end_time <= self.start_time:
                raise ValueError("end_time must be after start_time")

    def matches(self, source: SourceVersion) -> bool:
        source_session = getattr(source, "session_id", None) or source.context_session_id
        if self.session_id is not None and source_session != self.session_id:
            return False
        if self.start_time is not None:
            return source.source_time is not None and self.start_time <= source.source_time < self.end_time
        return True


class SourceMemoryRetrieval:
    def __init__(self, sources, provider, *, max_sources=8, max_characters=48000, timeout_seconds=8.0):
        if type(max_sources) is not int or not 1 <= max_sources <= 32 or max_characters < 2000 or timeout_seconds <= 0:
            raise ValueError("Invalid source retrieval bounds")
        self.sources, self.provider = sources, provider
        self.max_sources, self.max_characters, self.timeout_seconds = max_sources, max_characters, timeout_seconds

    @staticmethod
    def _candidates(recall, coverage):
        """Follow native observation-to-fact links once, never an arbitrary graph."""
        facts = recall.get("source_facts") or {}
        facts = facts if isinstance(facts, dict) else {}
        coverage["source_facts_truncated"] = recall.get("source_facts_truncated") is True
        for rank, item in enumerate(recall["results"][:64], start=1):
            if not isinstance(item, dict):
                coverage["rejected_candidates"] += 1
                continue
            scores = item.get("scores")
            scores = {key: value for key, value in scores.items()
                      if key in {"final", "reranker", "semantic", "keyword"}
                      and type(value) in {int, float} and -1e9 <= value <= 1e9 and math.isfinite(value)} if isinstance(scores, dict) else {}
            match = {"candidate_rank": rank, "native_scores": scores}
            document = item.get("document_id")
            direct = isinstance(document, str) and bool(document)
            if direct:
                yield document, match | {"basis": "direct_document"}
            links = item.get("source_fact_ids")
            if item.get("type") != "observation" or not isinstance(links, list) or not links:
                if not direct:
                    coverage["rejected_candidates"] += 1
                continue
            coverage["omitted_supporting_facts"] += max(0, len(links) - 16)
            seen = set()
            for identity in links[:16]:
                if not isinstance(identity, str) or not identity or identity in seen:
                    if not isinstance(identity, str) or not identity:
                        coverage["missing_supporting_facts"] += 1
                    continue
                seen.add(identity)
                fact = facts.get(identity)
                # Supporting facts must be leaves. Nested observations, cycles,
                # mismatched IDs and absent optional facts never authorize reads.
                if (not isinstance(fact, dict) or fact.get("id") != identity
                        or fact.get("type") not in {"world", "experience"}
                        or fact.get("source_fact_ids")
                        or not isinstance(fact.get("document_id"), str) or not fact["document_id"]):
                    coverage["missing_supporting_facts"] += 1
                    continue
                yield fact["document_id"], match | {"basis": "observation_support"}

    async def recall(self, owner_id, area_id, query, *, scope: SourceRecallScope | None = None):
        started = time.monotonic()
        scope = scope or SourceRecallScope()
        async with self.sources.transaction(owner_id) as connection:
            await self.sources._area(connection, owner_id, area_id)
            bank = await connection.fetchval(
                "select bank_name from jarvis_memory_banks where owner_id=$1 and area_id=$2", owner_id, area_id,
            )
        result = {"status": "available", "sources": [], "rejected_candidates": 0,
                  "omitted_sources": 0, "omitted_content": 0, "filtered_candidates": 0,
                  "policy_excluded_candidates": 0,
                  "unscanned_candidates": 0, "unscanned_documents": 0,
                  "missing_supporting_facts": 0, "omitted_supporting_facts": 0,
                  "source_facts_truncated": False, "elapsed_ms": 0, "vendor_usage": None}
        if bank is None:
            result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
            return result
        verified = []
        try:
            async with asyncio.timeout(self.timeout_seconds):
                recall = await self.provider.recall(bank, query)
                result["unscanned_candidates"] = max(0, len(recall["results"]) - 64)
                seen = set()
                for document_id, match in self._candidates(recall, result):
                    if document_id in seen:
                        continue
                    seen.add(document_id)
                    if len(seen) > 64:
                        result["unscanned_documents"] += 1
                        continue
                    try:
                        source = await self.sources.resolve_document(owner_id, area_id, document_id)
                        if source.bank_name != bank:
                            raise SourceError("knowledge_bank_mismatch", "The candidate is outside the selected bank.")
                        if source.semantic_index_exclusion is not None:
                            result["policy_excluded_candidates"] += 1
                            continue
                        if not scope.matches(source):
                            result["filtered_candidates"] += 1
                            continue
                        if len(verified) >= self.max_sources:
                            result["omitted_sources"] += 1
                            continue
                        document = await self.provider.document(bank, document_id)
                        self.provider.verify_document(source, document)
                        verified.append((source, match))
                    except (SourceError, HindsightError) as error:
                        if isinstance(error, HindsightError) and error.code != "hindsight_original_evidence_mismatch":
                            raise
                        result["rejected_candidates"] += 1
                # Recheck all originals after the final network await, not just
                # the source that happened to be fetched last.
                for source, match in verified:
                    try:
                        fresh = await self.sources.revalidate(source)
                        if fresh.semantic_index_exclusion is not None:
                            result["policy_excluded_candidates"] += 1
                            continue
                        if not scope.matches(fresh):
                            result["filtered_candidates"] += 1
                            continue
                        candidate = fresh.original_evidence() | {"content_included": True, "retrieval_match": match}
                        tentative = result | {"sources": [*result["sources"], candidate]}
                        # Reserve room for counters that may grow after this candidate.
                        if len(json.dumps(tentative, ensure_ascii=True)) > self.max_characters - 128:
                            candidate.pop("content")
                            candidate.update(content_included=False, content_omission_reason="character_budget")
                            tentative = result | {"sources": [*result["sources"], candidate]}
                        if len(json.dumps(tentative, ensure_ascii=True)) <= self.max_characters - 128:
                            result["sources"].append(candidate)
                            result["omitted_content"] += not candidate["content_included"]
                        else:
                            result["omitted_sources"] += 1
                    except SourceError:
                        result["rejected_candidates"] += 1
        except (HindsightError, TimeoutError):
            return result | {"status": "unavailable", "sources": [], "reason": "memory_provider_unavailable",
                             "elapsed_ms": round((time.monotonic() - started) * 1000)}
        result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return result
