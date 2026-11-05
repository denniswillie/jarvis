from __future__ import annotations

import asyncio
import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from app.memory.consolidation.job_handler import (
    CONSOLIDATE_EXPERIENCE_EVENT_JOB,
    ConsolidateExperienceEventJobHandler,
    ExperienceConsolidationRecoveryRule,
    consolidation_dedupe_key,
)
from app.memory.consolidation.openai_client import ConsolidationProviderTransientError
from app.memory.consolidation.service import PreparedConsolidation
from app.memory.jobs.handlers import MemoryJobTransientError, MemoryJobValidationError
from app.memory.jobs.models import MemoryJob, MemoryJobEnqueueResult
from test_consolidation_contracts import _cited_output, _output, _request
from test_consolidation_service import _event, _service_with_responses


class ConsolidationJobHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_commit_persists_validated_result(self) -> None:
        job = _job()
        event = replace(
            _event(datetime.now(UTC)),
            id=UUID(str(job.payload["event_id"])),
            session_id=job.session_id,
        )
        repository = AsyncMock()
        repository.begin_consolidation.return_value = event
        repository.store_consolidation_result.return_value = _consolidated_event(event)
        promotion_jobs = AsyncMock()
        service = AsyncMock()
        service.provider = "openai"
        service.model = "test-gpt"
        prepared = _prepared(event)
        service.consolidate.return_value = prepared
        handler = ConsolidateExperienceEventJobHandler(
            repository=repository,
            service=service,
            candidate_lease_seconds=1800,
            promotion_job_repository=promotion_jobs,
        )

        result = await handler.prepare(job)
        connection = AsyncMock()
        committed = await handler.commit(job, result, connection)

        self.assertEqual(committed["event_status"], "consolidated")
        repository.renew_event_candidate_leases.assert_awaited_once()
        repository.store_consolidation_result.assert_awaited_once()
        promotion_jobs.enqueue.assert_awaited_once()
        self.assertIs(
            promotion_jobs.enqueue.await_args.kwargs["connection"],
            connection,
        )

    async def test_retryable_provider_failure_restores_closed_state(self) -> None:
        job = _job(attempt_count=1, max_attempts=3)
        event = replace(
            _event(datetime.now(UTC)),
            id=UUID(str(job.payload["event_id"])),
            session_id=job.session_id,
        )
        repository = AsyncMock()
        repository.begin_consolidation.return_value = event
        service = AsyncMock()
        service.provider = "openai"
        service.model = "test-gpt"
        service.consolidate.side_effect = ConsolidationProviderTransientError("busy")
        handler = ConsolidateExperienceEventJobHandler(
            repository=repository,
            service=service,
            candidate_lease_seconds=1800,
        )

        with self.assertRaises(MemoryJobTransientError):
            await handler.prepare(job)

        repository.finish_consolidation_failure.assert_awaited_once_with(
            owner_id=job.owner_id,
            session_id=job.session_id,
            event_id=event.id,
            terminal=False,
        )

    async def test_recovery_enqueues_missing_job_and_renews_lease(self) -> None:
        event = _event(datetime.now(UTC))
        connection = AsyncMock()
        connection.fetch.return_value = [
            {
                "id": event.id,
                "owner_id": event.owner_id,
                "session_id": event.session_id,
                "status": "closed",
                "consolidation_job_status": None,
            }
        ]
        repository = AsyncMock()
        repository.enqueue.return_value = MemoryJobEnqueueResult(
            job=_queued_job(event),
            created=True,
        )
        rule = ExperienceConsolidationRecoveryRule(candidate_lease_seconds=1800)

        changed = await rule.reconcile(
            repository=repository,
            connection=connection,
            now=datetime.now(UTC),
        )

        self.assertEqual(changed, 1)
        repository.enqueue.assert_awaited_once()
        self.assertTrue(connection.execute.await_count >= 1)


    async def test_invalid_correction_is_terminal_without_persistence_or_promotion(self):
        job = _job()
        request = _request()
        event = replace(
            request.event, id=UUID(job.payload["event_id"]),
            owner_id=job.owner_id, session_id=job.session_id,
        )
        request = replace(request, event=event)
        bad = json.dumps(_cited_output(request, text="private unsupported quote"))
        service = _service_with_responses(request, bad, bad)
        repository = AsyncMock()
        repository.begin_consolidation.return_value = event
        promotion_jobs = AsyncMock()
        handler = ConsolidateExperienceEventJobHandler(
            repository=repository, service=service, candidate_lease_seconds=1800,
            promotion_job_repository=promotion_jobs,
        )
        with self.assertRaises(MemoryJobValidationError) as caught:
            await handler.prepare(job)
        self.assertEqual(service._client.generate.await_count, 2)
        self.assertIn("missing_literal_quotation", str(caught.exception))
        self.assertIn("model_generation=2", str(caught.exception))
        self.assertNotIn("private unsupported quote", str(caught.exception))
        repository.finish_consolidation_failure.assert_awaited_once_with(
            owner_id=job.owner_id, session_id=job.session_id,
            event_id=event.id, terminal=True,
        )
        repository.store_consolidation_result.assert_not_awaited()
        promotion_jobs.enqueue.assert_not_awaited()

    async def test_cancelled_generation_does_not_become_a_terminal_failure(self):
        job = _job()
        event = replace(
            _event(datetime.now(UTC)), id=UUID(job.payload["event_id"]),
            session_id=job.session_id,
        )
        repository = AsyncMock()
        repository.begin_consolidation.return_value = event
        service = AsyncMock()
        service.consolidate.side_effect = asyncio.CancelledError()
        handler = ConsolidateExperienceEventJobHandler(
            repository=repository, service=service, candidate_lease_seconds=1800,
        )
        with self.assertRaises(asyncio.CancelledError):
            await handler.prepare(job)
        repository.finish_consolidation_failure.assert_not_awaited()
        repository.store_consolidation_result.assert_not_awaited()

    async def test_already_consolidated_event_does_not_call_provider_again(self):
        job = _job()
        event = _consolidated_event(replace(
            _event(datetime.now(UTC)), id=UUID(job.payload["event_id"]),
            session_id=job.session_id,
        ))
        repository = AsyncMock()
        repository.begin_consolidation.return_value = event
        service = AsyncMock()
        handler = ConsolidateExperienceEventJobHandler(
            repository=repository, service=service, candidate_lease_seconds=1800,
        )
        prepared = await handler.prepare(job)
        self.assertEqual(prepared.event_id, event.id)
        service.consolidate.assert_not_awaited()
        repository.renew_event_candidate_leases.assert_not_awaited()


def _job(*, attempt_count: int = 1, max_attempts: int = 3) -> MemoryJob:
    now = datetime.now(UTC)
    event_id = uuid4()
    return MemoryJob(
        id=uuid4(),
        owner_id="owner",
        session_id=uuid4(),
        job_type=CONSOLIDATE_EXPERIENCE_EVENT_JOB,
        dedupe_key=consolidation_dedupe_key(event_id),
        payload={"event_id": str(event_id)},
        status="claimed",
        priority=0,
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        available_at=now,
        claimed_at=now,
        claimed_by="worker",
        claim_token=uuid4(),
        result=None,
        last_error=None,
        completed_at=None,
        created_at=now,
        updated_at=now,
    )


def _queued_job(event) -> MemoryJob:  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    return MemoryJob(
        id=uuid4(),
        owner_id=event.owner_id,
        session_id=event.session_id,
        job_type=CONSOLIDATE_EXPERIENCE_EVENT_JOB,
        dedupe_key=consolidation_dedupe_key(event.id),
        payload={"event_id": str(event.id)},
        status="queued",
        priority=0,
        attempt_count=0,
        max_attempts=3,
        available_at=now,
        claimed_at=None,
        claimed_by=None,
        claim_token=None,
        result=None,
        last_error=None,
        completed_at=None,
        created_at=now,
        updated_at=now,
    )


def _prepared(event) -> PreparedConsolidation:  # type: ignore[no-untyped-def]
    request = _request()
    from app.memory.consolidation.contracts import parse_consolidation_result

    return PreparedConsolidation(
        event_id=event.id,
        owner_id=event.owner_id,
        session_id=event.session_id,
        result=parse_consolidation_result(_output(request), request=request),
        provider="openai",
        model="test-gpt",
    )


def _consolidated_event(event):  # type: ignore[no-untyped-def]
    from dataclasses import replace

    prepared = _prepared(event)
    return replace(
        event,
        status="consolidated",
        title=prepared.result.event_title,
        summary=prepared.result.event_summary,
        importance=prepared.result.episode_importance,
        confidence=prepared.result.confidence,
        consolidation_result=prepared.result.model_dump(mode="json"),
        consolidation_provider="openai",
        consolidation_model="test-gpt",
        consolidated_at=datetime.now(UTC),
    )
