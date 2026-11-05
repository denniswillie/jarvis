"""One bounded ingestion/reconciliation step, suitable for an owned worker."""
from __future__ import annotations

import asyncio

from app.knowledge.hindsight import HindsightError
from app.knowledge.models import SourceError
from app.knowledge.outbox import IngestionClaimLost, RetentionPaused


class IngestionWorker:
    def __init__(self, repository, provider, *, request_deadline=25.0):
        if not 0 < request_deadline < repository.lease_seconds:
            raise ValueError("Provider step must fit within the worker lease")
        self.repository, self.provider = repository, provider
        self.request_deadline = request_deadline

    async def step(self, owner_id, worker_id):
        claim = await self.repository.claim(owner_id, worker_id)
        if claim is None:
            return None
        try:
            async with asyncio.timeout(self.request_deadline):
                if claim.operation == "retain":
                    return await self._retain(claim)
                return await self._delete(claim)
        except IngestionClaimLost:
            return "claim_lost"
        except HindsightError as error:
            try:
                return await self.repository.failure(claim, error.code, retryable=error.retryable)
            except IngestionClaimLost:
                return "claim_lost"
        except TimeoutError:
            try:
                return await self.repository.failure(claim, "memory_step_timeout", retryable=True)
            except IngestionClaimLost:
                return "claim_lost"
        except RetentionPaused:
            try:
                return await self.repository.settle(claim, "pending", reason="knowledge_area_archived")
            except IngestionClaimLost:
                return "claim_lost"
        except SourceError:
            try:
                return await self.repository.settle(claim, "cancelled", reason="source_unavailable")
            except IngestionClaimLost:
                return "claim_lost"
        # Cancellation deliberately leaves a leased intent. Recovery uses the
        # same operation UUID, never an inferred 'definitely not submitted'.

    async def _retain(self, claim):
        # Reconciliation can use retained archived originals through the owned
        # ledger, without opening them to ordinary memory recall.
        source = await self.repository.retention_source(claim)
        if source.generation != claim.generation:
            return await self.repository.settle(claim, "cancelled", reason="source_changed")
        if (getattr(source, "semantic_index_exclusion", None) is not None
                and await self.repository.skip_indexing(claim)):
            return "cancelled"
        await self.provider.ensure_bank(claim.bank_name)
        operation = await self.provider.operation(claim.bank_name, claim.id)
        if operation is None:
            source = await self.repository.prepare_retain(claim)
            await self.provider.retain(source, claim.id)
            return await self.repository.settle(claim, "accepted")
        if operation["status"] in {"pending", "processing"} or operation.get("next_retry_at"):
            return await self.repository.settle(claim, "accepted")
        if operation["status"] != "completed":
            return await self.repository.settle(claim, "failed", reason="hindsight_retention_failed")
        document = await self.provider.document(claim.bank_name, claim.document_id)
        self.provider.verify_document(source, document)
        await self.repository.retention_source(claim)
        return await self.repository.settle(claim, "searchable")

    async def _delete(self, claim):
        if not await self.repository.deletion_allowed(claim):
            return await self.repository.settle(claim, "failed", reason="current_source_cleanup_refused")
        operation_ids = await self.repository.cleanup_operations(claim)
        if operation_ids is None:
            retained = await self.repository.retention_state(claim)
            operation_ids = ()
            if retained is not None and retained["dispatch_count"]:
                operation_ids = await self.provider.terminal_operations(claim.bank_name, retained["id"])
                if operation_ids is None:
                    return await self.repository.settle(claim, "accepted", reason="waiting_for_retention_terminal")
            operation_ids = await self.repository.prepare_cleanup(claim, operation_ids)
        await self.provider.delete_document(claim.bank_name, claim.document_id)
        if not await self.provider.erase_operations(claim.bank_name, operation_ids):
            return await self.repository.settle(claim, "accepted", reason="waiting_for_operation_cleanup")
        # Reconfirm document absence after removing operations, not just before.
        await self.provider.delete_document(claim.bank_name, claim.document_id)
        return await self.repository.settle(claim, "done")

    async def run(self, owner_id, worker_id, stop: asyncio.Event, *, poll_seconds=1.0):
        if poll_seconds <= 0:
            raise ValueError("Invalid ingestion poll interval")
        while not stop.is_set():
            result = await self.step(owner_id, worker_id)
            if result is None:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=poll_seconds)
                except TimeoutError:
                    pass
