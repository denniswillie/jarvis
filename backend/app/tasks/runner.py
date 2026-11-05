from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
from typing import Protocol
from uuid import UUID, uuid4

from app.tasks.actions import TaskActionRepository
from app.tasks.budgets import TaskBudgetRepository
from app.tasks.models import BudgetExceeded, ClaimLost, Reconciliation, TaskClaim, TaskStatus, WorkerOutcome
from app.tasks.repository import TaskRepository


LOGGER = logging.getLogger(__name__)


def _claim_lost_diagnostic(claim, error, *, stage, disposition):
    """Bounded branch evidence, never exception text, source content or paths."""
    try:
        origins = {
            TaskRepository.start_runtime.__code__: "runtime_registration",
            TaskRepository.lock_claim.__code__: "canonical_claim_guard",
            TaskRepository.checkpoint.__code__: "checkpoint",
            TaskRepository.renew.__code__: "lease_renewal",
            TaskRunner._heartbeat.__code__: "heartbeat",
        }
        trace, origin = error.__traceback__, "unavailable"
        for _ in range(32):
            if trace is None:
                break
            if trace.tb_next is None:
                origin = origins.get(trace.tb_frame.f_code, "unavailable")
                break
            trace = trace.tb_next
        evidence = {
            "format": "jarvis.task_claim_lost.v1", "stage": stage,
            "origin": origin, "recovering": bool(claim.recovering),
            "task_id": str(claim.task.id) if isinstance(claim.task.id, UUID) else None,
            "attempt_id": str(claim.attempt_id) if isinstance(claim.attempt_id, UUID) else None,
            "task_revision": claim.task_revision if type(claim.task_revision) is int else None,
            "fence": claim.fence if type(claim.fence) is int else None,
            "disposition": disposition,
        }
        LOGGER.warning("task_claim_lost_diagnostic %s", json.dumps(evidence, sort_keys=True))
    except Exception:
        # Reporting must not change claim ownership or settlement.
        pass


@dataclass(frozen=True, slots=True)
class WorkerContext:
    claim: TaskClaim
    tasks: TaskRepository
    budgets: TaskBudgetRepository
    actions: TaskActionRepository

    @property
    def task(self):
        return self.claim.task

    async def checkpoint(self, payload: dict, *, provider_handle=None):
        await self.tasks.checkpoint(self.claim, payload, provider_handle=provider_handle)

    async def reserve(self, operation_key: str, **amounts):
        return await self.budgets.reserve(self.claim, operation_key, **amounts)

    async def dispatch(self, reservation_id):
        await self.budgets.submit(self.claim, reservation_id, provider_dispatch=True)

    async def settle(self, reservation_id, **amounts):
        await self.budgets.settle(self.task.owner_id, self.task.id, self.claim.attempt_id, reservation_id, **amounts)


class TaskWorkerProvider(Protocol):
    model: str

    async def execute(self, context: WorkerContext) -> WorkerOutcome:
        """Reserve/record dispatch before I/O; checkpoint handles and tool receipts."""
        ...

    async def reconcile(self, context: WorkerContext, *, cancel: bool) -> Reconciliation:
        """Inspect/cancel the existing handle. Never substitute a new submission."""
        ...


class TaskRunner:
    def __init__(self, repository: TaskRepository, providers: dict[str, TaskWorkerProvider], *, worker_id=None,
                 owner_id=None, host_id=None, capabilities=()):
        self.repository = repository
        self.providers = providers
        self.owner_id, self.host_id = owner_id, host_id
        self.capabilities = tuple(capabilities)
        self.worker_id = worker_id or f"task-worker-{uuid4()}"
        self._active: set[asyncio.Task] = set()
        self._stop = asyncio.Event()
        self._closing = False

    def context(self, claim):
        return WorkerContext(claim, self.repository, TaskBudgetRepository(self.repository), TaskActionRepository(self.repository))

    async def run(self):
        try:
            while not self._stop.is_set():
                try:
                    await self.run_cycle()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    LOGGER.warning("background task polling failed error_type=%s", type(error).__name__)
                try:
                    await asyncio.wait_for(self._stop.wait(), self.repository.limits.poll_seconds)
                except TimeoutError:
                    pass
        finally:
            await self.stop()

    async def run_cycle(self):
        for _ in range(max(0, self.repository.limits.global_concurrency - len(self._active))):
            if self._stop.is_set():
                return
            claim = await self.repository.claim_recovery(
                self.worker_id,owner_id=self.owner_id,host_id=self.host_id,backend_kinds=tuple(self.providers),
            )
            if claim is None:
                claim = await self.repository.claim_next(
                    self.worker_id,tuple(self.providers),owner_id=self.owner_id,
                    provider_models={kind:provider.model for kind,provider in self.providers.items()},
                    worker_capabilities=self.capabilities,
                )
            if claim is None:
                return
            task = asyncio.create_task(self._execute(claim))
            self._active.add(task)
            task.add_done_callback(self._finished)

    def _finished(self, task):
        self._active.discard(task)
        if not task.cancelled() and task.exception() is not None:
            LOGGER.warning("background attempt escaped error handling error_type=%s", type(task.exception()).__name__)

    async def _heartbeat(self, claim):
        while True:
            await asyncio.sleep(max(0.05, claim.task.budget.lease_seconds / 3))
            if not await self.repository.renew(claim):
                raise ClaimLost()

    async def _execute(self, claim):
        heartbeat = None
        work = None
        stage = "provider_selection"
        try:
            provider = self.providers.get(claim.task.backend_kind)
            context = self.context(claim)
            if claim.recovering:
                stage = "recovery"
                if claim.provider_state == "not_started":
                    result = Reconciliation("not_started")
                elif provider is None:
                    result = Reconciliation("unknown")
                else:
                    result = await asyncio.wait_for(provider.reconcile(context, cancel=claim.task.pending_status is not None),
                                                    timeout=max(0.1, claim.task.budget.lease_seconds / 2))
                if (result.state == "terminal" and result.outcome is not None
                        and await self._finish_owned_outcome(provider, context, result.outcome)):
                    return
                await self.repository.resolve_recovery(claim, result)
                return
            if provider is None:
                await self.repository.abandon(claim, reason="task_provider_unavailable", target=TaskStatus.PAUSED)
                return
            heartbeat = asyncio.create_task(self._heartbeat(claim))
            work = asyncio.create_task(provider.execute(context))
            stage = "execution_wait"
            remaining = max(0.001, claim.task.budget.active_seconds - claim.task.active_seconds)
            done, _ = await asyncio.wait((heartbeat, work), timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise BudgetExceeded("task_active_budget_reached")
            if heartbeat in done:
                stage = "heartbeat"
                await heartbeat
                raise ClaimLost()
            stage = "provider_execute"
            outcome = await work
            stage = "finish"
            if not await self._finish_owned_outcome(provider, context, outcome):
                await self.repository.finish(claim, outcome)
        except asyncio.CancelledError:
            await asyncio.shield(self.repository.abandon(claim, reason="worker_shutdown"))
            raise
        except ClaimLost as error:
            disposition = "recovery_owner_retained"
            try:
                if not claim.recovering:
                    # ClaimLost can also mean a still-owned manifest became
                    # invalid before runtime registration. Recheck canonical
                    # ownership: abandon is a no-op for replaced/cancelled
                    # claims and preserves dispatched effects as unknown.
                    # Undispatched retries get a NEW attempt/manifest, never
                    # reuse this attempt's invalid attached context.
                    disposition = "fenced_abandon_failed"
                    await self.repository.abandon(claim, reason="worker_claim_lost")
                    disposition = "fenced_abandon_returned"
            finally:
                # Returned is not proof of settlement: the fence may reject it.
                _claim_lost_diagnostic(claim, error, stage=stage, disposition=disposition)
        except BudgetExceeded as error:
            await self.repository.abandon(claim, reason=error.code, target=TaskStatus.PAUSED)
        except Exception as error:
            # Preserve unknown executor state, even if a local note committed.
            # Existing recovery must confirm process exit before its terminal
            # outcome is normalized through the same owner hook above.
            await self.repository.abandon(claim, reason="worker_" + type(error).__name__[:70], target=TaskStatus.FAILED)
        finally:
            pending = [task for task in (work, heartbeat) if task is not None and not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                _, unresolved = await asyncio.wait(pending, timeout=self.repository.limits.shutdown_seconds)
                for task in unresolved:
                    task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)

    @staticmethod
    async def _finish_owned_outcome(provider, context, outcome):
        finish = getattr(getattr(provider, "prepare", None), "finish_outcome", None)
        if callable(finish) and await finish(context, outcome):
            return True
        finish = getattr(provider, "finish_outcome", None)
        return bool(await finish(context, outcome)) if callable(finish) else False

    async def stop(self):
        if self._closing:
            return
        self._closing = True
        self._stop.set()
        active = tuple(self._active)
        for task in active:
            task.cancel()
        if active:
            await asyncio.wait(active, timeout=self.repository.limits.shutdown_seconds + 1)
