"""Opt-in observation of a provider guard's cancelled nested await.

No authority, SQL, deadlines or tasks live here. Outside an explicitly observed
writer, and in child tasks that merely inherit context, stage markers are inert.
"""
from asyncio import CancelledError, current_task
from contextlib import contextmanager
from contextvars import ContextVar
import time


MAX_GUARD_CANCELLATION_RECORDS = 32
_STAGES = frozenset({
    "pool_acquire", "transaction_enter", "memory_owner_lock", "task_owner_lock",
    "scope_revalidation", "canonical_source_lock", "dependency_revalidation",
    "response_source_lock", "unclassified_guard",
})
_ACTIVE = ContextVar("provider_guard_wait_observation", default=None)


def _clock():
    try:
        return time.monotonic()
    except Exception:
        return None


def _active():
    try:
        probe = _ACTIVE.get()
        return probe if probe is not None and probe["task"] is current_task() else None
    except Exception:
        return None


@contextmanager
def guard_wait_stage(stage):
    probe = _active()
    if probe is None:
        yield
        return
    stage = stage if isinstance(stage, str) and stage in _STAGES else "unclassified_guard"
    started = _clock()
    try:
        yield
    except CancelledError:
        # The innermost annotated wait wins; unwind must not overwrite it.
        if probe["cancelled"] is None:
            probe["cancelled"] = (stage, started, _clock())
        raise
    else:
        # Bounded by the fixed stage vocabulary, not by the number of queries.
        # Nested stages are inclusive; these totals must not be summed as a
        # disjoint breakdown. Emit them only if this guard is later cancelled.
        ended = _clock()
        if started is not None and ended is not None:
            totals = probe["completed_ms"]
            totals[stage] = totals.get(stage, 0.0) + max(0.0, (ended - started) * 1000)


class guard_context:
    """Wrap only the original __aenter__; delegate exit/suppression unchanged."""
    def __init__(self, context, stage):
        self.context, self.stage = context, stage

    async def __aenter__(self):
        with guard_wait_stage(self.stage):
            return await self.context.__aenter__()

    async def __aexit__(self, *error):
        return await self.context.__aexit__(*error)


class GuardCancellationDiagnostics:
    def __init__(self):
        self.records = 0

    def _record(self, trace, dispatch_id, sample, completed_ms):
        try:
            if self.records > MAX_GUARD_CANCELLATION_RECORDS:
                return
            self.records += 1
            if self.records > MAX_GUARD_CANCELLATION_RECORDS:
                trace.event("provider_guard_wait_truncated", status="truncated")
                return
            stage, started, ended = sample
            duration = None if started is None or ended is None else max(0.0, (ended - started) * 1000)
            trace.event("provider_guard_wait_cancelled", dispatch_id=dispatch_id,
                phase=stage, duration_ms=duration, status="await_cancelled",
                write_possible=False, send_returned=False, guard_present=True,
                boundary="completed_stages_inclusive",
                **{f"guard_completed_{name}_ms": value for name, value in completed_ms.items()})
        except Exception:
            # Logging failures cannot replace the original cancellation/error.
            pass

    @contextmanager
    def observe(self, trace, dispatch_id):
        if trace is None:
            yield
            return
        probe = {"task": current_task(), "cancelled": None, "completed_ms": {}}
        token = _ACTIVE.set(probe)
        started = _clock()
        try:
            yield
        except BaseException as error:
            sample = probe["cancelled"]
            if sample is None and isinstance(error, CancelledError):
                sample = ("unclassified_guard", started, _clock())
            if sample is not None:
                # A nested asyncio.timeout may translate cancellation to its
                # TimeoutError. This still describes the cancelled await only.
                self._record(trace, dispatch_id, sample, probe["completed_ms"])
            raise
        finally:
            _ACTIVE.reset(token)
