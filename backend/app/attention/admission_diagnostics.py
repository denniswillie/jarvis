"""Bounded process-local observations, never attention or delivery authority.

Only an existing voice poll opens an observation. Transaction outcomes are
published after claim_next returns (including transaction exit), not on reads,
maintenance ticks, or rolled-back attempts. No logging or I/O occurs here.
"""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import UTC, datetime
from functools import wraps
import re
from types import SimpleNamespace
from uuid import UUID


FORMAT = "jarvis.attention.admission_diagnostics.v1"
OUTCOMES = ("no_pending", "no_session", "route_not_ready", "foreground_busy",
            "sink_unavailable", "quiet_denied", "quiet_unavailable",
            "coalescing_blocked", "coalescing_completed", "claimed", "no_claim",
            "exception", "cancelled", "admission_unclassified")
MAX_EPISODES = 16
MAX_SAMPLES = 2
MAX_COUNT = 2**63 - 1
_CURRENT = ContextVar("attention_admission_diagnostic", default=None)
_PENDING = ContextVar("attention_admission_transaction_diagnostic", default=None)


def _identity(value):
    return str(UUID(str(value))) if value is not None else None


def _active():
    value = _CURRENT.get()
    return value if value is not None and value.active and value.task is asyncio.current_task() else None


class AdmissionDiagnostics:
    def __init__(self):
        self.owner = None
        self.sequence = 0
        self.counters = {key: {"count": 0, "first_at": None, "last_at": None} for key in OUTCOMES}
        self.episodes = {}
        self.dropped = {"samples": 0, "episodes": 0, "invalid": 0, "observations": 0}

    def drop(self, kind):
        self.dropped[kind] = min(MAX_COUNT, self.dropped[kind] + 1)

    def begin(self, owner):
        if (not isinstance(owner, str) or re.fullmatch(r"[A-Za-z0-9_.:@-]{1,200}", owner) is None
                or self.owner not in (None, owner)):
            self.drop("invalid")
            return None
        self.owner = owner
        if self.sequence == MAX_COUNT:
            self.drop("observations")
            return None
        self.sequence += 1
        return SimpleNamespace(collector=self, owner=owner, sequence=self.sequence,
                               task=asyncio.current_task(), active=True, outcomes=set())

    def add(self, observation, outcome, *, owner=None, session_id=None, episode_id=None, revision=None):
        if outcome not in self.counters or owner not in (None, observation.owner):
            self.drop("invalid")
            return
        session = _identity(session_id)
        episode = _identity(episode_id)
        if revision is not None and (type(revision) is not int or not 0 <= revision <= MAX_COUNT):
            self.drop("invalid")
            return
        # A single invocation cannot manufacture multiple completed checks.
        if outcome in observation.outcomes:
            return
        observation.outcomes.add(outcome)
        now = datetime.now(UTC).isoformat()
        counter = self.counters[outcome]
        counter["count"] = min(MAX_COUNT, counter["count"] + 1)
        counter["first_at"] = counter["first_at"] or now
        counter["last_at"] = now
        if outcome != "quiet_denied":
            return
        if episode is None or session is None or revision is None:
            self.drop("invalid")
            return
        key = (episode, revision)
        if key not in self.episodes:
            if len(self.episodes) == MAX_EPISODES:
                self.drop("episodes")
                self.drop("samples")
                return
            self.episodes[key] = {"owner_id": observation.owner, "episode_id": episode,
                                  "revision": revision, "completed_denials": 0, "samples": []}
        bucket = self.episodes[key]
        bucket["completed_denials"] = min(MAX_COUNT, bucket["completed_denials"] + 1)
        if len(bucket["samples"]) == MAX_SAMPLES:
            self.drop("samples")
            return
        bucket["samples"].append({"owner_id": observation.owner, "session_id": session,
            "episode_id": episode, "revision": revision, "sequence": observation.sequence,
            "completed_at": now, "transaction_exited_successfully": True})

    def snapshot(self):
        return deepcopy({"format": FORMAT, "status": "observed", "owner_id": self.owner,
            "observation_sequence": self.sequence, "counters": self.counters,
            "quiet_denials": list(self.episodes.values()), "dropped": self.dropped,
            "limits": {"episode_revisions": MAX_EPISODES, "samples_per_episode_revision": MAX_SAMPLES,
                       "counter_max": MAX_COUNT},
            "authority": "diagnostic_only_not_canonical_delivery_evidence",
            "meaning": "Completed admission observations, not maintenance ticks, tool reads, hearing, or delivery. "
                       "Coalescing completed includes an existing summary; it does not prove a new summary or claim."})


def record_admission(outcome, **identity):
    """Optional one-way hook. A broken observer must not break business work."""
    try:
        if (observation := _active()) is not None:
            observation.collector.add(observation, outcome, **identity)
    except Exception:
        pass


@contextmanager
def observe_admission(collector, owner):
    observation = None
    try:
        observation = collector.begin(owner)
    except Exception:
        pass
    token = _CURRENT.set(observation)
    try:
        yield
    except asyncio.CancelledError:
        record_admission("cancelled")
        raise
    except Exception:
        record_admission("exception")
        raise
    finally:
        if observation is not None:
            if not observation.outcomes:
                record_admission("admission_unclassified")
            observation.active = False
        _CURRENT.reset(token)


def stage_admission(outcome, **identity):
    try:
        pending = _PENDING.get()
        if _active() is not None and pending is not None and len(pending) < 4:
            pending.append((outcome, identity))
    except Exception:
        pass


def completed_claim(method):
    """Stage inside the existing transaction; publish only after its exit."""
    @wraps(method)
    async def wrapped(*args, **kwargs):
        if _active() is None:
            return await method(*args, **kwargs)
        pending = []
        token = _PENDING.set(pending)
        try:
            result = await method(*args, **kwargs)
        finally:
            _PENDING.reset(token)
        # Unreached on body/commit failure or cancellation. No automatic retry.
        for outcome, identity in pending:
            record_admission(outcome, **identity)
        if not pending:
            record_admission("no_claim")
        return result
    return wrapped


def shutdown_snapshot(service):
    try:
        result = service.admission_diagnostics.snapshot()
        tasks = (service._task, service._maintenance_task)
        result.update(service_closed=service._closed, voice_loop_started=tasks[0] is not None,
                      voice_loop_joined=tasks[0] is None or tasks[0].done(),
                      maintenance_loop_joined=tasks[1] is None or tasks[1].done(),
                      final=service._closed and all(task is None or task.done() for task in tasks))
        return result
    except Exception:
        return {"format": FORMAT, "status": "unavailable", "final": False,
                "reason": "admission_diagnostics_unavailable",
                "authority": "diagnostic_only_not_canonical_delivery_evidence"}
