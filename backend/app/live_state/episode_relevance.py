"""Ephemeral activity relevance, never admission or physical completion.

The intentionally small grammar accepts ``when/while/during <activity label>``
and ``activity: <label>``. Labels are compared whole, not by object keywords,
substring matches, speculative synonyms, or model calls. Compound and unfamiliar
cues remain unsupported. Activity originals retain their confidence and time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import re

from app.knowledge.models import SourceError
from app.knowledge.observations import OBSERVATION_NAMESPACE
from app.live_state.canonical import context_dependency
from app.live_state.models import FrameRef, SessionScope, SourceDependency
from app.live_state.perception_adapter import observation_from_record


ACTIVITY_TTL_SECONDS = 10
MIN_ACTIVITY_CONFIDENCE = 0.7
_COMPOUND_OR_UNCERTAIN = frozenset((
    "and", "or", "if", "unless", "until", "before", "after", "then", "not",
    "never", "maybe", "possibly", "unknown", "unclear",
))


def activity_label(value):
    if not isinstance(value, str) or not 0 < len(value) <= 160:
        return None
    label = " ".join(value.casefold().split())
    if not re.fullmatch(r"[\w][\w '\-]*", label) or set(label.split()) & _COMPOUND_OR_UNCERTAIN:
        return None
    return label


def cue_label(cue):
    match = re.fullmatch(r"(?:when\s+|while\s+|during\s+|activity:\s*)(.+?)\.?", cue.strip(), re.IGNORECASE)
    return activity_label(match[1]) if match else None


@dataclass(frozen=True, slots=True)
class ActivityEvidence:
    scope: SessionScope
    source: SourceDependency
    label: str
    confidence: float
    started_at: datetime
    captured_at: datetime
    expires_at: datetime
    frames: tuple[FrameRef, ...]


@dataclass(frozen=True, slots=True)
class CueRelevance:
    relevant: bool
    reason: str
    activity: ActivityEvidence | None = None
    cue: str | None = None


def relevance_for_check(check, episode_started_at, activity, *, now):
    if (check.lifecycle != "tracking" or check.resolution is not None
            or (check.snoozed_until is not None and check.snoozed_until > now)
            or (check.observation_window_expires_at is not None and check.observation_window_expires_at <= now)):
        return CueRelevance(False, "check_inactive")
    if not check.activation_cues:
        return CueRelevance(True, "explicit_unconditional")
    labels = tuple((cue, cue_label(cue)) for cue in check.activation_cues)
    if not any(label is not None for _, label in labels):
        return CueRelevance(False, "cue_grammar_unsupported")
    if activity is None or activity.scope != check.scope:
        return CueRelevance(False, "activity_unavailable")
    if not episode_started_at <= activity.started_at <= activity.captured_at <= now < activity.expires_at:
        return CueRelevance(False, "activity_outside_episode_or_window")
    if activity.confidence < MIN_ACTIVITY_CONFIDENCE or activity_label(activity.label) is None:
        return CueRelevance(False, "activity_uncertain")
    for cue, label in labels:
        if label is not None and label == activity_label(activity.label):
            return CueRelevance(True, "inferred_activity_relevance_not_completion", activity, cue)
    return CueRelevance(False, "activity_not_matching")


def current_activity_entry(snapshot, *, now):
    # Only the existing canonical working-memory adapter owns this entry.
    for entry in snapshot.entries:
        if (entry.entry_id == "working_observation" and entry.kind == "observation"
                and entry.scope == snapshot.scope and entry.validity == "active"
                and entry.evidence_kind == "observed" and entry.guard_binding is not None
                and len(entry.sources) == 1 and entry.sources[0].source_namespace == "context_event"
                and entry.observed_at is not None and entry.observed_at <= now
                and all(t is not None and t > now for t in (entry.fresh_until, entry.expires_at))):
            return entry
    return None


async def activity_original(canonical, connection, source, *, now):
    """Read an exact, removable original under the caller's canonical transaction."""
    record = await canonical.context._record(connection, canonical.scope.owner_id, source.source_id)
    if record.source_namespace != OBSERVATION_NAMESPACE or record.kind != "observation":
        return None
    await canonical._lock_roots(connection, (record.payload_id,))
    await canonical._validate_dependencies(connection, (source,))
    record = await canonical.context._record(connection, canonical.scope.owner_id, source.source_id)
    if (context_dependency(record) != source
            or (record.owner_id, record.session_id) != (canonical.scope.owner_id, canonical.scope.session_id)):
        raise SourceError("live_episode_activity_changed", "Activity original changed scope or revision.")
    observation, frames = observation_from_record(record)
    if (not frames or observation.primary_visual_frame_id not in {frame.frame_id for frame in frames}
            or observation.activity is None or observation.activity_confidence is None
            or not observation.ended_at <= now < observation.ended_at + timedelta(seconds=ACTIVITY_TTL_SECONDS)):
        return None
    return ActivityEvidence(canonical.scope, source, observation.activity, observation.activity_confidence,
        observation.started_at, observation.ended_at,
        observation.ended_at + timedelta(seconds=ACTIVITY_TTL_SECONDS), frames)


async def read_current_activity(canonical, connection, snapshot, *, now):
    entry = current_activity_entry(snapshot, now=now)
    if entry is None:
        return None
    await canonical._lock_roots(connection, entry.guard_binding.payload_root_ids)
    activity = await activity_original(canonical, connection, entry.sources[0], now=now)
    if activity is None:
        return None
    if activity.captured_at != entry.observed_at or activity.frames != entry.frames:
        raise SourceError("live_episode_activity_mismatch", "Activity projection disagrees with original evidence.")
    return ActivityEvidence(activity.scope, activity.source, activity.label, activity.confidence,
        activity.started_at, activity.captured_at,
        min(activity.expires_at, entry.fresh_until, entry.expires_at), activity.frames)
