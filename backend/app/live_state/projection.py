"""Bounded deterministic selection; combined context is rebound before publication."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from uuid import UUID, uuid4

from app.live_state.models import GuardedPayloadBinding, LiveSnapshot, SessionScope, SourceDependency, StateCursor
from app.voice_routes.live_context_content import encode_live_context_content
from app.voice_routes.contracts import ObservationWindow, RouteIdentity, VoiceContextUpdate


@dataclass(frozen=True, slots=True)
class VoiceSelection:
    scope: SessionScope
    cursor: StateCursor
    text: str
    dependencies: tuple[SourceDependency, ...]
    parent_payload_root_ids: tuple[UUID, ...]
    selected_entry_ids: tuple[str, ...]
    omitted_entry_ids: tuple[str, ...]
    expires_at: datetime | None
    # Filled only by the explicit Live historical-observation selection policy.
    # expires_at remains the preparation freshness bound for the selected text.
    observation_history: tuple[ObservationWindow, ...] = ()
    hard_expires_at: datetime | None = None


ContextBinder = Callable[[VoiceSelection], Awaitable[GuardedPayloadBinding]]
_PRIORITY = {"notification": 0, "observation": 1, "focus": 2, "task": 3,
             "situational_check": 4, "selected_memory": 5, "future_cue": 6}


def _compact_text(snapshot: LiveSnapshot, lines: tuple[str, ...] | list[str], omitted: int, *,
                  has_fresh_observation: bool = False) -> str:
    # These gaps must survive even when no evidence statement fits. In particular,
    # a missing task in this partial view is not evidence that no task exists.
    availability = (f"Readiness={snapshot.readiness};"
                    f"responsibilities={snapshot.capabilities.durable_responsibilities};"
                    f"history={snapshot.capabilities.historical_retrieval}.")
    missing = (() if has_fresh_observation else
               ("Current visual view unknown; prior observations are historical.",))
    return "\n".join((availability, *missing, *lines,
                      f"Partial; omitted={omitted}; not absence."))


def _live_encoded_context_bytes(text: str, cursor: StateCursor) -> int:
    # The adapter uses the same entire factual-content envelope. Cursor remains
    # in the immutable local selection/update/guards, not in provider content.
    return len(encode_live_context_content(text).encode("utf-8"))


def _compact_timestamp(value: datetime | None) -> str:
    if value is None:
        return "unknown"
    # UTC normalization and removal of fractional trailing zeroes are exact,
    # not rounding/truncation of capture time or an extension of freshness.
    value = value.astimezone(timezone.utc)
    stamp = value.strftime("%Y-%m-%dT%H:%M:%S")
    fraction = f"{value.microsecond:06d}".rstrip("0")
    return stamp + ("." + fraction if fraction else "") + "Z"


def _compact_entry_line(entry, text: str, now: datetime) -> tuple[str, bool]:
    # Facts, including unknowns, stay one complete quoted data value. Embedded
    # newlines/quotes cannot become renderer-owned metadata or role delimiters.
    quoted = json.dumps(text, ensure_ascii=False)
    if entry.kind == "observation":
        fresh = (entry.observed_at is not None and entry.fresh_until is not None
                 and entry.observed_at <= now < entry.fresh_until)
        state = "current" if fresh else "unknown"
        confidence = ("" if entry.confidence is None else f";confidence={entry.confidence}")
        return (f"Untrusted visual/{entry.evidence_kind} {state} "
                f"captured@{_compact_timestamp(entry.observed_at)};"
                f"fresh_until={_compact_timestamp(entry.fresh_until)}{confidence}: {quoted}", fresh)
    qualifier = getattr(entry, "lifecycle", None) or getattr(entry, "status", None) or entry.evidence_kind
    observed = entry.observed_at.isoformat() if entry.observed_at is not None else "unknown"
    return f"Untrusted [{entry.kind}; {qualifier}; observed_at={observed}] {quoted}", False


def select_voice_context(snapshot: LiveSnapshot, *, now: datetime,
                         max_chars: int = 4096, max_entries: int = 8,
                         max_encoded_bytes: int | None = None,
                         preserve_observation_history: bool = False) -> VoiceSelection:
    """Select whole entries before canonical binding, with no I/O.

    ``None`` preserves the existing Realtime projection. An encoded-byte budget
    opts into a compact Live projection: it covers the entire ASCII-escaped
    factual-context JSON document, not merely UTF-8 text or a token estimate.
    Availability and partial-view notices are mandatory; bounds too small for
    those notices raise rather than silently suppressing uncertainty.
    """
    if now.tzinfo is None or now.utcoffset() is None or not 128 <= max_chars <= 4096 or not 1 <= max_entries <= 16:
        raise ValueError("Invalid projection clock or bounds.")
    if max_encoded_bytes is not None and (type(max_encoded_bytes) is not int or max_encoded_bytes <= 0):
        raise ValueError("Encoded context budget must be a positive integer.")
    if type(preserve_observation_history) is not bool or (preserve_observation_history and max_encoded_bytes is None):
        raise ValueError("Observation history requires an explicit compact Live projection.")

    def compact_fits(text: str) -> bool:
        return len(text) <= max_chars and _live_encoded_context_bytes(text, snapshot.cursor) <= max_encoded_bytes

    if max_encoded_bytes is not None:
        mandatory = _compact_text(snapshot, (), len(snapshot.entries))
        if not compact_fits(mandatory):
            raise ValueError("Projection bounds cannot fit mandatory availability and omission notices.")
    if snapshot.readiness != "ready":
        if max_encoded_bytes is not None:
            omitted = tuple(entry.entry_id for entry in sorted(snapshot.entries,
                key=lambda entry: (_PRIORITY[entry.kind], entry.entry_id)))
            return VoiceSelection(snapshot.scope, snapshot.cursor, mandatory, (), (), (), omitted, None)
        return VoiceSelection(snapshot.scope, snapshot.cursor,
            "Context availability: " + snapshot.readiness + ".", (), (), (), (), None)
    lines, selected, omitted, deadlines = [], [], [], []
    observation_history, hard_deadlines = [], []
    has_fresh_observation = False
    dependencies = {}
    roots = set()
    used = 0
    ordered = sorted(snapshot.entries, key=lambda entry: (_PRIORITY[entry.kind], entry.entry_id))
    for entry in ordered:
        expiry = tuple(value for value in (entry.fresh_until, entry.expires_at) if value is not None)
        if entry.validity != "active" or any(value <= now for value in expiry) or entry.guard_binding is None:
            omitted.append(entry.entry_id)
            continue
        text = next((getattr(entry, field) for field in ("summary", "goal", "question", "relevance_cue")
                     if hasattr(entry, field)), "")
        if max_encoded_bytes is not None and (not text.strip()
                or (entry.observed_at is not None and entry.observed_at > now)):
            omitted.append(entry.entry_id)
            continue
        unknowns = getattr(entry, "unknowns", ())
        if unknowns:
            # Budget the entire evidence statement, including its limitations.
            # Omitting an oversized entry is safer than retaining only certainty.
            text += " Uncertainties: " + "; ".join(unknowns)
        qualifier = getattr(entry, "lifecycle", None) or getattr(entry, "status", None) or entry.evidence_kind
        observed = entry.observed_at.isoformat() if entry.observed_at is not None else "unknown"
        line = f"[{entry.kind}; {qualifier}; observed_at={observed}] {text}"
        fresh_observation = False
        if max_encoded_bytes is not None:
            line, fresh_observation = _compact_entry_line(entry, text, now)
        proposed = dict(dependencies)
        conflict = False
        for source in entry.sources:
            key = (source.source_namespace, source.source_id)
            if key in proposed and proposed[key] != source:
                conflict = True
                break
            proposed[key] = source
        proposed_roots = roots | set(entry.guard_binding.payload_root_ids)
        if (conflict or len(proposed) > 64 or len(proposed_roots) > 64
                or len(selected) >= max_entries or used + len(line) + bool(lines) > max_chars
                or (max_encoded_bytes is not None and not compact_fits(_compact_text(
                    snapshot, (*lines, line), len(snapshot.entries) - len(selected) - 1,
                    has_fresh_observation=has_fresh_observation or fresh_observation)))):
            omitted.append(entry.entry_id)
            continue
        lines.append(line)
        used += len(line) + (1 if len(lines) > 1 else 0)
        selected.append(entry.entry_id)
        dependencies, roots = proposed, proposed_roots
        deadlines.extend(expiry)
        has_fresh_observation = has_fresh_observation or fresh_observation
        if (preserve_observation_history and entry.kind == "observation"
                and entry.evidence_kind == "observed" and entry.sources
                and entry.observed_at is not None and entry.fresh_until is not None
                and entry.observed_at <= entry.fresh_until and entry.expires_at == entry.fresh_until):
            observation_history.append(ObservationWindow(observed_at=entry.observed_at,
                fresh_until=entry.fresh_until, dependencies=entry.sources))
        else:
            hard_deadlines.extend(expiry)
    return VoiceSelection(snapshot.scope, snapshot.cursor,
        (_compact_text(snapshot, lines, len(omitted), has_fresh_observation=has_fresh_observation)
         if max_encoded_bytes is not None else
         "\n".join(lines) if lines else "Context availability: no current source-backed observations."),
        tuple(sorted(dependencies.values(), key=lambda source: (source.source_namespace, str(source.source_id)))),
        tuple(sorted(roots, key=str)), tuple(selected), tuple(omitted), min(deadlines) if deadlines else None,
        observation_history=tuple(observation_history),
        hard_expires_at=min(hard_deadlines) if observation_history and hard_deadlines else None)


async def prepare_voice_context(selection: VoiceSelection, route: RouteIdentity, *,
                                bind: ContextBinder | None,
                                clock: Callable[[], datetime]) -> VoiceContextUpdate | None:
    """Background preparation only; the caller retains current-source/output guards.

    `bind` must bind the ACTUAL rendered text against these exact dependencies and
    parent roots through the canonical provenance path. Root union is not binding.
    """
    if selection.scope != route.scope:
        raise ValueError("Projection scope differs from voice route scope.")
    now = clock()
    if selection.expires_at is not None and selection.expires_at <= now:
        return None
    binding = None
    if selection.dependencies or selection.parent_payload_root_ids:
        if bind is None:
            raise ValueError("Source-backed context requires the canonical binder.")
        binding = await bind(selection)
        if binding.scope != selection.scope or set(binding.dependencies) != set(selection.dependencies):
            raise ValueError("Canonical binding does not cover this projection.")
    now = clock()
    if selection.expires_at is not None and selection.expires_at <= now:
        return None
    return VoiceContextUpdate(route=route, update_id=uuid4(), cursor=selection.cursor,
                              mode="snapshot", text=selection.text, dependencies=selection.dependencies,
                              guard_binding=binding, created_at=now,
                              expires_at=(selection.hard_expires_at if selection.observation_history
                                          else selection.expires_at),
                              observation_history=selection.observation_history)
