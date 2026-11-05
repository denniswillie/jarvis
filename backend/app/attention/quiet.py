"""Owner-wide DND in the existing source-backed capability preference ledger.

This capability is separate from responsibility_notifications. There is no task,
perception, memory, profile, or delivery-queue mutation here. Expiry is evaluated
against database time on every read; reconnect never creates a new expiry.

Integration boundary: attention claims/guarded writes, push claim/preparation,
direct visual writers and Live readonly-result callbacks consume this policy.
Native already-buffered output, client UI attention and resume coalescing still
need their existing-owner integrations. A prepared push may already be externally exposed;
this module cannot recall it or prove hearing. No emergency exemption is defined.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Literal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.tools.inputs import ToolInputError
from app.tools.invocation import ToolInvocationContext


CAPABILITY = "owner_attention_quiet"
FORMAT = "jarvis.attention.owner_quiet.v1"
# Same owner lock as TaskRepository.transaction(owner), not a second coordinator.
OWNER_LOCK = "select pg_advisory_xact_lock(hashtextextended(current_schema() || ':jarvis-task-owner:' || $1,0))"
# Readers retain exclusion from policy mutations without upgrading borrowed guards.
OWNER_SHARED_LOCK = "select pg_advisory_xact_lock_shared(hashtextextended(current_schema() || ':jarvis-task-owner:' || $1,0))"


class QuietIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["activate", "resume"]
    expires_at: AwareDatetime | None = None
    timezone: str | None = None

    @field_validator("expires_at", mode="before")
    @classmethod
    def exact_expiry(cls, value):
        if value is not None and not isinstance(value, (str, datetime)):
            raise ValueError("Use an offset-aware ISO datetime, not an epoch or duration.")
        return value

    @field_validator("timezone")
    @classmethod
    def known_timezone(cls, value):
        if value is not None:
            try:
                ZoneInfo(value)
            except (ZoneInfoNotFoundError, ValueError) as error:
                raise ValueError("Use an IANA timezone.") from error
        return value

    @model_validator(mode="after")
    def validate_scope(self):
        if self.action == "resume" and (self.expires_at is not None or self.timezone is not None):
            raise ValueError("Resume does not set an expiry or timezone.")
        if self.expires_at is not None and self.timezone is not None:
            if self.expires_at.utcoffset() != self.expires_at.astimezone(ZoneInfo(self.timezone)).utcoffset():
                raise ValueError("Expiry offset must match the named timezone at that instant.")
        return self


class QuietCommand(QuietIntent):
    expected_revision: int = Field(ge=0, strict=True)


class QuietEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    episode_id: UUID | None
    state: Literal["quiet", "resumed"]
    started_at: AwareDatetime | None
    expires_at: AwareDatetime | None
    timezone: str | None
    resumed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def valid_episode(self):
        if (self.episode_id is None) != (self.started_at is None):
            raise ValueError("Episode identity and start must be present together.")
        if self.state == "quiet" and (self.episode_id is None or self.resumed_at is not None):
            raise ValueError("Quiet requires an unresumed episode.")
        if self.state == "resumed" and self.resumed_at is None:
            raise ValueError("Resume requires a committed timestamp.")
        if self.expires_at is not None and (self.started_at is None or self.expires_at <= self.started_at):
            raise ValueError("Expiry must follow the episode start.")
        return self


class QuietPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["jarvis.attention.owner_quiet.v1"] = FORMAT
    intent: QuietIntent
    episode: QuietEpisode

    @model_validator(mode="after")
    def matches_intent(self):
        expected = "quiet" if self.intent.action == "activate" else "resumed"
        if self.episode.state != expected:
            raise ValueError("Policy intent and episode disagree.")
        if expected == "quiet" and (
            self.intent.expires_at != self.episode.expires_at or self.intent.timezone != self.episode.timezone
        ):
            raise ValueError("The exact accepted scope must be preserved.")
        return self


@dataclass(frozen=True)
class QuietSnapshot:
    owner_id: str
    revision: int
    evaluated_at: datetime
    policy: QuietPolicy | None
    available: bool = True

    @property
    def effective_state(self):
        if not self.available:
            return "unavailable"
        if self.policy is None:
            return "inactive"
        episode = self.policy.episode
        if episode.state == "resumed":
            return "resumed"
        if episode.expires_at is not None and self.evaluated_at >= episode.expires_at:
            return "expired"
        return "quiet"

    @property
    def suppresses_unsolicited(self):
        return self.effective_state in {"quiet", "unavailable"}

    def payload(self):
        return {
            "owner_id": self.owner_id, "scope": "owner_unsolicited_attention",
            "revision": self.revision, "effective_state": self.effective_state,
            "suppresses_unsolicited": self.suppresses_unsolicited,
            "evaluated_at": self.evaluated_at.isoformat(),
            "episode": self.policy.episode.model_dump(mode="json") if self.policy else None,
            "channels": ["voice", "push", "ui"],
            "direct_replies_allowed": True, "work_continues": True,
            "automatic_replay_requested": False,
        }


def _decode_policy(content):
    return QuietPolicy.model_validate(json.loads(content) if isinstance(content, str) else content)


async def read_owner_quiet(connection, owner: str) -> QuietSnapshot:
    """Read the latest revision only. Withdrawn policy never revives an older one."""
    row = await connection.fetchrow("""/* attention_quiet:current */
        select clock_timestamp() as evaluated_at,q.revision,p.content,p.availability
        from (select 1) anchor
        left join lateral (
            select revision,policy_payload_id from jarvis_capability_preferences
            where owner_id=$1 and capability=$2 and knowledge_area_id is null
            order by revision desc limit 1
        ) q on true
        left join jarvis_context_payloads p on p.owner_id=$1 and p.id=q.policy_payload_id
        """, owner, CAPABILITY)
    if row is None or not isinstance(row["evaluated_at"], datetime) or row["evaluated_at"].utcoffset() is None:
        raise ToolInputError("attention_quiet_unavailable", "Canonical quiet state could not be read.")
    revision = row["revision"]
    if revision is None:
        return QuietSnapshot(owner, 0, row["evaluated_at"], None)
    policy = None
    if row["availability"] == "available" and row["content"] is not None:
        try:
            policy = _decode_policy(row["content"])
        except (ValidationError, ValueError, TypeError):
            pass
    return QuietSnapshot(owner, revision, row["evaluated_at"], policy, available=policy is not None)


@dataclass(frozen=True)
class QuietDecision:
    allowed: bool
    reason: str
    channel: str
    revision: int | None = None
    episode_id: UUID | None = None


async def quiet_delivery_decision(connection, owner, *, kind: str, channel: str) -> QuietDecision:
    """Trusted callers classify initiation; model JSON cannot declare solicitation.

    Call within the existing owner transaction for an admission/write fence. This
    decision neither grants source authorization nor overrides duty/device quiet.
    Solicited replies do not wait for, mutate, or clear owner quiet state.
    """
    if channel not in {"voice", "push", "ui"} or kind not in {"solicited_reply", "unsolicited"}:
        raise ValueError("An explicit supported delivery kind and channel are required.")
    if kind == "solicited_reply":
        if channel != "voice":
            raise ValueError("A direct voice reply cannot exempt push or UI attention.")
        return QuietDecision(True, "solicited_reply", channel)
    snapshot = await read_owner_quiet(connection, owner)
    episode = snapshot.policy.episode if snapshot.policy else None
    return QuietDecision(
        not snapshot.suppresses_unsolicited,
        "owner_quiet_" + snapshot.effective_state,
        channel, snapshot.revision, episode.episode_id if episode else None,
    )


class OwnerQuietRepository:
    """Append-only owner preference revisions, with captured-conversation admission."""

    def __init__(self, pool):
        self.pool = pool

    async def get(self, owner):
        async with self.pool.acquire() as connection:
            return await read_owner_quiet(connection, owner)

    async def save(self, owner, session_id, source: ToolInvocationContext, request: QuietCommand):
        if (session_id is None or source.task is not None or source.request_event_id is None
                or source.request_revision is None):
            raise ToolInputError("request_evidence_required", "A current captured user conversation request is required.")
        intent = QuietIntent.model_validate(request.model_dump(exclude={"expected_revision"}))
        async with self.pool.acquire() as c, c.transaction():
            await c.execute(OWNER_LOCK, owner)
            original = await c.fetchval("""/* attention_quiet:authorized_source */
                select e.payload_id from jarvis_context_events e
                join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
                join jarvis_sessions s on s.owner_id=e.owner_id and s.id=e.session_id
                where e.owner_id=$1 and e.id=$2 and e.revision=$3 and e.session_id=$4
                  and s.status='active' and e.kind in ('user_request','user_transcript')
                  and e.trust='user_input' and e.finality in ('final','revised')
                  and p.availability='available' and p.content is not null
                  and not exists(select 1 from jarvis_context_events newer
                      where newer.owner_id=e.owner_id and newer.supersedes_event_id=e.id)
                for share of p
                """, owner, source.request_event_id, source.request_revision, session_id)
            if original is None:
                raise ToolInputError("request_evidence_unavailable", "The current owned user request is unavailable.")
            previous = await c.fetchrow("""/* attention_quiet:request */
                select q.revision,p.content,p.availability from jarvis_capability_preferences q
                left join jarvis_context_payloads p on p.owner_id=q.owner_id and p.id=q.policy_payload_id
                where q.owner_id=$1 and q.capability=$2 and q.knowledge_area_id is null
                  and q.request_event_id=$3 and q.request_revision=$4
                order by q.revision desc limit 1
                """, owner, CAPABILITY, source.request_event_id, source.request_revision)
            current = await read_owner_quiet(c, owner)
            if previous is not None:
                try:
                    prior = _decode_policy(previous["content"]) if previous["availability"] == "available" else None
                except (ValidationError, ValueError, TypeError):
                    prior = None
                if prior is None:
                    raise ToolInputError("quiet_receipt_unavailable", "The original quiet command receipt is unavailable; do not replay it.")
                if prior.intent != intent:
                    raise ToolInputError("request_conflict", "This user request already changed quiet state differently.")
                return {"quiet": current.payload(), "accepted_revision": previous["revision"], "request_replayed": True}
            if current.revision != request.expected_revision:
                raise ToolInputError("revision_conflict", "Read current owner quiet state before changing it.")
            now = current.evaluated_at
            old = current.policy.episode if current.policy else None
            if request.action == "activate":
                if request.expires_at is not None and request.expires_at <= now:
                    raise ToolInputError("quiet_expiry_not_future", "The exact requested expiry must be in the future.")
                continuing = current.effective_state == "quiet"
                episode = QuietEpisode(
                    episode_id=old.episode_id if continuing else uuid4(), state="quiet",
                    started_at=old.started_at if continuing else now,
                    expires_at=request.expires_at, timezone=request.timezone,
                )
            else:
                # A fresh explicit resume can recover withdrawn/unavailable policy.
                # It does not reconstruct a removed episode or replay any output.
                episode = QuietEpisode(
                    episode_id=old.episode_id if old else None, state="resumed",
                    started_at=old.started_at if old else None,
                    expires_at=old.expires_at if old else None,
                    timezone=old.timezone if old else None,
                    # Repeated explicit resume is not a later backlog boundary.
                    resumed_at=old.resumed_at if old and old.state == "resumed" else now,
                )
            policy = QuietPolicy(intent=intent, episode=episode)
            payload = await c.fetchval("""/* attention_quiet:payload */
                insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available',$2::jsonb) returning id
                """, owner, policy.model_dump_json())
            await c.execute("""/* attention_quiet:dependency */
                insert into jarvis_context_payload_dependencies(owner_id,source_payload_id,dependent_payload_id)
                values($1,$2,$3)
                """, owner, original, payload)
            await c.execute("""/* attention_quiet:append */
                insert into jarvis_capability_preferences
                    (owner_id,capability,revision,request_event_id,request_revision,policy_payload_id)
                values($1,$2,$3,$4,$5,$6)
                """, owner, CAPABILITY, current.revision + 1,
                source.request_event_id, source.request_revision, payload)
            snapshot = QuietSnapshot(owner, current.revision + 1, now, policy)
            return {"quiet": snapshot.payload(), "accepted_revision": snapshot.revision, "request_replayed": False}
