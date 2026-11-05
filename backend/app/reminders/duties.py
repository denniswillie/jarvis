"""Conversation-taught rules over the existing reminder/attention delivery path."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from typing import Literal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.tools.inputs import ToolInputError


class DutySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["scheduled"] = "scheduled"
    title: str = Field(min_length=1, max_length=200)
    details: str = Field(default="", max_length=2000)
    condition: str | None = Field(default=None, min_length=1, max_length=1000)
    first_at: AwareDatetime
    timezone: str = "UTC"
    every_days: int | None = Field(default=None, ge=1, le=3660, strict=True)
    basis: Literal["explicit_request", "confirmed_inference", "inferred_under_policy"] = "explicit_request"

    @field_validator("timezone")
    @classmethod
    def known_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError("Use an IANA timezone.") from error
        return value

    def occurrence(self, due: datetime, now: datetime):
        """Preserve local wall-clock cadence; collapse missed periods, not alerts."""
        local = due.astimezone(ZoneInfo(self.timezone))
        if self.every_days is None:
            return due, None
        elapsed = max(0, (now.astimezone(local.tzinfo).date() - local.date()).days // self.every_days)
        selected = local + timedelta(days=elapsed * self.every_days)
        if elapsed and selected > now:
            selected -= timedelta(days=self.every_days)
        return selected.astimezone(UTC), (selected + timedelta(days=self.every_days)).astimezone(UTC)


class SaveDuty(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: DutySpec
    responsibility_id: UUID | None = None
    expected_revision: int | None = Field(default=None, ge=1, strict=True)
    request_slot: int = Field(default=1, ge=1, le=20, strict=True)

    @model_validator(mode="after")
    def revision_pair(self):
        if (self.responsibility_id is None) != (self.expected_revision is None):
            raise ValueError("Updating needs both responsibility_id and expected_revision.")
        return self


class ChangeDuty(BaseModel):
    model_config = ConfigDict(extra="forbid")
    responsibility_id: UUID
    expected_revision: int = Field(ge=1, strict=True)
    operation: Literal["cancel", "snooze", "report_done", "report_not_done"]
    occurrence_id: UUID | None = None
    until: AwareDatetime | None = None
    request_slot: int = Field(default=1, ge=1, le=20, strict=True)

    @model_validator(mode="after")
    def targeted_occurrence(self):
        if self.operation != "cancel" and self.occurrence_id is None:
            raise ValueError("Select an occurrence from list_responsibilities.")
        if self.operation == "snooze" and self.until is None:
            raise ValueError("Snoozing requires an explicit until timestamp.")
        if self.operation != "snooze" and self.until is not None:
            raise ValueError("until applies only to snooze.")
        return self


def decoded(value):
    return json.loads(value) if isinstance(value, str) else value


def json_value(value):
    return json.loads(json.dumps(value, default=lambda item:
        item.isoformat() if isinstance(item, datetime) else str(item)))


class ResponsibilityRepository:
    def __init__(self, pool):
        self.pool = pool

    def episode_checks(self, *, enabled=False):
        """Same canonical owner/transaction; initialization never starts services."""
        from app.reminders.episode_checks import EpisodeCheckOwner
        return EpisodeCheckOwner(self, enabled=enabled)

    async def _lock(self, connection, owner):
        await connection.execute(
            "select pg_advisory_xact_lock(hashtextextended('jarvis-responsibilities:' || $1,0))", owner)

    async def _require_source(self, c, owner, session_id, source):
        if source.request_event_id is None or source.request_revision is None:
            raise ToolInputError("request_evidence_required", "A captured user request is required.")
        valid = await c.fetchval("""select e.payload_id from jarvis_context_events e
            join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
            where e.owner_id=$1 and e.id=$2 and e.revision=$3 and e.session_id=$4
              and e.kind in ('user_request','user_transcript') and e.trust='user_input'
              and e.finality in ('final','revised') and p.availability='available' and p.content is not null
              and not exists(select 1 from jarvis_context_events newer
                  where newer.owner_id=e.owner_id and newer.supersedes_event_id=e.id)""",
            owner, source.request_event_id, source.request_revision, session_id)
        if not valid:
            raise ToolInputError("request_evidence_unavailable", "The current owned user request is unavailable.")
        return valid

    async def _request(self, c, owner, session_id, source, operation, request):
        await self._require_source(c, owner, session_id, source)
        arguments = request.model_dump(mode="json")
        # Preserve pre-kind scheduled request fingerprints across this migration.
        if isinstance(request, SaveDuty):
            arguments["spec"].pop("kind", None)
        fingerprint = sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()
        old = await c.fetchrow("""select * from jarvis_responsibility_requests
            where owner_id=$1 and request_event_id=$2 and request_revision=$3 and operation=$4 and request_slot=$5""",
            owner, source.request_event_id, source.request_revision, operation, request.request_slot)
        if old and old["arguments_hash"] != fingerprint:
            raise ToolInputError("request_conflict", "This request slot already has another effect; inspect current duties.")
        return old, fingerprint

    async def _receipt(self, c, owner, source, operation, request, fingerprint, identity,
                       *, occurrence_id=None, report_state=None):
        report = report_state or {"report_done": "user_reported_done", "report_not_done": "user_reported_not_done"}.get(
            getattr(request, "operation", None))
        await c.execute("""insert into jarvis_responsibility_requests
            (owner_id,request_event_id,request_revision,operation,request_slot,arguments_hash,
             responsibility_id,occurrence_id,report_state) values($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
            owner, source.request_event_id, source.request_revision, operation, request.request_slot,
            fingerprint, identity, occurrence_id or getattr(request, "occurrence_id", None), report)

    @staticmethod
    def _require_scheduled(rule):
        if decoded(rule["spec"]).get("kind", "scheduled") != "scheduled":
            raise ToolInputError("episode_versions_required", "Use the episode-check owner with check and episode versions.")

    async def _rule(self, c, owner, identity, expected=None):
        row = await c.fetchrow("select * from jarvis_responsibilities where owner_id=$1 and id=$2 for update",
                               owner, identity)
        if row is None:
            raise ToolInputError("responsibility_not_found", "No responsibility exists in this owner scope.")
        if expected is not None and row["revision"] != expected:
            raise ToolInputError("revision_conflict", "The duty changed. List it before applying another change.")
        return row

    async def _revoke_reminders(self, c, owner, identities):
        identities = [identity for identity in identities if identity is not None]
        if identities:
            await c.execute("""update jarvis_reminders set status='completed',
                completed_at=coalesce(completed_at,clock_timestamp()),trigger_claim_token=null,
                trigger_claimed_at=null,updated_at=clock_timestamp(),
                metadata=metadata || jsonb_build_object('attention_cancelled_at',clock_timestamp())
                where owner_id=$1 and id=any($2::uuid[])""", owner, list(identities))

    async def _retire(self, c, owner, identity, status):
        rows = await c.fetch("""update jarvis_responsibility_occurrences set status=$3
            where owner_id=$1 and responsibility_id=$2 and status='open' returning reminder_id""",
            owner, identity, status)
        await self._revoke_reminders(c, owner, [row["reminder_id"] for row in rows])

    async def _reminder(self, c, owner, rule, occurrence_id, scheduled, *, not_before=None):
        spec = DutySpec.model_validate(decoded(rule["spec"]))
        return await c.fetchval("""insert into jarvis_reminders
            (owner_id,session_id,title,details,remind_at,condition,not_before,metadata)
            values($1,$2,$3,$4,$5,$6,$7,$8::jsonb) returning id""",
            owner, rule["session_id"], spec.title, spec.details,
            None if spec.condition else max(scheduled, not_before or scheduled),
            spec.condition, not_before or scheduled, json.dumps({
                "responsibility_id": str(rule["id"]), "occurrence_id": str(occurrence_id),
                "physical_action_status": "not_established"}))

    async def _materialize(self, c, owner, rule, now):
        if decoded(rule["spec"]).get("kind", "scheduled") != "scheduled" or rule["next_due_at"] is None:
            return
        spec = DutySpec.model_validate(decoded(rule["spec"]))
        scheduled, following = spec.occurrence(rule["next_due_at"], now)
        await self._retire(c, owner, rule["id"], "superseded")
        occurrence = uuid4()
        reminder = await self._reminder(c, owner, rule, occurrence, scheduled,
                                        not_before=rule["snoozed_until"])
        await c.execute("""insert into jarvis_responsibility_occurrences
            (id,owner_id,responsibility_id,rule_revision,scheduled_for,reminder_id)
            values($1,$2,$3,$4,$5,$6)""", occurrence, owner, rule["id"],
            rule["revision"], scheduled, reminder)
        await c.execute("""update jarvis_responsibilities set next_due_at=$3
            where owner_id=$1 and id=$2""", owner, rule["id"], following)

    async def materialize(self, owner, *, now=None):
        now = now or datetime.now(UTC)
        async with self.pool.acquire() as c, c.transaction():
            await self._lock(c, owner)
            rules = await c.fetch("""select * from jarvis_responsibilities where owner_id=$1
                and status='active' and next_due_at<=$2
                and coalesce(spec->>'kind','scheduled')='scheduled'
                and (snoozed_until is null or snoozed_until<=$2)
                order by next_due_at,id limit 100 for update""", owner, now)
            for rule in rules:
                await self._materialize(c, owner, rule, now)
            return len(rules)

    async def save(self, owner, session_id, source, request: SaveDuty):
        async with self.pool.acquire() as c, c.transaction():
            await self._lock(c, owner)
            old, fingerprint = await self._request(c, owner, session_id, source, "save", request)
            if old:
                return await self._view(c, owner, old["responsibility_id"], replayed=True)
            if request.spec.basis == "inferred_under_policy":
                from app.reminders.duty_preferences import read_preferences
                preferences = await read_preferences(c, owner)
                policy = preferences["policy"]
                if policy is None or policy["confirm_inferred_obligations"]:
                    raise ToolInputError("confirmation_required", "Confirm an inferred obligation before saving it.")
            body = request.spec.model_dump_json()
            if request.responsibility_id:
                current = await self._rule(c, owner, request.responsibility_id, request.expected_revision)
                self._require_scheduled(current)
                await self._retire(c, owner, request.responsibility_id, "superseded")
                rule = await c.fetchrow("""update jarvis_responsibilities set spec=$3::jsonb,
                    revision=revision+1,status='active',next_due_at=$4,snoozed_until=null,
                    session_id=$5,updated_at=clock_timestamp() where owner_id=$1 and id=$2 returning *""",
                    owner, request.responsibility_id, body, request.spec.first_at, session_id)
            else:
                rule = await c.fetchrow("""insert into jarvis_responsibilities
                    (owner_id,session_id,spec,next_due_at) values($1,$2,$3::jsonb,$4) returning *""",
                    owner, session_id, body, request.spec.first_at)
            await self._materialize(c, owner, rule, datetime.now(UTC))
            await self._receipt(c, owner, source, "save", request, fingerprint, rule["id"])
            return await self._view(c, owner, rule["id"])

    async def change(self, owner, session_id, source, request: ChangeDuty):
        operation = "change:" + request.operation
        async with self.pool.acquire() as c, c.transaction():
            await self._lock(c, owner)
            old, fingerprint = await self._request(c, owner, session_id, source, operation, request)
            if old:
                return await self._view(c, owner, old["responsibility_id"], replayed=True)
            rule = await self._rule(c, owner, request.responsibility_id, request.expected_revision)
            self._require_scheduled(rule)
            if request.operation == "cancel":
                await self._retire(c, owner, rule["id"], "cancelled")
                await c.execute("""update jarvis_responsibilities set status='cancelled',next_due_at=null,
                    revision=revision+1,updated_at=clock_timestamp() where owner_id=$1 and id=$2""", owner, rule["id"])
            else:
                occurrence = await c.fetchrow("""select * from jarvis_responsibility_occurrences
                    where owner_id=$1 and responsibility_id=$2 and id=$3 for update""",
                    owner, rule["id"], request.occurrence_id)
                if occurrence is None:
                    raise ToolInputError("occurrence_not_found", "Select an occurrence belonging to this duty.")
                if request.operation == "snooze":
                    if rule["status"] != "active" or occurrence["status"] != "open":
                        raise ToolInputError("occurrence_inactive", "Only an open occurrence of an active duty can be snoozed.")
                    if request.until <= datetime.now(UTC):
                        raise ToolInputError("invalid_snooze", "Snooze until a future, timezone-aware timestamp.")
                    await self._revoke_reminders(c, owner, [occurrence["reminder_id"]])
                    reminder = await self._reminder(c, owner, rule, occurrence["id"],
                                                    occurrence["scheduled_for"], not_before=request.until)
                    await c.execute("""update jarvis_responsibility_occurrences set reminder_id=$3
                        where owner_id=$1 and id=$2""", owner, occurrence["id"], reminder)
                    await c.execute("""update jarvis_responsibilities set snoozed_until=$3,
                        revision=revision+1,updated_at=clock_timestamp() where owner_id=$1 and id=$2""",
                        owner, rule["id"], request.until)
                else:
                    # Correct historical evidence without reviving old work.
                    latest = await c.fetchval("""select id from jarvis_responsibility_occurrences
                        where owner_id=$1 and responsibility_id=$2
                        order by created_at desc,id desc limit 1""", owner, rule["id"])
                    current = (latest == occurrence["id"] and occurrence["status"] in {"open", "closed"}
                               and rule["status"] in {"active", "completed"})
                    state = "user_reported_done" if request.operation == "report_done" else "user_reported_not_done"
                    await c.execute("""update jarvis_responsibility_occurrences set report_state=$3,
                        report_event_id=$4,report_revision=$5,reported_at=clock_timestamp(),
                        status=case when $6 and $3='user_reported_done' then 'closed'
                                    when $6 and $3='user_reported_not_done' then 'open' else status end
                        where owner_id=$1 and id=$2""",
                        owner, occurrence["id"], state, source.request_event_id, source.request_revision, current)
                    if request.operation == "report_done":
                        await self._revoke_reminders(c, owner, [occurrence["reminder_id"]])
                    spec = DutySpec.model_validate(decoded(rule["spec"]))
                    await c.execute("""update jarvis_responsibilities set revision=revision+1,
                        status=case when $3 then $4 else status end,
                        updated_at=clock_timestamp() where owner_id=$1 and id=$2""",
                        owner, rule["id"], current and spec.every_days is None,
                        "completed" if request.operation == "report_done" else "active")
            await self._receipt(c, owner, source, operation, request, fingerprint, rule["id"])
            return await self._view(c, owner, rule["id"])

    async def _view(self, c, owner, identity, *, replayed=False):
        row = await c.fetchrow("select * from jarvis_responsibilities where owner_id=$1 and id=$2", owner, identity)
        if row is None:
            raise ToolInputError("responsibility_not_found", "No responsibility exists in this owner scope.")
        occurrences = await c.fetch("""select o.*,r.status as reminder_status,r.triggered_at,
            r.not_before,coalesce(p.availability='available' and p.content is not null,false) as report_available
            from jarvis_responsibility_occurrences o
            left join jarvis_reminders r on r.owner_id=o.owner_id and r.id=o.reminder_id
            left join jarvis_context_events e on e.owner_id=o.owner_id and e.id=o.report_event_id
            left join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
            where o.owner_id=$1 and o.responsibility_id=$2 order by o.scheduled_for desc,o.created_at desc limit 21""",
            owner, identity)
        history = await c.fetch("""select q.report_state,q.request_event_id,q.request_revision,
            q.occurrence_id,q.created_at,coalesce(p.availability='available' and p.content is not null,false) as source_available
            from jarvis_responsibility_requests q
            join jarvis_context_events e on e.owner_id=q.owner_id and e.id=q.request_event_id
            join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
            where q.owner_id=$1 and q.responsibility_id=$2 and q.report_state is not null
            order by q.created_at desc limit 21""", owner, identity)
        items = []
        for occurrence in occurrences[:20]:
            item = dict(occurrence)
            for field in ("check_payload", "check_resolution"):
                if item.get(field) is not None:
                    item[field] = decoded(item[field])
            if not item.pop("report_available"):
                item["report_state"] = "unknown"
            items.append(item)
        reports = [dict(item) for item in history[:20]]
        for item in reports:
            if not item["source_available"]:
                item["report_state"] = "unknown"
        return json_value({"responsibility": dict(row) | {"spec": decoded(row["spec"])},
            "occurrences": items, "report_history": reports, "request_replayed": replayed,
            "has_more_occurrences": len(occurrences)>20, "has_more_reports": len(history)>20,
            "meaning": "Reminder completion/delivery is not action completion or proof of hearing. "
                       "user_reported_done means the user reported it, not that Jarvis observed or verified it. "
                       "Unknown means insufficient evidence, not that the action did not happen. "
                       "A later conflicting report must not be presented as reassurance."})

    async def list(self, owner, *, include_inactive=False, responsibility_id=None):
        async with self.pool.acquire() as c, c.transaction(isolation="repeatable_read", readonly=True):
            rows = await c.fetch("""select id from jarvis_responsibilities where owner_id=$1
                and ($2 or status='active') and ($3::uuid is null or id=$3)
                order by updated_at desc,id limit 51""", owner, include_inactive, responsibility_id)
            from app.reminders.duty_preferences import read_preferences
            return {"duties": [await self._view(c, owner, row["id"]) for row in rows[:50]],
                    "has_more": len(rows)>50,
                    "notification_preferences": await read_preferences(c, owner)}
