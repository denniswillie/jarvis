"""Conversation-owned reminder preferences using the existing preference ledger."""
from datetime import time
import json
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.reminders.duties import decoded
from app.tools.inputs import ToolInputError

CAPABILITY = "responsibility_notifications"


class NotificationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timezone: str = "UTC"
    quiet_start: time | None = None
    quiet_end: time | None = None
    paused_until: AwareDatetime | None = None
    confirm_inferred_obligations: bool = Field(default=True, strict=True)

    @field_validator("timezone")
    @classmethod
    def known_timezone(cls, value):
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError("Use an IANA timezone.") from error
        return value

    @model_validator(mode="after")
    def quiet_pair(self):
        if (self.quiet_start is None) != (self.quiet_end is None):
            raise ValueError("Set both quiet times, or neither.")
        if self.quiet_start is not None:
            if self.quiet_start == self.quiet_end:
                raise ValueError("Quiet hours must have different start and end times.")
            if self.quiet_start.tzinfo is not None or self.quiet_end.tzinfo is not None:
                raise ValueError("Quiet times are local wall-clock times; supply timezone separately.")
        return self


class SetNotificationPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0, strict=True)
    policy: NotificationPolicy


async def read_preferences(c, owner):
    row = await c.fetchrow("""select q.revision,p.content,p.availability from jarvis_capability_preferences q
        join jarvis_context_payloads p on p.owner_id=q.owner_id and p.id=q.policy_payload_id
        where q.owner_id=$1 and q.capability=$2 and q.knowledge_area_id is null
        order by q.revision desc limit 1""", owner, CAPABILITY)
    if row is None:
        return {"revision":0, "policy":NotificationPolicy().model_dump(mode="json")}
    return {"revision":row["revision"],
            "policy":decoded(row["content"]) if row["availability"]=="available" and row["content"] else None,
            "meaning":"These preferences govern duties, not ordinary replies or unrelated expert results. "
                      "Already-started speech or submitted push cannot be undone. Missing policy evidence pauses duty notices."}


class ResponsibilityPreferences:
    def __init__(self, repository):
        self.repository = repository

    async def save(self, owner, session_id, source, request: SetNotificationPreferences):
        async with self.repository.pool.acquire() as c, c.transaction():
            await self.repository._lock(c, owner)
            original = await self.repository._require_source(c, owner, session_id, source)
            old = await c.fetchrow("""select p.content from jarvis_capability_preferences q
                join jarvis_context_payloads p on p.owner_id=q.owner_id and p.id=q.policy_payload_id
                where q.owner_id=$1 and q.capability=$2 and q.request_event_id=$3 and q.request_revision=$4
                order by q.revision desc limit 1""", owner, CAPABILITY,
                source.request_event_id, source.request_revision)
            body = request.policy.model_dump(mode="json")
            if old:
                if decoded(old["content"]) != body:
                    raise ToolInputError("request_conflict", "This user request already changed preferences; inspect the current policy.")
                return {"notification_preferences":await read_preferences(c, owner), "request_replayed":True}
            current = await read_preferences(c, owner)
            if current["revision"] != request.expected_revision:
                raise ToolInputError("revision_conflict", "List responsibilities to obtain current notification preferences.")
            payload = await c.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available',$2::jsonb) returning id""", owner, json.dumps(body))
            await c.execute("""insert into jarvis_context_payload_dependencies
                (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""", owner, original, payload)
            await c.execute("""insert into jarvis_capability_preferences
                (owner_id,capability,revision,request_event_id,request_revision,policy_payload_id)
                values($1,$2,$3,$4,$5,$6)""", owner, CAPABILITY, current["revision"]+1,
                source.request_event_id, source.request_revision, payload)
            await c.execute("select refresh_jarvis_responsibility_quiet($1,clock_timestamp())", owner)
            return {"notification_preferences":await read_preferences(c, owner), "request_replayed":False}
