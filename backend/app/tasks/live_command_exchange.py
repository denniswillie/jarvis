"""Worker/API command exchange on existing task events, not another agent loop.

Only the active API route executes its domain commands. Dispatch acceptance is
durable before crossing that boundary; a lost reply never authorizes resubmission.
The domain's existing operation receipt remains the authority for the effect.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
from uuid import NAMESPACE_URL, UUID, uuid5

from app.tasks.live_commands import live_command_source
from app.tasks.models import TaskClaim, TaskError, digest, json_object, json_text


REQUEST = "live_command_requested"
STARTED = "live_command_dispatch_started"
REPLIED = "live_command_replied"
FORMAT = "jarvis.live_command_exchange.v1"


def _error(code):
    return TaskError(code, "The exact Live command is unavailable or changed; do not repeat its effect.")


def _bounded(value):
    result = json_object(value)
    if len(json_text(result).encode("utf-8")) > 64000:
        raise _error("live_command_payload_too_large")
    return result


def _identity(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 200 or "\x00" in value:
        raise _error("live_command_identity_invalid")
    return value


@dataclass(frozen=True)
class LiveCommandTicket:
    owner_id: str
    task_id: UUID
    event_id: UUID
    call_key: str


@dataclass(frozen=True)
class LiveCommandReceipt:
    """Trusted domain projection, not an arbitrary copied document/tool body.

Callers project their existing receipt fields and supply any additional source
dependencies. A worker or phone cannot supply this object across the wire.
"""
    result: dict
    source_payload_ids: tuple[UUID, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "result", _bounded(self.result))
        if len(self.source_payload_ids) > 64 or any(not isinstance(x, UUID) for x in self.source_payload_ids):
            raise ValueError("Command receipts require bounded canonical source roots")


@dataclass(frozen=True)
class LiveCommandInvocation:
    """Trusted API-side invocation of an actual, source-bound Hermes tool call."""
    tasks: object
    task: object
    claim: object
    source: object
    operation_id: UUID
    request_payload_id: UUID

    async def authorize(self, connection):
        current, _ = await self.tasks.lock_claim(connection, self.claim)
        source = await live_command_source(self.tasks, connection, current)
        if source is None or source.id != self.source.id or await self.tasks.context_problem(connection, current):
            raise _error("live_command_context_unavailable")
        roots = (source.payload_id, self.request_payload_id)
        for root in roots:
            if await connection.fetchval("select lock_jarvis_copy_source($1,$2)", current.owner_id, root) != "available":
                raise _error("live_command_source_withdrawn")
        return roots


class LiveCommandExchange:
    def __init__(self, repository):
        self.tasks = repository

    async def _lookup(self, connection, owner, task_id, kind, call_key):
        row = await connection.fetchrow("""select e.*,p.content,p.availability
            from jarvis_task_events e join jarvis_context_payloads p
            on p.owner_id=e.owner_id and p.id=e.payload_id
            where e.owner_id=$1 and e.task_id=$2 and e.event_type=$3
              and p.content->>'call_key'=$4 order by e.sequence limit 1""",
            owner, task_id, kind, call_key)
        return self._available(row)

    @staticmethod
    def _available(row):
        if row is None:
            return None
        if row["availability"] != "available" or row["content"] is None:
            raise _error("live_command_source_withdrawn")
        return dict(row) | {"content": json_object(row["content"])}

    async def _emit(self, connection, task, kind, body, *, attempt_id, roots):
        await self.tasks.emit(connection, task, kind, payload=body, attempt_id=attempt_id,
                              source_payload_ids=tuple(sorted(set(roots))))
        row = await self._lookup(connection, task.owner_id, task.id, kind, body["call_key"])
        if row is None:
            raise _error("live_command_event_unconfirmed")
        return row

    async def request(self, context, name, arguments, *, provider_call_id):
        """Trusted worker only. Caller charges its ordinary tool budget."""
        name, call_id = _identity(name), _identity(provider_call_id)
        arguments = _bounded(arguments)
        claim = context.claim
        key = str(uuid5(NAMESPACE_URL, f"{claim.task.id}:{claim.attempt_id}:{call_id}"))
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, _ = await self.tasks.lock_claim(connection, claim)
            source = await live_command_source(self.tasks, connection, task)
            if source is None:
                raise _error("live_command_profile_required")
            parameters = source.content["parameters"]
            if name not in {tool["name"] for tool in parameters["command_catalog"]}:
                raise _error("live_command_not_offered")
            body = {"format": FORMAT, "call_key": key, "provider_call_id": call_id,
                    "name": name, "arguments": arguments, "route": parameters["route"],
                    "task_revision": claim.task_revision, "attempt_id": str(claim.attempt_id),
                    "fence": claim.fence,
                    "backend_session_key": f"hermes:{task.id}:{claim.attempt_id}"}
            previous = await self._lookup(connection, task.owner_id, task.id, REQUEST, key)
            if previous is not None:
                if digest(previous["content"]) != digest(body):
                    raise _error("live_command_call_conflict")
                row = previous
            else:
                inputs = await connection.fetch("""select e.payload_id from jarvis_task_events e
                    where e.owner_id=$1 and e.task_id=$2 and e.event_type in ('work_input','live_command_replied')
                    order by e.sequence desc limit 512""", task.owner_id, task.id)
                row = await self._emit(connection, task, REQUEST, body, attempt_id=claim.attempt_id,
                                       roots=(source.payload_id, *(item["payload_id"] for item in inputs)))
            return LiveCommandTicket(task.owner_id, task.id, row["id"], key)

    async def response(self, ticket):
        row = await self.response_record(ticket)
        return _bounded(row["content"]["result"]) if row is not None else None

    async def response_record(self, ticket):
        async with self.tasks.pool.acquire() as connection:
            row = await self._lookup(connection, ticket.owner_id, ticket.task_id, REPLIED, ticket.call_key)
        if row is None:
            return None
        body = row["content"]
        if body.get("request_event_id") != str(ticket.event_id):
            raise _error("live_command_reply_conflict")
        return row

    async def wait(self, context, ticket, *, timeout=15, poll_seconds=.1):
        if not 0 < timeout <= 120 or not .01 <= poll_seconds <= 1:
            raise ValueError("Invalid command reply observation bounds")
        if ticket.owner_id != context.task.owner_id or ticket.task_id != context.task.id:
            raise _error("live_command_ticket_scope_mismatch")
        try:
            async with asyncio.timeout(timeout):
                while True:
                    result = await self.response(ticket)
                    if result is not None:
                        async with self.tasks.transaction(ticket.owner_id) as connection:
                            task, _ = await self.tasks.lock_claim(connection, context.claim)
                            if await self.tasks.context_problem(connection, task):
                                raise _error("live_command_context_unavailable")
                        return result
                    await asyncio.sleep(poll_seconds)
        except TimeoutError:
            # The command or its domain effect can exist after this observer stops.
            # Keep the original identity/receipt; never report "not executed".
            raise _error("live_command_reply_unconfirmed") from None

    async def dispatch(self, event_id, *, route, is_current, execute, invocation_context=False):
        """API process only; execute returns a source-bound domain receipt.

        There is no transaction/owner lock held while invoking the domain. The
        accepted marker is the command handoff boundary, not evidence of effect
        completion. Later cancellation must inspect the owning domain receipt.
        """
        owner = route.scope.owner_id
        async with self.tasks.transaction(owner) as connection:
            row = self._available(await connection.fetchrow("""select e.*,p.content,p.availability
                from jarvis_task_events e join jarvis_context_payloads p
                on p.owner_id=e.owner_id and p.id=e.payload_id
                where e.owner_id=$1 and e.id=$2""", owner, event_id))
            if row is None or row["event_type"] != REQUEST:
                raise _error("live_command_request_unavailable")
            body = row["content"]
            if body.get("format") != FORMAT:
                raise _error("live_command_request_invalid")
            if body.get("route") != route.model_dump(mode="json") or not is_current():
                return None
            task = await self.tasks.load(connection, owner, row["task_id"])
            claim = TaskClaim(task, UUID(body["attempt_id"]), body["task_revision"], body["fence"],
                              datetime.now(timezone.utc), "live", None, None)
            task, _ = await self.tasks.lock_claim(connection, claim)
            source = await live_command_source(self.tasks, connection, task)
            if (source is None or source.content["parameters"]["route"] != body["route"]
                    or row["attempt_id"] != claim.attempt_id or row["task_revision"] != claim.task_revision
                    or body["name"] not in {tool["name"] for tool in source.content["parameters"]["command_catalog"]}
                    or body["backend_session_key"] != f"hermes:{task.id}:{claim.attempt_id}"
                    or body["call_key"] != str(uuid5(NAMESPACE_URL,
                        f"{task.id}:{claim.attempt_id}:{_identity(body['provider_call_id'])}"))):
                raise _error("live_command_request_changed")
            reply = await self._lookup(connection, owner, task.id, REPLIED, body["call_key"])
            if reply is not None:
                return _bounded(reply["content"]["result"])
            if await self._lookup(connection, owner, task.id, STARTED, body["call_key"]) is not None:
                return None  # In flight or outcome unknown; never blindly redispatch.
            await self._emit(connection, task, STARTED,
                {"format": FORMAT, "call_key": body["call_key"], "request_event_id": str(event_id)},
                attempt_id=claim.attempt_id, roots=(row["payload_id"],))
            invocation = LiveCommandInvocation(self.tasks, task, claim, source,
                                                UUID(body["call_key"]), row["payload_id"])

        if not is_current():
            receipt = LiveCommandReceipt({"status": "failed", "outcome": "rejected",
                "effect_committed": False, "error": {"code": "live_command_route_closed"}})
        else:
            try:
                receipt = await execute(body["name"], _bounded(body["arguments"]),
                    provider_call_id=body["provider_call_id"],
                    backend_session_key=body["backend_session_key"], admit_if=is_current,
                    **({"invocation": invocation} if invocation_context else {}))
                if not isinstance(receipt, LiveCommandReceipt):
                    raise TypeError("A trusted projected domain receipt is required")
            except asyncio.CancelledError:
                raise  # Preserve STARTED as unknown; domain authority is not cancelled here.
            except Exception:
                receipt = LiveCommandReceipt({"status": "failed", "outcome": "outcome_unknown",
                    "effect_committed": None, "error": {"code": "live_command_outcome_unknown"}})

        async with self.tasks.transaction(owner) as connection:
            # Record the actual outcome even if the worker was cancelled after
            # acceptance. Do not revive its claim or turn that into another effect.
            current = await self.tasks.load(connection, owner, task.id)
            await self._emit(connection, current, REPLIED,
                {"format": FORMAT, "call_key": body["call_key"],
                 "request_event_id": str(event_id), "result": receipt.result},
                attempt_id=claim.attempt_id, roots=(row["payload_id"], *receipt.source_payload_ids))
        return receipt.result
