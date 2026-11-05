"""Expose the admitted API domain tools to the same native Hermes agent.

This is an RPC adapter over existing task events, not an interpreter or another
agent. Domain services own effects; immutable reply records back result claims.
"""
from copy import deepcopy
from uuid import UUID

from app.tasks.live_command_exchange import LiveCommandExchange, LiveCommandTicket
from app.tasks.live_commands import WORK_PROFILE, _command_catalog, live_command_source
from app.tasks.models import TaskError, json_object


class LiveWorkToolBridge:
    def __init__(self, base, catalog):
        self.base, self.context = base, base.context
        self.definitions = {item["name"]: deepcopy(item) for item in catalog}
        if self.definitions.keys() & {item["name"] for item in base.catalog()}:
            raise ValueError("Live work tools must have distinct names")
        self.exchange = LiveCommandExchange(self.context.tasks)
        self.result_roots = set()

    def __getattr__(self, name):
        return getattr(self.base, name)

    @classmethod
    async def attach(cls, base):
        async with base.context.tasks.transaction(base.context.task.owner_id) as connection:
            task, _ = await base.context.tasks.lock_claim(connection, base.context.claim)
            source = await live_command_source(base.context.tasks, connection, task)
            if source is None or source.content.get("profile") != WORK_PROFILE:
                return base
            catalog = _command_catalog(source.content["parameters"]["command_catalog"])
        return cls(base, catalog)

    def catalog(self):
        return self.base.catalog() + deepcopy(list(self.definitions.values()))

    async def invoke(self, name, arguments, *, provider_call_id):
        if name not in self.definitions:
            return await self.base.invoke(name, arguments, provider_call_id=provider_call_id)
        await self.base._check()
        reservation = await self.context.reserve("live-tool:" + provider_call_id, tool_steps=1)
        await self.context.budgets.submit(self.context.claim, reservation["id"])
        try:
            ticket = await self.exchange.request(self.context, name, arguments, provider_call_id=provider_call_id)
            result = await self.exchange.wait(self.context, ticket)
            record = await self.exchange.response_record(ticket)
            if record is None:
                raise TaskError("live_command_reply_unconfirmed", "The command reply is unavailable.")
            self.result_roots.add(record["payload_id"])
            if result.get("effect_committed") is True:
                result = result | {"task_operation": {
                    "operation_ref": "live_" + ticket.call_key.replace("-", ""),
                    "action_id": ticket.call_key, "state": "accepted", "replayed": False}}
            return result
        finally:
            await self.context.settle(reservation["id"], input_tokens=0, output_tokens=0, tool_steps=1)

    async def verified_operation(self, reference):
        if not isinstance(reference, str) or not reference.startswith("live_"):
            return await self.base.verified_operation(reference)
        try:
            key = str(UUID(reference[5:]))
        except ValueError:
            raise TaskError("task_result_operation_unconfirmed", "Unknown domain operation reference.") from None
        await self.base._check()
        async with self.context.tasks.transaction(self.context.task.owner_id) as connection:
            row = await self.exchange._lookup(connection, self.context.task.owner_id, self.context.task.id,
                                               "live_command_replied", key)
            if row is None or row["content"]["result"].get("effect_committed") is not True:
                raise TaskError("task_result_operation_unconfirmed", "No committed domain receipt exists.")
            if await connection.fetchval("select lock_jarvis_copy_source($1,$2)",
                                         self.context.task.owner_id, row["payload_id"]) != "available":
                raise TaskError("task_result_operation_unconfirmed", "The operation source was withdrawn.")
        self.result_roots.add(row["payload_id"])
        return json_object(row["content"]["result"]) | {"task_operation": {
            "operation_ref": reference, "action_id": key, "state": "accepted", "replayed": True}}

    async def guard(self, connection):
        roots = frozenset(self.result_roots)
        if not roots:
            return
        rows = await connection.fetch("""select id,availability from jarvis_context_payloads
            where owner_id=$1 and id=any($2::uuid[])""", self.context.task.owner_id, list(roots))
        if {row["id"] for row in rows if row["availability"] == "available"} != roots:
            raise TaskError("live_command_source_withdrawn", "A supplied domain result was withdrawn.")
