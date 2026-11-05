"""Retry-stable reservations. Unknown usage stays charged, never reset to zero."""
from __future__ import annotations

from uuid import UUID

from app.tasks.models import BudgetExceeded, ClaimLost, TaskClaim, TaskError
from app.tasks.repository import TaskRepository


USAGE_SQL = """select coalesce(sum(case when status='settled' then actual_input_tokens else input_tokens end),0)::bigint as input,
    coalesce(sum(case when status='settled' then actual_output_tokens else output_tokens end),0)::bigint as output,
    coalesce(sum(case when status='settled' then actual_tool_steps else tool_steps end),0)::bigint as steps
    from jarvis_task_reservations where owner_id=$1 and status<>'released'"""


def validate_amounts(input_tokens: int, output_tokens: int, tool_steps: int) -> None:
    if any(type(value) is not int or not 0 <= value <= 1_000_000_000 for value in (input_tokens, output_tokens, tool_steps)):
        raise ValueError("resource amounts must be bounded nonnegative integers")


class TaskBudgetRepository:
    def __init__(self, tasks: TaskRepository):
        self.tasks = tasks

    async def reserve(self, claim: TaskClaim, operation_key: str, *, input_tokens=0, output_tokens=0, tool_steps=0,
                      output_is_ceiling=False):
        validate_amounts(input_tokens, output_tokens, tool_steps)
        if not 1 <= len(operation_key.strip()) <= 200:
            raise ValueError("invalid reservation operation identity")
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, attempt = await self.tasks.lock_claim(connection, claim)
            if await self.tasks.context_problem(connection, task):
                raise ClaimLost()
            return await self.reserve_locked(connection, task, attempt, operation_key, input_tokens=input_tokens,
                                             output_tokens=output_tokens, tool_steps=tool_steps,
                                             output_is_ceiling=output_is_ceiling)

    async def reserve_locked(self, connection, task, attempt, operation_key, *, input_tokens=0, output_tokens=0, tool_steps=0,
                             output_is_ceiling=False):
        """Caller must already hold this owner's/task's locks in a transaction."""
        if not connection.is_in_transaction():
            raise ValueError("reservation requires its caller's transaction")
        validate_amounts(input_tokens, output_tokens, tool_steps)
        if type(output_is_ceiling) is not bool or (output_is_ceiling and output_tokens < 1):
            raise ValueError("A model output ceiling must be positive")
        existing = await connection.fetchrow("select * from jarvis_task_reservations where owner_id=$1 and task_id=$2 and operation_key=$3 for update", task.owner_id, task.id, operation_key)
        if existing:
            # A replay keeps its original allocation, never grows it. Released
            # operations may reacquire that same bound only if it still fits.
            if output_is_ceiling and 0 < existing["output_tokens"] <= output_tokens:
                output_tokens = existing["output_tokens"]
            if (existing["input_tokens"], existing["output_tokens"], existing["tool_steps"]) != (input_tokens, output_tokens, tool_steps):
                raise TaskError("task_reservation_conflict", "A logical operation was repeated with different resource bounds.")
            if existing["status"] != "released":
                return dict(existing)
        used = await connection.fetchrow(USAGE_SQL + " and task_id=$2", task.owner_id, task.id)
        daily = await connection.fetchrow(USAGE_SQL + " and accounting_day=(clock_timestamp() at time zone 'UTC')::date", task.owner_id)
        if output_tokens > task.budget.request_output_tokens:
            raise BudgetExceeded("task_request_output_budget_reached")
        if output_is_ceiling and not existing:
            task_remaining = task.budget.task_output_tokens - used["output"]
            daily_remaining = task.budget.owner_daily_tokens - daily["input"] - daily["output"] - input_tokens
            if task_remaining < 1:
                raise BudgetExceeded()
            if daily_remaining < 1:
                raise BudgetExceeded("task_owner_daily_budget_reached")
            output_tokens = min(output_tokens, task_remaining, daily_remaining)
        if (used["input"] + input_tokens > task.budget.task_input_tokens or used["output"] + output_tokens > task.budget.task_output_tokens
                or used["steps"] + tool_steps > task.budget.tool_steps):
            raise BudgetExceeded()
        if daily["input"] + daily["output"] + input_tokens + output_tokens > task.budget.owner_daily_tokens:
            raise BudgetExceeded("task_owner_daily_budget_reached")
        if existing:
            row = await connection.fetchrow("""update jarvis_task_reservations set status='reserved',attempt_id=$3,
                accounting_day=(clock_timestamp() at time zone 'UTC')::date where owner_id=$1 and id=$2 returning *""",
                task.owner_id, existing["id"], attempt["id"])
        else:
            row = await connection.fetchrow("""insert into jarvis_task_reservations(owner_id,task_id,attempt_id,operation_key,status,
                input_tokens,output_tokens,tool_steps) values($1,$2,$3,$4,'reserved',$5,$6,$7) returning *""",
                task.owner_id, task.id, attempt["id"], operation_key, input_tokens, output_tokens, tool_steps)
        await self.tasks.emit(connection, task, "budget_reserved", payload={"operation_key": operation_key, "input_tokens": input_tokens,
                                                                        "output_tokens": output_tokens, "tool_steps": tool_steps}, attempt_id=attempt["id"])
        return dict(row)

    async def submit(self, claim: TaskClaim, reservation_id: UUID, *, provider_dispatch: bool = False,
                     parent_provider_handle: str | None = None):
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, attempt = await self.tasks.lock_claim(connection, claim)
            if await self.tasks.context_problem(connection, task):
                raise ClaimLost()
            if parent_provider_handle is not None:
                if provider_dispatch:
                    raise ValueError("Nested model dispatch cannot replace runtime ownership")
                if attempt["provider_state"] != "live" or attempt["provider_handle"] != parent_provider_handle:
                    raise ClaimLost()
                if await connection.fetchval(
                    "select exists(select 1 from jarvis_task_reservations where owner_id=$1 and attempt_id=$2 "
                    "and is_provider and status in ('submitted','unknown'))", task.owner_id, attempt["id"],
                ):
                    raise TaskError("task_dispatch_pending", "An earlier model request has no terminal usage receipt.")
            if provider_dispatch and attempt["provider_state"] not in {"not_started", "terminal"}:
                raise TaskError("task_dispatch_pending", "An existing provider request must finish before another is submitted.")
            row = await connection.fetchrow("""select *,accounting_day=(clock_timestamp() at time zone 'UTC')::date as current_day
                from jarvis_task_reservations where owner_id=$1 and task_id=$2 and attempt_id=$3 and id=$4 and status='reserved' for update""",
                task.owner_id, task.id, attempt["id"], reservation_id)
            if not row:
                raise TaskError("task_operation_already_submitted", "This operation is not a fresh reserved submission.")
            if not row["current_day"]:
                daily = await connection.fetchrow(USAGE_SQL + " and accounting_day=(clock_timestamp() at time zone 'UTC')::date", task.owner_id)
                if daily["input"] + daily["output"] + row["input_tokens"] + row["output_tokens"] > task.budget.owner_daily_tokens:
                    raise BudgetExceeded("task_owner_daily_budget_reached")
            await connection.execute("""update jarvis_task_reservations set status='submitted',is_provider=$3,
                accounting_day=(clock_timestamp() at time zone 'UTC')::date where owner_id=$1 and id=$2""", task.owner_id, reservation_id, provider_dispatch or parent_provider_handle is not None)
            if provider_dispatch:
                await connection.execute("update jarvis_task_attempts set provider_state='submitting',provider_handle=null where owner_id=$1 and id=$2", task.owner_id, attempt["id"])
            await self.tasks.emit(connection, task, "dispatch_prepared" if provider_dispatch else "operation_submitting", attempt_id=attempt["id"])

    async def settle(self, owner_id: str, task_id: UUID, attempt_id: UUID, reservation_id: UUID,
                     *, input_tokens: int, output_tokens: int, tool_steps: int, provider_terminal: bool = False):
        validate_amounts(input_tokens, output_tokens, tool_steps)
        async with self.tasks.transaction(owner_id) as connection:
            task = await self.tasks.load(connection, owner_id, task_id, lock=True)
            row = await connection.fetchrow("""select * from jarvis_task_reservations
                where owner_id=$1 and task_id=$2 and id=$3 and attempt_id=$4 for update""", owner_id, task_id, reservation_id, attempt_id)
            if row is None or row["status"] not in {"submitted", "unknown", "settled"}:
                raise TaskError("task_reservation_not_submitted", "No matching submitted reservation exists.")
            if provider_terminal and not row["is_provider"]:
                raise TaskError("task_usage_not_provider", "A non-provider operation cannot terminate provider execution.")
            if row["status"] == "settled":
                if (row["actual_input_tokens"], row["actual_output_tokens"], row["actual_tool_steps"]) != (input_tokens, output_tokens, tool_steps):
                    raise TaskError("task_usage_conflict", "Usage was already settled with different counts.")
                return
            await connection.execute("""update jarvis_task_reservations set status='settled',actual_input_tokens=$3,
                actual_output_tokens=$4,actual_tool_steps=$5 where owner_id=$1 and id=$2""", owner_id, reservation_id, input_tokens, output_tokens, tool_steps)
            if provider_terminal:
                await connection.execute("""update jarvis_task_attempts set provider_state='terminal' where owner_id=$1 and id=$2
                    and not exists(select 1 from jarvis_task_reservations where owner_id=$1 and attempt_id=$2
                        and id<>$3 and is_provider and status in ('submitted','unknown'))""", owner_id, attempt_id, reservation_id)
            await self.tasks.emit(connection, task, "usage_settled", payload={"reservation_id": str(reservation_id),
                "input_tokens": input_tokens, "output_tokens": output_tokens, "tool_steps": tool_steps}, attempt_id=attempt_id)

    async def usage(self, owner_id: str, task_id: UUID):
        async with self.tasks.pool.acquire() as connection:
            await self.tasks.load(connection, owner_id, task_id)
            return dict(await connection.fetchrow(USAGE_SQL + " and task_id=$2", owner_id, task_id))
