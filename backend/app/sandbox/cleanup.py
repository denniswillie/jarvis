"""Durable, scoped native cleanup. Provider I/O never holds a database lock."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
from uuid import UUID, uuid4

from app.sandbox.models import SandboxError
from app.tasks.models import TaskError, TaskStatus
from app.tasks.repository import ACTIVE_ATTEMPTS

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CleanupClaim:
    owner_id: str = field(repr=False)
    task_id: UUID
    allocation_id: UUID
    token: UUID
    generation: int
    lease_seconds: int
    action: str = "kill"


def resource_ids(values):
    result = set(values)
    if len(result) > 100 or any(not isinstance(value,str) or not 1 <= len(value) <= 200
                              or value != value.strip() or "\x00" in value for value in result):
        raise ValueError("Invalid bounded native sandbox identities")
    return result


def known_ids(row):
    result = resource_ids(row["cleanup_observed_ids"])
    if row["sandbox_id"] is not None:
        result.add(row["sandbox_id"])
    return result


class SandboxCleanupRepository:
    def __init__(self, allocations, *, check_seconds=5, retry_seconds=30):
        if not 1 <= check_seconds <= 300 or not 1 <= retry_seconds <= 3600:
            raise ValueError("Invalid sandbox cleanup schedule")
        self.allocations, self.tasks = allocations, allocations.tasks
        self.check_seconds, self.retry_seconds = check_seconds, retry_seconds

    async def due_owners(self):
        async with self.tasks.pool.acquire() as connection:
            rows = await connection.fetch("""select owner_id from jarvis_sandbox_allocations
                where phase<>'cleaned' and next_cleanup_at<=clock_timestamp()
                and (cleanup_lease_until is null or cleanup_lease_until<=clock_timestamp())
                group by owner_id order by min(next_cleanup_at),owner_id limit 100""")
            return [row["owner_id"] for row in rows]

    async def _reason(self, connection, task, row, context_problem):
        if row["phase"] == "cleanup_requested":
            return row["reason"] or "sandbox_cleanup_requested"
        if not row["within_deadline"]:
            return "sandbox_deadline_reached"
        if row.get("reusable"):
            if context_problem:
                return context_problem
            if task.status in {TaskStatus.CANCELLED, TaskStatus.CANCEL_REQUESTED} or task.pending_status is not None:
                return "sandbox_task_cancelled"
            # Each retained workspace still depends on its current grant.
            try:
                await self.tasks.validate_grant(connection, task.owner_id, row["grant_id"])
            except TaskError:
                return "sandbox_grant_revoked"
            if row["phase"] == "pausing":
                # An expired pause lease has uncertain provider outcome. Never
                # reattach/retry pause while an old native request could arrive.
                return "sandbox_pause_outcome_unknown"
            if row["phase"] == "paused":
                return None
            if row["phase"] == "ready":
                active = await connection.fetchval("""select status=any($3::text[]) from jarvis_task_attempts
                    where owner_id=$1 and id=$2""", task.owner_id, row["attempt_id"], list(ACTIVE_ATTEMPTS))
                if active:
                    if task.revision != row["task_revision"] or task.current_attempt_id != row["attempt_id"]:
                        return "sandbox_task_superseded"
                    if task.active_seconds >= task.budget.active_seconds:
                        return "sandbox_task_budget_reached"
                    return None
                unfinished = await connection.fetchval("""select exists(select 1 from jarvis_sandbox_commands
                    where owner_id=$1 and allocation_id=$2 and phase in ('submitting','running','unknown'))""",
                    task.owner_id, row["id"])
                return "sandbox_command_unresolved" if unfinished else "sandbox_idle_pause"
        if task.revision != row["task_revision"] or task.current_attempt_id != row["attempt_id"]:
            return "sandbox_task_superseded"
        if task.status not in {TaskStatus.RUNNING,TaskStatus.UNKNOWN} or task.pending_status is not None:
            return "sandbox_task_inactive"
        if task.active_seconds >= task.budget.active_seconds:
            return "sandbox_task_budget_reached"
        if context_problem:
            return context_problem
        active = await connection.fetchval("""select status=any($3::text[]) from jarvis_task_attempts
            where owner_id=$1 and id=$2""", task.owner_id,row["attempt_id"],list(ACTIVE_ATTEMPTS))
        # Lease loss alone leaves a bounded window for task recovery. It never
        # authorizes stale execution; the immutable allocation deadline still ends it.
        return None if active else "sandbox_attempt_inactive"

    async def claim_next(self, owner, worker_id, *, lease_seconds=90, allocation_id=None):
        if not isinstance(worker_id,str) or not worker_id.strip() or len(worker_id)>100:
            raise ValueError("Invalid sandbox cleanup worker identity")
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 300:
            raise ValueError("Invalid sandbox cleanup lease")
        async with self.tasks.transaction(owner) as connection:
            candidates = await connection.fetch("""select id,task_id from jarvis_sandbox_allocations
                where owner_id=$1 and phase<>'cleaned' and ($2::uuid is null or id=$2)
                and (next_cleanup_at<=clock_timestamp() or $2::uuid is not null)
                and (cleanup_lease_until is null or cleanup_lease_until<=clock_timestamp())
                order by next_cleanup_at,id limit 32""",owner,allocation_id)
            for candidate in candidates:
                task = await self.tasks.load(connection,owner,candidate["task_id"],lock=True)
                context_problem = await self.tasks.context_problem(connection,task)
                row = await self.allocations._load(connection,owner,candidate["id"])
                reason = await self._reason(connection,task,row,context_problem)
                if reason is None:
                    await connection.execute("""update jarvis_sandbox_allocations set
                        next_cleanup_at=least(deadline,clock_timestamp()+$3*interval '1 second')
                        where owner_id=$1 and id=$2""",owner,row["id"],float(self.check_seconds))
                    continue
                if row["submission_token"] is None:
                    await connection.execute("""update jarvis_sandbox_allocations set phase='cleaned',
                        reason=$3,updated_at=clock_timestamp() where owner_id=$1 and id=$2""",owner,row["id"],reason)
                    await self.tasks.emit(connection,task,"sandbox_cleaned",
                        payload={"allocation_id":str(row["id"]),"never_submitted":True},
                        attempt_id=row["attempt_id"],revision=row["task_revision"])
                    continue
                token = uuid4()
                generation = row["cleanup_generation"]+1
                action = "pause" if reason == "sandbox_idle_pause" else "kill"
                await connection.execute("""update jarvis_sandbox_allocations set phase=$8,
                    reason=$3,cleanup_token=$4,cleanup_worker=$5,cleanup_generation=$6,
                    cleanup_lease_until=clock_timestamp()+$7*interval '1 second',
                    next_cleanup_at=clock_timestamp()+$7*interval '1 second',updated_at=clock_timestamp()
                    where owner_id=$1 and id=$2""",owner,row["id"],reason,token,worker_id,generation,float(lease_seconds),
                    "pausing" if action == "pause" else "cleanup_requested")
                return CleanupClaim(owner,task.id,row["id"],token,generation,lease_seconds,action)
        return None

    async def _locked(self, connection, claim):
        task = await self.tasks.load(connection,claim.owner_id,claim.task_id,lock=True)
        row = await connection.fetchrow("""select *,cleanup_lease_until>clock_timestamp() as cleanup_live
            from jarvis_sandbox_allocations where owner_id=$1 and task_id=$2 and id=$3 for update""",
            claim.owner_id,claim.task_id,claim.allocation_id)
        if (row is None or row["phase"]!=("pausing" if claim.action == "pause" else "cleanup_requested") or not row["cleanup_live"]
                or row["cleanup_token"]!=claim.token or row["cleanup_generation"]!=claim.generation):
            raise SandboxError("sandbox_cleanup_claim_lost")
        return task,row

    async def _renew(self, connection, claim):
        await connection.execute("""update jarvis_sandbox_allocations set
            cleanup_lease_until=clock_timestamp()+$3*interval '1 second',
            next_cleanup_at=clock_timestamp()+$3*interval '1 second'
            where owner_id=$1 and id=$2""",claim.owner_id,claim.allocation_id,float(claim.lease_seconds))

    async def authorize(self, claim):
        async with self.tasks.transaction(claim.owner_id) as connection:
            _,row = await self._locked(connection,claim)
            await self._renew(connection,claim)
            return dict(row)

    async def observe(self, claim, identities):
        identities = resource_ids(identities)
        async with self.tasks.transaction(claim.owner_id) as connection:
            _,row = await self._locked(connection,claim)
            identities = resource_ids(identities | known_ids(row))
            await connection.execute("""update jarvis_sandbox_allocations set cleanup_observed_ids=$3
                where owner_id=$1 and id=$2""",claim.owner_id,claim.allocation_id,sorted(identities))
            await self._renew(connection,claim)
            return identities

    async def finish_pause(self, claim, handle):
        async with self.tasks.transaction(claim.owner_id) as connection:
            task, row = await self._locked(connection, claim)
            if (claim.action != "pause" or handle.state != "paused" or handle.sandbox_id != row["sandbox_id"]
                    or handle.provider_scope != self.allocations.native_scope(row)
                    or handle.policy != self.allocations.policy(row)):
                raise SandboxError("sandbox_pause_unconfirmed")
            await connection.execute("""update jarvis_sandbox_allocations set phase='paused',
                cleanup_token=null,cleanup_lease_until=null,cleanup_error=null,
                next_cleanup_at=least(deadline,clock_timestamp()+$3*interval '1 second'),
                updated_at=clock_timestamp() where owner_id=$1 and id=$2""",
                claim.owner_id, claim.allocation_id, float(self.check_seconds))
            await self.tasks.emit(connection, task, "sandbox_paused",
                payload={"allocation_id": str(row["id"])}, attempt_id=row["attempt_id"])
            return {"allocation_id": str(row["id"]), "state": "paused", "error": None}

    async def finish(self, claim, *, absent_ids, discovery_empty, error=None):
        absent = resource_ids(absent_ids)
        if type(discovery_empty) is not bool or (error is not None and (not isinstance(error,str) or not 1<=len(error)<=100)):
            raise ValueError("Invalid cleanup evidence")
        async with self.tasks.transaction(claim.owner_id) as connection:
            task,row = await self._locked(connection,claim)
            expected = known_ids(row)
            complete = error is None and discovery_empty and bool(expected) and expected <= absent
            if not complete and error is None:
                error = "sandbox_native_resources_remain" if expected or not discovery_empty else "sandbox_creation_outcome_unresolved"
            phase = "cleaned" if complete else "cleanup_requested"
            await connection.execute("""update jarvis_sandbox_allocations set phase=$3,cleanup_error=$4,
                cleanup_token=null,cleanup_lease_until=null,next_cleanup_at=clock_timestamp()+$5*interval '1 second',
                updated_at=clock_timestamp() where owner_id=$1 and id=$2""",
                claim.owner_id,claim.allocation_id,phase,error,float(self.retry_seconds))
            if complete:
                await self.tasks.emit(connection,task,"sandbox_cleaned",
                    payload={"allocation_id":str(row["id"]),"native_absence_verified":True},
                    attempt_id=row["attempt_id"],revision=row["task_revision"])
            return {"allocation_id":str(row["id"]),"state":phase,"error":error}


class SandboxCleaner:
    """A replaceable provider observer. It never creates or resumes a sandbox."""
    def __init__(self, repository, provider, *, lease_seconds=90):
        if type(lease_seconds) is not int or not 1<=lease_seconds<=300 or lease_seconds<=provider.configuration.request_timeout_seconds+2:
            raise ValueError("Native request deadlines must fit inside the cleanup lease")
        self.repository,self.provider,self.lease_seconds = repository,provider,lease_seconds

    async def run_once(self, owner, worker_id, *, allocation_id=None):
        claim = await self.repository.claim_next(owner,worker_id,lease_seconds=self.lease_seconds, allocation_id=allocation_id)
        if claim is None:
            return None
        try:
            if claim.action == "pause":
                row = await self.repository.authorize(claim)
                handle = await self.provider.pause(self.repository.allocations.handle(row))
                return await self.repository.finish_pause(claim, handle)
            return await self.clean(claim)
        except SandboxError as error:
            if error.code=="sandbox_cleanup_claim_lost":
                return {"allocation_id":str(claim.allocation_id),"state":"lease_lost","error":error.code}
            return await self.repository.finish(claim,absent_ids=(),discovery_empty=False,error=error.code)

    async def clean(self, claim):
        row = await self.repository.authorize(claim)
        scope,policy = self.repository.allocations.native_scope(row),self.repository.allocations.policy(row)
        identities,absent,error = known_ids(row),set(),None
        try:
            identities.update(await self.provider.discover(scope,policy))
        except SandboxError as failure:
            error = failure.code
        # Persist discovered handles before kill. A crash after native deletion
        # must not erase the evidence needed to resolve an uncertain create.
        identities = await self.repository.observe(claim,identities)
        for identity in sorted(identities):
            await self.repository.authorize(claim)
            kill_error = None
            try:
                await self.provider.kill(identity,scope,policy)
            except SandboxError as failure:
                if failure.code=="sandbox_not_found":
                    absent.add(identity)
                    continue
                kill_error = failure.code
            await self.repository.authorize(claim)
            try:
                await self.provider.info(identity,scope,policy)
                error = error or kill_error or "sandbox_native_resource_still_present"
            except SandboxError as failure:
                if failure.code=="sandbox_not_found":
                    absent.add(identity)
                else:
                    error = error or kill_error or failure.code
        await self.repository.authorize(claim)
        empty = False
        try:
            remaining = await self.provider.discover(scope,policy)
            await self.repository.observe(claim,remaining)
            empty = not remaining
        except SandboxError as failure:
            error = error or failure.code
        return await self.repository.finish(claim,absent_ids=absent,discovery_empty=empty,error=error)

    async def run(self, stop, worker_id, *, poll_seconds=5, owner_id=None):
        if not 0<poll_seconds<=60:
            raise ValueError("Invalid sandbox cleanup polling interval")
        while not stop.is_set():
            try:
                for owner in ([owner_id] if owner_id is not None else await self.repository.due_owners()):
                    if stop.is_set():
                        break
                    await self.run_once(owner,worker_id)
            except Exception as error:
                # Exception bodies may include private SQL/provider details.
                logger.warning("Sandbox cleanup pass failed (%s)",type(error).__name__)
            try:
                await asyncio.wait_for(stop.wait(),timeout=poll_seconds)
            except TimeoutError:
                pass
