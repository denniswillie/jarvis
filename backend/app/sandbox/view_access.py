"""Read-only authorization for an existing desktop, never allocation/admission.

The provider is checked separately, outside SQL. A ledger handle is not evidence
that a native desktop is running. No method reserves, resumes or extends work.
"""
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from uuid import UUID

from app.errors import JarvisHTTPError
from app.sandbox.allocations import SandboxAllocationRepository
from app.sandbox.models import SandboxInfo, SandboxPolicy


def unavailable(code="sandbox_not_running", message="Your sandbox isn't running. Ask Jarvis to start it.", status=409):
    return JarvisHTTPError(status, code, message)


@dataclass(frozen=True)
class DesktopTarget:
    owner_id: str = field(repr=False)
    session_id: UUID
    task_id: UUID
    allocation_id: UUID
    generation: str
    title: str
    deadline: datetime
    handle: SandboxInfo = field(repr=False)

    def wire(self):
        return {"task_id": str(self.task_id), "allocation_id": str(self.allocation_id),
            "generation": self.generation, "title": self.title, "expires_at": self.deadline.isoformat()}


class SandboxViewAccess:
    def __init__(self, tasks, *, owner_id):
        self.tasks, self.owner_id = tasks, owner_id
        self.closed = False

    def check(self, owner_id):
        if self.closed:
            raise unavailable("sandbox_view_closed", "Sandbox viewing is unavailable.", 503)
        if owner_id != self.owner_id:
            raise unavailable("sandbox_view_owner_mismatch", "This sandbox belongs to another account.", 403)

    async def _session(self, connection, owner_id, session_id):
        self.check(owner_id)
        if not isinstance(session_id, UUID) or not await connection.fetchval(
            "select true from jarvis_sessions where owner_id=$1 and id=$2", owner_id, session_id):
            raise unavailable("sandbox_view_session_unavailable", "The current session is unavailable.", 404)

    async def _target(self, connection, owner_id, session_id, row):
        task = await self.tasks.load(connection, owner_id, row["task_id"])
        if task.session_id != session_id:
            raise unavailable("sandbox_view_session_mismatch", "Select work in the current session.", 403)
        if not row["within_deadline"]:
            raise unavailable("sandbox_view_access_ended", "This sandbox's viewing time has ended.", 410)
        if (row["phase"] != "ready"
                or row["sandbox_id"] is None):
            if row["phase"] in {"reserved", "creating", "uncertain", "preparing", "resuming", "pausing", "cleanup_requested"}:
                raise unavailable("sandbox_not_ready", "The sandbox isn't ready to view yet.")
            raise unavailable()
        if task.current_attempt_id != row["attempt_id"] or task.revision != row["task_revision"]:
            raise unavailable("sandbox_view_replaced", "This work has changed. Open its current sandbox.", 409)
        problem = await self.tasks.context_problem(connection, task, lock=False)
        if problem is not None:
            raise unavailable("sandbox_view_access_ended", "Access to this work has ended.", 403)
        attached = await connection.fetchval("""select true from jarvis_task_grants
            where owner_id=$1 and task_id=$2 and grant_id=$3""", owner_id, task.id, row["grant_id"])
        grant = await self.tasks.validate_grant(connection, owner_id, row["grant_id"], lock=False)
        handle = SandboxAllocationRepository.handle(row)
        grant_scope = json.loads(grant["scope"]) if isinstance(grant["scope"], str) else grant["scope"]
        if (not attached or grant["capability"] != "sandbox.execute" or handle.policy.template != "desktop"
                or SandboxPolicy.model_validate(grant_scope).fingerprint != handle.policy.fingerprint):
            raise unavailable("sandbox_desktop_unavailable", "This work has no desktop to view.", 409)
        # Attempt/revision identifies the holder after continuation. A stale
        # phone request cannot follow an allocation into different work.
        generation = hashlib.sha256(json.dumps([str(row["id"]), str(task.id), str(row["attempt_id"]),
            row["task_revision"], row["sandbox_id"], row["policy_digest"]], separators=(",", ":")).encode()).hexdigest()
        return DesktopTarget(owner_id, session_id, task.id, row["id"], generation,
            (task.title or "Your sandbox")[:200], row["deadline"], handle)

    async def resolve(self, owner_id, session_id, *, task_id, allocation_id=None, generation=None):
        self.check(owner_id)
        if not isinstance(task_id, UUID) or (allocation_id is not None and not isinstance(allocation_id, UUID)):
            raise unavailable("sandbox_selection_required", "Choose the work you'd like to view.", 422)
        async with self.tasks.pool.acquire() as connection:
            async with connection.transaction(readonly=True):
                await self._session(connection, owner_id, session_id)
                rows = await connection.fetch("""select *,deadline>clock_timestamp() as within_deadline
                    from jarvis_sandbox_allocations where owner_id=$1 and task_id=$2
                    and ($3::uuid is null or id=$3) and ($3::uuid is not null or phase='ready') limit 2""",
                    owner_id, task_id, allocation_id)
                if not rows:
                    raise unavailable()
                if len(rows) != 1:
                    raise unavailable("sandbox_selection_required", "Choose the sandbox you'd like to view.", 409)
                target = await self._target(connection, owner_id, session_id, rows[0])
        if generation is not None and generation != target.generation:
            raise unavailable("sandbox_view_replaced", "This sandbox view is no longer current.", 409)
        return target

    async def allocation(self, owner_id, session_id, allocation_id, generation=None):
        """Resolve a presentation's exact allocation UUID; never reserve one."""
        self.check(owner_id)
        async with self.tasks.pool.acquire() as connection:
            async with connection.transaction(readonly=True):
                await self._session(connection, owner_id, session_id)
                row = await connection.fetchrow("""select *,deadline>clock_timestamp() as within_deadline
                    from jarvis_sandbox_allocations where owner_id=$1 and id=$2""", owner_id, allocation_id)
                if row is None:
                    raise unavailable()
                target = await self._target(connection, owner_id, session_id, row)
        if generation is not None and target.generation != generation:
            raise unavailable("sandbox_view_replaced", "This sandbox view is no longer current.")
        return target

    async def choices(self, owner_id, session_id):
        """Return explicit choices, not a newest-task heuristic or native liveness."""
        async with self.tasks.pool.acquire() as connection:
            async with connection.transaction(readonly=True):
                await self._session(connection, owner_id, session_id)
                rows = await connection.fetch("""select a.*,a.deadline>clock_timestamp() as within_deadline
                    from jarvis_sandbox_allocations a join jarvis_background_tasks t
                    on t.owner_id=a.owner_id and t.id=a.task_id
                    where a.owner_id=$1 and t.session_id=$2 and a.phase='ready'
                    and a.deadline>clock_timestamp() and a.policy->>'template'='desktop'
                    order by a.id limit 9""", owner_id, session_id)
                targets = []
                for row in rows[:8]:
                    try:
                        targets.append(await self._target(connection, owner_id, session_id, row))
                    except JarvisHTTPError as error:
                        if error.status_code >= 500:
                            raise
                return targets, len(rows) > 8

    async def close(self):
        self.closed = True
