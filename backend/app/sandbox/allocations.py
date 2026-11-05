"""Task-owned allocation intent and late receipts, with no provider I/O in SQL.

A submitted creation is never retried from this ledger. An empty provider listing
is not proof that an unacknowledged create was rejected. Resource cleanup is
implemented by the separate native cleanup worker, not by a model assertion.
"""
from uuid import uuid4

from app.sandbox.models import SandboxError, SandboxInfo, SandboxPolicy, SandboxScope
from app.tasks.models import ClaimLost, TaskError, TaskStatus, json_object, json_text


class SandboxAllocationRepository:
    def __init__(self, tasks, *, claim=None):
        self.tasks = tasks
        self.claim = claim

    @staticmethod
    def scope(row):
        return SandboxScope(owner_id=row["owner_id"], task_id=row["task_id"],
            task_revision=row["task_revision"], attempt_id=row["attempt_id"], allocation_id=row["id"])

    @classmethod
    def native_scope(cls, row):
        value = row.get("native_scope")
        return SandboxScope.model_validate(json_object(value)) if value else cls.scope(row)

    @staticmethod
    def policy(row):
        policy = SandboxPolicy.model_validate(json_object(row["policy"]))
        if policy.fingerprint != row["policy_digest"]:
            raise SandboxError("sandbox_policy_fingerprint_mismatch")
        return policy

    @classmethod
    def handle(cls, row):
        if row["sandbox_id"] is None:
            raise SandboxError("sandbox_handle_unavailable")
        # This is a recorded handle, not a claim about current native liveness.
        return SandboxInfo(sandbox_id=row["sandbox_id"], scope=cls.scope(row), policy=cls.policy(row),
            native_scope=cls.native_scope(row), state="paused" if row["phase"] == "paused" else "running",
            expires_at=row["provider_expires_at"], template_id=row["template_id"])

    async def _load(self, connection, owner, identity):
        row = await connection.fetchrow("""select *,deadline>clock_timestamp() as within_deadline
            from jarvis_sandbox_allocations where owner_id=$1 and id=$2 for update""", owner, identity)
        if row is None:
            raise TaskError("sandbox_allocation_not_found", "No sandbox allocation exists for this owner.", 404)
        return row

    async def _authority(self, connection, claim, grant_id):
        task, attempt = await self.tasks.lock_claim(connection, claim)
        if await self.tasks.context_problem(connection, task):
            raise ClaimLost()
        attached = await connection.fetchval("""select exists(select 1 from jarvis_task_grants
            where owner_id=$1 and task_id=$2 and grant_id=$3)""", task.owner_id, task.id, grant_id)
        if not attached:
            raise TaskError("sandbox_grant_not_attached", "The sandbox grant is not attached to this task.", 403)
        grant = await self.tasks.validate_grant(connection, task.owner_id, grant_id)
        if grant["capability"] != "sandbox.execute":
            raise TaskError("sandbox_grant_scope", "A current sandbox execution grant is required.", 403)
        policy = SandboxPolicy.model_validate(json_object(grant["scope"]))
        return task, attempt, grant, policy

    async def _current(self, connection, claim, identity):
        # Preserve owner -> task/attempt -> grant -> allocation lock order.
        task, _ = await self.tasks.lock_claim(connection, claim)
        initial = await connection.fetchrow("""select grant_id from jarvis_sandbox_allocations
            where owner_id=$1 and task_id=$2 and id=$3""", task.owner_id, task.id, identity)
        if initial is None:
            raise TaskError("sandbox_allocation_not_found", "No sandbox allocation exists for this task.", 404)
        _, _, _, policy = await self._authority(connection, claim, initial["grant_id"])
        row = await self._load(connection, task.owner_id, identity)
        if (row["attempt_id"] != claim.attempt_id or row["task_revision"] != claim.task_revision
                or row["policy_digest"] != policy.fingerprint or not row["within_deadline"]):
            raise TaskError("sandbox_allocation_not_current", "The sandbox no longer belongs to this active work.", 403)
        return task, row

    async def reserve(self, claim, grant_id, *, reusable=False, continuation_id=None):
        if type(reusable) is not bool:
            raise TypeError("reusable must be a boolean selected by the trusted runtime")
        if continuation_id is not None:
            return await self.attach(claim, grant_id, continuation_id)
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, _, grant, policy = await self._authority(connection, claim, grant_id)
            previous = await connection.fetch("""select * from jarvis_sandbox_allocations
                where owner_id=$1 and (grant_id=$2 or (task_id=$3 and attempt_id=$4)) for update""",
                task.owner_id, grant_id, task.id, claim.attempt_id)
            existing = next((row for row in previous if row["attempt_id"] == claim.attempt_id), None)
            if any(row["task_id"] != task.id for row in previous):
                raise TaskError("sandbox_allocation_conflict", "A sandbox grant cannot move to another task.")
            if existing is not None:
                if (existing["task_id"] != task.id or existing["attempt_id"] != claim.attempt_id
                        or existing["task_revision"] != claim.task_revision or existing["grant_id"] != grant_id
                        or existing["policy_digest"] != policy.fingerprint):
                    raise TaskError("sandbox_allocation_conflict", "This grant or attempt already owns a different allocation.")
                return dict(existing)
            if any(row["phase"] != "cleaned" for row in previous):
                raise TaskError("sandbox_cleanup_required", "Confirm cleanup of the prior allocation before creating a replacement.")
            row = await connection.fetchrow("""insert into jarvis_sandbox_allocations
                (owner_id,task_id,task_revision,attempt_id,grant_id,policy,policy_digest,phase,deadline,reusable)
                values($1,$2,$3,$4,$5,$6::jsonb,$7,'reserved',least($8,clock_timestamp()+$9*interval '1 second'),$10)
                returning *""", task.owner_id, task.id, claim.task_revision, claim.attempt_id, grant_id,
                json_text(policy.model_dump(mode="json")), policy.fingerprint, grant["expires_at"],
                float(min(policy.lifetime_seconds, max(0, task.budget.active_seconds-task.active_seconds))), reusable)
            await self.tasks.emit(connection, task, "sandbox_reserved", payload={"allocation_id":str(row["id"])},
                                  attempt_id=claim.attempt_id)
            return dict(row)

    async def continuation_for_task(self, claim):
        """Resolve only this work or its server-validated predecessor, never a session."""
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, _ = await self.tasks.lock_claim(connection, claim)
            if await self.tasks.context_problem(connection, task):
                raise ClaimLost()
            rows = await connection.fetch("""select id,reusable,phase,attempt_id from jarvis_sandbox_allocations
                where owner_id=$1 and task_id=$2 order by created_at desc""", task.owner_id, task.id)
            if rows:
                # Current-attempt allocations retain legacy behavior. A continuing
                # allocation is selected by stable work identity, not recency alone.
                current = [r for r in rows if r["attempt_id"] == claim.attempt_id]
                if current:
                    return current[0]["id"] if current[0]["reusable"] else None
                retained = [r for r in rows if r["reusable"]]
                if retained:
                    if len(retained) != 1 or retained[0]["phase"] in {"cleaned", "cleanup_requested"}:
                        raise TaskError("sandbox_continuation_unavailable", "The prior workspace is unavailable; prior files are not confirmed retained.")
                    return retained[0]["id"]
                return None
            from app.live_state.admission import task_command_source
            from app.tasks.media_editing import CorrectedMediaInput, parse_media_input
            record = await task_command_source(self.tasks, connection, task)
            value = record.content.get("media_input") if record is not None else None
            if value is None:
                return None
            pin = parse_media_input(value)
            if not isinstance(pin, CorrectedMediaInput):
                return None
            # task_command_source validated the accepted output and its complete
            # source dependency chain; a model-provided allocation ID is never used.
            rows = await connection.fetch("""select id,reusable,phase from jarvis_sandbox_allocations
                where owner_id=$1 and task_id=$2""", task.owner_id, pin.target.task_id)
            if len(rows) != 1 or not rows[0]["reusable"] or rows[0]["phase"] in {"cleaned", "cleanup_requested"}:
                raise TaskError("sandbox_continuation_unavailable", "The selected work's workspace is unavailable; its files are not confirmed retained.")
            return rows[0]["id"]

    async def attach(self, claim, grant_id, identity):
        """Attach an exact trusted continuation reference; never select by user/session.

        Admission supplies the intended allocation ID. Owner, current execution
        grant, policy equality, expiry and quiescence are independently checked.
        Native ownership metadata and the original retention deadline never move.
        """
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, _, grant, policy = await self._authority(connection, claim, grant_id)
            row = await self._load(connection, task.owner_id, identity)
            same = (row["task_id"], row["attempt_id"], row["task_revision"], row["grant_id"]) == (
                task.id, claim.attempt_id, claim.task_revision, grant_id)
            if not row["reusable"] or row["policy_digest"] != policy.fingerprint or not row["within_deadline"]:
                raise TaskError("sandbox_continuation_unavailable", "The prior workspace is unavailable; its files are not confirmed retained.")
            if same and row["phase"] not in {"paused", "pausing", "cleaned", "cleanup_requested"}:
                return dict(row)
            if row["phase"] != "paused":
                raise TaskError("sandbox_continuation_busy", "The prior workspace has not reached a confirmed idle pause.")
            prior = await self.tasks.load(connection, task.owner_id, row["task_id"], lock=True)
            # A cancellation, removed input or expired grant cannot be bypassed by attachment.
            if (prior.status in {TaskStatus.CANCELLED, TaskStatus.CANCEL_REQUESTED}
                    or prior.pending_status is not None or await self.tasks.context_problem(connection, prior)):
                raise TaskError("sandbox_continuation_revoked", "The prior workspace authority was withdrawn.", 403)
            await self.tasks.validate_grant(connection, task.owner_id, row["grant_id"])
            active = await connection.fetchval("""select exists(select 1 from jarvis_task_attempts
                where owner_id=$1 and id=$2 and status=any($3::text[]))""",
                task.owner_id, row["attempt_id"], ["running", "recovering", "outcome_unknown"])
            # Resuming a paused SAME attempt is permitted only when it is the current claim.
            if active and not same:
                raise TaskError("sandbox_continuation_busy", "The prior attempt may still be executing.")
            await connection.execute("""update jarvis_sandbox_allocations set
                task_id=$3,task_revision=$4,attempt_id=$5,grant_id=$6,phase='resuming',
                deadline=least(deadline,$7),updated_at=clock_timestamp()
                where owner_id=$1 and id=$2""", task.owner_id, identity, task.id,
                claim.task_revision, claim.attempt_id, grant_id, grant["expires_at"])
            await self.tasks.emit(connection, task, "sandbox_attached",
                payload={"allocation_id": str(identity)}, attempt_id=claim.attempt_id)
            return dict(await self._load(connection, task.owner_id, identity))

    async def resumed(self, claim, identity, handle):
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            _, row = await self._current(connection, claim, identity)
            if (row["phase"] != "resuming" or handle.sandbox_id != row["sandbox_id"]
                    or handle.provider_scope != self.native_scope(row) or handle.state != "running"
                    or handle.policy != self.policy(row)):
                raise TaskError("sandbox_resume_unconfirmed", "The existing workspace did not resume.")
            await connection.execute("""update jarvis_sandbox_allocations set phase='ready',
                provider_expires_at=$3,updated_at=clock_timestamp() where owner_id=$1 and id=$2""",
                claim.task.owner_id, identity, handle.expires_at)
            return dict(await self._load(connection, claim.task.owner_id, identity))

    async def begin_creation(self, claim, identity):
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, row = await self._current(connection, claim, identity)
            if row["phase"] != "reserved" or row["submission_token"] is not None:
                raise TaskError("sandbox_creation_already_submitted", "Reconcile the existing allocation; do not create another VM.")
            token = uuid4()
            await connection.execute("""update jarvis_sandbox_allocations set phase='creating',submission_token=$3,
                submission_fence=$4,updated_at=clock_timestamp() where owner_id=$1 and id=$2""",
                task.owner_id, identity, token, claim.fence)
            await self.tasks.emit(connection, task, "sandbox_creating", payload={"allocation_id":str(identity)},
                                  attempt_id=claim.attempt_id)
            return token

    async def creation_uncertain(self, owner, identity, token):
        async with self.tasks.transaction(owner) as connection:
            row = await self._load(connection, owner, identity)
            self._submission(row, token)
            # A lost local acknowledgement cannot erase an already recorded VM.
            if row["phase"] == "creating":
                await connection.execute("""update jarvis_sandbox_allocations set phase='uncertain',
                    updated_at=clock_timestamp() where owner_id=$1 and id=$2""", owner, identity)

    @staticmethod
    def _submission(row, token):
        if token is None or row["submission_token"] != token:
            raise TaskError("sandbox_receipt_scope", "No matching sandbox submission exists.", 403)

    async def record_created(self, owner, identity, token, handle):
        if not isinstance(handle, SandboxInfo):
            raise TypeError("A trusted native sandbox receipt is required")
        async with self.tasks.transaction(owner) as connection:
            # Serializing on the owner also orders late receipts with task mutations.
            initial = await connection.fetchrow("select task_id from jarvis_sandbox_allocations where owner_id=$1 and id=$2", owner, identity)
            if initial is None:
                raise TaskError("sandbox_allocation_not_found", "No sandbox allocation exists for this owner.", 404)
            task = await self.tasks.load(connection, owner, initial["task_id"], lock=True)
            invalid_context = await self.tasks.context_problem(connection, task)
            row = await self._load(connection, owner, identity)
            self._submission(row, token)
            if handle.provider_scope != self.native_scope(row) or handle.policy != self.policy(row):
                raise TaskError("sandbox_receipt_scope", "The native receipt belongs to a different allocation.", 403)
            if row["sandbox_id"] is not None and row["sandbox_id"] != handle.sandbox_id:
                raise TaskError("sandbox_receipt_conflict", "The allocation already has a different native sandbox.")
            if row.get("reusable") and (row["phase"] in {"pausing", "paused", "resuming"}
                    or self.scope(row) != self.native_scope(row)):
                # A delayed creation acknowledgement cannot regress a retained
                # or transferred allocation into its original preparation path.
                return dict(row)
            live_attempt = await connection.fetchval("""select lease_until>clock_timestamp() and status='running'
                from jarvis_task_attempts where owner_id=$1 and id=$2""", owner, row["attempt_id"])
            obsolete = (invalid_context or not live_attempt or task.status != TaskStatus.RUNNING
                or task.current_attempt_id != row["attempt_id"] or task.revision != row["task_revision"]
                or not row["within_deadline"] or row["phase"] in {"cleanup_requested", "cleaned"})
            phase = "cleanup_requested" if obsolete else ("ready" if row["phase"] == "ready" else "preparing")
            await connection.execute("""update jarvis_sandbox_allocations set phase=$3,sandbox_id=$4,
                template_id=$5,provider_expires_at=$6,updated_at=clock_timestamp(),
                next_cleanup_at=case when $3='cleanup_requested' then clock_timestamp() else next_cleanup_at end
                where owner_id=$1 and id=$2""",
                owner, identity, phase, handle.sandbox_id, handle.template_id, handle.expires_at)
            return dict(await self._load(connection, owner, identity))

    async def ready(self, claim, identity):
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, row = await self._current(connection, claim, identity)
            if row["phase"] not in {"preparing", "ready"} or row["sandbox_id"] is None:
                raise TaskError("sandbox_not_prepared", "This allocation has not finished its trusted preparation.")
            await connection.execute("""update jarvis_sandbox_allocations set phase='ready',
                updated_at=clock_timestamp() where owner_id=$1 and id=$2""", task.owner_id, identity)
            return dict(await self._load(connection, task.owner_id, identity))

    async def current(self, claim, identity):
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            _, row = await self._current(connection, claim, identity)
            if row["phase"] != "ready":
                raise TaskError("sandbox_not_ready", "The existing sandbox needs preparation or recovery.")
            return dict(row)

    async def request_cleanup(self, owner, identity, *, reason="sandbox_cleanup_requested", claim=None, revoke=False):
        if not isinstance(reason, str) or not 1 <= len(reason) <= 100:
            raise ValueError("Use a bounded cleanup reason code")
        async with self.tasks.transaction(owner) as connection:
            row = await self._load(connection, owner, identity)
            if row.get("reusable") and not revoke:
                expected = claim or self.claim
                if (expected is None or expected.task.owner_id != owner
                        or (row["task_id"], row["attempt_id"], row["task_revision"]) !=
                           (expected.task.id, expected.attempt_id, expected.task_revision)):
                    raise TaskError("sandbox_cleanup_stale", "A previous attempt cannot clean a continuing workspace.", 403)
                current_fence = await connection.fetchval("""select fence from jarvis_background_tasks
                    where owner_id=$1 and id=$2""", owner, expected.task.id)
                if current_fence != expected.fence:
                    raise TaskError("sandbox_cleanup_stale", "A stale worker cannot clean a continuing workspace.", 403)
            # Only an intent that never crossed submission is already known absent.
            phase = "cleaned" if row["submission_token"] is None or row["phase"] == "cleaned" else "cleanup_requested"
            await connection.execute("""update jarvis_sandbox_allocations set phase=$3,reason=$4,
                updated_at=clock_timestamp(),next_cleanup_at=clock_timestamp()
                where owner_id=$1 and id=$2""", owner, identity, phase, reason)
            return dict(await self._load(connection, owner, identity))
