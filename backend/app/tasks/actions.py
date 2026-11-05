"""External effects cross one durable submission boundary exactly once.

No provider API calls occur here. A submission token authorizes recording a late
receipt, not executing a second submission. Native reconciliation is a trusted
adapter operation, never a model assertion that an action probably succeeded.
"""
from __future__ import annotations

from uuid import UUID, uuid4

from app.tasks.models import ClaimLost, TaskClaim, TaskError, digest, json_object
from app.tasks.budgets import TaskBudgetRepository
from app.tasks.repository import TaskRepository


class TaskActionRepository:
    def __init__(self, tasks: TaskRepository):
        self.tasks = tasks

    async def _receipt_payload(self, connection, owner_id, action, content):
        # Parameters already reuse the approved scope. Receipt/proof prose is
        # a new copy, but its arrival must still settle an effect after removal.
        source = action["parameters_payload_id"]
        state = await connection.fetchval("select lock_jarvis_copy_source($1,$2)", owner_id, source)
        if state == "available":
            return await self.tasks.payload(connection, owner_id, content, source_payload_ids=(source,))
        # Do not insert private text even transiently and then redact it. The
        # non-content outcome/fingerprint live on the existing action record.
        return await connection.fetchval("""insert into jarvis_context_payloads
            (owner_id,availability,content,removed_at) values($1,'unavailable',null,clock_timestamp()) returning id""", owner_id)

    async def prepare(self, claim: TaskClaim, operation_key: str, grant_id: UUID, capability: str, parameters: dict):
        if not 1 <= len(operation_key.strip()) <= 200:
            raise ValueError("invalid logical operation identity")
        parameters_digest = digest(parameters)
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, _ = await self.tasks.lock_claim(connection, claim)
            existing = await connection.fetchrow("select * from jarvis_task_actions where owner_id=$1 and task_id=$2 and operation_key=$3 for update", task.owner_id, task.id, operation_key)
            if existing:
                grant_capability = await connection.fetchval("select capability from jarvis_operation_grants where owner_id=$1 and id=$2", task.owner_id, grant_id)
                if existing["parameters_digest"] != parameters_digest or existing["grant_id"] != grant_id or grant_capability != capability:
                    raise TaskError("task_action_conflict", "An existing logical operation has different authorized parameters.")
                if existing["task_revision"] != task.revision and existing["status"] in {"prepared", "authorized", "revoked"}:
                    resumed = await connection.fetchval("""select exists(select 1 from jarvis_task_requests where owner_id=$1
                        and task_id=$2 and (operation='resume' or (operation='update' and $4))
                        and accepted_revision=$3)""", task.owner_id, task.id, task.revision,
                        operation_key == "managed-files-data:" + str(task.id))
                    if not resumed or existing["submission_token"] is not None or await self.tasks.context_problem(connection, task):
                        raise TaskError("task_action_reauthorization_required", "The old action requires a current explicit resume and unchanged authority.", 403)
                    available = await connection.fetchval("select availability='available' from jarvis_context_payloads where owner_id=$1 and id=$2 for share", task.owner_id, existing["parameters_payload_id"])
                    if not available:
                        raise TaskError("task_action_parameters_removed", "Removed parameters cannot be restored by a retry.", 403)
                    existing = await connection.fetchrow("""update jarvis_task_actions set task_revision=$3,status='prepared',updated_at=clock_timestamp()
                        where owner_id=$1 and id=$2 returning *""", task.owner_id, existing["id"], task.revision)
                    await self.tasks.emit(connection, task, "action_revalidated_after_resume", payload={"action_id": str(existing["id"])}, attempt_id=claim.attempt_id)
                return dict(existing)
            if await connection.fetchval("select exists(select 1 from jarvis_task_actions where owner_id=$1 and grant_id=$2)", task.owner_id, grant_id):
                raise TaskError("task_grant_already_bound", "This single-operation grant is already bound to another logical action.", 403)
            if await self.tasks.context_problem(connection, task):
                raise ClaimLost()
            grant = await self.tasks.validate_grant(connection, task.owner_id, grant_id)
            if grant["capability"] != capability or digest(json_object(grant["scope"])) != parameters_digest:
                raise TaskError("task_action_scope_mismatch", "The action does not match the exact authorized scope.", 403)
            payload_id = grant["scope_payload_id"]
            action = await connection.fetchrow("""insert into jarvis_task_actions(owner_id,task_id,task_revision,operation_key,grant_id,
                scope_revision,parameters_payload_id,parameters_digest,status) values($1,$2,$3,$4,$5,$6,$7,$8,'prepared') returning *""",
                task.owner_id, task.id, task.revision, operation_key, grant_id, grant["scope_revision"], payload_id, parameters_digest)
            await self.tasks.emit(connection, task, "action_prepared", payload={"action_id": str(action["id"]), "capability": capability}, attempt_id=claim.attempt_id)
            return dict(action)

    async def _locked_action(self, connection, claim, action_id):
        task, _ = await self.tasks.lock_claim(connection, claim)
        if await self.tasks.context_problem(connection, task):
            raise ClaimLost()
        action = await connection.fetchrow("select * from jarvis_task_actions where owner_id=$1 and task_id=$2 and id=$3 for update", task.owner_id, task.id, action_id)
        if not action or action["task_revision"] != task.revision:
            raise TaskError("task_action_not_current", "The action does not belong to this current task revision.")
        await self.tasks.validate_grant(connection, task.owner_id, action["grant_id"])
        available = await connection.fetchval("select availability='available' from jarvis_context_payloads where owner_id=$1 and id=$2 for share", task.owner_id, action["parameters_payload_id"])
        if not available:
            raise TaskError("task_action_parameters_removed", "Action parameters are no longer available.", 403)
        return task, action

    async def authorize(self, claim: TaskClaim, action_id: UUID):
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, action = await self._locked_action(connection, claim, action_id)
            if action["status"] == "authorized":
                return
            if action["status"] != "prepared":
                raise TaskError("task_action_state_conflict", "Only a prepared action can be authorized.")
            await connection.execute("update jarvis_task_actions set status='authorized',updated_at=clock_timestamp() where owner_id=$1 and id=$2", task.owner_id, action_id)
            await self.tasks.emit(connection, task, "action_authorized", payload={"action_id": str(action_id)}, attempt_id=claim.attempt_id)

    async def revoke_unsubmitted(self, claim: TaskClaim, action_id: UUID):
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, action = await self._locked_action(connection, claim, action_id)
            if action["status"] not in {"prepared", "authorized", "revoked"}:
                raise TaskError("task_action_already_submitted", "A submitted effect cannot be undone by revoking its record.")
            if action["status"] != "revoked":
                await connection.execute("update jarvis_task_actions set status='revoked',updated_at=clock_timestamp() where owner_id=$1 and id=$2", task.owner_id, action_id)
                await self.tasks.emit(connection, task, "action_revoked", payload={"action_id": str(action_id)}, attempt_id=claim.attempt_id)

    async def begin_submission(self, claim: TaskClaim, action_id: UUID, *, managed_plugins=None, managed_pin=None) -> UUID:
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, action = await self._locked_action(connection, claim, action_id)
            if action["status"] != "authorized":
                raise TaskError("task_action_already_submitted", "This action is not an unsubmitted authorized operation.")
            from app.plugins.managed import ManagedFilesDataRuntime, load_managed_task, submission_receipt
            managed = await load_managed_task(connection, task)
            submitting_job = managed is not None and action["grant_id"] == managed[1]
            if managed is not None:
                if not isinstance(managed_plugins, ManagedFilesDataRuntime):
                    raise TaskError("managed_submission_required", "Managed work must use its bound submission route.", 403)
                if submitting_job:
                    if managed_pin is not None:
                        raise TaskError("managed_new_work_requires_admission", "A continuation pin cannot submit a new managed job.", 403)
                    # Same owner lock and transaction as the token, before provider
                    # creation/preparation. Selection/prepared/authorized is no pin.
                    await managed_plugins.require_new_work(connection, task, managed[0])
                else:
                    from app.plugins.authority import SubmittedManagedJob
                    if (not isinstance(managed_pin, SubmittedManagedJob)
                            or action["operation_key"] != "tool-grant:" + action["grant_id"].hex):
                        raise TaskError("managed_submission_required", "Publication requires the exact submitted-job pin and operation.", 403)
                    grant = await self.tasks.validate_grant(connection, task.owner_id, action["grant_id"])
                    parameters = json_object(grant["scope"])
                    if action["parameters_digest"] != digest(parameters):
                        raise TaskError("task_action_scope_mismatch", "The action no longer matches its exact grant.", 403)
                    await managed_plugins.require_publication(connection, self.tasks, task,
                        action["grant_id"], parameters, expected=managed_pin)
            elif managed_pin is not None:
                raise TaskError("managed_submission_mismatch", "A managed pin cannot authorize another task.", 403)
            _, attempt = await self.tasks.lock_claim(connection, claim)
            reservation = await TaskBudgetRepository(self.tasks).reserve_locked(connection, task, attempt, "action:" + str(action_id), tool_steps=1)
            if reservation["status"] != "reserved":
                raise TaskError("task_action_budget_already_submitted", "This action's execution budget was already submitted.")
            await connection.execute("update jarvis_task_reservations set status='submitted' where owner_id=$1 and id=$2", task.owner_id, reservation["id"])
            token = uuid4()
            await connection.execute("""update jarvis_task_actions set status='submitting',submission_token=$3,updated_at=clock_timestamp()
                where owner_id=$1 and id=$2""", task.owner_id, action_id, token)
            await self.tasks.emit(connection, task, "action_submitting", payload={"action_id": str(action_id)}, attempt_id=claim.attempt_id)
            if submitting_job:
                # Local submission acknowledgement, not a fabricated provider
                # effect. Commit it with the token so restart is unambiguous.
                await self.record_receipt_in_transaction(connection, task.owner_id, task.id, action_id, token,
                    "accepted", submission_receipt(task, managed[0], token))
            return token

    async def record_receipt(self, owner_id: str, task_id: UUID, action_id: UUID, token: UUID,
                             status: str, receipt: dict):
        async with self.tasks.transaction(owner_id) as connection:
            await self.record_receipt_in_transaction(connection, owner_id, task_id, action_id, token, status, receipt)

    async def record_receipt_in_transaction(self, connection, owner_id, task_id, action_id, token, status, receipt):
        """Trusted local effects can commit publication and acknowledgement together."""
        if not connection.is_in_transaction():
            raise RuntimeError("Action receipts require a transaction")
        if status not in {"accepted", "rejected", "outcome_unknown"}:
            raise ValueError("invalid external submission outcome")
        task = await self.tasks.load(connection, owner_id, task_id, lock=True)
        action = await connection.fetchrow("""select a.*,p.content as receipt from jarvis_task_actions a
            left join jarvis_context_payloads p on p.owner_id=a.owner_id and p.id=a.receipt_payload_id
            where a.owner_id=$1 and a.task_id=$2 and a.id=$3 for update of a""", owner_id, task_id, action_id)
        if not action or action["submission_token"] != token:
            raise TaskError("task_action_receipt_scope", "No matching submission token exists.", 403)
        if action["status"] in {"accepted", "rejected", "reconciled"}:
            if status == "outcome_unknown":
                # A lost local acknowledgement cannot erase already committed proof.
                return
            fingerprint = action["receipt_digest"]
            if fingerprint is None and action["receipt"] is not None:
                fingerprint = digest(json_object(action["receipt"]))
            if action["status"] == status and fingerprint == digest(receipt):
                if action["receipt_digest"] is None:
                    # A matching retained legacy body is evidence; an already
                    # removed legacy receipt cannot supply a made-up digest.
                    await connection.execute("update jarvis_task_actions set receipt_digest=$3 where owner_id=$1 and id=$2",
                                             owner_id, action_id, fingerprint)
                return
            raise TaskError("task_action_receipt_conflict", "The action already has a different durable outcome.")
        if action["status"] not in {"submitting", "outcome_unknown"}:
            raise TaskError("task_action_not_submitted", "The action never crossed the submission boundary.")
        payload_id = await self._receipt_payload(connection, owner_id, action, receipt)
        await connection.execute("update jarvis_task_actions set status=$3,receipt_payload_id=$4,receipt_digest=$5,updated_at=clock_timestamp() where owner_id=$1 and id=$2",
                                 owner_id, action_id, status, payload_id, digest(receipt))
        if status != "outcome_unknown":
            await self._settle_action_budget(connection, owner_id, task_id, action_id)
        await self.tasks.emit(connection, task, "action_" + status, payload_id=payload_id, revision=action["task_revision"])
        await self.tasks.resolve_action_barrier(connection, task)

    async def reconcile(self, owner_id: str, task_id: UUID, action_id: UUID, *, outcome: str, proof: dict):
        if outcome not in {"accepted", "rejected", "not_submitted"} or not proof:
            raise ValueError("native reconciliation requires an explicit outcome and proof")
        async with self.tasks.transaction(owner_id) as connection:
            task = await self.tasks.load(connection, owner_id, task_id, lock=True)
            action = await connection.fetchrow("select * from jarvis_task_actions where owner_id=$1 and task_id=$2 and id=$3 for update", owner_id, task_id, action_id)
            if not action or action["status"] != "outcome_unknown":
                raise TaskError("task_action_reconciliation_conflict", "Only an uncertain action may be reconciled.")
            payload_id = await self._receipt_payload(connection, owner_id, action, proof)
            await connection.execute("""update jarvis_task_actions set status='reconciled',reconciled_outcome=$3,
                receipt_payload_id=$4,receipt_digest=$5,updated_at=clock_timestamp() where owner_id=$1 and id=$2""",
                owner_id, action_id, outcome, payload_id, digest(proof))
            await self._settle_action_budget(connection, owner_id, task_id, action_id)
            await self.tasks.emit(connection, task, "action_reconciled", payload_id=payload_id, revision=action["task_revision"])
            await self.tasks.resolve_action_barrier(connection, task)

    async def _settle_action_budget(self, connection, owner_id, task_id, action_id):
        await connection.execute("""update jarvis_task_reservations set status='settled',actual_input_tokens=0,
            actual_output_tokens=0,actual_tool_steps=1 where owner_id=$1 and task_id=$2 and operation_key=$3
            and status in ('submitted','unknown')""", owner_id, task_id, "action:" + str(action_id))

    async def get(self, owner_id: str, task_id: UUID, action_id: UUID):
        async with self.tasks.pool.acquire() as connection:
            await self.tasks.load(connection, owner_id, task_id)
            row = await connection.fetchrow("select * from jarvis_task_actions where owner_id=$1 and task_id=$2 and id=$3", owner_id, task_id, action_id)
            if not row:
                raise TaskError("task_action_not_found", "No action was found for this task.", 404)
            return dict(row)
