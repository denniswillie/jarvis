"""Durable command submission and removable receipts, independent of a tool-call ID."""
from uuid import uuid4

from app.sandbox.allocations import SandboxAllocationRepository
from app.sandbox.models import CommandReceipt, CommandRequest
from app.tasks.budgets import TaskBudgetRepository
from app.tasks.models import TaskError, digest, json_object


class SandboxCommandRepository:
    def __init__(self, tasks):
        self.tasks = tasks
        self.allocations = SandboxAllocationRepository(tasks)
        self.budgets = TaskBudgetRepository(tasks)

    async def _load(self, connection, owner, identity):
        row = await connection.fetchrow("""select c.*,p.content as parameters,p.availability,
            r.content as result from jarvis_sandbox_commands c
            join jarvis_context_payloads p on p.owner_id=c.owner_id and p.id=c.parameters_payload_id
            left join jarvis_context_payloads r on r.owner_id=c.owner_id and r.id=c.result_payload_id
            where c.owner_id=$1 and c.id=$2 for update of c""", owner, identity)
        if row is None:
            raise TaskError("sandbox_command_not_found", "No command exists for this owner.", 404)
        return dict(row)

    @staticmethod
    def operation_digest(allocation, task_id, operation_key):
        # Distinct turns may independently choose the same short operation key.
        # Retrying a logical command within its task still uses the original key.
        value = {"key": operation_key}
        if allocation.get("reusable"):
            value["task_id"] = str(task_id)
        return digest(value)

    async def prepare(self, claim, allocation_id, request):
        if not isinstance(request, CommandRequest):
            raise TypeError("A validated command request is required")
        parameters = request.model_dump(mode="json")
        fingerprint = digest(parameters)
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, allocation = await self.allocations._current(connection, claim, allocation_id)
            operation_digest = self.operation_digest(allocation, task.id, request.operation_key)
            if allocation["phase"] in {"cleaned", "cleanup_requested"}:
                raise TaskError("sandbox_not_current", "This sandbox is being removed.")
            existing = await connection.fetchval("""select id from jarvis_sandbox_commands
                where owner_id=$1 and allocation_id=$2 and operation_digest=$3""",
                task.owner_id, allocation_id, operation_digest)
            if existing:
                row = await self._load(connection, task.owner_id, existing)
                if row["request_digest"] != fingerprint:
                    raise TaskError("sandbox_command_conflict", "Reuse a command key only with identical parameters.")
                if row["availability"] != "available":
                    raise TaskError("sandbox_command_source_removed", "The original command source is unavailable.")
                return row
            manifest = await connection.fetchval("""select m.payload_id from jarvis_task_attempts a
                join jarvis_task_context_manifests m on m.owner_id=a.owner_id and m.id=a.context_manifest_id
                where a.owner_id=$1 and a.id=$2 and m.status='active'""", task.owner_id, claim.attempt_id)
            if manifest is None:
                raise TaskError("context_manifest_required", "Attach current task context before executing commands.")
            grant = await self.tasks.validate_grant(connection, task.owner_id, allocation["grant_id"])
            attempt = await connection.fetchrow("select * from jarvis_task_attempts where owner_id=$1 and id=$2",
                                                task.owner_id, claim.attempt_id)
            identity = uuid4()
            reservation = await self.budgets.reserve_locked(connection, task, attempt,
                "sandbox-command:"+identity.hex, tool_steps=1)
            payload = await self.tasks.payload(connection, task.owner_id, parameters,
                source_payload_ids=(task.description_payload_id, grant["scope_payload_id"], manifest))
            await connection.execute("""insert into jarvis_sandbox_commands
                (id,owner_id,task_id,task_revision,attempt_id,allocation_id,operation_digest,request_digest,
                 parameters_payload_id,reservation_id,phase)
                values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,'prepared')""",
                identity, task.owner_id, task.id, claim.task_revision, claim.attempt_id, allocation_id,
                operation_digest, fingerprint, payload, reservation["id"])
            return await self._load(connection, task.owner_id, identity)

    async def begin_submission(self, claim, identity):
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            initial = await self._load(connection, claim.task.owner_id, identity)
            task, allocation = await self.allocations._current(connection, claim, initial["allocation_id"])
            row = await self._load(connection, task.owner_id, identity)
            if row["task_id"] != task.id:
                raise TaskError("sandbox_command_not_current", "This command belongs to a previous task.", 403)
            if allocation["phase"] != "ready" or row["availability"] != "available":
                raise TaskError("sandbox_not_ready", "Current prepared sandbox and source are required.")
            if row["phase"] != "prepared":
                return None
            # Commands reserve no model tokens. Their one tool step is submitted
            # in the same transaction as the irreversible native-call boundary.
            submitted = await connection.fetchval("""update jarvis_task_reservations
                set status='submitted',accounting_day=(clock_timestamp() at time zone 'UTC')::date
                where owner_id=$1 and id=$2 and attempt_id=$3 and status='reserved' returning id""",
                task.owner_id, row["reservation_id"], claim.attempt_id)
            if submitted is None:
                raise TaskError("sandbox_budget_not_reserved", "The original command budget is unavailable.")
            token = uuid4()
            await connection.execute("""update jarvis_sandbox_commands set phase='submitting',
                submission_token=$3,updated_at=clock_timestamp() where owner_id=$1 and id=$2""",
                task.owner_id, identity, token)
            await self.tasks.emit(connection, task, "sandbox_command_submitting",
                payload={"command_id": str(identity)}, attempt_id=claim.attempt_id)
            return token

    async def uncertain(self, owner, identity, token):
        async with self.tasks.transaction(owner) as connection:
            row = await self._load(connection, owner, identity)
            self._submission(row, token)
            if row["phase"] in {"submitting", "running"}:
                await connection.execute("""update jarvis_sandbox_commands set phase='unknown',
                    updated_at=clock_timestamp() where owner_id=$1 and id=$2""", owner, identity)

    @staticmethod
    def _submission(row, token):
        if token is None or row["submission_token"] != token:
            raise TaskError("sandbox_command_receipt_scope", "No matching command submission exists.", 403)

    async def record(self, owner, identity, token, receipt):
        if not isinstance(receipt, CommandReceipt):
            raise TypeError("A trusted native command receipt is required")
        async with self.tasks.transaction(owner) as connection:
            row = await self._load(connection, owner, identity)
            self._submission(row, token)
            if (receipt.operation_id != identity or receipt.allocation_id != row["allocation_id"]
                    or receipt.request_digest != row["request_digest"]
                    or row["pid"] is not None and row["pid"] != receipt.pid):
                raise TaskError("sandbox_command_receipt_scope", "The receipt belongs to another command.", 403)
            content = receipt.model_dump(mode="json")
            fingerprint = receipt.fingerprint
            if row["phase"] == "completed":
                if receipt.phase == "completed" and row["result_digest"] != fingerprint:
                    raise TaskError("sandbox_command_receipt_conflict", "The final receipt cannot be replaced.")
            elif receipt.phase == "completed":
                payload = None
                state = await connection.fetchval("""select availability from jarvis_context_payloads
                    where owner_id=$1 and id=$2 for share""", owner, row["parameters_payload_id"])
                if state == "available":
                    # A provider's byte limit is not the SQL JSON limit:
                    # Unicode surrogate escaping can expand twelvefold. Preserve
                    # a bounded, explicitly incomplete copy and the full receipt
                    # digest; the native original remains until VM cleanup.
                    retained = content | {"stdout": receipt.stdout[:8000].replace("\x00", "\ufffd"),
                        "stderr": receipt.stderr[:8000].replace("\x00", "\ufffd"),
                        "receipt_truncated": len(receipt.stdout)>8000 or len(receipt.stderr)>8000,
                        "output_sanitized": "\x00" in receipt.stdout or "\x00" in receipt.stderr}
                    payload = await self.tasks.payload(connection, owner, retained,
                        source_payload_ids=(row["parameters_payload_id"],))
                await connection.execute("""update jarvis_sandbox_commands set phase='completed',pid=$3,
                    result_digest=$4,result_payload_id=$5,updated_at=clock_timestamp()
                    where owner_id=$1 and id=$2""", owner, identity, receipt.pid, fingerprint, payload)
            else:
                await connection.execute("""update jarvis_sandbox_commands set phase='running',pid=$3,
                    updated_at=clock_timestamp() where owner_id=$1 and id=$2""", owner, identity, receipt.pid)
            result = await self._load(connection, owner, identity)
        # Late usage may settle without restoring execution or copying forgotten text.
        if result["phase"] == "completed":
            await self.settle(result)
        return result

    async def settle(self, row):
        await self.budgets.settle(row["owner_id"], row["task_id"], row["attempt_id"], row["reservation_id"],
                                  input_tokens=0, output_tokens=0, tool_steps=1)

    async def current(self, claim, identity):
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            row = await self._load(connection, claim.task.owner_id, identity)
            task, _ = await self.allocations._current(connection, claim, row["allocation_id"])
            if row["task_id"] != task.id:
                raise TaskError("sandbox_command_not_current", "This command belongs to a previous task.", 403)
            if row["availability"] != "available":
                raise TaskError("sandbox_command_source_removed", "The original command source is unavailable.")
            return row

    async def find(self, claim, allocation_id, operation_key):
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, allocation = await self.allocations._current(connection, claim, allocation_id)
            identity = await connection.fetchval("""select id from jarvis_sandbox_commands
                where owner_id=$1 and allocation_id=$2 and operation_digest=$3""",
                task.owner_id, allocation_id, self.operation_digest(allocation, task.id, operation_key))
            if identity is None:
                raise TaskError("sandbox_command_not_found", "No command with this key exists.", 404)
            return await self._load(connection, task.owner_id, identity)

    async def list(self, claim, allocation_id):
        async with self.tasks.transaction(claim.task.owner_id) as connection:
            task, allocation = await self.allocations._current(connection, claim, allocation_id)
            rows = await connection.fetch("""select c.id,c.phase,p.content from jarvis_sandbox_commands c
                join jarvis_context_payloads p on p.owner_id=c.owner_id and p.id=c.parameters_payload_id
                where c.owner_id=$1 and c.allocation_id=$2 and p.availability='available'
                and (not $3::boolean or c.task_id=$4)
                order by c.created_at desc,c.id limit 21""", task.owner_id, allocation_id,
                bool(allocation.get("reusable")), task.id)
            return {"commands": [{"command_id": str(row["id"]), "state": row["phase"],
                "operation_key": json_object(row["content"])["operation_key"]} for row in rows[:20]],
                "has_older_commands": len(rows)>20}
