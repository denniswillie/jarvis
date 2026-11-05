"""Task-scoped native commands; Hermes still owns reasoning and tool selection."""
import asyncio

from app.sandbox.allocations import SandboxAllocationRepository
from app.sandbox.commands import SandboxCommandRepository
from app.sandbox.models import SandboxError
from app.tasks.models import json_object


class SandboxCommandBroker:
    def __init__(self, context, provider, grant_id, *, managed_plugins=None,
                 reusable=False, continuation_id=None):
        self.context, self.provider, self.grant_id = context, provider, grant_id
        self.allocations = SandboxAllocationRepository(context.tasks, claim=context.claim)
        self.reusable, self.continuation_id = reusable, continuation_id
        self.commands = SandboxCommandRepository(context.tasks)
        self.managed_plugins = managed_plugins

    async def _managed(self):
        from app.plugins.managed import ManagedFilesDataRuntime, load_managed_task
        async with self.context.tasks.transaction(self.context.task.owner_id) as connection:
            task, _ = await self.context.tasks.lock_claim(connection, self.context.claim)
            managed = await load_managed_task(connection, task)
            if managed is None:
                return None
            if managed[1] != self.grant_id or not isinstance(self.managed_plugins, ManagedFilesDataRuntime):
                raise SandboxError("managed_submission_required")
            return managed[0]

    async def allocation(self):
        if self.continuation_id is not None:
            # A follow-up may arrive before the periodic idle pass. Use the same
            # fenced cleaner for this exact allocation, never pause active work.
            from app.sandbox.cleanup import SandboxCleaner, SandboxCleanupRepository
            cleaner = SandboxCleaner(SandboxCleanupRepository(self.allocations), self.provider)
            await cleaner.run_once(self.context.task.owner_id, "sandbox-continuation",
                                   allocation_id=self.continuation_id)
        return await self.allocations.reserve(self.context.claim, self.grant_id,
            reusable=self.reusable, continuation_id=self.continuation_id)

    async def _ready(self, row):
        claim, owner = self.context.claim, self.context.task.owner_id
        job = await self._managed()
        bound_create = None
        if job is not None and row["phase"] == "reserved":
            bound_create = getattr(self.provider, "create_bound", None)
            if not callable(bound_create):
                raise SandboxError("managed_template_selection_unavailable")
        if job is not None:
            job = await self.managed_plugins.ensure_submitted(self.context, self.grant_id)
        if row["phase"] == "reserved":
            token = await self.allocations.begin_creation(claim, row["id"])
            try:
                scope, policy = self.allocations.scope(row), self.allocations.policy(row)
                if job is None:
                    handle = await self.provider.create(scope, policy)
                else:
                    handle = await bound_create(scope, policy, template_id=job.bundle.sandbox_template_id)
                row = await self.allocations.record_created(owner, row["id"], token, handle)
            except BaseException:
                await asyncio.shield(self.allocations.creation_uncertain(owner, row["id"], token))
                raise
        elif row["phase"] in {"creating", "uncertain"}:
            ids = await self.provider.discover(self.allocations.scope(row), self.allocations.policy(row))
            if len(ids) != 1:
                raise SandboxError("sandbox_creation_outcome_unresolved")
            handle = await self.provider.info(ids[0], self.allocations.scope(row), self.allocations.policy(row))
            row = await self.allocations.record_created(owner, row["id"], row["submission_token"], handle)
        if row["phase"] == "resuming":
            from datetime import datetime, timezone
            remaining = int((row["deadline"] - datetime.now(timezone.utc)).total_seconds())
            if remaining < 1:
                raise SandboxError("sandbox_continuation_expired")
            # Never fall back to create/prepare on a missing or failed resume.
            handle = await self.provider.resume(self.allocations.handle(row), timeout_seconds=remaining)
            row = await self.allocations.resumed(claim, row["id"], handle)
        if row["phase"] == "preparing":
            if job is not None:
                job.bundle.require_image(self.allocations.handle(row))
            # Trusted preparation is idempotent; it never runs the model command.
            await self.provider.prepare(self.allocations.handle(row))
            row = await self.allocations.ready(claim, row["id"])
        current = await self.allocations.current(claim, row["id"])
        if job is not None and current["phase"] == "ready":
            job.bundle.require_image(self.allocations.handle(current))
        return current

    async def execute(self, request):
        return await self._execute(request)

    async def execute_media(self, plan):
        from app.workspace.media_render import MediaRenderPlan
        if not isinstance(plan, MediaRenderPlan):
            raise SandboxError("managed_media_plan_required")
        return await self._execute(plan.command(), media_plan=plan)

    async def _execute(self, request, *, media_plan=None):
        from app.plugins.media import ManagedMediaJob
        job = await self._managed()
        if isinstance(job, ManagedMediaJob):
            job.bundle.require_command(request, plan=media_plan)
            if media_plan.owner_id != self.context.task.owner_id or media_plan.task_id != self.context.task.id:
                raise SandboxError("managed_media_scope_mismatch")
            if not callable(getattr(self.provider, "materialize_media_render", None)):
                raise SandboxError("managed_media_provider_unavailable")
        elif media_plan is not None:
            raise SandboxError("managed_media_binding_required")
        elif job is not None:
            job.bundle.require_command(request)
        allocation = await self.allocation()
        row = await self.commands.prepare(self.context.claim, allocation["id"], request)
        if row["phase"] == "prepared":
            allocation = await self._ready(allocation)
            if media_plan is not None:
                handle = self.allocations.handle(allocation)
                media_plan.require_allocation(handle)
                await self.provider.materialize_media_render(handle, media_plan)
                # Source/claim validity is checked again by the existing durable
                # submission transition. Materialization is not render success.
            token = await self.commands.begin_submission(self.context.claim, row["id"])
            if token is not None:
                try:
                    await self.provider.start_managed(self.allocations.handle(allocation), request,
                        operation_id=row["id"], request_digest=row["request_digest"])
                except BaseException:
                    await asyncio.shield(self.commands.uncertain(self.context.task.owner_id, row["id"], token))
                    raise
        return await self.observe(row["id"])

    async def observe(self, identity):
        row = await self.commands.current(self.context.claim, identity)
        if row["phase"] not in {"prepared", "completed"}:
            allocation = await self.allocations.current(self.context.claim, row["allocation_id"])
            try:
                receipt = await self.provider.command_receipt(self.allocations.handle(allocation),
                    identity, row["request_digest"])
                if receipt is not None:
                    row = await self.commands.record(self.context.task.owner_id, identity, row["submission_token"], receipt)
                else:
                    await self.commands.uncertain(self.context.task.owner_id, identity, row["submission_token"])
            except SandboxError:
                await self.commands.uncertain(self.context.task.owner_id, identity, row["submission_token"])
                raise
        row = await self.commands.current(self.context.claim, identity)
        if row["phase"] == "completed":
            await self.commands.settle(row)
        value = {"command_id": str(identity), "operation_key": json_object(row["parameters"])["operation_key"],
                 "state": row["phase"], "meaning": "Unknown or running is not failure; do not rerun under a new key."}
        if row["phase"] == "running":
            value["meaning"] = (
                "This command is still running, not the final task result. Continue checking sandbox_status "
                "with this operation_key while task budget remains, then inspect the output and continue the task. "
                "Do not pause the whole task merely because this call returned running. Do not rerun the command "
                "under a new key. If observation cannot finish within the available budget, report that limit honestly.")
        elif row["phase"] == "completed":
            value["meaning"] = (
                "The command has terminated. Inspect exit_code, timed_out, runner_error and output before deciding "
                "what to do next. Completion alone does not prove the requested application outcome; observe it.")
        if row["result"] is not None:
            receipt = json_object(row["result"])
            value.update({key: receipt[key] for key in
                ("started", "exit_code", "timed_out", "output_limited", "runner_error")})
            value.update(stdout=receipt["stdout"][:4000], stderr=receipt["stderr"][:4000],
                output_complete=len(receipt["stdout"])<=4000 and len(receipt["stderr"])<=4000
                    and not receipt["output_limited"] and not receipt.get("receipt_truncated", False)
                    and not receipt.get("output_sanitized", False))
        return {"status": "success", "sandbox_command": value}

    async def status(self, operation_key):
        allocation = await self.allocation()
        if operation_key is None:
            return {"status": "success", **await self.commands.list(self.context.claim, allocation["id"])}
        row = await self.commands.find(self.context.claim, allocation["id"], operation_key)
        return await self.observe(row["id"])
