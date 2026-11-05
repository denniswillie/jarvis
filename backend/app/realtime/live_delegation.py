"""Bounded session coordination, reusing durable task/Hermes owners."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from app.tasks.models import TaskError, TaskStatus
from app.attention.quiet_delivery import assert_unsolicited_allowed
from app.tasks.readonly import readonly_source
from app.voice_routes.live_results import LiveDelegationResult


async def current_result(tasks, connection, receipt, route):
    task, detail = await tasks.live_context_in_transaction(connection, receipt.task.owner_id, receipt.task.id)
    source = await readonly_source(tasks, connection, task)
    if (source is None or source.id != receipt.source.id or task.revision != receipt.task.revision
            or source.content["route"] != route.model_dump(mode="json")
            or source.content["delegation_id"] != receipt.delegation_id):
        raise TaskError("live_readonly_request_superseded", "The delegation no longer owns this task revision.")
    if task.status in {TaskStatus.CANCEL_REQUESTED, TaskStatus.CANCELLED, TaskStatus.UNKNOWN}:
        raise TaskError("live_readonly_task_cancelled", "This task cannot publish a result.")
    if task.status not in {TaskStatus.SUCCEEDED, TaskStatus.WAITING, TaskStatus.FAILED, TaskStatus.PAUSED}:
        return None
    result = detail.get("result")
    if not result or not detail.get("manifest_id") or not detail.get("manifest_payload_id"):
        raise TaskError("live_readonly_result_unavailable", "There is no source-backed current worker result.")
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise TaskError("live_readonly_answer_unavailable", "The worker did not supply a readable answer.")
    return LiveDelegationResult(route, receipt.delegation_id, task.id, task.revision,
        task.result_payload_id, answer, receipt.expires_at,
        tuple(sorted({source.payload_id, task.description_payload_id, task.result_payload_id,
                      detail["manifest_payload_id"]})))


@asynccontextmanager
async def result_guard(session, receipt, expected):
    if not session.ready or expected.expires_at <= datetime.now(timezone.utc):
        raise TaskError("live_readonly_route_unavailable", "The route or result expired.")
    async with session.live.scope_guard() as connection:
        # Scope guard already holds the canonical task-owner lock. A readonly
        # caption/delegation is NOT confirmed solicited-reply/command authority.
        await assert_unsolicited_allowed(connection, receipt.task.owner_id, origin="live_readonly_result")
        current = await current_result(session.readonly_admission.tasks, connection, receipt, session.request.route)
        if current != expected:
            raise TaskError("live_readonly_result_changed", "The task result changed before submission.")
        for root in current.roots:
            if await connection.fetchval("select lock_jarvis_copy_source($1,$2)", receipt.task.owner_id, root) != "available":
                raise TaskError("live_readonly_result_withdrawn", "A result source was withdrawn.")
        session._source_context.observe(current.roots)
        async with session._source_context.context_guard(connection=connection):
            if not session.ready or current.expires_at <= datetime.now(timezone.utc):
                raise TaskError("live_readonly_route_unavailable", "The route changed before submission.")
            yield connection
