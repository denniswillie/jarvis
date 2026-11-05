"""Observation admission to the existing task owner, not action authorization."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID, NAMESPACE_URL, uuid5

from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError, TaskStatus, digest, json_text

CAPABILITY = "live.readonly.v1"
FORMAT = "jarvis.live_readonly.v1"
FRAGMENT_FORMAT = "jarvis.live_provisional_fragment.v1"
PREFIX = "live-readonly:"


def now():
    return datetime.now(timezone.utc)


def source_ref(record):
    return {"id": str(record.id), "revision": record.revision,
            "payload_id": str(record.payload_id), "digest": record.content_digest}


def request_identity(route, delegation_id):
    return uuid5(NAMESPACE_URL, json_text({"format": FORMAT,
        "route": route.model_dump(mode="json"), "delegation_id": delegation_id}))


async def readonly_source(tasks, connection, task, *, lock=True):
    """None is ordinary admission; invalid read-only evidence raises.

    Never use this validator in place of _source_available for action grants.
    Existing worker/model/tool boundaries call it through context_problem.
    """
    sources = ContextRepository(tasks)
    record = await sources._record(connection, task.owner_id, task.request_event_id)
    if not record.source_namespace.startswith(PREFIX):
        return None
    body = record.content or {}
    if (record.kind != "task_update" or record.trust != "derived"
            or record.finality != "final" or not record.is_current
            or record.availability != "available" or record.revision != task.request_revision
            or record.session_id != task.session_id or body.get("format") != FORMAT
            or body.get("profile") != CAPABILITY or body.get("action_authority") is not False
            or task.backend_kind != "frontier"):
        raise TaskError("readonly_admission_unavailable", "The original read-only admission is unavailable.")
    try:
        expiry = datetime.fromisoformat(body["expires_at"])
        route, refs = body["route"], body["speech_sources"]
        if (expiry.tzinfo is None or expiry <= now() or not 1 <= len(refs) <= 64
                or route["scope"]["owner_id"] != task.owner_id
                or route["scope"]["session_id"] != str(task.session_id)
                or route["scope"].get("area_id") != (str(task.knowledge_area_id) if task.knowledge_area_id else None)):
            raise ValueError()
        identities = [UUID(ref["id"]) for ref in refs]
    except (KeyError, TypeError, ValueError):
        raise TaskError("readonly_admission_expired_or_invalid", "This observation assignment is not current.") from None
    originals = await sources._records(connection, task.owner_id, identities)
    user_seen = False
    for reference in refs:
        original = originals[UUID(reference["id"])]
        content = original.content or {}
        if (source_ref(original) != reference or not original.is_current
                or original.availability != "available" or original.session_id != task.session_id
                or original.kind != "observation" or original.trust != "derived"
                or content.get("format") != FRAGMENT_FORMAT or content.get("route") != route
                or content.get("audio_session_id") != body.get("audio_session_id")
                or content.get("request_boundary") != "unavailable"
                or content.get("speech_finality") != "provisional"):
            raise TaskError("readonly_observation_superseded", "The original provisional evidence changed.")
        user_seen |= content.get("speaker") == "user"
    if not user_seen:
        raise TaskError("readonly_user_evidence_missing", "No user speech was observed for this delegation.")
    if await connection.fetchval("select exists(select 1 from jarvis_task_grants where owner_id=$1 and task_id=$2)",
                                 task.owner_id, task.id):
        raise TaskError("readonly_task_has_grants", "Read-only work cannot acquire action grants.")
    if lock:
        for identity in sorted({record.payload_id, task.description_payload_id,
                                *(original.payload_id for original in originals.values())}):
            if await connection.fetchval("select lock_jarvis_copy_source($1,$2)", task.owner_id, identity) != "available":
                raise TaskError("readonly_source_withdrawn", "The delegated evidence was withdrawn.")
    return record


@dataclass(frozen=True)
class ReadonlyAdmissionReceipt:
    task: object
    source: object
    delegation_id: str
    expires_at: datetime


class ReadonlyTaskAdmission:
    def __init__(self, service):
        self.service, self.tasks = service, service.repository
        self.sources = ContextRepository(self.tasks)

    async def admit(self, connection, *, route, audio_session_id, delegation, observations,
                    visual_context, visual_roots=(), expires_at=None):
        """Called under the actual live-context guard; no generated user request."""
        if not connection.is_in_transaction():
            raise RuntimeError("Read-only admission requires an owner transaction")
        owner, session = route.scope.owner_id, route.scope.session_id
        await self.tasks.lock_owner(connection, owner)
        if (not 1 <= len(observations) <= 64 or not any((r.content or {}).get("speaker") == "user" for r in observations)
                or any(r.owner_id != owner or r.session_id != session for r in observations)):
            raise TaskError("readonly_speech_unavailable", "No exact scoped provisional speech is available.")
        key = request_identity(route, delegation.delegation_id)
        parameters = {"profile": CAPABILITY, "route": route.model_dump(mode="json"), "audio_session_id": audio_session_id,
            "delegation_id": delegation.delegation_id, "provider_event_id": delegation.event_id,
            "offset_ms": delegation.offset_ms, "speech_sources": [source_ref(r) for r in observations]}
        previous = await self.tasks._replay(connection, owner, key, "create", parameters)
        if previous is not None:
            source = await readonly_source(self.tasks, connection, previous)
            return ReadonlyAdmissionReceipt(previous, source, delegation.delegation_id,
                                             datetime.fromisoformat(source.content["expires_at"]))
        provider = self.service.providers.get("frontier")
        if provider is None or self.service.registry is None:
            raise TaskError("readonly_worker_unavailable", "No qualified read-only worker is available.", 503)
        await self.service.registry.require(owner, "frontier", provider.model, connection=connection,
                                            required_capabilities=(CAPABILITY,))
        await self.tasks._queue_capacity(connection, owner)
        area = route.scope.area_id
        if area is not None and not await connection.fetchval(
                "select true from jarvis_knowledge_areas where owner_id=$1 and id=$2 and status='active' for share", owner, area):
            raise TaskError("task_area_not_available", "The selected area is not active.")
        expiry = min(expires_at or now() + timedelta(minutes=3), now() + timedelta(minutes=3))
        if expiry <= now():
            raise TaskError("readonly_context_expired", "The selected context expired before admission.")
        roots = tuple(sorted({*(r.payload_id for r in observations), *visual_roots}))
        if len(roots) > 128:
            raise TaskError("readonly_source_budget", "The exact source set exceeds the admission bound.")
        content = parameters | {"format": FORMAT, "action_authority": False,
            "expires_at": expiry.isoformat(), "visual_context": visual_context,
            "gaps": ["Speech is provisional; no finalized user turn or action permission is inferred.",
                     "Provider offsets are approximate, not speech boundaries.",
                     "The selected fragment window may omit earlier speech; use the archive, not guesses."],
            "allowed": ["source_bound_reasoning", "authorized_visual_context", "memory_reads", "task_status"]}
        source = await self.sources.append_in_transaction(connection, owner, ContextIngress(
            session_id=session, source_namespace=PREFIX + str(route.route_id),
            source_event_id=delegation.delegation_id, receipt_id=digest(parameters), expected_revision=0,
            kind="task_update", trust="derived", source_time=now(), payload=content, source_payload_ids=roots))
        description = await self.tasks.payload(connection, owner, {
            "title": "Read-only Live question", "mode": "consult",
            "requested_outcome": "Answer the read-only question indicated by the captured delegation. "
                "Read its exact source and provisional speech observations. Preserve uncertainty. "
                "Use authorized visual evidence, memory retrieval and task status only. "
                "Do not perform or claim writes, reminders, terminal/browser work or external actions. "
                "If the request needs those actions, explain that limitation. Keep the answer concise.",
            "admission_profile": CAPABILITY}, source_payload_ids=(source.payload_id,))
        limits = self.tasks.limits
        budget = limits.model_copy(update={"active_seconds": min(120, limits.active_seconds),
            "tool_steps": min(12, limits.tool_steps), "max_attempts": 1,
            "request_output_tokens": min(2048, limits.request_output_tokens),
            "task_output_tokens": min(4096, limits.task_output_tokens)})
        task_id = await connection.fetchval("""insert into jarvis_background_tasks(owner_id,session_id,knowledge_area_id,
            backend_kind,model,description_payload_id,request_event_id,request_revision,context_watermark,budget)
            values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb) returning id""", owner, session, area,
            "frontier", provider.model, description, source.id, source.revision, source.sequence, budget.model_dump_json())
        task = await self.tasks.load(connection, owner, task_id)
        await readonly_source(self.tasks, connection, task)
        await self.tasks.emit(connection, task, "accepted", payload={"admission_profile": CAPABILITY,
            "source_event_id": str(source.id), "action_authority": False})
        await self.tasks._save_request(connection, task, key, "create", parameters)
        return ReadonlyAdmissionReceipt(task, source, delegation.delegation_id, expiry)

    async def cancel(self, receipt, *, reason):
        """Revoke this exact admission, including a completed-but-unsent result.

        Completion history is not rewritten to pretend computation never
        happened. A derived admission tombstone fences its future use; active
        execution additionally uses the existing cancellation/fence machinery.
        """
        if reason not in {"route_closed", "source_withdrawn", "superseded", "cancelled"}:
            raise ValueError("Unknown read-only revocation reason")
        async with self.tasks.transaction(receipt.task.owner_id) as connection:
            task = await self.tasks.load(connection, receipt.task.owner_id, receipt.task.id, lock=True)
            if task.request_event_id != receipt.source.id or task.revision != receipt.task.revision:
                return False
            source = await self.sources._record(connection, task.owner_id, receipt.source.id)
            revoked = False
            if source.is_current and source.availability == "available":
                await self.sources.append_in_transaction(connection, task.owner_id, ContextIngress(
                    session_id=source.session_id, source_namespace=source.source_namespace,
                    source_event_id=source.source_event_id, receipt_id="revoke:" + str(task.revision) + ":" + reason,
                    expected_revision=source.revision, kind="task_update", trust="derived",
                    source_time=now(), payload={"format": FORMAT, "profile": CAPABILITY,
                        "admission_revoked": True, "reason": reason, "action_authority": False,
                        "delegation_id": receipt.delegation_id, "task_id": str(task.id),
                        "task_revision": task.revision, "physical_completion": False}))
                revoked = True
            if task.status not in {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.PAUSED, TaskStatus.WAITING}:
                return revoked
            target = TaskStatus.CANCEL_REQUESTED if task.current_attempt_id else TaskStatus.CANCELLED
            await self.tasks.transition(connection, task, target, reason="readonly_" + reason)
            await connection.execute("""update jarvis_background_tasks set revision=revision+1,fence=fence+1,
                pending_status=$3,result_payload_id=null where owner_id=$1 and id=$2""",
                task.owner_id, task.id, "cancelled" if task.current_attempt_id else None)
            if task.current_attempt_id is not None:
                # Match canonical task cancellation: fencing stops old writes,
                # while lease expiry makes that SAME attempt recoverable now.
                # Keep its live handle until recovery proves joined shutdown;
                # neither cancellation nor expiry establishes completion.
                await connection.execute(
                    "update jarvis_task_attempts set lease_until=clock_timestamp() where owner_id=$1 and id=$2",
                    task.owner_id, task.current_attempt_id)
            await self.tasks.emit(connection, await self.tasks.load(connection, task.owner_id, task.id),
                                  "cancel_requested" if task.current_attempt_id else "cancelled",
                                  payload={"reason": "readonly_" + reason, "physical_completion": False})
            return True
