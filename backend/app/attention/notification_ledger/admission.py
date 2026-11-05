"""Router/voice task admission through the same canonical request identities.

No in-process cache is an admission authority. TaskService/TaskRepository retain
current-source validation, request fingerprints, grants and transactional dedupe.
"""
from types import SimpleNamespace
from uuid import NAMESPACE_URL, UUID, uuid5

from app.tasks.context_repository import ContextRepository
from app.tasks.models import (
    TaskCreate, TaskError, TaskMutation, TaskStatus, TaskVoiceSource, digest, json_object, json_text,
)
from app.tasks.voice_tools import MutationArguments, StartArguments
from app.tools.invocation import CanonicalRequestIdentity


def canonical_request_id(scope, names: CanonicalRequestIdentity, operation, *, task_ref=None, slot=1):
    if operation not in {"create", "update", "cancel", "resume"}:
        raise ValueError("Unsupported task operation")
    if type(slot) is not int or not 1 <= slot <= 20:
        raise ValueError("Task slots identify independent requests, not retries")
    if (operation == "create") != (task_ref is None) or (operation != "create" and slot != 1):
        raise ValueError("Task mutations require the canonical target and default slot")
    return uuid5(NAMESPACE_URL, json_text({
        "namespace": "jarvis.voice_task.v1", "owner": scope.owner_id,
        "session": str(scope.session_id), "source_namespace": names.namespace,
        "source_event_id": names.event_id, "operation": operation,
        "task_ref": task_ref, "slot": slot,
    }))


def _correction_conflict():
    return TaskError("task_request_conflict", "This proposal is not a compatible canonical transcript correction.")


async def _correction_plan(repository, sources, connection, owner_id, request, source, mode, slot,
                           *, lock_sources=False):
    """Prepare a CAS; recheck the same proof inside the owner's mutation hook."""
    receipt = await connection.fetchrow(
        "select * from jarvis_task_requests where owner_id=$1 and request_id=$2", owner_id, request.request_id)
    if receipt is None or receipt["operation"] != "create":
        raise _correction_conflict()
    task = await repository.load(connection, owner_id, receipt["task_id"])
    accepted = await connection.fetchrow("""select p.id,p.content,p.availability from jarvis_task_events e
        join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
        where e.owner_id=$1 and e.task_id=$2 and e.event_type='accepted' and e.task_revision=$3
        order by e.sequence limit 1""", owner_id, task.id, receipt["accepted_revision"])
    if accepted is None or accepted["availability"] != "available" or accepted["content"] is None:
        raise _correction_conflict()
    description = json_object(accepted["content"])
    original = description.get("instruction_request")
    try:
        initial = TaskVoiceSource(UUID(original["event_id"]), original["revision"], UUID(original["session_id"]))
    except (KeyError, TypeError, ValueError, AttributeError):
        raise _correction_conflict() from None
    first = await sources._record(connection, owner_id, initial.event_id)
    current = await sources._record(connection, owner_id, source.event_id)
    names = CanonicalRequestIdentity.from_record(current)
    scope = SimpleNamespace(owner_id=owner_id, session_id=source.session_id)
    if (names is None or not current.authorizes_request or current.kind != "user_transcript"
            or current.revision != source.revision or current.session_id != source.session_id
            or first.kind != "user_transcript" or first.trust != "user_input"
            or first.finality not in {"final", "revised"} or first.availability != "available"
            or first.revision != initial.revision or first.session_id != initial.session_id
            or initial.session_id != source.session_id or request.session_id != source.session_id
            or task.session_id != source.session_id or source.revision <= initial.revision
            or current.origin_sequence != first.origin_sequence
            or names != CanonicalRequestIdentity.from_record(first)
            or request.request_id != canonical_request_id(scope, names, "create", slot=slot)):
        raise _correction_conflict()
    # Change the objective, not the owner's display title or original admission
    # scope. The original fingerprint includes mode, backend, grants and whether
    # the area was explicitly selected rather than inferred from the session.
    if not isinstance(description.get("title"), str) or not isinstance(description.get("requested_outcome"), str):
        raise _correction_conflict()
    original_request = request.model_copy(update={"title": description["title"],
        "requested_outcome": description["requested_outcome"]})
    if (mode != description.get("mode") or digest(repository._create_parameters(
            original_request, source=initial, mode=mode)) != receipt["parameters_digest"]):
        raise _correction_conflict()
    if lock_sources:
        for payload_id in sorted({accepted["id"], first.payload_id, current.payload_id}):
            if await connection.fetchval("select lock_jarvis_copy_source($1,$2)", owner_id, payload_id) != "available":
                raise _correction_conflict()
    identity = uuid5(NAMESPACE_URL, json_text({"namespace": "jarvis.voice_task.asr_correction.v1",
        "create_request_id": str(request.request_id), "source": source.parameters()}))
    replay = await connection.fetchrow(
        "select * from jarvis_task_requests where owner_id=$1 and request_id=$2", owner_id, identity)
    if replay is not None:
        if replay["operation"] != "update" or replay["task_id"] != task.id:
            raise _correction_conflict()
        expected = replay["accepted_revision"] - 1
    else:
        previous = await sources._record(connection, owner_id, task.request_event_id)
        if (previous.session_id != source.session_id or previous.revision != task.request_revision
                or previous.revision >= source.revision or previous.availability != "available"
                or previous.origin_sequence != current.origin_sequence
                or CanonicalRequestIdentity.from_record(previous) != names
                or task.status in {TaskStatus.CANCELLED, TaskStatus.CANCEL_REQUESTED}):
            raise _correction_conflict()
        expected = task.revision
    mutation = TaskMutation(request_id=identity, expected_revision=expected, message=request.requested_outcome)
    if replay is not None and replay["parameters_digest"] != digest(repository._mutation_parameters(mutation, source)):
        raise _correction_conflict()
    return task.id, mutation


async def create_or_correct(service, owner_id, request, *, source, mode, slot=1, sources=None):
    """Shared router/native admission, with bounded durable ASR correction.

    Ordinary creation/replay retains the existing service fast path. No local
    cache, second task identity or retry loop becomes mutation authority.
    """
    try:
        return await service.create(owner_id, request, source=source, mode=mode)
    except TaskError as error:
        if error.code != "task_request_conflict" or not callable(getattr(service.repository, "read_transaction", None)):
            raise
    repository = service.repository
    sources = sources or ContextRepository(repository)
    async with repository.read_transaction() as connection:
        task_id, mutation = await _correction_plan(
            repository, sources, connection, owner_id, request, source, mode, slot)

    async def guard(connection, task):
        proved_id, proved_mutation = await _correction_plan(
            repository, sources, connection, owner_id, request, source, mode, slot, lock_sources=True)
        if proved_id != task_id or proved_mutation != mutation:
            raise _correction_conflict()
        # Uncertain submissions retain the owner's UNKNOWN/recovery barrier.
        # A settled submission is not permission to redo an external effect.
        if await connection.fetchval("""select exists(select 1 from jarvis_task_actions
            where owner_id=$1 and task_id=$2 and submission_token is not null
            and status not in ('submitting','outcome_unknown'))""", owner_id, task_id):
            raise TaskError("task_reconciliation_required",
                            "A submitted action needs explicit review before applying this transcript correction.")

    # TaskService's update delegates to this same owner. Its admission hook
    # commits the proof, CAS, fence and action transition atomically, without
    # nesting another owner transaction or changing shared contracts.
    return await repository.mutate(owner_id, task_id, "update", mutation, source=source, admission=guard)


class TaskAdmissionAdapter:
    def __init__(self, service, *, sources=None):
        self.service = service
        self.repository = service.repository
        self.sources = sources if sources is not None else ContextRepository(self.repository)

    async def _request(self, scope, event_id, revision, operation, *, task_ref=None, slot=1):
        source = TaskVoiceSource(event_id, revision, scope.session_id)
        record = await self.sources.get(scope.owner_id, event_id, session_id=scope.session_id)
        names = CanonicalRequestIdentity.from_record(record)
        if not record.authorizes_request or record.revision != revision or names is None:
            raise TaskError("task_request_not_finalized", "The exact finalized request is unavailable or changed.", 409)
        return canonical_request_id(scope, names, operation, task_ref=task_ref, slot=slot), source

    async def create(self, scope, event_id, revision, arguments: StartArguments):
        args = StartArguments.model_validate(arguments)
        identity, source = await self._request(scope, event_id, revision, "create", slot=args.request_slot)
        requested = set(args.requested_capabilities)
        environment = (("networked_desktop_sandbox" if "sandbox_network" in requested else "desktop_sandbox")
            if "sandbox_desktop" in requested else "networked_sandbox" if "sandbox_network" in requested else
            "offline_sandbox" if requested & {"sandbox_compute", "publish_files"} else "none")
        return await create_or_correct(self.service, scope.owner_id, TaskCreate(request_id=identity,
            title=args.title, requested_outcome=args.requested_outcome, session_id=scope.session_id,
            knowledge_area_id=args.knowledge_area_id, execution_environment=environment,
            publish_artifacts="publish_files" in requested, edit_memory="memory_edit" in requested),
            source=source, mode=args.mode, slot=args.request_slot, sources=self.sources)

    async def mutate(self, scope, event_id, revision, operation, arguments: MutationArguments):
        if operation not in {"update", "cancel", "resume"}:
            raise ValueError("Unsupported task mutation")
        args = MutationArguments.model_validate(arguments)
        task_id = await self.repository.resolve_reference(scope.owner_id, args.task_ref)
        identity, source = await self._request(scope, event_id, revision, operation, task_ref=args.task_ref)
        return await self.service.mutate(scope.owner_id, task_id, operation,
            TaskMutation(request_id=identity, expected_revision=args.expected_revision, message=args.message),
            source=source)

