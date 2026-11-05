"""Conversation tools over the existing episode owner and canonical receipts.

The model chooses an operation; it does not supply user-report timestamps or
invent resolution evidence. Durable mutations remain request-idempotent in the
owner. Tool receipts are source-bound snapshots, not permanent truth assertions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, ValidationError, model_validator

from app.live_state.models import ContractModel, Revision, SessionScope
from app.live_state.perception import CheckResolution, EvidenceReference, ObservationText, RequestTargetSpan
from app.reminders.episode_checks import EpisodeCheckOwner, reference
from app.reminders.episode_models import AdmitEpisodeCheck, ChangeEpisodeCheck, EpisodeCheckDraft, EpisodeCheckSpec
from app.tasks.models import TaskError, digest
from app.tools.inputs import ToolInputError, failed, input_error_payload, success
from app.tools.invocation import current_tool_invocation
from app.tools.provenance import SourceBoundToolResult, source_read_arguments
from app.tools.registry import ToolDefinition


class EpisodeCheckCommand(ContractModel):
    responsibility_id: UUID
    occurrence_id: UUID
    expected_revision: Revision
    expected_check_revision: Revision
    episode_id: UUID
    expected_episode_revision: Revision
    operation: Literal["revise", "report_done", "cancel", "snooze", "expire"]
    check: EpisodeCheckDraft | None = None
    rebind_episode: bool = Field(default=False, strict=True)
    until: AwareDatetime | None = None
    request_slot: int = Field(default=1, ge=1, le=20, strict=True)

    @model_validator(mode="after")
    def operation_shape(self):
        if (self.operation == "revise") != (self.check is not None):
            raise ValueError("Only revise supplies a replacement check.")
        if self.rebind_episode and self.operation != "revise":
            raise ValueError("Only an explicit revision may rebind an episode.")
        if (self.operation == "snooze") != (self.until is not None):
            raise ValueError("Only snooze supplies until.")
        return self


class ListEpisodeChecks(ContractModel):
    include_inactive: bool = Field(default=False, strict=True)
    limit: int = Field(default=4, ge=1, le=8, strict=True)


class TrackVisualMemory(ContractModel):
    """Observation-only admission; evidence, resolution and scheduling are not inputs."""
    title: str = Field(min_length=1, max_length=200)
    question: ObservationText
    object_association: str | None = Field(default=None, min_length=1, max_length=200)
    observation_window_expires_at: AwareDatetime | None = None
    request_slot: int = Field(default=1, ge=1, le=20, strict=True)
    target_spans: tuple[RequestTargetSpan, ...] = Field(default=(), max_length=4)


class _EpisodeResult(dict):
    """Trusted projections beside JSON, never reconstructed from model arguments."""
    def __init__(self, payload, checks):
        super().__init__(payload)
        self.checks = tuple(checks)
        self.fingerprint = digest(payload)


def _invalid(error):
    return failed("invalid_tool_input", "; ".join(
        ".".join(map(str, item["loc"])) + ": " + item["msg"]
        for item in error.errors(include_input=False, include_url=False)))


@dataclass(frozen=True)
class EpisodeCheckToolModule:
    owner: EpisodeCheckOwner
    name: ClassVar[str] = "episode_checks"

    def build_tools(self, context):
        if context.session_id is None or not getattr(context.settings, "RESPONSIVE_VOICE_ENABLED", False):
            return []
        scope = SessionScope(owner_id=context.owner_id, session_id=context.session_id)
        owner = self.owner

        def mutation_result(result):
            if result.outcome.status != "accepted":
                return {**failed("episode_tracking_unavailable", result.outcome.capability.reason
                    or "Persistent episode tracking is unavailable; do not promise it was saved."),
                    "tracking": result.outcome.model_dump(mode="json")}
            payload = success({"tracking": result.outcome.model_dump(mode="json"),
                "check": result.check.model_dump(mode="json") if result.check else None,
                "replayed": result.replayed, "as_of": owner.clock().isoformat(),
                "physical_action_verified": False})
            if result.model_interpretation is not None:
                payload["model_interpretation"] = result.model_interpretation
            return _EpisodeResult(payload, (result.check,) if result.check else ())

        async def track(arguments):
            try:
                request = AdmitEpisodeCheck.model_validate(arguments)
                return mutation_result(await owner.admit(scope, current_tool_invocation(), request))
            except ValidationError as error:
                return _invalid(error)
            except ToolInputError as error:
                return input_error_payload(error)

        async def track_visual_memory(arguments):
            try:
                command = TrackVisualMemory.model_validate(arguments)
                # Reuse the existing request-slot identity, original user source,
                # revisions and owner transaction. This is not a find_once watch
                # or permission to resolve a check or deliver an alert.
                request = AdmitEpisodeCheck(
                    spec=EpisodeCheckSpec(title=command.title),
                    check=EpisodeCheckDraft(question=command.question,
                        object_association=command.object_association,
                        target_spans=command.target_spans,
                        observation_window_expires_at=command.observation_window_expires_at),
                    request_slot=command.request_slot)
                result = mutation_result(await owner.admit(scope, current_tool_invocation(), request,
                                                          visual_memory=True))
                if not isinstance(result, _EpisodeResult):
                    return result
                return _EpisodeResult({**result, "observation_only": True,
                    "location_fact_recorded_by_this_call": False,
                    "meaning": "Accepted observation tracking, not an observed location or completed placement. "
                        "The check question comes from the canonical user request; model_interpretation is not user truth. "
                        "Unresolved request_grounding preserves tracking without accepting a concrete target. "
                        "Clarify if needed, not by repeatedly polling this tool. Literal spans prove wording only. "
                        "Visual findings require their own source evidence. This admission creates no recurring "
                        "duty, automatic visual resolution or notification authorization. "
                        "Use list_episode_checks and change_episode_check to revise or cancel this exact goal."},
                    result.checks)
            except ValidationError as error:
                return _invalid(error)
            except ToolInputError as error:
                return input_error_payload(error)

        async def change(arguments):
            try:
                command = EpisodeCheckCommand.model_validate(arguments)
                source = current_tool_invocation()
                values = command.model_dump()
                if command.operation == "report_done":
                    if owner.capability.status != "available":
                        return failed("episode_tracking_unavailable", "The episode owner is unavailable.")
                    # Resolve the trusted invocation, not an ID or timestamp from
                    # tool JSON. The owner revalidates it in the mutation transaction.
                    async with owner.repository.pool.acquire() as connection, connection.transaction():
                        await owner._begin(connection, scope, source)
                        event_ref = reference("context_event", source.request_event_id, source.request_revision)
                        event, = await owner._events(connection, scope, (event_ref,))
                    captured_at = event["source_time"] or event["recorded_at"]
                    episode = reference("episode", command.episode_id, command.expected_episode_revision)
                    values.update(operation="resolve", resolution=CheckResolution(
                        basis="user_report", resolved_at=captured_at,
                        applicable_episode_or_object_ref=episode,
                        source_refs=(EvidenceReference(source=event_ref, captured_at=captured_at,
                            episode_or_object_ref=episode),)))
                    # Original report time makes an identical retry fingerprint
                    # stable even after a different provider call ID or wall time.
                request = ChangeEpisodeCheck.model_validate(values)
                return mutation_result(await owner.change(scope, source, request))
            except ValidationError as error:
                return _invalid(error)
            except ToolInputError as error:
                return input_error_payload(error)

        async def list_checks(arguments):
            try:
                request = ListEpisodeChecks.model_validate(arguments)
                result = await owner.read(scope, include_inactive=request.include_inactive, limit=request.limit)
                payload = {"capability": result.capability.model_dump(mode="json"),
                    "checks": [item.check.model_dump(mode="json") | (
                        {"visual_resolution": item.payload.visual_resolution.model_dump(mode="json"),
                         "visual_policy_check_revision": item.check.revision}
                        if item.payload.visual_resolution is not None else {}) for item in result.snapshots],
                    "gaps": list(result.gaps), "has_more": result.has_more,
                    "as_of": owner.clock().isoformat(), "physical_action_verified": False}
                if result.capability.status != "available":
                    return {**failed("episode_tracking_unavailable", result.capability.reason
                        or "Cannot read persistent episode tracking."), **payload}
                return _EpisodeResult(success(payload), (item.check for item in result.snapshots))
            except ValidationError as error:
                return _invalid(error)

        async def capture_result(result, target):
            capture = target.capture
            if (not isinstance(result, _EpisodeResult) or digest(dict(result)) != result.fingerprint
                    or (capture.owner_id, capture.session_id) != (scope.owner_id, scope.session_id)):
                raise TaskError("episode_receipt_scope", "The episode receipt has no matching trusted projection.")
            repository = capture.repository
            transaction = (repository.publication.transaction(repository.tasks, scope.owner_id)
                if repository.publication is not None else repository.tasks.transaction(scope.owner_id))
            async with transaction as connection:
                await owner._projection_lock(connection, scope)
                roots, originals = set(), {}
                for expected in result.checks:
                    rule = await owner.repository._rule(connection, scope.owner_id,
                        expected.canonical_owner_ref.source_id, expected.canonical_owner_ref.revision)
                    owner._require_episode(rule)
                    occurrence = await owner._occurrence(connection, scope, rule, expected.id)
                    current = await owner._snapshot(connection, scope, rule, occurrence)
                    if current.check != expected:
                        raise TaskError("episode_receipt_changed", "The episode changed before its receipt was captured.")
                    roots.update(current.source_payload_ids)
                    dependencies = tuple(item for item in current.dependencies
                                         if item.source_namespace == "context_event")
                    for event in await owner._events(connection, scope, dependencies):
                        originals[event["id"]] = event
                if len(roots) > 64:
                    raise TaskError("episode_receipt_budget", "Read fewer checks to preserve their source evidence.")
                record = await capture.bound_tool_outcome(connection, target.call_id, target.tool_name,
                    dict(result), tuple(sorted(roots)))
                if originals:
                    await connection.executemany("""insert into jarvis_context_event_dependencies
                        (owner_id,dependent_payload_id,event_id,event_revision,source_payload_id,source_usage)
                        values($1,$2,$3,$4,$5,'current') on conflict do nothing""",
                        [(scope.owner_id, record.payload_id, event["id"], event["revision"], event["payload_id"])
                         for event in originals.values()])
                return SourceBoundToolResult(dict(result), record)

        return [
            ToolDefinition(name="track_visual_memory", mutates_state=True,
                description="Accept an explicit request to observe and remember an object's placement or visual "
                    "state, such as remembering where the user puts something. Accept the observation task even "
                    "before frames arrive; a current camera lookup is not a prerequisite. A no_frame result "
                    "does not itself reject tracking. Ask an observational question "
                    "about visible position/state and uncertainty, not a question containing a guessed answer. "
                    "Preserve the user's object wording and ambiguity; an assistant interpretation is not user "
                    "truth. Clarify if necessary rather than silently renaming the object. This is observation "
                    "tracking grounded by the server in the finalized request. Optional target_spans are Unicode "
                    "code-point [start,end) offsets into that exact wording, not source IDs or entity evidence. "
                    "A unique literal object_association can be grounded without offsets. Unsupported, deictic or "
                    "ambiguous targets remain unresolved while the literal request is retained. Span presence does "
                    "not prove intent, negation handling or coreference. Model question/association are interpretations. "
                    "This is observation "
                    "tracking, NOT find_once lost-item detection, a recurring obligation, an instruction to "
                    "announce every observation, or proof that placement happened. A visible location can be "
                    "recorded without seeing the release; do not infer the release or an out-of-view location. "
                    "Only tracking.status=accepted confirms admission, never that a location fact was saved. "
                    "Reuse request_slot and arguments on retry, including after interrupted speech. For an "
                    "existing goal use list_episode_checks then change_episode_check with exact IDs/revisions "
                    "to revise or cancel; do not start a duplicate. Unrelated conversation does not cancel it. "
                    "Ordinary factual memory without a requested visual inspection uses the existing memory tools.",
                input_schema=TrackVisualMemory.model_json_schema(), handler=track_visual_memory,
                result_capture=capture_result, capture_after_commit=True, audit_arguments=source_read_arguments),
            ToolDefinition(name="track_episode_check", mutates_state=True,
                description="Track an explicitly requested situational check across sessions, such as checking "
                    "belongings before leaving. This is not a scheduled or recurring reminder; use "
                    "save_responsibility for those. For observation-only placement/visual-memory requests use "
                    "track_visual_memory. Ask observational questions, preserve uncertainty, and do not "
                    "infer a permanent obligation without user confirmation. Persistent acceptance requires the "
                    "returned tracking.status=accepted. Reuse request_slot and arguments on retry; do not recreate "
                    "a check just because speech was interrupted. A new occurrence of an existing rule needs its "
                    "exact responsibility_id and expected_revision. visual_resolution is optional: configure it "
                    "ONLY when the current user explicitly authorizes automatic visual confirmation of a specific "
                    "natural-language condition, not merely because they requested tracking. Visibility alone "
                    "is not completion. Omit it for user-report-only checks. Tracking is not a safety guarantee.",
                input_schema=AdmitEpisodeCheck.model_json_schema(), handler=track,
                result_capture=capture_result, capture_after_commit=True, audit_arguments=source_read_arguments),
            ToolDefinition(name="change_episode_check", mutates_state=True,
                description="Revise, snooze, cancel or expire a tracked episode check or visual-memory goal, or record report_done "
                    "only when the current user explicitly reports the action happened. List checks first and "
                    "preserve exact rule/check/episode IDs and revisions. The server supplies original report "
                    "evidence. Never report completion from silence, an object merely being visible, expired "
                    "evidence, or notification playback. Cancel means stop tracking, not action completed. "
                    "A revision replaces the full check; rebind_episode is an explicit correction of the same "
                    "episode, not a new occurrence. A visual_resolution policy requires explicit user authorization; "
                    "list the current policy before editing it. Omitting it in a full replacement disables visual "
                    "settlement. Never use report_done for an automatic visual decision. "
                    "Preserve request_slot and arguments on retry.",
                input_schema=EpisodeCheckCommand.model_json_schema(), handler=change,
                result_capture=capture_result, capture_after_commit=True, audit_arguments=source_read_arguments),
            ToolDefinition(name="list_episode_checks",
                description="Read bounded source-backed episode checks and visual-memory goals, including their exact rule/check/episode "
                    "versions. include_inactive includes earlier resolved/cancelled tracking for past-action "
                    "questions. Results are as-of snapshots, not proof of current appliance or object state. "
                    "User reports remain reports, not independently verified actions. If gaps or has_more are "
                    "present, do not claim a complete history; unavailable evidence means you do not know.",
                input_schema=ListEpisodeChecks.model_json_schema(), handler=list_checks,
                result_capture=capture_result, audit_arguments=source_read_arguments),
        ]
