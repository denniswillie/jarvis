"""Episode operations over canonical responsibility rows and request receipts.

This adapter shares ResponsibilityRepository's owner lock, pool and transaction.
It neither starts a scheduler nor makes notification/delivery/physical-safety claims.
Captured user requests authorize configuration and user-report mutations. Explicit
visual policies additionally admit the bounded predicate adapter's canonical
decision receipts. Trusted-external-source settlement remains unsupported.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from hashlib import sha256
import re
from uuid import UUID, uuid4

from asyncpg import PostgresError

from app.live_state.models import ContractModel, SessionScope, SourceDependency
from app.live_state.perception import (
    CheckResolution, DurableResponsibilityCapability, PersistentTrackingOutcome, SituationalCheck,
    RequestTargetSpan, VisualRequestGrounding,
)
from app.reminders.duties import decoded
from app.reminders.episode_models import (
    AdmitEpisodeCheck, ChangeEpisodeCheck, EpisodeCheckPayload, EpisodeCheckSpec,
)
from app.tools.inputs import ToolInputError


STORAGE_ERRORS = (PostgresError, OSError, TimeoutError)


def reference(namespace, identity, revision):
    return SourceDependency(source_namespace=namespace, source_id=identity, revision=revision)


class EpisodeCheckResult(ContractModel):
    outcome: PersistentTrackingOutcome
    check: SituationalCheck | None = None
    replayed: bool = False
    model_interpretation: dict[str, str | None] | None = None


class EpisodeCheckSnapshot(ContractModel):
    """Source-guarded owner projection; not a pre-materialized voice payload."""
    check: SituationalCheck
    payload: EpisodeCheckPayload
    dependencies: tuple[SourceDependency, ...]
    source_payload_ids: tuple[UUID, ...]


class EpisodeCheckRead(ContractModel):
    capability: DurableResponsibilityCapability
    snapshots: tuple[EpisodeCheckSnapshot, ...] = ()
    gaps: tuple[str, ...] = ()
    has_more: bool = False


class EpisodeCheckOwner:
    def __init__(self, repository, *, enabled=False, clock=lambda: datetime.now(UTC)):
        self.repository, self.enabled, self.clock = repository, enabled, clock
        self._capability = DurableResponsibilityCapability(owner_namespace="responsibility",
            status="unavailable" if enabled else "disabled", reason="Episode owner has not initialized.")

    @property
    def capability(self):
        return self._capability

    async def initialize(self):
        """Read-only schema capability probe; no service, migration or polling loop."""
        if not self.enabled or self.repository.pool is None:
            return self.capability
        try:
            async with self.repository.pool.acquire() as c:
                await c.fetch("""select o.check_revision,o.episode_id,o.episode_revision,
                    o.check_payload,o.check_lifecycle,o.check_resolution,r.spec
                    from jarvis_responsibility_occurrences o
                    join jarvis_responsibilities r on r.owner_id=o.owner_id and r.id=o.responsibility_id limit 0""")
        except STORAGE_ERRORS:
            self._unavailable()
        else:
            self._capability = DurableResponsibilityCapability(owner_namespace="responsibility", status="available")
        return self.capability

    def _unavailable(self):
        self._capability = DurableResponsibilityCapability(owner_namespace="responsibility", status="unavailable",
            reason="Episode storage is unavailable; retry the same request identity after recovery.")

    def _unavailable_result(self, scope):
        return EpisodeCheckResult(outcome=PersistentTrackingOutcome(scope=scope,
            capability=self.capability, status="unavailable"))

    async def _projection_lock(self, c, scope):
        await self.repository._lock(c, scope.owner_id)
        if await c.fetchval("select owner_id from jarvis_context_streams where owner_id=$1 for update", scope.owner_id) is None:
            raise ToolInputError("check_evidence_unavailable", "Canonical context is unavailable.")

    async def _snapshot(self, c, scope, rule, occurrence):
        result = await self._result(c, scope, rule, occurrence)
        payload = EpisodeCheckPayload.model_validate(decoded(occurrence["check_payload"]))
        refs = (payload.admission_request, payload.last_request,
                *((payload.visual_request,) if payload.visual_request is not None else ()), *payload.dependencies,
                *(ref.source for ref in result.check.evidence_refs))
        # Deduplicate only exact references; conflicting versions fail closed.
        deps = tuple(dict.fromkeys(refs))
        if len({ref.source_id for ref in deps}) != len(deps):
            raise ToolInputError("check_evidence_unavailable", "The check retains conflicting source versions.")
        events = await self._events(c, scope, deps)
        return EpisodeCheckSnapshot(check=result.check, payload=payload,
            dependencies=(result.outcome.canonical_owner_ref, result.outcome.check_ref,
                          result.check.episode_or_object_ref, *deps),
            source_payload_ids=tuple(sorted({row["payload_id"] for row in events})))

    async def read(self, scope: SessionScope, *, include_inactive=False, limit=20):
        """Bounded background read, never an admission receipt or a voice-path cache."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Read limit must be between 1 and 100.")
        if self.capability.status != "available":
            return EpisodeCheckRead(capability=self.capability)
        snapshots, gaps = [], []
        try:
            async with self.repository.pool.acquire() as c, c.transaction():
                await self._projection_lock(c, scope)
                ids = await c.fetch("""select id,responsibility_id from jarvis_responsibility_occurrences
                    where owner_id=$1 and episode_id is not null and ($2 or status='open')
                    order by created_at,id limit $3""", scope.owner_id, include_inactive, limit + 1)
                for row in ids[:limit]:
                    rule = await self.repository._rule(c, scope.owner_id, row["responsibility_id"])
                    self._require_episode(rule)
                    occurrence = await self._occurrence(c, scope, rule, row["id"])
                    try:
                        snapshots.append(await self._snapshot(c, scope, rule, occurrence))
                    except ToolInputError:
                        gaps.append("episode_check_evidence_unavailable:" + str(row["id"]))
        except STORAGE_ERRORS:
            self._unavailable()
            return EpisodeCheckRead(capability=self.capability)
        return EpisodeCheckRead(capability=self.capability, snapshots=tuple(snapshots),
            gaps=tuple(gaps), has_more=len(ids) > limit)

    @asynccontextmanager
    async def guard(self, scope: SessionScope, *, responsibility: SourceDependency,
                    check: SourceDependency, episode: SourceDependency, connection=None):
        """Yield (connection, exact snapshot) under canonical locks for an effect.

        Caller-owned transactions may be borrowed; never call change/admit while
        holding this guard on a separate connection. It grants no settlement right.
        Projection bindings must materialize their exact body over returned roots
        and repeat this guard at effect time; payload survival alone is insufficient.
        """
        if self.capability.status != "available":
            raise ToolInputError("episode_owner_unavailable", "The canonical episode owner is unavailable.")
        for dep, namespace in ((responsibility, "responsibility"), (check, "responsibility_check"), (episode, "episode")):
            if dep.source_namespace != namespace or dep.generation is not None:
                raise ToolInputError("check_reference_invalid", "Use the canonical owner reference namespaces and revisions.")

        async def validate(c):
            if not c.is_in_transaction():
                raise RuntimeError("An episode effect guard requires an actual transaction.")
            await self._projection_lock(c, scope)
            rule = await self.repository._rule(c, scope.owner_id, responsibility.source_id, responsibility.revision)
            self._require_episode(rule)
            occurrence = await self._occurrence(c, scope, rule, check.source_id)
            if (occurrence["check_revision"], occurrence["episode_id"], occurrence["episode_revision"]) != (
                    check.revision, episode.source_id, episode.revision):
                raise ToolInputError("check_revision_conflict", "The canonical check or episode changed.")
            if rule["status"] != "active" or occurrence["check_lifecycle"] != "tracking":
                raise ToolInputError("episode_check_inactive", "This check no longer accepts current evidence.")
            snapshot = await self._snapshot(c, scope, rule, occurrence)
            now = self.clock()
            if ((snapshot.payload.observation_window_expires_at is not None and now >= snapshot.payload.observation_window_expires_at)
                    or (snapshot.payload.snoozed_until is not None and now < snapshot.payload.snoozed_until)):
                raise ToolInputError("episode_check_inactive", "This check is outside its active observation window.")
            return snapshot

        if connection is not None:
            yield connection, await validate(connection)
        else:
            async with self.repository.pool.acquire() as c, c.transaction():
                yield c, await validate(c)

    async def _begin(self, c, scope, source):
        await self.repository._lock(c, scope.owner_id)
        # Canonical ingress assigns its sequence under this row lock. This keeps
        # request supersession from racing authorization inside our transaction.
        stream = await c.fetchval("select owner_id from jarvis_context_streams where owner_id=$1 for update",
                                  scope.owner_id)
        if stream is None:
            raise ToolInputError("request_evidence_unavailable", "No owned captured request exists.")
        payload = await self.repository._require_source(c, scope.owner_id, scope.session_id, source)
        await self._lock_payloads(c, scope.owner_id, (payload,))

    async def _lock_payloads(self, c, owner, roots):
        if roots and not await c.fetchval("select lock_jarvis_copy_sources($1,$2::uuid[])", owner, list(set(roots))):
            raise ToolInputError("check_evidence_unavailable", "Original evidence is no longer available.")

    async def _events(self, c, scope, dependencies):
        rows = []
        for dep in dependencies:
            if dep.source_namespace != "context_event" or dep.generation is not None:
                raise ToolInputError("check_evidence_unavailable", "No qualified owner validates this evidence namespace.")
            row = await c.fetchrow("""select e.*,p.availability,p.content from jarvis_context_events e
                join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
                where e.owner_id=$1 and e.id=$2 and e.revision=$3
                and e.finality in ('final','revised') and p.availability='available' and p.content is not null
                and not exists(select 1 from jarvis_context_events n
                    where n.owner_id=e.owner_id and n.supersedes_event_id=e.id)""",
                scope.owner_id, dep.source_id, dep.revision)
            if row is None:
                raise ToolInputError("check_evidence_unavailable", "A required source is missing, withdrawn or superseded.")
            rows.append(row)
        await self._lock_payloads(c, scope.owner_id, [r["payload_id"] for r in rows])
        return rows

    @staticmethod
    def _require_episode(rule):
        if decoded(rule["spec"]).get("kind", "scheduled") != "episode_check":
            raise ToolInputError("scheduled_responsibility", "Scheduled duties keep their existing owner API.")
        EpisodeCheckSpec.model_validate(decoded(rule["spec"]))

    async def _occurrence(self, c, scope, rule, identity):
        row = await c.fetchrow("""select * from jarvis_responsibility_occurrences
            where owner_id=$1 and responsibility_id=$2 and id=$3 for update""", scope.owner_id, rule["id"], identity)
        if row is None or row["episode_id"] is None:
            raise ToolInputError("episode_check_not_found", "No episode check exists in this owner/rule scope.")
        return row

    async def _visual_grounding(self, c, scope, request_ref, draft):
        """Read current wording under the existing source/owner locks, never a quote cache."""
        event, = await self._events(c, scope, (request_ref,))
        content = decoded(event["content"])
        text = content.get("text") if isinstance(content, dict) else None
        if (event["kind"] not in {"user_request", "user_transcript"} or event["trust"] != "user_input"
                or not isinstance(text, str) or not text.strip() or len(text) > 1000):
            raise ToolInputError("request_wording_unavailable",
                "Current nonblank finalized user wording must fit the existing bounded observation question.")

        def whole_span(start, end):
            word = lambda char: char.isalnum() or char == "_"
            return (0 <= start < end <= len(text) and bool(text[start:end].strip())
                    and not (start and word(text[start - 1]) and word(text[start]))
                    and not (end < len(text) and word(text[end - 1]) and word(text[end])))

        spans = tuple(draft.target_spans)
        association = draft.object_association
        reason = "no_target_span"
        if spans:
            if (any(not whole_span(span.start, span.end) for span in spans)
                    or any(left.end > right.start for left, right in zip(spans, spans[1:]))):
                spans, reason = (), "invalid_target_span"
        elif association is not None:
            matches = [match for match in re.finditer(re.escape(association), text, flags=re.IGNORECASE)
                       if whole_span(match.start(), match.end())]
            if len(matches) == 1:
                spans = (RequestTargetSpan(start=matches[0].start(), end=matches[0].end()),)
            else:
                reason = "ambiguous_literal_match" if matches else "model_target_not_supported"
        if len(spans) > 1:
            reason = "multiple_targets_unresolved"
        elif spans:
            literal = text[spans[0].start:spans[0].end]
            if association is not None and literal.casefold() != association.casefold():
                reason = "model_target_not_supported"
            elif literal.casefold().strip() in {
                "it", "that", "this", "them", "these", "those", "this one", "that one"
            }:
                reason = "deictic_reference_unresolved"
            else:
                reason = "literal_span_validated"
        supported = reason == "literal_span_validated"
        return VisualRequestGrounding(request_ref=request_ref, request_session_id=event["session_id"],
            payload_id=event["payload_id"], wording_sha256=sha256(text.encode("utf-8")).hexdigest(),
            user_wording=text, target_spans=spans, status="literal_supported" if supported else "unresolved",
            reason=reason, clarification_needed=not supported)

    async def _result(self, c, scope, rule, occurrence, *, replayed=False):
        payload = EpisodeCheckPayload.model_validate(decoded(occurrence["check_payload"]))
        resolution = (CheckResolution.model_validate(decoded(occurrence["check_resolution"]))
                      if occurrence["check_resolution"] is not None else None)
        evidence_available = "unknown"
        grounding, interpretation = None, None
        question = payload.question
        if payload.visual_request is not None:
            # Keep source loss from preventing an otherwise authorized cancel.
            # _snapshot/guard still require the original roots; this fallback
            # contains neither the vanished quote nor the model's guessed target.
            try:
                grounding = await self._visual_grounding(c, scope, payload.visual_request, payload)
            except ToolInputError:
                question = "Original visual request unavailable; do not infer its target or completed state."
                evidence_available = "unavailable"
            else:
                question = grounding.user_wording
                interpretation = {"authority": "model_interpretation_not_user_truth",
                                  "question": payload.question, "object_association": payload.object_association}
        if resolution is not None:
            try:
                await self._events(c, scope, tuple(r.source for r in resolution.source_refs))
            except ToolInputError:
                evidence_available = "unavailable"
            else:
                evidence_available = "available"
        episode = reference("episode", occurrence["episode_id"], occurrence["episode_revision"])
        owner_ref = reference("responsibility", rule["id"], rule["revision"])
        check = SituationalCheck(id=occurrence["id"], revision=occurrence["check_revision"], scope=scope,
            episode_or_object_ref=episode, origin="explicit_request", continuity="durable",
            canonical_owner_ref=owner_ref, question=question, request_grounding=grounding,
            activation_cues=payload.activation_cues,
            dependency_versions=payload.dependencies, lifecycle=occurrence["check_lifecycle"],
            evidence_availability=evidence_available,
            evidence_refs=resolution.source_refs if resolution else (),
            observation_window_expires_at=payload.observation_window_expires_at,
            snoozed_until=payload.snoozed_until, resolution=resolution)
        return EpisodeCheckResult(outcome=PersistentTrackingOutcome(scope=scope, capability=self.capability,
            status="accepted", canonical_owner_ref=owner_ref,
            check_ref=reference("responsibility_check", check.id, check.revision)), check=check, replayed=replayed,
            model_interpretation=interpretation)

    async def admit(self, scope: SessionScope, source, request: AdmitEpisodeCheck, *, visual_memory=False):
        if self.capability.status != "available":
            return self._unavailable_result(scope)
        result = None
        try:
            async with self.repository.pool.acquire() as c, c.transaction():
                await self._begin(c, scope, source)
                old, fingerprint = await self.repository._request(c, scope.owner_id, scope.session_id,
                    source, "episode:admit", request)
                if old:
                    rule = await self.repository._rule(c, scope.owner_id, old["responsibility_id"])
                    self._require_episode(rule)
                    occurrence = await self._occurrence(c, scope, rule, old["occurrence_id"])
                    if visual_memory and EpisodeCheckPayload.model_validate(decoded(
                            occurrence["check_payload"])).visual_request is None:
                        raise ToolInputError("visual_grounding_unavailable",
                            "This earlier check has no visual request binding; do not silently upgrade its interpretation.")
                    result = await self._result(c, scope, rule, occurrence, replayed=True)
                else:
                    now = self.clock()
                    req_ref = reference("context_event", source.request_event_id, source.request_revision)
                    if visual_memory:
                        await self._visual_grounding(c, scope, req_ref, request.check)
                    if request.check.observation_window_expires_at is not None and request.check.observation_window_expires_at <= now:
                        raise ToolInputError("observation_window_expired", "Admit a current window, not expired tracking.")
                    await self._events(c, scope, request.check.dependencies)
                    if request.responsibility_id is not None:
                        rule = await self.repository._rule(c, scope.owner_id, request.responsibility_id, request.expected_revision)
                        self._require_episode(rule)
                        if rule["status"] == "cancelled" or EpisodeCheckSpec.model_validate(decoded(rule["spec"])) != request.spec:
                            raise ToolInputError("episode_rule_conflict", "Use an uncancelled matching rule or admit a new responsibility.")
                        rule = await c.fetchrow("""update jarvis_responsibilities set revision=revision+1,
                            status='active',updated_at=clock_timestamp() where owner_id=$1 and id=$2 returning *""",
                            scope.owner_id, rule["id"])
                    else:
                        rule = await c.fetchrow("""insert into jarvis_responsibilities
                            (owner_id,session_id,spec,next_due_at) values($1,$2,$3::jsonb,null) returning *""",
                            scope.owner_id, scope.session_id, request.spec.model_dump_json())
                    req_ref = reference("context_event", source.request_event_id, source.request_revision)
                    payload = EpisodeCheckPayload(**request.check.model_dump(), admission_request=req_ref,
                        last_request=req_ref, episode_started_at=now,
                        visual_request=req_ref if visual_memory else None)
                    occurrence = await c.fetchrow("""insert into jarvis_responsibility_occurrences
                        (id,owner_id,responsibility_id,rule_revision,scheduled_for,reminder_id,
                         episode_id,episode_revision,check_payload,check_lifecycle)
                        values($1,$2,$3,$4,$5,null,$6,1,$7::jsonb,'tracking') returning *""",
                        uuid4(), scope.owner_id, rule["id"], rule["revision"], now, uuid4(), payload.model_dump_json())
                    await self.repository._receipt(c, scope.owner_id, source, "episode:admit", request,
                        fingerprint, rule["id"], occurrence_id=occurrence["id"])
                    result = await self._result(c, scope, rule, occurrence)
        except STORAGE_ERRORS:
            self._unavailable()
            return self._unavailable_result(scope)
        # Exiting the transaction is part of acceptance, including commit failure.
        return result

    async def apply_visual_resolution(self, connection, scope, *, snapshot, decision_receipt):
        """Internal owner commit, not a model/tool command or a new user report.

        The resolver owns model work OUTSIDE this transaction. This method
        re-enters the existing owner guard, validates its canonical decision,
        and preserves every user-report field. Caller retains source/camera/cue
        applicability through commit on this SAME borrowed connection.
        """
        from app.reminders.visual_resolution import DECISION_NAMESPACE, resolution_from_receipt
        if not connection.is_in_transaction():
            raise RuntimeError("Visual settlement requires the guarded effect transaction")
        expected = snapshot.check
        async with self.guard(scope, responsibility=expected.canonical_owner_ref,
                check=reference("responsibility_check", expected.id, expected.revision),
                episode=expected.episode_or_object_ref, connection=connection) as (_, current):
            if current != snapshot:
                raise ToolInputError("visual_resolution_changed", "The configured check changed during evaluation.")
            decision_ref = reference("context_event", decision_receipt.id, decision_receipt.revision)
            row, = await self._events(connection, scope, (decision_ref,))
            if (row["kind"] != "observation" or row["trust"] != "derived"
                    or row["session_id"] != scope.session_id or row["source_namespace"] != DECISION_NAMESPACE
                    or row["payload_id"] != decision_receipt.payload_id
                    or decoded(row["content"]) != decision_receipt.content):
                raise ToolInputError("visual_decision_unavailable", "No matching canonical predicate decision exists.")
            resolution = resolution_from_receipt(current, decision_receipt, now=self.clock())
            await self._events(connection, scope, tuple(dict.fromkeys(ref.source for ref in resolution.source_refs)))
            rule = await self.repository._rule(connection, scope.owner_id,
                expected.canonical_owner_ref.source_id, expected.canonical_owner_ref.revision)
            occurrence = await self._occurrence(connection, scope, rule, expected.id)
            await self.repository._revoke_reminders(connection, scope.owner_id, [occurrence["reminder_id"]])
            occurrence = await connection.fetchrow("""update jarvis_responsibility_occurrences set
                check_revision=check_revision+1,episode_revision=$4,check_payload=$5::jsonb,
                check_lifecycle=$6,check_resolution=$7::jsonb,status=$8,rule_revision=$9,
                report_state=$10,report_event_id=$11,report_revision=$12,reported_at=$13
                where owner_id=$1 and responsibility_id=$2 and id=$3 returning *""",
                scope.owner_id, rule["id"], occurrence["id"], occurrence["episode_revision"],
                current.payload.model_dump_json(), "resolved", resolution.model_dump_json(), "closed", rule["revision"] + 1,
                occurrence["report_state"], occurrence["report_event_id"], occurrence["report_revision"], occurrence["reported_at"])
            remaining = await connection.fetchval("""select exists(select 1 from jarvis_responsibility_occurrences
                where owner_id=$1 and responsibility_id=$2 and status='open')""", scope.owner_id, rule["id"])
            rule = await connection.fetchrow("""update jarvis_responsibilities set revision=revision+1,status=$3,
                updated_at=clock_timestamp() where owner_id=$1 and id=$2 returning *""",
                scope.owner_id, rule["id"], "active" if remaining else "completed")
            # The derived context decision is the idempotent effect receipt.
            # Do not forge a user request in jarvis_responsibility_requests.
            return await self._result(connection, scope, rule, occurrence)

    async def _validate_resolution(self, c, scope, source, occurrence, payload, resolution, now):
        episode = reference("episode", occurrence["episode_id"], occurrence["episode_revision"])
        if resolution.applicable_episode_or_object_ref != episode or resolution.resolved_at > now:
            raise ToolInputError("resolution_episode_conflict", "Resolution must apply to the current episode and time.")
        if payload.observation_window_expires_at is not None and now >= payload.observation_window_expires_at:
            raise ToolInputError("observation_window_expired", "Revise the expired check before resolving it.")
        if resolution.basis != "user_report":
            raise ToolInputError("resolution_validator_unavailable", "Visual and trusted-source predicates are not yet qualified.")
        expected = reference("context_event", source.request_event_id, source.request_revision)
        if len(resolution.source_refs) != 1 or resolution.source_refs[0].source != expected:
            raise ToolInputError("resolution_request_mismatch", "User-report resolution needs this exact captured user request.")
        evidence = resolution.source_refs[0]
        if evidence.frame_id is not None:
            raise ToolInputError("resolution_basis_mismatch", "A user report cannot relabel visual evidence.")
        row, = await self._events(c, scope, (expected,))
        original_time = row["source_time"] or row["recorded_at"]
        if (row["session_id"] != scope.session_id or row["trust"] != "user_input"
                or row["kind"] not in {"user_request", "user_transcript"}
                or evidence.captured_at != original_time or original_time < payload.episode_started_at):
            raise ToolInputError("resolution_evidence_mismatch", "Preserve the original, current-episode user-report evidence.")
        await self._events(c, scope, payload.dependencies)

    async def change(self, scope: SessionScope, source, request: ChangeEpisodeCheck):
        if self.capability.status != "available":
            return self._unavailable_result(scope)
        result = None
        operation = "episode:" + request.operation
        try:
            async with self.repository.pool.acquire() as c, c.transaction():
                await self._begin(c, scope, source)
                old, fingerprint = await self.repository._request(c, scope.owner_id, scope.session_id, source, operation, request)
                rule = await self.repository._rule(c, scope.owner_id, request.responsibility_id,
                    None if old else request.expected_revision)
                self._require_episode(rule)
                occurrence = await self._occurrence(c, scope, rule, request.occurrence_id)
                if old:
                    result = await self._result(c, scope, rule, occurrence, replayed=True)
                else:
                    if (occurrence["check_revision"], occurrence["episode_id"], occurrence["episode_revision"]) != (
                            request.expected_check_revision, request.episode_id, request.expected_episode_revision):
                        raise ToolInputError("check_revision_conflict", "The check or episode changed; read current versions.")
                    if rule["status"] == "cancelled" or occurrence["check_lifecycle"] == "dismissed":
                        raise ToolInputError("episode_check_inactive", "Cancelled tracking cannot be revived by late evidence.")
                    now = self.clock()
                    payload = EpisodeCheckPayload.model_validate(decoded(occurrence["check_payload"]))
                    lifecycle, status = occurrence["check_lifecycle"], occurrence["status"]
                    resolution = None
                    episode_revision = occurrence["episode_revision"]
                    report, report_id, report_revision, reported_at = (
                        occurrence["report_state"], occurrence["report_event_id"], occurrence["report_revision"], occurrence["reported_at"])
                    if request.operation == "revise":
                        if request.check.observation_window_expires_at is not None and request.check.observation_window_expires_at <= now:
                            raise ToolInputError("observation_window_expired", "A revised observation window must be current.")
                        await self._events(c, scope, request.check.dependencies)
                        visual_request = (reference("context_event", source.request_event_id, source.request_revision)
                                          if payload.visual_request is not None else None)
                        if visual_request is not None:
                            await self._visual_grounding(c, scope, visual_request, request.check)
                        rebind = request.rebind_episode or request.check.object_association != payload.object_association
                        episode_revision += int(rebind)
                        payload = EpisodeCheckPayload(**request.check.model_dump(), admission_request=payload.admission_request,
                            last_request=reference("context_event", source.request_event_id, source.request_revision),
                            episode_started_at=now if rebind else payload.episode_started_at,
                            visual_request=visual_request)
                        lifecycle, status, report, report_id, report_revision, reported_at = "tracking", "open", "unknown", None, None, None
                    elif request.operation == "cancel":
                        lifecycle, status = "dismissed", "cancelled"
                    elif lifecycle != "tracking":
                        raise ToolInputError("episode_check_inactive", "Explicitly revise before reopening resolved or expired tracking.")
                    elif request.operation == "resolve":
                        await self._validate_resolution(c, scope, source, occurrence, payload, request.resolution, now)
                        resolution = request.resolution
                        lifecycle, status = "resolved", "closed"
                        report, report_id, report_revision, reported_at = "user_reported_done", source.request_event_id, source.request_revision, now
                    elif request.operation == "expire":
                        if payload.observation_window_expires_at is None or now < payload.observation_window_expires_at:
                            raise ToolInputError("observation_window_active", "Expiry requires an elapsed observation window.")
                        lifecycle = "expired"  # Keep the actual obligation open and unresolved.
                    elif request.operation == "snooze":
                        if request.until <= now:
                            raise ToolInputError("invalid_snooze", "Snooze until a future timestamp.")
                        payload = payload.model_copy(update={"snoozed_until": request.until})
                    payload = payload.model_copy(update={"last_request": reference("context_event", source.request_event_id, source.request_revision)})
                    await self.repository._revoke_reminders(c, scope.owner_id, [occurrence["reminder_id"]])
                    occurrence = await c.fetchrow("""update jarvis_responsibility_occurrences set
                        check_revision=check_revision+1,episode_revision=$4,check_payload=$5::jsonb,
                        check_lifecycle=$6,check_resolution=$7::jsonb,status=$8,rule_revision=$9,
                        report_state=$10,report_event_id=$11,report_revision=$12,reported_at=$13
                        where owner_id=$1 and responsibility_id=$2 and id=$3 returning *""",
                        scope.owner_id, rule["id"], occurrence["id"], episode_revision, payload.model_dump_json(),
                        lifecycle, resolution.model_dump_json() if resolution else None, status, rule["revision"] + 1,
                        report, report_id, report_revision, reported_at)
                    # Every unresolved episode survives new episodes and activity changes.
                    remaining = await c.fetchval("""select exists(select 1 from jarvis_responsibility_occurrences
                        where owner_id=$1 and responsibility_id=$2 and status='open')""", scope.owner_id, rule["id"])
                    rule_status = "active" if remaining else ("cancelled" if request.operation == "cancel" else "completed")
                    rule = await c.fetchrow("""update jarvis_responsibilities set revision=revision+1,status=$3,
                        updated_at=clock_timestamp() where owner_id=$1 and id=$2 returning *""", scope.owner_id, rule["id"], rule_status)
                    await self.repository._receipt(c, scope.owner_id, source, operation, request, fingerprint,
                        rule["id"], report_state="user_reported_done" if resolution else None)
                    result = await self._result(c, scope, rule, occurrence)
        except STORAGE_ERRORS:
            self._unavailable()
            return self._unavailable_result(scope)
        return result
