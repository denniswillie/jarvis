"""Native evidence in canonical attention rows, never a parallel delivery queue.

All methods belong on the attention worker lane, not a socket receive handler.
Existing owner/source guards remain authoritative. No method claims hearing,
manufactures PCM, starts a route, or cancels a task.
"""
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from app.attention.models import ACTIVE_PHASES, AttentionError, DeliveryClaim
from app.attention.notifications import NotificationOutput
from app.live_state.models import SourceDependency
from app.tasks.models import digest, json_object
from app.voice_routes.contracts import (
    DeliveryAttemptReference, DeliveryEvidence, DeliveryEvidenceState as State,
    NotificationOffer, RouteCapabilities, RouteIdentity, Support,
)

from .models import EvidenceResult, SubmissionTicket, VoiceRoutePlan, settlement_for


class VoiceNotificationLedger:
    def __init__(self, attention, *, clock=lambda: datetime.now(UTC),
                 validate_source_generation=None):
        self.attention = attention
        self.tasks = attention.tasks
        self.clock = clock
        self.validate_source_generation = validate_source_generation
        self._issuer = object()

    def _now(self):
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Ledger clocks must be timezone-aware")
        return value

    @staticmethod
    def _source(row, generation=None):
        return SourceDependency(source_namespace=row["source_namespace"],
            source_id=row["source_id"], revision=row["source_revision"], generation=generation)

    async def _validate_source(self, connection, claim, row, source):
        if source != self._source(row, source.generation):
            raise AttentionError("attention_source_changed", "The canonical source binding changed.")
        # Attention generations, route generations and content revisions are not
        # knowledge-source lifecycle generations. Never manufacture one.
        if source.generation is not None:
            if self.validate_source_generation is None or not await self.validate_source_generation(
                    connection, claim, source):
                raise AttentionError("attention_source_generation_unavailable",
                                     "No authoritative current source-generation binding is available.")

    @staticmethod
    def _claim_matches(claim, row, attempt):
        if (attempt is None or attempt["owner_id"] != claim.owner_id
                or attempt["intent_id"] != claim.intent_id
                or attempt["generation"] != claim.generation
                or attempt["session_id"] != claim.session_id
                or attempt["worker_id"] != claim.worker_id
                or row["source_payload_id"] != claim.source_payload_id
                or row["source_namespace"] != claim.source_namespace):
            raise AttentionError("attention_claim_lost", "The original owned attempt does not match.")

    async def _plan(self, connection, claim, attempt):
        if attempt.get("delivery_transport", "legacy_pcm_tts") == "legacy_pcm_tts":
            raise AttentionError("attention_legacy_attempt", "Legacy claims retain their original delivery path.")
        payload = await connection.fetchrow("""/* voice_ledger:plan */
            select content,availability from jarvis_context_payloads
            where owner_id=$1 and id=$2 for share""", claim.owner_id, attempt["plan_payload_id"])
        if payload is None or payload["availability"] != "available" or payload["content"] is None:
            raise AttentionError("attention_plan_unavailable", "The source-bound route plan is unavailable.")
        plan = VoiceRoutePlan.model_validate(json_object(payload["content"]))
        if digest(plan.model_dump(mode="json")) != attempt["plan_digest"]:
            raise AttentionError("attention_plan_conflict", "The immutable plan digest does not match.")
        offer = plan.offer
        if (offer.route.scope.owner_id != claim.owner_id or offer.route.scope.session_id != claim.session_id
                or offer.attempt.notification_intent_id != claim.intent_id
                or offer.attempt.attempt_id != claim.attempt_id
                or offer.attempt.attempt_generation != claim.generation
                or offer.attempt.source_payload_id != claim.source_payload_id
                or offer.offer_id != attempt["voice_offer_id"]
                or offer.route.route_id != attempt["voice_route_id"]
                or offer.route.route_generation != attempt["voice_route_generation"]
                or plan.audio_session_id != attempt["voice_audio_session_id"]
                or plan.protocol_version != attempt["voice_route_protocol_version"]
                or plan.capabilities.transport.value != attempt["delivery_transport"]):
            raise AttentionError("attention_plan_conflict", "The route plan is not bound to this attempt.")
        return plan

    async def prepare(self, claim: DeliveryClaim, *, route: RouteIdentity,
                      audio_session_id: UUID, capabilities: RouteCapabilities,
                      offer_id: UUID, expires_at: datetime, submission_boundary: str,
                      source_generation: int | None = None) -> NotificationOffer:
        """Prepare only an already admitted canonical claim; never create an intent."""
        if route.scope.owner_id != claim.owner_id or route.scope.session_id != claim.session_id:
            raise AttentionError("attention_route_scope", "The selected route does not own this claim.")
        async with self.attention.guard(claim) as (connection, row, attempt, output):
            self._claim_matches(claim, row, attempt)
            if attempt["phase"] != "claimed" or attempt["submitted_at"] is not None:
                raise AttentionError("attention_submission_state", "Preparation must precede submission.")
            old_plan = await self._plan(connection, claim, attempt) if attempt["plan_payload_id"] else None
            text = "".join(output.speech_texts) if isinstance(output, NotificationOutput) else "".join(
                chunk.text for chunk in output.content.speech)
            source = self._source(row, source_generation)
            await self._validate_source(connection, claim, row, source)
            now = self._now()
            deadline = min(expires_at, row["expires_at"]) if row["expires_at"] else expires_at
            if deadline <= now:
                raise AttentionError("attention_offer_expired", "No current speech offer is available.")
            offer = NotificationOffer(route=route, offer_id=offer_id,
                attempt=DeliveryAttemptReference(notification_intent_id=claim.intent_id,
                    attempt_id=claim.attempt_id, attempt_generation=claim.generation,
                    source=source, source_payload_id=claim.source_payload_id),
                text=text, offered_at=old_plan.offer.offered_at if old_plan else now, expires_at=deadline)
            plan = VoiceRoutePlan(audio_session_id=audio_session_id, offer=offer,
                capabilities=capabilities, submission_boundary=submission_boundary,
                source_payload_root_ids=(claim.source_payload_id,))
            if old_plan is not None:
                if plan != old_plan:
                    raise AttentionError("attention_plan_conflict", "An attempt cannot change its route plan.")
                return old_plan.offer
            if attempt["forwarded_samples"] or attempt["played_samples"] is not None or await connection.fetchval(
                    """/* voice_ledger:legacy_chunks */ select exists(select 1 from jarvis_attention_speech_chunks
                    where owner_id=$1 and attempt_id=$2)""", claim.owner_id, claim.attempt_id):
                raise AttentionError("attention_legacy_attempt", "A legacy output cannot become a native attempt.")
            content = plan.model_dump(mode="json")
            payload_id = await self.tasks.payload(connection, claim.owner_id, content,
                                                source_payload_ids=plan.source_payload_root_ids)
            await connection.execute("""/* voice_ledger:prepare */ update jarvis_attention_attempts
                set plan_payload_id=$3,plan_digest=$4,delivery_transport=$5,voice_route_id=$6,
                    voice_route_generation=$7,voice_offer_id=$8,voice_route_protocol_version=1,
                    voice_audio_session_id=$9 where owner_id=$1 and id=$2""",
                claim.owner_id, claim.attempt_id, payload_id, digest(content), capabilities.transport.value,
                route.route_id, route.route_generation, offer_id, audio_session_id)
            await self._append(connection, claim, DeliveryEvidence(
                evidence_id=uuid5(NAMESPACE_URL, f"jarvis.attention.offered.v1:{claim.attempt_id}:{offer_id}"),
                route=route, transport=capabilities.transport, observed_at=offer.offered_at,
                state=State.OFFERED, attribution="explicit", offer_id=offer_id, attempt=offer.attempt),
                roots=(claim.source_payload_id, payload_id))
            return offer

    @staticmethod
    def _same_offer(plan, offer):
        if plan.offer != offer:
            raise AttentionError("attention_offer_conflict", "The offer differs from its immutable plan.")

    async def commit_submission(self, claim, offer) -> SubmissionTicket | None:
        """Commit possible external exposure before returning a one-use write ticket.

        A duplicate returns None. It is never permission to resend an offer.
        """
        async with self.attention.guard(claim) as (connection, row, attempt, _output):
            self._claim_matches(claim, row, attempt)
            plan = await self._plan(connection, claim, attempt)
            self._same_offer(plan, offer)
            await self._validate_source(connection, claim, row, offer.attempt.source)
            if attempt["submitted_at"] is not None:
                return None
            if attempt["phase"] != "claimed" or offer.expires_at <= self._now():
                raise AttentionError("attention_submission_state", "A current unsubmitted offer is required.")
            await connection.execute("""/* voice_ledger:submit */ update jarvis_attention_attempts
                set phase='submitted',submitted_at=clock_timestamp() where owner_id=$1 and id=$2""",
                claim.owner_id, claim.attempt_id)
            await self._append(connection, claim, DeliveryEvidence(
                evidence_id=uuid5(NAMESPACE_URL, f"jarvis.attention.submitted.v1:{claim.attempt_id}:{offer.offer_id}"),
                route=offer.route, transport=plan.capabilities.transport, observed_at=self._now(),
                state=State.SUBMITTED, attribution="explicit", offer_id=offer.offer_id, attempt=offer.attempt),
                roots=(claim.source_payload_id, attempt["plan_payload_id"]))
        return SubmissionTicket(claim, offer, self._issuer)

    @asynccontextmanager
    async def guarded_write(self, ticket: SubmissionTicket):
        async with self.guarded_transaction(ticket):
            yield ticket.offer

    @asynccontextmanager
    async def guarded_transaction(self, ticket: SubmissionTicket):
        """Consume once, then hold current canonical guards for a bounded write.

        The adapter owns deadline/cancellation and actual output attribution.
        On exceptions it must settle unknown; do not retry this ticket.
        """
        if not isinstance(ticket, SubmissionTicket) or ticket._issuer is not self._issuer or ticket._used:
            raise AttentionError("attention_write_ticket_used", "No unused submission ticket exists.")
        ticket._used = True
        claim, offer = ticket.claim, ticket.offer
        async with self.attention.guard(claim) as (connection, row, attempt, _output):
            self._claim_matches(claim, row, attempt)
            plan = await self._plan(connection, claim, attempt)
            self._same_offer(plan, offer)
            if attempt["phase"] != "submitted" or offer.expires_at <= self._now():
                raise AttentionError("attention_submission_state", "The submitted offer is no longer writable.")
            await self._validate_source(connection, claim, row, offer.attempt.source)
            yield connection

    @asynccontextmanager
    async def output_guard(self, claim, offer):
        """Guard follow-up output for the SAME committed offer, not a retry ticket.

        The adapter owns response deduplication. This never creates a submission
        or permits a new offer; it only retains canonical locks through writes.
        """
        async with self.attention.guard(claim) as (connection, row, attempt, _output):
            self._claim_matches(claim, row, attempt)
            plan = await self._plan(connection, claim, attempt)
            self._same_offer(plan, offer)
            if (attempt["submitted_at"] is None or attempt["phase"] not in ACTIVE_PHASES
                    or offer.expires_at <= self._now()):
                raise AttentionError("attention_submission_state", "No current submitted offer exists.")
            await self._validate_source(connection, claim, row, offer.attempt.source)
            yield connection

    @asynccontextmanager
    async def _locked(self, claim):
        # Keep canonical source -> task -> intent -> attempt ordering. A late
        # event is allowed for a retired attempt, but never borrows a new claim.
        async with self.tasks.transaction(claim.owner_id) as connection:
            row, task = await self.attention._entry(connection, claim.owner_id, claim.intent_id, lock=True)
            attempt = await connection.fetchrow("""/* voice_ledger:attempt */ select * from jarvis_attention_attempts
                where owner_id=$1 and id=$2 for update""", claim.owner_id, claim.attempt_id)
            self._claim_matches(claim, row, attempt)
            yield connection, row, task, attempt

    async def _append(self, connection, claim, evidence, *, roots):
        content = evidence.model_dump(mode="json")
        signature = digest(content)
        previous = await connection.fetchrow("""/* voice_ledger:evidence_key */
            select evidence_digest from jarvis_attention_route_evidence
            where owner_id=$1 and evidence_id=$2""", claim.owner_id, evidence.evidence_id)
        if previous:
            if previous["evidence_digest"] != signature:
                raise AttentionError("attention_evidence_conflict", "This evidence ID was already used differently.")
            return False
        if evidence.playback is not None:
            previous_receipt = await connection.fetchval("""/* voice_ledger:receipt_key */
                select exists(select 1 from jarvis_attention_route_evidence where owner_id=$1
                    and attempt_id=$2 and playback_receipt_id=$3)""",
                claim.owner_id, claim.attempt_id, evidence.playback.receipt_id)
            if previous_receipt:
                raise AttentionError("attention_receipt_conflict", "A playback receipt cannot acquire a new evidence identity.")
        payload = await self.tasks.payload(connection, claim.owner_id, content, source_payload_ids=roots)
        await connection.execute("""/* voice_ledger:append */ insert into jarvis_attention_route_evidence
            (owner_id,evidence_id,attempt_id,voice_route_id,voice_route_generation,offer_id,output_id,
             evidence_state,observed_at,evidence_payload_id,evidence_digest,playback_receipt_id)
            values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)""",
            claim.owner_id, evidence.evidence_id, claim.attempt_id, evidence.route.route_id,
            evidence.route.route_generation, evidence.offer_id,
            evidence.output.output_id if evidence.output else None, evidence.state.value,
            evidence.observed_at, payload, signature,
            evidence.playback.receipt_id if evidence.playback else None)
        return True

    async def record(self, claim, evidence: DeliveryEvidence) -> EvidenceResult:
        if evidence.attribution != "explicit" or evidence.attempt is None:
            raise AttentionError("attention_unattributed_output", "Unattributed output is telemetry, not attempt evidence.")
        async with self._locked(claim) as (connection, row, task, attempt):
            # A digest match only identifies the event, not the caller's claim.
            # Check identity without loading payloads so a legitimate replay
            # remains readable after source/plan withdrawal.
            if (evidence.route.scope.owner_id != claim.owner_id
                    or evidence.route.scope.session_id != claim.session_id
                    or evidence.attempt.notification_intent_id != claim.intent_id
                    or evidence.attempt.attempt_id != claim.attempt_id
                    or evidence.attempt.attempt_generation != claim.generation
                    or evidence.attempt.source_payload_id != claim.source_payload_id
                    or evidence.attempt.source != self._source(row, evidence.attempt.source.generation)):
                raise AttentionError("attention_evidence_binding", "Evidence does not match the supplied claim.")
            # Idempotency remains readable after a payload is retracted. A
            # replay neither republishes content nor modifies terminal state.
            prior = await connection.fetchrow("""/* voice_ledger:evidence_key */
                select evidence_digest from jarvis_attention_route_evidence where owner_id=$1 and evidence_id=$2""",
                claim.owner_id, evidence.evidence_id)
            if prior is not None:
                if prior["evidence_digest"] != digest(evidence.model_dump(mode="json")):
                    raise AttentionError("attention_evidence_conflict", "This evidence ID was already used differently.")
                return EvidenceResult(False, attempt["phase"])
            plan = await self._plan(connection, claim, attempt)
            offer = plan.offer
            if (evidence.route != offer.route or evidence.offer_id != offer.offer_id
                    or evidence.attempt != offer.attempt or evidence.transport != plan.capabilities.transport):
                raise AttentionError("attention_evidence_binding", "Evidence does not match the original route and attempt.")
            _, problem = await self.attention._source(connection, row, task, lock=True)
            if problem:
                raise AttentionError("attention_source_unavailable", "No current owned evidence source is available.")
            await self._validate_source(connection, claim, row, offer.attempt.source)
            if evidence.observed_at < offer.offered_at:
                raise AttentionError("attention_evidence_time", "Evidence predates its offer.")
            submitted = attempt["submitted_at"] is not None
            if evidence.state not in {State.OFFERED, State.FAILED, State.UNKNOWN, State.INTERRUPTED} and not submitted:
                raise AttentionError("attention_submission_state", "Evidence cannot retroactively create submission.")
            if evidence.output is not None:
                if plan.capabilities.notification_output_attribution != Support.SUPPORTED:
                    raise AttentionError("attention_output_unsupported", "This route cannot attribute notification output.")
                bindings = await connection.fetch("""/* voice_ledger:output_binding */
                    select p.content from jarvis_attention_route_evidence e join jarvis_context_payloads p
                    on p.owner_id=e.owner_id and p.id=e.evidence_payload_id
                    where e.owner_id=$1 and e.attempt_id=$2 and e.output_id=$3
                      and e.evidence_state in ('output_started','output_finished')
                      and p.availability='available'""", claim.owner_id, claim.attempt_id, evidence.output.output_id)
                if any(json_object(item["content"])["output"] != evidence.output.model_dump(mode="json")
                       for item in bindings):
                    raise AttentionError("attention_output_conflict", "An observed output identity cannot change.")
                if evidence.state == State.CLIENT_PLAYBACK_OBSERVED:
                    if plan.capabilities.client_playback_evidence != Support.SUPPORTED:
                        raise AttentionError("attention_playback_unsupported", "This route has no client render evidence.")
                    if not bindings:
                        raise AttentionError("attention_output_unbound", "Playback needs a prior attributed output binding.")
            inserted = await self._append(connection, claim, evidence,
                roots=(claim.source_payload_id, attempt["plan_payload_id"]))
            settlement = settlement_for(attempt["phase"], submitted=submitted,
                                        state=evidence.state, reason=evidence.reason)
            if inserted and settlement is not None:
                await self._settle_locked(connection, claim, row, settlement)
            return EvidenceResult(inserted, settlement.attempt_phase if settlement else attempt["phase"])

    async def _settle_locked(self, connection, claim, row, settlement):
        await connection.execute("""/* voice_ledger:settle_attempt */ update jarvis_attention_attempts
            set phase=$3,reason=$4,ended_at=clock_timestamp() where owner_id=$1 and id=$2""",
            claim.owner_id, claim.attempt_id, settlement.attempt_phase, settlement.reason)
        if row["generation"] == claim.generation and row["state"] in {"claimed", "awaiting_receipt"}:
            await connection.execute("""/* voice_ledger:settle_intent */ update jarvis_attention_intents
                set state=$3,reason=$4,updated_at=clock_timestamp()
                where owner_id=$1 and id=$2 and generation=$5""",
                claim.owner_id, claim.intent_id, settlement.intent_state, settlement.reason, claim.generation)

    async def settle(self, claim, *, reason, interrupted=False):
        """Retire this native attempt without requiring still-available source prose."""
        if not isinstance(reason, str) or not 1 <= len(reason) <= 512:
            raise ValueError("A bounded settlement reason is required")
        async with self._locked(claim) as (connection, row, _task, attempt):
            if attempt.get("delivery_transport", "legacy_pcm_tts") == "legacy_pcm_tts":
                raise AttentionError("attention_legacy_attempt", "Use canonical legacy settlement for this attempt.")
            settlement = settlement_for(attempt["phase"], submitted=attempt["submitted_at"] is not None,
                state=State.INTERRUPTED if interrupted else State.UNKNOWN, reason=reason)
            if settlement:
                await self._settle_locked(connection, claim, row, settlement)

    async def recover(self, owner_id):
        """Reuse legacy-compatible lease recovery without a TTS/client prerequisite."""
        async with self.tasks.transaction(owner_id) as connection:
            await self.attention._recover(connection, owner_id)

