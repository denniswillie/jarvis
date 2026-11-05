"""One selected native speaker using canonical attention claims and evidence.

Provider callbacks only enqueue bounded evidence. Canonical persistence runs on
this worker, never in the transport reader. Completion remains unknown hearing.
"""
import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from uuid import UUID, uuid4

from app.attention.admission_diagnostics import record_admission
from app.attention.models import ACTIVE_PHASES, AttentionError
from app.attention.notification_ledger.repository import VoiceNotificationLedger
from app.voice_routes.contracts import DeliveryEvidence, DeliveryEvidenceState, InterruptSpeechCommand


LOGGER = logging.getLogger(__name__)


class NativeAttentionSink:
    def __init__(self, session, runtime, repository, admission, *, clock):
        self.session, self.runtime, self.repository = session, runtime, repository
        self.admission, self.clock = admission, clock
        self.ledger = VoiceNotificationLedger(repository, clock=clock)
        self._deliveries = {}
        self._events = asyncio.Queue(maxsize=64)
        self._closed = False
        self._worker = None
        self.failure = None

    def start(self):
        if self._worker is not None or self._closed:
            raise RuntimeError("Native notification sink already started or closed")
        self._worker = asyncio.create_task(self._consume(), name="native-attention-evidence")
        self.session._attention_delivery_factory = self.create_delivery
        self.session.try_attention = self.try_attention

    async def try_attention(self, repository, speech=None, *, worker_id, receipt_seconds):
        if self._closed or self.failure is not None or not self.runtime.ready:
            record_admission("route_not_ready")
            return None
        if repository is not self.repository or len(self._deliveries) >= 64:
            record_admission("sink_unavailable")
            return None
        return await self.admission(repository, None, worker_id=worker_id, receipt_seconds=receipt_seconds)

    def create_delivery(self, repository, speech, claim, session, *, receipt_seconds):
        if repository is not self.repository or session is not self.session or self._closed:
            raise AttentionError("attention_sink_changed", "The selected notification sink changed.")
        return NativeVoiceDelivery(self, claim, receipt_seconds=receipt_seconds)

    def register(self, delivery):
        offer = delivery.offer
        if self._closed or len(self._deliveries) >= 64 or offer.offer_id in self._deliveries:
            raise AttentionError("attention_sink_capacity", "No new native notification slot is available.")
        self._deliveries[offer.offer_id] = delivery

    def on_event(self, event):
        if not isinstance(event, DeliveryEvidence) or event.attribution != "explicit":
            return
        delivery = self._deliveries.get(event.offer_id)
        if (self._closed or delivery is None or event.route != self.runtime.request.route
                or event.attempt != delivery.offer.attempt):
            raise AttentionError("attention_evidence_binding", "No owned notification matches this event.")
        try:
            self._events.put_nowait((delivery, event))
        except asyncio.QueueFull:
            self.failure = "attention_evidence_overflow"
            raise AttentionError(self.failure, "Native evidence requires reconstruction.") from None

    @asynccontextmanager
    async def guard(self, offer):
        delivery = self._deliveries.get(offer.offer_id)
        self._guard_delivery(delivery, offer)
        ticket = delivery.ticket
        if ticket is None:
            raise AttentionError("attention_not_submitted", "The notification has no committed submission.")
        manager = (self.ledger.output_guard(delivery.claim, offer) if ticket._used
                   else self.ledger.guarded_transaction(ticket))
        async with manager as connection:
            self._guard_delivery(delivery, offer)
            yield connection

    def _guard_delivery(self, delivery, offer):
        # Readiness is not identity. In particular, a visual-context expiry
        # during canonical lock acquisition does not replace this offer.
        # Neither result authorizes output without the ledger/source guards.
        if delivery is None or not delivery.owns_offer(offer):
            raise AttentionError("attention_offer_not_current", "The owned notification is no longer current.")
        output_current = getattr(self.runtime, "notification_output_current", None)
        if (not self.runtime.ready
                and not (callable(output_current) and output_current(offer) is True)):
            raise AttentionError("attention_route_not_ready", "The notification route is not ready for output.")

    async def _consume(self):
        try:
            while True:
                delivery, event = await self._events.get()
                try:
                    result = await self.ledger.record(delivery.claim, event)
                    if result.attempt_phase not in ACTIVE_PHASES:
                        delivery.output_finished = event.state == DeliveryEvidenceState.OUTPUT_FINISHED
                        delivery._done.set()
                except AttentionError:
                    # A late event still belongs to its original attempt. It
                    # cannot acquire the newer speaker's interruption authority.
                    if self.session._attention_delivery is delivery:
                        await delivery.stop("attention_evidence_source_unavailable")
                finally:
                    self._events.task_done()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.failure = "attention_evidence_writer_failed"
            self.runtime._fail(self.failure)

    async def close(self):
        if self._closed:
            return
        self._closed = True
        deliveries = set(self._deliveries.values())
        selected = self.session._attention_delivery
        if isinstance(selected, NativeVoiceDelivery) and selected.sink is self:
            # A claimed delivery may be waiting before it has a registered offer.
            deliveries.add(selected)
        for delivery in deliveries:
            if delivery.task is not None and not delivery.task.done():
                await delivery.stop("native_attention_closed")
        await asyncio.gather(*(item.task for item in deliveries
            if item.task is not None), return_exceptions=True)
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)


class NativeVoiceDelivery:
    def __init__(self, sink, claim, *, receipt_seconds):
        self.sink, self.claim = sink, claim
        self.repository, self.session, self.runtime = sink.repository, sink.session, sink.runtime
        self.receipt_seconds = receipt_seconds
        self.task = None
        self.offer = self.ticket = None
        self.revoked = self.output_finished = self._started = False
        self._settling = False
        self.reason = "native_delivery_unconfirmed"
        self._done = asyncio.Event()
        self._clear_task = None
        self._route = self.runtime.request.route
        self._audio_session_id = self.session.context.audio_session_id
        self._readiness_diagnostics = 0

    def current(self):
        return self._ownership_rejection() is None and self.runtime.ready

    def _ownership_rejection(self):
        if self.revoked:
            return "delivery_revoked"
        if self.session._closed.is_set():
            return "session_closed"
        if self.sink._closed:
            return "sink_closed"
        if self.session._attention_delivery is not self:
            return "delivery_replaced"
        if self.runtime.request.route != self._route:
            return "route_changed"
        if self.session.context.audio_session_id != self._audio_session_id:
            return "audio_session_changed"
        return None

    def _readiness_diagnostic(self, stage, reason):
        # Fixed local predicates only: no source prose, provider errors or credentials.
        if self._readiness_diagnostics >= 8:
            return
        self._readiness_diagnostics += 1
        LOGGER.info("attention_delivery_readiness stage=%s reason=%s attempt_id=%s intent_id=%s generation=%s",
                    stage, reason, self.claim.attempt_id, self.claim.intent_id, self.claim.generation)

    async def _wait_until_ready(self, stage, deadline):
        """Wait only before exposure, on the same claim and original deadline.

        This is not canonical authority. Preparation/submission still acquire
        their existing source, owner, revision, lease and expiry guards. Once a
        submission may have happened, no readiness recovery or resend occurs.
        """
        waiting = False
        while True:
            rejection = self._ownership_rejection()
            if rejection is not None:
                self._readiness_diagnostic(stage, rejection)
                raise AttentionError("attention_not_current", "The selected route is no longer current.")
            if self.sink.clock() >= deadline:
                self._readiness_diagnostic(stage, "original_deadline_expired")
                raise AttentionError("attention_offer_expired", "The original notification window expired.")
            if self.runtime.ready:
                if waiting:
                    self._readiness_diagnostic(stage, "route_ready")
                return
            if not waiting:
                self._readiness_diagnostic(stage, "route_not_ready")
                waiting = True
            await asyncio.sleep(0.05)

    def accepts_offer(self, offer):
        return self.owns_offer(offer) and self.runtime.ready

    def owns_offer(self, offer):
        """In-memory ownership only; never a substitute for canonical guards."""
        return (not self.revoked and not self.session._closed.is_set()
                and not self.sink._closed and self.session._attention_delivery is self
                and self.offer == offer and self.ticket is not None
                and offer.route == self.runtime.request.route)

    async def _heartbeat(self):
        while not self._done.is_set() and self._ownership_rejection() is None:
            await asyncio.sleep(self.repository.lease_seconds / 3)
            if self._done.is_set() or self._ownership_rejection() is not None:
                return
            try:
                await self.repository.renew(self.claim)
            except AttentionError:
                if not self._done.is_set():
                    await self.stop("native_source_revoked")
                return

    async def run(self):
        self._started = True
        heartbeat = asyncio.create_task(self._heartbeat())
        try:
            async with asyncio.timeout(300 + self.receipt_seconds):
                deadline = self.sink.clock() + timedelta(seconds=300)
                await self._wait_until_ready("before_prepare", deadline)
                self.offer = await self.sink.ledger.prepare(self.claim, route=self.runtime.request.route,
                    audio_session_id=UUID(self.session.context.audio_session_id),
                    capabilities=self.runtime.adapter.capabilities, offer_id=uuid4(),
                    expires_at=deadline,
                    submission_boundary="realtime.conversation.item.create.authorized_notification_offer")
                self.sink.register(self)
                await self._wait_until_ready("before_submission", self.offer.expires_at)
                self.ticket = await self.sink.ledger.commit_submission(self.claim, self.offer)
                if self.ticket is None:
                    raise AttentionError("attention_already_submitted", "The original offer must not be retried.")
                result = await self.runtime.offer_notification(self.offer)
                # The event pump may already have enqueued this exact evidence.
                # Canonical evidence IDs make this replay idempotent, not a resend.
                self.sink.on_event(result)
                await self._done.wait()
        except asyncio.CancelledError:
            if not self.revoked:
                self.reason = "native_delivery_cancelled"
        except TimeoutError:
            self.reason = "native_delivery_timeout"
        except AttentionError as error:
            if not self.revoked:
                self.reason = error.code
        except Exception:
            self.reason = "native_delivery_failed"
        finally:
            self._settling = True
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            if not self.output_finished:
                self.revoked = True
                await self._clear()
                with suppress(Exception):
                    settle = self.sink.ledger.settle if self.offer is not None else self.repository.settle
                    await settle(self.claim, reason=self.reason, interrupted=self.reason == "user_interrupted")
            elif self._clear_task is not None:
                # Completion can race a source-withdrawal fence. Do not release
                # the output owner while that old fence is still queued.
                await asyncio.shield(self._clear_task)
            await self.session.release_attention(self)

    async def _clear(self):
        await asyncio.shield(self._request_clear())

    def _request_clear(self):
        if self._clear_task is None:
            self._clear_task = asyncio.create_task(self._send_clear())
        return self._clear_task

    async def _send_clear(self):
        if self.session._closed.is_set() or self.session._attention_delivery is not self:
            return
        try:
            command = InterruptSpeechCommand(route=self.runtime.request.route,
                command_id=uuid4(), requested_at=self.sink.clock(), reason=self.reason)
            fence = getattr(self.runtime, "fence_notification_output", None)
            receipt = (await fence(command, delivery=self) if fence is not None
                       else await self.runtime.coordinator.interrupt_speech(command))
            if receipt.status != "completed" or receipt.output_fenced is not True:
                self.session._mark_closing("native_attention_fence_unconfirmed")
        except Exception:
            self.session._mark_closing("native_attention_fence_unconfirmed")

    def request_stop(self, reason):
        """Revoke synchronously; the delivery retains ownership through its fence."""
        was_revoked = self.revoked
        self.revoked, self.reason = True, reason
        if (not was_revoked and self._started and not self._settling and self.task is not None
                and self.task is not asyncio.current_task() and not self.task.done()):
            self.task.cancel()
        fence = self._request_clear()
        controls = getattr(self.session, "_response_controls", None)
        if controls is not None:
            # Source withdrawal also enters here without a user cancel event.
            # Its old clear must fence newer response writes just as barge-in does.
            controls.hold_fence(str(id(self)), fence)
        return fence

    async def stop(self, reason):
        await asyncio.shield(self.request_stop(reason))
