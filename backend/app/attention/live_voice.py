"""Speech-eligible Live offers with canonical claims and unknown hearing.

Live attributes append acknowledgments, not generated audio, to an offer. Keep
that distinction: release the delivery lease after a correlated context receipt,
without cancelling the continuous conversation or claiming the user heard it.
The provider session retains source guards and expiry fences after settlement.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
import json
from uuid import UUID, uuid4

from app.attention.models import AttentionError
from app.attention.notification_ledger.repository import VoiceNotificationLedger
from app.attention.notifications import NotificationOutput
from app.voice_routes.contracts import (
    DeliveryEvidence, DeliveryEvidenceState as Evidence, InterruptSpeechCommand,
    RouteOperation, Support,
)


def commentary_fits(text: str) -> bool:
    """Conservative encoded-byte bound, not a tokenizer or billing estimate."""
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        return False
    document = json.dumps({"kind": "authorized_speakable_result", "text": text},
                          ensure_ascii=True, separators=(",", ":"))
    return len(document.encode("utf-8")) <= 500


class LiveAttentionSink:
    """One session-owned offer; all DB work stays on the attention worker lane.

The caller supplies existing eligibility policy and the selected adapter's
submit method. Neither callback is an authorization substitute for the ledger.
This sink never creates intents, generates speech, or owns background tasks.
"""

    def __init__(self, session, repository, *, clock, admission, submit_offer):
        if not callable(admission) or not callable(submit_offer):
            raise TypeError("An eligibility policy and selected submit method are required")
        self.session, self.repository, self.clock = session, repository, clock
        self.admission, self.submit_offer = admission, submit_offer
        self.ledger = VoiceNotificationLedger(repository, clock=clock)
        self._admission_lock = asyncio.Lock()
        self._delivery = None
        self._started = self._closed = False
        self.failure = None

    def start(self):
        if self._started or self._closed:
            raise RuntimeError("Live notification sink already started or closed")
        capabilities = self.session.adapter.capabilities
        operations = {item.operation: item.support for item in capabilities.operations}
        if (operations.get(RouteOperation.OFFER_NOTIFICATION) != Support.SUPPORTED
                or capabilities.notification_output_attribution != Support.UNSUPPORTED):
            raise ValueError("This sink requires speech offers with explicitly unavailable audio attribution")
        self._started = True
        self.session.try_attention = self.try_attention

    def available(self):
        return (self._started and not self._closed and self.failure is None
                and self.session.ready and not self.session._closed.is_set())

    def _admit(self, row, output):
        if not self.available():
            return "live_notification_route_unavailable"
        text = ("".join(output.speech_texts) if isinstance(output, NotificationOutput)
                else "".join(chunk.text for chunk in output.content.speech))
        if not commentary_fits(text):
            # Do not acquire a claim then truncate its exact source-bound text.
            return "live_notification_append_budget"
        return self.admission(row, output)

    async def try_attention(self, repository, speech=None, *, worker_id, receipt_seconds):
        if not 0 < receipt_seconds <= 300:
            raise ValueError("Live notification receipt deadline must be bounded")
        async with self._admission_lock:
            if (repository is not self.repository or not self.available()
                    or self._delivery is not None or self.session._attention_delivery is not None):
                return None
            claim = await repository.claim_next(self.session.context.owner_id,
                self.session.context.jarvis_session_id, worker_id, admission=self._admit)
            if claim is None:
                return None
            # Disconnect can race the database claim without waiting on its lock.
            if not self.available():
                await repository.settle(claim, reason="live_notification_route_unavailable")
                return None
            delivery = LiveNotificationDelivery(self, claim, receipt_seconds=receipt_seconds)
            self._delivery = self.session._attention_delivery = delivery
            delivery.task = asyncio.create_task(delivery.run(), name="live-notification-delivery")
            return delivery

    @asynccontextmanager
    async def guard(self, offer):
        delivery = self._delivery
        if delivery is None or not delivery.accepts_offer(offer) or delivery.ticket is None:
            raise AttentionError("attention_offer_not_current", "No current Live offer owns this write.")
        # Live sends exactly one commentary append. Reusing an already-consumed
        # ticket is forbidden, even when a provider acknowledgment was lost.
        async with self.ledger.guarded_transaction(delivery.ticket) as connection:
            if not delivery.accepts_offer(offer):
                raise AttentionError("attention_offer_not_current", "The Live offer was retired.")
            sources = self.session._source_context
            roots = (offer.attempt.source_payload_id,)
            async with sources.context_guard(connection=connection, extra_sources=roots):
                sources.observe(roots)
                yield connection

    def on_event(self, event):
        """Bounded enqueue only; called by the sole provider-event consumer."""
        if not isinstance(event, DeliveryEvidence) or event.attribution != "explicit":
            return
        delivery = self._delivery
        if delivery is None or delivery.offer is None or event.offer_id != delivery.offer.offer_id:
            return  # An old attempt never acquires the next offer's authority.
        if (event.route != delivery.offer.route or event.attempt != delivery.offer.attempt
                or event.transport != self.session.adapter.capabilities.transport
                or event.output is not None or event.playback is not None):
            self.failure = "live_notification_evidence_mismatch"
            self.session._mark_closing(self.failure)
            raise AttentionError(self.failure, "Live cannot attribute this output to the offer.")
        try:
            delivery.events.put_nowait(event)
        except asyncio.QueueFull:
            self.failure = "live_notification_evidence_overflow"
            self.session._mark_closing(self.failure)
            raise AttentionError(self.failure, "Live notification evidence requires recovery.") from None

    async def close(self):
        self._closed = True
        if getattr(self.session, "try_attention", None) == self.try_attention:
            self.session.try_attention = None
        delivery = self._delivery
        if delivery is not None:
            await delivery.stop("live_notification_sink_closed")
            if delivery.task is not asyncio.current_task():
                await asyncio.gather(delivery.task, return_exceptions=True)


class LiveNotificationDelivery:
    def __init__(self, sink, claim, *, receipt_seconds):
        self.sink, self.claim, self.session = sink, claim, sink.session
        self.receipt_seconds = receipt_seconds
        self.task = self.offer = self.ticket = self._fence_task = None
        self.events = asyncio.Queue(maxsize=64)
        self.revoked = self.provider_received = self._started = False
        self._settling = False
        self.reason = "live_notification_delivery_unconfirmed"

    def current(self):
        return (not self.revoked and self.sink.available()
                and self.sink._delivery is self and self.session._attention_delivery is self)

    def accepts_offer(self, offer):
        return self.current() and self.offer == offer

    async def _heartbeat(self):
        while self.current():
            await asyncio.sleep(self.sink.repository.lease_seconds / 3)
            if not self.current():
                return
            try:
                await self.sink.repository.renew(self.claim)
            except Exception:
                await self.stop("live_notification_source_unavailable")
                return

    async def run(self):
        self._started = True
        heartbeat = asyncio.create_task(self._heartbeat(), name="live-notification-lease")
        try:
            async with asyncio.timeout(self.receipt_seconds):
                if not self.current():
                    raise AttentionError("attention_not_current", "The Live route is not current.")
                self.offer = await self.sink.ledger.prepare(self.claim,
                    route=self.session.request.route,
                    audio_session_id=UUID(self.session.context.audio_session_id),
                    capabilities=self.session.adapter.capabilities, offer_id=uuid4(),
                    expires_at=self.sink.clock() + timedelta(seconds=300),
                    submission_boundary="live.session.commentary.append.authorized_notification_offer")
                if not commentary_fits(self.offer.text):
                    raise AttentionError("live_notification_append_budget", "The exact offer exceeds the Live budget.")
                self.ticket = await self.sink.ledger.commit_submission(self.claim, self.offer)
                if self.ticket is None:
                    raise AttentionError("attention_already_submitted", "A submitted offer must not be retried.")
                result = await self.sink.submit_offer(self.offer)
                self.sink.on_event(result)
                while self.current():
                    event = await self.events.get()
                    try:
                        await self.sink.ledger.record(self.claim, event)
                        if event.state == Evidence.PROVIDER_RECEIVED:
                            # Only the correlated commentary ACK establishes
                            # injection. It establishes no speech or hearing.
                            await self.sink.ledger.settle(self.claim,
                                reason="live_commentary_received_hearing_unknown")
                            self.provider_received = True
                            return
                        if event.state in {Evidence.UNKNOWN, Evidence.FAILED, Evidence.INTERRUPTED}:
                            self.reason = event.reason or "live_notification_delivery_unconfirmed"
                            return
                    finally:
                        self.events.task_done()
        except asyncio.CancelledError:
            if not self.revoked:
                self.reason = "live_notification_cancelled"
        except TimeoutError:
            self.reason = "live_notification_receipt_timeout"
        except AttentionError as error:
            if not self.revoked:
                self.reason = error.code
        except Exception:
            self.reason = "live_notification_delivery_failed"
        finally:
            # Stop requests may arrive from provider failure, source retirement,
            # and session shutdown concurrently. Once cleanup begins, none may
            # cancel its first await and strand the canonical claim/owner refs.
            self._settling = True
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            if not self.provider_received:
                self.revoked = True
                if self.ticket is not None and self.ticket._used:
                    await self._fence()
                try:
                    settle = self.sink.ledger.settle if self.offer is not None else self.sink.repository.settle
                    await settle(self.claim, reason=self.reason,
                                 interrupted=self.reason == "user_interrupted")
                except Exception:
                    self.sink.failure = "live_notification_settlement_unconfirmed"
                    self.session._mark_closing(self.sink.failure)
            if self.sink._delivery is self:
                self.sink._delivery = None
            if self.session._attention_delivery is self:
                self.session._attention_delivery = None

    async def _fence(self):
        if self._fence_task is None:
            self._fence_task = asyncio.create_task(self._send_fence(), name="live-notification-fence")
        await asyncio.shield(self._fence_task)

    async def _send_fence(self):
        if self.session._closed.is_set() or self.session._attention_delivery is not self:
            return
        try:
            async with asyncio.timeout(5):
                receipt = await self.session.coordinator.interrupt_speech(InterruptSpeechCommand(
                    route=self.session.request.route, command_id=uuid4(),
                    requested_at=self.sink.clock(), reason=self.reason))
                if receipt.status != "completed" or receipt.output_fenced is not True:
                    self.sink.failure = "live_notification_fence_unconfirmed"
        except Exception:
            self.sink.failure = "live_notification_fence_unconfirmed"
        finally:
            # This is explicit retirement, never natural conversational barge-in.
            self.session._mark_closing(self.sink.failure or "live_notification_requires_replacement")

    async def stop(self, reason):
        first_stop = not self.revoked
        self.revoked = True
        if first_stop and not self._settling:
            self.reason = reason
        # Cancelling a task before its coroutine enters skips its finally block.
        # Let that task enter, observe revocation and settle the unsubmitted
        # claim. close() still joins it before returning ownership.
        # Only the first stop may cancel running work, never settlement. This
        # also preserves a pre-entry stop when close() arrives after entry.
        if (first_stop and not self._settling and self._started and self.task is not None
                and self.task is not asyncio.current_task() and not self.task.done()):
            self.task.cancel()
        if not self._settling and not self.provider_received and self.ticket is not None and self.ticket._used:
            await self._fence()
