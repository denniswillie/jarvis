from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from math import isfinite
from typing import Any
from uuid import UUID

from app.images import ImageReference, ImageUnavailableError, ResolvedImage
from app.perception.frame_window import RecentVisualFrameWindow
from app.visual_memory.validation import VisualFrameValidationError
from app.realtime.openai_client import (
    OpenAIRealtimeConnection, build_ambient_image_event, build_image_delete_event,
    build_requested_image_event,
)
from app.realtime.visual_context import (
    ProviderImageStatus, VisualContextNotReadyError, VisualContextState,
    VisualFrameMetadata, VisualResponseSnapshot, VisualSyncDeadline,
    VisualAvailability, VisualAvailabilityStatus, VisualAvailabilityReason,
    ProviderImageItem, RequestedImageDeliveryError,
)


LOGGER = logging.getLogger(__name__)
VISUAL_POLL_SECONDS = 0.25
VISUAL_WRITE_TIMEOUT_SECONDS = 0.5
VISUAL_ACK_TIMEOUT_SECONDS = 2.0
RECENT_VISUAL_IDENTITIES = 32
MAX_UNSETTLED_VISUAL_RESPONSES = 32


@dataclass
class _ImageOperation:
    kind: str
    item_id: str
    request_id: str
    started_monotonic: float
    write_started: bool = False
    cancelled: bool = False
    provider_call_id: str | None = None
    image: ResolvedImage | None = field(default=None, repr=False)
    failure_reason: str | None = None


@dataclass
class _RequestedImageWork:
    reference: ImageReference
    turn_generation: int
    is_request_active: Callable[[], bool]
    task: asyncio.Task[Any] | None


class RealtimeVisualCoordinator:
    """One bounded image ledger and writer for passive and requested evidence.

    Metadata transitions are synchronous on the session event loop. Neither the
    ingestion callback nor the receive loop waits for network acknowledgements.
    Response preparation fences only image mutations, never audio transport.
    """

    def __init__(
        self, *, state: VisualContextState, frame_window: RecentVisualFrameWindow,
        provider: OpenAIRealtimeConnection, max_image_bytes: int,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
        ack_timeout_seconds: float = VISUAL_ACK_TIMEOUT_SECONDS,
        write_timeout_seconds: float = VISUAL_WRITE_TIMEOUT_SECONDS,
        image_resolution_timeout_seconds: float = 5.0,
        on_image_delivery: Callable[[ImageReference | None, str, str], None] | None = None,
    ) -> None:
        if not isfinite(image_resolution_timeout_seconds) or image_resolution_timeout_seconds <= 0:
            raise ValueError("Image resolution timeout must be finite and positive.")
        self.image_resolution_timeout_seconds = image_resolution_timeout_seconds
        self.state = state
        self._window = frame_window
        self._provider = provider
        self._max_image_bytes = max_image_bytes
        self._on_image_delivery = on_image_delivery
        self._image_references: OrderedDict[str, ImageReference | None] = OrderedDict()
        self._now = now
        self._monotonic = monotonic
        self._ack_timeout = ack_timeout_seconds
        self._write_timeout = write_timeout_seconds
        self._wake = asyncio.Event()
        self._revision = 0
        self._closed = False
        self._worker: asyncio.Task[None] | None = None
        self._write_task: asyncio.Task[None] | None = None
        self._operation: _ImageOperation | None = None
        self._fence_generation: UUID | None = None
        self._preparation: VisualResponseSnapshot | None = None
        self._response_request_id: str | None = None
        self._response_id: str | None = None
        self._prior_responses: dict[str, tuple[VisualResponseSnapshot, str | None]] = {}
        self._recent_cutoffs: deque[dict[str, object]] = deque(maxlen=8)
        self._last_dispatch = float("-inf")
        self._retry_after = float("-inf")
        self._create_times: dict[str, float] = {}
        self._recent_items: deque[str] = deque(maxlen=RECENT_VISUAL_IDENTITIES)
        self._recent_requests: deque[str] = deque(maxlen=RECENT_VISUAL_IDENTITIES)
        self._delivery_failure: str | None = None
        self._disable_creates = False
        self._replacement_count = 0
        self._last_ack_latency: float | None = None
        self._requested_work: dict[str, _RequestedImageWork] = {}
        self._retired_image_calls: deque[tuple[str, str]] = deque(maxlen=RECENT_VISUAL_IDENTITIES)

    def start(self) -> None:
        if self._closed or self._worker is not None or not self.state.policy.enabled:
            return
        self._worker = asyncio.create_task(self._run())
        self.notify_frame_available()

    def notify_frame_available(self) -> None:
        if not self._closed:
            self._revision += 1
            self._wake.set()

    def is_passive_item(self, item_id: str) -> bool:
        return item_id in self._recent_items or any(
            item.item_id == item_id for item in self.state.items
        )

    @property
    def camera_stream_active(self) -> bool:
        key = self.state.connection
        return self._window.stream_lease(key.owner_id, key.session_id) is not None

    def diagnostics(self) -> dict[str, object]:
        return {
            **self.state.to_metadata(now=self._now(), monotonic_now=self._monotonic()),
            "failure_reason": self._delivery_failure,
            "queue_replacement_count": self._replacement_count,
            "last_create_ack_latency_seconds": self._last_ack_latency,
            "image_fenced": self._fence_generation is not None or self._prior_ack_pending(),
            "unsettled_prior_responses": len(self._prior_responses),
            "recent_response_cutoffs": list(self._recent_cutoffs),
            "requested_image_count": len(self._requested_work),
        }

    def _retired_image_reason(self, provider_call_id: str) -> str | None:
        return next((reason for call_id, reason in reversed(self._retired_image_calls) if call_id == provider_call_id), None)

    def _validate_requested_reference(self, reference: ImageReference) -> None:
        key = self.state.connection
        if self._closed:
            raise RequestedImageDeliveryError("connection_closed", "The image connection is closed.")
        if reference.owner_id != key.owner_id:
            raise ImageUnavailableError("frame_unavailable", "The selected image is unavailable.")
        now, monotonic_now = self._now(), self._monotonic()
        if reference.raw_image_expires_at is not None and now >= reference.raw_image_expires_at:
            raise ImageUnavailableError("frame_unavailable", "The selected image has expired.")
        if reference.source == "historical":
            return
        if reference.session_id != key.session_id:
            raise ImageUnavailableError("frame_unavailable", "The selected image is unavailable.")
        lease = self._window.stream_lease(key.owner_id, key.session_id)
        if lease is None or (lease.visual_session_id, lease.connection_id) != (
            reference.visual_session_id, reference.stream_connection_id,
        ):
            raise ImageUnavailableError("source_changed", "The camera stream has changed.")
        sample = self._window.get_sample(key.owner_id, key.session_id, frame_id=reference.frame_id)
        if sample is None or (
            sample.frame.captured_at != reference.captured_at
            or sample.frame.server_received_at != reference.server_received_at
            or sample.frame.visual_session_id != reference.visual_session_id
            or sample.frame.frame_sequence != reference.frame_sequence
            or sample.sampled_at != reference.sampled_at
            or sample.sampled_monotonic != reference.sampled_monotonic
        ):
            raise ImageUnavailableError("frame_unavailable", "The selected image is no longer available.")
        freshness = reference.freshness(
            now=now, monotonic_now=monotonic_now,
            max_age_seconds=self.state.policy.max_context_age_seconds,
        )
        if freshness.status != "fresh":
            code = "stale_frame" if freshness.status == "stale" else "clock_uncertain"
            raise ImageUnavailableError(code, "The selected image cannot establish a fresh current view.")
        if reference.raw_image_expires_at is not None and (
            monotonic_now - sample.sampled_monotonic
            >= (reference.raw_image_expires_at - sample.sampled_at).total_seconds()
        ):
            raise ImageUnavailableError("frame_unavailable", "The selected image has expired.")

    def _validate_requested_work(self, provider_call_id: str, work: _RequestedImageWork) -> None:
        if not work.is_request_active():
            raise RequestedImageDeliveryError("request_cancelled", "The image request is no longer active.")
        self._validate_requested_reference(work.reference)
        pin = self.state.requested_pin(provider_call_id)
        if pin is not None:
            if self._monotonic() < pin.admitted_monotonic:
                raise ImageUnavailableError("clock_uncertain", "The image delivery clock is uncertain.")
            if pin.expired(now=self._now(), monotonic_now=self._monotonic()):
                raise ImageUnavailableError("frame_unavailable", "The selected image has expired.")

    async def deliver_requested_image(
        self,
        image: ResolvedImage,
        *,
        provider_call_id: str,
        turn_generation: int,
        is_request_active: Callable[[], bool],
    ) -> ProviderImageItem:
        """Await outside the provider receiver and speech lock; success requires ACK.

        The caller supplies a synchronous cancellation/authorization guard. Domain
        resolution happens before this call; pixels are held only during its
        bounded write/ACK operation, not in the ledger or completed-call cache.
        """
        self._validate_requested_reference(image.reference)
        if not provider_call_id.strip() or not isinstance(turn_generation, int) or isinstance(turn_generation, bool) or turn_generation < 0:
            raise ValueError("Requested images require a tool call and turn generation.")
        if not image.image_bytes or len(image.image_bytes) > self._max_image_bytes:
            raise ImageUnavailableError("invalid_image", "The selected image exceeds the delivery limits.")
        previous = self._requested_work.get(provider_call_id)
        if previous is not None:
            if previous.reference != image.reference or previous.turn_generation != turn_generation:
                raise RequestedImageDeliveryError("request_conflict", "The image request identity has changed.")
            self._validate_requested_work(provider_call_id, previous)
            existing = self.state.requested_item(provider_call_id)
            if previous.task is None and existing is not None and existing.status == ProviderImageStatus.ACKNOWLEDGED:
                return existing
            raise RequestedImageDeliveryError("request_in_progress", "This image request is already being processed.")
        retired_reason = self._retired_image_reason(provider_call_id)
        if retired_reason is not None:
            raise RequestedImageDeliveryError(retired_reason, "This image request has already ended.")
        if len(self._requested_work) >= self.state.policy.max_image_items:
            raise RequestedImageDeliveryError("image_capacity", "Too many image requests are pending.")
        work = _RequestedImageWork(image.reference, turn_generation, is_request_active, asyncio.current_task())
        self._requested_work[provider_call_id] = work
        operation: _ImageOperation | None = None
        delivered = False
        failure_reason = "request_cancelled"
        try:
            async with asyncio.timeout(self.state.policy.sync_timeout_seconds):
                while True:
                    revision = self._revision
                    if self._requested_work.get(provider_call_id) is not work:
                        raise RequestedImageDeliveryError(
                            self._retired_image_reason(provider_call_id) or "request_cancelled",
                            "The image request is no longer active.",
                        )
                    self._validate_requested_work(provider_call_id, work)
                    self._refresh_local()
                    if operation is not None and operation.failure_reason is not None:
                        raise RequestedImageDeliveryError(operation.failure_reason, "The image could not be attached.")
                    item = self.state.requested_item(provider_call_id)
                    if item is not None and item.status == ProviderImageStatus.ACKNOWLEDGED:
                        delivered = True
                        return item
                    if self._disable_creates:
                        raise RequestedImageDeliveryError("provider_rejected_image_input", "Image input is unavailable on this connection.")
                    writing = self._write_task is not None and not self._write_task.done()
                    if (
                        item is None and not writing and self._operation is None
                        and self._fence_generation is None and not self._prior_ack_pending()
                    ):
                        current_frame = None
                        if image.reference.source == "current":
                            sample = self._window.get_sample(
                                self.state.connection.owner_id, self.state.connection.session_id,
                                frame_id=image.reference.frame_id,
                            )
                            if sample is None:
                                raise ImageUnavailableError("frame_unavailable", "The selected image is unavailable.")
                            current_frame = VisualFrameMetadata.from_sensory_frame(
                                sample.frame, connection=self.state.connection,
                                sampled_at=sample.sampled_at, sampled_monotonic=sample.sampled_monotonic,
                            )
                        item = self.state.reserve_requested_create(
                            reference=image.reference, current_frame=current_frame,
                            provider_call_id=provider_call_id, turn_generation=turn_generation,
                            now=self._now(), monotonic_now=self._monotonic(),
                        )
                        if item is not None:
                            if item.status == ProviderImageStatus.ACKNOWLEDGED:
                                delivered = True
                                return item
                            operation = _ImageOperation(
                                "create", item.item_id, item.create_request_id, self._monotonic(),
                                provider_call_id=provider_call_id, image=image,
                            )
                            self._recent_items.append(item.item_id)
                            self._schedule(operation)
                    self._tick()
                    await self._wait_for_change(revision, VISUAL_POLL_SECONDS)
        except TimeoutError as error:
            failure_reason = "image_delivery_timeout"
            raise RequestedImageDeliveryError(failure_reason, "The image attachment could not be confirmed in time.") from error
        except (ImageUnavailableError, RequestedImageDeliveryError) as error:
            failure_reason = error.code
            raise
        finally:
            if delivered:
                work.task = None
            else:
                self.cancel_requested_image(
                    provider_call_id=provider_call_id, turn_generation=turn_generation,
                    reason=failure_reason,
                )
            # Acknowledged inputs and unknown writes both need eventual cleanup.
            if not self._closed and self._worker is None and self.state.items:
                self._worker = asyncio.create_task(self._run())
            self.notify_frame_available()

    def cancel_requested_image(
        self, *, provider_call_id: str, turn_generation: int, reason: str = "request_cancelled",
    ) -> bool:
        work = self._requested_work.get(provider_call_id)
        if work is None or work.turn_generation != turn_generation:
            return False
        del self._requested_work[provider_call_id]
        self._retired_image_calls.append((provider_call_id, reason))
        operation = self._operation
        if operation is not None and operation.provider_call_id == provider_call_id:
            operation.cancelled = True
            operation.failure_reason = reason
            if self._write_task is not None and not self._write_task.done():
                self._write_task.cancel()
            self._write_failed(operation)
        self.state.release_requested_image(provider_call_id=provider_call_id, turn_generation=turn_generation)
        if work.task is not None and work.task is not asyncio.current_task() and not work.task.done():
            work.task.cancel()
        self.notify_frame_available()
        return True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.state.close()
        self._wake.set()
        tasks = [
            task for task in (
                self._worker, self._write_task,
                *(work.task for work in self._requested_work.values()),
            )
            if task is not None and task is not asyncio.current_task()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._operation = None
        self._preparation = None
        self._fence_generation = None
        self._response_request_id = self._response_id = None
        self._prior_responses.clear()
        self._recent_cutoffs.clear()
        self._create_times.clear()
        self._recent_items.clear()
        self._recent_requests.clear()
        self._requested_work.clear()
        self._retired_image_calls.clear()

    async def _run(self) -> None:
        while not self._closed:
            self._wake.clear()
            try:
                self._tick()
            except Exception:
                # Do not include provider errors or media in routine diagnostics.
                self._delivery_failure = "visual_worker_failed"
                self._disable_creates = True
                LOGGER.warning("passive visual worker failed")
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=VISUAL_POLL_SECONDS)
            except TimeoutError:
                pass

    def _refresh_local(self) -> None:
        key = self.state.connection
        frame = self._window.latest(
            key.owner_id, key.session_id,
            max_age_seconds=self.state.policy.max_context_age_seconds,
        )
        stream = self._window.active_stream(key.owner_id, key.session_id)
        if stream is None and frame is not None:
            stream = frame.visual_session_id
        if stream != self.state.visual_session_id:
            self.state.activate_visual_stream(stream)
        if frame is None or frame.visual_session_id != stream:
            return
        metadata = VisualFrameMetadata.from_sensory_frame(
            frame, connection=key, sampled_at=self._now(),
            sampled_monotonic=self._monotonic(),
        )
        pending = self.state.pending_candidate(now=self._now(), monotonic_now=self._monotonic())
        if self.state.offer_frame(metadata) and pending is not None:
            self._replacement_count += 1

    def _prior_ack_pending(self) -> bool:
        return any(response_id is None for _, response_id in self._prior_responses.values())

    def _tick(self, *, preparing: bool = False) -> None:
        if self._closed:
            return
        self._refresh_local()
        for call_id, work in tuple(self._requested_work.items()):
            try:
                self._validate_requested_work(call_id, work)
            except (ImageUnavailableError, RequestedImageDeliveryError) as error:
                self.cancel_requested_image(provider_call_id=call_id, turn_generation=work.turn_generation, reason=error.code)
        operation = self._operation
        if operation is not None and self._monotonic() - operation.started_monotonic >= self._ack_timeout:
            self._mark_unknown(operation, "image_ack_timeout")
        if self._prior_ack_pending() or (self._fence_generation is not None and not preparing):
            return
        if self._write_task is not None and not self._write_task.done():
            return
        if self._operation is not None or self._monotonic() < self._retry_after:
            return
        if any(item.status in {
            ProviderImageStatus.CREATE_UNKNOWN, ProviderImageStatus.DELETE_UNKNOWN,
        } for item in self.state.items):
            return
        retiring = self.state.retirable_item_ids(now=self._now(), monotonic_now=self._monotonic())
        if retiring:
            item_id = retiring[0]
            request_id = self.state.begin_delete(
                connection_epoch=self.state.connection.connection_epoch, item_id=item_id,
            )
            if request_id is not None:
                self._schedule(_ImageOperation("delete", item_id, request_id, self._monotonic()))
            return
        if self._requested_work or self._disable_creates or self._monotonic() - self._last_dispatch < self.state.policy.min_interval_seconds:
            return
        item = self.state.reserve_create(now=self._now(), monotonic_now=self._monotonic())
        if item is not None:
            self._recent_items.append(item.item_id)
            self._schedule(_ImageOperation(
                "create", item.item_id, item.create_request_id, self._monotonic(),
            ))

    def _schedule(self, operation: _ImageOperation) -> None:
        # The ledger/fence sees an operation before its task can enter the writer.
        self._operation = operation
        self._recent_requests.append(operation.request_id)
        self._write_task = asyncio.create_task(self._write(operation))

    def _remember_image_reference(self, operation: _ImageOperation, *, frame=None) -> None:
        if self._on_image_delivery is None:
            return
        reference = operation.image.reference if operation.image is not None else None
        if reference is None and frame is not None:
            key = self.state.connection
            sample = self._window.get_sample(key.owner_id, key.session_id, frame_id=frame.frame_id)
            lease = self._window.stream_lease(key.owner_id, key.session_id)
            if sample is not None and lease is not None and lease.visual_session_id == frame.visual_session_id:
                try:
                    reference = ImageReference(
                        frame_id=frame.frame_id, owner_id=key.owner_id, session_id=key.session_id,
                        source="current", captured_at=frame.captured_at, content_type=frame.content_type,
                        width=frame.width, height=frame.height, raw_image_expires_at=frame.raw_image_expires_at,
                        visual_session_id=frame.visual_session_id, frame_sequence=frame.frame_sequence,
                        stream_connection_id=lease.connection_id, server_received_at=frame.server_received_at,
                        sampled_at=sample.sampled_at, sampled_monotonic=sample.sampled_monotonic,
                    )
                except (TypeError, ValueError):
                    reference = None
        self._image_references[operation.item_id] = reference
        self._image_references.move_to_end(operation.item_id)
        while len(self._image_references) > RECENT_VISUAL_IDENTITIES:
            self._image_references.popitem(last=False)

    def _capture_image_delivery(self, item_id: str, state: str) -> None:
        if self._on_image_delivery is None or item_id not in self._image_references:
            return
        try:
            self._on_image_delivery(self._image_references[item_id], state, item_id)
        except Exception:
            # Optional evidence capture must not alter media admission or retry it.
            LOGGER.warning("image context capture callback failed")

    def _payload_at_write(self, operation: _ImageOperation) -> dict[str, Any] | None:
        if self._closed or operation.cancelled:
            return None
        if operation.kind == "delete":
            operation.write_started = True
            return build_image_delete_event(operation.item_id, operation.request_id)
        if operation.provider_call_id is not None:
            work = self._requested_work.get(operation.provider_call_id)
            if work is None or operation.image is None:
                return None
            try:
                self._validate_requested_work(operation.provider_call_id, work)
                if self._fence_generation is not None or self._prior_ack_pending():
                    operation.failure_reason = "image_context_fenced"
                    return None
                item = self.state.requested_item(operation.provider_call_id)
                if item is None:
                    return None
                if work.reference.source == "current":
                    sample = self._window.get_sample(
                        self.state.connection.owner_id, self.state.connection.session_id,
                        frame_id=work.reference.frame_id,
                    )
                    if sample is None or sample.frame.image_bytes != operation.image.image_bytes:
                        raise ImageUnavailableError("frame_unavailable", "The selected image has changed.")
                payload = build_requested_image_event(item, operation.image)
                if not self.state.mark_submitted(
                    item_id=item.item_id, now=self._now(), monotonic_now=self._monotonic(),
                ):
                    return None
                operation.write_started = True
                self._remember_image_reference(operation)
                self._create_times[item.item_id] = self._monotonic()
                return payload
            except (ImageUnavailableError, RequestedImageDeliveryError) as error:
                operation.failure_reason = error.code
                return None
            except (ValueError, VisualFrameValidationError):
                operation.failure_reason = "invalid_image"
                return None
        # Re-read after waiting for the common writer; never retain queued pixels.
        self._refresh_local()
        key = self.state.connection
        frame = self._window.latest(
            key.owner_id, key.session_id,
            max_age_seconds=self.state.policy.max_context_age_seconds,
        )
        item = next((item for item in self.state.items if item.item_id == operation.item_id), None)
        if item is None:
            return None
        if (
            frame is None or frame.frame_id != item.frame.frame_id
            or len(frame.image_bytes) > self._max_image_bytes
            or frame.content_type != "image/jpeg"
        ):
            self.state.cancel_reserved(connection_epoch=key.connection_epoch, item_id=item.item_id)
            if frame is not None and (
                len(frame.image_bytes) > self._max_image_bytes or frame.content_type != "image/jpeg"
            ):
                self._delivery_failure = "image_payload_rejected"
                self._disable_creates = True
            return None
        payload = build_ambient_image_event(item, frame)
        if not self.state.mark_submitted(
            item_id=item.item_id, now=self._now(), monotonic_now=self._monotonic(),
        ):
            return None
        operation.write_started = True
        self._remember_image_reference(operation, frame=frame)
        self._last_dispatch = self._monotonic()
        self._create_times[item.item_id] = self._last_dispatch
        return payload

    async def _revalidated_payload_at_write(self, operation: _ImageOperation) -> dict[str, Any] | None:
        if self._closed or operation.cancelled or operation.image is None:
            return None
        try:
            if operation.image.revalidate is not None:
                await operation.image.revalidate()
        except (ImageUnavailableError, RequestedImageDeliveryError) as error:
            operation.failure_reason = error.code
            return None
        # Recheck session cancellation, expiry and source identity after the await.
        return self._payload_at_write(operation)

    async def _write(self, operation: _ImageOperation) -> None:
        try:
            if operation.provider_call_id is not None:
                async with asyncio.timeout(self._write_timeout):
                    if operation.image is not None and operation.image.revalidate is not None:
                        sent = await self._provider.send_event(
                            lambda: self._revalidated_payload_at_write(operation),
                        )
                    else:
                        sent = await self._provider.send_event(lambda: self._payload_at_write(operation))
            else:
                sent = await self._provider.send_visual_event(
                    lambda: self._payload_at_write(operation),
                    timeout_seconds=self._write_timeout,
                )
            if not sent:
                self._not_written(operation)
            elif operation.kind == "create":
                self._capture_image_delivery(operation.item_id, "socket_written")
        except asyncio.CancelledError:
            if not self._closed:
                self._write_failed(operation)
            raise
        except Exception:
            if not self._closed:
                self._write_failed(operation)
        finally:
            operation.image = None
            self.notify_frame_available()

    def _not_written(self, operation: _ImageOperation) -> None:
        if operation.kind == "create":
            self._capture_image_delivery(operation.item_id, "not_written")
        if operation.provider_call_id is not None and operation.failure_reason is None:
            operation.failure_reason = "image_not_written"
        if operation.kind == "create":
            self.state.cancel_reserved(
                connection_epoch=self.state.connection.connection_epoch, item_id=operation.item_id,
            )
            if self._operation is operation:
                self._operation = None
        else:
            self.state.cancel_unwritten_delete(
                connection_epoch=self.state.connection.connection_epoch,
                item_id=operation.item_id, request_id=operation.request_id,
            )
            if self._operation is operation:
                self._operation = None
        self._retry_after = self._monotonic() + VISUAL_POLL_SECONDS

    def _write_failed(self, operation: _ImageOperation) -> None:
        if not operation.write_started:
            self._not_written(operation)
            self._delivery_failure = "image_writer_busy"
        else:
            self._mark_unknown(operation, "image_write_outcome_unknown")

    def _mark_unknown(self, operation: _ImageOperation, reason: str) -> None:
        if operation.kind == "create":
            self._capture_image_delivery(operation.item_id, "write_outcome_unknown")
        if operation.provider_call_id is not None:
            operation.failure_reason = operation.failure_reason or reason
        key = self.state.connection.connection_epoch
        changed = (
            self.state.mark_create_timeout(connection_epoch=key, item_id=operation.item_id)
            if operation.kind == "create" else
            self.state.mark_delete_timeout(
                connection_epoch=key, item_id=operation.item_id, request_id=operation.request_id,
            )
        )
        if changed:
            self._delivery_failure = reason
        if self._operation is operation:
            self._operation = None

    def handle_provider_event(self, payload: dict[str, Any]) -> bool:
        """Consume this connection's image events without blocking the receiver."""
        if self._closed:
            return False
        event_type = payload.get("type")
        raw_item = payload.get("item")
        item_id = raw_item.get("id") if isinstance(raw_item, dict) else payload.get("item_id")
        if not isinstance(item_id, str):
            item_id = None
        request_id = None
        if event_type == "error":
            error = payload.get("error")
            request_id = error.get("event_id") if isinstance(error, dict) else None
            if not isinstance(request_id, str):
                return False
            item = next((
                item for item in self.state.items
                if request_id in {item.create_request_id, item.delete_request_id}
            ), None)
            if item is None:
                return request_id in self._recent_requests
            item_id = item.item_id
        else:
            if item_id is None or not self.is_passive_item(item_id):
                return False
            item = next((item for item in self.state.items if item.item_id == item_id), None)
            if item is None:
                return True
        epoch = self.state.connection.connection_epoch
        changed = False
        if event_type in {"conversation.item.created", "conversation.item.done"}:
            changed = self.state.acknowledge_create(
                connection_epoch=epoch, item_id=item.item_id, acknowledged_at=self._now(),
            )
            if changed:
                self._capture_image_delivery(item.item_id, "provider_acknowledged")
                started = self._create_times.get(item.item_id)
                if started is not None:
                    self._last_ack_latency = max(0.0, self._monotonic() - started)
                self._delivery_failure = None
        elif event_type == "conversation.item.deleted" and item.delete_request_id:
            changed = self.state.acknowledge_delete(
                connection_epoch=epoch, item_id=item.item_id, request_id=item.delete_request_id,
            )
            if changed:
                self._create_times.pop(item.item_id, None)
                self._delivery_failure = None
        elif event_type == "error":
            if request_id == item.create_request_id:
                changed = self.state.reject_create(connection_epoch=epoch, item_id=item.item_id)
                if changed:
                    self._capture_image_delivery(item.item_id, "provider_rejected")
                    self._create_times.pop(item.item_id, None)
                    self._disable_creates = True
                    if self._operation is not None and self._operation.item_id == item.item_id:
                        self._operation.failure_reason = "provider_rejected_image_input"
                    self._delivery_failure = "provider_rejected_image_input"
                    LOGGER.warning(
                        "passive image input rejected; verify configured Realtime model image support"
                    )
            else:
                error = payload.get("error", {})
                if error.get("code") in {"item_not_found", "conversation_item_not_found"}:
                    changed = self.state.acknowledge_delete(
                        connection_epoch=epoch, item_id=item.item_id, request_id=request_id,
                    )
                    if changed:
                        self._create_times.pop(item.item_id, None)
                else:
                    changed = self.state.mark_delete_timeout(
                        connection_epoch=epoch, item_id=item.item_id, request_id=request_id,
                    )
                self._delivery_failure = None if not any(
                    entry.item_id == item.item_id for entry in self.state.items
                ) else "image_delete_rejected"
        if changed and self._operation is not None and self._operation.item_id == item.item_id:
            self._operation = None
        self.notify_frame_available()
        # item.added is recognized but not treated as fully accepted until item.done.
        return True

    async def prepare_response(
        self, *, turn_id: str, generation_id: UUID, requested_call_ids: tuple[str, ...] = (),
    ) -> VisualResponseSnapshot:
        """Stage 2C calls this outside the speech-admission lock, then revalidates.

        The returned immutable cutoff is single-use. The image fence survives
        response submission until its correlated response.created event arrives.
        """
        if self._closed:
            raise VisualContextNotReadyError("The visual connection is closed.")
        if any(work.task is not None for work in self._requested_work.values()):
            raise VisualContextNotReadyError("Requested image tool work must settle before response preparation.")
        if self._preparation is not None and self._response_request_id is not None:
            if len(self._prior_responses) >= MAX_UNSETTLED_VISUAL_RESPONSES:
                raise VisualContextNotReadyError("Too many response outcomes remain unknown.")
            # Cancellation is not settlement. Keep every old input pinned until
            # that response's own done/error, including a response.created arriving late.
            snapshot = self.state.active_snapshot
            assert snapshot is not None
            self._prior_responses[self._response_request_id] = (snapshot, self._response_id)
            self.state.hold_active_response()
            self._preparation = None
            self._response_request_id = self._response_id = None
            self._fence_generation = None
        if self._fence_generation is not None or self.state.active_snapshot is not None:
            raise VisualContextNotReadyError("Another visual preparation is still active.")
        deadline = VisualSyncDeadline(self._monotonic(), self.state.policy.sync_timeout_seconds)
        self._fence_generation = generation_id
        try:
            while True:
                if self._closed:
                    raise VisualContextNotReadyError("Visual connection closed during preparation.")
                revision = self._revision
                self._tick(preparing=True)
                writing = self._write_task is not None and not self._write_task.done()
                if not writing and self._operation is None and not self._prior_ack_pending():
                    try:
                        snapshot = self.state.prepare_response(
                            turn_id=turn_id, generation_id=generation_id,
                            now=self._now(), monotonic_now=self._monotonic(),
                            requested_call_ids=requested_call_ids,
                        )
                        break
                    except VisualContextNotReadyError:
                        pass
                remaining = deadline.remaining(self._monotonic())
                if remaining <= 0:
                    # A queued, never-written operation cannot escape this cutoff.
                    operation = self._operation
                    if operation is not None and not operation.write_started:
                        operation.cancelled = True
                        if self._write_task is not None:
                            self._write_task.cancel()
                        self._not_written(operation)
                    snapshot = self.state.prepare_response(
                        turn_id=turn_id, generation_id=generation_id,
                        now=self._now(), monotonic_now=self._monotonic(), allow_degraded=True,
                        synchronization_blocked=self._prior_ack_pending(),
                        requested_call_ids=requested_call_ids,
                    )
                    break
                await self._wait_for_change(revision, min(remaining, VISUAL_POLL_SECONDS))
            self._preparation = snapshot
            for call_id in requested_call_ids:
                self._requested_work.pop(call_id, None)
                self._retired_image_calls.append((call_id, "request_settled"))
            return snapshot
        except BaseException:
            if self._fence_generation == generation_id:
                self._fence_generation = None
                self.notify_frame_available()
            raise

    async def _wait_for_change(self, revision: int, timeout: float) -> None:
        if self._revision != revision:
            return
        self._wake.clear()
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=timeout)
        except TimeoutError:
            pass

    def mark_response_submitted(self, preparation: VisualResponseSnapshot, *, request_id: str) -> bool:
        """Call immediately before enqueueing response.create inside speech admission."""
        if (
            self._closed or self._preparation is not preparation
            or self._response_request_id is not None
            or preparation.connection != self.state.connection
        ):
            return False
        self._refresh_local()
        # The source may disconnect or age while response.create waits for the
        # ordered writer. Preserve the recorded image set, but revoke freshness.
        availability = preparation.availability
        synchronized = preparation.synchronized
        for pin in preparation.requested_images:
            try:
                self._validate_requested_reference(pin.reference)
            except (ImageUnavailableError, RequestedImageDeliveryError):
                return False
            if pin.expired(now=self._now(), monotonic_now=self._monotonic()):
                return False
        if preparation.visual_session_id != self.state.visual_session_id:
            synchronized = False
            availability = VisualAvailability(
                VisualAvailabilityStatus.DEGRADED, VisualAvailabilityReason.STREAM_CLEANUP_REQUIRED,
            )
        elif preparation.current_frame is not None:
            freshness = preparation.current_frame.freshness(
                now=self._now(), monotonic_now=self._monotonic(),
                max_age_seconds=self.state.policy.max_context_age_seconds,
            )
            if freshness.status != "fresh":
                availability = VisualAvailability(
                    VisualAvailabilityStatus.STALE if freshness.status == "stale" else VisualAvailabilityStatus.DEGRADED,
                    VisualAvailabilityReason.STALE_FRAME if freshness.status == "stale" else VisualAvailabilityReason.CLOCK_UNCERTAIN,
                    freshness.age_seconds,
                )
            else:
                availability = replace(availability, age_seconds=freshness.age_seconds)
        self._response_request_id = request_id
        self.state.active_snapshot = replace(
            preparation, response_submitted_at=self._now(),
            availability=availability, synchronized=synchronized,
        )
        return True

    def response_created(self, *, request_id: str, response_id: str) -> bool:
        prior = self._prior_responses.get(request_id)
        if prior is not None and not self._closed:
            snapshot, known_id = prior
            if known_id not in {None, response_id}:
                return False
            self._prior_responses[request_id] = (
                replace(snapshot, provider_response_created_at=self._now()), response_id,
            )
            self.notify_frame_available()
            return True
        if self._closed or request_id != self._response_request_id or self.state.active_snapshot is None:
            return False
        self._response_id = response_id
        self.state.active_snapshot = replace(
            self.state.active_snapshot, provider_response_created_at=self._now(),
        )
        self._fence_generation = None
        self.notify_frame_available()
        return True

    def settle_response(self, *, request_id: str, response_id: str | None = None) -> bool:
        prior = self._prior_responses.get(request_id)
        if prior is not None and not self._closed:
            snapshot, known_id = prior
            if response_id is not None and known_id not in {None, response_id}:
                return False
            self.state.settle_response(
                connection_epoch=self.state.connection.connection_epoch,
                generation_id=snapshot.generation_id,
            )
            self._recent_cutoffs.append(snapshot.to_metadata())
            del self._prior_responses[request_id]
            self.notify_frame_available()
            return True
        if (
            self._closed or request_id != self._response_request_id
            or (response_id is not None and self._response_id not in {None, response_id})
            or self._preparation is None
        ):
            return False
        if self.state.active_snapshot is not None:
            self._recent_cutoffs.append(self.state.active_snapshot.to_metadata())
        self.state.settle_response(
            connection_epoch=self.state.connection.connection_epoch,
            generation_id=self._preparation.generation_id,
        )
        self._preparation = None
        self._response_request_id = self._response_id = None
        self._fence_generation = None
        self.notify_frame_available()
        return True

    def cancel_preparation(self, preparation: VisualResponseSnapshot) -> bool:
        if self._preparation is not preparation or self._response_request_id is not None:
            return False
        self.state.settle_response(
            connection_epoch=self.state.connection.connection_epoch,
            generation_id=preparation.generation_id,
        )
        self._preparation = None
        self._fence_generation = None
        self.notify_frame_available()
        return True
