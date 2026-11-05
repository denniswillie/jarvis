from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

from app.images import ImageReference
from app.perception.frame_window import SensoryVisualFrame
from app.perception.freshness import (
    FrameFreshness,
    assess_frame_freshness,
    require_aware_timestamp,
)

if TYPE_CHECKING:
    from app.settings import JarvisSettings


RESPONSE_IMAGE_BUDGET = 2
LIVE_IMAGE_RESERVE = 2
PASSIVE_VISUAL_INPUT_KIND = "ambient_visual"


@dataclass(frozen=True)
class VisualDeliveryPolicy:
    enabled: bool = True
    min_interval_seconds: float = 1.0
    max_context_age_seconds: float = 5.0
    sync_timeout_seconds: float = 0.5
    max_image_items: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("Visual delivery enabled must be a boolean.")
        for value in (
            self.min_interval_seconds, self.max_context_age_seconds, self.sync_timeout_seconds
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError("Visual delivery durations must be positive and finite.")
        if (
            not isinstance(self.max_image_items, int)
            or isinstance(self.max_image_items, bool)
            or self.max_image_items < RESPONSE_IMAGE_BUDGET + LIVE_IMAGE_RESERVE
        ):
            raise ValueError("Visual delivery requires at least four image slots.")

    @classmethod
    def from_settings(cls, settings: JarvisSettings) -> VisualDeliveryPolicy:
        return cls(
            enabled=settings.VISUAL_REALTIME_ENABLED,
            min_interval_seconds=settings.VISUAL_REALTIME_MIN_INTERVAL_SECONDS,
            max_context_age_seconds=settings.VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS,
            sync_timeout_seconds=settings.VISUAL_REALTIME_SYNC_TIMEOUT_SECONDS,
            max_image_items=settings.VISUAL_REALTIME_MAX_IMAGE_ITEMS,
        )


@dataclass(frozen=True)
class VisualSyncDeadline:
    started_monotonic: float
    timeout_seconds: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.started_monotonic)
            or not isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("Visual synchronization needs a finite, positive budget.")

    def remaining(self, monotonic_now: float) -> float:
        if not isfinite(monotonic_now) or monotonic_now < self.started_monotonic:
            raise ValueError("Visual synchronization requires a non-regressing clock.")
        return max(0.0, self.timeout_seconds - (monotonic_now - self.started_monotonic))


@dataclass(frozen=True)
class VisualConnectionKey:
    owner_id: str
    session_id: UUID
    connection_epoch: UUID

    def __post_init__(self) -> None:
        if not self.owner_id.strip():
            raise ValueError("Visual context requires an owner.")
        if not isinstance(self.session_id, UUID) or not isinstance(self.connection_epoch, UUID):
            raise ValueError("Visual context requires distinct session and connection identifiers.")


@dataclass(frozen=True)
class VisualFrameMetadata:
    """Safe projection of sensory evidence, never another image-byte buffer."""

    connection: VisualConnectionKey
    visual_session_id: UUID
    frame_id: UUID
    frame_sequence: int
    source: str
    captured_at: datetime
    server_received_at: datetime
    sampled_at: datetime
    sampled_monotonic: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.frame_sequence, int)
            or isinstance(self.frame_sequence, bool)
            or self.frame_sequence < 0
        ):
            raise ValueError("Visual frame sequence must be a nonnegative integer.")
        if not isinstance(self.visual_session_id, UUID) or not isinstance(self.frame_id, UUID):
            raise ValueError("Visual frame identifiers must be UUIDs.")
        if self.source not in {"glasses", "iphone_camera"}:
            raise ValueError("Unsupported visual frame source.")
        for timestamp in (self.captured_at, self.server_received_at, self.sampled_at):
            require_aware_timestamp(timestamp)
        if not isfinite(self.sampled_monotonic):
            raise ValueError("Visual frame sample time must be finite.")

    @classmethod
    def from_sensory_frame(
        cls,
        frame: SensoryVisualFrame,
        *,
        connection: VisualConnectionKey,
        sampled_at: datetime,
        sampled_monotonic: float,
    ) -> VisualFrameMetadata:
        if (frame.owner_id, frame.session_id) != (connection.owner_id, connection.session_id):
            raise ValueError("Visual frame does not belong to this connection's owner/session.")
        return cls(
            connection=connection,
            visual_session_id=frame.visual_session_id,
            frame_id=frame.frame_id,
            frame_sequence=frame.frame_sequence,
            source=frame.source,
            captured_at=frame.captured_at,
            server_received_at=frame.server_received_at,
            sampled_at=sampled_at,
            sampled_monotonic=sampled_monotonic,
        )

    def freshness(
        self, *, now: datetime, monotonic_now: float, max_age_seconds: float
    ) -> FrameFreshness:
        return assess_frame_freshness(
            captured_at=self.captured_at,
            server_received_at=self.server_received_at,
            sampled_at=self.sampled_at,
            sampled_monotonic=self.sampled_monotonic,
            now=now,
            monotonic_now=monotonic_now,
            max_age_seconds=max_age_seconds,
        )

    def is_newer_than(self, previous: VisualFrameMetadata) -> bool:
        # Stream changes are explicit; delayed uploads cannot rewind a stream.
        return (
            self.connection == previous.connection
            and self.visual_session_id == previous.visual_session_id
            and self.frame_id != previous.frame_id
            and self.frame_sequence > previous.frame_sequence
            and self.server_received_at >= previous.server_received_at
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "input_kind": PASSIVE_VISUAL_INPUT_KIND,
            "owner_id": self.connection.owner_id,
            "session_id": str(self.connection.session_id),
            "connection_epoch": str(self.connection.connection_epoch),
            "visual_session_id": str(self.visual_session_id),
            "frame_id": str(self.frame_id),
            "frame_sequence": self.frame_sequence,
            "source": self.source,
            "captured_at": self.captured_at.isoformat(),
            "server_received_at": self.server_received_at.isoformat(),
        }


class ProviderImageStatus(str, Enum):
    RESERVED = "reserved"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    CREATE_UNKNOWN = "create_unknown"
    DELETE_PENDING = "delete_pending"
    DELETE_UNKNOWN = "delete_unknown"


@dataclass(frozen=True)
class ProviderImageItem:
    item_id: str
    create_request_id: str
    frame: VisualFrameMetadata | ImageReference
    status: ProviderImageStatus = ProviderImageStatus.RESERVED
    submitted_at: datetime | None = None
    acknowledged_at: datetime | None = None
    delete_request_id: str | None = None
    delivery_origin: Literal["passive", "requested"] = "passive"
    image_reference: ImageReference | None = None
    provider_call_id: str | None = None
    turn_generation: int | None = None

    @property
    def is_historical(self) -> bool:
        return isinstance(self.frame, ImageReference) and self.frame.source == "historical"

    def to_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "item_id": self.item_id,
            "create_request_id": self.create_request_id,
            "delete_request_id": self.delete_request_id,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "frame": self.frame.to_metadata(),
        }
        if self.delivery_origin == "requested":
            metadata.update(
                delivery_origin=self.delivery_origin,
                provider_call_id=self.provider_call_id,
                turn_generation=self.turn_generation,
                image=self.image_reference.to_metadata() if self.image_reference else None,
            )
        return metadata


@dataclass(frozen=True)
class RequestedImagePin:
    provider_call_id: str
    turn_generation: int
    item_id: str
    reference: ImageReference
    admitted_at: datetime
    admitted_monotonic: float

    def expired(self, *, now: datetime, monotonic_now: float) -> bool:
        expires_at = self.reference.raw_image_expires_at
        if expires_at is None:
            return False
        return (
            now >= expires_at
            or monotonic_now - self.admitted_monotonic
            >= (expires_at - self.admitted_at).total_seconds()
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "provider_call_id": self.provider_call_id,
            "turn_generation": self.turn_generation,
            "item_id": self.item_id,
            "image": self.reference.to_metadata(),
        }


class RequestedImageDeliveryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class VisualAvailabilityStatus(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


class VisualAvailabilityReason(str, Enum):
    DISABLED = "disabled"
    CONNECTION_CLOSED = "connection_closed"
    NO_VISUAL_STREAM = "no_visual_stream"
    NO_FRAME = "no_frame"
    AWAITING_ACKNOWLEDGEMENT = "awaiting_acknowledgement"
    STALE_FRAME = "stale_frame"
    CLOCK_UNCERTAIN = "clock_uncertain"
    PROVIDER_OUTCOME_UNKNOWN = "provider_outcome_unknown"
    PROVIDER_REJECTED = "provider_rejected"
    STREAM_CLEANUP_REQUIRED = "stream_cleanup_required"
    CONTEXT_UNSYNCHRONIZED = "context_unsynchronized"


@dataclass(frozen=True)
class VisualAvailability:
    status: VisualAvailabilityStatus
    reason: VisualAvailabilityReason | None = None
    age_seconds: float | None = None

    def to_metadata(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reason": self.reason.value if self.reason else None,
            "age_seconds": self.age_seconds,
        }


@dataclass(frozen=True)
class VisualResponseSnapshot:
    connection: VisualConnectionKey
    visual_session_id: UUID | None
    turn_id: str
    generation_id: UUID
    prepared_at: datetime
    prepared_monotonic: float
    synchronized: bool
    availability: VisualAvailability
    acknowledged_items: tuple[ProviderImageItem, ...]
    possibly_present_items: tuple[ProviderImageItem, ...]
    response_submitted_at: datetime | None = None
    provider_response_created_at: datetime | None = None
    requested_images: tuple[RequestedImagePin, ...] = ()

    @property
    def pinned_item_ids(self) -> frozenset[str]:
        return frozenset(
            item.item_id for item in self.acknowledged_items + self.possibly_present_items
        )

    @property
    def current_frame(self) -> VisualFrameMetadata | None:
        if not self.synchronized or self.availability.status != VisualAvailabilityStatus.FRESH:
            return None
        return max(
            (
                item.frame for item in self.acknowledged_items
                if isinstance(item.frame, VisualFrameMetadata)
                and item.frame.visual_session_id == self.visual_session_id
            ),
            key=lambda frame: (frame.server_received_at, frame.frame_sequence),
            default=None,
        )

    def observation_is_older(self, *, observed_at: datetime) -> bool:
        """Compare evidence provenance, never the working-memory DB update time."""
        require_aware_timestamp(observed_at)
        frame = self.current_frame
        return frame is not None and observed_at < frame.captured_at

    def to_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "owner_id": self.connection.owner_id,
            "session_id": str(self.connection.session_id),
            "connection_epoch": str(self.connection.connection_epoch),
            "visual_session_id": str(self.visual_session_id) if self.visual_session_id else None,
            "turn_id": self.turn_id,
            "generation_id": str(self.generation_id),
            "prepared_at": self.prepared_at.isoformat(),
            "response_submitted_at": self.response_submitted_at.isoformat() if self.response_submitted_at else None,
            "provider_response_created_at": self.provider_response_created_at.isoformat() if self.provider_response_created_at else None,
            "synchronized": self.synchronized,
            "availability": self.availability.to_metadata(),
            "acknowledged_items": [item.to_metadata() for item in self.acknowledged_items],
            "possibly_present_items": [item.to_metadata() for item in self.possibly_present_items],
        }
        if self.requested_images:
            metadata["requested_images"] = [pin.to_metadata() for pin in self.requested_images]
        historical = [item.to_metadata() for item in self.acknowledged_items if item.is_historical]
        if historical:
            metadata["historical_images"] = historical
        return metadata


class VisualContextNotReadyError(RuntimeError):
    pass


class VisualContextState:
    """Connection-local, metadata-only state; no tasks, transport, or speech.

    Stage 2B owns serialization, clocks, timers, and provider event routing.
    It must fence image mutations around prepare_response and response creation.
    These synchronous transitions neither provide that I/O fence nor send events.
    """

    def __init__(
        self,
        *,
        connection: VisualConnectionKey,
        policy: VisualDeliveryPolicy,
        visual_session_id: UUID | None = None,
    ) -> None:
        self.connection = connection
        self.policy = policy
        self.visual_session_id = visual_session_id
        self.latest_local: VisualFrameMetadata | None = None
        self.latest_submitted: VisualFrameMetadata | None = None
        self.latest_acknowledged: VisualFrameMetadata | None = None
        self.active_snapshot: VisualResponseSnapshot | None = None
        self._held_response_pins: dict[UUID, frozenset[str]] = {}
        self._requested_pins: dict[str, RequestedImagePin] = {}
        self._pending: VisualFrameMetadata | None = None
        self._items: dict[str, ProviderImageItem] = {}
        self._closed = False
        self._provider_rejected = False

    @property
    def items(self) -> tuple[ProviderImageItem, ...]:
        return tuple(self._items.values())

    @property
    def pinned_item_ids(self) -> frozenset[str]:
        pins = self.active_snapshot.pinned_item_ids if self.active_snapshot else frozenset()
        return pins.union(
            (pin.item_id for pin in self._requested_pins.values()),
            *self._held_response_pins.values(),
        )

    @property
    def requested_pins(self) -> tuple[RequestedImagePin, ...]:
        return tuple(self._requested_pins.values())

    def requested_pin(self, provider_call_id: str) -> RequestedImagePin | None:
        return self._requested_pins.get(provider_call_id)

    def requested_item(self, provider_call_id: str) -> ProviderImageItem | None:
        pin = self._requested_pins.get(provider_call_id)
        return self._items.get(pin.item_id) if pin is not None else None

    def hold_active_response(self) -> None:
        """Retain cancelled-but-unsettled inputs while a newer response is prepared."""
        if self.active_snapshot is not None:
            self._held_response_pins[self.active_snapshot.generation_id] = self.active_snapshot.pinned_item_ids
            self.active_snapshot = None

    @property
    def free_slots(self) -> int:
        return self.policy.max_image_items - len(self._items)

    def activate_visual_stream(self, visual_session_id: UUID | None) -> None:
        if self._closed:
            raise RuntimeError("A retired visual connection cannot be reused.")
        if visual_session_id == self.visual_session_id:
            return
        self.visual_session_id = visual_session_id
        self.latest_local = self.latest_submitted = self.latest_acknowledged = None
        self._pending = None
        self._provider_rejected = False
        # Old provider items and generation pins survive until actual cleanup.

    def _matches_current_stream(self, frame: VisualFrameMetadata | ImageReference) -> bool:
        return (
            isinstance(frame, VisualFrameMetadata)
            and frame.connection == self.connection
            and frame.visual_session_id == self.visual_session_id
        )

    def _is_fresh(self, frame: VisualFrameMetadata, now: datetime, monotonic_now: float) -> bool:
        return frame.freshness(
            now=now, monotonic_now=monotonic_now,
            max_age_seconds=self.policy.max_context_age_seconds,
        ).status == "fresh"

    def offer_frame(self, frame: VisualFrameMetadata) -> bool:
        if self._closed or not self.policy.enabled or not self._matches_current_stream(frame):
            return False
        if self.latest_local is not None and not frame.is_newer_than(self.latest_local):
            return False
        self.latest_local = self._pending = frame
        return True

    def pending_candidate(self, *, now: datetime, monotonic_now: float) -> VisualFrameMetadata | None:
        if self._pending is not None and not self._is_fresh(self._pending, now, monotonic_now):
            self._pending = None
        return self._pending

    def reserve_create(self, *, now: datetime, monotonic_now: float) -> ProviderImageItem | None:
        candidate = self.pending_candidate(now=now, monotonic_now=monotonic_now)
        if self._closed or not self.policy.enabled or candidate is None or self.free_slots == 0:
            return None
        if any(
            item.status in {ProviderImageStatus.RESERVED, ProviderImageStatus.SUBMITTED, ProviderImageStatus.CREATE_UNKNOWN}
            or (not item.is_historical and not self._matches_current_stream(item.frame))
            for item in self._items.values()
        ):
            return None
        item = ProviderImageItem(uuid4().hex, uuid4().hex, candidate)
        self._items[item.item_id] = item
        self._pending = None
        return item

    def reserve_requested_create(
        self,
        *,
        reference: ImageReference,
        current_frame: VisualFrameMetadata | None,
        provider_call_id: str,
        turn_generation: int,
        now: datetime,
        monotonic_now: float,
    ) -> ProviderImageItem | None:
        """Reserve in the shared ledger, or pin an identical acknowledged input."""
        if self._closed:
            raise RequestedImageDeliveryError("connection_closed", "The image connection is closed.")
        if not provider_call_id.strip() or not isinstance(turn_generation, int) or isinstance(turn_generation, bool) or turn_generation < 0:
            raise ValueError("Requested images require a tool call and turn generation.")
        require_aware_timestamp(now)
        if not isfinite(monotonic_now):
            raise ValueError("Requested images require finite admission timing.")
        if reference.owner_id != self.connection.owner_id:
            raise RequestedImageDeliveryError("frame_unavailable", "The selected image is unavailable.")
        if reference.source == "current":
            if (
                reference.session_id != self.connection.session_id
                or current_frame is None
                or current_frame.frame_id != reference.frame_id
                or current_frame.captured_at != reference.captured_at
                or current_frame.visual_session_id != reference.visual_session_id
                or not self._matches_current_stream(current_frame)
                or not self._is_fresh(current_frame, now, monotonic_now)
            ):
                raise RequestedImageDeliveryError("source_changed", "The current image is no longer eligible.")
            frame: VisualFrameMetadata | ImageReference = current_frame
        else:
            if current_frame is not None:
                raise ValueError("Historical images cannot carry current-frame metadata.")
            frame = reference
        previous = self._requested_pins.get(provider_call_id)
        if previous is not None:
            if previous.reference != reference or previous.turn_generation != turn_generation:
                raise RequestedImageDeliveryError("request_conflict", "The image request identity has changed.")
            return self._items.get(previous.item_id)
        if len(self._requested_pins) >= self.policy.max_image_items:
            return None
        reusable = next((
            item for item in self._items.values()
            if item.status == ProviderImageStatus.ACKNOWLEDGED
            and item.is_historical == (reference.source == "historical")
            and item.frame.frame_id == reference.frame_id
            and item.frame.captured_at == reference.captured_at
            and (
                item.frame == reference if item.is_historical
                else self._matches_current_stream(item.frame)
            )
        ), None)
        requested_ids = {pin.item_id for pin in self._requested_pins.values()}
        if (reusable is None or reusable.item_id not in requested_ids) and len(requested_ids) >= self.policy.max_image_items - LIVE_IMAGE_RESERVE:
            return None
        if reusable is None:
            if self.free_slots <= 0 or any(
                item.status != ProviderImageStatus.ACKNOWLEDGED for item in self._items.values()
            ):
                return None
            reusable = ProviderImageItem(
                uuid4().hex, uuid4().hex, frame,
                delivery_origin="requested", image_reference=reference,
                provider_call_id=provider_call_id, turn_generation=turn_generation,
            )
        pin = RequestedImagePin(
            provider_call_id, turn_generation, reusable.item_id, reference, now, monotonic_now,
        )
        if pin.expired(now=now, monotonic_now=monotonic_now):
            raise RequestedImageDeliveryError("frame_unavailable", "The selected image has expired.")
        self._items[reusable.item_id] = reusable
        self._requested_pins[provider_call_id] = pin
        return reusable

    def release_requested_image(self, *, provider_call_id: str, turn_generation: int) -> bool:
        pin = self._requested_pins.get(provider_call_id)
        if pin is None or pin.turn_generation != turn_generation:
            return False
        del self._requested_pins[provider_call_id]
        if pin.item_id not in self.pinned_item_ids:
            self.cancel_reserved(connection_epoch=self.connection.connection_epoch, item_id=pin.item_id)
        return True

    def cancel_reserved(self, *, connection_epoch: UUID, item_id: str) -> bool:
        item = self._scoped_item(connection_epoch, item_id)
        if item is None or item.status != ProviderImageStatus.RESERVED:
            return False
        del self._items[item_id]
        if item.delivery_origin == "passive" and self._matches_current_stream(item.frame) and self.latest_local == item.frame:
            self._pending = item.frame
        return True

    def mark_submitted(self, *, item_id: str, now: datetime, monotonic_now: float) -> bool:
        """Call before enqueueing bytes; afterward a timeout cannot prove absence."""
        item = self._items.get(item_id)
        if item is None or item.status != ProviderImageStatus.RESERVED:
            return False
        if item.delivery_origin == "requested":
            pin = self._requested_pins.get(item.provider_call_id)
            if pin is None or pin.expired(now=now, monotonic_now=monotonic_now):
                return False
            if not item.is_historical and (
                not self._matches_current_stream(item.frame)
                or not self._is_fresh(item.frame, now, monotonic_now)
            ):
                return False
            self._items[item_id] = replace(item, status=ProviderImageStatus.SUBMITTED, submitted_at=now)
            if not item.is_historical and (
                self.latest_submitted is None or item.frame.is_newer_than(self.latest_submitted)
            ):
                self.latest_submitted = item.frame
            return True
        if (
            not self._matches_current_stream(item.frame)
            or self.latest_local != item.frame
            or not self._is_fresh(item.frame, now, monotonic_now)
        ):
            self.cancel_reserved(connection_epoch=self.connection.connection_epoch, item_id=item_id)
            return False
        self._items[item_id] = replace(item, status=ProviderImageStatus.SUBMITTED, submitted_at=now)
        self.latest_submitted = item.frame
        return True

    def acknowledge_create(self, *, connection_epoch: UUID, item_id: str, acknowledged_at: datetime) -> bool:
        require_aware_timestamp(acknowledged_at)
        item = self._scoped_item(connection_epoch, item_id)
        if item is None or item.status not in {ProviderImageStatus.SUBMITTED, ProviderImageStatus.CREATE_UNKNOWN}:
            return False
        self._items[item_id] = replace(
            item, status=ProviderImageStatus.ACKNOWLEDGED, acknowledged_at=acknowledged_at
        )
        if self._matches_current_stream(item.frame):
            if self.latest_acknowledged is None or item.frame.is_newer_than(self.latest_acknowledged):
                self.latest_acknowledged = item.frame
            self._provider_rejected = False
        return True

    def mark_create_timeout(self, *, connection_epoch: UUID, item_id: str) -> bool:
        item = self._scoped_item(connection_epoch, item_id)
        if item is None or item.status != ProviderImageStatus.SUBMITTED:
            return False
        self._items[item_id] = replace(item, status=ProviderImageStatus.CREATE_UNKNOWN)
        return True

    def reject_create(self, *, connection_epoch: UUID, item_id: str) -> bool:
        item = self._scoped_item(connection_epoch, item_id)
        if item is None or item.status not in {ProviderImageStatus.SUBMITTED, ProviderImageStatus.CREATE_UNKNOWN}:
            return False
        del self._items[item_id]
        if self._matches_current_stream(item.frame):
            self._provider_rejected = True
        return True

    def begin_delete(self, *, connection_epoch: UUID, item_id: str) -> str | None:
        item = self._scoped_item(connection_epoch, item_id)
        if item is None or item.status != ProviderImageStatus.ACKNOWLEDGED:
            return None
        if item_id in self.pinned_item_ids:
            return None
        request_id = uuid4().hex
        self._items[item_id] = replace(
            item, status=ProviderImageStatus.DELETE_PENDING, delete_request_id=request_id
        )
        return request_id

    def cancel_unwritten_delete(
        self, *, connection_epoch: UUID, item_id: str, request_id: str
    ) -> bool:
        """Rollback only when the writer proves the delete was never sent."""
        item = self._scoped_item(connection_epoch, item_id)
        if item is None or item.status != ProviderImageStatus.DELETE_PENDING or item.delete_request_id != request_id:
            return False
        self._items[item_id] = replace(
            item, status=ProviderImageStatus.ACKNOWLEDGED, delete_request_id=None,
        )
        return True

    def mark_delete_timeout(self, *, connection_epoch: UUID, item_id: str, request_id: str) -> bool:
        item = self._scoped_item(connection_epoch, item_id)
        if item is None or item.status != ProviderImageStatus.DELETE_PENDING or item.delete_request_id != request_id:
            return False
        self._items[item_id] = replace(item, status=ProviderImageStatus.DELETE_UNKNOWN)
        return True

    def acknowledge_delete(self, *, connection_epoch: UUID, item_id: str, request_id: str) -> bool:
        """Also usable for definitive not-found after a known accepted create.

        An unresolved create has no admitted delete request and cannot be freed
        by a speculative delete-not-found response.
        """
        item = self._scoped_item(connection_epoch, item_id)
        if (
            item is None
            or item.status not in {ProviderImageStatus.DELETE_PENDING, ProviderImageStatus.DELETE_UNKNOWN}
            or item.delete_request_id != request_id
        ):
            return False
        del self._items[item_id]
        return True

    def _scoped_item(self, connection_epoch: UUID, item_id: str) -> ProviderImageItem | None:
        if self._closed or connection_epoch != self.connection.connection_epoch:
            return None
        return self._items.get(item_id)

    def retirable_item_ids(self, *, now: datetime, monotonic_now: float) -> tuple[str, ...]:
        pinned = self.pinned_item_ids
        eligible = sorted(
            (
                item for item in self._items.values()
                if item.status == ProviderImageStatus.ACKNOWLEDGED
                and item.item_id not in pinned
                and item.delivery_origin == "passive"
                and self._matches_current_stream(item.frame)
                and self._is_fresh(item.frame, now, monotonic_now)
            ),
            key=lambda item: (item.frame.server_received_at, item.frame.frame_sequence),
            reverse=True,
        )
        keep_count = 1 if self.active_snapshot or self._held_response_pins else RESPONSE_IMAGE_BUDGET
        keep_count = max(0, keep_count - len({pin.item_id for pin in self._requested_pins.values()}))
        keep = {item.item_id for item in eligible[:keep_count]}
        return tuple(
            item.item_id for item in self._items.values()
            if item.status == ProviderImageStatus.ACKNOWLEDGED
            and item.item_id not in pinned and item.item_id not in keep
        )

    def availability(self, *, now: datetime, monotonic_now: float) -> VisualAvailability:
        status, reason = VisualAvailabilityStatus, VisualAvailabilityReason
        if self._closed:
            return VisualAvailability(status.UNAVAILABLE, reason.CONNECTION_CLOSED)
        if not self.policy.enabled and not any(
            item.delivery_origin == "requested" and not item.is_historical
            and item.status == ProviderImageStatus.ACKNOWLEDGED
            for item in self._items.values()
        ):
            return VisualAvailability(status.UNAVAILABLE, reason.DISABLED)
        if self.visual_session_id is None:
            return VisualAvailability(status.UNAVAILABLE, reason.NO_VISUAL_STREAM)
        if any(not item.is_historical and not self._matches_current_stream(item.frame) for item in self._items.values()):
            return VisualAvailability(status.DEGRADED, reason.STREAM_CLEANUP_REQUIRED)
        if any(not item.is_historical and item.status in {ProviderImageStatus.CREATE_UNKNOWN, ProviderImageStatus.DELETE_UNKNOWN} for item in self._items.values()):
            return VisualAvailability(status.DEGRADED, reason.PROVIDER_OUTCOME_UNKNOWN)
        if self._provider_rejected:
            return VisualAvailability(status.DEGRADED, reason.PROVIDER_REJECTED)
        acknowledged = [
            item.frame for item in self._items.values()
            if item.status == ProviderImageStatus.ACKNOWLEDGED and not item.is_historical
        ]
        frame = max(
            acknowledged,
            key=lambda value: (value.server_received_at, value.frame_sequence),
            default=self.latest_local,
        )
        if frame is None:
            return VisualAvailability(status.UNAVAILABLE, reason.NO_FRAME)
        freshness = frame.freshness(
            now=now, monotonic_now=monotonic_now,
            max_age_seconds=self.policy.max_context_age_seconds,
        )
        if freshness.status == "uncertain":
            return VisualAvailability(status.DEGRADED, reason.CLOCK_UNCERTAIN, freshness.age_seconds)
        if freshness.status == "stale":
            return VisualAvailability(status.STALE, reason.STALE_FRAME, freshness.age_seconds)
        if not acknowledged:
            return VisualAvailability(status.UNAVAILABLE, reason.AWAITING_ACKNOWLEDGEMENT, freshness.age_seconds)
        return VisualAvailability(status.FRESH, age_seconds=freshness.age_seconds)

    def prepare_response(
        self,
        *,
        turn_id: str,
        generation_id: UUID,
        now: datetime,
        monotonic_now: float,
        allow_degraded: bool = False,
        synchronization_blocked: bool = False,
        requested_call_ids: tuple[str, ...] = (),
    ) -> VisualResponseSnapshot:
        """Pin a local cutoff inside the future sender's ordered boundary.

        This synchronous method does not submit a response or establish provider
        context ordering. Stage 2B must keep its image fence through the matched
        response-created acknowledgement, including for degraded cutoffs.
        """
        if self._closed or self.active_snapshot is not None:
            raise VisualContextNotReadyError("Connection closed or prior generation is not settled.")
        if not turn_id.strip() or not isinstance(generation_id, UUID):
            raise ValueError("A response cutoff requires turn and generation identities.")
        require_aware_timestamp(now)
        if not isfinite(monotonic_now):
            raise ValueError("Response cutoff time must be finite.")
        if len(set(requested_call_ids)) != len(requested_call_ids):
            raise ValueError("Requested image calls must be unique.")
        if set(requested_call_ids) != set(self._requested_pins):
            raise VisualContextNotReadyError("Requested images require their own settled tool continuation.")
        requested = tuple(self._requested_pins[call_id] for call_id in requested_call_ids)
        if any(
            str(pin.turn_generation) != turn_id
            or pin.expired(now=now, monotonic_now=monotonic_now)
            or self._items.get(pin.item_id) is None
            or self._items[pin.item_id].status != ProviderImageStatus.ACKNOWLEDGED
            for pin in requested
        ):
            raise VisualContextNotReadyError("Requested image inputs are not acknowledged and eligible.")
        image_budget = max(RESPONSE_IMAGE_BUDGET, len({pin.item_id for pin in requested}))
        synchronized = (
            not synchronization_blocked
            and len(self._items) <= image_budget
            and self.free_slots >= LIVE_IMAGE_RESERVE
            and all(
                item.status == ProviderImageStatus.ACKNOWLEDGED
                and (
                    item.is_historical
                    or (
                        self._matches_current_stream(item.frame)
                        and self._is_fresh(item.frame, now, monotonic_now)
                    )
                )
                for item in self._items.values()
            )
        )
        if not synchronized and not allow_degraded:
            raise VisualContextNotReadyError("Image operations or cleanup must settle before a healthy cutoff.")
        available = self.availability(now=now, monotonic_now=monotonic_now)
        if not synchronized and available.status != VisualAvailabilityStatus.UNAVAILABLE:
            available = VisualAvailability(
                VisualAvailabilityStatus.DEGRADED, VisualAvailabilityReason.CONTEXT_UNSYNCHRONIZED
            )
        snapshot = VisualResponseSnapshot(
            connection=self.connection,
            visual_session_id=self.visual_session_id,
            turn_id=turn_id,
            generation_id=generation_id,
            prepared_at=now,
            prepared_monotonic=monotonic_now,
            synchronized=synchronized,
            availability=available,
            acknowledged_items=tuple(item for item in self._items.values() if item.status == ProviderImageStatus.ACKNOWLEDGED),
            possibly_present_items=tuple(item for item in self._items.values() if item.status not in {ProviderImageStatus.RESERVED, ProviderImageStatus.ACKNOWLEDGED}),
            requested_images=requested,
        )
        self.active_snapshot = snapshot
        for call_id in requested_call_ids:
            del self._requested_pins[call_id]
        return snapshot

    def settle_response(self, *, connection_epoch: UUID, generation_id: UUID) -> bool:
        """Release only on definite settlement or confirmed never-submitted cancellation."""
        if connection_epoch != self.connection.connection_epoch:
            return False
        if self.active_snapshot is not None and generation_id == self.active_snapshot.generation_id:
            self.active_snapshot = None
            return True
        return self._held_response_pins.pop(generation_id, None) is not None

    def close(self) -> None:
        self._closed = True
        self._items.clear()
        self._held_response_pins.clear()
        self._requested_pins.clear()
        self._pending = None
        self.latest_local = self.latest_submitted = self.latest_acknowledged = None
        self.active_snapshot = None

    def to_metadata(self, *, now: datetime, monotonic_now: float) -> dict[str, object]:
        candidate = self.pending_candidate(now=now, monotonic_now=monotonic_now)
        return {
            "input_kind": PASSIVE_VISUAL_INPUT_KIND,
            "availability": self.availability(now=now, monotonic_now=monotonic_now).to_metadata(),
            "latest_local": self.latest_local.to_metadata() if self.latest_local else None,
            "latest_submitted": self.latest_submitted.to_metadata() if self.latest_submitted else None,
            "latest_acknowledged": self.latest_acknowledged.to_metadata() if self.latest_acknowledged else None,
            "pending_frame_id": str(candidate.frame_id) if candidate else None,
            "image_item_count": len(self._items),
            "max_image_items": self.policy.max_image_items,
            "items": [item.to_metadata() for item in self._items.values()],
            "active_response": self.active_snapshot.to_metadata() if self.active_snapshot else None,
        }
