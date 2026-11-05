from __future__ import annotations

import unittest
from uuid import uuid4

from app.runtime.events import (
    DomainEvent,
    DomainEventBus,
    DomainEventName,
    VisualFrameAcceptedPayload,
)


class DomainEventBusTests(unittest.IsolatedAsyncioTestCase):
    async def test_handlers_run_in_registration_order_and_failures_are_isolated(self) -> None:
        bus = DomainEventBus()
        calls: list[str] = []

        async def first(event: DomainEvent) -> None:
            calls.append("first")

        async def failing(event: DomainEvent) -> None:
            calls.append("failing")
            raise RuntimeError("expected failure")

        async def last(event: DomainEvent) -> None:
            calls.append("last")

        bus.subscribe(DomainEventName.VISUAL_FRAME_ACCEPTED, first)
        bus.subscribe(DomainEventName.VISUAL_FRAME_ACCEPTED, failing)
        bus.subscribe(DomainEventName.VISUAL_FRAME_ACCEPTED, last)

        result = await bus.publish(_visual_frame_event())

        self.assertEqual(calls, ["first", "failing", "last"])
        self.assertEqual(result.delivered_count, 2)
        self.assertEqual(result.failed_count, 1)

    async def test_duplicate_subscription_is_rejected_without_changing_order(self) -> None:
        bus = DomainEventBus()

        async def handler(event: DomainEvent) -> None:
            return None

        bus.subscribe(DomainEventName.VISUAL_FRAME_ACCEPTED, handler)
        with self.assertRaisesRegex(ValueError, "already subscribed"):
            bus.subscribe(DomainEventName.VISUAL_FRAME_ACCEPTED, handler)

        result = await bus.publish(_visual_frame_event())
        self.assertEqual(result.delivered_count, 1)

    async def test_duplicate_bound_method_subscription_is_rejected(self) -> None:
        bus = DomainEventBus()

        class Receiver:
            async def handle(self, event: DomainEvent) -> None:
                return None

        receiver = Receiver()
        bus.subscribe(DomainEventName.VISUAL_FRAME_ACCEPTED, receiver.handle)

        with self.assertRaisesRegex(ValueError, "already subscribed"):
            bus.subscribe(DomainEventName.VISUAL_FRAME_ACCEPTED, receiver.handle)

    def test_event_rejects_payload_for_a_different_name(self) -> None:
        with self.assertRaisesRegex(TypeError, "requires payload type"):
            DomainEvent(
                name=DomainEventName.MEMORY_ITEM_CHANGED,
                payload=VisualFrameAcceptedPayload(
                    owner_id="owner",
                    session_id=uuid4(),
                    frame_id=uuid4(),
                    frame_sequence=1,
                ),
            )


def _visual_frame_event() -> DomainEvent:
    return DomainEvent(
        name=DomainEventName.VISUAL_FRAME_ACCEPTED,
        payload=VisualFrameAcceptedPayload(
            owner_id="owner",
            session_id=uuid4(),
            frame_id=uuid4(),
            frame_sequence=1,
        ),
    )
