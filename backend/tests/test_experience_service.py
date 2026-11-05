import asyncio
import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import EvidenceAttachmentOutcome, ExperienceEventService
from app.sessions.models import JarvisMessage


class _EmptyExperienceRepository:
    def __init__(self) -> None:
        self.get_open_event_calls = 0

    async def get_open_event(self, *args, **kwargs):
        self.get_open_event_calls += 1
        return None


class ExperienceEventServiceTests(unittest.IsolatedAsyncioTestCase):
    def _service(
        self,
        repository: _EmptyExperienceRepository,
    ) -> ExperienceEventService:
        return ExperienceEventService(
            repository=repository,  # type: ignore[arg-type]
            segmenter=ExperienceEventSegmenter(
                EventSegmentationPolicy(
                    idle_close_seconds=12,
                    max_event_seconds=120,
                )
            ),
        )

    async def test_session_close_waits_for_active_work_before_finalizing(self) -> None:
        repository = _EmptyExperienceRepository()
        service = self._service(repository)
        session_id = uuid4()

        async with service.activity(owner_id="owner", session_id=session_id):
            service.schedule_session_close(
                owner_id="owner",
                session_id=session_id,
                closed_at=datetime.now(UTC),
            )
            await asyncio.sleep(0)
            self.assertEqual(repository.get_open_event_calls, 0)

        await service.close()

        self.assertEqual(repository.get_open_event_calls, 1)

    async def test_close_without_open_event_is_idempotent(self) -> None:
        repository = _EmptyExperienceRepository()
        service = self._service(repository)
        session_id = uuid4()
        closed_at = datetime.now(UTC)

        service.schedule_session_close(
            owner_id="owner",
            session_id=session_id,
            closed_at=closed_at,
        )
        service.schedule_session_close(
            owner_id="owner",
            session_id=session_id,
            closed_at=closed_at,
        )
        await service.close()

        self.assertEqual(repository.get_open_event_calls, 1)

    async def test_attach_message_uses_explicit_owner_scope(self) -> None:
        repository = _EmptyExperienceRepository()
        service = self._service(repository)
        session_id = uuid4()
        message = JarvisMessage(
            id=uuid4(),
            session_id=session_id,
            role="user",
            modality="audio",
            content="Remember this message.",
            provider_item_id=None,
            provider_response_id=None,
            created_at=datetime.now(UTC),
        )
        outcome = EvidenceAttachmentOutcome(False, None, "test")
        service._attach = AsyncMock(return_value=outcome)  # type: ignore[method-assign]
        service._update_working_memory_pointer = AsyncMock()  # type: ignore[method-assign]

        result = await service.attach_message(owner_id="owner", message=message)

        self.assertEqual(result, outcome)
        service._attach.assert_awaited_once()
        self.assertEqual(service._attach.await_args.kwargs["owner_id"], "owner")
