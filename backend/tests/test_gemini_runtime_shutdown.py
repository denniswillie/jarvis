from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.realtime.gemini.persistence import FinalTranscript
from app.realtime.gemini.session import GeminiAudioSession
from app.realtime.session_manager import RealtimeSessionManager
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEventBus
from app.runtime.services import JarvisRuntimeServices
from test_gemini_audio_session import Phone, Wire, until
from test_gemini_domain_integration import context, settings


class Resource:
    def __init__(self, label, events):
        self.label, self.events, self.closed = label, events, False

    async def close(self):
        self.closed = True
        self.events.append(self.label + ".closed")


class GptShape:
    """The existing manager contract has close(), but no pre-storage hook."""

    def __init__(self, events):
        self.events = events

    async def close(self, reason):
        self.events.append("gpt.closed")


class GeminiRuntimeShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_gemini_drain_precedes_storage_without_moving_gpt_close(self):
        events, manager = [], RealtimeSessionManager()
        pool = Resource("pool", events)
        release, domain_entered, attachment_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Messages:
            async def create_or_reuse_gemini_message(self, **fields):
                if pool.closed:
                    raise AssertionError("Message insert reached a closed pool")
                return SimpleNamespace(id=uuid4(), **fields)

            async def gemini_message_has_experience_evidence(self, **fields):
                return False

        class Experience(Resource):
            async def attach_message(self, *, owner_id, message):
                attachment_entered.set()
                await release.wait()
                if self.closed or pool.closed:
                    raise AssertionError("Experience attachment outlived storage")
                events.append("message.attached")
                return SimpleNamespace(attached=True)

        experience = Experience("experience", events)
        phone, ctx = Phone(), context()
        session = GeminiAudioSession(
            websocket=phone, context=ctx, settings=settings(), repository=Messages(),
            experience_event_service=experience, connection_factory=Wire,
        )
        await manager.register(audio_session_id=ctx.audio_session_id, owner_id=ctx.owner_id,
                               jarvis_session_id=ctx.jarvis_session_id, session=session)
        await manager.register(audio_session_id="gpt", owner_id="gpt-owner",
                               jarvis_session_id=uuid4(), session=GptShape(events))
        running = asyncio.create_task(session.run())
        await until(lambda: any(event["type"] == "session.created" for event in phone.events))

        async def admitted_work():
            domain_entered.set()
            await release.wait()
            if pool.closed:
                raise AssertionError("Admitted work outlived the database")
            events.append("domain.finished")

        work = asyncio.create_task(admitted_work())
        session._domain_tasks.add(work)
        work.add_done_callback(session._domain_tasks.discard)
        session.journal.submit(FinalTranscript(
            logical_item_id="accepted", role="user", content="A normal conversation fact.",
            occurred_at=datetime.now(UTC), provider_item_id="accepted-input",
        ))
        await domain_entered.wait()
        await attachment_entered.wait()
        services = JarvisRuntimeServices(
            settings=settings(), realtime_session_manager=manager,
            sensory_frame_window=object(), latest_visual_frame_buffer=object(),
            domain_event_bus=DomainEventBus(), model_limiter=PriorityAsyncLimiter(1),
            db_pool=pool, experience_event_service=experience,
        )
        closing = asyncio.create_task(services.close())
        try:
            await until(session._closed.is_set)
            self.assertFalse(pool.closed)
            self.assertFalse(experience.closed)
            self.assertNotIn("gpt.closed", events)
            release.set()
            await closing
            await running
            await services.close()
            self.assertLess(events.index("domain.finished"), events.index("pool.closed"))
            self.assertLess(events.index("message.attached"), events.index("experience.closed"))
            self.assertLess(events.index("experience.closed"), events.index("pool.closed"))
            self.assertLess(events.index("pool.closed"), events.index("gpt.closed"))
            self.assertEqual(events.count("gpt.closed"), 1)
            self.assertEqual(session.diagnostics()["domain_tasks"], 0)
            self.assertFalse(session.journal.diagnostics()["degraded"])
            self.assertFalse((await manager.get_session_diagnostics(ctx.jarvis_session_id))["realtime_connected"])
        finally:
            release.set()
            await session.close("test_finished")
            await asyncio.gather(work, running, closing, return_exceptions=True)

    async def test_pre_drain_blocks_new_opt_in_sessions_but_preserves_gpt_registration(self):
        events, manager = [], RealtimeSessionManager()
        await manager.register(audio_session_id="old-gpt", owner_id="owner",
                               jarvis_session_id=uuid4(), session=GptShape(events))
        await manager.prepare_for_resource_close()
        self.assertEqual(events, [])
        self.assertTrue(await manager.register(audio_session_id="new-gpt", owner_id="owner",
                                                jarvis_session_id=uuid4(), session=GptShape(events)))

        async def prepared():
            events.append("unexpected-new-gemini")

        self.assertFalse(await manager.register(
            audio_session_id="late-gemini", owner_id="owner", jarvis_session_id=uuid4(),
            session=SimpleNamespace(prepare_for_resource_close=prepared),
        ))
        await manager.close_all()
        self.assertEqual(events, ["gpt.closed", "gpt.closed"])

    async def test_one_failed_hook_does_not_skip_other_opt_in_drains(self):
        events, manager = [], RealtimeSessionManager()

        async def failed():
            events.append("failed")
            raise RuntimeError("Controlled drain failure")

        async def completed():
            events.append("completed")

        for label, hook in (("first", failed), ("second", completed)):
            await manager.register(audio_session_id=label, owner_id="owner", jarvis_session_id=uuid4(),
                                   session=SimpleNamespace(prepare_for_resource_close=hook))
        with self.assertRaises(RuntimeError):
            await manager.prepare_for_resource_close()
        self.assertEqual(events, ["failed", "completed"])
        await manager.close_all()
