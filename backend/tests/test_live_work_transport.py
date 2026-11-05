import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.realtime.live_session import create_live_session
from app.realtime.live_work import LiveWorkBridge, LiveWorkRuntime
from app.tasks.models import TaskError


class LiveWorkTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_factory_installs_work_without_starting_a_provider(self):
        context = SimpleNamespace(owner_id="fixture", jarvis_session_id=uuid4())
        services = SimpleNamespace(background_task_service=SimpleNamespace(repository=SimpleNamespace(pool=None)))
        def construct(**values):
            return SimpleNamespace(context=context, services=services, application=SimpleNamespace(),
                configuration=SimpleNamespace(model="gpt-live-1", voice="marin"))
        with patch("app.realtime.live_session.LiveAudioSession", side_effect=construct):
            session = await create_live_session(websocket=object(), context=context, settings=object(),
                repository=object(), runtime_services=services, offer="test", capture_mode="auto", client_capabilities={})
        self.assertIsInstance(session.work, LiveWorkBridge)
        self.assertIs(session.delegation_handler, session.work)
        self.assertFalse(session.work._started)
        self.assertIsNone(session.work.receipt)
        self.assertIn("Delegate", session.configuration.instructions)
        await services.live_work_runtime.close()

    async def test_explicit_handler_is_not_replaced(self):
        handler = AsyncMock()
        sentinel = object()
        with patch("app.realtime.live_session.LiveAudioSession", return_value=sentinel) as constructor:
            session = await create_live_session(websocket=object(), context=object(), settings=object(),
                repository=object(), runtime_services=SimpleNamespace(), offer="test", capture_mode="auto",
                client_capabilities={}, delegation_handler=handler)
        self.assertIs(session, sentinel)
        self.assertIs(constructor.call_args.kwargs["delegation_handler"], handler)

    async def test_captions_only_forward_user_fragments_and_idle_does_not_spin_on_old_input(self):
        bridge = object.__new__(LiveWorkBridge)
        assistant = SimpleNamespace(fragment=SimpleNamespace(speaker="assistant", event_id="assistant"))
        user = SimpleNamespace(fragment=SimpleNamespace(speaker="user", event_id="user"))
        bridge.pending_parts, bridge.seen_parts = (assistant, user), set()
        bridge._closed = bridge._detached = False
        bridge.receipt, bridge.route = SimpleNamespace(task=SimpleNamespace(id=uuid4())), object()
        async def observations(parts):
            self.assertEqual(parts, (user,))
            return ("original",)
        async def append(*args):
            bridge._closed = True
            return False
        bridge.session = SimpleNamespace(capture=SimpleNamespace(_key=lambda part: part.fragment.event_id,
                                                                   observations=observations))
        bridge.mailbox = SimpleNamespace(append_live_observations=AsyncMock(side_effect=append))
        await bridge._captions()
        self.assertEqual(bridge.seen_parts, {"user"})
        bridge.mailbox.append_live_observations.assert_awaited_once()

    async def test_audio_detach_wakes_delivery_without_cancelling_work(self):
        bridge = object.__new__(LiveWorkBridge)
        bridge._closed = bridge._detached = False
        bridge._started = True
        bridge._outbox = asyncio.Queue(maxsize=16)
        bridge.session = SimpleNamespace(ready=True)
        bridge.publish = AsyncMock()
        bridge.service = SimpleNamespace(cancel=AsyncMock())
        waiting = asyncio.create_task(bridge._deliver())
        await asyncio.sleep(0)
        bridge.detach()
        await asyncio.wait_for(waiting, .5)
        bridge.publish.assert_not_awaited()
        bridge.service.cancel.assert_not_awaited()

    async def test_followup_uses_explicit_work_not_a_newest_task_lookup(self):
        bridge = object.__new__(LiveWorkBridge)
        bridge._closed = False
        bridge.runtime = SimpleNamespace(closed=False)
        bridge.route = object()
        bridge.receipt = SimpleNamespace(task=SimpleNamespace(id=uuid4()))
        bridge.session = SimpleNamespace(capture=SimpleNamespace(_key=lambda value: value))
        bridge.mailbox = SimpleNamespace(append_live=AsyncMock())
        bridge.seen_parts = set()
        bridge._started = True
        packet = SimpleNamespace(route=bridge.route, window=SimpleNamespace(parts=("part",),
            request=SimpleNamespace(delegation_id="second")))
        await bridge(packet, publish=AsyncMock(), is_current=lambda: True)
        bridge.mailbox.append_live.assert_awaited_once_with(bridge.receipt.task.id, packet)
        self.assertEqual(bridge.delegation_id, "second")
        self.assertEqual(bridge.seen_parts, {"part"})

    async def test_closed_runtime_rejects_new_input_before_storage(self):
        bridge = object.__new__(LiveWorkBridge)
        bridge._closed = False
        bridge.runtime = SimpleNamespace(closed=True)
        with self.assertRaises(TaskError):
            await bridge(object(), publish=AsyncMock(), is_current=lambda: True)

    async def test_runtime_close_joins_transport_not_worker_execution(self):
        runtime = LiveWorkRuntime()
        binding = SimpleNamespace(close=AsyncMock())
        # SimpleNamespace is unhashable; a normal binding is identity-hashable.
        class Binding:
            close = binding.close
        instance = Binding()
        runtime.bindings.add(instance)
        await runtime.close()
        self.assertTrue(runtime.closed)
        binding.close.assert_awaited_once()
