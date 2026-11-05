"""Prepared adapter only; actual transport/domain fixtures, no external calls."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.realtime.live_delegation_input import LiveDelegationInput
from app.realtime.live_session import LiveAudioSession, create_live_session
from app.tasks.models import TaskError
from app.voice_routes.live_adapter import LiveDelegationWindow, LiveRouteError
from app.voice_routes.live_protocol import LiveDelegationRequest
from test_live_voice_adapter import LiveFixture
import test_live_readonly_delegation as readonly_fixtures


class WindowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = LiveFixture()
        await self.f.start()

    async def asyncTearDown(self):
        await self.f.close()

    async def speech(self, identity, text):
        await self.f.inject(type='session.input_transcript.delta', event_id=identity,
                           delta=text, start_ms=0, end_ms=50)

    async def delegate(self):
        await self.f.inject(type='session.delegation.created', event_id='delegate', offset_ms=25,
                           delegation={'id': 'd1', 'type': 'delegation', 'target': 'client'})

    async def test_freezes_at_actual_reader_event_not_async_consumer_or_offset(self):
        await self.speech('before', ' edit this  please ')
        await self.delegate()
        # Same approximate times, different provider arrival order. The later
        # fragment cannot become the earlier delegation's instruction.
        await self.speech('after', 'actually cancel that')
        await self.f.until(lambda: len(self.f.route.transcript_parts) == 2)
        stream = self.f.route.delegation_requests()
        request = await anext(stream)
        await stream.aclose()
        window = self.f.route.delegation_window(request)
        self.assertEqual([p.fragment.delta for p in window.parts], [' edit this  please '])
        self.assertEqual(window.received_part_count, 1)
        with self.assertRaises(LiveRouteError):
            self.f.route.delegation_window(replace(request, event_id='forged'))

    async def test_window_bounds_whole_fragments_and_exposes_omissions(self):
        for i in range(70):
            await self.speech(f't{i}', f'[{i}]' + '\u03bb' * 510)
        await self.delegate()
        await self.f.until(lambda: 'd1' in self.f.route._delegations)
        window = self.f.route.delegation_window(self.f.route._delegations['d1'])
        all_parts = self.f.route.transcript_parts
        # Existing captions split provider deltas at the 4096-character caption
        # boundary. The window counts those exact recorded parts, not events.
        self.assertEqual(window.received_part_count, len(all_parts))
        self.assertGreaterEqual(window.received_part_count, 70)
        self.assertEqual(window.parts, all_parts[-len(window.parts):])
        self.assertEqual(window.omitted_prefix_parts, len(all_parts) - len(window.parts))
        self.assertGreater(window.omitted_prefix_parts, 6)  # byte bound also applies
        self.assertLessEqual(sum(len(p.fragment.delta.encode()) for p in window.parts), 32768)
        self.assertIn('[69]' + '\u03bb' * 510, ''.join(p.fragment.delta for p in window.parts))

    async def test_many_short_fragments_do_not_cut_the_instruction_opening(self):
        for i in range(180):
            await self.speech(f'short-{i}', f'word{i} ')
        await self.delegate()
        await self.f.until(lambda: 'd1' in self.f.route._delegations)
        window = self.f.route.delegation_window(self.f.route._delegations['d1'])
        self.assertEqual(len(window.parts), 180)
        self.assertEqual(window.omitted_prefix_parts, 0)
        self.assertEqual(''.join(part.fragment.delta for part in window.parts),
                         ''.join(f'word{i} ' for i in range(180)))


class PacketTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = readonly_fixtures.ReadonlyOwnerTests.asyncSetUp
    asyncTearDown = readonly_fixtures.ReadonlyOwnerTests.asyncTearDown

    def packet(self, **changes):
        fixture = LiveFixture()
        self.update = fixture.context(route=self.route)
        args = dict(route=self.route, audio_session_id=self.context.audio_session_id,
            window=LiveDelegationWindow(self.request, self.parts, 8),
            observations=self.observations, live_context=self.update)
        args.update(changes)
        return LiveDelegationInput.from_window(**args)

    async def test_real_committed_fragments_keep_original_text_refs_and_uncertainty(self):
        packet = self.packet()
        data = packet.as_context()
        self.assertEqual(data['dialogue'][0]['delta'], self.parts[0].fragment.delta)
        self.assertEqual(data['dialogue'][0]['context_event_id'], str(self.observations[0].id))
        self.assertEqual(packet.observation_payload_ids, (self.observations[0].payload_id,))
        self.assertEqual(data['window']['omitted_prefix_parts'], 7)
        self.assertEqual(data['window']['speech_finality'], 'unavailable')
        self.assertFalse(self.observations[0].authorizes_request)
        data['dialogue'][0]['delta'] = 'mutated projection'
        self.assertEqual(packet.as_context()['dialogue'][0]['delta'], self.parts[0].fragment.delta)

    async def test_capture_and_packet_keep_more_than_64_originals(self):
        parts, originals = [], []
        for i in range(90):
            part = replace(self.parts[0], transcript_id=uuid4(), revision=1,
                           fragment=replace(self.parts[0].fragment, event_id=f'short-{i}', delta=f'word{i} '))
            original = self.observations[0]
            content = original.content | {'caption_id': str(part.transcript_id), 'caption_revision': 1,
                'provider_event_id': part.fragment.event_id, 'delta': part.fragment.delta}
            record = original.model_copy(update={'id': uuid4(), 'payload_id': uuid4(),
                                                 'sequence': i+1, 'content': content})
            parts.append(part)
            originals.append(record)
            self.captions._originals[self.captions._key(part)] = record
        captured = await self.captions.observations(tuple(parts))
        self.assertEqual(captured, tuple(originals))
        packet = self.packet(window=LiveDelegationWindow(self.request, tuple(parts), len(parts)),
                             observations=captured)
        self.assertEqual(len(packet.as_context()['dialogue']), 90)
        self.assertEqual(packet.as_context()['dialogue'][0]['delta'], 'word0 ')

    async def test_wrong_owner_session_source_and_changed_words_rejected(self):
        record = self.observations[0]
        for changes in ({'owner_id': 'other'}, {'session_id': uuid4()}, {'availability': 'withdrawn'},
                        {'content': record.content | {'delta': 'invented words'}},
                        {'content': record.content | {'request_boundary': 'user_confirmed'}}):
            with self.subTest(fields=list(changes)), self.assertRaises(TaskError):
                self.packet(observations=(record.model_copy(update=changes),))

    async def test_same_queue_handoff_releases_source_guard_and_creates_no_readonly_receipt(self):
        packet = self.packet()
        locked, delivered = False, []
        @asynccontextmanager
        async def guard(update):
            nonlocal locked
            locked = True
            try:
                yield self.db
            finally:
                locked = False
        async def handler(value, *, publish, is_current):
            self.assertFalse(locked)
            self.assertTrue(is_current())
            self.assertIs(publish, publish_result)
            delivered.append(value)
            session._closed.set()
        publish_result = AsyncMock()
        session = SimpleNamespace(ready=True, capture=self.captions, work=None,
            live=SimpleNamespace(service=SimpleNamespace(snapshot=lambda: object()), bind=object()),
            request=SimpleNamespace(route=self.route), context=self.context, _source_guard=guard,
            delegation_handler=handler, adapter=SimpleNamespace(submit_delegation_result=publish_result),
            _closed=asyncio.Event(), _readonly_queue=asyncio.Queue(), _readonly_receipts={}, _readonly_jobs={},
            _readonly_notice=AsyncMock())
        session._handoff_delegation = lambda window: LiveAudioSession._handoff_delegation(session, window)
        session._readonly_queue.put_nowait(packet.window)
        with patch('app.realtime.live_session.select_voice_context', return_value=object()), \
                patch('app.realtime.live_session.prepare_voice_context', AsyncMock(return_value=self.update)):
            await LiveAudioSession._admit_readonly_requests(session)
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].context_json, packet.context_json)
        self.assertFalse(session._readonly_receipts)
        self.assertFalse(session._readonly_jobs)
        session._readonly_notice.assert_not_awaited()
        publish_result.assert_not_awaited()  # No fabricated accepted/saved reply.

    async def test_handoff_timeout_does_not_assert_no_task_was_admitted(self):
        closed = asyncio.Event()
        async def handoff(window):
            closed.set()
            raise TimeoutError()
        session = SimpleNamespace(_closed=closed, delegation_handler=object(),
            _readonly_queue=asyncio.Queue(), _handoff_delegation=handoff, _readonly_notice=AsyncMock())
        session._readonly_queue.put_nowait(self.packet().window)
        await LiveAudioSession._admit_readonly_requests(session)
        self.assertIsNone(session._readonly_notice.await_args.kwargs['admitted'])


class ResultTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = readonly_fixtures.ReadonlyResultTransportTests.asyncSetUp
    asyncTearDown = readonly_fixtures.ReadonlyResultTransportTests.asyncTearDown
    result = readonly_fixtures.ReadonlyResultTransportTests.result
    ready = readonly_fixtures.ReadonlyResultTransportTests.ready
    guard = readonly_fixtures.ReadonlyResultTransportTests.guard

    async def test_acceptance_then_actual_completion_each_source_guarded_replay_deduplicated(self):
        await self.ready()
        accepted = replace(self.result('Accepted for background work.'), kind='delegation_result')
        completed = replace(accepted, result_payload_id=uuid4(), text='The corrected file was saved.',
                            task_revision=2, roots=(uuid4(),))
        first = await self.f.route.submit_delegation_result(accepted, guard=self.guard)
        final = await self.f.route.submit_delegation_result(completed, guard=self.guard)
        self.assertTrue(first.provider_received and final.provider_received)
        before = len(self.f.wire.socket.sent)
        self.assertEqual(await self.f.route.submit_delegation_result(accepted, guard=self.guard), first)
        self.assertEqual(len(self.f.wire.socket.sent), before)
        self.assertEqual(self.guarded, [accepted, completed])
        with self.assertRaises(LiveRouteError):
            await self.f.route.submit_delegation_result(replace(completed, text='False new text'), guard=self.guard)
        self.assertFalse(hasattr(final, 'heard'))

    async def test_all_numbered_chunks_correlated_and_same_payload_on_other_delegation_independent(self):
        await self.ready()
        result = replace(self.result('actual bounded result ' * 70), kind='delegation_result')
        await self.f.route.submit_delegation_result(result, guard=self.guard)
        frames = [item for item in self.f.wire.socket.sent if item['type'] in
                  {'session.thinking.append', 'session.commentary.append'}]
        # Startup context may be uncorrelated. Restrict to this actual reply.
        frames = [item for item in frames if item['event_id'].startswith('dr_')]
        self.assertGreater(len(frames), 1)
        self.assertTrue(all(frame['delegation_id'] == 'd1' for frame in frames))
        self.assertTrue(all(json.loads(frame['content'])['result_id'] == str(result.result_payload_id)
                            for frame in frames))
        await self.f.inject(type='session.delegation.created', event_id='delegate2', offset_ms=30,
                           delegation={'id': 'd2', 'type': 'delegation', 'target': 'client'})
        await self.f.until(lambda: 'd2' in self.f.route._delegations)
        second = await self.f.route.submit_delegation_result(replace(result, delegation_id='d2'), guard=self.guard)
        self.assertTrue(second.provider_received)

    async def test_revoked_delegation_blocks_later_result_payload(self):
        await self.ready()
        self.f.route.revoke_delegation_result('d1')
        self.assertFalse(self.f.route.requires_fenced_replacement)
        receipt = await self.f.route.submit_delegation_result(self.result(), guard=self.guard)
        self.assertEqual(receipt.status, 'rejected')
        self.assertEqual(self.guarded, [])

    async def test_revoking_submitted_result_requires_route_replacement(self):
        await self.ready()
        receipt = await self.f.route.submit_delegation_result(self.result(), guard=self.guard)
        self.assertTrue(receipt.provider_received)
        self.f.route.revoke_delegation_result('d1')
        self.assertTrue(self.f.route.requires_fenced_replacement)


class FactoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_factory_forwards_server_handler_without_starting_it(self):
        handler = AsyncMock()
        session = object()
        with patch("app.realtime.live_session.LiveAudioSession", return_value=session) as constructor:
            result = await create_live_session(websocket=object(), context=object(),
                settings=object(), repository=object(), runtime_services=object(),
                offer="unused", capture_mode="auto", client_capabilities={},
                delegation_handler=handler)
        self.assertIs(result, session)
        self.assertIs(constructor.call_args.kwargs["delegation_handler"], handler)
        handler.assert_not_awaited()

    async def test_default_factory_keeps_readonly_composition(self):
        with patch("app.realtime.live_session.LiveAudioSession") as constructor:
            await create_live_session(websocket=object(), context=object(), settings=object(),
                repository=object(), runtime_services=object(), offer="unused",
                capture_mode="auto", client_capabilities={})
        self.assertIsNone(constructor.call_args.kwargs["delegation_handler"])
