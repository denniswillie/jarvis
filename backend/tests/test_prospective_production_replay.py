"""R09 replay: real pipeline/service/adapters plus durable prospective records.

Only perception inference, frame/observation storage, and native network/phone
I/O are controlled fixtures. No fixture fabricates a delivery outcome or focus
completion. Both audio adapters must resolve the real repository themselves.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime, timedelta
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.experience.repository import ExperienceRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.memory.prospective.repository import ProspectiveRepository
from app.memory.prospective.service import ProspectiveMemoryService
from app.perception.contracts import FastProspectiveMatch
from app.realtime.gemini.session import GeminiAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.realtime.session_manager import RealtimeSessionManager
from app.reminders.repository import ReminderRepository
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.visual_memory.interrupts import VisualInterruptService
from evals.glasses_reliability.search_tool_probe import FocusOnlyContext, NativePhone
from evals.gemini_live.interruption import DatabaseFixtures
from evals.perception_memory.fixture_cleanup import cleanup_fixture_owners
from test_fast_perception_pipeline import _Harness
from test_gemini_audio_session import Wire as GeminiWire, ENCODED


async def until(predicate, timeout=10):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.005)


class GPTWire:
    def __init__(self):
        self.sent = []

    async def send_event(self, event):
        payload = event() if callable(event) else event
        if payload is None:
            return False
        self.sent.append(payload)
        return True

    async def close(self):
        pass


@unittest.skipUnless(os.getenv("JARVIS_TEST_DB_URL"), "Explicit isolated-owner database run required")
class ProspectiveProductionReplayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        output = Path(tempfile.mkdtemp(
            prefix="prospective-replay-", dir=os.getenv("JARVIS_TEST_ARTIFACT_DIR"),
        ))
        self.fixtures = DatabaseFixtures(os.environ["JARVIS_TEST_DB_URL"], output)
        self.addAsyncCleanup(self.cleanup)
        await self.fixtures.connect()

    async def cleanup(self):
        await self.fixtures.cleanup()
        self.assertEqual(self.fixtures.cleanup_result["remaining_rows"], 0)

    @asynccontextmanager
    async def environment(self, provider, *, mode="find_once"):
        owner, session_id = await self.fixtures.create("r09-" + provider)
        async with AsyncExitStack() as stack:
            stack.push_async_callback(cleanup_fixture_owners, self.fixtures.dsn, {owner})
            settings = JarvisSettings(_env_file=None, GEMINI_API_KEY="fixture-key")
            repository = ProspectiveRepository(self.fixtures.pool)
            sessions = SessionRepository(self.fixtures.pool)
            manager = RealtimeSessionManager()
            service = ProspectiveMemoryService(repository=repository,
                                               realtime_session_manager=manager, settings=settings)
            focus = await service.set_visual_focus(
                owner_id=owner, target="toilet roll", description="white toilet roll",
                purpose="Tell me where it is", mode=mode, expires_in_seconds=3600,
            )
            phone, wires = NativePhone(), []
            context = RealtimeSessionContext(
                audio_session_id=str(uuid4()), owner_id=owner, jarvis_session_id=session_id,
                client_kind="phone_audio", session=await sessions.get_session(owner, session_id),
            )
            experience = ExperienceEventService(
                repository=ExperienceRepository(self.fixtures.pool),
                segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)),
            )
            stack.push_async_callback(experience.close)
            running = None
            if provider == "gemini":
                def factory(**kwargs):
                    wire = GeminiWire(**kwargs)
                    wires.append(wire)
                    return wire
                session = GeminiAudioSession(
                    websocket=phone, context=context, settings=settings, repository=sessions,
                    connection_factory=factory, experience_event_service=experience,
                )
                running = asyncio.create_task(session.run())
                async def join():
                    await asyncio.wait_for(running, 10)
                stack.push_async_callback(join)
                stack.push_async_callback(session.close, "replay_finished")
                await until(lambda: session._ready)
            else:
                session = RealtimeAudioSession(
                    websocket=phone, context=context, settings=settings, repository=sessions,
                    experience_event_service=experience, tool_modules=(), context_providers=(),
                )
                session.openai = GPTWire()
                wires.append(session.openai)
                session._realtime_context_ready.set()
                stack.push_async_callback(session.close, "replay_finished")
            await manager.register(audio_session_id=context.audio_session_id, owner_id=owner,
                                   jarvis_session_id=session_id, session=session)
            stack.push_async_callback(manager.unregister, audio_session_id=context.audio_session_id,
                                      owner_id=owner, jarvis_session_id=session_id)
            harness = _Harness()
            stack.push_async_callback(harness.pipeline.close)
            harness.owner_id, harness.session_id = owner, session_id
            harness.pipeline._prospective_service = service
            harness.pipeline._context_builder._session_repository = sessions
            harness.pipeline._context_builder._prospective_context_builder = FocusOnlyContext(repository)
            generic = SimpleNamespace(
                recent_visual_interrupt_exists=AsyncMock(return_value=False),
                create_visual_interrupt=AsyncMock(side_effect=lambda **kw: SimpleNamespace(id=uuid4())),
            )
            harness.pipeline._interrupt_service = VisualInterruptService(
                repository=generic, realtime_session_manager=manager, dedupe_seconds=10,
            )
            original = harness.fast_analyzer.analyze
            captured = []

            async def analyze(request):
                output = await original(request)
                output = output.model_copy(update={
                    "interrupt_candidate": output.interrupt_candidate.model_copy(update={
                        "message": "The toilet roll is on the desk.", "reason": "Tracked target visible",
                    }),
                    "prospective_matches": [FastProspectiveMatch(
                        subject_type="visual_focus", subject_id=focus.id, confidence=.99,
                        observed_condition="The toilet roll is on the desk.",
                        supporting_frame_ids=[request.primary_frame.frame_id],
                        suggested_message="The toilet roll is on the desk.",
                    )],
                })
                captured.append((request, output))
                return output

            harness.fast_analyzer.analyze = analyze
            env = SimpleNamespace(
                provider=provider, owner=owner, session_id=session_id, settings=settings,
                repository=repository, service=service, manager=manager, focus=focus,
                phone=phone, session=session, wires=wires, running=running, harness=harness,
                generic=generic, captured=captured,
            )
            yield env

    async def enqueue(self, env, *, sequence=1):
        frame = env.harness.frame(sequence=sequence, captured_at=datetime.now(UTC))
        env.harness.put(frame, b"controlled-perception-fixture")
        await env.harness.pipeline.enqueue(frame=frame, image_bytes=b"controlled-perception-fixture")
        await asyncio.wait_for(env.harness.pipeline.drain(), 10)
        self.assertEqual(env.harness.visual_repository.failed, [])
        return frame

    def creates(self, env):
        if env.provider == "openai":
            return [e for e in env.wires[0].sent if e.get("type") == "response.create"]
        return [e for wire in env.wires for e in wire.sent
                if e.get("clientContent", {}).get("turnComplete")]

    async def finish(self, env, *, interrupted=False):
        if env.provider == "gemini":
            await env.wires[-1].publish(
                {"interrupted": True, "turnComplete": True} if interrupted else {
                    "modelTurn": {"parts": [{"inlineData": {
                        "mimeType": "audio/pcm;rate=24000", "data": ENCODED,
                    }}]},
                    "outputTranscription": {"text": "The toilet roll is on the desk."},
                    "generationComplete": True, "turnComplete": True,
                }
            )
        else:
            response_id = "replay-" + uuid4().hex
            await env.session._handle_openai_event({
                "type": "response.created", "response": {
                    "id": response_id, "metadata": self.creates(env)[-1]["response"]["metadata"],
                },
            })
            if interrupted:
                await env.session._handle_client_event({"type": "response.cancel"})
            else:
                await env.session._handle_openai_event({
                    "type": "response.output_audio.delta", "response_id": response_id,
                    "item_id": response_id + "-audio", "delta": ENCODED,
                })
                await env.session._handle_openai_event({
                    "type": "response.output_audio.done", "response_id": response_id,
                    "item_id": response_id + "-audio",
                })
            await env.session._handle_openai_event({
                "type": "response.done", "response": {
                    "id": response_id, "status": "cancelled" if interrupted else "completed",
                },
            })
        await until(lambda: not env.service._pending_deliveries)

    async def events(self, env):
        async with self.fixtures.pool.acquire() as conn:
            return [dict(row) for row in await conn.fetch(
                "select id, visual_focus_id, reminder_id, event_type, metadata, provider_response_id "
                "from jarvis_prospective_delivery_events "
                "where owner_id=$1 order by created_at, id", env.owner,
            )]

    async def test_r09_one_forwarded_alert_completes_find_once_and_late_detection_cannot_repeat(self):
        for provider in ("openai", "gemini"):
            with self.subTest(provider=provider):
                async with self.environment(provider) as env:
                    await self.enqueue(env)
                    self.assertEqual(len(self.creates(env)), 1)
                    env.generic.create_visual_interrupt.assert_not_awaited()
                    self.assertIsNotNone(await env.repository.get_active_visual_focus(owner_id=env.owner))
                    await self.finish(env)
                    self.assertIsNone(await env.repository.get_active_visual_focus(owner_id=env.owner))
                    self.assertEqual([r["event_type"] for r in await self.events(env)], ["delivered_to_session"])
                    self.assertEqual(len([e for e in env.phone.events if e["type"] == "audio.output.delta"]), 1)
                    await self.enqueue(env, sequence=2)
                    self.assertEqual(len(self.creates(env)), 1)
                    self.assertEqual(len(await self.events(env)), 1)

    async def test_both_route_orderings_and_simultaneous_candidates_have_one_owner(self):
        for provider in ("openai", "gemini"):
            for order in ("generic_first", "tracked_first", "simultaneous"):
                with self.subTest(provider=provider, order=order):
                    async with self.environment(provider) as env:
                        frame = env.harness.frame(sequence=1, captured_at=datetime.now(UTC))
                        env.harness.put(frame, b"fixture")
                        context = await env.harness.pipeline._context_builder.build(frame)
                        request = env.harness.pipeline._build_fast_request(
                            frame=frame, image_bytes=b"fixture", context=context,
                        )
                        output = await env.harness.fast_analyzer.analyze(request)
                        async def generic():
                            await env.harness.pipeline._apply_completed_result(
                                frame=frame, output=output, send_diagnostic=None, observation_id=uuid4(),
                            )
                        async def tracked():
                            await env.service.process_matches(request=request, matches=output.prospective_matches)
                        if order == "generic_first":
                            await generic()
                            await tracked()
                        elif order == "tracked_first":
                            await tracked()
                            await generic()
                        else:
                            await asyncio.gather(generic(), tracked(), tracked())
                        self.assertEqual(len(self.creates(env)), 1)
                        env.generic.create_visual_interrupt.assert_not_awaited()
                        await self.finish(env)
                        self.assertEqual([e["event_type"] for e in await self.events(env)], ["delivered_to_session"])

    async def test_monitor_keeps_existing_dedupe_then_can_notify_again(self):
        for provider in ("openai", "gemini"):
            with self.subTest(provider=provider):
                async with self.environment(provider, mode="monitor") as env:
                    await self.enqueue(env)
                    await self.finish(env)
                    await self.enqueue(env, sequence=2)
                    self.assertEqual(len(self.creates(env)), 1)
                    async with self.fixtures.pool.acquire() as conn:
                        await conn.execute(
                            "update jarvis_prospective_delivery_events set created_at = now() - interval '11 seconds' "
                            "where owner_id=$1", env.owner,
                        )
                    await self.enqueue(env, sequence=3)
                    self.assertEqual(len(self.creates(env)), 2)
                    await self.finish(env)
                    self.assertEqual(len(await self.events(env)), 2)
                    self.assertEqual((await env.repository.get_active_visual_focus(owner_id=env.owner)).id,
                                     env.focus.id)

    async def test_clear_and_replace_while_dispatch_is_queued_never_speak_old_target(self):
        for provider in ("openai", "gemini"):
            for action in ("clear", "replace"):
                with self.subTest(provider=provider, action=action):
                    async with self.environment(provider) as env:
                        entered, release = asyncio.Event(), asyncio.Event()
                        begin = env.repository.begin_delivery
                        async def delayed(**kw):
                            result = await begin(**kw)
                            entered.set()
                            await release.wait()
                            return result
                        env.repository.begin_delivery = delayed
                        task = asyncio.create_task(self.enqueue(env))
                        try:
                            await asyncio.wait_for(entered.wait(), 5)
                            if action == "clear":
                                await env.service.clear_visual_focus(owner_id=env.owner)
                            else:
                                await env.service.set_visual_focus(
                                    owner_id=env.owner, target="keys", description="silver keys",
                                    purpose="find them", mode="find_once", expires_in_seconds=3600,
                                )
                        finally:
                            release.set()
                            await task
                        await until(lambda: not env.service._pending_deliveries)
                        self.assertFalse(self.creates(env))
                        self.assertFalse(env.phone.audio)
                        events = await self.events(env)
                        self.assertEqual(len(events), 1)
                        self.assertEqual(events[0]["event_type"], "failed")
                        metadata = json.loads(events[0]["metadata"])
                        self.assertEqual(metadata["reason"], "focus_cleared" if action == "clear" else "focus_replaced")

    async def test_interrupted_and_disconnected_deliveries_do_not_complete_search(self):
        for provider in ("openai", "gemini"):
            for outcome in ("interrupted", "disconnected"):
                with self.subTest(provider=provider, outcome=outcome):
                    async with self.environment(provider) as env:
                        await self.enqueue(env)
                        if outcome == "interrupted":
                            await self.finish(env, interrupted=True)
                        else:
                            await env.session.close("peer_closed")
                            await until(lambda: not env.service._pending_deliveries)
                        self.assertIsNotNone(await env.repository.get_active_visual_focus(owner_id=env.owner))
                        self.assertEqual([r["event_type"] for r in await self.events(env)], ["interrupted"])

    async def test_rejected_delivery_retries_on_later_evidence_without_generic_fallback(self):
        for provider in ("openai", "gemini"):
            with self.subTest(provider=provider):
                async with self.environment(provider) as env:
                    if provider == "openai":
                        env.session._realtime_context_ready.clear()
                    else:
                        env.session._ready = False
                    await self.enqueue(env)
                    await until(lambda: not env.service._pending_deliveries)
                    first = (await self.events(env))[0]
                    self.assertIn(first["event_type"], ("failed", "missed"))
                    self.assertTrue(json.loads(first["metadata"])["reason"])
                    self.assertFalse(self.creates(env))
                    env.generic.create_visual_interrupt.assert_not_awaited()
                    if provider == "openai":
                        env.session._realtime_context_ready.set()
                    else:
                        env.session._ready = True
                    async with self.fixtures.pool.acquire() as conn:
                        await conn.execute(
                            "update jarvis_prospective_delivery_events set created_at = now() - interval '11 seconds' "
                            "where owner_id=$1", env.owner,
                        )
                    await self.enqueue(env, sequence=2)
                    late = await env.repository.resolve_delivery_event(
                        owner_id=env.owner, event_id=first["id"], outcome="delivered",
                        provider_response_id="stale-contradictory-callback",
                    )
                    self.assertEqual(late.event_type, first["event_type"])
                    self.assertIsNotNone(await env.repository.get_active_visual_focus(owner_id=env.owner))
                    await self.finish(env)
                    self.assertEqual(len(self.creates(env)), 1)
                    self.assertEqual([e["event_type"] for e in await self.events(env)],
                                     [first["event_type"], "delivered_to_session"])
                    self.assertIsNone(await env.repository.get_active_visual_focus(owner_id=env.owner))

    async def test_conditional_reminder_owns_its_alert_and_is_not_cancelled_with_visual_focus(self):
        for provider in ("openai", "gemini"):
            with self.subTest(provider=provider):
                async with self.environment(provider) as env:
                    reminders = ReminderRepository(self.fixtures.pool)
                    reminder = await reminders.create_reminder(
                        owner_id=env.owner, session_id=env.session_id,
                        title="Synthetic conditional reminder", details="Tell me when the toilet roll is visible",
                        remind_at=datetime.now(UTC) + timedelta(days=30),
                        condition="The toilet roll is visible", metadata={"synthetic_fixture": True},
                    )
                    builder = env.harness.pipeline._context_builder._prospective_context_builder
                    async def build(**kw):
                        base = await builder.build(**kw)
                        return replace(base, conditional_reminders=tuple(
                            await reminders.list_active_reminders(owner_id=env.owner)
                        ))
                    env.harness.pipeline._context_builder._prospective_context_builder = SimpleNamespace(build=build)
                    analyze = env.harness.fast_analyzer.analyze
                    async def conditional(request):
                        output = await analyze(request)
                        return output.model_copy(update={"prospective_matches": [
                            output.prospective_matches[0].model_copy(update={
                                "subject_type": "conditional_reminder", "subject_id": reminder.id,
                            })
                        ]})
                    env.harness.fast_analyzer.analyze = conditional
                    await self.enqueue(env)
                    async with self.fixtures.pool.acquire() as conn:
                        await conn.execute(
                            "update jarvis_reminders set trigger_claimed_at = now() - interval '5 minutes' "
                            "where owner_id=$1 and id=$2", env.owner, reminder.id,
                        )
                    self.assertIsNone(await env.repository.try_claim_subject(
                        owner_id=env.owner, subject_type="conditional_reminder", subject_id=reminder.id,
                        claim_timeout_seconds=120,
                    ))
                    await env.service.clear_visual_focus(owner_id=env.owner)
                    self.assertFalse(any(a.revoked for a in env.service._pending_deliveries.values()))
                    await self.finish(env)
                    event = (await self.events(env))[0]
                    self.assertEqual(event["reminder_id"], reminder.id)
                    self.assertIsNone(event["visual_focus_id"])
                    self.assertEqual(event["event_type"], "delivered_to_session")
                    async with self.fixtures.pool.acquire() as conn:
                        row = await conn.fetchrow(
                            "select status, triggered_at, trigger_claim_token from jarvis_reminders "
                            "where owner_id=$1 and id=$2", env.owner, reminder.id,
                        )
                    self.assertEqual(row["status"], "completed")
                    self.assertIsNotNone(row["triggered_at"])
                    self.assertIsNone(row["trigger_claim_token"])
                    env.generic.create_visual_interrupt.assert_not_awaited()

    async def test_gpt_ambiguous_write_preserves_durable_fence_until_late_success(self):
        async with self.environment("openai") as env:
            send = env.wires[0].send_event
            async def ambiguous(event):
                sent = await send(event)
                if sent and env.wires[0].sent[-1]["type"] == "response.create":
                    raise RuntimeError("Controlled acknowledgement loss")
                return sent
            env.wires[0].send_event = ambiguous
            await self.enqueue(env)
            self.assertEqual([r["event_type"] for r in await self.events(env)], ["dispatched_to_session"])
            active = await env.repository.get_active_visual_focus(owner_id=env.owner)
            self.assertEqual(active.delivery_status["status"], "awaiting_outcome")
            self.assertTrue(env.session._pending_response.submitted)
            self.assertTrue(any(e.get("code") == "prospective_delivery_outcome_unknown"
                                for e in env.phone.events))
            env.wires[0].send_event = send
            await self.finish(env)
            self.assertEqual([r["event_type"] for r in await self.events(env)], ["delivered_to_session"])
            self.assertEqual(len(self.creates(env)), 1)

    async def test_timeout_is_not_reported_as_success_or_replayed(self):
        for provider in ("openai", "gemini"):
            with self.subTest(provider=provider):
                async with self.environment(provider) as env:
                    await self.enqueue(env)
                    if provider == "openai":
                        request_id = env.session._pending_response.request_id
                        with patch("app.realtime.session.OPENAI_RESPONSE_START_TIMEOUT_SECONDS", .001):
                            await env.session._watch_response_start(request_id)
                        active = await env.repository.get_active_visual_focus(owner_id=env.owner)
                        self.assertEqual(active.delivery_status["status"], "awaiting_outcome")
                        async with self.fixtures.pool.acquire() as conn:
                            await conn.execute(
                                "update jarvis_visual_focuses set trigger_claimed_at = now() - interval '5 minutes' "
                                "where owner_id=$1 and id=$2", env.owner, env.focus.id,
                            )
                        await self.enqueue(env, sequence=2)
                    else:
                        await env.session._queue.put(("response_timeout", env.session._pending_responses[0]))
                        await asyncio.wait_for(env.running, 10)
                        self.assertEqual([r["event_type"] for r in await self.events(env)], ["interrupted"])
                    self.assertEqual(len(self.creates(env)), 1)
                    self.assertIsNotNone(await env.repository.get_active_visual_focus(owner_id=env.owner))
