"""Regression cases for exact prospective speech ownership on both adapters."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.memory.prospective.delivery import ProspectiveDeliveryAttempt
from app.memory.prospective.service import ProspectiveMemoryService
from app.perception.contracts import FastProspectiveMatch
from app.realtime.session_manager import RealtimeSessionManager
import test_generic_visual_speech as gpt_fixture
import test_gemini_proactive as gemini_fixture
from test_fast_perception_pipeline import _Harness
from test_prospective_service import _request, _candidate, _settings


def attempt(*, persist=None, preflight=None, **values):
    return ProspectiveDeliveryAttempt(
        owner_id="owner", event_id=uuid4(), subject_type="visual_focus",
        subject_id=uuid4(), preflight=preflight or AsyncMock(return_value=True),
        persist=persist or AsyncMock(), **values,
    )


def repository():
    return SimpleNamespace(
        try_claim_subject=AsyncMock(return_value=uuid4()),
        reserve_delivery_event=AsyncMock(side_effect=lambda **kw: SimpleNamespace(id=uuid4())),
        release_subject_claim=AsyncMock(), begin_delivery=AsyncMock(return_value=True),
        resolve_delivery_event=AsyncMock(return_value=SimpleNamespace()),
        mark_event_dispatched=AsyncMock(),
    )


class DeliveryAttemptTests(unittest.IsolatedAsyncioTestCase):
    async def test_expired_attempt_is_rejected_before_database_preflight(self):
        delivery = attempt(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        self.assertFalse(await delivery.prepare())
        delivery.preflight.assert_not_awaited()
        self.assertEqual(delivery.reason, "focus_expired")

    async def test_failed_persistence_retries_the_original_success_not_close_interruption(self):
        persist = AsyncMock(side_effect=[RuntimeError("storage unavailable"), None])
        delivery = attempt(persist=persist)
        with self.assertRaises(RuntimeError):
            await delivery.on_outcome("delivered", "response-1")
        delivery.revoke("focus_replaced")
        await delivery.on_outcome("interrupted", None)
        self.assertEqual([call.args for call in persist.await_args_list],
                         [("delivered", "response-1"), ("delivered", "response-1")])
        self.assertTrue(delivery.settled)
        self.assertEqual(delivery.reason, "completed_forwarded_answer")

    async def test_revocation_during_preflight_wins(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def preflight():
            entered.set()
            await release.wait()
            return True

        delivery = attempt(preflight=preflight)
        task = asyncio.create_task(delivery.prepare())
        await asyncio.wait_for(entered.wait(), 1)
        delivery.revoke("focus_cleared")
        release.set()
        self.assertFalse(await asyncio.wait_for(task, 1))
        self.assertEqual(delivery.reason, "focus_cleared")

    async def test_concurrent_callbacks_persist_once(self):
        delivery = attempt()
        await asyncio.gather(delivery.on_outcome("delivered", "response"),
                             delivery.on_outcome("missed", None))
        delivery.persist.assert_awaited_once_with("delivered", "response")


class ServiceOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_errors_before_provider_ownership_release_the_reserved_claim(self):
        for error in (RuntimeError("admission failure"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                repo = repository()
                async def speak(**kw):
                    raise error
                service = ProspectiveMemoryService(
                    repository=repo,
                    realtime_session_manager=SimpleNamespace(speak_prospective_interrupt=speak),
                    settings=_settings(max_candidates=2),
                )
                request = _request()
                with self.assertRaises(type(error)):
                    await service.process_matches(request=request, matches=[_candidate(request, confidence=.99)])
                result = repo.resolve_delivery_event.await_args.kwargs
                self.assertEqual(result["outcome"], "failed")
                self.assertEqual(result["metadata"]["reason"],
                                 "admission_cancelled" if isinstance(error, asyncio.CancelledError) else "admission_failed")
                self.assertEqual(service._pending_deliveries, {})

    async def test_application_focus_expiry_is_carried_to_the_delivery_attempt(self):
        repo, captured = repository(), []

        async def speak(**kw):
            captured.append(kw["delivery_attempt"])
            return None

        request = _request()
        candidate = _candidate(request, confidence=.99)
        expiry = datetime.now(UTC) + timedelta(minutes=5)
        request = replace(request, active_focus={
            "id": str(candidate.subject_id), "expires_at": expiry.isoformat(),
        })
        service = ProspectiveMemoryService(
            repository=repo,
            realtime_session_manager=SimpleNamespace(speak_prospective_interrupt=speak),
            settings=_settings(max_candidates=2),
        )
        await service.process_matches(request=request, matches=[candidate])
        self.assertEqual(captured[0].expires_at, expiry)

    async def test_explicit_stop_revokes_expired_pending_work_and_bounds_phone_notification(self):
        repo = repository()
        repo.clear_active_visual_focus = AsyncMock(return_value=None)
        service = ProspectiveMemoryService(
            repository=repo, realtime_session_manager=RealtimeSessionManager(),
            settings=_settings(max_candidates=2),
        )
        delivery = attempt()
        delivery.on_revoke = asyncio.Event().wait
        service._pending_deliveries[delivery.event_id] = delivery
        with patch("app.memory.prospective.service.REVOCATION_NOTIFICATION_TIMEOUT_SECONDS", .01):
            with self.assertLogs("app.memory.prospective.service", level="WARNING"):
                self.assertIsNone(await service.clear_visual_focus(owner_id="owner"))
        self.assertTrue(delivery.revoked)
        self.assertFalse(delivery.revocation_notified)

    async def test_provider_owned_async_rejection_is_not_overwritten_by_missed(self):
        repo = repository()
        deliveries = []

        async def speak(**kw):
            delivery = kw["delivery_attempt"]
            delivery.provider_owned = True
            delivery.reason = "user_turn_busy"
            deliveries.append(delivery)
            return None

        service = ProspectiveMemoryService(
            repository=repo,
            realtime_session_manager=SimpleNamespace(speak_prospective_interrupt=speak),
            settings=_settings(max_candidates=2),
        )
        request = _request()
        await service.process_matches(request=request, matches=[_candidate(request, confidence=.99)])
        repo.resolve_delivery_event.assert_not_awaited()
        await deliveries[0].on_outcome("failed", None)
        self.assertEqual(repo.resolve_delivery_event.await_args.kwargs["outcome"], "failed")
        self.assertEqual(repo.resolve_delivery_event.await_args.kwargs["metadata"]["reason"], "user_turn_busy")
        self.assertEqual(service._pending_deliveries, {})

    async def test_absent_session_has_truthful_reason_and_no_false_dispatch(self):
        repo = repository()
        service = ProspectiveMemoryService(
            repository=repo, realtime_session_manager=RealtimeSessionManager(),
            settings=_settings(max_candidates=2),
        )
        request = _request()
        await service.process_matches(request=request, matches=[_candidate(request, confidence=.99)])
        result = repo.resolve_delivery_event.await_args.kwargs
        self.assertEqual(result["outcome"], "missed")
        self.assertEqual(result["metadata"]["reason"], "no_connected_session")
        self.assertEqual(result["metadata"]["playback_status"], "unconfirmed")
        repo.mark_event_dispatched.assert_not_awaited()

    async def test_replacement_and_clear_revoke_only_matching_owner_focus_attempts(self):
        repo = repository()
        focus_id = uuid4()
        repo.set_visual_focus = AsyncMock(return_value=SimpleNamespace(id=focus_id))
        repo.clear_active_visual_focus = AsyncMock(return_value=SimpleNamespace(id=focus_id))
        settings = _settings(max_candidates=2)
        settings.PROSPECTIVE_FIND_ONCE_DEFAULT_TTL_SECONDS = 300
        settings.PROSPECTIVE_MONITOR_DEFAULT_TTL_SECONDS = 3600
        service = ProspectiveMemoryService(
            repository=repo, realtime_session_manager=RealtimeSessionManager(), settings=settings,
        )
        old, new, foreign, reminder = attempt(), attempt(), attempt(), attempt()
        new.subject_id = focus_id
        foreign.owner_id = "another-owner"
        reminder.subject_type = "conditional_reminder"
        service._pending_deliveries = {d.event_id: d for d in (old, new, foreign, reminder)}
        await service.set_visual_focus(owner_id="owner", target="keys", description="silver keys",
                                       purpose="find them", mode="find_once", expires_in_seconds=None)
        self.assertTrue(old.revoked)
        self.assertFalse(new.revoked)
        self.assertFalse(foreign.revoked)
        self.assertFalse(reminder.revoked)
        await service.clear_visual_focus(owner_id="owner")
        self.assertTrue(new.revoked)
        self.assertEqual(new.reason, "focus_cleared")


class TrackedPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_dual_candidates_use_only_tracked_route_even_when_delivery_is_rejected(self):
        for accepted in (True, False):
            with self.subTest(accepted=accepted):
                harness = _Harness()
                self.addAsyncCleanup(harness.pipeline.close)
                repo, calls = repository(), []

                async def speak(**kw):
                    calls.append(kw)
                    delivery = kw["delivery_attempt"]
                    delivery.provider_owned = True
                    if accepted:
                        self.assertTrue(await delivery.prepare())
                    await delivery.on_outcome("delivered" if accepted else "failed", "response")
                    return kw["preferred_jarvis_session_id"] if accepted else None

                harness.pipeline._prospective_service = ProspectiveMemoryService(
                    repository=repo,
                    realtime_session_manager=SimpleNamespace(speak_prospective_interrupt=speak),
                    settings=_settings(max_candidates=2),
                )
                original = harness.fast_analyzer.analyze

                async def analyze(request):
                    output = await original(request)
                    return output.model_copy(update={"prospective_matches": [
                        FastProspectiveMatch(
                            subject_type="visual_focus", subject_id=uuid4(), confidence=.99,
                            observed_condition="The tracked target is visible.",
                            supporting_frame_ids=[request.primary_frame.frame_id],
                            suggested_message="The target is here.",
                        )
                    ]})

                harness.fast_analyzer.analyze = analyze
                frame = harness.frame(sequence=1, captured_at=datetime.now(UTC))
                harness.put(frame, b"fixture")
                await harness.pipeline.enqueue(frame=frame, image_bytes=b"fixture", send_diagnostic=None)
                await asyncio.wait_for(harness.pipeline.drain(), 2)
                self.assertEqual(len(calls), 1)
                self.assertEqual(harness.interrupt_service.calls, [])
                self.assertEqual(len(harness.perception_repository.inserted), 1)
                self.assertEqual(repo.resolve_delivery_event.await_count, 1)


class GPTAttemptTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.h = gpt_fixture.GenericVisualSpeechTests()
        await self.h.asyncSetUp()
        self.addAsyncCleanup(self.h.asyncTearDown)

    async def speak(self, delivery):
        return await self.h.session.speak_prospective_interrupt(
            message="Your keys are on the desk.", event_id=delivery.event_id,
            subject_type=delivery.subject_type, subject_id=delivery.subject_id,
            on_outcome=delivery.on_outcome, delivery_attempt=delivery,
        )

    async def test_close_retries_observed_success_without_reporting_an_interruption(self):
        persist = AsyncMock(side_effect=[RuntimeError("storage temporarily unavailable"), None])
        delivery = attempt(persist=persist)
        self.assertTrue(await self.speak(delivery))
        await self.h.acknowledge()
        with self.assertLogs("app.realtime.session", level="WARNING"):
            await self.h.finish()
        self.assertFalse(delivery.settled)
        self.assertFalse(any(kind == "prospective.delivered" for kind, _ in self.h.app_events))
        await self.h.session.close("peer_closed")
        self.assertTrue(delivery.settled)
        self.assertEqual([call.args for call in persist.await_args_list],
                         [("delivered", "response-1"), ("delivered", "response-1")])
        self.assertFalse(any(kind == "prospective.interrupted" for kind, _ in self.h.app_events))
        delivered = [payload for kind, payload in self.h.app_events if kind == "prospective.delivered"]
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0]["response_id"], "response-1")

    async def test_error_before_native_writer_is_a_known_failed_delivery(self):
        self.h.session.openai.send_event = AsyncMock(side_effect=RuntimeError("not connected"))
        delivery = attempt()
        with self.assertRaises(RuntimeError):
            await self.speak(delivery)
        delivery.persist.assert_awaited_once_with("failed", None)
        self.assertIsNone(self.h.session._pending_response)

    async def test_ambiguous_write_can_receive_a_late_success_without_another_speech(self):
        self.h.provider.fail_create = True
        delivery = attempt()
        with self.assertRaises(RuntimeError):
            await self.speak(delivery)
        delivery.persist.assert_not_awaited()
        self.assertTrue(self.h.session._pending_response.submitted)
        self.h.provider.fail_create = False
        await self.h.acknowledge()
        await self.h.finish()
        delivery.persist.assert_awaited_once_with("delivered", "response-1")
        self.assertEqual(len(self.h.creates()), 1)

    async def test_revocation_clears_only_the_exact_buffered_prospective_response(self):
        delivery = attempt()
        self.assertTrue(await self.speak(delivery))
        self.assertEqual(self.h.creates()[0]["response"]["tool_choice"], "none")
        await self.h.acknowledge()
        await self.h.session._handle_openai_event({
            "type": "response.output_audio.delta", "delta": "AA==",
            "response_id": "response-1", "item_id": "speech-item",
        })
        delivery.revoke("focus_replaced")
        await delivery.notify_revoked()
        await delivery.notify_revoked()
        clears = [payload for kind, payload in self.h.app_events if kind == "audio.output.clear"]
        self.assertEqual(clears, [{
            "provider": "openai", "response_id": "response-1",
            "item_id": "speech-item", "reason": "prospective_revoked",
        }])
        await self.h.finish()
        delivery.persist.assert_awaited_once_with("interrupted", "response-1")

    async def test_cancelled_during_preflight_never_submits_speech(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def preflight():
            entered.set()
            await release.wait()
            return True

        delivery = attempt(preflight=preflight)
        task = asyncio.create_task(self.speak(delivery))
        await asyncio.wait_for(entered.wait(), 1)
        delivery.revoke("focus_replaced")
        release.set()
        self.assertFalse(await asyncio.wait_for(task, 1))
        self.assertFalse(self.h.creates())
        self.assertTrue(delivery.provider_owned)
        self.assertTrue(delivery.settled)
        self.assertEqual(delivery.reason, "focus_replaced")

    async def test_cancelled_after_submission_does_not_forward_late_audio(self):
        delivery = attempt()
        self.assertTrue(await self.speak(delivery))
        await self.h.acknowledge()
        delivery.revoke("focus_cleared")
        await self.h.finish()
        self.assertFalse(any(kind == "audio.output.delta" for kind, _ in self.h.app_events))
        delivery.persist.assert_awaited_once_with("interrupted", "response-1")

    async def test_revoked_audio_already_queued_at_phone_writer_is_not_forwarded(self):
        delivery = attempt()
        self.assertTrue(await self.speak(delivery))
        await self.h.acknowledge()
        entered = asyncio.Event()
        session = self.h.session
        real_send = gpt_fixture.RealtimeAudioSession._send_app_event.__get__(session)

        async def send(kind, **payload):
            entered.set()
            return await real_send(kind, **payload)

        session._send_app_event = send
        async with session._app_sender._lock:
            task = asyncio.create_task(session._handle_openai_event({
                "type": "response.output_audio.delta", "delta": "AA==",
                "response_id": "response-1", "item_id": "speech",
            }))
            await asyncio.wait_for(entered.wait(), 1)
            delivery.revoke("focus_cleared")
        await asyncio.wait_for(task, 1)
        session._app_sender._websocket.send_text.assert_not_awaited()
        self.assertFalse(session._active_responses["response-1"].has_forwarded_answer)
        await self.h.finish()
        delivery.persist.assert_awaited_once_with("interrupted", "response-1")

    async def test_expiry_after_submission_suppresses_late_audio(self):
        delivery = attempt(expires_at=datetime.now(UTC) + timedelta(hours=1))
        self.assertTrue(await self.speak(delivery))
        await self.h.acknowledge()
        delivery.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await self.h.finish()
        self.assertFalse(any(kind == "audio.output.delta" for kind, _ in self.h.app_events))
        delivery.persist.assert_awaited_once_with("interrupted", "response-1")
        self.assertEqual(delivery.reason, "focus_expired")

    async def test_success_requires_forwarded_answer_and_completion(self):
        delivery = attempt()
        self.assertTrue(await self.speak(delivery))
        delivery.persist.assert_not_awaited()
        await self.h.acknowledge()
        await self.h.finish()
        delivery.persist.assert_awaited_once_with("delivered", "response-1")


class GeminiAttemptTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.h = gemini_fixture.GeminiProactiveTests()
        await self.h.asyncSetUp()
        self.addAsyncCleanup(self.h.asyncTearDown)

    async def speak(self, delivery):
        return await self.h.session.speak_prospective_interrupt(
            message="Your keys are on the desk.", event_id=delivery.event_id,
            subject_type=delivery.subject_type, subject_id=delivery.subject_id,
            on_outcome=delivery.on_outcome, delivery_attempt=delivery,
        )

    async def test_revocation_clears_only_the_exact_buffered_prospective_response(self):
        delivery = attempt()
        self.assertTrue(await self.speak(delivery))
        await self.h.answer(terminal=False)
        await gemini_fixture.until(lambda: bool(self.h.events("audio.output.delta")))
        audio = self.h.events("audio.output.delta")[0]
        delivery.revoke("focus_replaced")
        await delivery.notify_revoked()
        await delivery.notify_revoked()
        clears = self.h.events("audio.output.clear")
        self.assertEqual(len(clears), 1)
        self.assertEqual(clears[0]["provider"], "gemini")
        self.assertEqual(clears[0]["response_id"], audio["response_id"])
        self.assertEqual(clears[0]["item_id"], audio["item_id"])

    async def test_busy_rejection_has_one_owned_outcome_and_reason(self):
        await self.h.append()
        delivery = attempt()
        self.assertFalse(await self.speak(delivery))
        await gemini_fixture.until(lambda: delivery.settled)
        self.assertTrue(delivery.provider_owned)
        self.assertEqual(delivery.reason, "user_turn_busy")
        delivery.persist.assert_awaited_once_with("failed", None)
        self.assertFalse(self.h.native_deliveries())

    async def test_cancelled_during_preflight_never_submits_speech(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def preflight():
            entered.set()
            await release.wait()
            return True

        delivery = attempt(preflight=preflight)
        task = asyncio.create_task(self.speak(delivery))
        await asyncio.wait_for(entered.wait(), 1)
        delivery.revoke("focus_replaced")
        release.set()
        self.assertFalse(await asyncio.wait_for(task, 1))
        await gemini_fixture.until(lambda: delivery.settled)
        self.assertFalse(self.h.native_deliveries())
        self.assertEqual(delivery.reason, "focus_replaced")

    async def test_cancelled_after_submission_does_not_forward_late_audio(self):
        delivery = attempt()
        self.assertTrue(await self.speak(delivery))
        delivery.revoke("focus_cleared")
        await self.h.answer()
        await gemini_fixture.until(lambda: delivery.settled)
        self.assertFalse(self.h.events("audio.output.delta"))
        self.assertEqual(delivery.observed_outcome[0], "interrupted")

    async def test_revoked_audio_already_queued_at_phone_writer_is_not_forwarded(self):
        delivery = attempt()
        self.assertTrue(await self.speak(delivery))
        turn = self.h.session._pending_responses[0]
        async with self.h.session._sender._lock:
            await self.h.answer()
            await gemini_fixture.until(lambda: turn.provider_started)
            delivery.revoke("focus_cleared")
        await gemini_fixture.until(lambda: delivery.settled)
        self.assertFalse(self.h.events("audio.output.delta"))
        self.assertEqual(turn.output.forwarded_audio_bytes, 0)
        self.assertEqual(delivery.observed_outcome[0], "interrupted")

    async def test_expiry_after_submission_suppresses_late_audio(self):
        delivery = attempt(expires_at=datetime.now(UTC) + timedelta(hours=1))
        self.assertTrue(await self.speak(delivery))
        delivery.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await self.h.answer()
        await gemini_fixture.until(lambda: delivery.settled)
        self.assertFalse(self.h.events("audio.output.delta"))
        self.assertEqual(delivery.reason, "focus_expired")
        self.assertEqual(delivery.observed_outcome[0], "interrupted")

    async def test_success_requires_forwarded_answer_and_completion(self):
        delivery = attempt()
        self.assertTrue(await self.speak(delivery))
        delivery.persist.assert_not_awaited()
        await self.h.answer()
        await gemini_fixture.until(lambda: delivery.settled)
        self.assertEqual(delivery.observed_outcome[0], "delivered")


class ProspectiveOwnerRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_foreign_preferred_session_never_receives_search(self):
        manager = RealtimeSessionManager()
        foreign_id = uuid4()
        speak = AsyncMock(return_value=True)
        await manager.register(audio_session_id="foreign", owner_id="foreign",
                               jarvis_session_id=foreign_id,
                               session=SimpleNamespace(speak_prospective_interrupt=speak))
        delivery = attempt()
        result = await manager.speak_prospective_interrupt(
            owner_id="owner", preferred_jarvis_session_id=foreign_id,
            message="Private search", event_id=delivery.event_id,
            subject_type=delivery.subject_type, subject_id=delivery.subject_id,
            on_outcome=delivery.on_outcome, delivery_attempt=delivery,
        )
        self.assertIsNone(result)
        speak.assert_not_awaited()
