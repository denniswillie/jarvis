from __future__ import annotations

import asyncio
import json
import time
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import test_generic_visual_speech as speech_fixtures
import test_realtime_output_phases as phase_fixtures
import test_realtime_shared_view as shared_fixtures
from test_current_image_service import _jpeg
from test_historical_image_service import _evidence, _service
from visual_frame_fixtures import _frame

from app.images import ImageReference, ImageUnavailableError, ResolvedImage
from app.errors import JarvisHTTPError
from app.memory.tools import create_get_memory_image_tool
from app.perception.current_image import CurrentImageService
from app.perception.tools import CurrentImageToolModule
from app.perception.frame_window import RecentVisualFrameWindow
from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.visual_delivery import RealtimeVisualCoordinator
from app.tools.executor import JarvisToolExecutor
from app.tools.module import FunctionToolModule, ToolBuildContext
from app.tools.registry import ToolDefinition, create_tool_registry


class _Clock:
    def __init__(self):
        self.origin = time.monotonic()
        self.offset = 0.0

    def monotonic(self):
        return time.monotonic() + self.offset

    def now(self):
        return datetime(2026, 9, 9, 12, tzinfo=UTC) + timedelta(
            seconds=self.monotonic() - self.origin,
        )


class _Executions:
    def __init__(self):
        self.records = {}
        self.finalized = []

    async def start_execution_with_status(self, **kwargs):
        call_id = kwargs["provider_call_id"]
        created = call_id not in self.records
        if created:
            self.records[call_id] = SimpleNamespace(status="started", result=None, error=None)
        return self.records[call_id], created

    async def finalize_execution(self, **kwargs):
        self.finalized.append(kwargs)
        record = self.records[kwargs["provider_call_id"]]
        record.status = kwargs["status"]
        record.result = kwargs.get("result")
        record.error = kwargs.get("error")
        return record


class _Wire(shared_fixtures._Wire):
    def __init__(self, session):
        super().__init__(session)
        self.output_image_statuses = []
        self.fail_output = False

    async def send(self, raw):
        event = json.loads(raw)
        output = event.get("item", {}).get("type") == "function_call_output"
        if output:
            self.output_image_statuses.append(tuple(
                item.status.value for item in self.session._visual_context.items
            ))
        await super().send(raw)
        if output and self.fail_output:
            raise RuntimeError("unknown function output write outcome")


class RealtimeImageToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = speech_fixtures.GenericVisualSpeechTests()
        await self.fixture.asyncSetUp()
        self.session = self.fixture.session
        self.clock = _Clock()
        self.session._now = self.clock.now
        self.session._monotonic = self.clock.monotonic
        self.state = self.session._visual_context
        self.state.policy = replace(self.state.policy, enabled=False, sync_timeout_seconds=0.15)
        self.window = RecentVisualFrameWindow(
            max_frames_per_session=8, ttl_seconds=30,
            now=self.clock.now, monotonic=self.clock.monotonic,
        )
        self.lease = self.window.begin_stream("owner", self.session.context.jarvis_session_id, uuid4())
        self.provider = OpenAIRealtimeConnection(settings=self.session.settings, owner_id="owner")
        self.wire = _Wire(self.session)
        self.provider._websocket = self.wire
        self.session.openai = self.provider
        self.delivery = RealtimeVisualCoordinator(
            state=self.state, frame_window=self.window, provider=self.provider,
            max_image_bytes=512_000, now=self.clock.now, monotonic=self.clock.monotonic,
            image_resolution_timeout_seconds=0.15,
        )
        self.session._visual_delivery = self.delivery
        self.images = [self.historical(), self.historical()]
        self.handler = AsyncMock(side_effect=self.select)
        self.resolver = AsyncMock(side_effect=self.resolve)
        self.ordinary = AsyncMock(return_value={"status": "success", "value": "ordinary result"})
        module = FunctionToolModule("fake-image-module", lambda context: (
            ToolDefinition(
                name="fake_image_lookup", description="Select exact test evidence.",
                input_schema={"type": "object", "properties": {"which": {"type": "integer"}}},
                handler=self.handler, image_resolver=self.resolver,
            ),
            ToolDefinition(
                name="ordinary_example", description="Ordinary test tool.",
                input_schema={"type": "object", "properties": {}}, handler=self.ordinary,
            ),
        ))
        self.session.tool_registry = create_tool_registry(
            modules=(module,), context=ToolBuildContext(
                settings=self.session.settings, owner_id="owner", session_id=self.session.context.jarvis_session_id,
            ),
        )
        self.executions = _Executions()
        self.session.tool_executor = JarvisToolExecutor(
            self.session.tool_registry, execution_repository=self.executions,
            owner_id="owner", session_id=self.session.context.jarvis_session_id,
        )
        self.receiver = asyncio.create_task(self.receive())

    async def asyncTearDown(self):
        await self.session.close("test_complete")
        await self.delivery.close()
        self.receiver.cancel()
        await asyncio.gather(self.receiver, return_exceptions=True)

    def historical(self):
        return ResolvedImage(
            reference=ImageReference(
                frame_id=uuid4(), owner_id="owner", session_id=self.session.context.jarvis_session_id,
                source="historical", captured_at=self.clock.now() - timedelta(hours=1),
                content_type="image/jpeg", width=48, height=32,
                raw_image_expires_at=self.clock.now() + timedelta(seconds=60),
            ),
            image_bytes=_jpeg(),
        )

    async def select(self, arguments):
        return {
            "status": "success", "image_status": "selected",
            "image": self.images[arguments.get("which", 0)].reference.to_metadata(),
        }

    async def resolve(self, selection):
        self.assertFalse(self.session._response_admission_lock.locked())
        self.assertIsNot(asyncio.current_task(), self.receiver)
        return next(image for image in self.images if str(image.reference.frame_id) == selection["image"]["frame_id"])

    async def receive(self):
        self.session._provider_receive_task = asyncio.current_task()
        while True:
            event = await self.wire.incoming.get()
            try:
                await self.session._handle_openai_event(event)
            finally:
                self.wire.incoming.task_done()

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                if self.receiver.done():
                    await self.receiver
                await asyncio.sleep(0.001)

    async def start(self, *, kind="normal_user_turn", metadata=None, task=None):
        if kind == "normal_user_turn":
            await self.fixture.append()
            await self.session._handle_client_event({"type": "audio.input.commit"})
            await self.session._handle_client_event({"type": "response.create"})
        else:
            async with self.session._response_admission_lock:
                reservation = self.session._reserve_response_locked(kind, metadata or {})
            await self.session._send_reserved_response(reservation, instructions=task)
        await self.until(lambda: "response-1" in self.session._active_responses)

    def call(self, call_id="image-1", *, name="fake_image_lookup", which=0, response_id="response-1", arguments=None):
        item = {
            "type": "function_call", "id": call_id, "call_id": call_id,
            "name": name, "arguments": json.dumps({"which": which} if arguments is None else arguments),
        }
        self.wire.incoming.put_nowait({
            "type": "response.output_item.done", "response_id": response_id, "item": item,
        })
        return item

    def done(self, response_id="response-1", *, status="completed", output=None):
        response = {"id": response_id, "status": status}
        if output is not None:
            response["output"] = output
        self.wire.incoming.put_nowait({"type": "response.done", "response": response})

    def outputs(self):
        return [
            (event["item"]["call_id"], json.loads(event["item"]["output"]))
            for event in self.wire.events if event.get("item", {}).get("type") == "function_call_output"
        ]

    def image_events(self):
        return [
            event for event in self.wire.events
            if any(part.get("type") == "input_image" for part in event.get("item", {}).get("content", []))
        ]

    async def continuation(self):
        await self.until(lambda: "response-2" in self.session._active_responses)
        self.assertEqual(len(self.wire.responses), 2)
        return self.wire.responses[1]

    async def blocked_resolver(self, *, ignore_cancel=False):
        entered, release = asyncio.Event(), asyncio.Event()

        async def resolve(selection):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                if not ignore_cancel:
                    raise
                await release.wait()
            return await self.resolve(selection)

        self.resolver.side_effect = resolve
        self.call()
        await asyncio.wait_for(entered.wait(), 1)
        return release

    async def test_ack_precedes_output_and_one_continuation_when_parent_finishes_first(self):
        await self.start()
        self.wire.auto_image_ack = False
        self.call()
        await self.until(lambda: bool(self.image_events()))
        self.done()
        await self.until(lambda: self.session._active_responses["response-1"].completed_response is not None)
        self.assertEqual(self.outputs(), [])
        self.assertTrue(self.session._normal_turn_pending)
        self.assertEqual(len(self.wire.responses), 1)
        self.wire.ack_image(self.image_events()[0])
        recorded = await self.continuation()
        result = self.outputs()[0][1]
        self.assertEqual(result["image_status"], "attached")
        self.assertEqual(result["image"]["captured_at"], self.images[0].reference.captured_at.isoformat())
        self.assertEqual(self.wire.output_image_statuses[0], ("acknowledged",))
        self.assertEqual(recorded["cutoff"].requested_images[0].provider_call_id, "image-1")
        self.assertIn(result["provider_item_id"], recorded["images"])
        self.assertIsNone(recorded["cutoff"].current_frame)
        self.assertNotIn("input", recorded["event"]["response"])
        self.assertNotIn("conversation", recorded["event"]["response"])
        output_index = next(i for i, event in enumerate(self.wire.events) if event.get("item", {}).get("call_id") == "image-1")
        self.assertLess(self.wire.events.index(self.image_events()[0]), output_index)
        self.assertLess(output_index, self.wire.events.index(recorded["event"]))
        self.assertFalse(self.session._image_tool_requests)

    async def test_completion_before_parent_and_duplicate_events_do_not_repeat_work(self):
        await self.start()
        item = self.call()
        await self.until(lambda: len(self.outputs()) == 1)
        self.wire.incoming.put_nowait({**item, "type": "response.function_call_arguments.done", "response_id": "response-1"})
        self.call()
        await self.wire.incoming.join()
        self.assertEqual(len(self.wire.responses), 1)
        self.done()
        self.done()
        await self.continuation()
        self.handler.assert_awaited_once()
        self.resolver.assert_awaited_once()
        self.assertEqual(len(self.image_events()), 1)
        self.assertEqual(len(self.outputs()), 1)

    async def test_mixed_calls_wait_for_every_output_and_transfer_both_images(self):
        await self.start()
        entered, release = asyncio.Event(), asyncio.Event()

        async def resolve(selection):
            if selection["image"]["frame_id"] == str(self.images[0].reference.frame_id):
                entered.set()
                await release.wait()
            return await self.resolve(selection)

        self.resolver.side_effect = resolve
        self.call("slow", which=0)
        await entered.wait()
        self.call("fast", which=1)
        self.call("ordinary", name="ordinary_example")
        self.done()
        await self.until(lambda: len(self.outputs()) == 2)
        self.assertEqual(len(self.wire.responses), 1)
        release.set()
        recorded = await self.continuation()
        self.assertEqual({call for call, _ in self.outputs()}, {"slow", "fast", "ordinary"})
        self.assertEqual({pin.provider_call_id for pin in recorded["cutoff"].requested_images}, {"slow", "fast"})
        self.assertEqual(len(self.outputs()), 3)

    async def test_final_envelope_admits_a_missing_function_event(self):
        await self.start()
        self.done(output=[{
            "type": "function_call", "call_id": "final-only", "name": "fake_image_lookup", "arguments": "{}",
        }])
        await self.continuation()
        self.assertEqual(self.outputs()[0][0], "final-only")
        self.handler.assert_awaited_once()

    async def test_ordinary_tool_stays_on_receiver_and_does_not_use_image_adapter(self):
        await self.start()

        async def ordinary(arguments):
            self.assertIs(asyncio.current_task(), self.receiver)
            self.assertFalse(self.session._image_tool_tasks)
            return {"status": "success", "value": "unchanged"}

        self.ordinary.side_effect = ordinary
        self.call(name="ordinary_example")
        self.done()
        await self.continuation()
        self.resolver.assert_not_awaited()
        self.assertFalse(self.image_events())
        self.assertEqual(self.outputs()[0][1], {"status": "success", "value": "unchanged"})

    async def test_failed_selection_skips_resolver_and_continues_with_domain_error(self):
        await self.start()
        self.handler.side_effect = None
        self.handler.return_value = {"status": "failed", "error": {"code": "no_frame", "message": "No frame."}}
        self.call()
        self.done()
        await self.continuation()
        self.resolver.assert_not_awaited()
        self.assertEqual(self.outputs()[0][1]["error"]["code"], "no_frame")
        self.assertEqual(self.outputs()[0][1]["image_status"], "unavailable")

    async def test_resolver_error_is_safe_and_does_not_close_audio(self):
        await self.start()
        self.resolver.side_effect = RuntimeError("SECRET_MEDIA_URL_OR_STORAGE_PATH")
        with self.assertLogs("app.realtime.tool_images", level="INFO") as logs:
            self.call()
            self.done()
            await self.continuation()
        self.assertEqual(self.outputs()[0][1]["error"]["code"], "image_resolution_failed")
        self.assertNotIn("SECRET_MEDIA", json.dumps(self.outputs()) + " ".join(logs.output))
        self.assertFalse(self.session._closed.is_set())

    async def test_resolver_timeout_returns_terminal_failure(self):
        await self.start()
        await self.blocked_resolver()
        self.done()
        await self.continuation()
        self.assertEqual(self.outputs()[0][1]["error"]["code"], "image_resolution_timeout")
        self.assertFalse(self.session._image_tool_tasks)

    async def test_ack_timeout_does_not_report_selected_as_attached(self):
        await self.start()
        self.wire.auto_image_ack = False
        self.call()
        self.done()
        await self.continuation()
        self.assertEqual(self.outputs()[0][1]["error"]["code"], "image_delivery_timeout")
        self.assertEqual(self.executions.records["image-1"].result["image_status"], "selected")
        self.assertFalse(self.state.requested_pins)
        self.assertFalse(self.session._closed.is_set())

    async def test_provider_rejection_has_one_failure_output_and_no_requested_pin(self):
        await self.start()
        self.wire.auto_image_ack = False
        self.call()
        await self.until(lambda: bool(self.image_events()))
        self.wire.incoming.put_nowait({
            "type": "error", "error": {"event_id": self.image_events()[0]["event_id"], "code": "invalid_request_error"},
        })
        self.done()
        await self.continuation()
        self.assertEqual(self.outputs()[0][1]["error"]["code"], "provider_rejected_image_input")
        self.assertFalse(self.state.requested_pins)
        self.assertEqual(len(self.outputs()), 1)

    async def test_durable_replay_resolves_exact_selection_without_rerunning_handler(self):
        await self.start()
        selection = await self.select({"which": 1})
        self.executions.records["replay"] = SimpleNamespace(status="completed", result=selection, error=None)
        self.call("replay", which=0)
        self.done()
        await self.continuation()
        self.handler.assert_not_awaited()
        self.assertEqual(self.resolver.await_args.args[0], selection)
        self.assertEqual(self.outputs()[0][1]["image"]["frame_id"], str(self.images[1].reference.frame_id))
        self.assertEqual(selection["image_status"], "selected")
        self.assertNotIn("provider_item_id", selection)

    async def test_expired_replay_is_revalidated_on_this_connection(self):
        await self.start()
        selection = await self.select({})
        self.executions.records["replay"] = SimpleNamespace(status="completed", result=selection, error=None)
        self.clock.offset += 61
        self.call("replay")
        self.done()
        await self.continuation()
        self.handler.assert_not_awaited()
        self.resolver.assert_awaited_once()
        self.assertEqual(self.outputs()[0][1]["error"]["code"], "frame_unavailable")
        self.assertFalse(self.image_events())

    async def test_budget_is_shared_across_image_and_ordinary_continuations(self):
        await self.start()
        self.session._tool_budgets["response-1"].max_total_calls = 1
        self.call()
        self.done()
        await self.continuation()
        self.call("over-budget", name="ordinary_example", response_id="response-2")
        self.done("response-2")
        await self.until(lambda: len(self.wire.responses) == 3)
        self.ordinary.assert_not_awaited()
        self.assertEqual(self.outputs()[-1][1]["error"]["code"], "tool_call_limit_reached")
        self.resolver.assert_awaited_once()

    async def test_audio_and_receiver_progress_during_resolution_then_cancel_old_work(self):
        await self.start()
        release = await self.blocked_resolver(ignore_cancel=True)
        self.wire.incoming.put_nowait({"type": "input_audio_transcription.completed", "transcript": "A delayed transcript."})
        await self.until(lambda: any(kind == "user.transcript.done" for kind, _ in self.fixture.app_events))
        await asyncio.wait_for(self.fixture.append(), 1)
        self.assertTrue(any(event["type"] == "input_audio_buffer.append" for event in self.wire.events[1:]))
        release.set()
        self.done()
        await self.until(lambda: not self.session._image_tool_tasks)
        await self.wire.incoming.join()
        self.assertFalse(self.image_events())
        self.assertEqual(self.outputs()[0][1]["image_status"], "unavailable")
        self.assertEqual(len(self.wire.responses), 1)
        self.assertTrue(self.session._normal_turn_pending)
        self.assertTrue(self.session._user_audio_pending)

    async def test_barge_in_during_ack_wait_releases_pin_and_late_ack_is_cleaned(self):
        await self.start()
        self.wire.auto_image_ack = False
        self.call()
        await self.until(lambda: bool(self.image_events()))
        await self.fixture.append()
        self.done(status="cancelled")
        self.wire.ack_image(self.image_events()[0])
        await self.until(lambda: not self.session._image_tool_tasks and not self.state.items)
        self.assertFalse(self.state.requested_pins)
        self.assertEqual(self.outputs()[0][1]["image_status"], "unavailable")
        self.assertEqual(len(self.wire.responses), 1)

    async def test_barge_in_while_image_write_is_queued_prevents_pixels(self):
        await self.start()
        await self.provider._send_lock.acquire()
        cancel = None
        try:
            self.call()
            await self.until(lambda: self.provider._foreground_waiters > 0)
            cancel = asyncio.create_task(self.session._handle_client_event({"type": "response.cancel"}))
            await self.until(lambda: self.session._image_tool_requests["image-1"].cancelled)
        finally:
            self.provider._send_lock.release()
        await cancel
        self.done(status="cancelled")
        await self.until(lambda: not self.session._image_tool_tasks)
        self.assertFalse(self.image_events())
        self.assertEqual(self.outputs()[0][1]["image_status"], "unavailable")

    async def test_barge_in_while_function_output_is_queued_replaces_success(self):
        await self.start()
        self.wire.auto_image_ack = False
        self.call()
        await self.until(lambda: bool(self.image_events()))
        await self.provider._send_lock.acquire()
        cancel = None
        try:
            self.wire.ack_image(self.image_events()[0])
            await self.until(lambda: self.provider._foreground_waiters > 0)
            cancel = asyncio.create_task(self.session._handle_client_event({"type": "response.cancel"}))
            await self.until(lambda: self.session._image_tool_requests["image-1"].cancelled)
        finally:
            self.provider._send_lock.release()
        await cancel
        self.done(status="cancelled")
        await self.until(lambda: not self.session._image_tool_tasks)
        self.assertEqual(len(self.outputs()), 1)
        self.assertEqual(self.outputs()[0][1]["image_status"], "unavailable")

    async def test_close_during_resolver_ignoring_cancellation_has_no_late_writes(self):
        await self.start()
        release = await self.blocked_resolver(ignore_cancel=True)
        close = asyncio.create_task(self.session.close("client_closed"))
        await self.until(self.session._closed.is_set)
        count = len(self.wire.events)
        release.set()
        await asyncio.wait_for(close, 1)
        self.assertEqual(len(self.wire.events), count)
        self.assertFalse(self.session._image_tool_tasks)
        self.assertFalse(self.session._image_tool_requests)
        self.assertFalse(self.state.items)

    async def test_close_during_ack_wait_releases_tasks_and_pixels(self):
        await self.start()
        self.wire.auto_image_ack = False
        self.call()
        await self.until(lambda: bool(self.image_events()))
        count = len(self.wire.events)
        await asyncio.wait_for(self.session.close("client_closed"), 1)
        self.wire.ack_image(self.image_events()[0])
        await self.wire.incoming.join()
        self.assertEqual(len(self.wire.events), count)
        self.assertFalse(self.outputs())
        self.assertFalse(self.session._image_tool_tasks)
        self.assertFalse(self.state.items)

    async def test_close_before_background_coroutine_starts_does_not_execute_tool(self):
        await self.start()
        await self.session._handle_function_call_item({
            "call_id": "not-started", "name": "fake_image_lookup", "arguments": "{}",
        }, response_id="response-1")
        await self.session.close("client_closed")
        self.handler.assert_not_awaited()
        self.resolver.assert_not_awaited()
        self.assertFalse(self.outputs())
        self.assertFalse(self.session._image_tool_tasks)

    async def test_failed_parent_cancels_pending_work_without_continuation(self):
        await self.start()
        await self.blocked_resolver()
        self.done(status="failed")
        await self.until(lambda: "response-1" in self.session._retired_response_ids)
        self.assertEqual(len(self.wire.responses), 1)
        self.assertFalse(self.image_events())
        self.assertFalse(self.session._normal_turn_pending)

    async def test_terminal_visual_dismissal_cannot_be_revived_by_image_completion(self):
        await self.start(kind="visual_interrupt", metadata={"visual_interrupt_id": str(uuid4())}, task="Check the candidate.")
        await self.blocked_resolver()
        self.session._visual_interrupt_control_executor.execute = AsyncMock(return_value={"status": "success"})
        self.call("dismiss", name="dismiss_visual_interrupt")
        self.done()
        await self.until(lambda: "response-1" in self.session._retired_response_ids)
        self.assertEqual(len(self.wire.responses), 1)
        self.assertFalse(self.image_events())
        self.assertEqual(len(self.outputs()), 2)
        self.assertIsNone(self.session._tool_followup)

    async def delivery_identity(self, kind, *, fail=False):
        event_id = str(uuid4())
        if kind == "reminder":
            metadata = {"reminder_id": str(uuid4()), "reminder_event_id": event_id}
            self.session.reminder_repository = SimpleNamespace(
                mark_event_delivery=AsyncMock(), set_event_provider_response_id=AsyncMock(),
            )
            self.session._pending_reminder_events[event_id] = metadata["reminder_id"]
        elif kind == "prospective":
            metadata = {"prospective_event_id": event_id, "prospective_subject_type": "commitment", "prospective_subject_id": str(uuid4())}
            self.outcome = AsyncMock()
            self.session._pending_prospective_deliveries[event_id] = ("commitment", metadata["prospective_subject_id"], self.outcome)
        else:
            metadata = {"visual_interrupt_id": event_id}
        await self.start(kind=kind, metadata=metadata, task="ORIGINAL_DELIVERY_TASK")
        await self.session._send_complete_session_update(SimpleNamespace(
            as_prompt_payload=lambda: {"working_memory": {"marker": "REFRESHED_CONTEXT"}},
        ))
        if fail:
            self.resolver.side_effect = ImageUnavailableError("frame_unavailable", "The selected frame expired.")
        self.call()
        self.done()
        recorded = await self.continuation()
        continuation = self.session._active_responses["response-2"]
        self.assertEqual(continuation.task_instructions, "ORIGINAL_DELIVERY_TASK")
        self.assertEqual(continuation.delivery_kind, kind)
        self.assertIn("REFRESHED_CONTEXT", recorded["event"]["response"]["instructions"])
        for key, value in metadata.items():
            self.assertEqual(continuation.metadata[key], value)
        if kind == "reminder":
            self.session.reminder_repository.mark_event_delivery.assert_not_awaited()
        elif kind == "prospective":
            self.outcome.assert_not_awaited()
        self.assertEqual(continuation.metadata["jarvis_tool_chain_id"], "response-1")

    async def test_reminder_image_continuation_preserves_delivery_task(self):
        await self.delivery_identity("reminder")

    async def test_prospective_image_failure_preserves_delivery_task(self):
        await self.delivery_identity("prospective", fail=True)

    async def test_visual_image_continuation_preserves_delivery_task(self):
        await self.delivery_identity("visual_interrupt")

    async def test_expiry_after_attachment_is_labelled_unavailable_in_answer_context(self):
        await self.start()
        self.call()
        await self.until(lambda: len(self.outputs()) == 1 and not self.session._image_tool_tasks)
        self.clock.offset += 61
        self.delivery.notify_frame_available()
        await self.until(lambda: not self.state.requested_pins)
        self.done()
        recorded = await self.continuation()
        self.assertIn("unavailable_requested_images", recorded["event"]["response"]["instructions"])
        self.assertIn(str(self.images[0].reference.frame_id), recorded["event"]["response"]["instructions"])
        self.assertFalse(recorded["cutoff"].requested_images)
        self.assertIsNone(recorded["cutoff"].current_frame)

    async def test_expiry_at_response_writer_falls_back_without_duplicate_response(self):
        await self.start()
        self.call()
        await self.until(lambda: len(self.outputs()) == 1 and not self.session._image_tool_tasks)
        await self.provider._send_lock.acquire()
        try:
            self.done()
            await self.until(lambda: self.provider._foreground_waiters > 0)
            self.clock.offset += 61
        finally:
            self.provider._send_lock.release()
        recorded = await self.continuation()
        self.assertIn("unavailable_requested_images", recorded["event"]["response"]["instructions"])
        self.assertFalse(recorded["cutoff"].requested_images)
        self.assertEqual(len(self.outputs()), 1)

    async def test_barge_in_between_parent_settlement_and_admission_releases_image(self):
        await self.start()
        self.call()
        await self.until(lambda: len(self.outputs()) == 1 and not self.session._image_tool_tasks)

        async def app_event(kind, **payload):
            await self.fixture._record_event(kind, **payload)
            if kind == "response.done" and payload.get("response_id") == "response-1":
                await self.fixture.append()

        self.session._send_app_event = AsyncMock(side_effect=app_event)
        self.done()
        await self.wire.incoming.join()
        self.assertEqual(len(self.wire.responses), 1)
        self.assertFalse(self.state.requested_pins)
        self.assertFalse(self.session._image_tool_requests)
        self.assertTrue(self.session._user_audio_pending)

    async def test_absent_image_coordinator_returns_failure_through_normal_continuation(self):
        await self.start()
        self.session._visual_delivery = None
        self.call()
        self.done()
        await self.continuation()
        self.assertEqual(self.outputs()[0][1]["error"]["code"], "image_delivery_unavailable")
        self.resolver.assert_not_awaited()

    async def test_unknown_function_output_write_is_not_retried_or_continued(self):
        await self.start()
        self.wire.fail_output = True
        self.call()
        self.done()
        await self.until(self.session._closed.is_set)
        self.assertEqual(len(self.outputs()), 1)
        self.assertEqual(len(self.wire.responses), 1)

    async def test_requested_resolution_can_outlast_passive_sync_budget(self):
        self.delivery.image_resolution_timeout_seconds = 0.6

        async def delayed(selection):
            await asyncio.sleep(0.2)
            return await self.resolve(selection)

        self.resolver.side_effect = delayed
        await self.start()
        self.call()
        self.done()
        await self.continuation()
        self.assertEqual(self.state.policy.sync_timeout_seconds, 0.15)
        self.assertEqual(self.outputs()[0][1]["image_status"], "attached")

    async def test_camera_stream_metadata_is_separate_from_attached_pixels(self):
        metadata = self.session._visual_prompt_metadata()
        self.assertTrue(metadata["camera_stream_active"])
        self.assertIsNone(metadata["latest_acknowledged"])
        self.window.end_stream(self.lease)
        self.assertFalse(self.session._visual_prompt_metadata()["camera_stream_active"])

    async def test_failed_provider_image_response_gets_one_notice_and_next_turn_works(self):
        await self.start()
        self.call()
        self.done()
        await self.continuation()
        self.done("response-2", status="failed")
        await self.until(lambda: "response-3" in self.session._active_responses)
        notice = self.wire.responses[2]["event"]["response"]
        self.assertEqual(notice["tool_choice"], "none")
        self.assertEqual(notice["tools"], [])
        self.assertEqual(notice["input"], [])
        self.assertEqual(notice["conversation"], "none")
        self.assertIn("Do not describe or infer its contents", notice["instructions"])
        self.assertTrue(any(kind == "error" and payload["code"] == "response_failed"
                            and payload["recoverable"] for kind, payload in self.fixture.app_events))
        self.done("response-2", status="failed")
        await self.wire.incoming.join()
        self.assertEqual(len(self.wire.responses), 3)
        await phase_fixtures.output_item(self.session, "response-3", "failure-notice", "final_answer")
        await phase_fixtures.speech(self.session, "response-3", "failure-notice", "I could not read that image.")
        await phase_fixtures.finish(self.session, "response-3")
        await self.fixture.append()
        await self.session._handle_client_event({"type": "audio.input.commit"})
        await self.session._handle_client_event({"type": "response.create"})
        await self.until(lambda: "response-4" in self.session._active_responses)
        await phase_fixtures.output_item(self.session, "response-4", "ordinary-answer", "final_answer")
        await phase_fixtures.speech(self.session, "response-4", "ordinary-answer", "Four.")
        await phase_fixtures.finish(self.session, "response-4")
        self.assertFalse(self.session._normal_turn_pending)
        self.assertFalse(self.session._closed.is_set())

    async def test_failed_notice_does_not_start_a_retry_loop(self):
        await self.start()
        self.call()
        self.done()
        await self.continuation()
        self.done("response-2", status="failed")
        await self.until(lambda: "response-3" in self.session._active_responses)
        self.done("response-3", status="failed")
        await self.until(lambda: not self.session._normal_turn_pending)
        self.assertEqual(len(self.wire.responses), 3)
        self.assertFalse(self.session._closed.is_set())

    async def test_interrupted_image_failure_does_not_speak_a_notice(self):
        await self.start()
        self.call()
        self.done()
        await self.continuation()
        self.session._active_responses["response-2"].interrupted = True
        self.done("response-2", status="failed")
        await self.until(lambda: not self.session._normal_turn_pending)
        self.assertEqual(len(self.wire.responses), 2)
        self.assertFalse(any(kind == "error" and payload["code"] == "response_failed"
                             for kind, payload in self.fixture.app_events))

    async def test_failed_image_turn_is_followed_by_a_usable_normal_spoken_turn(self):
        await self.start()
        self.resolver.side_effect = ImageUnavailableError("frame_unavailable", "Unavailable.")
        self.call()
        self.done()
        await self.continuation()
        await phase_fixtures.output_item(self.session, "response-2", "limitation", "final_answer")
        await phase_fixtures.speech(self.session, "response-2", "limitation", "That image is unavailable.")
        await phase_fixtures.finish(self.session, "response-2")
        await self.fixture.append()
        await self.session._handle_client_event({"type": "audio.input.commit"})
        await self.session._handle_client_event({"type": "response.create"})
        await self.until(lambda: "response-3" in self.session._active_responses)
        await phase_fixtures.output_item(self.session, "response-3", "ordinary-answer", "final_answer")
        await phase_fixtures.speech(self.session, "response-3", "ordinary-answer", "Four.")
        await phase_fixtures.finish(self.session, "response-3")
        self.assertFalse(self.session._normal_turn_pending)
        self.assertFalse(self.session._closed.is_set())
        self.assertFalse(self.session._image_tool_requests)
        self.assertFalse(self.state.requested_pins)

    async def test_barge_in_cancels_the_production_historical_storage_resolver(self):
        reference, repository, loader, arguments = self.register_historical_tool()
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def download(**kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        loader.download_object.side_effect = download
        await self.start()
        self.call(name="get_memory_image", arguments=arguments)
        await asyncio.wait_for(entered.wait(), 1)
        await self.fixture.append()
        self.done(status="cancelled")
        await self.until(lambda: not self.session._image_tool_tasks)
        self.assertTrue(cancelled.is_set())
        self.assertFalse(self.image_events())
        self.assertFalse(self.state.requested_pins)
        self.assertTrue(self.session._user_audio_pending)

    def register_historical_tool(self):
        evidence = _evidence(now=self.clock.now())
        repository = SimpleNamespace(get_image_evidence=AsyncMock(return_value=evidence))
        loader = SimpleNamespace(download_object=AsyncMock(return_value=_jpeg()))
        service = _service(repository, loader, now=self.clock.now)
        self.session.tool_registry.register(create_get_memory_image_tool(service=service, owner_id="owner"))
        arguments = {
            "record_id": str(evidence.reference.record_id),
            "record_type": evidence.reference.record_type,
        }
        return evidence.reference, repository, loader, arguments

    async def test_historical_pixels_attach_without_a_current_frame_or_stream(self):
        reference, repository, loader, arguments = self.register_historical_tool()
        await self.start()
        self.call(name="get_memory_image", arguments=arguments)
        self.done()
        recorded = await self.continuation()
        result = self.outputs()[0][1]
        self.assertEqual(result["image_status"], "attached")
        self.assertEqual(recorded["cutoff"].requested_images[0].reference, reference)
        self.assertNotEqual(reference.session_id, self.session.context.jarvis_session_id)
        self.assertIsNone(recorded["cutoff"].current_frame)
        self.assertEqual(len(self.image_events()), 1)
        self.assertIn("NOT the current camera view", self.image_events()[0]["item"]["content"][0]["text"])
        self.assertEqual(repository.get_image_evidence.await_count, 4)
        loader.download_object.assert_awaited_once()
        self.handler.assert_not_awaited()
        self.resolver.assert_not_awaited()

    async def test_missing_historical_object_does_not_fall_back_to_text_or_current_pixels(self):
        reference, repository, loader, arguments = self.register_historical_tool()
        loader.download_object.side_effect = JarvisHTTPError(
            502, "visual_frame_storage_download_failed", "missing", details={"status_code": 404},
        )
        await self.start()
        self.call(name="get_memory_image", arguments=arguments)
        self.done()
        recorded = await self.continuation()
        self.assertEqual(self.outputs()[0][1]["image_status"], "unavailable")
        self.assertEqual(self.outputs()[0][1]["error"]["code"], "frame_unavailable")
        self.assertFalse(self.image_events())
        self.assertFalse(recorded["cutoff"].requested_images)
        self.assertIsNone(recorded["cutoff"].current_frame)
        self.assertEqual(self.executions.records["image-1"].result["image_status"], "selected")

    async def test_forgetting_while_historical_image_waits_for_writer_prevents_delivery(self):
        reference, repository, loader, arguments = self.register_historical_tool()
        await self.start()
        await self.provider._send_lock.acquire()
        try:
            self.call(name="get_memory_image", arguments=arguments)
            await self.until(lambda: self.provider._foreground_waiters > 0)
            repository.get_image_evidence.return_value = None
        finally:
            self.provider._send_lock.release()
        self.done()
        await self.continuation()
        self.assertEqual(self.outputs()[0][1]["error"]["code"], "frame_unavailable")
        self.assertFalse(self.image_events())
        self.assertFalse(self.state.requested_pins)
        loader.download_object.assert_awaited_once()

    async def test_current_and_historical_requests_share_delivery_without_replacing_current_view(self):
        historical, repository, loader, arguments = self.register_historical_tool()
        frame = replace(
            _frame("owner", self.session.context.jarvis_session_id),
            visual_session_id=self.lease.visual_session_id, frame_sequence=1,
            captured_at=self.clock.now(), server_received_at=self.clock.now(),
            raw_image_expires_at=self.clock.now() + timedelta(seconds=30),
            image_bytes=_jpeg(), width=48, height=32,
        )
        self.window.put(frame, stream_lease=self.lease)
        current_service = CurrentImageService(
            frame_window=self.window, max_age_seconds=10, max_image_bytes=512_000,
            max_width=1024, max_height=1024, now=self.clock.now, monotonic=self.clock.monotonic,
        )
        self.session.tool_registry.register_many(CurrentImageToolModule(current_service).build_tools(ToolBuildContext(
            settings=self.session.settings, owner_id="owner", session_id=self.session.context.jarvis_session_id,
        )))
        await self.start()
        self.call("current", name="get_current_camera_view", arguments={})
        self.call("historical", name="get_memory_image", arguments=arguments)
        self.done()
        recorded = await self.continuation()
        self.assertEqual(recorded["cutoff"].current_frame.frame_id, frame.frame_id)
        self.assertEqual({pin.reference.source for pin in recorded["cutoff"].requested_images}, {"current", "historical"})
        self.assertEqual({pin.reference.frame_id for pin in recorded["cutoff"].requested_images}, {frame.frame_id, historical.frame_id})
        self.assertEqual({result["image_status"] for _, result in self.outputs()}, {"attached"})
        self.assertEqual(len(self.image_events()), 2)
        self.assertEqual(self.window.stream_lease("owner", self.session.context.jarvis_session_id), self.lease)

    async def test_current_pixels_reach_final_answer_without_a_secondary_model(self):
        frame = replace(
            _frame("owner", self.session.context.jarvis_session_id),
            visual_session_id=self.lease.visual_session_id, frame_sequence=1,
            captured_at=self.clock.now(), server_received_at=self.clock.now(),
            raw_image_expires_at=self.clock.now() + timedelta(seconds=30),
            image_bytes=_jpeg(), width=48, height=32,
        )
        self.window.put(frame, stream_lease=self.lease)
        service = CurrentImageService(
            frame_window=self.window, max_age_seconds=10, max_image_bytes=512_000,
            max_width=1024, max_height=1024, now=self.clock.now, monotonic=self.clock.monotonic,
        )
        reference = service.select_current_image(owner_id="owner", session_id=self.session.context.jarvis_session_id)
        self.images[0] = service.resolve_current_image(
            owner_id="owner", session_id=self.session.context.jarvis_session_id, reference=reference,
        )
        self.session.tool_registry.register_many(CurrentImageToolModule(service).build_tools(ToolBuildContext(
            settings=self.session.settings, owner_id="owner", session_id=self.session.context.jarvis_session_id,
        )))
        await self.start()
        self.call(name="get_current_camera_view", arguments={})
        self.done()
        recorded = await self.continuation()
        self.assertEqual(recorded["cutoff"].current_frame.frame_id, frame.frame_id)
        self.assertEqual(recorded["cutoff"].requested_images[0].reference, reference)
        self.assertTrue(self.image_events())
        self.handler.assert_not_awaited()
        self.resolver.assert_not_awaited()
        await phase_fixtures.output_item(self.session, "response-2", "answer", "final_answer")
        await phase_fixtures.speech(self.session, "response-2", "answer", "Is the sign on your left the one you mean?")
        await phase_fixtures.finish(self.session, "response-2")
        self.assertFalse(self.session._normal_turn_pending)
        self.assertFalse(self.state.pinned_item_ids)
        self.assertEqual(self.executions.records["image-1"].result["image_status"], "selected")
        self.assertNotIn("image_bytes", json.dumps(self.executions.finalized, default=str))
