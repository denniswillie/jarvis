"""Opt-in native revision, interruption, reconnect and mixed-notification acceptance.

Reuses the existing native delivery observer and production worker/session/inbox.
The worker and inbox coordinator outlive both actual phone websocket connections.
Input is synthetic speech, playback/receipts use the real Swift simulator client.
An explicit test-driver inbox replay represents a user replay action, not a
claim that the foreground model or an unfinished phone-inbox UI supports it.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack, suppress
from datetime import UTC, datetime, timedelta
import json
import logging
from pathlib import Path
import time
from types import SimpleNamespace
from uuid import UUID, uuid4, uuid5

from openai import AsyncOpenAI
from websockets.asyncio.server import serve

from app.attention.models import InboxMutation
from app.attention.repository import AttentionRepository
from app.attention.service import AttentionService
from app.attention.tts import OpenAISpeechProvider, SpeechConfiguration
from app.experience.repository import ToolExecutionRepository
from app.realtime.events import encode_event
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.realtime.session_manager import RealtimeSessionManager
from app.reminders.delivery import ReminderDeliveryService
from app.reminders.repository import ReminderRepository
from app.runtime.feature_config import AttentionVoiceConfiguration, FrontierConfiguration, TaskLimits
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes import HermesTaskProvider
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from app.tasks.session_context import SessionTaskContextService
from app.tasks.voice_tools import BackgroundTaskToolModule
from app.tasks.worker import TaskWorkerRole, WorkerContextFactory, WorkerRoleConfiguration
from evals.gemini_live.transport_probe import write_wav
from evals.second_brain.native_expert_delivery import (
    ObservedRealtime, SocketPhone, delivery_checks, history_matches, microphone, prepare_speech, snapshot,
)
from evals.second_brain.native_voice_tasks import (
    ObservedExecutor, completed_audio_ids, foreground_context_providers, wait_until,
)
from evals.second_brain.source_fixture import source_fixture


def final_history(entries, items):
    """Select acknowledgments of final attempts, retaining older ones in artifacts."""
    expected = set()
    for entry in entries:
        if not entry["attempts"]:
            continue
        attempt = UUID(str(entry["attempts"][-1]["id"]))
        namespace = "expert" if entry["source_namespace"] == "task_result" else "notification"
        expected.update(uuid5(attempt, f"{namespace}:{index}").hex
                        for index in range(len(entry["expected_texts"])))
    return [item for item in items if item["id"] in expected]


def recovery_checks(result, entries):
    """Frozen mechanical gates; prose reasoning/status accuracy remains independent."""
    task_ids = {task["id"] for task in result.get("tasks", [])}
    initial = result.get("accepted_task_ids", [])
    final_tasks = result.get("tasks", [])
    first = result.get("interrupted_intent")
    replayed = result.get("explicit_replays", [])
    before = result.get("before_replay", [])
    before_by_id = {row["intent_id"]: row for row in before}
    final_by_id = {row["intent_id"]: row for row in entries}
    counts_match = (bool(before) and all(row["attempt_count"] == 1 for row in before)
        and set(before_by_id) == set(final_by_id)) and all(
        row["attempt_count"] == before_by_id[identity]["attempt_count"] + (identity in replayed)
        for identity, row in final_by_id.items())
    before_tasks = {task["id"]: task for task in result.get("tasks_before_interrupt", [])}
    current_tasks = {task["id"]: task for task in final_tasks}
    common = delivery_checks(entries, expected=3)
    common.pop("one_attempt_per_result")
    return common | {
        # Explicitly authorized replay adds an attempt, not another task or result.
        "only_authorized_extra_attempts": counts_match,
        "same_accepted_tasks": len(initial) == len(task_ids) == 2 and set(initial) == task_ids,
        "exact_user_revision": len(final_tasks) == 2 and sorted(t["revision"] for t in final_tasks) == [1, 2],
        "tasks_succeeded": len(final_tasks) == 2 and all(t["state"] == "succeeded" for t in final_tasks),
        "interruption_recorded": result.get("interrupt_state") == "unknown"
            and result.get("client_interruption_receipt") is True,
        "unknown_not_automatically_replayed": first is not None
            and result.get("interrupted_state_after_reconnect") == "unknown"
            and first in before_by_id and before_by_id[first]["state"] == "unknown",
        "explicit_replay_only": first in replayed and len(replayed) == len(set(replayed)) and counts_match,
        "completed_result_unchanged": first is not None and bool(before_tasks)
            and all(identity in current_tasks and current_tasks[identity]["revision"] == task["revision"]
                and (task["state"] != "succeeded" or (
                    current_tasks[identity]["result_payload_id"] == task["result_payload_id"]))
                for identity, task in before_tasks.items()),
        "same_conversation_new_audio_connection": len(result.get("connections", [])) == 2
            and len({row["audio_session_id"] for row in result["connections"]}) == 2
            and len({row["session_id"] for row in result["connections"]}) == 1,
        "mixed_source_delivery": sorted(row["source_namespace"] for row in entries)
            == ["reminder_event", "task_result", "task_result"],
        "provider_accepted_final_history": history_matches(
            entries, final_history(entries, result.get("native_history_items", []))),
        "foreground_responded_after_reconnect": bool(result.get("status_response_texts")),
        "superseded_results_unavailable": all(not row["source_available"] for row in result.get("superseded_deliveries", [])),
        "all_capture_flushed": result.get("capture_flushed") == [True, True],
        "no_application_errors": not result.get("application_errors"),
    }


async def current_snapshot(tasks, inbox, owner, phone, result):
    current, entries = await snapshot(tasks, inbox, owner, phone)
    revisions = {str(task.id): task.revision for task in current}
    selected, superseded = [], []
    for entry in entries:
        if (entry["source_namespace"] == "task_result"
                and revisions.get(entry["task_id"]) != entry["task_revision"]):
            superseded.append(entry)
        else:
            selected.append(entry)
    result["superseded_deliveries"] = superseded
    return current, selected


async def task_snapshot(tasks, owner):
    return [{"id": str(task.id), "state": task.status.value, "revision": task.revision,
             "model": task.model, "result_payload_id": str(task.result_payload_id) if task.result_payload_id else None}
            for task in await tasks.list(owner, limit=20)]


async def run(args):
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    prepare_speech(args.output / "speech", "multitask")
    prepare_speech(args.output / "revision", "revise")
    prepare_speech(args.output / "status", "status")
    settings = JarvisSettings()
    started = time.monotonic()
    result = {
        "format": "jarvis.native_expert_recovery.v1", "ok": False, "case": "recovery",
        "native_foreground": True, "native_frontier_reasoning": True, "native_tts": True,
        "frontier_model": args.frontier_model, "foreground_model": settings.OPENAI_REALTIME_MODEL,
        "synthetic_microphone": True, "physical_glasses": False, "human_hearing": "not_established",
        "semantic_review": "pending", "explicit_replay_origin": "authorized_fixture_inbox_action_not_voice_tool",
        "tools": [], "worker_events": [], "connections": [], "native_history_items": [],
        "capture_flushed": [], "explicit_replays": [], "application_errors": [],
    }
    phones, native, worker_job = [], None, None

    def active_native():
        return sorted(str(run.context.task.id) for run in tuple(native._runs.values())
                      if run.process is not None and run.process.returncode is None) if native else []

    def observed_phone():
        return SimpleNamespace(events=[event for phone in phones for event in phone.events])

    try:
        async with source_fixture(settings.SUPABASE_DB_URL, args.output,
                                  test_name="native-expert-recovery") as fixture:
            tasks = TaskRepository(fixture.pool, TaskLimits())
            context = ContextRepository(tasks)
            messages = SessionRepository(fixture.pool)
            catalog = TaskWorkerRegistry(fixture.pool)
            service = TaskService(tasks, {"frontier": ConfiguredTaskProvider(args.frontier_model)}, registry=catalog)
            inbox = AttentionRepository(tasks)
            attention = AttentionService(inbox)
            service.attention_service = attention
            manager = RealtimeSessionManager()
            manager.attention_notifications_enabled, manager.attention_repository = True, inbox
            session = await messages.create_session(fixture.owner, "Synthetic expert recovery")
            stop, finished = asyncio.Event(), asyncio.Event()
            connection_gate = asyncio.Lock()

            async with AsyncExitStack() as lifetime:
                configuration = WorkerRoleConfiguration(python=args.hermes_python,
                    work_root=args.output / "native-state", host_id="recovery-" + uuid4().hex)

                class ObservedHermes(HermesTaskProvider):
                    async def execute(self, context):
                        row = {"task_id": str(context.task.id), "revision": context.task.revision,
                               "started_seconds": time.monotonic() - started}
                        result["worker_events"].append(row)
                        try:
                            outcome = await super().execute(context)
                            row.update(state=outcome.status.value, reason=outcome.reason)
                            return outcome
                        finally:
                            row["ended_seconds"] = time.monotonic() - started

                native = ObservedHermes(FrontierConfiguration(model=args.frontier_model,
                    reasoning_effort="medium", api_key=settings.OPENAI_API_KEY),
                    configuration.native(), prepare=WorkerContextFactory(fixture.pool))
                worker = TaskWorkerRole(tasks, {"frontier": native},
                    owner_id=fixture.owner, configuration=configuration)
                worker_job = asyncio.create_task(worker.run(stop))

                async def drain_worker():
                    stop.set()
                    await asyncio.wait_for(worker_job, 30)
                lifetime.push_async_callback(drain_worker)
                async with asyncio.timeout(30):
                    while not await catalog.ready(fixture.owner, "frontier", args.frontier_model):
                        if worker_job.done():
                            await worker_job
                            raise RuntimeError("Native worker stopped before readiness")
                        await asyncio.sleep(.1)
                client = await lifetime.enter_async_context(AsyncOpenAI(
                    api_key=settings.OPENAI_API_KEY, max_retries=0))
                attention.start_voice(manager=manager, owner=fixture.owner,
                    speech=OpenAISpeechProvider(client, SpeechConfiguration()),
                    config=AttentionVoiceConfiguration(api_key=settings.OPENAI_API_KEY))
                lifetime.push_async_callback(attention.close)
                lifetime.push_async_callback(manager.close_all)

                async def idle_response(voice, phone, jobs, boundary):
                    await wait_until(lambda: bool(completed_audio_ids([
                        row["event"] for row in phone.observations[boundary:]
                        if not str(row["event"].get("response_id", "")).startswith("tts_")]))
                        and not voice._normal_turn_pending and not voice._native_playback,
                        running=jobs, timeout=120)

                async def handle(socket):
                    if socket.request.headers.get("Authorization") != "Bearer fixture-token":
                        await socket.close(code=1008)
                        return
                    async with connection_gate:
                        if len(phones) >= 2 or finished.is_set():
                            await socket.close(code=1008)
                            return
                        index = len(phones) + 1
                        phone = SocketPhone(socket, active_native)
                        phone.started = started
                        phones.append(phone)
                        audio_id = "recovery-audio-" + uuid4().hex
                        result["connections"].append({"audio_session_id": audio_id,
                            "session_id": str(session.id), "index": index})
                        capture = SessionContextCapture(context, owner_id=fixture.owner,
                            session_id=session.id, provider="openai", audio_session_id=audio_id)
                        voice = RealtimeAudioSession(websocket=phone,
                            context=RealtimeSessionContext(audio_session_id=audio_id, owner_id=fixture.owner,
                                jarvis_session_id=session.id, client_kind="phone_audio", session=session),
                            settings=settings, repository=messages,
                            tool_modules=(BackgroundTaskToolModule(service),),
                            context_providers=foreground_context_providers(SessionTaskContextService(context)),
                            context_capture=capture, tool_execution_repository=ToolExecutionRepository(fixture.pool))
                        voice.openai = ObservedRealtime(settings=settings, owner_id=fixture.owner,
                                                        history=result["native_history_items"])
                        voice.tool_executor = ObservedExecutor(voice.tool_executor, result["tools"], started)
                        session_job = None
                        try:
                            if not await manager.register(audio_session_id=audio_id, owner_id=fixture.owner,
                                                          jarvis_session_id=session.id, session=voice):
                                raise RuntimeError("Overlapping foreground registration")
                            session_job = asyncio.create_task(voice.run())
                            jobs = (worker_job, session_job, attention._task)
                            await wait_until(lambda: any(event["type"] == "session.created" for event in phone.events),
                                             running=jobs, timeout=30)
                            if index == 1:
                                await microphone(voice, args.output / "speech", "multitask")
                                await idle_response(voice, phone, jobs, 0)
                                accepted = await task_snapshot(tasks, fixture.owner)
                                if len(accepted) != 2:
                                    raise RuntimeError("Exactly two native delegated tasks are required")
                                result["accepted_task_ids"] = [row["id"] for row in accepted]
                                boundary = len(phone.observations)
                                await microphone(voice, args.output / "revision", "revise")
                                await idle_response(voice, phone, jobs, boundary)
                                result["tasks_after_revision"] = await task_snapshot(tasks, fixture.owner)
                                if sorted(row["revision"] for row in result["tasks_after_revision"]) != [1, 2]:
                                    raise RuntimeError("Foreground did not revise exactly one accepted task")
                                await wait_until(lambda: voice._attention_delivery is not None
                                    and any(event["type"] == "audio.output.delta"
                                        and event.get("response_id") == voice._attention_delivery.response_id
                                        for event in phone.events), running=jobs, timeout=180)
                                delivery = voice._attention_delivery
                                result["interrupted_intent"] = str(delivery.claim.intent_id)
                                result["tasks_before_interrupt"] = await task_snapshot(tasks, fixture.owner)
                                result["native_tasks_at_interrupt"] = active_native()
                                await voice._app_sender.send_text(encode_event("fixture.interrupt"))
                                async with asyncio.timeout(20):
                                    while True:
                                        detail = await inbox.detail(fixture.owner, delivery.claim.intent_id)
                                        if detail["state"] == "unknown" and voice._attention_delivery is not delivery:
                                            break
                                        await asyncio.sleep(.1)
                                result["interrupt_state"] = detail["state"]
                                result["interrupted_attempt"] = detail["attempts"][-1]
                                result["client_interruption_receipt"] = any(
                                    row["event"]["type"] == "playback.interrupted"
                                    and row["event"].get("response_id") == delivery.response_id
                                    for row in phone.client_events)
                                result["native_tasks_at_disconnect"] = active_native()
                                await voice._app_sender.send_text(encode_event("fixture.disconnect"))
                                await asyncio.wait_for(asyncio.shield(session_job), 30)
                                result["worker_alive_after_disconnect"] = not worker_job.done()
                            else:
                                first = await inbox.detail(fixture.owner, UUID(result["interrupted_intent"]))
                                result["interrupted_state_after_reconnect"] = first["state"]
                                boundary = len(phone.observations)
                                await microphone(voice, args.output / "status", "status")
                                await idle_response(voice, phone, jobs, boundary)
                                result["status_response_texts"] = [
                                    row["event"].get("text") for row in phone.observations[boundary:]
                                    if row["event"]["type"] == "transcript.done"
                                    and not str(row["event"].get("response_id", "")).startswith("tts_")]
                                # Let any genuinely pending occurrence finish. Unknown/failed
                                # delivery cannot be re-enqueued without this explicit action.
                                async with asyncio.timeout(args.delivery_timeout):
                                    while True:
                                        current, entries = await current_snapshot(tasks, inbox, fixture.owner, observed_phone(), result)
                                        if len(entries) == 3 and all(row["state"] in {"unknown", "failed", "played"} for row in entries):
                                            break
                                        if session_job.done():
                                            raise RuntimeError("Reconnected session stopped before settling")
                                        await asyncio.sleep(1)
                                result["before_replay"] = [
                                    {key: row[key] for key in ("intent_id", "state", "attempt_count")} for row in entries]
                                for row in entries:
                                    if row["state"] in {"unknown", "failed"}:
                                        identity = UUID(row["intent_id"])
                                        detail = await inbox.detail(fixture.owner, identity)
                                        await inbox.mutate(fixture.owner, identity, "replay",
                                            InboxMutation(request_id=uuid4(), expected_generation=detail["generation"]))
                                        result["explicit_replays"].append(row["intent_id"])
                                async with asyncio.timeout(args.delivery_timeout):
                                    while True:
                                        current, entries = await current_snapshot(tasks, inbox, fixture.owner, observed_phone(), result)
                                        if len(entries) == 3 and all(row["state"] == "played" for row in entries):
                                            break
                                        if session_job.done():
                                            raise RuntimeError("Reconnected session stopped during replay")
                                        await asyncio.sleep(1)
                                result["tasks"] = await task_snapshot(tasks, fixture.owner)
                                result["deliveries"] = entries
                                result["capture_flushed"].append(await capture.flush())
                                result["application_errors"] = [event for p in phones for event in p.events
                                                               if event["type"] == "error"]
                                result["checks"] = recovery_checks(result, entries) | {
                                    "worker_survived_disconnect": result.get("worker_alive_after_disconnect") is True,
                                }
                                async with fixture.pool.acquire() as connection:
                                    attempts = await connection.fetch(
                                        "select task_id,task_revision,count(*) as attempts from jarvis_task_attempts where owner_id=$1 group by task_id,task_revision",
                                        fixture.owner)
                                result["task_attempts"] = [dict(row) for row in attempts]
                                result["checks"]["no_worker_rerun_after_interruption"] = (
                                    2 <= len(attempts) <= 3 and all(row["attempts"] == 1 for row in attempts))
                                result["ok"] = all(result["checks"].values())
                                await voice._app_sender.send_text(encode_event("fixture.result",
                                    ok=result["ok"], case="recovery", state="played", checks=result["checks"],
                                    native_foreground=True, native_frontier_reasoning=True, native_tts=True,
                                    expected_tasks=2, connections=2))
                        except Exception as error:
                            result.update(ok=False, error_type=type(error).__name__, failed_connection=index)
                            with suppress(Exception):
                                _, entries = await current_snapshot(tasks, inbox, fixture.owner, observed_phone(), result)
                                result["deliveries"] = entries
                                result["tasks"] = await task_snapshot(tasks, fixture.owner)
                            with suppress(Exception):
                                await socket.send(encode_event("fixture.result", ok=False, error_type=type(error).__name__))
                            finished.set()
                        finally:
                            await voice.close("recovery_fixture_connection_finished")
                            if session_job is not None:
                                with suppress(Exception):
                                    await asyncio.wait_for(session_job, 20)
                            if index == 1:
                                result["capture_flushed"].append(await capture.flush())
                            await capture.close()
                            await manager.unregister(audio_session_id=audio_id, owner_id=fixture.owner, jarvis_session_id=session.id)
                            if index == 1 and not finished.is_set():
                                reminders = ReminderRepository(fixture.pool)
                                reminder = await reminders.create_reminder(owner_id=fixture.owner, session_id=session.id,
                                    title="Check the synthetic fixture", details="This reminder does not prove any physical action.",
                                    remind_at=datetime.now(UTC)-timedelta(seconds=1), condition=None)
                                claimed = await reminders.claim_due_reminders(owner_id=fixture.owner,
                                    claim_timeout_seconds=120, max_attempts=3)
                                delivery_service = ReminderDeliveryService(owner_id=fixture.owner, repository=reminders,
                                    session_repository=messages, realtime_session_manager=manager, generator=None)
                                await delivery_service.deliver(next(row for row in claimed if row.id == reminder.id))
                                result["reminder_queued_while_disconnected"] = True
                            if index == 2:
                                finished.set()

                async with serve(handle, "127.0.0.1", 0, max_size=65536) as server:
                    print(json.dumps({"ready": True, "url": "ws://127.0.0.1:" + str(server.sockets[0].getsockname()[1])}), flush=True)
                    await asyncio.wait_for(finished.wait(), args.delivery_timeout * 2 + 480)
            result["native_processes_remaining"] = active_native()
            if result["native_processes_remaining"]:
                result["ok"] = False
        result["fixture_cleanup"] = "completed"
    except BaseException as error:
        result.update(ok=False, error_type=type(error).__name__)
        if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise
    finally:
        result["elapsed_ms"] = (time.monotonic() - started) * 1000
        result["observations"] = [row for phone in phones for row in phone.observations]
        result["client_events"] = [row for phone in phones for row in phone.client_events]
        result["audio_files"] = []
        for index, phone in enumerate(phones, 1):
            for ordinal, (item, pcm) in enumerate(phone.audio.items(), 1):
                filename = f"connection-{index}-forwarded-{ordinal}.wav"
                write_wav(args.output / filename, bytes(pcm))
                result["audio_files"].append({"item_id": item, "file": filename})
        (args.output / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({key: result.get(key) for key in (
        "ok", "error_type", "checks", "fixture_cleanup", "native_processes_remaining")}), flush=True)
    return result["ok"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hermes-python", type=Path, required=True)
    parser.add_argument("--frontier-model", default="gpt-5.6-sol")
    parser.add_argument("--delivery-timeout", type=int, default=1200)
    parser.add_argument("--allow-native-models", action="store_true", required=True)
    logging.disable(logging.CRITICAL)
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 1)
