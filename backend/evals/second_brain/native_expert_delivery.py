"""Opt-in real foreground -> Hermes -> automatic expert TTS -> simulator receipts.

Only microphone speech and the loopback test driver are synthetic. No foreground
tools, worker results, TTS output, turn admission or playback receipts are mocked.
Personal storage is never used: the existing manifested source fixture owns all
rows, and cleanup drains the session/worker before removing its schema.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import AsyncExitStack, suppress
import json
import logging
import os
from types import SimpleNamespace
from pathlib import Path
import subprocess
import time
import traceback
from uuid import UUID, uuid4

from openai import AsyncOpenAI
from websockets.asyncio.server import serve

from app.attention.notifications import speech_attribution, speech_chunks
from app.attention.repository import AttentionRepository
from app.attention.service import AttentionService
from app.attention.tts import OpenAISpeechProvider, SpeechConfiguration
from app.experience.repository import ToolExecutionRepository
from app.knowledge.current_context import SourceContextProvider
from app.knowledge.runtime import KnowledgeRuntime
from app.knowledge.selection import configure_selected_memory
from app.realtime.events import encode_event
from app.realtime.models import RealtimeSessionContext
from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.session import RealtimeAudioSession
from app.realtime.session_manager import RealtimeSessionManager
from app.runtime.feature_config import (
    AttentionVoiceConfiguration, DEFAULT_FRONTIER_MODEL, FrontierConfiguration, TaskLimits,
    KnowledgeConfiguration, MemoryAuthority,
)
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes import HermesTaskProvider
from app.tasks.repository import TaskRepository
from app.tasks.memory_corrections import MEMORY_CORRECTION_CAPABILITY
from app.tasks.models import TaskError, TaskVoiceSource, digest, json_object
from app.tools.invocation import current_tool_invocation
from app.tasks.service import TaskService
from app.tasks.session_context import SessionTaskContextService
from app.tasks.voice_tools import BackgroundTaskToolModule
from app.tasks.worker import TaskWorkerRole, WorkerContextFactory, WorkerRoleConfiguration
from evals.gemini_live.transport_probe import read_pcm, write_wav
from evals.second_brain.native_voice_tasks import (
    ObservedExecutor, ObservedPhone, UTTERANCES, completed_audio_ids,
    foreground_context_providers, safe_app_event, wait_until,
)
from evals.second_brain.source_fixture import source_fixture
from evals.second_brain.native_memory_lifecycle import (
    context_reset_notice, load_cases, snapshot as source_snapshot,
)
from evals.second_brain.hindsight_semantic_trace import drain


AMBIGUOUS_MEMORY_CASES = ("ambiguous_correction", "expert_ambiguous_correction")
MEMORY_CASES = ("precise_correction", "exact_history", *AMBIGUOUS_MEMORY_CASES)
MEMORY_CLARIFICATION = "Version one only."


def memory_case_for(name):
    if name == "exact_history":
        return {"id": name, "sources": [], "targets": ["mass-after"],
            "utterance": "Please correct only my body mass entry for September fourteenth twenty twenty-six "
                "to eighty-one point zero five kilograms instead of eighty-one point three five. "
                "Preserve my other entries and calculate the change from September twelfth."}
    if name not in MEMORY_CASES:
        return None
    cases, _ = load_cases()
    original = "ambiguous_correction" if name == "expert_ambiguous_correction" else name
    case = dict(next(case for case in cases if case["id"] == original))
    if name == "expert_ambiguous_correction":
        case.update(id=name, utterance=(
            "Please delegate this memory correction to the background expert. " + case["utterance"]
            + " If the expert needs clarification, have it ask me before changing anything."))
    return case


def scenario_utterances(name):
    case = memory_case_for(name)
    utterances = {name: case["utterance"] if case else UTTERANCES[name],
                  "ordinary": UTTERANCES["ordinary"]}
    if name == "exact_history":
        utterances.update(
            exact_record="Please record two body mass measurements for me, in kilograms. "
                "On September twelfth twenty twenty-six, eighty point two. On September fourteenth, "
                "eighty-one point three five. Also record a separate running distance of three kilometres "
                "on September fourteenth twenty twenty-six.",
            exact_compare="Calculate the change between my two saved body mass entries.",
            exact_recall="What were my body mass measurements on September twelfth and fourteenth "
                "twenty twenty-six, what is the change between them, and what running distance did I record?")
    if name in AMBIGUOUS_MEMORY_CASES:
        utterances["memory_clarification"] = MEMORY_CLARIFICATION
    return utterances


def memory_effect_checks(case, before, after, actions, foreground_tools, *, clarified=False):
    """Frozen effect oracles, not semantic grading or model instructions."""
    waiting = case["id"] in AMBIGUOUS_MEMORY_CASES and not clarified
    targets = () if waiting else tuple(case.get("targets", ("cedar-v1",)))
    unchanged = set(before) - set(targets)
    accepted = [row for row in actions if row["status"] == "accepted"
                and row["capability"] == MEMORY_CORRECTION_CAPABILITY]
    return {
        "untargeted_originals_unchanged": all(after[key] == before[key] for key in unchanged),
        "exact_correction_effect_count": len(actions) == len(accepted) == len(targets),
        "one_versioned_replacement": all(
            after[key]["revision"] == before[key]["revision"] + 1
            and after[key]["state"] == "active" and after[key]["trust"] == "derived"
            and after[key]["original_content_retained"] for key in targets),
        "no_foreground_replacement": not any(
            row["tool"] == "correct_memory_source" for row in foreground_tools),
    }


def exception_details(error):
    """Synthetic-fixture diagnostics without arbitrary exception prose or locals."""
    details, seen = [], set()
    current = error
    while current is not None and id(current) not in seen and len(details) < 8:
        seen.add(id(current))
        item = {"error_type": type(current).__name__, "error_code": getattr(current, "code", None),
                "sqlstate": getattr(current, "sqlstate", None),
                "frames": [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                           for frame in traceback.extract_tb(current.__traceback__)]}
        if type(current).__module__.startswith("asyncpg."):
            item["database_message"] = str(current)
        details.append(item)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return details


async def memory_actions(fixture):
    async with fixture.pool.acquire() as connection:
        rows = await connection.fetch(
            """select a.id,a.task_id,a.task_revision,a.status,a.grant_id,g.capability,
                      p.availability as receipt_availability,p.content as receipt
               from jarvis_task_actions a
               left join jarvis_operation_grants g on g.owner_id=a.owner_id and g.id=a.grant_id
               left join jarvis_context_payloads p on p.owner_id=a.owner_id and p.id=a.receipt_payload_id
               where a.owner_id=$1 order by a.created_at,a.id""", fixture.owner)
    return [dict(row) | {"receipt": json_object(row["receipt"]) if row["receipt"] is not None else None}
            for row in rows]


async def memory_effects(fixture, bindings):
    return await source_snapshot(fixture, bindings), await memory_actions(fixture)


async def memory_failure_snapshot(fixture, bindings):
    # An action-query failure must not hide the independently readable originals.
    evidence = {}
    for name, operation in (("after", lambda: source_snapshot(fixture, bindings)),
                            ("actions", lambda: memory_actions(fixture))):
        try:
            evidence[name] = await operation()
        except Exception as error:
            evidence[name + "_error"] = exception_details(error)
    return evidence


def completed_source_reset(phone, *, session_id, audio_session_id):
    pending = None
    for event in phone.events:
        identity = context_reset_notice(event, session_id=session_id,
            audio_session_id=audio_session_id, pending=pending)
        if identity is None:
            continue
        if event["type"] == "session.context_invalidated":
            pending = identity
        elif pending is not None and any(
            row["event"].get("type") == "session.context_cleared"
            and row["event"].get("reset_id") == pending
            and row["event"].get("audio_session_id") == audio_session_id
            and type(row["event"].get("protocol_version")) is int
            and row["event"]["protocol_version"] == 1 for row in phone.client_events):
            return pending
    return None


class ObservedAdmissionRepository(TaskRepository):
    """Observe real writes; only a separate committed read proves acceptance."""
    def __init__(self, pool, limits, *, observations):
        super().__init__(pool, limits)
        self.admission_observations = observations

    async def _save_request(self, connection, task, request_id, operation, parameters):
        await super()._save_request(connection, task, request_id, operation, parameters)
        invocation = current_tool_invocation()
        if (invocation.task is None and invocation.provider_call_id
                and invocation.request_event_id is not None and "voice_source" in parameters):
            self.admission_observations.append({
                "owner_id": task.owner_id, "request_id": str(request_id), "task_id": str(task.id),
                "operation": operation, "parameters_digest": digest(parameters),
                "accepted_revision": task.revision, "call_id": invocation.provider_call_id,
                "request_event_id": str(invocation.request_event_id),
                "request_revision": invocation.request_revision,
                "voice_source": parameters["voice_source"],
                "memory_edit": parameters.get("edit_memory") is True,
                "committed": False,
            })

    async def refresh_admissions(self, owner_id, session_id, tool_rows):
        names = {"create": "start_background_task", "update": "update_background_task"}
        pending = [record for record in self.admission_observations
                   if record["owner_id"] == owner_id and record["operation"] in names]
        if not pending:
            return
        async with self.pool.acquire() as connection:
            for observed in pending:
                matching = [row for row in tool_rows
                            if row.get("call_id") == observed["call_id"]
                            and row["tool"] == names[observed["operation"]]]
                if not matching:
                    continue
                # The observer runs before its transaction commits. Never use
                # the attempted write or current task state as an acceptance
                # substitute. Match the immutable receipt and exact user source.
                receipt = await connection.fetchrow("""select r.task_id,r.operation,
                        r.parameters_digest,r.accepted_revision,e.session_id
                    from jarvis_task_requests r join jarvis_context_events e
                        on e.owner_id=r.owner_id and e.id=$3 and e.revision=$4
                    where r.owner_id=$1 and r.request_id=$2 and e.session_id=$5
                        and e.kind in ('user_request','user_transcript') and e.trust='user_input'
                        and e.finality in ('final','revised')""",
                    owner_id, UUID(observed["request_id"]), UUID(observed["request_event_id"]),
                    observed["request_revision"], session_id)
                if receipt is None or any((
                    str(receipt["task_id"]) != observed["task_id"],
                    receipt["operation"] != observed["operation"],
                    receipt["parameters_digest"] != observed["parameters_digest"],
                    receipt["accepted_revision"] != observed["accepted_revision"],
                )):
                    continue
                expected_source = TaskVoiceSource(UUID(observed["request_event_id"]),
                    observed["request_revision"], receipt["session_id"]).parameters()
                if observed["voice_source"] != expected_source:
                    continue
                observed["committed"] = True
                proof = {**observed, "session_id": str(receipt["session_id"]),
                         "basis": "exact_committed_task_request"}
                for row in matching:
                    # Do not fabricate a handler response or claim that its
                    # acknowledgment was delivered to the foreground provider.
                    row["durable_admission"] = proof


def accepted_task_admissions(tool_rows, operation, *, memory_edit=False):
    accepted = {}
    names = {"create": "start_background_task", "update": "update_background_task"}
    for row in tool_rows:
        proof = row.get("durable_admission", {})
        if (row.get("tool") == names.get(operation) and proof.get("call_id") == row.get("call_id")
                and proof.get("committed") is True and proof.get("basis") == "exact_committed_task_request"
                and proof.get("operation") == operation and (not memory_edit or proof.get("memory_edit") is True)):
            accepted[proof["request_id"]] = row
    return list(accepted.values())


class FixtureConversation:
    """One scenario over bounded real transports, never a replayed user request."""
    def __init__(self):
        self.rows = []
        self.current = None
        self.failure = None
        self.resets = []

    @property
    def events(self):
        return [event for row in self.rows for event in row.phone.events]

    def check(self, jobs=()):
        if self.failure is not None:
            raise self.failure
        for job in jobs:
            if job.done():
                job.result()
                raise TaskError("native_memory_runtime_stopped", "A required runtime stopped before delivery.")

    async def ready(self):
        async with asyncio.timeout(45):
            while True:
                self.check()
                row = self.current
                if row is not None and not row.job.done() and not any(
                        event["type"] == "session.context_invalidated" for event in row.phone.events):
                    return row
                await asyncio.sleep(.05)


async def exercise_memory_delivery(*, conversation, tasks, inbox, fixture, session_id, bindings,
                                   case, speech_dir, jobs, timeout, result):
    """Allow natural foreground clarification; retain a separate expert-first case."""
    memory = result["memory"]

    def admissions():
        return accepted_task_admissions(result["tools"], "create", memory_edit=True)

    async def speak(name):
        row = await conversation.ready()
        boundary, tool_boundary = len(conversation.events), len(result["tools"])
        await microphone(row.voice, speech_dir, name)
        memory.setdefault("spoken_inputs", []).append(name)
        async with asyncio.timeout(120):
            while True:
                conversation.check(jobs)
                new_tools = result["tools"][tool_boundary:]
                await tasks.refresh_admissions(fixture.owner, session_id, new_tools)
                admitted = bool(accepted_task_admissions(new_tools, "create")
                                or accepted_task_admissions(new_tools, "update"))
                foreground = [event for event in conversation.events[boundary:]
                    if not str(event.get("response_id", "")).startswith("tts_")]
                current = conversation.current
                idle = current is not None and not current.voice._native_playback \
                    and not current.voice._normal_turn_pending
                if admitted or (completed_audio_ids(foreground) and idle):
                    return foreground
                await asyncio.sleep(.1)

    async def deliveries(*, exclude=()):
        async with asyncio.timeout(timeout):
            while True:
                conversation.check(jobs)
                current, entries = await snapshot(tasks, inbox, fixture.owner, conversation)
                selected = [row for row in entries if row["intent_id"] not in exclude]
                if len(selected) == 1 and selected[0]["state"] == "played":
                    return current, selected
                if any(task.status.value in {"failed", "cancelled", "paused", "outcome_unknown"}
                       for task in current):
                    raise TaskError("native_memory_task_stopped", "The memory task did not finish normally.")
                await asyncio.sleep(.25)

    memory["phase"] = "spoken_request"
    foreground = await speak(case["id"])
    current = await tasks.list(fixture.owner, limit=20)
    foreground_clarified = False
    if case["id"] == "ambiguous_correction" and not current:
        after, actions = await memory_effects(fixture, bindings)
        checks = memory_effect_checks(case, memory["before"], after, actions, result["tools"])
        checks["foreground_reply_played"] = bool(completed_audio_ids(foreground))
        memory["foreground_clarification"] = {
            "texts": [event.get("text") for event in foreground if event["type"] == "transcript.done"],
            "after": after, "actions": actions, "checks": checks,
            "semantic_review": "Verify this actually asks which version; audio completion alone is not clarification."}
        if not all(checks.values()):
            raise TaskError("native_memory_premature_mutation", "Memory changed before the user's clarification.")
        memory["phase"] = "spoken_clarification"
        await speak("memory_clarification")
        foreground_clarified = True
        current = await tasks.list(fixture.owner, limit=20)

    memory["phase"] = "delegation_admission"
    if len(current) != 1 or len(admissions()) != 1:
        raise TaskError("native_memory_delegation_missing",
                        "The actual voice request did not admit one memory-editing task.")
    task_id = current[0].id
    memory["task_id"] = str(task_id)
    memory["phase"] = "initial_expert_delivery"
    current, entries = await deliveries()
    result["deliveries"] = entries
    after, actions = await memory_effects(fixture, bindings)
    memory["initial"] = {"after": after, "actions": actions, "deliveries": entries,
                         "task_states": [task.status.value for task in current],
                         "task_revisions": {str(task.id): task.revision for task in current}}
    checks = memory_effect_checks(case, memory["before"], after, actions,
                                  result["tools"], clarified=foreground_clarified)
    expert_clarified = case["id"] in AMBIGUOUS_MEMORY_CASES and not foreground_clarified
    if expert_clarified:
        checks["expert_requested_clarification"] = (
            len(current) == 1 and current[0].id == task_id
            and current[0].status.value == "waiting_for_input")
        memory["clarification_checks"] = checks
        if not all(checks.values()):
            result["checks"], result["deliveries"] = checks, entries
            return
        first_entries = entries
        memory["phase"] = "spoken_clarification"
        await speak("memory_clarification")
        memory["phase"] = "corrected_expert_delivery"
        current, new_entries = await deliveries(exclude={row["intent_id"] for row in first_entries})
        entries = first_entries + new_entries
        after, actions = await memory_effects(fixture, bindings)
        checks = memory_effect_checks(case, memory["before"], after, actions,
                                      result["tools"], clarified=True)
        updates = accepted_task_admissions(result["tools"], "update")
        checks["actual_spoken_task_update"] = (
            len(updates) == 1 and updates[0]["durable_admission"]["task_id"] == str(task_id)
            and updates[0]["durable_admission"]["accepted_revision"]
                == memory["initial"]["task_revisions"][str(task_id)] + 1)
        checks["clarification_before_mutation"] = all(memory["clarification_checks"].values())
    elif foreground_clarified:
        checks["clarification_before_mutation"] = all(memory["foreground_clarification"]["checks"].values())
    checks.update(delivery_checks(entries, expected=2 if expert_clarified else 1))
    checks["same_task_completed"] = (
        len(current) == 1 and current[0].id == task_id and current[0].status.value == "succeeded")
    checks["provider_accepted_expert_history"] = history_matches(entries, result["native_history_items"])
    checks["expert_authorship"] = expert_authorship_matches(entries, result["worker_events"],
        model=result["frontier_model"], owner_id=fixture.owner)
    memory.update(phase="completed", after=after, actions=actions,
                  clarification_origin="expert" if expert_clarified else "foreground" if foreground_clarified else None)
    result["tasks"] = [{"id": str(task.id), "state": task.status.value,
                        "revision": task.revision, "model": task.model} for task in current]
    result["deliveries"], result["checks"] = entries, checks



async def exercise_exact_history(*, services, settings, messages, knowledge, conversation,
                                 tasks, inbox, fixture, session_id, bindings, case,
                                 speech_dir, jobs, timeout, result):
    """One spoken record/calculate/correct sequence, then a genuinely fresh session."""
    from decimal import Decimal
    from app.knowledge.measurements import MeasurementRepository, MeasurementQuery
    from evals.second_brain.native_voice_tasks import answer
    exact = MeasurementRepository(knowledge.sources, tasks, owner_id=fixture.owner, area_id=fixture.area)
    answers = result["measurement_answers"] = []
    for name in ("exact_record", "exact_compare"):
        row = await conversation.ready()
        boundary = len(row.phone.events)
        await microphone(row.voice, speech_dir, name)
        await wait_until(lambda: bool(completed_audio_ids(row.phone.events[boundary:]))
            and not row.voice._native_playback and not row.voice._normal_turn_pending,
            running=(*jobs, row.job), timeout=90)
        ids = completed_audio_ids(row.phone.events[boundary:])
        answers.append({"utterance": name, "text": [e.get("text") for e in row.phone.events[boundary:]
            if e["type"] == "transcript.done" and e.get("response_id") in ids]})
    before = await exact.listing(session_id, MeasurementQuery())
    originals = {Decimal(row["measurement"]["value"]): row for row in before["records"]}
    if len(before["records"]) != 3 or set(originals) != {Decimal("80.2"), Decimal("81.35"), Decimal("3")}:
        raise TaskError("native_exact_records_mismatch", "The spoken request did not store three exact entries.")
    for label, value in (("mass-before", "80.2"), ("mass-after", "81.35"), ("running", "3")):
        bindings[label] = await fixture.sources.read(
            fixture.owner, fixture.area, UUID(originals[Decimal(value)]["measurement_id"]))
    result["memory"]["before"] = await source_snapshot(fixture, bindings)
    result["measurement_before"] = before
    await exercise_memory_delivery(conversation=conversation, tasks=tasks, inbox=inbox,
        fixture=fixture, session_id=session_id, bindings=bindings, case=case,
        speech_dir=speech_dir, jobs=jobs, timeout=timeout, result=result)
    after = await exact.listing(session_id, MeasurementQuery())
    by_id = {row["measurement_id"]: row for row in after["records"]}
    old = originals[Decimal("81.35")]
    revised = by_id.get(old["measurement_id"])
    control = originals[Decimal("3")]
    result["measurement_after"] = after
    result["checks"].update(
        exact_record_count=len(after["records"]) == 3,
        exact_corrected_value=bool(revised) and revised["revision"] == 2
            and Decimal(revised["measurement"]["value"]) == Decimal("81.05"),
        unrelated_exact_record_preserved=all(
            by_id.get(control["measurement_id"], {}).get(field) == control[field]
            for field in ("measurement_id", "revision", "measurement")),
        original_difference_calculated=any(row["tool"] == "calculate_measurements"
            and row.get("result", {}).get("calculation", {}).get("value") == "1.15"
            for row in result["tools"]),
        corrected_difference_calculated=any(row["name"] == "calculate_measurements"
            and row.get("result", {}).get("calculation", {}).get("value") == "0.85"
            for row in result["worker_tools"]),
    )
    fresh = await messages.create_session(fixture.owner, "Synthetic fresh exact-history recall")
    phone = ObservedPhone(lambda: [])
    audio_id = "fresh-exact-" + uuid4().hex
    capture = SessionContextCapture(knowledge.context_repository, owner_id=fixture.owner,
        session_id=fresh.id, provider="openai", audio_session_id=audio_id)
    voice = RealtimeAudioSession(websocket=phone,
        context=RealtimeSessionContext(audio_session_id=audio_id, owner_id=fixture.owner,
            jarvis_session_id=fresh.id, client_kind="phone_audio", session=fresh),
        settings=settings, repository=messages, tool_modules=services.tool_modules,
        context_providers=services.assistant_context_providers, context_capture=capture,
        tool_execution_repository=ToolExecutionRepository(fixture.pool))
    voice.tool_executor = ObservedExecutor(voice.tool_executor, result["tools"], phone.started)
    job = asyncio.create_task(voice.run())
    try:
        await wait_until(lambda: any(e["type"] == "session.created" for e in phone.events),
            running=(job,), timeout=30)
        answer_row = await answer(phone, speech_dir, "exact_recall", running=(job,))
        answers.append(answer_row)
        result["checks"]["fresh_session_answer"] = fresh.id != session_id and bool(answer_row["response_ids"])
        result["checks"]["fresh_session_capture"] = await capture.flush()
        result["checks"]["fresh_session_no_errors"] = not any(e["type"] == "error" for e in phone.events)
    finally:
        result["fresh_session_events"] = [safe_app_event(e) for e in phone.events]
        await voice.close("native_exact_history_finished")
        try:
            await asyncio.wait_for(job, 20)
        finally:
            await capture.close()


def authored_worker_results(events, *, model):
    """Use actual worker outcomes, not the public inbox's stripped projection."""
    for event in events:
        payload = event.get("payload") or {}
        author = payload.get("authorship") or {}
        if (author.get("kind") == "expert" and author.get("model") == model
                and author.get("owner_id") and author.get("owner_id") == event.get("owner_id")
                and author.get("task_id") and author.get("task_id") == event.get("task_id")
                and type(author.get("task_revision")) is int
                and author["task_revision"] == event.get("task_revision")):
            yield payload


def expert_authorship_matches(entries, events, *, model, owner_id):
    outcomes = list(authored_worker_results(events, model=model))
    return bool(entries) and all(any(
        payload["authorship"]["owner_id"] == owner_id
        and payload["authorship"]["task_id"] == entry["task_id"]
        and payload["authorship"]["task_revision"] == entry["task_revision"]
        and [part["text"] for part in payload.get("speech", [])] == entry["expected_texts"]
        for payload in outcomes) for entry in entries)


def delivery_checks(entries, *, expected):
    """Require authored text, exact durable sample receipts and no extra attempt."""
    return {
        "intent_count": len(entries) == expected,
        "all_played": bool(entries) and all(row["state"] == "played" for row in entries),
        "one_attempt_per_result": bool(entries) and all(row["attempt_count"] == 1 for row in entries),
        "exact_expert_text": bool(entries) and all(
            row["expected_texts"] and row["expected_texts"] == row["actual_texts"] for row in entries),
        "complete_chunk_receipts": bool(entries) and all(
            len(row["chunks"]) == len(row["expected_texts"])
            and [chunk["chunk_index"] for chunk in row["chunks"]] == list(range(len(row["expected_texts"])))
            and all(chunk["completed"] and chunk["generated"]
                    and chunk["forwarded_samples"] == chunk["played_samples"] > 0
                    for chunk in row["chunks"])
            for row in entries),
    }



def history_label(namespace):
    return "[" + speech_attribution(namespace)[2] + "; playback is not confirmed. Do not repeat automatically.]\n"


HISTORY_LABEL = history_label("task_result")
HISTORY_LABELS = tuple(history_label(namespace)
                      for namespace in ("task_result", "reminder_event", "visual_alert"))


def history_matches(entries, items):
    expected = sorted(history_label(row.get("source_namespace", "task_result")) + text
                      for row in entries for text in row["expected_texts"])
    actual = sorted(text for item in items for text in item["texts"])
    identities = [item["id"] for item in items]
    return (bool(expected) and expected == actual and len(identities) == len(set(identities))
            and all(isinstance(identity, str) and len(identity) <= 32 for identity in identities))


class ObservedRealtime(OpenAIRealtimeConnection):
    """Record provider acknowledgments without changing provider traffic."""
    def __init__(self, *, history, **kwargs):
        super().__init__(**kwargs)
        self.history = history

    async def receive_event(self):
        event = await super().receive_event()
        item = event.get("item") or {}
        if event.get("type") in {"conversation.item.created", "conversation.item.added"}:
            texts = [part["text"] for part in item.get("content", [])
                     if part.get("type") == "output_text"
                     and isinstance(part.get("text"), str) and part["text"].startswith(HISTORY_LABELS)]
            if texts:
                self.history.append({"id": item.get("id"), "texts": texts})
        return event


class SocketPhone(ObservedPhone):
    """Observe actual socket traffic; never manufacture playback acknowledgments."""
    def __init__(self, socket, active_native):
        super().__init__(active_native)
        self.socket = socket
        self.client_events = []

    async def iter_text(self):
        async for raw in self.socket:
            event = json.loads(raw)
            self.client_events.append({
                "seconds": time.monotonic() - self.started, "event": safe_app_event(event),
            })
            yield raw

    async def send_json(self, event):
        await super().send_json(event)
        await self.socket.send(json.dumps(event))
        self.observations[-1]["socket_write_completed"] = True

    async def close(self, **kwargs):
        await super().close(**kwargs)
        await self.socket.close(**kwargs)


def prepare_speech(directory, case):
    directory.mkdir(mode=0o700)
    for name, text in scenario_utterances(case).items():
        original, aiff, wav = (directory / (name + suffix) for suffix in (".txt", ".aiff", ".wav"))
        original.write_text(text)
        subprocess.run(["/usr/bin/say", "-r", "185", "-f", str(original), "-o", str(aiff)], check=True)
        subprocess.run(["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16@24000",
                        "-c", "1", str(aiff), str(wav)], check=True)
        aiff.unlink()


async def microphone(voice, directory, name):
    """Ask the real Swift client to send paced fixture PCM through its input API."""
    pcm = read_pcm(directory / (name + ".wav"))
    segment = name + "-" + uuid4().hex
    for offset in range(0, len(pcm), 4800):
        chunk = pcm[offset:offset + 4800]
        await voice._app_sender.send_text(encode_event(
            "fixture.input.delta", segment_id=segment, audio_base64=base64.b64encode(chunk).decode()))
        await asyncio.sleep(len(chunk) / 48000)
    await voice._app_sender.send_text(encode_event("fixture.input.commit", segment_id=segment))
    return time.monotonic()


async def snapshot(tasks, inbox, owner, phone):
    current = await tasks.list(owner, limit=20)
    async with inbox.pool.acquire() as connection:
        identities = await connection.fetch(
            "select id from jarvis_attention_intents where owner_id=$1 order by created_at,id", owner)
    entries = []
    for identity in identities:
        detail = await inbox.detail(owner, identity["id"])
        expert = detail.get("expert_result") or {}
        notification = detail.get("notification") or {}
        attempts = detail["attempts"]
        attempt_id = attempts[-1]["id"] if attempts else None
        response_id = "tts_" + attempt_id.hex if attempt_id is not None else None
        # Completion commits the intent, attempt and chunk receipts together.
        # Read the selected attempt's receipts AFTER its summary, never combine
        # pre-completion chunks with a newer "played" summary. Nonterminal
        # summaries still fail the existing oracle and are polled again.
        chunks = []
        if attempt_id is not None:
            async with inbox.pool.acquire() as connection:
                chunks = [dict(row) for row in await connection.fetch(
                    """select attempt_id,chunk_index,forwarded_samples,played_samples,first_bytes_ms,
                        generated_at is not null as generated,playback_completed_at is not null as completed
                        from jarvis_attention_speech_chunks where owner_id=$1 and attempt_id=$2
                        order by chunk_index""", owner, attempt_id)]
        entries.append({
            "intent_id": str(identity["id"]), "state": detail["state"],
            "attempt_count": len(attempts), "attempts": attempts, "expert_result": expert,
            "source_namespace": detail.get("source_namespace", "task_result"),
            "source_available": detail.get("source_available", True),
            "task_id": str(detail.get("task_id")), "task_revision": detail.get("task_revision"),
            "notification": notification,
            "expected_texts": list(speech_chunks(notification["message"])) if notification else
                [item["text"] for item in expert.get("speech", [])],
            "actual_texts": [event.get("text") for event in phone.events
                             if event.get("type") == "transcript.done" and event.get("response_id") == response_id],
            "chunks": [row for row in chunks if row["attempt_id"] == attempt_id],
        })
    return current, entries


async def exercise_ordinary_delivery(*, voice, phone, tasks, inbox, fixture, args,
                                     speech_dir, jobs, session_job, capture, active_native, expected, result):
    await microphone(voice, speech_dir, args.case)
    await wait_until(lambda: bool(completed_audio_ids(phone.events))
                     and not voice._native_playback and not voice._normal_turn_pending,
                     running=jobs, timeout=90)
    boundary = len(phone.observations)
    result["native_tasks_before_ordinary"] = active_native()
    input_end = await microphone(voice, speech_dir, "ordinary")
    await wait_until(lambda: bool(completed_audio_ids(
        [row["event"] for row in phone.observations[boundary:]
         if not str(row["event"].get("response_id", "")).startswith("tts_")])),
        running=jobs, timeout=90)
    ordinary = [row for row in phone.observations[boundary:]
                if not str(row["event"].get("response_id", "")).startswith("tts_")]
    output = [row for row in ordinary if row["event"]["type"] == "audio.output.delta"]
    result["ordinary_texts"] = [row["event"].get("text") for row in ordinary
                                if row["event"]["type"] == "transcript.done"]
    result["ordinary_audio_overlap_tasks"] = max(
        (len(row["native_running_tasks"]) for row in output), default=0)
    result["ordinary_input_end_to_forwarded_audio_ms"] = (
        (output[0]["seconds"] - (input_end - phone.started)) * 1000 if output else None)
    async with asyncio.timeout(args.delivery_timeout):
        while True:
            current, entries = await snapshot(tasks, inbox, fixture.owner, phone)
            if len(entries) == expected and all(row["state"] == "played" for row in entries):
                break
            if current and all(task.status.value in {"failed", "cancelled", "paused", "outcome_unknown"}
                               for task in current):
                break
            if session_job.done():
                raise RuntimeError("Voice session stopped before delivery")
            await asyncio.sleep(1)
    result["tasks"] = [{"id": str(task.id), "state": task.status.value,
                        "revision": task.revision, "model": task.model} for task in current]
    result["deliveries"] = entries
    result["checks"] = delivery_checks(entries, expected=expected) | {
        "provider_accepted_expert_history": history_matches(entries, result["native_history_items"]),
        "tasks_succeeded": len(current) == expected
            and all(task.status.value == "succeeded" for task in current),
        "foreground_available_during_all_tasks": result["ordinary_audio_overlap_tasks"] >= expected,
        "capture_flushed": await capture.flush(),
        "no_application_errors": not any(e["type"] == "error" for e in phone.events),
    }


async def drain_seed_sources(fixture, worker, report, bindings):
    # Spoken-record scenarios begin with no seeded originals. An empty queue is
    # not an ingestion failure or something to wait on.
    if bindings:
        await drain(fixture, worker, report, label="delegated-memory-seed")


async def run(args):
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    speech_dir = args.output / "speech"
    prepare_speech(speech_dir, args.case)
    expected = 2 if args.case == "multitask" else 1
    memory_case = memory_case_for(args.case)
    settings = JarvisSettings()
    result = {
        "format": "jarvis.native_expert_delivery.v1", "ok": False,
        "case": args.case, "expected_tasks": expected, "native_foreground": True,
        "native_frontier_reasoning": False, "native_tts": False, "synthetic_microphone": True,
        "physical_glasses": False, "human_hearing": "not_established",
        "semantic_review": "pending", "frontier_model": args.frontier_model,
        "foreground_model": settings.OPENAI_REALTIME_MODEL, "tools": [], "worker_events": [],
        "task_admissions": [],
        "native_history_items": [], "native_memory": memory_case is not None, "worker_tools": [],
    }
    started = time.monotonic()
    conversation = FixtureConversation()
    native = None

    def active_native():
        return sorted(str(run.context.task.id) for run in tuple(native._runs.values())
                      if run.process is not None and run.process.returncode is None) if native else []

    try:
        async with AsyncExitStack() as resources:
            fixture = await resources.enter_async_context(source_fixture(
                settings.SUPABASE_DB_URL, args.output, test_name="native-expert-delivery"))
            tasks = ObservedAdmissionRepository(fixture.pool, TaskLimits(), observations=result["task_admissions"])
            sources = ContextRepository(tasks)
            messages = SessionRepository(fixture.pool)
            catalog = TaskWorkerRegistry(fixture.pool)
            service = TaskService(tasks, {"frontier": ConfiguredTaskProvider(args.frontier_model)}, registry=catalog)
            inbox = AttentionRepository(tasks)
            attention = AttentionService(inbox)
            service.attention_service = attention
            knowledge, bindings = None, {}
            configured = settings.model_copy(update={"JARVIS_OWNER_ID": fixture.owner})
            if memory_case is not None:
                knowledge = await KnowledgeRuntime.create(
                    pool=fixture.pool, owner_id=fixture.owner, tasks=tasks,
                    configuration=KnowledgeConfiguration(
                        base_url=os.environ["JARVIS_S04_MEMORY_URL"],
                        api_key=os.environ["JARVIS_S04_MEMORY_TOKEN"], area_id=fixture.area))
                resources.push_async_callback(knowledge.close)
                sources = knowledge.context_repository
                for original in memory_case["sources"]:
                    bindings[original["id"]] = await fixture.publish(
                        original["text"], namespace="fixture:delegated-memory", source_key=original["id"])
                result["memory"] = {"before": await source_snapshot(fixture, bindings),
                    "case_id": memory_case["id"], "utterance": memory_case["utterance"],
                    "semantic_review": "pending_independent_review",
                    "source_bindings": {key: value.original_evidence() for key, value in bindings.items()}}
                await drain_seed_sources(fixture, knowledge.worker, result, bindings)
                knowledge.start()
            services = SimpleNamespace(settings=configured, background_task_service=service,
                knowledge_runtime=knowledge, knowledge_issue=None, memory_authority=MemoryAuthority.HINDSIGHT,
                tool_modules=(BackgroundTaskToolModule(service),),
                assistant_context_providers=foreground_context_providers(SessionTaskContextService(sources)))
            if knowledge is not None:
                services.assistant_context_providers += (SourceContextProvider(
                    knowledge.current_context, fixture.area, selections=knowledge.area_selections),)
                configure_selected_memory(services)
            manager = RealtimeSessionManager()
            session = await messages.create_session(fixture.owner, "Synthetic integrated expert delivery")
            manager.attention_notifications_enabled, manager.attention_repository = True, inbox
            completed, stop = asyncio.Event(), asyncio.Event()
            connection_gate = asyncio.Lock()
            driver = None
            configuration = WorkerRoleConfiguration(
                python=args.hermes_python, work_root=args.output / "native-state",
                host_id="integrated-voice-" + uuid4().hex)
            class ObservedHermes(HermesTaskProvider):
                async def execute(self, context):
                    row = {"task_id": str(context.task.id), "owner_id": context.task.owner_id,
                           "task_revision": context.claim.task_revision,
                           "started_seconds": time.monotonic() - started}
                    result["worker_events"].append(row)
                    try:
                        outcome = await super().execute(context)
                        row.update(status=outcome.status.value, reason=outcome.reason, payload=outcome.payload)
                        return outcome
                    except BaseException as error:
                        row["error"] = exception_details(error)
                        raise
                    finally:
                        row["ended_seconds"] = time.monotonic() - started

            class ObservedWorkerContext(WorkerContextFactory):
                async def __call__(self, context):
                    prepared = await super().__call__(context)
                    invoke = prepared.tools.invoke
                    async def observe(name, arguments, *, provider_call_id):
                        row = {"task_id": str(context.task.id), "name": name, "arguments": arguments,
                               "provider_call_id": provider_call_id,
                               "started_seconds": time.monotonic() - started}
                        result["worker_tools"].append(row)
                        try:
                            response = await invoke(name, arguments, provider_call_id=provider_call_id)
                            row["result"] = response
                            return response
                        except BaseException as error:
                            row["error"] = exception_details(error)
                            raise
                        finally:
                            row["ended_seconds"] = time.monotonic() - started
                    prepared.tools.invoke = observe
                    return prepared

            native = ObservedHermes(FrontierConfiguration(
                model=args.frontier_model, reasoning_effort="medium", api_key=settings.OPENAI_API_KEY),
                configuration.native(), prepare=ObservedWorkerContext(fixture.pool, knowledge=knowledge))
            worker = TaskWorkerRole(tasks, {"frontier": native},
                owner_id=fixture.owner, configuration=configuration)
            worker_job = asyncio.create_task(worker.run(stop))
            async def drain_worker():
                stop.set()
                await asyncio.wait_for(worker_job, 25)
            resources.push_async_callback(drain_worker)
            async with asyncio.timeout(30):
                while not await catalog.ready(fixture.owner, "frontier", args.frontier_model):
                    if worker_job.done():
                        await worker_job
                        raise RuntimeError("Worker stopped before readiness")
                    await asyncio.sleep(.1)


            client = await resources.enter_async_context(
                AsyncOpenAI(api_key=settings.OPENAI_API_KEY, max_retries=0))
            attention.start_voice(manager=manager, owner=fixture.owner,
                speech=OpenAISpeechProvider(client, SpeechConfiguration()),
                config=AttentionVoiceConfiguration(api_key=settings.OPENAI_API_KEY))
            resources.push_async_callback(attention.close)
            resources.push_async_callback(manager.close_all)
            jobs = (worker_job, attention._task)

            async def stop_driver():
                if driver is not None:
                    if not driver.done():
                        driver.cancel()
                    with suppress(asyncio.CancelledError):
                        await driver
            resources.push_async_callback(stop_driver)

            async def exercise():
                try:
                    row = await conversation.ready()
                    if memory_case is not None:
                        exercise_memory = exercise_exact_history if args.case == "exact_history" else exercise_memory_delivery
                        extra = {"services": services, "settings": configured, "messages": messages,
                                 "knowledge": knowledge} if args.case == "exact_history" else {}
                        await exercise_memory(conversation=conversation, tasks=tasks, inbox=inbox,
                            fixture=fixture, session_id=session.id, bindings=bindings, case=memory_case, speech_dir=speech_dir,
                            jobs=jobs, timeout=args.delivery_timeout, result=result, **extra)
                        row = await conversation.ready()
                        result["checks"].update(
                            capture_flushed=all(result.get("capture_flushed", [])) and await row.capture.flush(),
                            no_application_errors=not any(event["type"] == "error" for event in conversation.events),
                            same_conversation_across_connections=all(
                                item.voice.context.jarvis_session_id == session.id for item in conversation.rows),
                            reset_handshakes_acknowledged=len(conversation.resets) == len(conversation.rows) - 1)
                        result["canonical_history"] = [
                            record.as_context() for record in
                            (await sources.voice_history(fixture.owner, session.id))["records"]]
                    else:
                        await exercise_ordinary_delivery(voice=row.voice, phone=row.phone, tasks=tasks,
                            inbox=inbox, fixture=fixture, args=args, speech_dir=speech_dir,
                            jobs=(*jobs, row.job), session_job=row.job, capture=row.capture,
                            active_native=active_native, expected=expected, result=result)
                    result["native_frontier_reasoning"] = bool(list(
                        authored_worker_results(result["worker_events"], model=args.frontier_model)))
                    result["native_tts"] = bool(result.get("deliveries")) and all(
                        entry["chunks"] and all(chunk["generated"] for chunk in entry["chunks"])
                        for entry in result["deliveries"])
                    result["connections"] = len(conversation.rows)
                    result["ok"] = all(result["checks"].values())
                    result["state"] = "played" if result["ok"] else "incomplete"
                    payload = {key: result[key] for key in ("ok", "state", "case", "checks",
                        "native_foreground", "native_frontier_reasoning", "native_tts",
                        "native_memory", "expected_tasks", "connections")}
                    if "ordinary_audio_overlap_tasks" in result:
                        payload["ordinary_audio_overlap_tasks"] = result["ordinary_audio_overlap_tasks"]
                    await row.voice._app_sender.send_text(encode_event("fixture.result", **payload))
                except Exception as error:
                    result.update(ok=False, error_type=type(error).__name__, error_code=getattr(error, "code", None),
                                  error_details=exception_details(error))
                    if memory_case is not None:
                        result["memory"]["failure_snapshot"] = await memory_failure_snapshot(fixture, bindings)
                    try:
                        current, entries = await snapshot(tasks, inbox, fixture.owner, conversation)
                        result["deliveries"] = entries
                        result["tasks"] = [{"id": str(task.id), "state": task.status.value} for task in current]
                    except Exception as snapshot_error:
                        result["delivery_snapshot_error"] = exception_details(snapshot_error)
                    # Report while the socket and its writer are still alive.
                    with suppress(Exception):
                        row = await conversation.ready()
                        await row.voice._app_sender.send_text(encode_event("fixture.result", ok=False,
                            error_type=type(error).__name__, error_code=getattr(error, "code", None)))
                finally:
                    completed.set()

            async def handle(socket):
                nonlocal driver
                if socket.request.headers.get("Authorization") != "Bearer fixture-token":
                    await socket.close(code=1008)
                    return
                async with connection_gate:
                    if completed.is_set() or len(conversation.rows) >= (4 if memory_case else 1):
                        await socket.close(code=1008)
                        return
                    phone = SocketPhone(socket, active_native)
                    phone.started = started
                    audio_id = "integrated-audio-" + uuid4().hex
                    capture = SessionContextCapture(sources, owner_id=fixture.owner,
                        session_id=session.id, provider="openai", audio_session_id=audio_id)
                    voice = RealtimeAudioSession(websocket=phone,
                        context=RealtimeSessionContext(audio_session_id=audio_id, owner_id=fixture.owner,
                            jarvis_session_id=session.id, client_kind="phone_audio", session=session),
                        settings=configured, repository=messages, tool_modules=services.tool_modules,
                        context_providers=services.assistant_context_providers,
                        context_capture=capture, tool_execution_repository=ToolExecutionRepository(fixture.pool))
                    voice.openai = ObservedRealtime(settings=settings, owner_id=fixture.owner,
                                                    history=result["native_history_items"])
                    voice.tool_executor = ObservedExecutor(voice.tool_executor, result["tools"], started)
                    row = SimpleNamespace(phone=phone, voice=voice, capture=capture, job=None)
                    finished = None
                    registered = False
                    try:
                        registered = await manager.register(audio_session_id=audio_id, owner_id=fixture.owner,
                            jarvis_session_id=session.id, session=voice)
                        if not registered:
                            raise RuntimeError("Fixture session registration failed")
                        row.job = asyncio.create_task(voice.run())
                        conversation.rows.append(row)
                        await wait_until(lambda: any(event["type"] == "session.created" for event in phone.events),
                                         running=(*jobs, row.job), timeout=30)
                        conversation.current = row
                        if driver is None:
                            driver = asyncio.create_task(exercise())
                        finished = asyncio.create_task(completed.wait())
                        await asyncio.wait((row.job, finished), return_when=asyncio.FIRST_COMPLETED)
                        if row.job.done() and not completed.is_set():
                            await row.job
                            reset = completed_source_reset(phone, session_id=session.id, audio_session_id=audio_id)
                            if memory_case is None or reset is None:
                                raise TaskError("native_voice_connection_stopped", "Voice stopped without a completed source reset.")
                            conversation.resets.append(reset)
                    except Exception as error:
                        conversation.failure = error
                        result.update(ok=False, error_type=type(error).__name__, error_code=getattr(error, "code", None))
                        with suppress(Exception):
                            await socket.send(encode_event("fixture.result", ok=False,
                                error_type=type(error).__name__, error_code=getattr(error, "code", None)))
                        if driver is None:
                            completed.set()
                    finally:
                        if conversation.current is row:
                            conversation.current = None
                        if finished is not None:
                            finished.cancel()
                            with suppress(asyncio.CancelledError):
                                await finished
                        await voice.close("integrated_fixture_connection_finished")
                        if row.job is not None:
                            with suppress(Exception):
                                await asyncio.wait_for(row.job, 20)
                        result.setdefault("capture_flushed", []).append(await capture.flush())
                        await capture.close()
                        if registered:
                            await manager.unregister(audio_session_id=audio_id, owner_id=fixture.owner,
                                                     jarvis_session_id=session.id)

            async with serve(handle, "127.0.0.1", 0, max_size=65536) as server:
                print(json.dumps({"ready": True,
                    "url": "ws://127.0.0.1:" + str(server.sockets[0].getsockname()[1])
                        + "?session_id=" + str(session.id)}), flush=True)
                await asyncio.wait_for(completed.wait(), args.delivery_timeout * 2 + 480)
                if driver is not None:
                    await driver
        result["fixture_cleanup"] = "completed"
    except Exception as error:
        result.update(ok=False, error_type=type(error).__name__, error_code=getattr(error, "code", None))
    finally:
        result["native_processes_remaining"] = active_native()
        if result["native_processes_remaining"]:
            result["ok"] = False
        result["native_frontier_reasoning"] = bool(list(
                        authored_worker_results(result["worker_events"], model=args.frontier_model)))
        result["native_tts"] = any(str(event.get("response_id", "")).startswith("tts_")
            and event["type"] == "audio.output.delta" for event in conversation.events)
        result["connections"] = len(conversation.rows)
        result["context_resets"] = conversation.resets
        result["elapsed_ms"] = (time.monotonic() - started) * 1000
        result["observations"] = [event for row in conversation.rows for event in row.phone.observations]
        result["client_events"] = [event for row in conversation.rows for event in row.phone.client_events]
        result["audio_files"] = []
        for connection_index, row in enumerate(conversation.rows, 1):
            for index, (item, pcm) in enumerate(row.phone.audio.items(), 1):
                filename = f"connection-{connection_index}-forwarded-{index}.wav"
                write_wav(args.output / filename, bytes(pcm))
                result["audio_files"].append({"item_id": item, "file": filename,
                    "origin": "expert_tts" if item.startswith("tts_") else "foreground"})
        (args.output / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({key: result.get(key) for key in
        ("ok", "error_type", "checks", "ordinary_texts", "ordinary_audio_overlap_tasks",
         "fixture_cleanup", "native_processes_remaining")}), flush=True)
    return result["ok"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hermes-python", type=Path, required=True)
    parser.add_argument("--frontier-model", default=DEFAULT_FRONTIER_MODEL)
    parser.add_argument("--case", choices=("background", "multitask", *MEMORY_CASES), default="multitask")
    parser.add_argument("--delivery-timeout", type=int, default=360)
    parser.add_argument("--allow-native-models", action="store_true", required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    raise SystemExit(0 if asyncio.run(run(args)) else 1)
