"""Actual isolated memory -> normal Live State -> controlled voice writer.

The native memory service must already exist. Only the voice wire/client media
receipts are controlled. No selected memories, canonical bindings or bridge
outputs are injected. Calendar labels are metadata, not scheduler time travel.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceError
from app.knowledge.runtime import KnowledgeRuntime
from app.live_state.canonical import context_dependency, memory_dependency
from app.live_state.projection import select_voice_context
from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.live_state_bridge import install_responsive_voice
from app.realtime.models import RealtimeSessionContext
from app.runtime.events import DomainEventBus
from app.runtime.feature_config import MemoryAuthority
from app.settings import JarvisSettings
from app.tasks.models import TaskError
from app.voice_routes.contracts import CommandReceipt
from evals.responsive_voice.memory_lifecycle_qualification import (
    Bounds, INDEPENDENT, ORIGINAL, REPLACEMENT, LifecycleCase, QualificationFailure,
    configured_inputs, failure, require, run_qualification, save,
)
from evals.responsive_voice.memory_live_state_diagnostics import CanonicalRefreshDiagnostics


STAGES = ("canonical_ingestion", "fresh_enrichment_and_nonblocking_control",
          "correction_revokes_old_context", "fresh_corrected_context",
          "deletion_revokes_context", "fresh_no_resurrection_positive_control")
QUERY = "What do you remember about my spare workshop key and my bicycle's frame serial?"
BOUNDS = Bounds(drain_seconds=90, max_steps=180, lifecycle_seconds=240, cleanup_seconds=180)
CLIENT_CAPS = {
    "supported_versions": [1], "supported_transports": ["native_voice"],
    "supported_media_topologies": ["phone_to_provider"], "input_codecs": ["opus"],
    "output_codecs": ["opus"], "local_output_fencing": True, "client_playback_boundaries": [],
}


def now():
    return datetime.now(UTC)


class CallBudget:
    def __init__(self):
        self.recalls = 0
        self.main_http = self.cleanup_http = 0
        self.cleaning = False

    def recall(self):
        require(self.recalls < 8, "actual_recall_budget_exhausted")
        self.recalls += 1

    def http(self):
        field, limit = ("cleanup_http", 160) if self.cleaning else ("main_http", 400)
        count = getattr(self, field)
        require(count < limit, "native_http_budget_exhausted")
        setattr(self, field, count + 1)

    def install(self, client):
        original = client._call

        async def counted(*args, **kwargs):
            self.http()
            return await original(*args, **kwargs)
        client._call = counted

    def report(self):
        return {"actual_recall_queries": self.recalls, "runtime_main_http_attempts": self.main_http,
                "runtime_cleanup_http_attempts": self.cleanup_http,
                "outside_runtime_inventory_calls": "One preflight; one independent bank audit if cleanup reaches it.",
                "underlying_model_calls": None, "model_tokens": None,
                "usage_limitation": "HTTP requests are not provider-internal LLM/embedding calls."}


class ObservedRetrieval:
    """Hold a REAL result, never substitute its candidates or original evidence."""
    def __init__(self, original, budget, trace, *, hold=False):
        self.original, self.budget, self.trace = original, budget, trace
        self.hold = hold
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        if not hold:
            self.release.set()

    async def recall(self, *args, **kwargs):
        self.budget.recall()
        started = time.monotonic()
        result = await self.original.recall(*args, **kwargs)
        record = {"query_number": self.budget.recalls, "status": result.get("status"),
                  "actual_retrieval_ms": round((time.monotonic() - started) * 1000, 3),
                  "source_versions": [{"id": item.get("source_id"), "revision": item.get("revision"),
                      "source_time": item.get("source_time"), "trust": item.get("trust")}
                      for item in result.get("sources", ())]}
        self.trace.append(record)
        if self.hold:
            self.hold = False
            self.entered.set()
            held = time.monotonic()
            try:
                await self.release.wait()
            finally:
                record["test_barrier_ms"] = round((time.monotonic() - held) * 1000, 3)
        return result


class ControlledClient:
    """Synthetic selected-client and fence receipts; never playback evidence."""
    def __init__(self):
        self.session = None
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(payload)
        if payload["type"] == "voice_route.selected":
            await self.session._handle_client_event({"type": "voice_route.ready", "protocol_version": 1,
                "audio_session_id": payload["audio_session_id"], "route": payload["route"],
                "client_event_id": str(uuid4())})
        elif payload["type"] in {"voice_route.interrupt", "voice_route.close"}:
            command = payload["payload"]
            receipt = CommandReceipt(route=command["route"], command_id=command["command_id"],
                operation="close" if payload["type"] == "voice_route.close" else "interrupt_speech",
                observed_at=now(), status="completed", output_fenced=True)
            await self.session._handle_client_event({"type": "voice_route.event", "protocol_version": 1,
                "audio_session_id": payload["audio_session_id"], "payload": receipt.model_dump(mode="json"),
                "client_event_id": str(uuid4())})
        elif payload["type"] == "session.context_invalidated":
            await self.session._handle_client_event({"type": "session.context_cleared", "protocol_version": 1,
                "audio_session_id": payload["audio_session_id"], "reset_id": payload["context_reset"]["id"]})

    async def send_text(self, raw):
        await self.send_json(json.loads(raw))

    async def close(self, *args, **kwargs):
        pass


class ControlledProviderWire:
    def __init__(self):
        self.events, self.responses = [], []
        self.incoming = asyncio.Queue()

    async def send(self, raw):
        event = json.loads(raw)
        self.events.append(event)
        if event["type"] == "response.create":
            self.responses.append(event)
            self.incoming.put_nowait({"type": "response.created", "response": {
                "id": "controlled-" + str(len(self.responses)),
                "metadata": event["response"]["metadata"]}})
        elif event["type"] == "response.cancel" and event.get("response_id"):
            self.incoming.put_nowait({"type": "response.done", "response": {
                "id": event["response_id"], "status": "cancelled", "output": []}})
        elif event["type"] == "conversation.item.delete":
            self.incoming.put_nowait({"type": "conversation.item.deleted", "item_id": event["item_id"]})

    async def close(self):
        pass

    def context_text(self):
        return "\n".join(part.get("text", "") for event in self.events
            if event["type"] == "conversation.item.create"
            for part in event.get("item", {}).get("content", ()))


async def until(predicate, *, timeout=12, receiver=None, code="integration_wait_timeout"):
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                if receiver is not None and receiver.done():
                    await receiver
                    raise QualificationFailure("controlled_provider_receiver_stopped")
                await asyncio.sleep(.01)
    except TimeoutError:
        raise QualificationFailure(code) from None


def verify_selected(snapshot, expected, request, *, absent=()):
    entries = [item for item in snapshot.entries if item.kind == "selected_memory" and item.validity == "active"]
    absent = set(absent)
    require(not any(dependency.source_namespace == "knowledge_source" and dependency.source_id in absent
                    for item in entries for dependency in item.sources), "withdrawn_memory_in_live_state")
    selected = []
    for source in expected:
        dependency = memory_dependency(source)
        matches = [item for item in entries if dependency in item.sources]
        require(len(matches) == 1, "expected_source_not_selected_in_live_state")
        entry = matches[0]
        require(entry.observed_at == source.source_time, "live_state_source_timestamp_changed")
        require(entry.evidence_kind == "canonical_record", "live_state_source_provenance_changed")
        require(context_dependency(request) in entry.sources, "selected_memory_not_bound_to_current_intent")
        require(entry.guard_binding is not None, "selected_memory_has_no_canonical_guard")
        require(json.dumps(source.content, ensure_ascii=True, sort_keys=True) in entry.summary,
                "selected_memory_original_changed")
        selected.append(entry)
    return selected


class MemoryLiveStateCase(LifecycleCase):
    def __init__(self, *args):
        super().__init__(*args)
        self.voice = None
        self.budget = CallBudget()
        self.refresh_diagnostics = CanonicalRefreshDiagnostics()
        self.report.update(format="jarvis.responsive_voice.memory_live_state.v1",
            stages={stage: {"status": "unreached"} for stage in STAGES},
            actual_retrievals=[], live_state_checks=[], voice_fixture_runs=[],
            voice_evidence="Controlled provider writer/client fences; NOT frontend, speech or provider-model acceptance.")
        self.report["limitations"].append(
            "Normal install_responsive_voice composition; media/SDP readiness and voice receipts are controlled. "
            "No perception-frame, notification or proactive acceptance in this memory-only sidecar.")

    async def fresh(self, day):
        await super().fresh(day)
        self.budget.install(self.runtime.client)

    async def close_runtime(self):
        await self.close_voice()
        await super().close_runtime()

    async def close_voice(self):
        voice, self.voice = self.voice, None
        if voice is None:
            return
        voice.retrieval.release.set()
        before_close = None
        if voice.attachment is not None:
            attachment = voice.attachment
            before_close = {
                "subscription_needs_resync": bool(attachment._subscription
                    and attachment._subscription._needs_resync),
                "live_state_needs_resync": attachment.live.service.needs_resync,
                "live_state_readiness": attachment.live.service.snapshot().readiness,
                "live_state_last_rejection": attachment.live.service.last_rejection,
                "runtime_failure_type": type(attachment.live.failure).__name__ if attachment.live.failure else None,
                "historical_worker_last_error": attachment.live.enrichment.worker.last_error,
                "historical_worker_failures": attachment.live.enrichment.worker.failure_count,
            }
        try:
            if voice.attachment is not None:
                await voice.attachment.close("memory_live_state_fixture_complete")
            else:
                await voice.session.close("memory_live_state_fixture_complete")
        finally:
            if voice.receiver is not None:
                voice.receiver.cancel()
                await asyncio.gather(voice.receiver, return_exceptions=True)
            self.report["voice_fixture_runs"].append({
                "session_id": str(voice.session.context.jarvis_session_id),
                "provider_writes": len(voice.wire.events), "response_creates": len(voice.wire.responses),
                "attachment_failure": voice.attachment.failure if voice.attachment else None,
                "live_worker_failure": voice.attachment.live.worker.last_error if voice.attachment else None,
                "historical_status": voice.attachment.live.enrichment.status if voice.attachment else None,
                "historical_error": voice.attachment.live.enrichment.last_error if voice.attachment else None,
                "context_revoked": voice.session._context_revoked,
                "before_close": before_close,
            })
            self.report["call_usage"] = self.budget.report()
            diagnostic_path = self.path.parent / "canonical-refresh-diagnostics.json"
            save(diagnostic_path, self.refresh_diagnostics.document())
            self.report["canonical_refresh_diagnostics"] = str(diagnostic_path)
            self.checkpoint()

    async def attach_voice(self, *, hold=False):
        require(self.voice is None, "old_voice_runtime_not_retired")
        retrieval = ObservedRetrieval(self.runtime.retrieval, self.budget,
            self.report["actual_retrievals"], hold=hold)
        self.runtime.retrieval = retrieval
        services = SimpleNamespace(knowledge_runtime=self.runtime, memory_authority=MemoryAuthority.HINDSIGHT,
            domain_event_bus=DomainEventBus(), responsibility_issue="responsibilities_disabled")
        client, wire = ControlledClient(), ControlledProviderWire()
        session = DirectRealtimeAudioSession(offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            websocket=client, context=RealtimeSessionContext(audio_session_id="memory-live-" + uuid4().hex,
                owner_id=self.owner, jarvis_session_id=self.session, client_kind="rayban_audio",
                session=SimpleNamespace()), settings=JarvisSettings(_env_file=None),
            repository=SimpleNamespace(create_message=AsyncMock(return_value=SimpleNamespace(id=uuid4())),
                get_recent_messages=AsyncMock(return_value=[])), tool_modules=(), context_providers=())
        voice = self.voice = SimpleNamespace(session=session, wire=wire, client=client,
            attachment=None, receiver=None, retrieval=retrieval)
        client.session = session
        session._source_context.pool = self.pool
        session.openai._websocket = wire

        async def receive():
            session._provider_receive_task = asyncio.current_task()
            while True:
                event = await wire.incoming.get()
                try:
                    await session._handle_openai_event(event)
                finally:
                    wire.incoming.task_done()
        voice.receiver = asyncio.create_task(receive())
        voice.attachment = await install_responsive_voice(session, services,
            SimpleNamespace(RESPONSIVE_VOICE_ENABLED=True), CLIENT_CAPS, clock=now)
        self.refresh_diagnostics.observe(voice.attachment.live.enrichment)
        snapshot = voice.attachment.live.service.snapshot()
        require(not any(item.kind == "selected_memory" for item in snapshot.entries), "fresh_runtime_carried_old_memory_ram")
        self.report.setdefault("fresh_live_state_epochs", []).append(str(snapshot.cursor.epoch))
        require(len(set(self.report["fresh_live_state_epochs"])) == len(self.report["fresh_live_state_epochs"]),
                "live_state_epoch_reused")
        session._realtime_context_ready.set()
        await until(lambda: voice.attachment.route.ready and voice.attachment.live.service.snapshot().readiness == "ready",
                    receiver=voice.receiver, code="voice_or_live_state_not_ready")
        return voice

    async def emit(self, voice, kind, **payload):
        voice.wire.incoming.put_nowait({"type": kind, **payload})
        await asyncio.wait_for(voice.wire.incoming.join(), timeout=2)
        if voice.receiver.done():
            await voice.receiver

    async def ask(self, voice):
        item = "controlled-input-" + uuid4().hex
        await self.emit(voice, "input_audio_buffer.speech_started", item_id=item)
        await self.emit(voice, "input_audio_buffer.committed", item_id=item)

    async def inspect_selection(self, voice, expected, request, *, label, absent=(), forbidden=()):
        live = voice.attachment.live

        def delivered():
            snapshot = live.service.snapshot()
            dependencies = {dependency for item in snapshot.entries if item.kind == "selected_memory"
                            for dependency in item.sources}
            return (all(memory_dependency(source) in dependencies for source in expected)
                    and all(source.content["text"] in voice.wire.context_text() for source in expected))
        await until(delivered, receiver=voice.receiver, timeout=15, code="native_memory_not_delivered_to_live_context")
        snapshot = live.service.snapshot()
        selected = verify_selected(snapshot, expected, request, absent=absent)
        projection = select_voice_context(snapshot, now=now())
        require(all(item.entry_id in projection.selected_entry_ids for item in selected), "voice_selection_omitted_expected_memory")
        require(not any(text in projection.text or text in voice.wire.context_text() for text in forbidden),
                "obsolete_text_reached_fresh_voice_context")
        for item in selected:
            require(await live.binder.available(item.guard_binding), "selected_binding_not_current")
        update = voice.attachment._last_update
        require(update is not None and update.guard_binding is not None, "voice_bridge_has_no_bound_update")
        require(all(memory_dependency(source) in update.dependencies for source in expected), "voice_bridge_lost_source_versions")
        async with live.guard(update.guard_binding):
            pass
        self.report["live_state_checks"].append({"label": label, "session_id": str(self.session),
            "state_epoch": str(snapshot.cursor.epoch), "state_revision": snapshot.cursor.revision,
            "canonical_intent": context_dependency(request).model_dump(mode="json"),
            "selected_sources": [{"source_id": str(source.source_id), "revision": source.revision,
                "generation": source.generation, "source_time": source.source_time.isoformat(), "trust": source.trust,
                "payload_roots": [str(root) for root in entry.guard_binding.payload_root_ids]}
                for source, entry in zip(expected, selected, strict=True)],
            "voice_context_update_id": str(update.update_id),
            "canonical_update_guard_passed": True, "provider_writer_observed": True,
            "playback_or_consumption_claimed": False})
        self.checkpoint()
        return selected

    async def prove_revoked(self, voice, selected, label):
        require(not await voice.attachment.live.binder.available(selected.guard_binding), "obsolete_binding_still_authorized")
        before = len(voice.wire.responses)
        await self.ask(voice)
        await until(lambda: voice.session._context_revoked or voice.session._closed.is_set()
                    or not voice.attachment.route.ready, receiver=voice.receiver, timeout=8,
                    code="obsolete_remote_context_not_fenced")
        await until(lambda: not voice.session._direct_tasks, receiver=voice.receiver, timeout=5)
        require(len(voice.wire.responses) == before, "response_created_from_revoked_context")
        self.report.setdefault("invalidation", []).append({"label": label,
            "canonical_binding_rejected": True, "no_new_response_create": True,
            "session_or_route_fenced": True})

    async def exercise(self):
        async def ingestion():
            await self.fresh(0)
            event, _, original = await self.capture(ORIGINAL, 0)
            _, _, independent = await self.capture(INDEPENDENT, 0)
            await self.drain()
            return event, original, independent
        original_event, original, independent = await self.stage("canonical_ingestion", ingestion)

        async def initial():
            await self.fresh(1)
            voice = await self.attach_voice(hold=True)
            _, request, _ = await self.capture(QUERY, 1, kind="user_request")
            await asyncio.wait_for(voice.retrieval.entered.wait(), timeout=12)
            try:
                started = time.monotonic()
                await self.ask(voice)
                await until(lambda: bool(voice.wire.responses) and bool(voice.session._active_responses),
                            receiver=voice.receiver, timeout=1.5, code="held_recall_blocked_response_control")
                response_ms = (time.monotonic() - started) * 1000
                await voice.session._handle_client_event({"type": "response.cancel"})
                await until(lambda: any(event["type"] == "response.cancel" for event in voice.wire.events),
                            receiver=voice.receiver, timeout=1.5, code="held_recall_blocked_cancellation")
                require(not voice.retrieval.release.is_set(), "retrieval_barrier_released_before_fast_control")
                self.report["held_retrieval_fast_lane"] = {"response_create_ms": round(response_ms, 3),
                    "turn_and_cancel_ms": round((time.monotonic() - started) * 1000, 3),
                    "actual_result_still_held": True, "controlled_provider_only": True}
            finally:
                voice.retrieval.release.set()
            selected = await self.inspect_selection(voice, (original, independent), request, label="original")
            return voice, selected[0]
        original_voice, original_selected = await self.stage("fresh_enrichment_and_nonblocking_control", initial)

        async def correction():
            _, request, command = await self.capture(
                f"Correct source {original.source_id} revision {original.revision}. Replacement: {REPLACEMENT}",
                7, kind="user_request")
            await self.invoke("correct_memory_source", {"source_id": str(original.source_id),
                "expected_revision": original.revision, "replacement": REPLACEMENT}, request)
            corrected = await self.runtime.sources.read(self.owner, self.area, original.source_id)
            require(corrected.revision == 2 and corrected.trust == "derived", "versioned_correction_missing")
            await self.prove_revoked(original_voice, original_selected, "correction")
            await self.close_voice()
            await self.drain()
            return corrected, command
        corrected, correction_command = await self.stage("correction_revokes_old_context", correction)

        async def corrected_context():
            await self.fresh(7)
            voice = await self.attach_voice()
            _, request, _ = await self.capture(QUERY, 7, kind="user_request")
            selected = await self.inspect_selection(voice, (corrected, independent), request,
                label="corrected", forbidden=(ORIGINAL,))
            return voice, selected[0]
        corrected_voice, corrected_selected = await self.stage("fresh_corrected_context", corrected_context)

        async def deletion():
            targets = (corrected, correction_command)
            _, request, _ = await self.capture("Permanently delete the originals and historical copies for "
                + "; ".join(f"source {item.source_id} revision {item.revision}" for item in targets)
                + ". Keep all other facts.", 90, kind="user_request")
            await self.invoke("delete_memory_source_group", {"sources": [
                {"source_id": str(item.source_id), "expected_revision": item.revision} for item in targets]}, request)
            await self.prove_revoked(corrected_voice, corrected_selected, "deletion")
            await self.close_voice()
            await self.drain()
            for source in (original, corrected, correction_command):
                require(await self.runtime.client.document(self.bank, source.document_id) is None,
                        "deleted_native_document_remains")
            await self.unavailable(original, revision=1)
        await self.stage("deletion_revokes_context", deletion)

        async def final_context():
            await self.fresh(90)
            try:
                await self.runtime.context_for(self.owner).append(self.owner, original_event)
            except (SourceError, TaskError):
                self.report["original_receipt_replay"] = "rejected"
            else:
                self.report["original_receipt_replay"] = "returned_without_readable_original"
            await self.unavailable(original, revision=1)
            voice = await self.attach_voice()
            _, request, _ = await self.capture(QUERY, 90, kind="user_request")
            await self.inspect_selection(voice, (independent,), request, label="after_deletion",
                absent=(original.source_id, correction_command.source_id), forbidden=(ORIGINAL, REPLACEMENT))
            require(self.budget.recalls <= 8, "actual_recall_budget_exceeded")
        await self.stage("fresh_no_resurrection_positive_control", final_context)

    async def cleanup_native(self):
        await self.close_voice()
        self.budget.cleaning = True
        if self.runtime is None:
            self.runtime = await KnowledgeRuntime.create(pool=self.pool, owner_id=self.owner,
                                                         configuration=self.configuration)
            self.budget.install(self.runtime.client)
        try:
            return await super().cleanup_native()
        finally:
            self.report["call_usage"] = self.budget.report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", required=True, action="store_true")
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    require(args.report_dir.is_absolute() and not args.report_dir.exists(), "new_absolute_report_directory_required")
    # Freeze budgets in the parent before the runner makes its first native call.
    args.report_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    save(args.report_dir.parent / (args.report_dir.name + "-budget.json"), {
        "format": "jarvis.memory_live_state.budget.v1", "frozen_at": now().isoformat(),
        "main_seconds": 240, "cleanup_seconds": 180, "drain_seconds": 90, "steps_per_drain": 180,
        "actual_recall_query_cap": 8, "runtime_main_http_cap": 400, "runtime_cleanup_http_cap": 160,
        "additional_inventory_requests": 2, "banks": 1,
        "synthetic_fact_assertions": 3, "fact_limit": 4,
        "whole_run_retries": 0, "voice_provider": "controlled_writer_no_network",
        "frontend_speech_proactivity_clock_acceptance": False})
    dsn, resolution = configured_inputs()
    require(resolution.configured, "actual_memory_configuration_unavailable")
    report = asyncio.run(run_qualification(dsn, resolution.configuration, args.report_dir,
        bounds=BOUNDS, case_factory=MemoryLiveStateCase))
    print(json.dumps({"status": report["status"], "accepted": report["accepted"],
        "report": str(args.report_dir / "qualification.json"), "call_usage": report.get("call_usage")}))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
