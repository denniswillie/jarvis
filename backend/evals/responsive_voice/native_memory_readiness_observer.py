"""Opt-in, in-process observation of canonical memory readiness for replay hosts.

No polling task, tool dispatch, provider operation, worker claim or business write
is performed here. A consume uses one bounded repeatable-read, READ ONLY snapshot.
The frontend's connection/ordinal labels remain declarations, not backend evidence
of PCM delivery or playback. The unimplemented frontend trigger must validate its
own live control connection, prepared input occurrence and playback predicate.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import logging
import time
from pathlib import Path
from uuid import UUID

from app.knowledge.sources import VERSION_SQL, SourceRepository
from app.realtime.telemetry import LOGGER, PROCESS_ID
from app.tasks.context_repository import RECORD_SQL, context_record
from app.tasks.models import digest
from evals.responsive_voice.native_memory_readiness import (
    CanonicalMemorySnapshot, ReadinessRequest, _control, evaluate_memory_readiness,
)
from evals.responsive_voice.native_memory_replay_control import EvidenceRef, Scope


@dataclass(frozen=True)
class MemoryObserverConfig:
    invocation_id: UUID
    manifest_sha256: str
    max_consumes: int = 32
    max_trace_records: int = 512
    read_timeout_seconds: float = 3.0
    configuration_path: Path | None = None
    status_path: Path | None = None
    approved_gates: tuple[dict, ...] = ()
    origin_invocation_id: UUID | None = None
    phase_start_seconds: float = 0
    source_anchor_sha256: str | None = None

    def __post_init__(self):
        if (not isinstance(self.invocation_id, UUID)
                or not isinstance(self.manifest_sha256, str) or len(self.manifest_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.manifest_sha256)
                or type(self.max_consumes) is not int or not 1 <= self.max_consumes <= 64
                or type(self.max_trace_records) is not int or not 8 <= self.max_trace_records <= 2048
                or type(self.read_timeout_seconds) not in (int, float)
                or not 0 < self.read_timeout_seconds <= 3):
            raise ValueError("Invalid bounded memory observer configuration")
        if self.configuration_path is not None or self.status_path is not None:
            from evals.responsive_voice.native_memory_readiness_control import validate_host_config
            validate_host_config(self)
        elif self.approved_gates or self.origin_invocation_id is not None or self.source_anchor_sha256 is not None:
            raise ValueError("File controls require complete explicit host configuration")


def _identity(value):
    """Native UUIDs differ in casing; opaque provider identities must not."""
    if not isinstance(value, str) or not 0 < len(value) <= 512:
        raise ValueError("Invalid observed identity")
    try:
        return str(UUID(value))
    except ValueError:
        return value


def _object(value):
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, dict):
        raise ValueError("Expected canonical object")
    return value


def _reference(value, label):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    # This is an inline receipt reference, not a claim that an artifact was written.
    return EvidenceRef("inline/" + label + ".json", hashlib.sha256(encoded).hexdigest(),
                       "scoped_in_process_observation")


class _Pending(Exception):
    def __init__(self, reason):
        self.reason = reason


class MemoryReadinessObserver(logging.Handler):
    """Attach only inside an explicitly configured, isolated replay host.

    Logs are consumed directly from the existing RealtimeTrace logger, never from
    caller JSON or archived files. Truncation, duplicate conflicting bindings,
    missing telemetry and closed sessions fail closed. No timestamp joins occur.
    """
    EVENTS = frozenset({"input_commit_admitted", "response_source_latency",
        "response_write_gate", "provider_response_started", "session_closed_forwarded"})

    def __init__(self, host_scope, config, *, clock=lambda: datetime.now(UTC)):
        super().__init__()
        if not isinstance(config, MemoryObserverConfig):
            raise ValueError("Explicit MemoryObserverConfig required")
        self.config, self.clock, self.host_scope = config, clock, host_scope
        self.scope = Scope(host_scope.run_id, host_scope.owner_id, host_scope.database_schema,
                           str(host_scope.area_id), host_scope.memory_bank_id)
        self.pool = self.repository = None
        self.started_at = None
        self.active = False
        self.issue = None
        self.records = []
        self.admissions, self.prepared, self.writes, self.responses = {}, {}, {}, {}
        self.closed_sessions, self.nonces, self.inflight = set(), set(), set()
        self.query_count = 0
        self.bridge = None
        self._shutdown_receipt = None

    def start(self, runtime, *, started_at):
        if self.active or self.pool is not None:
            raise ValueError("Observer cannot restart")
        knowledge = runtime.knowledge_runtime
        if (knowledge.owner_id != self.scope.owner_id
                or str(knowledge.configuration.area_id) != self.scope.area_id
                or knowledge.sources.pool is not runtime.db_pool
                or knowledge.context_repository.pool is not runtime.db_pool):
            raise ValueError("Observer runtime scope mismatch")
        self.pool, self.repository = runtime.db_pool, knowledge.context_repository
        self.started_at, self.active = started_at, True
        LOGGER.addHandler(self)
        try:
            if self.config.configuration_path is not None:
                from evals.responsive_voice.native_memory_readiness_control import MemoryReadinessFileBridge
                self.bridge = MemoryReadinessFileBridge(self)
                self.bridge.start()
        except BaseException:
            self.active = False
            LOGGER.removeHandler(self)
            raise

    def status(self):
        if self._shutdown_receipt is not None:
            return dict(self._shutdown_receipt)
        return {"status": "active" if self.active else "not_started",
                "observer_joined": False, "io_loop_joined": self.bridge is None,
                "failure_code": self.issue, "file_control": self.bridge.status() if self.bridge else None}

    def _unique(self, table, key, value):
        prior = table.get(key)
        if prior is not None and prior != value:
            self.issue = "conflicting_observed_binding"
        else:
            table[key] = value

    def emit(self, record):
        if not self.active or self.issue or record.name != LOGGER.name or record.msg != "realtime_trace %s":
            return
        try:
            if not isinstance(record.args, tuple) or len(record.args) != 1:
                return
            data = record.args[0]
            if not isinstance(data, str) or len(data.encode()) > 8192:
                self.issue = "trace_record_limit"
                return
            value = _object(data)
            if (value.get("schema") != 1 or value.get("process_id") != PROCESS_ID
                    or value.get("provider") != "openai" or value.get("event") not in self.EVENTS):
                return
            if value["event"] == "response_source_latency" and not (
                    value.get("phase") == "prepare" and value.get("status") == "completed"):
                return
            if len(self.records) >= self.config.max_trace_records:
                self.issue = "trace_record_limit"
                return
            session, audio = _identity(value["jarvis_session_id"]), _identity(value["audio_session_id"])
            base = session, audio
            self.records.append(value)
            event = value["event"]
            if event == "session_closed_forwarded":
                self.closed_sessions.add(base)
            elif event == "input_commit_admitted":
                # Automatic commits lack this manual pair and are not candidates.
                if not value.get("segment_id") or not value.get("client_event_id"):
                    return
                key = (*base, _identity(value["segment_id"]), _identity(value["client_event_id"]))
                generation = value.get("turn_generation")
                if type(generation) is not int or not 0 <= generation < 2**63:
                    self.issue = "invalid_admission_generation"
                    return
                request_id = value["request_id"]
                if not isinstance(request_id, str) or not 0 < len(request_id) <= 512:
                    raise ValueError("Invalid request identity")
                self._unique(self.admissions, key, (request_id, generation, value))
            elif event == "response_source_latency":
                if value.get("source_snapshot_id"):
                    self.prepared[(*base, value["request_id"])] = str(UUID(value["source_snapshot_id"]))
            elif event == "response_write_gate" and value.get("status") == "accepted":
                key = (*base, value["request_id"])
                if value.get("boundary") == "payload_at_write" and key in self.prepared:
                    self._unique(self.writes, key, (self.prepared[key], value.get("turn_generation")))
            elif event == "provider_response_started":
                key = (*base, value.get("request_id"))
                if key in self.writes:
                    self._unique(self.responses, (*base, value["response_id"]),
                                 (value["request_id"], *self.writes[key]))
        except Exception:
            # Diagnostic failure never modifies voice admission or writes.
            self.issue = "invalid_observed_trace"

    def _empty(self, request):
        return CanonicalMemorySnapshot(self.scope, request.nonce, self.clock())

    def _failure(self, request, reason, *, status="unavailable"):
        result = evaluate_memory_readiness(request, self._empty(request), now=self.clock())
        return result | {"status": status, "reason": reason}

    async def observe(self, request):
        """One fresh consume; callers do not get a sticky ready bit or an auto-retry.

        Host/operator config binds invocation/manifest. Connection ID and input
        ordinals are echoed declarations for frontend validation, not observations.
        Exact segment/client-event IDs must match a real backend commit admission.
        """
        if not isinstance(request, ReadinessRequest):
            raise ValueError("Typed readiness request required")
        if (not self.active or self.issue or request.scope != self.scope
                or request.invocation_id != str(self.config.invocation_id)
                or request.manifest_sha256 != self.config.manifest_sha256):
            return self._failure(request, self.issue or "observer_scope_unavailable")
        if (request.nonce in self.nonces or len(self.nonces) >= self.config.max_consumes or self.inflight):
            return self._failure(request, "consume_reused_busy_or_budget_exhausted")
        now = self.clock()
        remaining = min((request.expires_at-now).total_seconds(),
                        (self.host_scope.expires_at-now).total_seconds(), self.config.read_timeout_seconds)
        if now < request.issued_at or remaining <= 0:
            return self._failure(request, "control_deadline", status="expired")
        self.nonces.add(request.nonce)
        base = str(request.session_id), _identity(request.audio_session_id)
        if base in self.closed_sessions:
            return self._failure(request, "audio_session_closed")
        admitted = self.admissions.get((*base, _identity(request.segment_id), _identity(request.client_event_id)))
        if admitted is None:
            return self._failure(request, "manual_commit_admission_unavailable")
        response = self.responses.get((*base, request.response_id))
        if response is None or response[2] != admitted[1]:
            return self._failure(request, "admitted_response_snapshot_unavailable")
        task = asyncio.current_task()
        self.inflight.add(task)
        before = self.query_count
        started = time.monotonic()
        try:
            async with asyncio.timeout(remaining):
                snapshot, graph = await self._read(request, admitted, response)
            if not self.active or self.issue or base in self.closed_sessions:
                return self._failure(request, self.issue or "observer_closed_during_read")
            result = evaluate_memory_readiness(request, snapshot, now=self.clock())
            if result['status'] == 'ready':
                message = snapshot.output.content['conversation_message']
                policy = message['realtime_output']
                result['final_output_policy'] = {
                    'response_kind': message['response_kind'],
                    'audio_session_id': message['audio_session_id'],
                    'realtime_output': {key:policy[key] for key in
                        ('item_id','phase','phase_anomaly','delivery_disposition')},
                }
            return result | {"observer": {"queries": self.query_count-before,
                "elapsed_seconds": time.monotonic()-started, "read_only": True,
                "isolation": "repeatable_read", "automatic_polling": False},
                "inline_evidence": {"admission": admitted[2], "continuation": graph},
                "continuation_basis": "admitted_response_source_dependencies_not_tool_trigger_or_model_use"}
        except _Pending as error:
            return self._failure(request, error.reason, status="waiting")
        except TimeoutError:
            return self._failure(request, "canonical_read_timeout")
        except asyncio.CancelledError:
            raise
        except Exception:
            return self._failure(request, "canonical_read_unavailable")
        finally:
            self.inflight.discard(task)

    async def _read(self, request, admitted, response):
        owner, session = self.scope.owner_id, request.session_id
        namespace = "voice:" + digest({"provider": "openai", "audio_session_id": request.audio_session_id})
        response_request, snapshot_id, generation = response
        admission = _control(request) | {"event": "input_commit_admitted", "request_id": admitted[0],
            "turn_generation": admitted[1], "evidence": _reference(admitted[2], "admission")}
        async with self.pool.acquire() as connection, connection.transaction(
                readonly=True, isolation="repeatable_read"):
            async def rows(sql, *args):
                self.query_count += 1
                if self.query_count - initial > 9:
                    raise ValueError("Snapshot query budget")
                return await connection.fetch(sql, *args)
            initial = self.query_count
            bound = await rows("""select current_schema() as schema, b.bank_name from jarvis_sessions s
                join jarvis_memory_banks b on b.owner_id=s.owner_id and b.area_id=$3
                join jarvis_knowledge_areas a on a.owner_id=b.owner_id and a.id=b.area_id
                where s.owner_id=$1 and s.id=$2 and s.status='active' and s.created_at >= $4
                and a.status='active' limit 2""", owner, session, UUID(self.scope.area_id), self.started_at)
            if len(bound) != 1 or dict(bound[0]) != {"schema": self.scope.database_schema,
                                                    "bank_name": self.scope.memory_bank_id}:
                raise _Pending("fresh_session_scope_unavailable")
            async def current(category, identity):
                found = await rows(RECORD_SQL + """ where e.owner_id=$1 and e.session_id=$2
                    and e.source_namespace=$3 and e.source_event_id=$4 order by e.revision desc limit 1""",
                    owner, session, namespace, category + ":" + digest({"identity": identity}))
                return context_record(found[0]) if found else None
            output = await current("output", request.item_id)
            canonical = await current("input", "commit:" + admitted[0])
            result = CanonicalMemorySnapshot(self.scope, request.nonce, self.clock(),
                request=canonical, output=output, admission=admission)
            if output is None or canonical is None:
                return result, None
            message = output.content.get("conversation_message", {}) if output.content else {}
            policy = message.get("realtime_output", {}) if isinstance(message, dict) else {}
            if (not isinstance(message, dict) or not isinstance(policy, dict)
                    or message.get("response_kind") != "normal_user_turn"
                    or message.get("audio_session_id") != request.audio_session_id
                    or policy.get("item_id") != request.item_id or policy.get("phase") != "final_answer"
                    or policy.get("phase_anomaly") is not False
                    or policy.get("delivery_disposition") != "forwarded"):
                raise _Pending("canonical_final_output_policy_unavailable")
            graph_rows = await rows("""select p.id,p.availability,p.content,p.content_digest
                from jarvis_context_payloads p join jarvis_context_payload_dependencies d
                on d.owner_id=p.owner_id and d.source_payload_id=p.id
                where p.owner_id=$1 and p.id=$2 and d.dependent_payload_id=$3 limit 2""",
                owner, UUID(snapshot_id), output.payload_id)
            if len(graph_rows) != 1 or graph_rows[0]["availability"] != "available":
                raise _Pending("exact_output_snapshot_edge_unavailable")
            source_content = _object(graph_rows[0]["content"])
            if any(source_content.get(k) != v for k, v in {
                    "kind": "live_response_dependencies", "request_id": response_request,
                    "session_id": str(session), "audio_session_id": request.audio_session_id}.items()):
                raise _Pending("response_snapshot_binding_mismatch")
            graph = {"authority": "LiveResponseSources.prepare_and_RealtimeAudioSession._register_response",
                "snapshot_payload_id": snapshot_id, "snapshot_content_digest": graph_rows[0]["content_digest"],
                "response_request_id": response_request, "output_payload_id": str(output.payload_id),
                "response_id": request.response_id, "turn_generation": generation}
            if request.canonical_requirement == "finalized_input":
                return result, graph
            tools = await rows(RECORD_SQL + """ where e.owner_id=$1 and e.session_id=$2
                and e.source_namespace=$3 and e.kind='tool_outcome'
                and p.content->>'tool_name'='remember' and p.content->>'input_id'=$4
                and exists(select 1 from jarvis_context_payload_dependencies d where d.owner_id=e.owner_id
                    and d.source_payload_id=e.payload_id and d.dependent_payload_id=$5) limit 2""",
                owner, session, namespace, "commit:" + admitted[0], UUID(snapshot_id))
            if len(tools) != 1:
                raise _Pending("remember_snapshot_edge_missing_or_ambiguous")
            tool = context_record(tools[0])
            content = tool.content or {}
            saved = content.get("result") or {}
            pinned = saved.get("source") or {}
            executions = await rows("""select status,result from jarvis_tool_executions
                where owner_id=$1 and session_id=$2 and provider_call_id=$3
                and tool_name='remember' limit 2""", owner, session, content.get("provider_call_id"))
            if (len(executions) != 1 or executions[0]["status"] != "completed"
                    or _object(executions[0]["result"]).get("source_bound_tool_result") != {
                        "protocol_version": 1, "context_event_id": str(tool.id)}):
                raise _Pending("completed_source_bound_execution_unavailable")
            versions = await rows(VERSION_SQL + " and v.source_id=$3 and v.revision=$4 limit 2",
                owner, UUID(self.scope.area_id), UUID(pinned["source_id"]), pinned["revision"])
            if len(versions) != 1:
                raise _Pending("exact_saved_source_unavailable")
            source = SourceRepository._version(versions[0])
            memory_rows = await rows(RECORD_SQL + " where e.owner_id=$1 and e.id=$2 limit 2",
                                     owner, source.context_event_id)
            if len(memory_rows) != 1:
                raise _Pending("exact_memory_event_unavailable")
            retains = await rows("""select id,owner_id,document_id,generation,operation,status,dispatch_count
                from jarvis_memory_outbox where owner_id=$1 and document_id=$2
                and operation='retain' and generation=$3 limit 2""", owner, source.document_id, source.generation)
            graph.update(tool_event_id=str(tool.id), tool_revision=tool.revision,
                         tool_payload_id=str(tool.payload_id))
            continuation = {"kind": "observed_tool_result_continuation", "owner_id": owner,
                "session_id": str(session), "audio_session_id": request.audio_session_id,
                "connection_id": request.connection_id, "response_id": request.response_id,
                "request_id": admitted[0], "turn_generation": admitted[1],
                "provider_call_id": content.get("provider_call_id"), "tool_event_id": str(tool.id),
                "tool_revision": tool.revision, "tool_payload_id": str(tool.payload_id),
                "request_event_id": str(canonical.id), "request_revision": canonical.revision,
                "authority": graph["authority"], "evidence": _reference(graph, "continuation")}
            return replace(result, observed_at=self.clock(), tool_outcome=tool,
                memory_event=context_record(memory_rows[0]), sources=(source,),
                retains=tuple(dict(row) for row in retains), continuation=continuation), graph

    async def shutdown(self):
        self.active = False
        LOGGER.removeHandler(self)
        tasks = set(self.inflight)
        if self.bridge is not None and self.bridge.task is not None:
            tasks.add(self.bridge.task)
        tasks.discard(asyncio.current_task())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=self.config.read_timeout_seconds)
            if done:
                await asyncio.gather(*done, return_exceptions=True)
            if pending:
                raise RuntimeError("Memory observer reads remain unjoined")
        self._shutdown_receipt = {"status": "closed", "joined": True, "observer_joined": True,
                "io_loop_joined": True, "polling_tasks": 0, "consumes": len(self.nonces),
                "queries": self.query_count, "trace_records": len(self.records), "issue": self.issue,
                "failure_code": self.issue or (self.bridge.failure if self.bridge else None),
                "file_control": self.bridge.status() if self.bridge else None}
        if self.bridge is not None:
            self.bridge.publish_status(self._shutdown_receipt)
        return dict(self._shutdown_receipt)
