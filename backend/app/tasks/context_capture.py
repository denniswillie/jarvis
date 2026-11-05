"""Nonblocking, provider-neutral capture of explicitly correlated voice evidence.

This adapter is optional. It does not own turn admission, retry domain actions,
store PCM, or infer what the user heard from a generated transcript.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
import logging
from math import isfinite
import re
from time import monotonic
import traceback
from typing import Any
from uuid import UUID, uuid4

from app.experience.evidence import durable_tool_payload
from app.images import ImageReference, image_source_payload_ids
from app.tasks.context_events import ContextIngress, ContextRecord
from app.realtime.request_sources import (
    MAX_UNWRITTEN_REQUESTS, OfferedUserRequest, ResponseRequestSources, RetainedUserRequest, source_error,
)
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError, digest


LOGGER = logging.getLogger(__name__)
_CAPTURE_IO_ISSUES = frozenset({
    "context_capture_persistence_failed", "context_capture_flush_timeout", "context_request_lookup_failed",
})


@dataclass
class CapturedOutput:
    response_id: str
    item_id: str
    text: str = ""
    provider_audio_bytes: int = 0
    forwarded_audio_bytes: int = 0
    interrupted: bool = False
    completed: bool = False
    terminal: bool = False
    checkpoint_characters: int = -1
    checkpoint_bytes: int = -1
    provenance: dict | None = None
    played_samples: int | None = None
    playback_state: str = "unknown"
    source_payload_ids: tuple[UUID, ...] = ()
    conversation_metadata: dict | None = None


class SessionContextCapture:
    """One trusted owner/session/audio connection; source IDs never come from tools."""

    def __init__(self, repository: ContextRepository, *, owner_id: str, session_id: UUID,
                 provider: str, audio_session_id: str, max_sources: int = 512,
                 max_pending: int = 128, write_timeout: float = 3.0,
                 committed_record_observer=None):
        if (provider not in {"openai", "gemini"} or not audio_session_id or max_pending < 2 or max_sources < 1
                or not isfinite(write_timeout) or write_timeout <= 0):
            raise ValueError("invalid canonical capture configuration")
        self.repository = repository
        self.owner_id, self.session_id = owner_id, session_id
        self.provider, self.audio_session_id = provider, audio_session_id
        # Hashing bounds the namespace without depending on provider ID length.
        self.namespace = "voice:" + digest({"provider": provider, "audio_session_id": audio_session_id})
        self.max_sources, self.max_pending = max_sources, max_pending
        self.write_timeout = write_timeout
        self._inputs: dict[str, str | None] = {}
        self._input_versions: dict[str, int] = {}
        self._request_records: dict[str, ContextRecord] = {}
        # One bounded pending set; each origin preserves its actual disposition.
        self._unwritten_inputs: OrderedDict[str, RetainedUserRequest] = OrderedDict()
        self._selected_call_sources: dict[str, OfferedUserRequest] = {}
        self.unwritten_dropped = 0
        self._outputs: dict[str, CapturedOutput] = {}
        self._image_deliveries: dict[str, tuple[ImageReference | None, str]] = {}
        self._calls: dict[str, str | None] = {}
        self._commits: OrderedDict[str, str] = OrderedDict()
        self._generations: dict[int, str] = {}
        self._provider_inputs: dict[str, str] = {}
        self._input_provider_ids: dict[str, str] = {}
        self._responses: dict[str, str | None] = {}
        self._response_sources: dict[str, tuple[UUID, ...]] = {}
        self._response_generations: dict[str, int | None] = {}
        self._response_binding_conflicts: set[str] = set()
        self._pending: OrderedDict[str, ContextIngress] = OrderedDict()
        self._last_fingerprints: dict[str, str] = {}
        self._source_times: dict[str, datetime] = {}
        self._ingress_sequence = 0
        self._failed_gap: ContextIngress | None = None
        self._writer: asyncio.Task | None = None
        self._active_source: str | None = None
        self._write_progress = asyncio.Event()
        self._closed = False
        self.issue: str | None = None
        self.request_authority_issue: str | None = None
        self.failed_events = 0
        # Optional synchronous admission only. Never await perception/model work
        # in the canonical writer or turn its failure into a capture failure.
        self.committed_record_observer = committed_record_observer
        self.observer_issue: str | None = None

    def _source(self, category: str, identity: str) -> str:
        return category + ":" + digest({"identity": identity})

    def _event(self, source: str, *, kind: str, trust: str, payload: dict,
               finality: str = "final", source_payload_ids: tuple[UUID, ...] = ()) -> ContextIngress:
        content = {"audio_provider": self.provider, "audio_session_id": self.audio_session_id,
                   "source_time_basis": "application_ingress_not_physical_capture", **payload}
        source_time = self._source_times.setdefault(source, datetime.now(UTC))
        return ContextIngress(session_id=self.session_id, source_namespace=self.namespace,
            source_event_id=source, receipt_id=digest({"kind": kind, "finality": finality, "payload": content}),
            kind=kind, trust=trust, finality=finality, payload=content, source_time=source_time,
            source_payload_ids=source_payload_ids)

    def _failure(self, code: str) -> None:
        self.issue = code
        if code not in _CAPTURE_IO_ISSUES:
            self.request_authority_issue = self.request_authority_issue or code
        self.failed_events += 1
        LOGGER.warning("Canonical voice capture incomplete provider=%s code=%s", self.provider, code)
        if not self._closed:
            # One reserved slot ensures overflow itself has a durable description.
            self._pending["capture-gap"] = self._event("capture-gap", kind="context_gap", trust="derived",
                payload={"reason": code, "failed_events": self.failed_events, "coverage": "incomplete",
                         "request_authority_issue": self.request_authority_issue})
            self._failed_gap = None
            self._start_writer()

    def _start_writer(self) -> None:
        if self._writer is None or self._writer.done():
            self._writer = asyncio.create_task(self._write_pending())

    def _put(self, source: str, **kwargs) -> bool:
        if self._closed:
            self._failure("capture_after_close")
            return False
        if source not in self._pending and len(self._pending) >= self.max_pending - 1:
            self._failure("context_capture_queue_full")
            return False
        if source not in self._last_fingerprints and len(self._last_fingerprints) >= self.max_sources * 4:
            self._failure("context_capture_source_limit")
            return False
        try:
            event = self._event(source, **kwargs)
        except (KeyError, ValueError, TypeError):
            self._failure("context_capture_payload_invalid")
            return False
        if self._last_fingerprints.get(source) == event.receipt_id:
            return True
        self._last_fingerprints[source] = event.receipt_id
        self._ingress_sequence += 1
        # A -> B -> A is a new revision, not a replay of the first A. A repeated
        # identical callback remains a no-op; each actual change has an ordinal.
        event = event.model_copy(update={"receipt_id": f"{self._ingress_sequence}:{event.receipt_id}"})
        # Coalesce snapshots of the same source, never different utterances.
        self._pending[source] = event
        self._write_progress.set()
        if self._failed_gap is not None:
            self._pending["capture-gap"] = self._failed_gap
            self._failed_gap = None
        self._start_writer()
        return True

    async def _write_pending(self) -> None:
        while self._pending:
            # Keep one writer and the existing bounded queue. Finalized speech
            # must not drain unrelated visual/output snapshots before it can
            # authorize a tool. Preserve FIFO among finalized user inputs.
            key = next((key for key, pending in self._pending.items()
                        if pending.kind == "user_transcript" and pending.finality in {"final", "revised"}),
                       next(iter(self._pending)))
            event = self._pending.pop(key)
            self._active_source = key
            started = monotonic()
            deadline = asyncio.timeout(self.write_timeout)
            try:
                async with deadline:
                    record = await self.repository.append(self.owner_id, event)
                    identity = event.payload.get("input_id")
                    if (isinstance(record, ContextRecord) and record.kind == "user_transcript"
                            and identity in self._inputs and record.owner_id == self.owner_id
                            and record.session_id == self.session_id and record.source_namespace == self.namespace):
                        self._request_records[identity] = record
                    if isinstance(record, ContextRecord) and self.committed_record_observer is not None:
                        try:
                            self.committed_record_observer(record)
                        except Exception:
                            self.observer_issue = "committed_record_observer_failed"
                            LOGGER.warning("Canonical capture observer failed; committed source is preserved")
            except Exception as error:
                if isinstance(error, TaskError) and error.code == "context_source_removed":
                    # Intentional removal wins over an already-queued copy.
                    # Its durable tombstone is not a lost microphone transcript.
                    continue
                if isinstance(error, TaskError):
                    # A canonical guard/finality failure is not downgraded to
                    # an unrelated I/O gap, even if a later write also times out.
                    self.request_authority_issue = self.request_authority_issue or error.code
                self._log_write_failure(event, error, started=started,
                    deadline_exceeded=deadline.expired(), is_gap=key == "capture-gap")
                # Do not log private speech or database error contents. A failed
                # gap write must not generate an endless stream of new gap writes.
                if key == "capture-gap":
                    self.issue = "context_capture_persistence_failed"
                    self._failed_gap = event
                else:
                    self._failure("context_capture_persistence_failed")
            finally:
                self._active_source = None
                self._write_progress.set()

    def _log_write_failure(self, event, error, *, started, deadline_exceeded, is_gap):
        """Classify failed writes without logging exception text, SQL or payloads."""
        def token(value):
            return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_]{1,80}", value) else None

        cause = error.__cause__
        trace = cause.__traceback__ if cause is not None else error.__traceback__
        frames = []
        for frame, line in traceback.walk_tb(trace):
            filename = frame.f_code.co_filename.replace("\\", "/")
            if "/app/" not in filename:
                continue
            relative = filename.rsplit("/app/", 1)[1]
            if re.fullmatch(r"[A-Za-z0-9_./-]{1,160}", relative):
                frames.append({"file": "app/" + relative,
                    "function": token(frame.f_code.co_name), "line": line})
        diagnostic = {
            "provider": self.provider, "kind": event.kind, "finality": event.finality,
            "error_type": token(type(error).__name__),
            "cause_type": token(type(cause).__name__) if cause is not None else None,
            "task_error_code": token(error.code) if isinstance(error, TaskError) else None,
            "sqlstate": token(getattr(error, "sqlstate", None)),
            "deadline_exceeded": deadline_exceeded,
            "elapsed_ms": round((monotonic() - started) * 1000, 3),
            "deadline_ms": self.write_timeout * 1000,
            "pending_sources": len(self._pending), "is_gap": is_gap,
            "application_frames": frames[-8:],
        }
        LOGGER.warning("Canonical voice capture write diagnostic=%s",
            json.dumps(diagnostic, sort_keys=True))

    def _room(self, collection: dict, identity: str) -> bool:
        if identity not in collection and len(collection) >= self.max_sources:
            self._failure("context_capture_source_limit")
            return False
        return True

    def input_admitted(self, identity: str) -> None:
        if not self._room(self._inputs, identity) or identity in self._inputs:
            return
        self._inputs[identity] = None
        self._put(self._source("input", identity), kind="user_transcript", trust="user_input",
            finality="unavailable", payload={"input_id": identity, "text": None, "transcript_status": "pending"})

    def input_final(self, identity: str, text: str) -> None:
        if not self._room(self._inputs, identity):
            return
        if not text.strip():
            self.input_unavailable(identity, reason="empty_provider_transcript")
            return
        if len(text) > 64_000:
            self._failure("context_transcript_too_large")
            self.input_unavailable(identity, reason="transcript_exceeds_capture_limit")
            return
        if self._inputs.get(identity) == text:
            return
        revised = self._inputs.get(identity) is not None
        self._inputs[identity] = text
        self._input_versions[identity] = self._input_versions.get(identity, 0) + 1
        self._put(self._source("input", identity), kind="user_transcript", trust="user_input",
            finality="revised" if revised else "final",
            payload={"input_id": identity, "text": text, "transcript_status": "finalized",
                     "capture_input_version": self._input_versions[identity],
                     "provider_item_id": self._input_provider_ids.get(identity, identity if self.provider == "gemini" else None)})

    def input_unavailable(self, identity: str, *, reason: str) -> None:
        if not self._room(self._inputs, identity) or self._inputs.get(identity) is not None:
            return
        self._inputs[identity] = None
        self._put(self._source("input", identity), kind="user_transcript", trust="user_input",
            finality="unavailable", payload={"input_id": identity, "text": None, "transcript_status": reason})

    def admit_openai_commit(self, request_id: str, generation: int) -> None:
        if not self._room(self._commits, request_id) or not self._room(self._generations, generation):
            return
        identity = "commit:" + request_id
        self._commits[request_id] = identity
        self._generations[generation] = identity
        self.input_admitted(identity)

    def openai_committed(self, item_id: str) -> None:
        if item_id in self._provider_inputs or not self._room(self._provider_inputs, item_id):
            return
        # Manual commits and their ACKs share one ordered WebSocket connection.
        # Never assign a late ACK to the most recent user generation.
        identity = self._commits.popitem(last=False)[1] if self._commits else "item:" + item_id
        self._provider_inputs[item_id] = identity
        self._input_provider_ids[identity] = item_id
        self.input_admitted(identity)

    def openai_commit_rejected(self, request_id: str) -> None:
        identity = self._commits.pop(request_id, None)
        if identity is not None:
            self.input_unavailable(identity, reason="provider_commit_rejected")

    def openai_transcript(self, item_id: str | None, text: str | None) -> None:
        if not isinstance(item_id, str) or not item_id.strip():
            self._failure("context_transcript_identity_missing")
            return
        # An unattributed provider item can be retained, but cannot grant a tool
        # authority over some other locally admitted input.
        identity = self._provider_inputs.get(item_id, "item:" + item_id)
        if self._room(self._input_provider_ids, identity):
            self._input_provider_ids[identity] = item_id
        if text is None:
            self.input_unavailable(identity, reason="provider_transcription_failed")
        else:
            self.input_final(identity, text)

    def bind_openai_response(self, response_id: str, generation: int | None, *,
                             source_payload_id: UUID | None = None) -> None:
        if (not isinstance(response_id, str) or not response_id.strip() or "\x00" in response_id
                or len(response_id.encode("utf-8")) > 512):
            self._failure("context_response_identity_invalid")
            return
        if generation is not None and (type(generation) is not int or not 0 <= generation < 2**63):
            self._failure("context_response_generation_invalid")
            return
        if not self._room(self._responses, response_id):
            return
        parents = (source_payload_id,) if source_payload_id is not None else ()
        if response_id in self._responses:
            if response_id in self._response_binding_conflicts:
                return
            if self._response_sources[response_id] != parents:
                reason = "context_response_source_conflict"
            elif (self._response_generations[response_id] != generation
                  or (self._responses[response_id] is not None
                      and self._responses[response_id] != self._generations.get(generation))):
                reason = "context_response_input_conflict"
            else:
                # A generation admitted later cannot retroactively resolve an
                # already-unresolved response. Identical duplicates do no work.
                return
            self._response_binding_conflicts.add(response_id)
            self._responses[response_id] = None
            self._failure(reason)
            # Replace existing snapshots through the normal bounded writer, not
            # an admission-time database lookup or a new association journal.
            for output in self._outputs.values():
                if output.response_id == response_id:
                    self._checkpoint(output)
            return
        self._response_sources[response_id] = parents
        self._response_generations[response_id] = generation
        self._responses[response_id] = self._generations.get(generation)

    def response_input_association(self, response_id: str) -> dict[str, Any] | None:
        """Content-free source binding, NOT finalized request or playback proof.

        This synchronous projection performs no persistence or request lookup.
        Existing assistant checkpoints retain it; a response without an output
        has only this in-memory diagnostic. Canonical availability/revision must
        be resolved independently from the exact scoped source at evaluation.
        """
        if self.provider != "openai" or response_id not in self._responses:
            return None
        if any(not isinstance(value, str) or not value.strip() or "\x00" in value
               or len(value.encode("utf-8")) > 512
               for value in (self.owner_id, self.audio_session_id, response_id)):
            return None
        identity = self._responses[response_id]
        if identity is not None and (len(identity.encode("utf-8")) > 512 or "\x00" in identity):
            identity = None
        status = ("conflict" if response_id in self._response_binding_conflicts else
                  "source_bound" if identity is not None else "unresolved")
        return {
            "schema_version": 1, "authority": "session_context_capture",
            "owner_id": self.owner_id, "session_id": str(self.session_id),
            "audio_provider": self.provider, "audio_session_id": self.audio_session_id,
            "source_namespace": self.namespace, "response_id": response_id,
            "turn_generation": self._response_generations[response_id],
            "binding_status": status, "input_id": identity,
            "source_event_id": self._source("input", identity) if identity is not None else None,
            "canonical_event_id": None, "canonical_revision": None,
            "heard_state": "unknown", "full_chain_proven": False,
        }

    def bind_openai_call(self, call_id: str, response_id: str) -> None:
        self.bind_call(call_id, self._responses.get(response_id))

    def cached_request(self, identity):
        record = self._request_records.get(identity)
        return record if self._request_matches(identity, self._input_versions.get(identity), record) else None

    def _request_matches(self, identity, version, record):
        content = record.content or {} if isinstance(record, ContextRecord) else {}
        return (not self._closed and self.request_authority_issue is None
                and type(version) is int and version > 0 and self._input_versions.get(identity) == version
                and self._inputs.get(identity) is not None
                and isinstance(record, ContextRecord) and record.authorizes_request
                and record.owner_id == self.owner_id and record.session_id == self.session_id
                and record.source_namespace == self.namespace
                and record.source_event_id == self._source("input", identity)
                and content.get("input_id") == identity and content.get("text") == self._inputs[identity]
                and type(content.get("capture_input_version")) is int
                and content["capture_input_version"] == version)

    def retain_unwritten(self, generation, scope):
        self._retain_request(generation, scope, "retained_unwritten")

    def retain_cancelled_empty(self, generation, scope, offer=None):
        """Called only after the response owner seals a proven no-output cancellation."""
        if offer is not None:
            if (not isinstance(offer, ResponseRequestSources) or offer.generation != generation
                    or not offer.scope.same(scope) or len(offer.requests) > MAX_UNWRITTEN_REQUESTS + 1):
                return
            # A second interruption must not forget earlier unfulfilled requests.
            # No tools were admitted for this response, so none of its offered
            # requests acquired an effect here. Preserve their original kinds.
            for request in offer.requests:
                if (request.kind in {"retained_unwritten", "retained_cancelled_empty"}
                        and (request.record is None or request.accepts(self.cached_request(request.input_id), self))):
                    self._retain_input(request.input_id, scope, request.kind)
        self._retain_request(generation, scope, "retained_cancelled_empty")

    def _retain_request(self, generation, scope, kind):
        identity = self._generations.get(generation)
        self._retain_input(identity, scope, kind)

    def _retain_input(self, identity, scope, kind):
        if (self._closed or self.request_authority_issue is not None or identity is None
                or (scope.owner_id, scope.session_id, scope.audio_session_id)
                != (self.owner_id, self.session_id, self.audio_session_id)):
            return
        self._unwritten_inputs[identity] = RetainedUserRequest(scope, kind)
        while len(self._unwritten_inputs) > MAX_UNWRITTEN_REQUESTS:
            self._unwritten_inputs.popitem(last=False)
            self.unwritten_dropped += 1

    def request_sources(self, generation, scope, followup=None):
        current = self._generations.get(generation)
        if self._closed or self.request_authority_issue is not None or current is None:
            return None
        carried = {}
        previous = getattr(followup, "source_offer", None)
        if previous is not None:
            if (not isinstance(previous, ResponseRequestSources)
                    or previous.generation != generation or not previous.scope.same(scope)
                    or len(previous.requests) > MAX_UNWRITTEN_REQUESTS + 1):
                raise source_error()
            for request in previous.requests:
                cached = self.cached_request(request.input_id)
                if ((request.record is not None or (request.kind == "current" and cached is not None))
                        and not request.accepts(cached, self)):
                    raise source_error()
                if request.input_id != current:
                    carried[request.input_id] = request
        for identity, retained in tuple(self._unwritten_inputs.items()):
            if not retained.scope.same(scope):
                self._unwritten_inputs.pop(identity)
        identities = [(current, "current")]
        identities.extend((identity, retained.kind) for identity, retained in self._unwritten_inputs.items()
                          if identity != current)
        # A continuation retains actual tool provenance, not a new replay duty.
        if followup is not None and followup.call_request_ids:
            for request in (followup.request, *followup.additional_requests):
                if request is not None:
                    matches = [identity for identity, record in self._request_records.items()
                               if record.id == request.event_id and record.revision == request.revision]
                    for identity in matches:
                        if identity not in {key for key, _ in identities}:
                            identities.append((identity, "tool_request"))
        # The parent offer was consumed from the pre-submit pending set after
        # possible write. Unselected requests are still context for this exact
        # owned continuation, not a new-turn queue or automatic tool retry.
        for identity, request in carried.items():
            if identity not in {key for key, _ in identities}:
                identities.append((identity, request.kind))
        if len(identities) <= 1:
            return None
        if len(identities) > MAX_UNWRITTEN_REQUESTS + 1:
            raise source_error()
        offered = tuple(
            replace(carried[identity], reference="request_" + uuid4().hex, kind=kind)
            if identity in carried else OfferedUserRequest("request_" + uuid4().hex, identity, kind,
                self.namespace, self._source("input", identity), self._input_provider_ids.get(identity),
                self._input_versions.get(identity, 0), self.cached_request(identity))
            for identity, kind in identities)
        return ResponseRequestSources(scope, generation, offered)

    def consume_unwritten(self, offer):
        if isinstance(offer, ResponseRequestSources):
            for request in offer.requests:
                if request.record is not None:
                    # Offered to a possibly written response, NOT fulfilled.
                    self._unwritten_inputs.pop(request.input_id, None)

    def bind_selected_call(self, call_id, request):
        if (self._closed or not isinstance(request, OfferedUserRequest)
                or request.namespace != self.namespace
                or (call_id in self._calls and self._calls[call_id] != request.input_id)
                or (call_id in self._selected_call_sources and self._selected_call_sources[call_id] != request)
                or not self._room(self._calls, call_id)):
            raise source_error()
        self._calls[call_id] = request.input_id
        self._selected_call_sources[call_id] = request

    def bind_call(self, call_id: str, input_id: str | None) -> None:
        if not self._room(self._calls, call_id):
            return
        if call_id in self._calls and self._calls[call_id] != input_id:
            self._failure("context_call_identity_conflict")
            return
        self._calls[call_id] = input_id

    async def request_for_call(self, call_id: str | None):
        identity = self._calls.get(call_id)
        selected = self._selected_call_sources.get(call_id)
        version = self._input_versions.get(identity)
        record = None
        if (not self._closed and self.request_authority_issue is None and identity is not None
                and self._inputs.get(identity) is not None):
            try:
                async with asyncio.timeout(self.write_timeout):
                    source = self._source("input", identity)
                    while True:
                        self._write_progress.clear()
                        if (self._closed or self.request_authority_issue is not None
                                or self._input_versions.get(identity) != version):
                            break
                        if source not in self._pending and self._active_source != source:
                            record = await self.repository.request_evidence(self.owner_id, self.session_id,
                                source_namespace=self.namespace, source_event_id=source)
                            break
                        # Cancelling this waiter never cancels the owned writer.
                        await self._write_progress.wait()
            except TaskError:
                pass
            except Exception:
                self._failure("context_request_lookup_failed")
        if (self._calls.get(call_id) != identity or self._selected_call_sources.get(call_id) != selected
                or not self._request_matches(identity, version, record)):
            record = None
        if selected is not None and not selected.accepts(record, self):
            # The executor catches this BEFORE invoking even a handler which
            # does not itself require request provenance. Never fall back.
            raise source_error()
        return record

    def _output(self, item_id: str, response_id: str) -> CapturedOutput | None:
        if not self._room(self._outputs, item_id):
            return None
        output = self._outputs.setdefault(item_id, CapturedOutput(response_id, item_id))
        if output.response_id != response_id:
            self._failure("context_output_identity_conflict")
            return None
        if response_id in self._response_sources:
            output.source_payload_ids = self._response_sources[response_id]
        return output

    def observe_output(self, event: dict[str, Any], *, forwarded: bool) -> None:
        kind = event.get("type")
        item_id, response_id = event.get("item_id"), event.get("response_id")
        if kind == "response.done" and isinstance(response_id, str):
            self.finish_response(response_id, interrupted=bool(event.get("interrupted")),
                                 completed=event.get("status", "completed") == "completed")
            return
        if kind not in {"audio.output.delta", "audio.output.done", "audio.output.clear", "transcript.delta", "transcript.done"}:
            return
        if not isinstance(item_id, str) or not isinstance(response_id, str):
            self._failure("context_output_identity_missing")
            return
        output = self._output(item_id, response_id)
        if output is None:
            return
        if kind == "audio.output.delta":
            try:
                data = event.get("audio_base64", "")
                if not isinstance(data, str) or len(data) > 12 * 1024 * 1024:
                    raise ValueError("invalid audio")
                count = len(base64.b64decode(data, validate=True))
                if count % 2:
                    raise ValueError("incomplete PCM16 sample")
            except (ValueError, binascii.Error):
                self._failure("context_output_audio_invalid")
                return
            output.provider_audio_bytes += count
            if forwarded:
                output.forwarded_audio_bytes += count
        elif kind in {"transcript.delta", "transcript.done"}:
            text = event.get("text")
            if not isinstance(text, str):
                self._failure("context_output_text_invalid")
                return
            text = text if kind == "transcript.done" else output.text + text
            if len(text) > 64_000:
                self._failure("context_transcript_too_large")
                return
            output.text = text
        if event.get("interrupted") or kind == "audio.output.clear":
            output.interrupted = True
        if (kind.endswith(".done") or kind.endswith(".clear") or output.checkpoint_characters < 0
                or len(output.text) - output.checkpoint_characters >= 1024
                or output.provider_audio_bytes - output.checkpoint_bytes >= 48_000):
            self._checkpoint(output)

    def assistant_message(self, *, response_id, item_id, text, metadata) -> bool:
        """Queue a native reply copy only after the session's delivery-policy gate.

        Successful queue admission is not a durable receipt or proof of hearing.
        The canonical writer commits the message, and records persistence gaps.
        """
        output = self._outputs.get(item_id)
        delivery = metadata.get("realtime_output") if isinstance(metadata, dict) else None
        if (self.provider != "openai" or output is None or output.response_id != response_id
                or not output.terminal or output.interrupted or output.provenance is not None
                or not isinstance(delivery, dict) or delivery.get("delivery_disposition") != "forwarded"
                or delivery.get("item_id") != item_id
                or metadata.get("audio_session_id") != self.audio_session_id):
            self._failure("context_message_source_unavailable")
            return False
        if not isinstance(text, str) or not text.strip() or text.strip() != output.text.strip():
            self._failure("context_message_content_conflict")
            return False
        admitted = {**metadata, "realtime_output": dict(delivery)}
        if output.conversation_metadata is not None and output.conversation_metadata != admitted:
            self._failure("context_message_identity_conflict")
            return False
        output.conversation_metadata = admitted
        return self._checkpoint(output)

    def _checkpoint(self, output: CapturedOutput, *, terminal: bool = False) -> bool:
        output.terminal = output.terminal or terminal
        output.checkpoint_characters = len(output.text)
        output.checkpoint_bytes = output.provider_audio_bytes
        return self._put(self._source("output", output.item_id), kind="assistant_output", trust="assistant_output",
            source_payload_ids=output.source_payload_ids,
            finality="final" if output.terminal else "provisional", payload={
                "response_id": output.response_id, "item_id": output.item_id, "text": output.text,
                "provider_audio_bytes": output.provider_audio_bytes, "forwarded_audio_bytes": output.forwarded_audio_bytes,
                "completion_state": ("interrupted" if output.interrupted else "completed" if output.completed
                                     else "incomplete" if output.terminal else "partial"),
                "playback_status": output.playback_state, "played_samples": output.played_samples,
                "heard_transcript": None,
                **({"conversation_message": output.conversation_metadata}
                   if output.conversation_metadata is not None else {}),
                **(output.provenance or {}),
                # Last so provider/provenance data cannot replace owner evidence.
                **({"response_input_association": self.response_input_association(output.response_id)}
                   if self.provider == "openai" and output.provenance is None else {})})

    def expert_speech(self, **kwargs):
        """Compatibility entry point for expert-only capture callers."""
        return self.tts_speech(**kwargs)

    def tts_speech(self, *, response_id, item_id, text, samples, complete, interrupted, provenance,
                      source_payload_id: UUID | None = None, authored_by="frontier_expert"):
        """Journal attributed media without inventing a heard prefix."""
        output = self._output(item_id, response_id)
        if output is None:
            return
        parents = (source_payload_id,) if source_payload_id is not None else ()
        if output.source_payload_ids and output.source_payload_ids != parents:
            self._failure("context_output_source_conflict")
            return
        output.source_payload_ids = parents
        output.text = text
        output.provider_audio_bytes = output.forwarded_audio_bytes = samples * 2
        output.completed, output.interrupted = complete, interrupted
        output.provenance = {**provenance, "audio_provider":"openai_tts", "authored_by":authored_by,
            "text_represents":("expert_script_not_a_verified_heard_transcript" if authored_by == "frontier_expert"
                               else "notification_script_not_a_verified_heard_transcript")}
        self._checkpoint(output, terminal=complete or interrupted)

    def finish_response(self, response_id: str, *, interrupted: bool, completed: bool = False) -> None:
        for output in self._outputs.values():
            if output.response_id == response_id:
                output.interrupted = output.interrupted or interrupted
                output.completed = completed and not output.interrupted
                self._checkpoint(output, terminal=True)

    def gemini_output(self, native_output) -> None:
        """Use native cumulative counters, including suppressed post-interruption audio."""
        output = self._output(native_output.item_id, native_output.response_id)
        if output is None:
            return
        if len(native_output.transcript) > 64_000:
            self._failure("context_transcript_too_large")
            return
        output.text = native_output.transcript
        output.provider_audio_bytes = native_output.provider_audio_bytes
        output.forwarded_audio_bytes = native_output.forwarded_audio_bytes
        output.interrupted = output.interrupted or native_output.interrupted
        output.completed = native_output.completed and not output.interrupted
        if (native_output.completed or native_output.interrupted or output.checkpoint_characters < 0
                or len(output.text) - output.checkpoint_characters >= 1024
                or output.provider_audio_bytes - output.checkpoint_bytes >= 48_000):
            self._checkpoint(output, terminal=native_output.completed)

    def playback(self, item_id: str, *, end_ms: int, content_index: int = 0) -> bool:
        output = self._outputs.get(item_id)
        if (output is None or type(end_ms) is not int or type(content_index) is not int
                or content_index != 0 or not 0 <= end_ms <= output.forwarded_audio_bytes * 1000 // 48_000):
            return False
        identity = self._source("playback", item_id + ":" + str(end_ms))
        self._put(identity, kind="played_output_boundary", trust="sensor", payload={
            "item_id": item_id, "response_id": output.response_id, "audio_end_ms": end_ms,
            "content_index": content_index, "source": "client_playback_truncated",
            "heard_transcript": None, "word_alignment_available": False})
        return True

    def playback_receipt(self, item_id, *, response_id, samples, state):
        output = self._outputs.get(item_id)
        if (output is None or output.response_id != response_id or type(samples) is not int
                or not 0 <= samples <= output.forwarded_audio_bytes // 2
                or samples < (output.played_samples or 0)
                or state not in {"started", "progress", "interrupted", "completed"}):
            return False
        if output.playback_state == "client_reported_completed" and state != "completed":
            return True
        if output.played_samples == samples and output.playback_state == "client_reported_" + state:
            return True
        output.played_samples = samples
        output.playback_state = "client_reported_" + state
        # Progress is a revision of one item's boundary, not a new observation
        # identity for every audio packet. Pending revisions can coalesce while
        # retaining distinct started/interrupted/completed evidence.
        source = self._source("playback", f"{item_id}:{state}")
        # The larger lower bound was observed now, not at the first progress
        # callback. Identity stays stable, but the revision's evidence time does
        # not move a later playback observation into the past.
        self._source_times[source] = datetime.now(UTC)
        self._put(source,
            kind="played_output_boundary", trust="sensor", payload={
                "item_id":item_id, "response_id":response_id, "played_samples":samples,
                "sample_rate":24000, "source":"client_av_audio_data_played_back",
                "state":state, "boundary":"played_sample_lower_bound",
                "heard_transcript":None, "word_alignment_available":False})
        self._checkpoint(output)
        return True

    def image_delivery(self, reference: ImageReference | None, state: str, delivery_id: str) -> None:
        """Capture transport evidence, never a claim that the model understood pixels."""
        allowed = {"socket_written", "provider_acknowledged", "write_outcome_unknown", "not_written", "provider_rejected"}
        if (state not in allowed or not delivery_id or len(delivery_id) > 512
                or reference is not None and (reference.owner_id != self.owner_id
                    or reference.source == "current" and reference.session_id != self.session_id)
                or self.provider != "openai" and state == "provider_acknowledged"):
            self._failure("context_image_evidence_invalid")
            return
        if not self._room(self._image_deliveries, delivery_id):
            return
        previous = self._image_deliveries.get(delivery_id)
        if previous is not None:
            if previous[0] != reference:
                self._failure("context_image_identity_conflict")
                return
            # A native ACK can arrive before the writer coroutine returns.
            # Do not overwrite a definitive receipt with a late local status.
            if previous[1] in {"provider_acknowledged", "provider_rejected"}:
                if state in {"socket_written", "write_outcome_unknown", "not_written"}:
                    return
        self._image_deliveries[delivery_id] = (reference, state)
        # Each transport state is a separate immutable receipt about these
        # exact pixels. An ACK is not a correction of the captured observation:
        # revising one source here invalidated already-consumed task manifests.
        # Keep ordinary current-source dependencies, so actual correction or
        # removal of a receipt/original still fences every dependent task.
        source = self._source("image", digest({"delivery_id": delivery_id, "state": state}))
        if reference is not None:
            self._source_times.setdefault(source, reference.captured_at)
        self._put(
            source, kind="observation", trust="sensor",
            source_payload_ids=reference.source_payload_ids if reference is not None else (),
            payload={
                "images": [reference.to_metadata()] if reference is not None else [],
                "source_time_basis": "image_capture" if reference is not None else "application_ingress_not_physical_capture",
                "delivery_id": delivery_id,
                "delivery_id_basis": "provider_item" if self.provider == "openai" else "application_image_write",
                "delivery_state": state,
                "delivery_state_scope": "immutable_receipt",
                "image_view": "captured_snapshot" if reference is not None else "unavailable",
                "current_view_verified": False,
                "provider_acknowledged": state == "provider_acknowledged",
                "model_understanding": "unknown",
                "evidence_gap": "image_reference_unavailable" if reference is None else None,
            },
        )


    async def bound_tool_outcome(self, connection, call_id, name, result, sources, *, publication_area_id=None):
        if self._closed or not isinstance(call_id, str) or not call_id:
            raise TaskError("tool_context_unavailable", "A current tool-call identity is required.")
        event = self._event(self._source("bound-tool", call_id), kind="tool_outcome", trust="tool_receipt",
            payload={"provider_call_id":call_id, "tool_name":name, "result":result,
                     "input_id":self._calls.get(call_id)}, source_payload_ids=sources)
        scope = {"publication_area_id": publication_area_id} if publication_area_id is not None else {}
        return await self.repository.append_in_transaction(connection, self.owner_id, event, **scope)


    def tool_outcome(self, call_id: str | None, name: str, result: dict, *, mutates_state: bool = False) -> None:
        if call_id is None:
            return
        try:
            sources = image_source_payload_ids(result, owner_id=self.owner_id)
            safe = durable_tool_payload(result, status=str(result.get("status", "failed")))
        except (ValueError, TypeError):
            self._failure("context_tool_payload_invalid")
            return
        self._put(self._source("tool", call_id), kind="tool_outcome", trust="tool_receipt",
            source_payload_ids=sources,
            payload={"provider_call_id": call_id, "tool_name": name, "result": safe,
                     "mutates_state": mutates_state is True,
                     "input_id": self._calls.get(call_id)})

    async def flush(self) -> bool:
        if self._failed_gap is not None and not self._closed:
            self._pending["capture-gap"] = self._failed_gap
            self._failed_gap = None
            self._start_writer()
        try:
            async with asyncio.timeout(self.write_timeout):
                while self._writer is not None and not self._writer.done():
                    await asyncio.shield(self._writer)
        except TimeoutError:
            self._failure("context_capture_flush_timeout")
        return self.issue is None and not self._pending

    def context_reset(self, reset_id, *, client_cleared, input_was_pending):
        self._unwritten_inputs.clear()
        self._put(self._source("context-reset", reset_id), kind="context_gap", trust="derived",
            payload={"reason":"source_context_rebuild", "reset_id":reset_id,
                     "client_reported_audio_cleared":client_cleared,
                     "unfinished_input_at_reset":input_was_pending,
                     "meaning":"A reset is not task cancellation, a recovered transcript, or proof of hearing."})

    async def close(self) -> None:
        if self._closed:
            return
        for identity, text in self._inputs.items():
            if text is None:
                self.input_unavailable(identity, reason="session_closed_without_transcript")
        for output in self._outputs.values():
            if not output.terminal:
                output.interrupted = True
            self._checkpoint(output, terminal=True)
        await self.flush()
        self._closed = True
        self._write_progress.set()
        if self._writer is not None and not self._writer.done():
            self._writer.cancel()
            await asyncio.gather(self._writer, return_exceptions=True)
        self._pending.clear()
        self._unwritten_inputs.clear()
        self._request_records.clear()
        self._selected_call_sources.clear()

    def diagnostics(self) -> dict:
        return {"namespace": self.namespace, "issue": self.issue, "failed_events": self.failed_events,
                "request_authority_issue": self.request_authority_issue,
                "unwritten_requests": sum(r.kind == "retained_unwritten" for r in self._unwritten_inputs.values()),
                "cancelled_empty_requests": sum(r.kind == "retained_cancelled_empty" for r in self._unwritten_inputs.values()),
                "unwritten_dropped": self.unwritten_dropped,
                "pending_sources": len(self._pending), "closed": self._closed,
                "missing_transcripts": sum(text is None for text in self._inputs.values())}
