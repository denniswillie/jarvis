"""Capture Live captions as partial evidence, never invented turns or outputs.

SessionContextCapture remains the bounded canonical writer. Its final-input and
response/output APIs are intentionally NOT used: Live fragments supply neither
boundary. A future trusted request-confirmation ingress must establish that
boundary before TaskAdmissionAdapter can accept a delegated operation.
"""
import asyncio
from collections import OrderedDict

from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.models import TaskError, digest
from app.tasks.readonly import FRAGMENT_FORMAT


class LiveCaptionCapture:
    def __init__(self, capture, context, *, observation_backed=False):
        if (not isinstance(capture, SessionContextCapture)
                or (capture.owner_id, capture.session_id, capture.audio_session_id, capture.provider)
                != (context.owner_id, context.jarvis_session_id, context.audio_session_id, "openai")):
            raise ValueError("Live capture must belong to the authenticated audio connection")
        self.capture = capture
        self.observation_backed = observation_backed
        self._pending = OrderedDict()
        self._originals = {}
        self._writer = None
        self.issue = None
        self._closed = False
        self._progress = asyncio.Event()

    def transcript(self, event, parts, *, roots=()):
        selected = tuple(part for part in parts if part.transcript_id == event.transcript_id
                         and part.revision <= event.revision)
        if (event.final or event.output is not None or not selected or len(selected) > 256
                or "".join(part.fragment.delta for part in selected) != event.text):
            self.capture._failure("live_caption_provenance_unavailable")
            return False
        # Anchor the existing canonical source clock before asynchronous writes
        # or caption coalescing. Later revisions must not move its start time.
        source = self.capture._source("live-caption", str(event.transcript_id))
        self.capture._source_times.setdefault(source, event.observed_at)
        if self.observation_backed:
            if self._closed or self.issue or len(selected) > 128 or len(self._pending) >= 16:
                self.capture._failure("live_observation_capacity_or_capture_unavailable")
                return False
            self._pending[event.transcript_id] = (event, selected, tuple(roots))
            self._progress.set()
            if self._writer is None or self._writer.done():
                self._writer = asyncio.create_task(self._drain(), name="live-provisional-observation-capture")
            return True
        return self._caption(event, selected, roots if event.role == "assistant" else ())

    def _caption(self, event, selected, roots):
        return self.capture._put(self.capture._source("live-caption", str(event.transcript_id)),
            kind="user_transcript" if event.role == "user" else "assistant_output",
            trust="user_input" if event.role == "user" else "assistant_output",
            finality="provisional", source_payload_ids=tuple(roots),
            payload={"voice_route":"live", "route":event.route.model_dump(mode="json"),
                "caption_id":str(event.transcript_id), "caption_revision":event.revision,
                "observed_at":event.observed_at.isoformat(),
                "text":event.text, "speaker":event.role, "completion_state":"partial",
                "request_boundary":"unavailable", "output_identity":None, "heard_transcript":None,
                "playback_status":"unknown", "provider_timing":"approximate_not_a_turn_boundary",
                "fragments":[{"provider_event_id":part.fragment.event_id,
                    "start_ms":part.fragment.start_ms, "end_ms":part.fragment.end_ms,
                    "delta":part.fragment.delta} for part in selected]})

    @staticmethod
    def _key(part):
        return (part.transcript_id, part.revision, part.fragment.event_id)

    async def _drain(self):
        try:
            while self._pending:
                _, (event, parts, roots) = self._pending.popitem(last=False)
                for part in parts:
                    key = self._key(part)
                    if key in self._originals:
                        continue
                    if len(self._originals) >= 1024:
                        raise TaskError("live_observation_capacity", "The bounded fragment archive is full.")
                    fragment = part.fragment
                    content = {"format": FRAGMENT_FORMAT, "voice_route": "live",
                        "route": event.route.model_dump(mode="json"), "audio_provider": "openai",
                        "audio_session_id": self.capture.audio_session_id,
                        "caption_id": str(part.transcript_id), "caption_revision": part.revision,
                        "provider_event_id": fragment.event_id, "speaker": fragment.speaker,
                        "delta": fragment.delta, "start_ms": fragment.start_ms, "end_ms": fragment.end_ms,
                        "speech_finality": "provisional", "request_boundary": "unavailable",
                        "provider_timing": "approximate_not_a_turn_boundary", "heard_transcript": None,
                        "source_time_basis": "application_ingress_not_physical_capture"}
                    identity = self.capture._source("live-fragment", ":".join(map(str, key)))
                    # Finality describes a received-fragment OBSERVATION, never
                    # a finalized utterance. Display captions depend on these
                    # removable originals; there is no second transcript store.
                    async with asyncio.timeout(self.capture.write_timeout):
                        record = await self.capture.repository.append(self.capture.owner_id, ContextIngress(
                            session_id=self.capture.session_id, source_namespace=self.capture.namespace,
                            source_event_id=identity, receipt_id=digest(content), expected_revision=0,
                            kind="observation", trust="derived", source_time=event.observed_at,
                            payload=content, source_payload_ids=roots if fragment.speaker == "assistant" else ()))
                    self._originals[key] = record
                    self._progress.set()  # append() has exited its commit boundary.
                parents = tuple(self._originals[self._key(part)].payload_id for part in parts)
                if not self._caption(event, parts, parents):
                    raise TaskError("live_caption_capture_failed", "The derived caption was not captured.")
        except asyncio.CancelledError:
            raise
        except Exception:
            self.issue = "live_observation_capture_failed"
            self.capture._failure(self.issue)
            self._pending.clear()
            self._progress.set()

    async def observations(self, parts):
        """A frozen received window, not an inferred speech boundary."""
        if not self.observation_backed:
            raise TaskError("live_observation_capture_disabled", "Observation capture is not initialized.")
        # The provider reader already froze a text-budgeted window. Preserve
        # every selected original; tiny transcription deltas are not turns.
        selected = tuple(parts)
        if not selected:
            raise TaskError("live_observation_gap", "The exact delegated fragments have not been captured.")
        # Delegation and transcript consumers are independent. Wait for the
        # exact received window even when the caption consumer has not started
        # its writer yet. Newer unrelated fragments neither satisfy nor extend
        # this bounded barrier. No provider or client reader waits here.
        async with asyncio.timeout(self.capture.write_timeout + 1):
            while True:
                self._progress.clear()
                if self.issue or self._closed:
                    raise TaskError("live_observation_gap", "The delegated fragments could not be committed.")
                if all(self._key(part) in self._originals for part in selected):
                    break
                await self._progress.wait()
        return tuple(self._originals[self._key(part)] for part in selected)

    async def close(self):
        self._closed = True
        self._progress.set()
        if self._writer is not None:
            self._writer.cancel()
            await asyncio.gather(self._writer, return_exceptions=True)
        self._pending.clear()

    def delegation_unavailable(self, request, route):
        return self.capture._put(self.capture._source("live-delegation", request.delegation_id),
            kind="context_gap", trust="derived", payload={"voice_route":"live",
                "route":route.model_dump(mode="json"), "provider_event_id":request.event_id,
                "delegation_id":request.delegation_id, "offset_ms":request.offset_ms,
                "reason":"canonical_request_boundary_unavailable", "task_admitted":False})
