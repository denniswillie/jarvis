"""Provider/DB-free current single-voice iOS control fixture.

Actual NativeAttentionSink, NativeVoiceDelivery, VoiceNotificationLedger and
notification response-admission code run over the existing in-memory canonical
repository fixture. Transport observations and fences are controlled, not RTP,
SQL, task execution, normal-app startup, acoustic playback or hearing evidence.
No authored TTS path, provider client, microphone or camera is started.

Future loopback qualification only:
  PYTHONPATH=.:tests python -m evals.second_brain.ios_single_voice --output NEW_DIR
Pass ready.url to JARVIS_DELIVERY_FIXTURE_URL for the four SingleVoice selectors.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

from websockets.asyncio.server import serve

from app.attention.models import AttentionError
from app.attention.native_voice import NativeAttentionSink
from app.realtime.session import RealtimeAudioSession
from app.voice_routes.contracts import CommandReceipt, DeliveryEvidence, OutputIdentity, RouteIdentity
from test_notification_ledger import CanonicalFixture, NOW, capabilities


CASES = frozenset({"completion", "priority_deduplication", "cancel_reconnect", "source_reset"})
PROTOCOL = "jarvis.ios_single_voice_control.v1"


class Probe:
    def __init__(self, case):
        self.case = case
        self.repository = CanonicalFixture()
        self.repository.lease_seconds = 60
        self.repository.renew = self.renew
        self.repository.settle = self.settle_unprepared
        self.session_id = self.repository.scope.session_id
        self.audio_id = uuid4()
        self.reset_id = uuid4()
        self.connections = 0
        self.waiting_reconnect = False
        self.closed = False
        self.socket = None
        self.delivery = None
        self.finished_evidence = None
        self.writes = []
        self.fences = []
        self.reservations = []
        self.checks = {}
        self.session = SimpleNamespace(
            _closed=asyncio.Event(), _attention_delivery=None, context=SimpleNamespace(audio_session_id=str(self.audio_id)),
            _response_admission_lock=asyncio.Lock(), _realtime_context_ready=asyncio.Event(), _turn_generation=1,
            _user_audio_pending=False, _normal_turn_pending=False, _pending_response=None,
            _active_responses={}, _native_playback={},
            _reserve_response_locked=self.reserve, _send_reserved_response=self.send_reserved,
            release_attention=self.release, _mark_closing=lambda reason: self.session._closed.set())
        self.session._realtime_context_ready.set()
        self.runtime = SimpleNamespace(
            ready=True, request=SimpleNamespace(route=RouteIdentity(scope=self.repository.scope, route_id=uuid4(), route_generation=1)),
            adapter=SimpleNamespace(capabilities=capabilities()),
            coordinator=SimpleNamespace(interrupt_speech=self.interrupt), offer_notification=self.offer,
            _fail=lambda reason: self.session._closed.set())
        self.sink = NativeAttentionSink(self.session, self.runtime, self.repository, self.admit, clock=lambda: NOW)
        self.sink.start()

    async def renew(self, claim):
        if claim != self.repository.claim:
            raise AttentionError("fixture_wrong_claim", "Foreign fixture claim.")

    async def settle_unprepared(self, claim, **kwargs):
        raise AssertionError("Unexpected unprepared fixture settlement")

    async def admit(self, repository, speech, **kwargs):
        if repository is not self.repository or speech is not None:
            raise AssertionError("Legacy speech provider crossed native admission")
        self.checks["no_legacy_speech_provider"] = True
        return self.delivery

    def reserve(self, kind, metadata):
        if kind != "prospective":
            raise AssertionError("Notification manufactured a user turn")
        self.reservations.append((kind, metadata))
        return SimpleNamespace(task_instructions=None)

    async def send_reserved(self, reservation):
        if not reservation.task_instructions:
            raise AssertionError("Missing normal notification instructions")
        self.writes.append("response")
        return True

    async def release(self, delivery):
        if self.session._attention_delivery is delivery:
            self.session._attention_delivery = None

    async def interrupt(self, command):
        self.fences.append(command)
        return CommandReceipt(route=command.route, command_id=command.command_id,
            operation="interrupt_speech", observed_at=NOW, status="completed", output_fenced=True)

    async def send(self, kind, **values):
        await self.socket.send(json.dumps({"type": kind, **values}))

    async def receive(self, kind):
        raw = await asyncio.wait_for(self.socket.recv(), 5)
        if len(raw) > 65536:
            raise ValueError("fixture_event_oversize")
        event = json.loads(raw)
        if not isinstance(event, dict) or event.get("type") != kind:
            raise ValueError("fixture_unexpected_control")
        UUID(event["client_event_id"])
        return event

    async def offer(self, offer):
        if self.case == "source_reset":
            self.repository.payloads[self.repository.claim.source_payload_id]["availability"] = "deleted"
        async with self.sink.guard(offer):
            self.writes.append("offer")
        if self.case == "priority_deduplication":
            self.session._normal_turn_pending = True
            admitted = await RealtimeAudioSession.request_route_notification(self.session, offer, turn_generation=1)
            self.checks["user_turn_priority"] = admitted is False and not self.reservations
            self.session._normal_turn_pending = False
        async with self.sink.guard(offer):
            admitted = await RealtimeAudioSession.request_route_notification(self.session, offer, turn_generation=1)
            if not admitted:
                raise AssertionError("Current single-voice notification refused")
        await self.send("fixture.notification_submitted", fixture_protocol=PROTOCOL, case=self.case)
        if self.case == "cancel_reconnect":
            await asyncio.Future()
        self.finished_evidence = DeliveryEvidence(
            evidence_id=uuid4(), route=offer.route, transport="native_voice", observed_at=NOW,
            state="output_finished", attribution="explicit", offer_id=offer.offer_id, attempt=offer.attempt,
            output=OutputIdentity(output_id=uuid4(), provider_response_ref="synthetic-response"))
        return self.finished_evidence

    async def handshake(self):
        await self.send("audio.provider.selected", provider="openai", protocol_version=1,
            model="controlled-transport-no-model", audio_session_id=str(self.audio_id))
        await self.send("fixture.protocol", fixture_protocol=PROTOCOL, case=self.case,
            native_model=False, native_tts=False, canonical_database=False)
        await self.send("session.created", session_id=str(self.session_id), audio_session_id=str(self.audio_id))

    def reset_fields(self):
        return dict(session_id=str(self.session_id), audio_session_id=str(self.audio_id),
            context_reset={"protocol_version": 1, "id": str(self.reset_id), "preserve_session": True})

    async def first_connection(self):
        self.delivery = self.sink.create_delivery(self.repository, None, self.repository.claim,
            self.session, receipt_seconds=1)
        self.session._attention_delivery = self.delivery
        await self.sink.try_attention(self.repository, object(), worker_id="controlled-worker", receipt_seconds=1)
        self.delivery.task = asyncio.create_task(self.delivery.run())
        if self.case == "cancel_reconnect":
            await self.receive("response.cancel")
            await self.delivery.stop("user_interrupted")
        await asyncio.wait_for(self.delivery.task, 5)
        await asyncio.wait_for(self.sink._events.join(), 1)
        if self.case == "priority_deduplication":
            self.sink.on_event(self.finished_evidence)
            await asyncio.wait_for(self.sink._events.join(), 1)
            self.checks["duplicate_evidence_no_resubmission"] = self.repository.operations.count("submit") == 1 and self.writes.count("response") == 1
        if self.case == "source_reset":
            self.checks["withdrawn_source_no_external_write"] = not self.writes
            await self.send("session.context_invalidated", **self.reset_fields())
            event = await self.receive("session.context_cleared")
            if (type(event.get("protocol_version")) is not int or event["protocol_version"] != 1
                    or UUID(event.get("reset_id", "")) != self.reset_id
                    or event.get("audio_session_id") != str(self.audio_id)):
                raise ValueError("fixture_reset_ack_binding")
            self.checks["matching_reset_ack"] = True
            self.waiting_reconnect = True
            await self.send("session.closed", **self.reset_fields())
            return None
        if self.case == "cancel_reconnect":
            self.checks["cancel_settled_interrupted"] = self.repository.attempt["phase"] == "interrupted"
            self.checks["owner_retired_after_fence"] = self.session._attention_delivery is None and len(self.fences) == 1
            self.waiting_reconnect = True
            await self.send("fixture.disconnect", fixture_protocol=PROTOCOL, case=self.case)
            return None
        return self.result()

    async def replacement_connection(self):
        # Same conversation, new audio route. Late old-attempt observations have
        # no authority to interrupt this unrelated output owner.
        new_owner = object()
        self.session._attention_delivery = new_owner
        self.runtime.request.route = RouteIdentity(scope=self.repository.scope, route_id=uuid4(), route_generation=2)
        before = len(self.fences)
        old = self.delivery.offer
        event = DeliveryEvidence(evidence_id=uuid4(), route=old.route, transport="native_voice",
            observed_at=NOW, state="unknown", attribution="explicit", offer_id=old.offer_id,
            attempt=old.attempt, reason="late_old_route_observation")
        rejected = False
        try:
            self.sink.on_event(event)
        except AttentionError as error:
            rejected = error.code == "attention_evidence_binding"
        self.checks["retired_route_cannot_affect_replacement"] = (
            rejected and self.session._attention_delivery is new_owner and len(self.fences) == before)
        self.checks["same_conversation"] = self.connections == 2
        if self.case == "source_reset":
            self.checks["withdrawn_result_not_replayed"] = not self.writes
        return self.result()

    def result(self):
        r = self.repository
        self.checks.update(
            one_submission=r.operations.count("submit") == 1,
            native_voice_plan=r.attempt["delivery_transport"] == "native_voice",
            no_legacy_chunks=not r.has_chunks,
            no_played_claim=r.intent["state"] == "unknown" and r.attempt["played_samples"] is None,
            no_manufactured_user_turn=self.session._normal_turn_pending is False)
        if self.case in {"completion", "priority_deduplication"}:
            self.checks["one_notification_response"] = self.writes == ["offer", "response"] and len(self.reservations) == 1
            self.checks["output_finished_not_hearing"] = self.delivery.output_finished and r.attempt["phase"] == "unknown"
        return dict(ok=all(self.checks.values()), fixture_protocol=PROTOCOL, case=self.case,
            checks=dict(self.checks), state=r.intent["state"], attempt_phase=r.attempt["phase"],
            connections=self.connections, external_response_requests=self.writes.count("response"),
            native_model=False, native_tts=False, canonical_database=False,
            task_execution="not_exercised", memory_effect="not_exercised",
            physical_playback_measured=False, human_hearing="unknown")

    async def close(self):
        if not self.closed:
            self.closed = True
            await self.sink.close()


class FixtureServer:
    def __init__(self):
        self.probes = {}
        self.results = {}
        self.complete = asyncio.Event()

    async def handle(self, socket):
        probe = None
        case = None
        try:
            if socket.request.headers.get("Authorization") != "Bearer fixture-token":
                await socket.close(code=1008)
                return
            target = urlsplit(socket.request.path)
            query = parse_qs(target.query, strict_parsing=True)
            if (target.path != "/single-voice" or set(query) - {"case", "probe_id", "audio_provider", "session_id"}
                    or any(len(values) != 1 for values in query.values())
                    or query.get("audio_provider") != ["openai"]):
                raise ValueError("fixture_route_binding")
            case = query["case"][0]
            if case not in CASES:
                raise ValueError("fixture_case")
            key = UUID(query["probe_id"][0])
            if key not in self.probes:
                if "session_id" in query:
                    raise ValueError("fixture_fresh_conversation_required")
                if len(self.probes) >= 4 or case in self.results or any(p.case == case for p in self.probes.values()):
                    raise ValueError("fixture_case_already_admitted")
                probe = self.probes[key] = Probe(case)
            else:
                probe = self.probes[key]
                if (probe.closed or probe.case != case or not probe.waiting_reconnect or probe.connections != 1
                        or query.get("session_id") != [str(probe.session_id)]):
                    raise ValueError("fixture_reconnect_binding")
                probe.waiting_reconnect = False
                probe.audio_id = uuid4()
                probe.session.context.audio_session_id = str(probe.audio_id)
            probe.socket = socket
            probe.connections += 1
            async with asyncio.timeout(12):
                await probe.handshake()
                result = await (probe.first_connection() if probe.connections == 1 else probe.replacement_connection())
            if result is not None:
                self.results[case] = result
                await probe.send("fixture.result", **result)
                await probe.close()
        except Exception as error:
            result = dict(ok=False, fixture_protocol=PROTOCOL, case=case, error_type=type(error).__name__,
                          native_model=False, native_tts=False, canonical_database=False)
            if case in CASES:
                self.results[case] = result
            with suppress(Exception):
                await socket.send(json.dumps({"type": "fixture.result", **result}))
            if probe is not None:
                await probe.close()
        finally:
            if len(self.results) == len(CASES):
                self.complete.set()

    async def close(self):
        await asyncio.gather(*(probe.close() for probe in self.probes.values()))


async def run(output: Path, timeout: float = 180):
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    fixture = FixtureServer()
    timed_out = False
    try:
        async with serve(fixture.handle, "127.0.0.1", 0, max_size=65536) as server:
            ready = {"ready": True, "url": f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/single-voice",
                     "fixture_protocol": PROTOCOL, "native_model": False, "canonical_database": False}
            (output / "ready.json").write_text(json.dumps(ready, indent=2) + "\n")
            print(json.dumps(ready), flush=True)
            try:
                await asyncio.wait_for(fixture.complete.wait(), timeout)
            except TimeoutError:
                timed_out = True
    finally:
        await fixture.close()
    result = {"ok": not timed_out and len(fixture.results) == len(CASES) and all(r["ok"] for r in fixture.results.values()),
              "timed_out": timed_out, "cases": fixture.results, "local_fixture_closed": True,
              "native_model": False, "canonical_database": False, "human_hearing": "unknown"}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result["ok"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(run(args.output)) else 1)
