"""No sockets, provider clients, SQL, simulator or Xcode: actual owner/ledger over controlled boundaries."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from urllib.parse import urlencode
from uuid import uuid4

from evals.second_brain.ios_single_voice import CASES, PROTOCOL, FixtureServer


class Socket:
    def __init__(self, case, probe_id, *, session_id=None, bearer="fixture-token", wrong_ack=False):
        query = {"case": case, "probe_id": str(probe_id), "audio_provider": "openai"}
        if session_id is not None:
            query["session_id"] = session_id
        self.request = SimpleNamespace(path="/single-voice?" + urlencode(query),
            headers={"Authorization": "Bearer " + bearer})
        self.sent = []
        self.incoming = asyncio.Queue()
        self.closed = None
        self.wrong_ack = wrong_ack

    async def send(self, raw):
        event = json.loads(raw)
        self.sent.append(event)
        if event["type"] == "fixture.notification_submitted" and event["case"] == "cancel_reconnect":
            self.incoming.put_nowait(json.dumps({"type": "response.cancel", "client_event_id": str(uuid4())}))
        if event["type"] == "session.context_invalidated":
            self.incoming.put_nowait(json.dumps({"type": "session.context_cleared", "client_event_id": str(uuid4()),
                "protocol_version": 1, "reset_id": str(uuid4()) if self.wrong_ack else event["context_reset"]["id"],
                "audio_session_id": event["audio_session_id"]}))

    async def recv(self):
        return await self.incoming.get()

    async def close(self, code=1000):
        self.closed = code


class SingleVoiceFixtureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = FixtureServer()

    async def asyncTearDown(self):
        await asyncio.wait_for(self.server.close(), 2)

    async def run_case(self, case):
        identity = uuid4()
        first = Socket(case, identity)
        await asyncio.wait_for(self.server.handle(first), 3)
        sockets = [first]
        if case in {"source_reset", "cancel_reconnect"} and case not in self.server.results:
            session = next(row["session_id"] for row in first.sent if row["type"] == "session.created")
            second = Socket(case, identity, session_id=session)
            await asyncio.wait_for(self.server.handle(second), 3)
            sockets.append(second)
        result = self.server.results[case]
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["native_model"])
        self.assertFalse(result["native_tts"])
        self.assertFalse(result["canonical_database"])
        self.assertFalse(result["physical_playback_measured"])
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(result["human_hearing"], "unknown")
        for socket in sockets:
            self.assertFalse(any(row["type"].startswith("audio.output.") for row in socket.sent))
        probe = self.server.probes[identity]
        self.assertTrue(probe.closed)
        self.assertTrue(probe.sink._worker.done())
        self.assertTrue(probe.delivery.task.done())
        return result, probe, sockets

    async def test_completion_uses_actual_ledger_and_notification_admission(self):
        result, probe, _ = await self.run_case("completion")
        self.assertEqual(probe.writes, ["offer", "response"])
        self.assertEqual(probe.repository.operations.count("submit"), 1)
        self.assertEqual(probe.repository.attempt["phase"], "unknown")
        self.assertTrue(probe.delivery.output_finished)
        self.assertEqual(probe.reservations[0][0], "prospective")
        self.assertTrue(result["checks"]["output_finished_not_hearing"])

    async def test_user_priority_and_duplicate_observation_do_not_resubmit(self):
        result, probe, _ = await self.run_case("priority_deduplication")
        self.assertTrue(result["checks"]["user_turn_priority"])
        self.assertTrue(result["checks"]["duplicate_evidence_no_resubmission"])
        self.assertEqual(len(probe.reservations), 1)

    async def test_interruption_reconnect_and_old_route_evidence_preserve_new_owner(self):
        result, probe, sockets = await self.run_case("cancel_reconnect")
        self.assertEqual(result["attempt_phase"], "interrupted")
        self.assertEqual(len(probe.fences), 1)
        self.assertTrue(result["checks"]["retired_route_cannot_affect_replacement"])
        self.assertEqual(len(sockets), 2)
        sessions = [next(e for e in s.sent if e["type"] == "session.created") for s in sockets]
        self.assertEqual(sessions[0]["session_id"], sessions[1]["session_id"])
        self.assertNotEqual(sessions[0]["audio_session_id"], sessions[1]["audio_session_id"])

    async def test_withdrawal_rechecks_actual_source_guard_and_requires_ack(self):
        result, probe, sockets = await self.run_case("source_reset")
        self.assertEqual(probe.writes, [])
        self.assertEqual(probe.repository.payloads[probe.repository.claim.source_payload_id]["availability"], "deleted")
        self.assertTrue(result["checks"]["matching_reset_ack"])
        self.assertTrue(result["checks"]["withdrawn_result_not_replayed"])
        self.assertEqual([e["type"] for e in sockets[0].sent][-2:],
                         ["session.context_invalidated", "session.closed"])
        self.assertNotEqual(result["memory_effect"], "succeeded")

    async def test_wrong_reset_ack_fails_without_reconnect_or_replay(self):
        identity = uuid4()
        socket = Socket("source_reset", identity, wrong_ack=True)
        await asyncio.wait_for(self.server.handle(socket), 3)
        self.assertFalse(self.server.results["source_reset"]["ok"])
        self.assertFalse(any(e["type"] == "session.closed" for e in socket.sent))
        self.assertTrue(self.server.probes[identity].closed)
        self.assertEqual(self.server.probes[identity].writes, [])

    async def test_wrong_session_reconnect_is_rejected(self):
        identity = uuid4()
        first = Socket("cancel_reconnect", identity)
        await asyncio.wait_for(self.server.handle(first), 3)
        second = Socket("cancel_reconnect", identity, session_id=str(uuid4()))
        await asyncio.wait_for(self.server.handle(second), 3)
        self.assertFalse(self.server.results["cancel_reconnect"]["ok"])
        self.assertFalse(any(e["type"] == "session.created" for e in second.sent))
        self.assertTrue(self.server.probes[identity].closed)

    async def test_missing_bearer_does_not_allocate_owner(self):
        socket = Socket("completion", uuid4(), bearer="wrong")
        await self.server.handle(socket)
        self.assertEqual(socket.closed, 1008)
        self.assertEqual(self.server.probes, {})
        self.assertEqual(socket.sent, [])

    async def test_four_cases_close_all_owned_workers_and_signal_completion(self):
        for case in sorted(CASES):
            await self.run_case(case)
        self.assertTrue(self.server.complete.is_set())
        self.assertEqual(set(self.server.results), CASES)
        self.assertTrue(all(probe.closed for probe in self.server.probes.values()))

    def test_swift_replacements_keep_current_control_guards_and_no_authored_tts(self):
        path = Path(__file__).resolve().parents[2] / "ios/Jarvis/JarvisAudioProviderTests/JarvisRealtimeClientTests.swift"
        source = path.read_text()
        self.assertNotIn("runNativeExpertFixture", source)
        start = source.index("    // These replace paid expert-authored TTS probes.")
        end = source.index("final class JarvisContextResetTests", start)
        current = source[start:end]
        for name in [
            "testSingleVoiceNotificationSettlesWithoutPlaybackClaim",
            "testSingleVoiceNotificationHonorsUserPriorityAndDeduplicates",
            "testSingleVoiceInterruptionAndReconnectFenceOldAttempt",
            "testSingleVoiceSourceResetRejectsWithdrawnResult",
        ]:
            self.assertIn("func " + name, current)
        for marker in ["sendResponseCancel()", "sendContextResetCleared", "pendingReset",
                       "retired_route_cannot_affect_replacement", 'components.path == "/single-voice"',
                       'case "audio.output.delta", "audio.output.done", "audio.output.clear":']:
            self.assertIn(marker, current)
        self.assertNotIn("AudioPlaybackService()", current)
        self.assertNotIn("enqueuePCM16", current)
        self.assertIn('event.values["state"] as? String, "unknown"', current)
        self.assertIn(PROTOCOL, current)


if __name__ == "__main__":
    unittest.main()
