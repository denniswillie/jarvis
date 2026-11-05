"""Offline Live receipt -> safe child drain -> existing runner evidence.

The actual Live session/adapter/protocol run over existing HTTP/socket/store
fixtures. No native app, service, SQL, model, or network is started here.
"""
import asyncio
import io
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.realtime import live_session
from evals.responsive_voice.frontend_host_qualification import (
    ChildOutputDrain, NATIVE_MEDIA, NO_MEDIA, configure_native_diagnostic_logging,
    native_route_observation, sanitize_live_finalization, sanitized_live_finalization_diagnostic,
)
from evals.responsive_voice.native_replay_run import provider_evidence, run_once
from tests import test_live_session as session_fixtures
from tests import test_native_replay_run as runner_fixtures


LOGGER = "app.realtime.live_session.finalization"
PRIVATE = "Bearer sk-private https://private.invalid/raw private-context-audio"
AUDIO = "audio_offline_finalization"
SESSION = str(uuid4())
ROUTE = str(uuid4())


def diagnostic(**changes):
    return {
        "schema_version": 1, "stage": "live_finalization", "voice_route": "live",
        "audio_session_id": AUDIO, "jarvis_session_id": SESSION,
        "route_id": ROUTE, "route_generation": 1, "provider_session_id": "live_fixture",
        "creation_submitted": True, "session_closed_received": True,
        "receipt_kind": "session.closed", "finalization_outcome": "confirmed",
        "reason": "close_requested", "usage_seconds": 1.25, "shutdown_elapsed_ms": 2.5,
        "remote_closure_confirmed": True, **changes,
    }


def line(value):
    return "INFO " + LOGGER + " live_finalization_diagnostic=" + json.dumps(value)


def media_event(**changes):
    return {"type": "live.media.prepare", "provider_session_id": "live_fixture",
            "audio_session_id": AUDIO, "route_id": ROUTE, "route_generation": 1, **changes}


def evidence(records=(), *, events=None, dropped=0, realtime=()):
    return provider_evidence(realtime, True, voice_route="live",
        native_events=[media_event()] if events is None else events,
        live_finalizations=records, live_finalizations_dropped=dropped)


async def drain_lines(lines, *, cap=256):
    """Actual bounded pipe reader with in-memory process streams, never a server."""
    stdout, stderr = asyncio.StreamReader(), asyncio.StreamReader()
    child = SimpleNamespace(pid=123, stdout=stdout, stderr=stderr)
    log = io.BytesIO()
    drain = ChildOutputDrain(child, log, byte_limit=cap)
    stdout.feed_data(b"x"*70_000+b"\n")
    for value in lines:
        data = (value+"\n").encode()
        # Exercise chunk boundaries independently from complete JSON lines.
        stdout.feed_data(data[:31])
        stdout.feed_data(data[31:])
    stdout.feed_eof()
    stderr.feed_eof()
    result = await drain.wait_closed()
    return result, log.getvalue()


class LiveFinalizationProjectionTests(unittest.TestCase):
    def test_closed_schema_retains_typed_evidence_not_raw_provider_material(self):
        result = sanitized_live_finalization_diagnostic(line(diagnostic(
            context=PRIVATE, audio=PRIVATE, url=PRIVATE, headers={"Authorization": PRIVATE},
            api_key=PRIVATE, body=PRIVATE, exception_text=PRIVATE)))
        self.assertEqual(result, diagnostic())
        self.assertNotIn(PRIVATE, json.dumps(result))
        self.assertTrue(result["remote_closure_confirmed"])

    def test_identity_validation_is_bounded_and_confirmation_fails_closed(self):
        for key in ("audio_session_id", "provider_session_id", "jarvis_session_id", "route_id"):
            for value in (None, True, [], {}, PRIVATE, "sk-key", "SK-key", "x"*257):
                with self.subTest(field=key, value_type=type(value).__name__):
                    result = sanitize_live_finalization(diagnostic(**{key:value}))
                    self.assertIsNone(result[key])
                    self.assertFalse(result["remote_closure_confirmed"])
                    self.assertEqual(result["finalization_outcome"], "unknown")
                    self.assertNotIn(PRIVATE, json.dumps(result))
        for value in (True, 0, -1, 2**31, "1", [], {}):
            with self.subTest(generation=value):
                result = sanitize_live_finalization(diagnostic(route_generation=value))
                self.assertIsNone(result["route_generation"])
                self.assertFalse(result["remote_closure_confirmed"])

    def test_flags_kind_and_outcome_cannot_be_coerced_into_confirmation(self):
        for key in ("creation_submitted", "session_closed_received"):
            for value in (False, 1, 0, "true", [], {}, None):
                with self.subTest(field=key, value=value):
                    result = sanitize_live_finalization(diagnostic(**{key:value}))
                    self.assertFalse(result["remote_closure_confirmed"])
                    self.assertIs(result[key], False if value is False else None)
        for key in ("receipt_kind", "finalization_outcome"):
            for value in ("HTTP200", "closed", True, [], {}, None):
                with self.subTest(field=key, value=value):
                    self.assertFalse(sanitize_live_finalization(diagnostic(**{key:value}))["remote_closure_confirmed"])
        self.assertTrue(sanitize_live_finalization(diagnostic(remote_closure_confirmed=False))["remote_closure_confirmed"])

    def test_reason_enum_is_allowlisted_and_not_arbitrary_exception_text(self):
        for value in ("close_requested", "expired", "content", "remote_hangup", "connection_lost", "unknown"):
            with self.subTest(reason=value):
                self.assertTrue(sanitize_live_finalization(diagnostic(reason=value))["remote_closure_confirmed"])
        for value in (PRIVATE, [], {}, True, None, "provider_not_requested", "receipt_unavailable"):
            with self.subTest(reason=value):
                result = sanitize_live_finalization(diagnostic(reason=value))
                self.assertFalse(result["remote_closure_confirmed"])
                self.assertNotIn(PRIVATE, json.dumps(result))

    def test_usage_and_elapsed_are_finite_bounded_measurements(self):
        for key, bound in (("usage_seconds", 86_400), ("shutdown_elapsed_ms", 120_000)):
            for value in (0, 1.25, bound):
                with self.subTest(field=key, value=value):
                    self.assertEqual(sanitize_live_finalization(diagnostic(**{key:value}))[key], value)
            for value in (True, -1, bound+1, float("nan"), float("inf"), "1", [], {}, None):
                with self.subTest(field=key, invalid=type(value).__name__):
                    self.assertIsNone(sanitize_live_finalization(diagnostic(**{key:value}))[key])

    def test_parser_rejects_wrong_logger_duplicate_keys_oversize_and_malformed_json(self):
        candidates = (
            None, b"bytes", line(diagnostic()).replace(LOGGER, "app.realtime.webrtc"),
            line(diagnostic()).replace(LOGGER, "prefix"+LOGGER),
            "INFO "+LOGGER+" live_finalization_diagnostic={",
            "INFO "+LOGGER+' live_finalization_diagnostic={"schema_version":1,"schema_version":1}',
            line(diagnostic(extra="x"*8192)), line(diagnostic(extra="\u20ac"*3000)).replace("\\u20ac","\u20ac"),
            "\ud800", "INFO "+LOGGER+" live_finalization_diagnostic="+("["*2000)+("]"*2000),
        )
        for value in candidates:
            with self.subTest(kind=type(value).__name__, size=len(value) if value is not None else None):
                self.assertIsNone(sanitized_live_finalization_diagnostic(value))
        for changes in ({"schema_version":True}, {"schema_version":2}, {"stage":[]}, {"voice_route":"realtime"}):
            with self.subTest(changes=changes):
                self.assertIsNone(sanitize_live_finalization(diagnostic(**changes)))

    def test_local_close_http_success_and_claimed_boolean_are_not_provider_receipts(self):
        raw = diagnostic(session_closed_received=False, receipt_kind=None, finalization_outcome="unknown",
                         http_status=200, socket_closed=True, app_closed=True, shutdown_confirmed=True)
        result = evidence([raw], realtime=[{"call_id":"rtc_old","http_status":200,"closure_outcome":"accepted"}])
        self.assertEqual(result["finalization_outcome"], "unknown")
        self.assertFalse(result["remote_closure_confirmed"])
        self.assertNotIn("rtc_old", json.dumps(result))
        self.assertFalse(result["realtime_hangup_applicable"])

    def test_bound_duplicate_receipt_is_idempotent(self):
        result = evidence([diagnostic(), diagnostic()])
        self.assertTrue(result["remote_closure_confirmed"])
        self.assertEqual(result["live_finalization_diagnostics"], [diagnostic()])
        self.assertEqual(result["provider_session_ids"], ["live_fixture"])
        self.assertFalse(result["provider_attempt_budget_violated"])

    def test_stale_audio_route_generation_and_conflicting_bindings_cannot_confirm(self):
        for changes in ({"audio_session_id":"other_audio"}, {"route_id":str(uuid4())},
                        {"route_generation":2}, {"provider_session_id":"live_foreign"}):
            with self.subTest(changes=changes):
                result = evidence([diagnostic(**changes)])
                self.assertFalse(result["remote_closure_confirmed"])
                self.assertEqual(result["finalization_outcome"], "unknown")
        for changes in ({"jarvis_session_id":str(uuid4())}, {"usage_seconds":3},
                        {"session_closed_received":False,"finalization_outcome":"unknown"}):
            with self.subTest(conflicting=changes):
                result = evidence([diagnostic(), diagnostic(**changes)])
                self.assertFalse(result["remote_closure_confirmed"])
                self.assertEqual(result["finalization_binding_conflicts"], ["live_fixture"])

    def test_second_created_session_is_counted_even_without_media_prepare(self):
        second = diagnostic(provider_session_id="live_other", audio_session_id="other_audio",
                            route_id=str(uuid4()))
        result = evidence([diagnostic(), second])
        self.assertTrue(result["provider_attempt_budget_violated"])
        self.assertEqual(result["provider_session_ids"], ["live_fixture","live_other"])
        self.assertFalse(result["remote_closure_confirmed"])

    def test_missing_invalid_unbound_or_lost_evidence_remains_unknown(self):
        self.assertEqual(evidence()["finalization_outcome"], "unavailable")
        for records, dropped in (([diagnostic(), {}],0), ([diagnostic()],1),
                                 ([diagnostic()],True), ([diagnostic()],-1),
                                 ([diagnostic()]*17,0),
                                 ([diagnostic(),diagnostic(provider_session_id=None)],0)):
            with self.subTest(dropped=dropped, records=len(records)):
                result = evidence(records,dropped=dropped)
                self.assertFalse(result["remote_closure_confirmed"])
                self.assertEqual(result["finalization_outcome"], "unknown")
        self.assertEqual(len(evidence([diagnostic()]*17)["live_finalization_diagnostics"]),1)

    def test_no_creation_is_distinct_from_unknown_creation_and_before_prepare_close(self):
        raw = diagnostic(creation_submitted=False,session_closed_received=False,receipt_kind=None,
                         provider_session_id=None,finalization_outcome="not_requested",reason="provider_not_requested")
        result = evidence([raw],events=[])
        self.assertEqual(result["finalization_outcome"],"not_requested")
        self.assertFalse(result["remote_closure_confirmed"])
        self.assertTrue(evidence([diagnostic()],events=[])["remote_closure_confirmed"])

    def test_realtime_projection_ignores_live_receipts_and_never_confirms_http200(self):
        records = [{"call_id":"rtc_fixture","stage":"hangup_http","closure_outcome":"accepted","http_status":200}]
        before = provider_evidence(records,True)
        after = provider_evidence(records,True,live_finalizations=[diagnostic()],live_finalizations_dropped=1)
        self.assertEqual(after,before)
        self.assertFalse(after["remote_closure_confirmed"])
        self.assertEqual(after["closure_by_call"],{"rtc_fixture":"accepted"})

    def test_native_logger_opt_in_does_not_enable_global_or_network_debug(self):
        names = ("", "app.realtime", LOGGER, "httpx", "httpcore", "websockets.client")
        loggers = {name:logging.getLogger(name) for name in names}
        original = {name:logger.level for name,logger in loggers.items()}
        for name,logger in loggers.items():
            self.addCleanup(logger.setLevel,original[name])
            logger.setLevel(logging.WARNING)
        configure_native_diagnostic_logging(NO_MEDIA)
        self.assertTrue(all(logger.level==logging.WARNING for logger in loggers.values()))
        configure_native_diagnostic_logging(NATIVE_MEDIA)
        self.assertEqual(loggers[LOGGER].level,logging.INFO)
        self.assertTrue(all(logger.level==logging.WARNING for name,logger in loggers.items() if name!=LOGGER))


class LiveFinalizationSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = session_fixtures.LiveSessionTests("test_actual_composition_gates_media_and_joins_single_readers")
        await self.fixture.asyncSetUp()
        self.addAsyncCleanup(self.fixture.asyncTearDown)
        self.lines = []
        logger = logging.getLogger(LOGGER)
        level = logger.level
        self.addCleanup(logger.setLevel,level)
        logger.setLevel(logging.INFO)
        handler = logging.Handler()
        handler.emit = lambda record:self.lines.append("INFO "+record.name+" "+record.getMessage())
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler,handler)

    def observations(self):
        return [native_route_observation(message) for message in self.fixture.phone.messages]

    async def project(self):
        drained, log = await drain_lines(self.lines)
        self.assertTrue(drained["complete"])
        self.assertFalse(drained["errors"])
        self.assertTrue(drained["truncated"])
        self.assertLessEqual(len(log),256)
        self.assertTrue(all(drained["drained_to_eof"].values()))
        result = provider_evidence(drained["provider_diagnostics"],True,voice_route="live",
            native_events=self.observations(),
            live_finalizations=drained["live_finalization_diagnostics"],
            live_finalizations_dropped=drained["live_finalization_diagnostics_dropped"])
        return result, drained

    async def test_actual_provider_close_receipt_survives_log_cap_and_single_joined_shutdown(self):
        session = await self.fixture.start()
        await session.close("offline_close")
        await session.close("duplicate_close")
        result, drained = await self.project()
        self.assertEqual(len(self.lines),1)
        self.assertEqual(len(self.fixture.wire.requests),1)
        self.assertEqual(sum(item["type"]=="session.close" for item in self.fixture.wire.socket.sent),1)
        self.assertIsNotNone(session.adapter.finalization)
        self.assertEqual(result["finalization_outcome"],"confirmed")
        self.assertTrue(result["remote_closure_confirmed"])
        self.assertEqual(result["provider_session_ids"],["live_fixture"])
        self.assertEqual(drained["provider_diagnostics"],[])
        self.assertFalse(result["provider_attempt_budget_violated"])
        self.assertNotIn("fixture-secret",json.dumps(result))
        self.assertNotIn("answer",json.dumps(result["live_finalization_diagnostics"]))

    async def test_local_shutdown_without_session_closed_receipt_stays_unknown(self):
        session = await self.fixture.start()
        self.fixture.wire.socket.auto_close = False
        session.adapter._io_timeout = .02
        await session.close("offline_no_receipt")
        result, _ = await self.project()
        self.assertTrue(session.shutdown_confirmed)
        self.assertTrue(self.fixture.phone.closed and self.fixture.wire.socket.closed)
        self.assertIsNone(session.adapter.finalization)
        self.assertEqual(result["finalization_outcome"],"unknown")
        self.assertFalse(result["remote_closure_confirmed"])
        self.assertEqual(len(self.fixture.wire.requests),1)

    async def test_provider_initiated_finalization_is_reported_without_another_close_or_creation(self):
        session = await self.fixture.start()
        self.fixture.wire.socket.incoming.put_nowait(json.dumps({
            "type":"session.closed","event_id":"spontaneous_closed",
            "session":{"id":"live_fixture"},"reason":"remote_hangup","usage":{"seconds":2.25},
            "private_context":PRIVATE}))
        await self.fixture.until(lambda:session.adapter.finalization is not None)
        await session.close("join_provider_close")
        result, _ = await self.project()
        self.assertTrue(result["remote_closure_confirmed"])
        self.assertEqual(result["live_finalization_diagnostics"][0]["reason"],"remote_hangup")
        self.assertEqual(result["live_finalization_diagnostics"][0]["usage_seconds"],2.25)
        self.assertEqual(sum(item["type"]=="session.close" for item in self.fixture.wire.socket.sent),0)
        self.assertEqual(len(self.fixture.wire.requests),1)
        self.assertNotIn(PRIVATE,json.dumps(result))

    async def test_wrong_provider_session_closed_cannot_become_finalization(self):
        session = await self.fixture.start()
        self.fixture.wire.socket.auto_close = False
        session.adapter._io_timeout = .02
        self.fixture.wire.socket.incoming.put_nowait(json.dumps({
            "type":"session.closed","event_id":"foreign_closed",
            "session":{"id":"foreign_session"},"reason":"close_requested"}))
        await session.close("offline_wrong_binding")
        result, _ = await self.project()
        self.assertIsNone(session.adapter.finalization)
        self.assertFalse(result["remote_closure_confirmed"])
        self.assertNotIn("foreign_session",json.dumps(result))

    async def test_malformed_provider_usage_is_not_an_authenticated_finalization_receipt(self):
        session = await self.fixture.start()
        self.fixture.wire.socket.auto_close = False
        session.adapter._io_timeout = .02
        self.fixture.wire.socket.incoming.put_nowait(json.dumps({
            "type":"session.closed","event_id":"invalid_usage",
            "session":{"id":"live_fixture"},"reason":"close_requested","usage":{"seconds":True}}))
        await session.close("offline_invalid_receipt")
        result, _ = await self.project()
        self.assertIsNone(session.adapter.finalization)
        self.assertEqual(result["finalization_outcome"],"unknown")

    async def test_phone_forged_closed_event_never_enters_authenticated_reader(self):
        session = await self.fixture.start()
        self.fixture.wire.socket.auto_close = False
        session.adapter._io_timeout = .02
        self.fixture.phone.incoming.put_nowait({
            "type":"session.closed","event_id":"forged_phone","session":{"id":"live_fixture"},
            "reason":"close_requested"})
        await session.close("offline_forged_receipt")
        result, _ = await self.project()
        self.assertIsNone(session.adapter.finalization)
        self.assertFalse(result["remote_closure_confirmed"])

    async def test_unstarted_session_reports_not_requested_without_provider_work(self):
        session = await self.fixture.make()
        await session.close("offline_not_admitted")
        result, _ = await self.project()
        self.assertEqual(result["finalization_outcome"],"not_requested")
        self.assertFalse(result["remote_closure_confirmed"])
        self.assertFalse(self.fixture.wire.requests)

    async def test_runner_consumes_actual_adapter_receipt_after_host_join_without_extra_launch(self):
        session = await self.fixture.start()
        await session.close("offline_runner_boundary")
        _, drained = await self.project()
        observations = self.observations()
        class Host(runner_fixtures.Fixture):
            async def serve(self,manifest,stop,ready):
                result = await super().serve(manifest,stop,ready)
                result["child_output_drain"] = drained
                result["child"]["websocket_trace"][0]["events"] = observations
                return result
        with tempfile.TemporaryDirectory() as directory:
            fixture = Host(Path(directory))
            fixture.live_baseline()
            result = await run_once(fixture.options,host=fixture,process=fixture)
            self.assertEqual(result["status"],"lifecycle_complete",result.get("fatal"))
            self.assertEqual((fixture.launches,fixture.worker_starts,fixture.cleanups),(1,0,1))
            self.assertTrue(result["provider_evidence"]["remote_closure_confirmed"])
            self.assertEqual(result["provider_evidence"]["finalization_outcome"],"confirmed")
            self.assertTrue(result["local_cleanup_complete"])
            self.assertFalse(result["voice_pass"])
            self.assertLess(fixture.events.index("canonical_export"),fixture.events.index("host_stopped"))
            self.assertLess(fixture.events.index("host_stopped"),fixture.events.index("fixture_cleanup"))


class LiveFinalizationDrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_record_cap_is_explicit_and_prevents_global_closure_claim(self):
        result, _ = await drain_lines([line(diagnostic(shutdown_elapsed_ms=i)) for i in range(20)])
        self.assertEqual(len(result["live_finalization_diagnostics"]),16)
        self.assertEqual(result["live_finalization_diagnostics_dropped"],4)
        self.assertTrue(result["complete"])
        projected = evidence(result["live_finalization_diagnostics"],
            dropped=result["live_finalization_diagnostics_dropped"])
        self.assertFalse(projected["remote_closure_confirmed"])
        self.assertEqual(projected["finalization_outcome"],"unknown")
