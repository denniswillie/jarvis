"""Offline repository execution against transactional fixtures, not SQL acceptance.

These tests execute the production ledger and state mapping. The DB double
models owner transactions, payload lineage and exact SQL operation results; it
does not establish PostgreSQL migration/constraint correctness.
"""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
import unittest
from uuid import uuid4

from pydantic import ValidationError

from app.attention.models import AttentionError, DeliveryClaim
from app.attention.notifications import NotificationContent, NotificationOutput
from app.attention.notification_ledger.models import VoiceRoutePlan, settlement_for
from app.attention.notification_ledger.repository import VoiceNotificationLedger
from app.live_state.models import SessionScope
from app.voice_routes.contracts import (
    ClientPlaybackObservation, DeliveryEvidence, DeliveryEvidenceState as State,
    OperationCapability, OutputIdentity, RouteCapabilities, RouteIdentity,
    RouteOperation, Support, VoiceTransport,
)


NOW = datetime(2026, 10, 30, 16, 0, tzinfo=UTC)


def capabilities(*, playback=False, attribution=True, transport="native_voice"):
    return RouteCapabilities(transport=transport, media_topology="phone_to_provider",
        control_owner="backend", credential_mode="server_only", input_codec="opus", output_codec="opus",
        vad_owner="provider", explicit_input_commit="supported", explicit_response_create="supported",
        full_duplex="supported", image_input="unsupported", provider_context_receipts="supported",
        notification_output_attribution="supported" if attribution else "unsupported",
        client_playback_evidence="supported" if playback else "unsupported", provider_interruption="supported",
        operations=tuple(OperationCapability(operation=operation, support="supported") for operation in RouteOperation))


class Connection:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def tag(sql):
        return sql.split("voice_ledger:", 1)[1].split(" */", 1)[0]

    async def execute(self, sql, *args):
        tag = self.tag(sql)
        s = self.store
        s.operations.append(tag)
        if tag == "prepare":
            keys = ("plan_payload_id", "plan_digest", "delivery_transport", "voice_route_id",
                    "voice_route_generation", "voice_offer_id", "voice_audio_session_id")
            s.attempt.update(dict(zip(keys, args[2:])), voice_route_protocol_version=1)
        elif tag == "submit":
            s.attempt.update(phase="submitted", submitted_at=NOW)
        elif tag == "append":
            keys = ("owner_id", "evidence_id", "attempt_id", "voice_route_id", "voice_route_generation",
                    "offer_id", "output_id", "evidence_state", "observed_at", "evidence_payload_id",
                    "evidence_digest", "playback_receipt_id")
            item = dict(zip(keys, args))
            key = (item["owner_id"], item["evidence_id"])
            if key in s.evidence:
                raise AssertionError("duplicate DB evidence identity")
            s.evidence[key] = item
        elif tag == "settle_attempt":
            s.attempt.update(phase=args[2], reason=args[3], ended_at=NOW)
        elif tag == "settle_intent":
            if args[4] == s.intent["generation"]:
                s.intent.update(state=args[2], reason=args[3])
        else:
            raise AssertionError(f"Unsupported SQL execution: {tag}")
        return "OK"

    async def fetchrow(self, sql, *args):
        tag = self.tag(sql)
        if tag == "plan":
            return deepcopy(self.store.payloads.get(args[1]))
        if tag == "attempt":
            return deepcopy(self.store.attempt) if args[:2] == (
                self.store.attempt["owner_id"], self.store.attempt["id"]) else None
        if tag == "evidence_key":
            return deepcopy(self.store.evidence.get(tuple(args)))
        raise AssertionError(f"Unsupported SQL row: {tag}")

    async def fetchval(self, sql, *args):
        tag = self.tag(sql)
        if tag == "legacy_chunks":
            return self.store.has_chunks
        if tag == "receipt_key":
            return any((e["owner_id"], e["attempt_id"], e["playback_receipt_id"]) == args
                       for e in self.store.evidence.values())
        raise AssertionError(f"Unsupported SQL value: {tag}")

    async def fetch(self, sql, *args):
        if self.tag(sql) != "output_binding":
            raise AssertionError("Unsupported SQL rows")
        return [{"content": deepcopy(self.store.payloads[e["evidence_payload_id"]]["content"])}
            for e in self.store.evidence.values()
            if (e["owner_id"], e["attempt_id"], e["output_id"]) == args
            and e["evidence_state"] in {"output_started", "output_finished"}
            and self.store.payloads[e["evidence_payload_id"]]["availability"] == "available"]


class CanonicalFixture:
    """Minimal canonical attention authority; no alternate runtime implementation."""
    def __init__(self):
        self.scope = SessionScope(owner_id="Personal owner: offline", session_id=uuid4(), area_id=uuid4())
        self.claim = DeliveryClaim(self.scope.owner_id, uuid4(), uuid4(), 3, self.scope.session_id,
                                   "native-worker", uuid4(), "reminder_event")
        self.intent = dict(id=self.claim.intent_id, owner_id=self.claim.owner_id,
            generation=3, source_id=uuid4(), source_revision=1, source_namespace="reminder_event",
            source_payload_id=self.claim.source_payload_id, state="claimed", expires_at=NOW+timedelta(minutes=5))
        self.attempt = dict(id=self.claim.attempt_id, owner_id=self.claim.owner_id, intent_id=self.claim.intent_id,
            generation=3, session_id=self.claim.session_id, worker_id=self.claim.worker_id, channel="voice",
            phase="claimed", submitted_at=None, forwarded_samples=0, played_samples=None,
            plan_payload_id=None, plan_digest=None, delivery_transport="legacy_pcm_tts",
            voice_route_id=None, voice_route_generation=None, voice_offer_id=None,
            voice_route_protocol_version=None, voice_audio_session_id=None)
        self.output = NotificationOutput(NotificationContent(source_namespace="reminder_event",
            source_id=self.intent["source_id"], reminder_id=uuid4(), title="Check the umbrella", details="",
            message="Reminder: Check the umbrella"))
        self.payloads = {self.claim.source_payload_id: dict(availability="available", content={"original": True})}
        self.evidence = {}
        self.dependencies = {}
        self.operations = []
        self.lock = asyncio.Lock()
        self.connection = Connection(self)
        self.tasks = self
        self.commits = 0
        self.has_chunks = False
        self.source_problem = None
        self.lease_valid = True
        self.recovery_calls = []

    @asynccontextmanager
    async def transaction(self, owner):
        if owner != self.claim.owner_id:
            raise AttentionError("wrong_owner", "The fixture has no such owner")
        async with self.lock:
            snapshot = deepcopy((self.intent, self.attempt, self.payloads, self.evidence, self.dependencies))
            try:
                yield self.connection
            except BaseException:
                self.intent, self.attempt, self.payloads, self.evidence, self.dependencies = snapshot
                raise
            else:
                self.commits += 1

    async def payload(self, connection, owner, content, *, source_payload_ids=()):
        if not self.lock.locked():
            raise AssertionError("payload escaped transaction")
        for root in source_payload_ids:
            if self.payloads[root]["availability"] != "available":
                raise AttentionError("source_removed", "No late copy")
        identity = uuid4()
        self.payloads[identity] = dict(availability="available", content=deepcopy(content))
        self.dependencies[identity] = tuple(source_payload_ids)
        return identity

    async def _entry(self, connection, owner, intent_id, *, lock=False):
        if (owner, intent_id) != (self.claim.owner_id, self.claim.intent_id):
            raise AttentionError("attention_not_found", "No such intent")
        return deepcopy(self.intent), None

    async def _source(self, connection, row, task, *, lock=False):
        if self.payloads[self.claim.source_payload_id]["availability"] != "available":
            return None, "source_removed"
        return self.output, self.source_problem

    @asynccontextmanager
    async def guard(self, claim):
        async with self.transaction(claim.owner_id) as connection:
            if (claim != self.claim or self.intent["generation"] != claim.generation
                    or self.intent["state"] != "claimed" or not self.lease_valid):
                raise AttentionError("attention_claim_lost", "No current claim")
            output, problem = await self._source(connection, self.intent, None, lock=True)
            if problem:
                raise AttentionError("attention_source_unavailable", "Source unavailable")
            yield connection, deepcopy(self.intent), deepcopy(self.attempt), output

    async def _recover(self, connection, owner):
        self.recovery_calls.append((connection, owner))


class LedgerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = CanonicalFixture()
        self.ledger = VoiceNotificationLedger(self.f, clock=lambda: NOW)
        self.route = RouteIdentity(scope=self.f.scope, route_id=uuid4(), route_generation=7)
        self.options = dict(route=self.route, audio_session_id=uuid4(), capabilities=capabilities(),
            offer_id=uuid4(), expires_at=NOW+timedelta(minutes=1), submission_boundary="response.create.with_notification")

    async def prepare(self, **changes):
        return await self.ledger.prepare(self.f.claim, **(self.options | changes))

    def event(self, offer, state, **changes):
        return DeliveryEvidence(evidence_id=uuid4(), route=offer.route, transport="native_voice", observed_at=NOW,
            state=state, attribution="explicit", offer_id=offer.offer_id, attempt=offer.attempt, **changes)

    async def submitted(self, **changes):
        offer = await self.prepare(**changes)
        ticket = await self.ledger.commit_submission(self.f.claim, offer)
        return offer, ticket

    async def test_plan_uses_canonical_text_roots_and_real_ledger_attempt(self):
        offer = await self.prepare()
        self.assertEqual(offer.text, self.f.output.content.message)
        self.assertEqual(offer.attempt.attempt_id, self.f.claim.attempt_id)
        self.assertEqual(offer.attempt.attempt_generation, 3)
        self.assertEqual(offer.route.route_generation, 7)
        self.assertEqual(self.f.attempt["delivery_transport"], "native_voice")
        self.assertEqual(self.f.dependencies[self.f.attempt["plan_payload_id"]], (self.f.claim.source_payload_id,))
        self.assertEqual(self.f.attempt["forwarded_samples"], 0)
        self.assertIsNone(self.f.attempt["played_samples"])
        self.assertEqual(len(self.f.evidence), 1)

    async def test_preparation_replay_has_no_duplicate_payloads_or_evidence(self):
        first = await self.prepare()
        counts = (len(self.f.payloads), len(self.f.evidence))
        self.assertEqual(await self.prepare(), first)
        self.assertEqual(counts, (len(self.f.payloads), len(self.f.evidence)))
        with self.assertRaises(AttentionError):
            await self.prepare(offer_id=uuid4())

    async def test_commit_precedes_write_and_duplicate_does_not_issue_ticket(self):
        offer, ticket = await self.submitted()
        commits = self.f.commits
        self.assertIsNone(await self.ledger.commit_submission(self.f.claim, offer))
        async with self.ledger.guarded_write(ticket) as exposed:
            self.assertGreaterEqual(self.f.commits, commits)
            self.assertEqual(self.f.attempt["phase"], "submitted")
            self.assertEqual(exposed, offer)
        with self.assertRaises(AttentionError):
            async with self.ledger.guarded_write(ticket):
                self.fail("A used ticket was exposed again")

    async def test_other_ledger_instance_cannot_consume_ticket(self):
        _, ticket = await self.submitted()
        other = VoiceNotificationLedger(self.f, clock=lambda: NOW)
        with self.assertRaises(AttentionError):
            async with other.guarded_write(ticket):
                self.fail("Foreign ticket accepted")

    async def test_source_guard_rechecked_after_submission(self):
        _, ticket = await self.submitted()
        self.f.source_problem = "withdrawn"
        with self.assertRaises(AttentionError):
            async with self.ledger.guarded_write(ticket):
                self.fail("Withdrawn source exposed")
        await self.ledger.settle(self.f.claim, reason="source_withdrawn_after_submission")
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.assertEqual(self.f.intent["state"], "unknown")

    async def test_write_failure_settles_unknown_without_replay(self):
        offer, ticket = await self.submitted()
        with self.assertRaises(TimeoutError):
            async with self.ledger.guarded_write(ticket):
                raise TimeoutError("simulated provider write ambiguity")
        await self.ledger.settle(self.f.claim, reason="write_timeout")
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.assertEqual(self.f.intent["generation"], 3)
        with self.assertRaises(AttentionError):
            await self.ledger.commit_submission(self.f.claim, offer)

    async def test_output_finish_without_phone_proof_is_unknown(self):
        offer, _ = await self.submitted()
        output = OutputIdentity(output_id=uuid4(), provider_response_ref="actual-response")
        started = self.event(offer, State.OUTPUT_STARTED, output=output)
        await self.ledger.record(self.f.claim, started)
        result = await self.ledger.record(self.f.claim, self.event(offer, State.OUTPUT_FINISHED, output=output))
        self.assertEqual(result.attempt_phase, "unknown")
        self.assertEqual(result.hearing, "not_established")
        self.assertEqual(self.f.intent["state"], "unknown")
        self.assertEqual(self.f.attempt["reason"], "native_completion_unconfirmed")
        self.assertEqual(self.f.attempt["forwarded_samples"], 0)

    async def test_provider_receipt_is_not_playback(self):
        offer, _ = await self.submitted()
        result = await self.ledger.record(self.f.claim, self.event(offer, State.PROVIDER_RECEIVED))
        self.assertEqual(result.attempt_phase, "submitted")
        self.assertEqual(self.f.intent["state"], "claimed")

    async def test_even_client_render_lower_bound_cannot_set_played(self):
        offer, _ = await self.submitted(capabilities=capabilities(playback=True))
        output = OutputIdentity(output_id=uuid4())
        await self.ledger.record(self.f.claim, self.event(offer, State.OUTPUT_STARTED, output=output))
        playback = self.event(offer, State.CLIENT_PLAYBACK_OBSERVED, output=output,
            playback=ClientPlaybackObservation(receipt_id=uuid4(), observed_at=NOW, rendered_ms=120))
        result = await self.ledger.record(self.f.claim, playback)
        self.assertEqual(result.attempt_phase, "submitted")
        self.assertEqual(self.f.intent["state"], "claimed")
        self.assertIsNone(self.f.attempt["played_samples"])
        with self.assertRaises(AttentionError):
            await self.ledger.record(self.f.claim, playback.model_copy(update={"evidence_id": uuid4()}))

    async def test_native_render_is_rejected_if_unsupported(self):
        offer, _ = await self.submitted()
        output = OutputIdentity(output_id=uuid4())
        await self.ledger.record(self.f.claim, self.event(offer, State.OUTPUT_STARTED, output=output))
        with self.assertRaises(AttentionError):
            await self.ledger.record(self.f.claim, self.event(offer, State.CLIENT_PLAYBACK_OBSERVED,
                output=output, playback=ClientPlaybackObservation(receipt_id=uuid4(), observed_at=NOW)))

    async def test_playback_requires_preexisting_output_binding(self):
        offer, _ = await self.submitted(capabilities=capabilities(playback=True))
        with self.assertRaises(AttentionError):
            await self.ledger.record(self.f.claim, self.event(offer, State.CLIENT_PLAYBACK_OBSERVED,
                output=OutputIdentity(output_id=uuid4()),
                playback=ClientPlaybackObservation(receipt_id=uuid4(), observed_at=NOW)))

    async def test_explicit_output_attribution_cannot_be_invented_from_time(self):
        offer, _ = await self.submitted(capabilities=capabilities(attribution=False))
        with self.assertRaises(AttentionError):
            await self.ledger.record(self.f.claim, self.event(offer, State.OUTPUT_STARTED,
                output=OutputIdentity(output_id=uuid4())))
        unattributed = DeliveryEvidence(evidence_id=uuid4(), route=self.route, transport="native_voice",
            observed_at=NOW, state=State.OUTPUT_STARTED, output=OutputIdentity(output_id=uuid4()))
        with self.assertRaises(AttentionError):
            await self.ledger.record(self.f.claim, unattributed)

    async def test_evidence_dedupes_and_conflicting_reuse_rolls_back(self):
        offer, _ = await self.submitted()
        event = self.event(offer, State.PROVIDER_RECEIVED)
        self.assertTrue((await self.ledger.record(self.f.claim, event)).inserted)
        count = len(self.f.payloads)
        self.assertFalse((await self.ledger.record(self.f.claim, event)).inserted)
        with self.assertRaises(AttentionError):
            await self.ledger.record(self.f.claim, event.model_copy(update={"reason": "changed"}))
        self.assertEqual(len(self.f.payloads), count)

    async def test_wrong_route_attempt_generation_source_and_offer_are_rejected(self):
        offer, _ = await self.submitted()
        base = self.event(offer, State.PROVIDER_RECEIVED)
        cases = [
            {"route": self.route.model_copy(update={"route_generation": 8})},
            {"offer_id": uuid4()},
            {"attempt": offer.attempt.model_copy(update={"attempt_generation": 4})},
            {"attempt": offer.attempt.model_copy(update={"source_payload_id": uuid4()})},
            {"attempt": offer.attempt.model_copy(update={"source": offer.attempt.source.model_copy(update={"revision": 2})})},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(AttentionError):
                await self.ledger.record(self.f.claim, base.model_copy(update=changes))
        for change in ({"worker_id": "another"}, {"generation": 4}, {"session_id": uuid4()}):
            with self.subTest(claim=change), self.assertRaises(AttentionError):
                await self.ledger.record(replace(self.f.claim, **change), base)

    async def test_existing_evidence_replay_must_still_match_supplied_claim(self):
        # Regression for the known early replay return: same-owner evidence must
        # not be accepted while inspecting a different valid original attempt.
        offer, _ = await self.submitted()
        event = self.event(offer, State.PROVIDER_RECEIVED)
        await self.ledger.record(self.f.claim, event)
        original_claim = self.f.claim
        original_intent, original_attempt = deepcopy((self.f.intent, self.f.attempt))
        cases = [
            ({"attempt_id": uuid4()}, {}),
            ({"intent_id": uuid4()}, {}),
            ({"generation": 4}, {}),
            ({"session_id": uuid4()}, {}),
            ({"source_payload_id": uuid4()}, {}),
            ({"source_namespace": "task_result"}, {}),
            ({}, {"source_id": uuid4()}),
            ({}, {"source_revision": 2}),
        ]
        for claim_changes, source_changes in cases:
            with self.subTest(claim=claim_changes, source=source_changes):
                another = replace(original_claim, **claim_changes)
                self.f.claim = another
                self.f.intent = original_intent | dict(id=another.intent_id, generation=another.generation,
                    source_payload_id=another.source_payload_id, source_namespace=another.source_namespace) | source_changes
                self.f.attempt = original_attempt | dict(id=another.attempt_id, intent_id=another.intent_id,
                    generation=another.generation, session_id=another.session_id)
                before = deepcopy((self.f.intent, self.f.attempt, self.f.payloads,
                                   self.f.evidence, self.f.dependencies, self.f.operations))
                with self.assertRaises(AttentionError):
                    await self.ledger.record(another, event)
                self.assertEqual(before, (self.f.intent, self.f.attempt, self.f.payloads,
                                          self.f.evidence, self.f.dependencies, self.f.operations))

    async def test_identical_replay_survives_withdrawal_without_reviving_terminal_state(self):
        offer, _ = await self.submitted()
        event = self.event(offer, State.OUTPUT_FINISHED, output=OutputIdentity(output_id=uuid4()))
        await self.ledger.record(self.f.claim, event)
        self.assertEqual(self.f.attempt["phase"], "unknown")
        for payload in self.f.payloads.values():
            payload.update(availability="deleted", content=None)
        self.f.intent.update(generation=4, state="claimed")
        before = deepcopy((self.f.intent, self.f.attempt, self.f.payloads,
                           self.f.evidence, self.f.dependencies, self.f.operations))
        result = await self.ledger.record(self.f.claim, event)
        self.assertFalse(result.inserted)
        self.assertEqual(result.attempt_phase, "unknown")
        with self.assertRaises(AttentionError):
            await self.ledger.record(self.f.claim, event.model_copy(update={"reason": "conflicting-replay"}))
        self.assertEqual(before, (self.f.intent, self.f.attempt, self.f.payloads,
                                  self.f.evidence, self.f.dependencies, self.f.operations))

    async def test_late_evidence_does_not_settle_replacement_generation(self):
        offer, _ = await self.submitted()
        await self.ledger.settle(self.f.claim, reason="barge_in", interrupted=True)
        self.f.intent.update(generation=4, state="claimed")
        result = await self.ledger.record(self.f.claim, self.event(offer, State.OUTPUT_FINISHED,
            output=OutputIdentity(output_id=uuid4())))
        self.assertEqual(result.attempt_phase, "interrupted")
        self.assertEqual(self.f.intent["generation"], 4)
        self.assertEqual(self.f.intent["state"], "claimed")

    async def test_output_identity_is_immutable(self):
        offer, _ = await self.submitted()
        output = OutputIdentity(output_id=uuid4(), provider_item_ref="one")
        await self.ledger.record(self.f.claim, self.event(offer, State.OUTPUT_STARTED, output=output))
        with self.assertRaises(AttentionError):
            await self.ledger.record(self.f.claim, self.event(offer, State.OUTPUT_FINISHED,
                output=output.model_copy(update={"provider_item_ref": "two"})))

    async def test_provider_event_cannot_create_submission_retroactively(self):
        offer = await self.prepare()
        with self.assertRaises(AttentionError):
            await self.ledger.record(self.f.claim, self.event(offer, State.SUBMITTED))
        self.assertIsNone(self.f.attempt["submitted_at"])

    async def test_unsubmitted_failure_and_submitted_interrupt_have_distinct_settlement(self):
        offer = await self.prepare()
        await self.ledger.record(self.f.claim, self.event(offer, State.FAILED))
        self.assertEqual(self.f.attempt["phase"], "failed")
        self.assertEqual(self.f.intent["state"], "failed")

    async def test_generation_must_have_authoritative_validation(self):
        with self.assertRaises(AttentionError):
            await self.prepare(source_generation=2)
        self.assertIsNone(self.f.attempt["plan_payload_id"])
        calls = []
        async def current(connection, claim, source):
            calls.append(source)
            return source.generation == 2
        self.ledger.validate_source_generation = current
        offer, _ = await self.submitted(source_generation=2)
        self.assertEqual(offer.attempt.source.generation, 2)
        self.assertEqual(len(calls), 2)

    async def test_legacy_prepared_claim_cannot_be_converted_or_settled(self):
        self.f.attempt.update(plan_payload_id=uuid4(), plan_digest="0"*64)
        before = deepcopy(self.f.attempt)
        with self.assertRaises(AttentionError):
            await self.prepare()
        with self.assertRaises(AttentionError):
            await self.ledger.settle(self.f.claim, reason="not_ours")
        self.assertEqual(before, self.f.attempt)

    async def test_legacy_chunk_presence_blocks_conversion(self):
        self.f.has_chunks = True
        with self.assertRaises(AttentionError):
            await self.prepare()
        self.assertEqual(self.f.attempt["delivery_transport"], "legacy_pcm_tts")

    async def test_oversize_canonical_text_fails_without_silent_truncation(self):
        self.f.output = NotificationOutput(self.f.output.content.model_copy(update={"message": "x"*4097}))
        with self.assertRaises(ValidationError):
            await self.prepare()
        self.assertIsNone(self.f.attempt["plan_payload_id"])

    async def test_recovery_delegates_existing_policy_without_tts(self):
        await self.ledger.recover(self.f.claim.owner_id)
        self.assertEqual(self.f.recovery_calls, [(self.f.connection, self.f.claim.owner_id)])
        self.assertEqual(self.f.commits, 1)

    async def test_plan_root_cannot_be_omitted_or_substituted(self):
        await self.prepare()
        content = deepcopy(self.f.payloads[self.f.attempt["plan_payload_id"]]["content"])
        for roots in ([], [str(uuid4())]):
            with self.subTest(roots=roots), self.assertRaises(ValidationError):
                VoiceRoutePlan.model_validate(content | {"source_payload_root_ids": roots})


class PolicyAndSchemaTests(unittest.TestCase):
    def test_all_nonterminal_evidence_preserves_phase(self):
        for state in (State.OFFERED, State.SUBMITTED, State.PROVIDER_RECEIVED,
                      State.OUTPUT_STARTED, State.CLIENT_PLAYBACK_OBSERVED):
            with self.subTest(state=state):
                self.assertIsNone(settlement_for("submitted", submitted=True, state=state))

    def test_terminal_attempts_never_revive(self):
        for phase in ("played", "unknown", "failed", "interrupted", "revoked"):
            for state in State:
                with self.subTest(phase=phase, state=state):
                    self.assertIsNone(settlement_for(phase, submitted=True, state=state))

    def test_sql_is_additive_and_does_not_translate_native_into_pcm(self):
        # Static packaging guard only. PostgreSQL execution is a separate gate.
        sql = (Path(__file__).parents[1] / "app/migrations/sql/0039_attention_voice_routes.sql").read_text()
        self.assertIn("default 'legacy_pcm_tts'", sql)
        self.assertIn("primary key(owner_id,evidence_id)", sql)
        self.assertIn("references jarvis_attention_attempts(owner_id,id,voice_route_id,voice_route_generation,voice_offer_id)", sql)
        self.assertIn("phase not in ('forwarding','awaiting_receipt','played')", sql)
        self.assertNotIn("update jarvis_attention_intents", sql)
        self.assertNotIn("update jarvis_attention_attempts set", sql)

