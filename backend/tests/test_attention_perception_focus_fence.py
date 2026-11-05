"""Persistent applicability through real attention/ledger code, offline SQL double.

No PostgreSQL concurrency or migration acceptance is claimed. The fixture models
canonical rows and transactions; assertions cover guard reuse, exact bindings,
lock requests and same-connection output, not a replacement attention authority.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.attention.models import AttentionError
from app.attention.notifications import NotificationOutput, VisualNotificationContent
from app.attention.notification_ledger.repository import VoiceNotificationLedger
from app.attention.repository import AttentionRepository
from app.attention.visual import visual_source
from app.tasks.models import digest
from app.voice_routes.contracts import RouteIdentity
from test_notification_ledger import CanonicalFixture, Connection, NOW, capabilities


DEFAULT_FOCUS = object()


class VisualConnection(Connection):
    def __init__(self, store):
        super().__init__(store)
        self.subject_reads = []

    async def fetchrow(self, sql, *args):
        if "voice_ledger:" in sql:
            return await super().fetchrow(sql, *args)
        s = self.store
        if "from jarvis_attention_visual_subjects" in sql:
            self.subject_reads.append((args, "for share" in sql, s.lock.locked()))
            return deepcopy(s.subjects.get(tuple(args)))
        if "from jarvis_context_payloads" in sql:
            if args[0] != s.claim.owner_id:
                return None
            payload = s.payloads.get(args[1])
            if payload is None:
                return None
            return deepcopy(payload) | {"id": args[1]}
        if "from jarvis_attention_intents" in sql:
            return deepcopy(s.intent) if args[:2] == (s.claim.owner_id, s.claim.intent_id) else None
        if "from jarvis_attention_attempts" in sql:
            a = s.attempt
            matches = args == (s.claim.owner_id, a["id"], a["intent_id"], a["generation"],
                a["session_id"], a["worker_id"], ["claimed", "submitted", "forwarding", "awaiting_receipt"])
            return deepcopy(a) if matches and a["phase"] in args[-1] and s.lease_valid else None
        raise AssertionError(f"Unsupported row query: {sql}")

    async def fetchval(self, sql, *args):
        if "voice_ledger:" in sql:
            return await super().fetchval(sql, *args)
        s = self.store
        if sql == "select $1::timestamptz<=clock_timestamp()":
            return args[0] <= NOW
        if "insert into jarvis_attention_intents" in sql:
            owner, namespace, source, payload, session, expiry, coalesce = args
            s.intent.update(owner_id=owner, source_namespace=namespace, source_id=source,
                source_revision=1, source_payload_id=payload, preferred_session_id=session,
                expires_at=expiry, coalesce_key=coalesce, task_id=None, task_revision=None,
                state="pending", generation=1, attempt_count=0, reason=None)
            s.claim = replace(s.claim, source_payload_id=payload, source_namespace=namespace, generation=1)
            s.attempt["phase"] = "failed"
            return s.intent["id"]
        if "from jarvis_sessions" in sql:
            return s.session_active and args == (s.claim.owner_id, s.claim.session_id)
        if "join jarvis_visual_delivery_sources" in sql:
            owner, event_id, message, kind, payload, session = args
            return s.event_available and (owner, event_id, message, kind, payload, session) == (
                s.claim.owner_id, s.event["id"], s.event["message"], s.kind, s.observation, s.scope.session_id)
        if "insert into jarvis_attention_attempts" in sql:
            owner, intent, generation, session, worker, _lease = args
            s.attempt.update(id=uuid4(), owner_id=owner, intent_id=intent, generation=generation,
                session_id=session, worker_id=worker, phase="claimed")
            return s.attempt["id"]
        if "from jarvis_attention_attempts" in sql:
            return s.attempt["phase"] in args[1]
        raise AssertionError(f"Unsupported scalar query: {sql}")

    async def fetch(self, sql, *args):
        if "voice_ledger:" in sql:
            return await super().fetch(sql, *args)
        s = self.store
        if "from visual_frames" in sql:
            owner, session, frames = args
            return [dict(id=s.frame, captured_at=NOW, server_received_at=NOW)] if (
                owner, session, frames) == (s.claim.owner_id, s.scope.session_id, [s.frame]) else []
        # Match the outer candidate query before its nested attempts subquery.
        if " ".join(sql.split()).startswith("select id from jarvis_attention_intents"):
            eligible = {"pending", "failed", "unknown"} if "source_namespace=any" in sql else {"pending"}
            return [{"id": s.intent["id"]}] if s.intent["state"] in eligible else []
        if "from jarvis_attention_attempts" in sql:
            return []  # No expired leases in these focused applicability cases.
        raise AssertionError(f"Unsupported rows query: {sql}")

    async def execute(self, sql, *args):
        if "voice_ledger:" in sql:
            return await super().execute(sql, *args)
        s = self.store
        if "insert into jarvis_context_payload_dependencies" in sql:
            owner, source, dependent = args
            assert owner == s.claim.owner_id
            s.dependencies[dependent] = (source,)
        elif "update jarvis_attention_intents" in sql:
            if "state='claimed'" in sql:
                s.intent.update(state="claimed", attempt_count=s.intent["attempt_count"] + 1)
            elif "reason=$4" in sql:
                s.intent.update(state=args[2], reason=args[3])
            elif "reason=$3" in sql:
                s.intent.update(reason=args[2], state=args[3] if len(args) == 4 else "expired")
            else:
                raise AssertionError(f"Unsupported intent update: {sql}")
            if "generation=generation+1" in sql:
                s.intent["generation"] += 1
        else:
            raise AssertionError(f"Unexpected write (including subject settlement): {sql}")
        return "OK"


class VisualFixture(CanonicalFixture):
    def __init__(self, *, kind="visual", reminder=False):
        super().__init__()
        self.pool = SimpleNamespace()
        self.connection = VisualConnection(self)
        self.kind = kind
        self.frame, self.observation, self.focus_id = uuid4(), uuid4(), uuid4()
        self.subjects = {}
        self.focus = dict(owner_id=self.claim.owner_id, subject_type="visual_focus",
            subject_id=self.focus_id, generation=7, state="active", expires_at=NOW+timedelta(minutes=4),
            claim_token=uuid4(), completed_event_id=None)
        self.subjects[self.claim.owner_id, "visual_focus", self.focus_id] = self.focus
        self.reference = {"subject_id": str(self.focus_id), "generation": 7}
        self.event = dict(id=uuid4(), session_id=self.scope.session_id, message="A possible target is visible.",
            visual_frame_id=self.frame, visual_focus_id=self.focus_id, reminder_id=None,
            claim_token=self.focus["claim_token"], evidence={"supporting_frame_ids": [str(self.frame)]})
        if reminder:
            identity = uuid4()
            self.tracked = self.focus | dict(subject_type="conditional_reminder", subject_id=identity)
            self.subjects[self.claim.owner_id, "conditional_reminder", identity] = self.tracked
            self.event.update(visual_focus_id=None, reminder_id=identity)
        else:
            self.tracked = self.focus
        self.event_available = self.session_active = True
        self.repository = AttentionRepository(self)

    async def enqueue(self, *, reference=DEFAULT_FOCUS):
        options = {} if reference is False else {
            "perception_focus": self.reference if reference is DEFAULT_FOCUS else reference}
        async with self.transaction(self.claim.owner_id) as connection:
            return await self.repository.enqueue_visual(connection, owner_id=self.claim.owner_id,
                kind=self.kind, event=self.event, observation_payload_id=self.observation,
                max_age_seconds=300, **options)

    async def claim_next(self):
        claim = await self.repository.claim_next(self.claim.owner_id, self.claim.session_id, "focus-fixture")
        if claim is not None:
            self.claim = claim
        return claim

    def ledger(self):
        return VoiceNotificationLedger(self.repository, clock=lambda: NOW)

    async def prepare(self, ledger):
        return await ledger.prepare(self.claim,
            route=RouteIdentity(scope=self.scope, route_id=uuid4(), route_generation=1),
            audio_session_id=uuid4(), capabilities=capabilities(), offer_id=uuid4(),
            expires_at=NOW+timedelta(minutes=1), submission_boundary="response.create.with_notification")


class PerceptionFocusFenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_producers_persist_exact_reference_and_original_lineage(self):
        for kind in ("visual", "prospective"):
            with self.subTest(kind=kind):
                f = VisualFixture(kind=kind)
                await f.enqueue()
                body = f.payloads[f.claim.source_payload_id]["content"]
                self.assertEqual(body["perception_focus"], f.reference)
                self.assertEqual(f.dependencies[f.claim.source_payload_id], (f.observation,))
                self.assertEqual(f.intent["expires_at"], f.focus["expires_at"])
                self.assertEqual(f.focus["state"], "active")

    async def test_none_preserves_old_payload_and_digest_and_has_no_focus_queries(self):
        for reference in (False, None):
            with self.subTest(reference=reference):
                f = VisualFixture()
                await f.enqueue(reference=reference)
                body = f.payloads[f.claim.source_payload_id]["content"]
                self.assertNotIn("perception_focus", body)
                output, problem = await visual_source(f.connection, f.intent)
                self.assertIsNone(problem)
                self.assertEqual(output.result_digest, digest(body))
                self.assertEqual(f.connection.subject_reads, [])
                f.focus.update(state="cancelled", generation=8)
                self.assertIsNotNone(await f.claim_next())

    async def test_bad_binding_fails_before_creating_payload(self):
        f = VisualFixture()
        bad = [True, [], {}, {"subject_id": "not-a-uuid", "generation": 1},
            f.reference | {"generation": True}, f.reference | {"generation": 0},
            f.reference | {"generation": "7"}, f.reference | {"generation": 7.0},
            f.reference | {"generation": 2**63}, f.reference | {"model_guess": True}]
        for reference in bad:
            with self.subTest(reference=reference):
                before = deepcopy(f.payloads)
                with self.assertRaises(AttentionError) as raised:
                    await f.enqueue(reference=reference)
                self.assertEqual(raised.exception.code, "visual_attention_focus_invalid")
                self.assertEqual(f.payloads, before)

    async def test_invalid_focus_is_rejected_at_enqueue(self):
        for changes in ({"generation": 8}, {"state": "cancelled"}, {"state": "completed"},
                        {"expires_at": NOW}):
            with self.subTest(changes=changes):
                f = VisualFixture()
                f.focus.update(changes)
                with self.assertRaises(AttentionError):
                    await f.enqueue()
                self.assertEqual(len(f.payloads), 1)

    async def test_missing_or_other_owner_focus_does_not_authorize(self):
        f = VisualFixture()
        f.subjects = {("different-owner", "visual_focus", f.focus_id): f.focus}
        with self.assertRaises(AttentionError):
            await f.enqueue()

    async def test_persisted_binding_survives_repository_reconstruction(self):
        for kind in ("visual", "prospective"):
            for changes in ({"generation": 8}, {"state": "cancelled"}, {"expires_at": NOW}):
                with self.subTest(kind=kind, changes=changes):
                    f = VisualFixture(kind=kind)
                    await f.enqueue()
                    # JSON roundtrip and fresh repository: no closure/session reference.
                    payload = f.payloads[f.claim.source_payload_id]
                    payload["content"] = json.dumps(payload["content"])
                    f.focus.update(changes)
                    restored = AttentionRepository(f)
                    output, problem = await restored._source(f.connection, f.intent, None)
                    self.assertIsNone(output)
                    self.assertIn(problem, {"perception_focus_changed", "perception_focus_expired"})

    async def test_unrelated_changes_and_generic_claim_churn_do_not_invalidate(self):
        f = VisualFixture()
        await f.enqueue()
        f.focus.update(updated_at=NOW+timedelta(seconds=1), claim_token=uuid4())
        f.subjects[f.claim.owner_id, "visual_focus", uuid4()] = f.focus | {"generation": 99}
        output, problem = await visual_source(f.connection, f.intent)
        self.assertIsInstance(output, NotificationOutput)
        self.assertIsNone(problem)

    async def test_prospective_retains_its_separate_claim_token_guard(self):
        f = VisualFixture(kind="prospective", reminder=True)
        await f.enqueue()
        f.tracked["claim_token"] = uuid4()
        output, problem = await visual_source(f.connection, f.intent, lock=True)
        self.assertIsNone(output)
        self.assertEqual(problem, "visual_subject_changed")
        f.tracked["claim_token"] = f.event["claim_token"]
        f.focus["generation"] += 1
        output, problem = await visual_source(f.connection, f.intent, lock=True)
        self.assertIsNone(output)
        self.assertEqual(problem, "perception_focus_changed")

    async def test_only_same_prospective_event_can_survive_its_own_completion(self):
        for kind in ("visual", "prospective"):
            with self.subTest(kind=kind):
                f = VisualFixture(kind=kind)
                await f.enqueue()
                f.focus.update(state="completed", claim_token=None, completed_event_id=str(f.event["id"]))
                output, problem = await visual_source(f.connection, f.intent)
                self.assertEqual(output is not None, kind == "prospective")
                self.assertEqual(problem, None if kind == "prospective" else "perception_focus_changed")
                f.focus["completed_event_id"] = str(uuid4())
                self.assertEqual((await visual_source(f.connection, f.intent))[1], "perception_focus_changed")

    async def test_legacy_prospective_still_checks_original_subject(self):
        f = VisualFixture(kind="prospective")
        await f.enqueue(reference=False)
        f.focus["generation"] += 1
        self.assertEqual((await visual_source(f.connection, f.intent))[1], "visual_subject_changed")

    async def test_payload_event_and_evidence_guards_remain_effective(self):
        f = VisualFixture()
        await f.enqueue()
        payload = f.payloads[f.claim.source_payload_id]
        payload["availability"] = "unavailable"
        self.assertEqual((await visual_source(f.connection, f.intent))[1], "source_payload_unavailable")
        payload["availability"] = "available"
        f.event_available = False
        self.assertEqual((await visual_source(f.connection, f.intent))[1], "notification_source_changed")
        f.event_available = True
        payload["content"]["expires_at"] = NOW.isoformat()
        self.assertEqual((await visual_source(f.connection, f.intent))[1], "visual_evidence_expired")

    async def test_malformed_persisted_condition_is_not_treated_as_legacy(self):
        f = VisualFixture()
        await f.enqueue()
        f.payloads[f.claim.source_payload_id]["content"]["perception_focus"] = {"generation": True}
        self.assertEqual((await visual_source(f.connection, f.intent))[1], "notification_source_invalid")

    async def test_claim_rechecks_and_revokes_cancelled_assignment(self):
        f = VisualFixture()
        await f.enqueue()
        f.focus["state"] = "cancelled"
        self.assertIsNone(await f.claim_next())
        self.assertEqual((f.intent["state"], f.intent["reason"]), ("revoked", "perception_focus_changed"))

    async def test_recovery_rechecks_pending_and_unknown_without_replaying(self):
        for state in ("pending", "unknown"):
            with self.subTest(state=state):
                f = VisualFixture()
                await f.enqueue()
                f.intent["state"] = state
                f.focus["generation"] += 1
                await f.ledger().recover(f.claim.owner_id)
                self.assertEqual(f.intent["state"], "revoked")
                self.assertEqual(f.intent["reason"], "perception_focus_changed")
                self.assertEqual(f.intent["attempt_count"], 0)

    async def test_preparation_rechecks_after_claim(self):
        f = VisualFixture()
        await f.enqueue()
        self.assertIsNotNone(await f.claim_next())
        f.focus["generation"] += 1
        with self.assertRaises(AttentionError) as raised:
            await f.prepare(f.ledger())
        self.assertEqual(raised.exception.code, "attention_source_unavailable")
        self.assertIsNone(f.attempt["plan_payload_id"])

    async def test_submission_rechecks_after_preparation(self):
        f = VisualFixture()
        await f.enqueue()
        await f.claim_next()
        ledger = f.ledger()
        offer = await f.prepare(ledger)
        f.focus["state"] = "cancelled"
        with self.assertRaises(AttentionError):
            await ledger.commit_submission(f.claim, offer)
        self.assertIsNone(f.attempt["submitted_at"])

    async def test_provider_ticket_rechecks_after_committed_submission(self):
        f = VisualFixture()
        await f.enqueue()
        await f.claim_next()
        ledger = f.ledger()
        offer = await f.prepare(ledger)
        ticket = await ledger.commit_submission(f.claim, offer)
        f.focus["generation"] += 1
        with self.assertRaises(AttentionError):
            async with ledger.guarded_transaction(ticket):
                self.fail("Stale assignment reached provider write")
        self.assertTrue(ticket._used)

    async def test_native_output_guard_uses_existing_connection_and_rechecks_each_write(self):
        for kind in ("visual", "prospective"):
            with self.subTest(kind=kind):
                f = VisualFixture(kind=kind)
                await f.enqueue()
                await f.claim_next()
                ledger = f.ledger()
                offer = await f.prepare(ledger)
                ticket = await ledger.commit_submission(f.claim, offer)
                async with ledger.guarded_transaction(ticket) as connection:
                    self.assertIs(connection, f.connection)
                    self.assertTrue(f.lock.locked())
                    self.assertEqual(f.connection.subject_reads[-1], (
                        (f.claim.owner_id, "visual_focus", f.focus_id), True, True))
                # Reconstruct the ledger, as recovery/reconnect would; the plan
                # and original binding remain authoritative, not this object.
                ledger = f.ledger()
                async with ledger.output_guard(f.claim, offer) as connection:
                    self.assertIs(connection, f.connection)
                    self.assertTrue(f.lock.locked())
                f.focus["state"] = "cancelled"
                with self.assertRaises(AttentionError):
                    async with ledger.output_guard(f.claim, offer):
                        self.fail("Late output used a cancelled assignment")

    async def test_original_claim_and_payload_guards_still_protect_native_output(self):
        for failure in ("claim", "source"):
            with self.subTest(failure=failure):
                f = VisualFixture()
                await f.enqueue()
                await f.claim_next()
                ledger = f.ledger()
                offer = await f.prepare(ledger)
                await ledger.commit_submission(f.claim, offer)
                if failure == "claim":
                    f.lease_valid = False
                else:
                    f.payloads[f.claim.source_payload_id]["availability"] = "unavailable"
                with self.assertRaises(AttentionError):
                    async with ledger.output_guard(f.claim, offer):
                        self.fail("Original canonical guard was bypassed")

    async def test_exact_focus_is_locked_once_for_prospective_output(self):
        f = VisualFixture(kind="prospective")
        await f.enqueue()
        await f.claim_next()
        f.connection.subject_reads.clear()
        async with f.repository.guard(f.claim):
            self.assertEqual(len(f.connection.subject_reads), 1)
            self.assertTrue(f.connection.subject_reads[0][1])

    async def test_optional_field_keeps_legacy_model_roundtrip_stable(self):
        f = VisualFixture()
        await f.enqueue(reference=False)
        body = f.payloads[f.claim.source_payload_id]["content"]
        self.assertEqual(VisualNotificationContent.model_validate(body).model_dump(mode="json"), body)
        self.assertEqual(VisualNotificationContent.model_validate(body | {"perception_focus": None})
                         .model_dump(mode="json"), body)
