"""Offline owner API qualification with scoped, transactional SQL fixtures.

This is not PostgreSQL trigger/locking qualification or real-world visual acceptance.
Production request validation, owner transitions, receipts and projections run here.
"""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from pydantic import ValidationError

from app.live_state.models import SessionScope
from app.live_state.perception import CheckResolution, EvidenceReference
from app.reminders.duties import ChangeDuty, DutySpec, ResponsibilityRepository, SaveDuty
from app.reminders.episode_checks import EpisodeCheckOwner, reference
from app.reminders.episode_models import AdmitEpisodeCheck, ChangeEpisodeCheck, EpisodeCheckDraft, EpisodeCheckSpec
from app.tools.inputs import ToolInputError
from app.tools.invocation import ToolInvocationContext


NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)


class Store:
    def __init__(self):
        self.rules, self.occurrences, self.receipts, self.events, self.reminders = {}, {}, {}, {}, {}
        self.streams = set()
        self.lock = asyncio.Lock()
        self.active = False
        self.schema = True
        self.fail_commit = False
        self.acquisitions = 0
        self.queries = []

    @asynccontextmanager
    async def acquire(self):
        self.acquisitions += 1
        yield self

    @asynccontextmanager
    async def transaction(self, **kwargs):
        async with self.lock:
            saved = deepcopy((self.rules, self.occurrences, self.receipts, self.events, self.reminders))
            self.active = True
            try:
                yield self
                if self.fail_commit:
                    self.fail_commit = False
                    raise OSError("fixture commit failed")
            except BaseException:
                self.rules, self.occurrences, self.receipts, self.events, self.reminders = saved
                raise
            finally:
                self.active = False

    def is_in_transaction(self):
        return self.active

    def sql(self, query):
        text = " ".join(query.lower().split())
        self.queries.append(text)
        if "limit 0" not in text:
            assert self.active, text
        return text

    def event(self, owner, session, *, at=NOW, **changes):
        identity, payload = uuid4(), uuid4()
        self.streams.add(owner)
        row = dict(id=identity, payload_id=payload, owner_id=owner, session_id=session,
            revision=1, kind="user_transcript", trust="user_input", finality="final",
            source_time=at, recorded_at=at, availability="available", content={"text": "fixture request"}, current=True)
        row.update(changes)
        self.events[identity] = row
        return ToolInvocationContext(provider_call_id=str(uuid4()), request_event_id=identity, request_revision=1)

    def valid_event(self, owner, identity, revision, session=None):
        row = self.events.get(identity)
        return row if row and row["owner_id"] == owner and row["revision"] == revision and row["current"] and (
            row["availability"] == "available" and row["content"] is not None and row["finality"] in {"final", "revised"}
            and (session is None or row["session_id"] == session)) else None

    def occurrence(self, args, *, episode):
        identity, owner, rule, revision, scheduled = args[:5]
        row = dict(id=identity, owner_id=owner, responsibility_id=rule, rule_revision=revision,
            scheduled_for=scheduled, reminder_id=None if episode else args[5], created_at=scheduled,
            status="open", report_state="unknown", report_event_id=None, report_revision=None, reported_at=None,
            check_revision=1, episode_id=args[5] if episode else None, episode_revision=1 if episode else None,
            check_payload=json.loads(args[6]) if episode else None, check_lifecycle="tracking" if episode else None,
            check_resolution=None)
        self.occurrences[identity] = row
        return dict(row)

    async def fetchrow(self, query, *args):
        q = self.sql(query)
        if q.startswith("select * from jarvis_responsibility_requests"):
            return self.receipts.get(tuple(args))
        if q.startswith("select * from jarvis_responsibilities"):
            row = self.rules.get(args[1])
            return dict(row) if row and row["owner_id"] == args[0] else None
        if q.startswith("select * from jarvis_responsibility_occurrences"):
            row = self.occurrences.get(args[2])
            return dict(row) if row and (row["owner_id"], row["responsibility_id"]) == args[:2] else None
        if q.startswith("select e.*,p.availability"):
            row = self.valid_event(*args)
            return dict(row) if row else None
        if q.startswith("insert into jarvis_responsibilities"):
            identity = uuid4()
            row = dict(id=identity, owner_id=args[0], session_id=args[1], spec=json.loads(args[2]),
                next_due_at=args[3] if len(args) == 4 else None, revision=1, status="active",
                snoozed_until=None, created_at=NOW, updated_at=NOW)
            self.rules[identity] = row
            return dict(row)
        if q.startswith("update jarvis_responsibilities"):
            row = self.rules[args[1]]
            assert row["owner_id"] == args[0]
            if "set spec=" in q:
                row.update(spec=json.loads(args[2]), next_due_at=args[3], session_id=args[4], snoozed_until=None)
            row.update(revision=row["revision"] + 1, status=args[2] if "status=$3" in q else "active")
            return dict(row)
        if q.startswith("insert into jarvis_responsibility_occurrences"):
            return self.occurrence(args, episode=True)
        if q.startswith("update jarvis_responsibility_occurrences set check_revision"):
            row = self.occurrences[args[2]]
            assert (row["owner_id"], row["responsibility_id"]) == args[:2]
            row.update(check_revision=row["check_revision"] + 1, episode_revision=args[3], check_payload=json.loads(args[4]),
                check_lifecycle=args[5], check_resolution=json.loads(args[6]) if args[6] else None, status=args[7],
                rule_revision=args[8], report_state=args[9], report_event_id=args[10], report_revision=args[11], reported_at=args[12])
            return dict(row)
        raise AssertionError(q)

    async def fetchval(self, query, *args):
        q = self.sql(query)
        if q.startswith("select owner_id from jarvis_context_streams"):
            return args[0] if args[0] in self.streams else None
        if q.startswith("select e.payload_id"):
            row = self.valid_event(*args)
            return row["payload_id"] if row and row["kind"] in {"user_request", "user_transcript"} and row["trust"] == "user_input" else None
        if q.startswith("select lock_jarvis_copy_sources"):
            return bool(args[1]) and all(any(e["owner_id"] == args[0] and e["payload_id"] == root
                and e["availability"] == "available" and e["content"] is not None for e in self.events.values()) for root in args[1])
        if q.startswith("select exists(select 1 from jarvis_responsibility_occurrences"):
            return any((r["owner_id"], r["responsibility_id"]) == args[:2] and r["status"] == "open" for r in self.occurrences.values())
        if q.startswith("insert into jarvis_reminders"):
            identity = uuid4()
            self.reminders[identity] = dict(id=identity, owner_id=args[0], status="active", triggered_at=None, not_before=args[6])
            return identity
        raise AssertionError(q)

    async def execute(self, query, *args):
        q = self.sql(query)
        if q.startswith("select pg_advisory_xact_lock"):
            return None
        if q.startswith("insert into jarvis_responsibility_requests"):
            key = tuple(args[:5])
            assert key not in self.receipts
            self.receipts[key] = dict(owner_id=args[0], request_event_id=args[1], request_revision=args[2],
                operation=args[3], request_slot=args[4], arguments_hash=args[5], responsibility_id=args[6],
                occurrence_id=args[7], report_state=args[8], created_at=NOW)
            return None
        if q.startswith("insert into jarvis_responsibility_occurrences"):
            self.occurrence(args, episode=False)
            return None
        if q.startswith("update jarvis_responsibilities set next_due_at"):
            self.rules[args[1]]["next_due_at"] = args[2]
            return None
        if q.startswith("update jarvis_reminders set status='completed'"):
            for identity in args[1]:
                assert self.reminders[identity]["owner_id"] == args[0]
                self.reminders[identity]["status"] = "completed"
            return None
        raise AssertionError(q)

    async def fetch(self, query, *args):
        q = self.sql(query)
        if "limit 0" in q:
            if not self.schema:
                raise OSError("schema unavailable")
            return []
        if q.startswith("select id,responsibility_id from jarvis_responsibility_occurrences"):
            return [dict(r) for r in self.occurrences.values() if r["owner_id"] == args[0]
                    and r["episode_id"] is not None and (args[1] or r["status"] == "open")][:args[2]]
        if q.startswith("select * from jarvis_responsibilities"):
            assert "coalesce(spec->>'kind','scheduled')='scheduled'" in q
            return [dict(r) for r in self.rules.values() if r["owner_id"] == args[0] and r["status"] == "active"
                and r["spec"].get("kind", "scheduled") == "scheduled" and r["next_due_at"] is not None
                and r["next_due_at"] <= args[1]]
        if q.startswith("update jarvis_responsibility_occurrences set status=$3"):
            changed = []
            for row in self.occurrences.values():
                if (row["owner_id"], row["responsibility_id"]) == args[:2] and row["status"] == "open":
                    row["status"] = args[2]
                    changed.append(dict(row))
            return changed
        if q.startswith("select o.*,r.status"):
            assert "left join jarvis_reminders" in q
            rows = []
            for row in self.occurrences.values():
                if (row["owner_id"], row["responsibility_id"]) != args[:2]:
                    continue
                reminder = self.reminders.get(row["reminder_id"], {})
                event = self.events.get(row["report_event_id"], {})
                rows.append(dict(row) | dict(reminder_status=reminder.get("status"), triggered_at=reminder.get("triggered_at"),
                    not_before=reminder.get("not_before"), report_available=event.get("availability") == "available"))
            return sorted(rows, key=lambda r:r["scheduled_for"], reverse=True)[:21]
        if q.startswith("select q.report_state"):
            return [dict(r) | dict(source_available=self.events[r["request_event_id"]]["availability"] == "available")
                for r in self.receipts.values() if (r["owner_id"], r["responsibility_id"]) == args[:2] and r["report_state"] is not None][:21]
        raise AssertionError(q)


class EpisodeOwnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = Store()
        self.repo = ResponsibilityRepository(self.store)
        self.scope = SessionScope(owner_id="episode-fixture", session_id=uuid4())
        self.now = NOW
        self.owner = EpisodeCheckOwner(self.repo, enabled=True, clock=lambda:self.now)
        await self.owner.initialize()

    def source(self, **kwargs):
        return self.store.event(self.scope.owner_id, self.scope.session_id, at=self.now, **kwargs)

    def request(self, **kwargs):
        return AdmitEpisodeCheck(spec=EpisodeCheckSpec(title="Before leaving"), check=EpisodeCheckDraft(
            question="What do the controls indicate?", object_association="stove:kitchen",
            observation_window_expires_at=NOW + timedelta(minutes=5)), **kwargs)

    async def admit(self):
        source, request = self.source(), self.request()
        return await self.owner.admit(self.scope, source, request), source, request

    def change_request(self, result, operation, **kwargs):
        check = result.check
        return ChangeEpisodeCheck(responsibility_id=check.canonical_owner_ref.source_id,
            expected_revision=check.canonical_owner_ref.revision, occurrence_id=check.id,
            expected_check_revision=check.revision, episode_id=check.episode_or_object_ref.source_id,
            expected_episode_revision=check.episode_or_object_ref.revision, operation=operation, **kwargs)

    def resolution(self, result, source, **kwargs):
        return CheckResolution(basis="user_report", applicable_episode_or_object_ref=result.check.episode_or_object_ref,
            source_refs=(EvidenceReference(source=reference("context_event", source.request_event_id, 1),
                captured_at=self.now, episode_or_object_ref=result.check.episode_or_object_ref),),
            resolved_at=self.now, **kwargs)

    async def error(self, code, coroutine):
        with self.assertRaises(ToolInputError) as caught:
            await coroutine
        self.assertEqual(caught.exception.code, code)

    async def test_disabled_and_uninitialized_do_not_claim_acceptance(self):
        for enabled in (False, True):
            owner = self.repo.episode_checks(enabled=enabled)
            calls = self.store.acquisitions
            result = await owner.admit(self.scope, self.source(), self.request())
            self.assertEqual(result.outcome.status, "unavailable")
            self.assertEqual(result.outcome.capability.status, "unavailable" if enabled else "disabled")
            self.assertEqual(self.store.acquisitions, calls)
        self.assertFalse(self.store.rules)

    async def test_schema_probe_failure_is_unavailable_not_flag_enabled(self):
        self.store.schema = False
        owner = self.repo.episode_checks(enabled=True)
        self.assertEqual((await owner.initialize()).status, "unavailable")
        self.assertEqual((await owner.admit(self.scope, self.source(), self.request())).outcome.status, "unavailable")

    async def test_admission_commits_one_canonical_occurrence_without_reminder(self):
        result, source, request = await self.admit()
        self.assertEqual(result.outcome.status, "accepted")
        self.assertFalse(self.store.active)
        self.assertEqual(result.check.continuity, "durable")
        rule = self.store.rules[result.check.canonical_owner_ref.source_id]
        occurrence = self.store.occurrences[result.check.id]
        self.assertIsNone(rule["next_due_at"])
        self.assertIsNone(occurrence["reminder_id"])
        self.assertFalse(self.store.reminders)
        self.assertNotEqual(occurrence["episode_id"], occurrence["id"])
        replay = await self.owner.admit(self.scope, source, request)
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.check, result.check)
        self.assertEqual(len(self.store.occurrences), 1)

    async def test_concurrent_same_identity_serializes_to_one_effect(self):
        source, request = self.source(), self.request()
        first, second = await asyncio.gather(self.owner.admit(self.scope, source, request),
                                             self.owner.admit(self.scope, source, request))
        self.assertEqual(first.check.id, second.check.id)
        self.assertEqual(first.replayed + second.replayed, 1)

    async def test_conflicting_request_slot_has_no_second_effect(self):
        _, source, request = await self.admit()
        changed = request.model_copy(update={"check": EpisodeCheckDraft(question="Different question")})
        await self.error("request_conflict", self.owner.admit(self.scope, source, changed))
        self.assertEqual(len(self.store.rules), 1)

    async def test_commit_failure_returns_no_acceptance_and_same_identity_retries(self):
        source, request = self.source(), self.request()
        self.store.fail_commit = True
        result = await self.owner.admit(self.scope, source, request)
        self.assertEqual(result.outcome.status, "unavailable")
        self.assertFalse(self.store.rules)
        await self.owner.initialize()
        self.assertEqual((await self.owner.admit(self.scope, source, request)).outcome.status, "accepted")

    async def test_admission_requires_current_owned_final_user_evidence(self):
        cases = ({"trust":"sensor"}, {"kind":"observation"}, {"finality":"provisional"},
                 {"current":False}, {"availability":"deleted", "content":None}, {"session_id":uuid4()}, {"owner_id":"other"})
        for changes in cases:
            with self.subTest(changes=changes):
                source = self.source()
                self.store.events[source.request_event_id].update(changes)
                await self.error("request_evidence_unavailable", self.owner.admit(self.scope, source, self.request()))
        await self.error("request_evidence_required", self.owner.admit(self.scope, ToolInvocationContext(), self.request()))
        self.assertFalse(self.store.rules)

    async def test_inference_cue_keeps_provenance_but_cannot_authorize(self):
        cue = self.source(kind="observation", trust="sensor")
        check = EpisodeCheckDraft(question="Inspect the controls", cue_origin="policy_permitted_inference",
            dependencies=(reference("context_event", cue.request_event_id, 1),))
        request = AdmitEpisodeCheck(spec=EpisodeCheckSpec(title="Confirmed check"), check=check)
        await self.error("request_evidence_unavailable", self.owner.admit(self.scope, cue, request))
        result = await self.owner.admit(self.scope, self.source(), request)
        payload = self.store.occurrences[result.check.id]["check_payload"]
        self.assertEqual(payload["cue_origin"], "policy_permitted_inference")
        self.assertEqual(result.check.origin, "explicit_request")
        with self.assertRaises(ValidationError):
            EpisodeCheckSpec(title="Unconfirmed", basis="inferred_under_policy")

    async def test_new_episode_has_new_identity_and_preserves_old_open_check(self):
        first, _, _ = await self.admit()
        self.now += timedelta(seconds=10)
        second = await self.owner.admit(self.scope, self.source(), self.request(
            responsibility_id=first.check.canonical_owner_ref.source_id, expected_revision=1))
        self.assertNotEqual(first.check.id, second.check.id)
        self.assertNotEqual(first.check.episode_or_object_ref.source_id, second.check.episode_or_object_ref.source_id)
        self.assertEqual(self.store.occurrences[first.check.id]["check_revision"], 1)
        self.assertEqual(self.store.occurrences[first.check.id]["status"], "open")

    async def test_rule_check_episode_versions_and_identity_are_independent(self):
        result, _, _ = await self.admit()
        original = self.change_request(result, "cancel")
        for field, value, code in (("expected_revision",2,"revision_conflict"),
                ("expected_check_revision",2,"check_revision_conflict"),
                ("expected_episode_revision",2,"check_revision_conflict"), ("episode_id",uuid4(),"check_revision_conflict")):
            with self.subTest(field=field):
                await self.error(code, self.owner.change(self.scope, self.source(), original.model_copy(update={field:value})))
        self.assertEqual(self.store.occurrences[result.check.id]["check_revision"], 1)

    async def test_object_correction_advances_episode_without_reusing_old_evidence(self):
        result, _, _ = await self.admit()
        self.now += timedelta(seconds=10)
        revised = await self.owner.change(self.scope, self.source(), self.change_request(result, "revise",
            check=EpisodeCheckDraft(question="Inspect this control", object_association="other:stove")))
        self.assertEqual(revised.check.episode_or_object_ref.source_id, result.check.episode_or_object_ref.source_id)
        self.assertEqual(revised.check.episode_or_object_ref.revision, 2)
        self.assertEqual(revised.check.revision, 2)
        source = self.source()
        await self.error("resolution_episode_conflict", self.owner.change(self.scope, source,
            self.change_request(revised, "resolve", resolution=self.resolution(result, source))))

    async def test_user_report_resolution_retains_basis_time_and_report_history(self):
        result, _, _ = await self.admit()
        self.now += timedelta(seconds=10)
        source = self.source()
        request = self.change_request(result, "resolve", resolution=self.resolution(result, source))
        resolved = await self.owner.change(self.scope, source, request)
        self.assertEqual(resolved.check.lifecycle, "resolved")
        self.assertEqual(resolved.check.resolution.basis, "user_report")
        self.assertEqual(resolved.check.resolution.source_refs[0].captured_at, self.now)
        self.assertEqual(self.store.occurrences[result.check.id]["report_state"], "user_reported_done")
        self.assertTrue((await self.owner.change(self.scope, source, request)).replayed)
        async with self.store.acquire() as c, c.transaction():
            view = await self.repo._view(c, self.scope.owner_id, result.check.canonical_owner_ref.source_id)
        self.assertEqual(len(view["report_history"]), 1)
        self.assertIsNone(view["occurrences"][0]["reminder_status"])

    async def test_resolution_rejects_retimed_and_previous_episode_user_report(self):
        result, _, _ = await self.admit()
        self.now += timedelta(seconds=10)
        for offset in (-20, 1):
            source = self.source()
            self.store.events[source.request_event_id]["source_time"] = self.now + timedelta(seconds=offset)
            await self.error("resolution_evidence_mismatch", self.owner.change(self.scope, source,
                self.change_request(result, "resolve", resolution=self.resolution(result, source))))
        self.assertEqual(self.store.occurrences[result.check.id]["status"], "open")

    async def test_visual_and_trusted_resolution_fail_unavailable_without_mutating(self):
        result, _, _ = await self.admit()
        for basis in ("visual_observation", "trusted_source"):
            source = self.source()
            evidence = EvidenceReference(source=reference("context_event", source.request_event_id, 1),
                captured_at=self.now, episode_or_object_ref=result.check.episode_or_object_ref,
                frame_id=uuid4() if basis == "visual_observation" else None,
                visibility="visible" if basis == "visual_observation" else None)
            resolution = CheckResolution(basis=basis, source_refs=(evidence,), resolved_at=self.now,
                applicable_episode_or_object_ref=result.check.episode_or_object_ref)
            await self.error("resolution_validator_unavailable", self.owner.change(self.scope, source,
                self.change_request(result, "resolve", resolution=resolution)))
        self.assertIsNone(self.store.occurrences[result.check.id]["check_resolution"])

    async def test_out_of_view_evidence_does_not_make_resolution(self):
        result, _, _ = await self.admit()
        for visibility in ("not_visible", "unclear"):
            with self.assertRaises(ValidationError):
                CheckResolution(basis="visual_observation", source_refs=(EvidenceReference(
                    source=reference("context_event", self.source().request_event_id, 1), captured_at=self.now,
                    episode_or_object_ref=result.check.episode_or_object_ref, frame_id=uuid4(), visibility=visibility),),
                    resolved_at=self.now, applicable_episode_or_object_ref=result.check.episode_or_object_ref)
        self.assertEqual(self.store.occurrences[result.check.id]["status"], "open")

    async def test_expiry_leaves_obligation_open_without_resolution(self):
        result, _, _ = await self.admit()
        self.now += timedelta(minutes=6)
        expired = await self.owner.change(self.scope, self.source(), self.change_request(result, "expire"))
        self.assertEqual(expired.check.lifecycle, "expired")
        self.assertIsNone(expired.check.resolution)
        self.assertEqual(self.store.rules[result.check.canonical_owner_ref.source_id]["status"], "active")
        self.assertEqual(self.store.occurrences[result.check.id]["status"], "open")

    async def test_snooze_and_cancellation_never_complete_physical_action(self):
        result, _, _ = await self.admit()
        source = self.source()
        snoozed = await self.owner.change(self.scope, source,
            self.change_request(result, "snooze", until=self.now + timedelta(minutes=1)))
        self.assertEqual(snoozed.check.lifecycle, "tracking")
        self.assertFalse(self.store.reminders)
        cancelled = await self.owner.change(self.scope, self.source(), self.change_request(snoozed, "cancel"))
        self.assertEqual(cancelled.check.lifecycle, "dismissed")
        self.assertIsNone(cancelled.check.resolution)
        self.assertEqual(self.store.occurrences[result.check.id]["report_state"], "unknown")
        source = self.source()
        await self.error("episode_check_inactive", self.owner.change(self.scope, source,
            self.change_request(cancelled, "resolve", resolution=self.resolution(cancelled, source))))

    async def test_projection_uses_exact_canonical_versions_and_original_roots(self):
        result, source, _ = await self.admit()
        read = await self.owner.read(self.scope)
        snapshot, = read.snapshots
        self.assertEqual(snapshot.check.id, result.check.id)
        self.assertEqual(snapshot.source_payload_ids, (self.store.events[source.request_event_id]["payload_id"],))
        self.assertIn(result.check.episode_or_object_ref, snapshot.dependencies)
        refs = dict(responsibility=result.check.canonical_owner_ref, check=result.outcome.check_ref,
                    episode=result.check.episode_or_object_ref)
        async with self.owner.guard(self.scope, **refs) as (connection, guarded):
            self.assertTrue(connection.is_in_transaction())
            self.assertEqual(guarded, snapshot)
        self.assertFalse(self.store.active)
        await self.owner.change(self.scope, self.source(), self.change_request(result, "cancel"))
        with self.assertRaises(ToolInputError):
            async with self.owner.guard(self.scope, **refs):
                self.fail("stale versions admitted")

    async def test_borrowed_guard_withdrawal_and_cross_owner_fail_closed(self):
        result, source, _ = await self.admit()
        refs = dict(responsibility=result.check.canonical_owner_ref, check=result.outcome.check_ref,
                    episode=result.check.episode_or_object_ref)
        async with self.store.acquire() as c, c.transaction():
            async with self.owner.guard(self.scope, connection=c, **refs) as (borrowed, _):
                self.assertIs(borrowed, c)
        other = SessionScope(owner_id="other", session_id=uuid4())
        self.store.event(other.owner_id, other.session_id)
        self.assertFalse((await self.owner.read(other)).snapshots)
        with self.assertRaises(ToolInputError):
            async with self.owner.guard(other, **refs):
                self.fail("foreign check admitted")
        self.store.events[source.request_event_id]["availability"] = "deleted"
        read = await self.owner.read(self.scope)
        self.assertFalse(read.snapshots)
        self.assertEqual(len(read.gaps), 1)
        self.assertEqual(self.store.occurrences[result.check.id]["status"], "open")

    async def test_legacy_scheduled_save_materialize_and_receipt_hash_unchanged(self):
        source = self.source()
        request = SaveDuty(spec=DutySpec(title="Laundry", first_at=NOW, every_days=1))
        # Saving materializes against wall time; keep it on the same fixture
        # timeline as the explicit two-day scheduler advance below.
        materialize = self.repo._materialize

        async def materialize_at_fixture_time(connection, owner, rule, wall_time):
            return await materialize(connection, owner, rule, NOW)

        with patch.object(self.repo, "_materialize", side_effect=materialize_at_fixture_time):
            first = await self.repo.save(self.scope.owner_id, self.scope.session_id, source, request)
            replay = await self.repo.save(self.scope.owner_id, self.scope.session_id, source, request)
        self.assertTrue(replay["request_replayed"])
        body = request.model_dump(mode="json")
        body["spec"].pop("kind")
        receipt, = self.store.receipts.values()
        self.assertEqual(receipt["arguments_hash"], sha256(json.dumps(body, sort_keys=True).encode()).hexdigest())
        self.assertIsNone(first["occurrences"][0]["episode_id"])
        self.assertIsNotNone(first["occurrences"][0]["reminder_id"])
        await self.admit()
        episodes_before = {k:dict(v) for k,v in self.store.occurrences.items() if v["episode_id"] is not None}
        # The scheduler also handles historical rows without the kind discriminator.
        scheduled = next(r for r in self.store.rules.values() if r["spec"].get("kind") == "scheduled")
        scheduled["spec"].pop("kind")
        self.assertEqual(await self.repo.materialize(self.scope.owner_id, now=NOW + timedelta(days=2)), 1)
        self.assertEqual({k:v for k,v in self.store.occurrences.items() if v["episode_id"] is not None}, episodes_before)

    async def test_scheduled_api_cannot_bypass_episode_version_checks(self):
        result, _, _ = await self.admit()
        identity = result.check.canonical_owner_ref.source_id
        await self.error("episode_versions_required", self.repo.change(self.scope.owner_id, self.scope.session_id,
            self.source(), ChangeDuty(responsibility_id=identity, expected_revision=1, operation="cancel")))
        await self.error("episode_versions_required", self.repo.save(self.scope.owner_id, self.scope.session_id,
            self.source(), SaveDuty(responsibility_id=identity, expected_revision=1,
                spec=DutySpec(title="Wrong conversion", first_at=NOW))))
        self.assertEqual(self.store.rules[identity]["revision"], 1)


if __name__ == "__main__":
    unittest.main()
