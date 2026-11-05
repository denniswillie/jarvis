"""Real composer/binder/repository logic; bounded in-memory persistence, no DB."""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
import unittest
from uuid import UUID, uuid4

from app.knowledge.models import SourceError
from app.live_state.canonical import CanonicalContextBinder, context_dependency
from app.live_state.models import (
    AggregateRef, FocusEntry, GuardedPayloadBinding, LiveSnapshot, ObservationEntry,
    SessionScope, SourceDependency, StateCursor,
)
from app.live_state.perception_focus import FOCUS_SUBJECT
from app.live_state.perception_runtime import SessionPerceptionRuntime
from app.runtime.events import DomainEventBus
from app.tasks.context_events import ContextRecord


NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)


class FocusPersistence:
    """Only the exercised SQL subset; unexpected SQL fails, writes are bounded.

    Uses actual TaskRepository.payload and ContextRepository record decoding and
    simulates recursive source removal. It does not qualify PostgreSQL triggers,
    lock scheduling, migrations or provider/device behavior.
    """

    def __init__(self, scope):
        self.scope = scope
        self.focuses, self.records, self.payloads = [], {}, {}
        self.parents, self.event_edges = {}, []
        self.calls = []
        self._lock = asyncio.Lock()
        self._transaction = False

    @asynccontextmanager
    async def transaction(self, owner):
        assert owner == self.scope.owner_id
        async with self._lock:
            self._transaction = True
            saved = deepcopy((self.payloads, self.parents, self.event_edges))
            try:
                yield self
            except BaseException:
                self.payloads, self.parents, self.event_edges = saved
                raise
            finally:
                self._transaction = False

    def is_in_transaction(self):
        return self._transaction

    def available(self, root):
        return root in self.payloads and self.payloads[root] is not None and all(
            self.available(parent) for parent in self.parents.get(root, ()))

    async def fetchval(self, sql, *args):
        self.calls.append((sql, args))
        if "from jarvis_sessions" in sql:
            return args == (self.scope.owner_id, self.scope.session_id)
        if "lock_jarvis_copy_sources" in sql:
            return all(self.available(root) for root in args[1])
        if "insert into jarvis_context_payloads" in sql:
            assert len(self.payloads) < 100, "unbounded payload creation"
            root = uuid4()
            self.payloads[root] = json.loads(args[1])
            return root
        if "select content from jarvis_context_payloads" in sql:
            return self.payloads.get(args[1])
        raise AssertionError(sql)

    async def execute(self, sql, *args):
        self.calls.append((sql, args))
        if "pg_advisory_xact_lock" in sql:
            return
        if "insert into jarvis_context_payload_dependencies" in sql:
            assert self.available(args[1])
            self.parents.setdefault(args[2], set()).add(args[1])
            return
        raise AssertionError(sql)

    async def executemany(self, sql, rows):
        if "insert into jarvis_context_event_dependencies" not in sql:
            raise AssertionError(sql)
        self.event_edges.extend(rows)

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        if "from jarvis_visual_focuses f" in sql:
            return [deepcopy(row) for row in self.focuses
                    if row["owner_id"] == args[0] and row["completed_at"] is None
                    and row["cleared_at"] is None and row["expires_at"] > args[1]][:2]
        if "from jarvis_context_events e join jarvis_context_payloads p" in sql:
            records = list(self.records.values())
            if "p.content#>>'{result,visual_focus,id}'" in sql:
                owner, session, focus_id, ids = args
                records = [record for record in records if record.owner_id == owner
                    and record.session_id == session and (ids is None or record.id in ids)
                    and (record.content or {}).get("result", {}).get("visual_focus", {}).get("id") == focus_id]
                records.sort(key=lambda record: record.sequence, reverse=True)
                records = records[:16]
            else:
                records = [record for record in records if record.owner_id == args[0] and record.id in args[1]]
            return [record.model_dump() | {
                "availability": "available" if self.available(record.payload_id) else "unavailable",
                "content": self.payloads.get(record.payload_id)} for record in records]
        raise AssertionError(sql)

    def add_record(self, *, kind="tool_outcome", content, **changes):
        values = dict(id=uuid4(), owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            sequence=len(self.records) + 1, origin_sequence=len(self.records) + 1,
            source_namespace="voice:openai", source_event_id=str(uuid4()), revision=3,
            supersedes_event_id=None, kind=kind, trust="tool_receipt" if kind == "tool_outcome" else "derived",
            finality="final", source_time=NOW, recorded_at=NOW, payload_id=uuid4(),
            availability="available", content_digest="a" * 64, content=content, is_current=True)
        record = ContextRecord(**(values | changes))
        self.records[record.id] = record
        self.payloads[record.payload_id] = record.content
        return record

    def add_focus(self, *, mode="find_once", generation=7, target="house keys",
                  revision=1, episode_revision=1):
        row = dict(id=uuid4(), owner_id=self.scope.owner_id, target=target,
            description="A small key on a red ring", purpose="Find the misplaced item", mode=mode,
            created_at=NOW - timedelta(seconds=1), expires_at=NOW + timedelta(minutes=5),
            completed_at=None, cleared_at=None, subject_generation=generation,
            subject_state="active", subject_expires_at=NOW + timedelta(minutes=5),
            descriptor_digest="b" * 64, descriptor_matches=True, created_from_message_id=None,
            revision=revision, episode_revision=episode_revision, check_resolution=None,
            completion_evidence=None, trigger_claim_token=None, trigger_claimed_at=None,
            delivery_status=None)
        self.focuses = [row]
        return row, self.receipt(row)

    def receipt(self, row, **changes):
        rendered = {key: row[key] for key in ("target", "description", "purpose", "mode")}
        rendered.update(id=str(row["id"]), created_at=row["created_at"].isoformat(),
            expires_at=row["expires_at"].isoformat(), completed_at=None, cleared_at=None,
            revision=row["revision"], episode_revision=row["episode_revision"],
            check_resolution=row["check_resolution"])
        return self.add_record(content={"tool_name": "set_visual_focus", "provider_call_id": str(uuid4()),
            "mutates_state": True, "result": {"status": "success", "visual_focus": rendered}}, **changes)


class SessionPerceptionFocusTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="canonical-focus-fixture", session_id=uuid4())
        self.now = NOW
        self.io = FocusPersistence(self.scope)
        self.snapshot = LiveSnapshot(scope=self.scope, cursor=StateCursor(epoch=uuid4(), revision=1),
                                     as_of=NOW, readiness="ready")
        sources = SimpleNamespace(pool=None, transaction=self.io.transaction)
        self.canonical = CanonicalContextBinder(sources, self.scope, clock=lambda: self.now)
        self.runtime = SessionPerceptionRuntime(service=SimpleNamespace(snapshot=lambda: self.snapshot),
            canonical=self.canonical, bus=DomainEventBus(), clock=lambda: self.now, enabled=True)

    async def asyncTearDown(self):
        await self.runtime.close()

    async def refresh(self):
        await self.runtime._apply(await self.runtime._read())
        return self.runtime.bridge.composition.brief

    async def current(self, brief, binding):
        async with self.runtime.binder.guard(brief, binding):
            return True

    async def test_actual_composition_binds_focus_counter_and_original_receipt(self):
        row, receipt = self.io.add_focus()
        brief = await self.refresh()
        question = brief.questions[0]
        self.assertEqual(len(brief.questions), 1)
        self.assertEqual(question.focus_ref.source_namespace, FOCUS_SUBJECT)
        self.assertEqual(question.focus_ref.source_id, row["id"])
        self.assertEqual((question.focus_ref.revision, question.focus_ref.generation), (7, 7))
        self.assertEqual(question.check_ref, SourceDependency(source_namespace="visual_focus",
            source_id=row["id"], revision=row["revision"], generation=None))
        self.assertEqual(question.episode_or_object_ref, SourceDependency(source_namespace="episode",
            source_id=row["id"], revision=row["episode_revision"], generation=None))
        self.assertEqual(set(brief.dependencies), {question.focus_ref, question.check_ref,
            question.episode_or_object_ref, context_dependency(receipt)})
        self.assertIn("house keys", question.question)
        self.assertIn("uncertainty", question.question)
        self.assertIn(context_dependency(receipt), brief.dependencies)
        binding = self.runtime.bridge._binding
        root = binding.payload_root_ids[0]
        self.assertEqual(binding.dependencies, brief.dependencies)
        self.assertEqual(self.io.parents[root], {receipt.payload_id})
        self.assertEqual(self.io.payloads[root]["brief"], brief.model_dump(mode="json"))
        self.assertEqual(self.io.event_edges[0][2:], (receipt.id, 3, receipt.payload_id, "current"))
        self.assertTrue(await self.current(brief, binding))
        self.assertEqual(self.runtime.diagnostics()["visual_focus_input"], "available")
        self.assertTrue(brief.scene_change_cues)
        self.assertEqual(brief.durable_responsibility.status, "unavailable")

    async def test_monitor_is_observational_and_does_not_mutate_watch(self):
        row, _ = self.io.add_focus(mode="monitor")
        before = deepcopy(row)
        brief = await self.refresh()
        self.assertEqual(row, before)
        self.assertEqual(brief.questions[0].check_ref, SourceDependency(source_namespace="visual_focus",
            source_id=row["id"], revision=row["revision"], generation=None))
        self.assertEqual(brief.questions[0].episode_or_object_ref, SourceDependency(source_namespace="episode",
            source_id=row["id"], revision=row["episode_revision"], generation=None))
        self.assertIsNone(row["check_resolution"])
        self.assertIsNone(row["completed_at"])
        self.assertIsNone(row["completion_evidence"])
        self.assertFalse(any(sql.lstrip().lower().startswith("update") for sql, _ in self.io.calls))

    async def test_replacement_rejects_old_assignment_and_selects_new_focus(self):
        old, _ = self.io.add_focus()
        brief = await self.refresh()
        binding = self.runtime.bridge._binding
        replacement, _ = self.io.add_focus(target="umbrella")
        with self.assertRaises(SourceError):
            await self.current(brief, binding)
        current = await self.refresh()
        self.assertEqual(current.questions[0].focus_ref.source_id, replacement["id"])
        self.assertNotEqual(old["id"], replacement["id"])
        self.assertTrue(await self.current(current, self.runtime.bridge._binding))

    async def test_unrelated_snapshot_and_delivery_bookkeeping_preserve_assignment(self):
        row, _ = self.io.add_focus()
        brief = await self.refresh()
        binding = self.runtime.bridge._binding
        self.snapshot = self.snapshot.model_copy(update={"cursor": StateCursor(
            epoch=self.snapshot.cursor.epoch, revision=999)})
        row["trigger_claim_token"] = uuid4()
        row["trigger_claimed_at"] = NOW
        self.assertTrue(await self.current(brief, binding))
        self.assertEqual(await self.refresh(), brief)
        self.assertEqual(self.runtime.bridge._binding, binding)

    async def test_new_get_receipt_does_not_invalidate_old_exact_receipt(self):
        row, _ = self.io.add_focus()
        brief = await self.refresh()
        binding = self.runtime.bridge._binding
        new = self.io.receipt(row)
        content = dict(new.content, tool_name="get_visual_focus")
        self.io.records[new.id] = new.model_copy(update={"content": content})
        self.io.payloads[new.payload_id] = content
        self.assertTrue(await self.current(brief, binding))

    async def test_real_counter_change_rejects_same_focus_identity(self):
        row, _ = self.io.add_focus()
        brief = await self.refresh()
        binding = self.runtime.bridge._binding
        row["subject_generation"] += 1
        with self.assertRaises(SourceError):
            await self.current(brief, binding)
        current = await self.refresh()
        self.assertEqual(current.questions[0].focus_ref.revision, 8)

    async def test_semantic_revision_invalidates_without_changing_subject_or_episode(self):
        row, _ = self.io.add_focus()
        brief = await self.refresh()
        binding = self.runtime.bridge._binding
        original = brief.questions[0]
        row["revision"] += 1
        with self.assertRaises(SourceError):
            await self.current(brief, binding)
        current = await self.refresh()
        question = current.questions[0]
        self.assertEqual(question.check_ref.revision, row["revision"])
        self.assertEqual(question.focus_ref, original.focus_ref)
        self.assertEqual(question.episode_or_object_ref, original.episode_or_object_ref)
        self.assertNotEqual(question.id, original.id)
        self.assertTrue(await self.current(current, self.runtime.bridge._binding))

    async def test_episode_rebind_requires_new_matching_receipt_and_rejects_old_assignment(self):
        row, _ = self.io.add_focus()
        brief = await self.refresh()
        binding = self.runtime.bridge._binding
        original = brief.questions[0]
        row["revision"] += 1
        row["episode_revision"] += 1
        with self.assertRaises(SourceError):
            await self.current(brief, binding)
        self.assertEqual((await self.refresh()).questions, ())
        self.assertEqual(self.runtime.focus_omission, "perception_focus_receipt_unavailable")
        receipt = self.io.receipt(row)
        current = await self.refresh()
        question = current.questions[0]
        self.assertEqual(question.check_ref, SourceDependency(source_namespace="visual_focus",
            source_id=row["id"], revision=row["revision"], generation=None))
        self.assertEqual(question.episode_or_object_ref, SourceDependency(source_namespace="episode",
            source_id=row["id"], revision=row["episode_revision"], generation=None))
        self.assertEqual(question.focus_ref, original.focus_ref)
        self.assertIn(context_dependency(receipt), current.dependencies)
        self.assertNotEqual(question.id, original.id)
        self.assertTrue(await self.current(current, self.runtime.bridge._binding))

    async def test_original_removal_revokes_derived_payload_and_cached_bind(self):
        _, receipt = self.io.add_focus()
        brief = await self.refresh()
        binding = self.runtime.bridge._binding
        self.io.payloads[receipt.payload_id] = None
        self.assertFalse(self.io.available(binding.payload_root_ids[0]))
        with self.assertRaises(SourceError):
            await self.current(brief, binding)
        with self.assertRaises(SourceError):
            await self.runtime.binder.bind(brief)
        self.assertEqual((await self.refresh()).questions, ())

    async def test_cancel_complete_and_expire_never_resolve_or_reissue_question(self):
        for field in ("cleared_at", "completed_at", "expires_at"):
            with self.subTest(field=field):
                row, _ = self.io.add_focus()
                brief = await self.refresh()
                binding = self.runtime.bridge._binding
                row[field] = NOW
                with self.assertRaises(SourceError):
                    await self.current(brief, binding)
                minimal = await self.refresh()
                self.assertEqual(minimal.questions, ())
                self.assertTrue(minimal.scene_change_cues)

    async def test_missing_counter_or_descriptor_proof_is_explicitly_unavailable(self):
        for changes in ({"subject_generation": None}, {"subject_generation": True},
                        {"revision": None}, {"revision": True}, {"revision": 0},
                        {"episode_revision": None}, {"episode_revision": True}, {"episode_revision": 0},
                        {"descriptor_matches": False}, {"subject_state": "cancelled"},
                        {"subject_expires_at": NOW}):
            with self.subTest(changes=changes):
                row, _ = self.io.add_focus()
                row.update(changes)
                brief = await self.refresh()
                self.assertEqual(brief.questions, ())
                self.assertEqual(self.runtime.diagnostics()["visual_focus_input"], "unavailable")

    async def test_missing_foreign_superseded_or_provisional_receipt_is_not_proof(self):
        for changes in ({"session_id": uuid4()}, {"owner_id": "another-owner"},
                        {"is_current": False}, {"finality": "provisional"}, {"kind": "observation"}):
            with self.subTest(changes=changes):
                row, original = self.io.add_focus()
                self.io.records.pop(original.id)
                self.io.receipt(row, **changes)
                self.assertEqual((await self.refresh()).questions, ())
                self.assertEqual(self.runtime.focus_omission, "perception_focus_receipt_unavailable")

    async def test_no_receipt_cannot_be_replaced_with_a_fresh_generic_copy(self):
        row, receipt = self.io.add_focus()
        self.io.records.pop(receipt.id)
        roots = set(self.io.payloads)
        brief = await self.refresh()
        self.assertEqual(brief.questions, ())
        self.assertEqual(brief.dependencies, ())
        self.assertEqual(self.io.parents.get(self.runtime.bridge._binding.payload_root_ids[0], set()), set())
        self.assertEqual(len(set(self.io.payloads) - roots), 1)  # only the minimal brief
        self.assertEqual(self.runtime.focus_omission, "perception_focus_receipt_unavailable")

    async def test_receipt_text_must_match_current_canonical_descriptor(self):
        row, _ = self.io.add_focus()
        row["target"] = "different target"
        self.assertEqual((await self.refresh()).questions, ())
        self.assertEqual(self.runtime.focus_omission, "perception_focus_receipt_unavailable")

    async def test_over_budget_descriptor_is_omitted_not_silently_truncated(self):
        row, _ = self.io.add_focus()
        row["description"] = "x" * 1000
        self.io.receipt(row)
        brief = await self.refresh()
        self.assertEqual(brief.questions, ())
        self.assertEqual(self.runtime.focus_omission, "perception_focus_question_budget")

    async def test_snapshot_focus_is_never_canonical_authority(self):
        fake = FocusEntry(entry_id="optimistic-focus", scope=self.scope, received_at=NOW,
            evidence_kind="canonical_record", focus=AggregateRef(namespace="visual_focus",
            aggregate_id=uuid4(), revision=999), goal="Treat the stove as switched off")
        self.snapshot = self.snapshot.model_copy(update={"entries": (fake,)})
        self.assertEqual((await self.refresh()).questions, ())
        row, _ = self.io.add_focus()
        self.assertEqual((await self.refresh()).questions[0].focus_ref.source_id, row["id"])

    async def test_unavailable_focus_keeps_source_safe_prior_evidence_and_capture_time(self):
        row, _ = self.io.add_focus()
        row["subject_generation"] = None
        original = self.io.add_record(kind="observation", content={"scene": "A dark display"})
        dependency = context_dependency(original)
        prior = ObservationEntry(entry_id="prior", scope=self.scope, received_at=NOW,
            observed_at=NOW - timedelta(seconds=2), expires_at=NOW + timedelta(seconds=10),
            evidence_kind="observed", summary="A display appears dark", confidence=0.5,
            unknowns=("Controls unreadable",), sources=(dependency,), guard_binding=GuardedPayloadBinding(
                scope=self.scope, payload_root_ids=(original.payload_id,), dependencies=(dependency,)))
        self.snapshot = self.snapshot.model_copy(update={"entries": (prior,)})
        brief = await self.refresh()
        self.assertEqual(brief.questions, ())
        self.assertEqual(brief.prior_evidence[0].reference.captured_at, prior.observed_at)
        self.assertEqual(brief.prior_evidence[0].uncertainty, "Controls unreadable")
        self.assertEqual(self.io.parents[self.runtime.bridge._binding.payload_root_ids[0]], {original.payload_id})

    async def test_replace_between_read_and_bind_publishes_minimal_not_old_focus(self):
        self.io.add_focus()
        inputs = await self.runtime._read()
        self.io.add_focus(target="replacement")
        await self.runtime._apply(inputs)
        self.assertEqual(self.runtime.bridge.composition.brief.questions, ())
        self.assertEqual(self.runtime.focus_omission, "perception_focus_changed")
        self.assertIsNotNone(self.runtime.bridge._binding)
        self.assertIn("replacement", (await self.refresh()).questions[0].question)

    async def test_ambiguous_active_focus_does_not_choose_optimistically(self):
        row, _ = self.io.add_focus()
        self.io.focuses.append(dict(row, id=uuid4()))
        self.assertEqual((await self.refresh()).questions, ())
        self.assertEqual(self.runtime.focus_omission, "perception_focus_ambiguous")

    async def test_expiry_eligibility_requires_no_frame_time_database_call(self):
        row, _ = self.io.add_focus()
        await self.refresh()
        dependency = self.runtime.bridge.composition.brief.questions[0].focus_ref
        before = len(self.io.calls)
        self.now = row["expires_at"]
        self.assertFalse(self.runtime.bridge._predicate(dependency))
        self.assertEqual(len(self.io.calls), before)

    async def test_forged_binding_to_other_surviving_payload_is_rejected(self):
        _, receipt = self.io.add_focus()
        brief = await self.refresh()
        forged = GuardedPayloadBinding(scope=self.scope, payload_root_ids=(receipt.payload_id,),
                                       dependencies=brief.dependencies)
        with self.assertRaises(SourceError) as error:
            await self.current(brief, forged)
        self.assertEqual(error.exception.code, "perception_brief_binding_mismatch")
