"""Offline canonical focus partition/bind/effect checks, not owner settlement.

The SQL fixture stores the actual full brief and removable receipt. The mixed
owner test spies on the responsibility proof boundary; the separate existing
episode integration suite exercises the real responsibility owner.
"""
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceError
from app.live_state.canonical import CanonicalContextBinder, context_dependency
from app.live_state.episode_context import EPISODE_NAMESPACES
from app.live_state.models import LiveSnapshot, SessionScope, SourceDependency, StateCursor
from app.live_state.perception import (
    CheckResolution, DurableResponsibilityCapability, EvidenceReference, PerceptionBrief,
)
from app.live_state.perception_focus import CanonicalFocusInputs, FOCUS_CHECK, FOCUS_SUBJECT
from app.live_state.perception_runtime import CanonicalPerceptionBinder, SessionPerceptionRuntime
from app.runtime.events import DomainEventBus
from tests.test_session_perception_focus import FocusPersistence, NOW


class TypedFocusPersistence(FocusPersistence):
    def add_focus(self, **kwargs):
        row, receipt = super().add_focus(**kwargs)
        row.setdefault("revision", 1)
        row.setdefault("episode_revision", 1)
        return row, receipt


class FocusPartitionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="focus-partition", session_id=uuid4())
        self.now = NOW
        self.io = TypedFocusPersistence(self.scope)
        self.canonical = CanonicalContextBinder(
            SimpleNamespace(pool=None, transaction=self.io.transaction), self.scope,
            clock=lambda: self.now)
        self.inputs = CanonicalFocusInputs(self.canonical)
        self.binder = CanonicalPerceptionBinder(self.canonical, focus=self.inputs)

    async def assignment(self):
        self.row, self.receipt = self.io.add_focus()
        self.selected = await self.inputs.read()
        return PerceptionBrief(id=uuid4(), version=1, scope=self.scope, scope_epoch=uuid4(),
            based_on_live_revision=1, questions=(self.selected.question,),
            dependencies=self.selected.dependencies,
            durable_responsibility=DurableResponsibilityCapability(
                owner_namespace="responsibility", status="unavailable"),
            created_at=self.now, fresh_until=self.now + timedelta(seconds=5),
            expires_at=self.now + timedelta(seconds=10))

    async def test_focus_without_responsibility_owner_binds_and_guards_full_document(self):
        brief = await self.assignment()
        binding = await self.binder.bind(brief)
        root, = binding.payload_root_ids
        self.assertIsNone(self.canonical.episode_context)
        self.assertEqual(binding.dependencies, brief.dependencies)
        self.assertEqual(self.io.parents[root], {self.receipt.payload_id})
        self.assertEqual(self.io.payloads[root]["brief"], brief.model_dump(mode="json"))
        self.assertEqual(self.io.event_edges[0][2:],
            (self.receipt.id, self.receipt.revision, self.receipt.payload_id, "current"))
        async with self.binder.guard(brief, binding) as connection:
            self.assertIs(connection, self.io)
            self.assertTrue(connection.is_in_transaction())
        self.assertEqual(await self.binder.bind(brief), binding)
        self.assertFalse(any("jarvis_responsibility_occurrences" in sql for sql, _ in self.io.calls))

    async def test_focus_episode_never_reaches_even_an_available_responsibility_owner(self):
        brief = await self.assignment()

        def unexpected():
            self.fail("A focus-owned episode reached the responsibility owner")

        self.canonical.episode_context = SimpleNamespace(owner=object(), require_owner=unexpected,
            validate_assignment=AsyncMock())
        binding = await self.binder.bind(brief)
        async with self.binder.guard(brief, binding):
            pass
        self.canonical.episode_context.validate_assignment.assert_not_awaited()

    async def test_mixed_partition_retains_responsibility_refs_and_full_original_brief(self):
        brief = await self.assignment()
        responsibility_refs = tuple(SourceDependency(source_namespace=namespace,
            source_id=uuid4(), revision=2) for namespace in
            ("responsibility", "responsibility_check", "episode"))
        brief = brief.model_copy(update={"dependencies": (*brief.dependencies, *responsibility_refs)})
        proof = SimpleNamespace(source_payload_ids=(), dependencies=())
        self.canonical.episode_context = SimpleNamespace(owner=object(), validate_assignment=AsyncMock())
        original = self.canonical._validate_dependencies

        async def canonical_proof(connection, dependencies):
            self.assertEqual(tuple(ref for ref in dependencies if ref.source_namespace in EPISODE_NAMESPACES),
                             responsibility_refs)
            self.assertNotIn(self.selected.question.episode_or_object_ref, dependencies)
            await original(connection, tuple(ref for ref in dependencies
                if ref.source_namespace not in EPISODE_NAMESPACES))
            return (proof,)

        self.canonical._validate_dependencies = AsyncMock(side_effect=canonical_proof)
        validate = self.inputs.validate = AsyncMock(wraps=self.inputs.validate)
        binding = await self.binder.bind(brief)
        async with self.binder.guard(brief, binding):
            pass
        self.assertEqual(self.io.payloads[binding.payload_root_ids[0]]["brief"], brief.model_dump(mode="json"))
        for call in validate.await_args_list:
            self.assertIs(call.args[1], brief)
        for call in self.canonical.episode_context.validate_assignment.await_args_list:
            self.assertIs(call.args[1], brief)
            self.assertEqual(call.args[2], (proof,))
        self.assertGreaterEqual(self.canonical.episode_context.validate_assignment.await_count, 4)

    async def test_arbitrary_episode_is_not_whitelisted_by_focus_assignment(self):
        brief = await self.assignment()
        orphan = SourceDependency(source_namespace="episode", source_id=uuid4(), revision=1)
        with self.assertRaises(SourceError) as error:
            await self.binder.bind(brief.model_copy(update={"dependencies": (*brief.dependencies, orphan)}))
        self.assertEqual(error.exception.code, "perception_brief_dependency_unsupported")

    async def test_missing_subject_still_invokes_focus_owner_validation(self):
        brief = await self.assignment()
        malformed = brief.model_copy(update={"dependencies": tuple(ref for ref in brief.dependencies
            if ref.source_namespace != FOCUS_SUBJECT)})
        validate = self.inputs.validate = AsyncMock(wraps=self.inputs.validate)
        with self.assertRaises(SourceError) as error:
            await self.binder.bind(malformed)
        self.assertEqual(error.exception.code, "perception_focus_dependency_invalid")
        validate.assert_awaited_once_with(self.io, malformed)

    async def test_orphan_partial_duplicate_and_wrong_version_refs_cannot_materialize(self):
        brief = await self.assignment()
        question, = brief.questions
        bad_episode = question.episode_or_object_ref.model_copy(update={"revision": 99})
        cases = {
            "missing_check": brief.model_copy(update={"dependencies": tuple(ref for ref in brief.dependencies
                if ref.source_namespace != FOCUS_CHECK)}),
            "missing_episode": brief.model_copy(update={"dependencies": tuple(ref for ref in brief.dependencies
                if ref.source_namespace != "episode")}),
            "missing_receipt": brief.model_copy(update={"dependencies": tuple(ref for ref in brief.dependencies
                if ref.source_namespace != "context_event")}),
            "no_question": brief.model_copy(update={"questions": ()}),
            "extra_check": brief.model_copy(update={"dependencies": (*brief.dependencies,
                SourceDependency(source_namespace=FOCUS_CHECK, source_id=uuid4(), revision=1))}),
            "duplicate_check": brief.model_copy(update={"dependencies": (*brief.dependencies, question.check_ref)}),
            "wrong_episode": brief.model_copy(update={
                "questions": (question.model_copy(update={"episode_or_object_ref": bad_episode}),),
                "dependencies": tuple(bad_episode if ref == question.episode_or_object_ref else ref
                                      for ref in brief.dependencies)}),
            "invented_question": brief.model_copy(update={"questions": (question.model_copy(update={"id": uuid4()}),)}),
        }
        roots = set(self.io.payloads)
        for label, malformed in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(SourceError):
                    await self.binder.bind(malformed)
                self.assertEqual(set(self.io.payloads), roots)

    async def test_absent_focus_adapter_cannot_admit_focus_references(self):
        brief = await self.assignment()
        with self.assertRaises(SourceError) as error:
            await CanonicalPerceptionBinder(self.canonical).bind(brief)
        self.assertEqual(error.exception.code, "perception_brief_dependency_unsupported")

    async def test_effect_boundary_rechecks_versions_cancel_expiry_and_original(self):
        for change in ("revision", "episode_revision", "subject_generation", "cancel", "expiry", "source"):
            with self.subTest(change=change):
                brief = await self.assignment()
                binding = await self.binder.bind(brief)
                if change in ("revision", "episode_revision", "subject_generation"):
                    self.row[change] += 1
                elif change == "cancel":
                    self.row["cleared_at"] = self.now
                elif change == "expiry":
                    self.row["expires_at"] = self.now
                else:
                    self.io.payloads[self.receipt.payload_id] = None
                with self.assertRaises(SourceError):
                    async with self.binder.guard(brief, binding):
                        self.fail("A stale focus assignment admitted an effect")

    async def test_session_runtime_selects_typed_focus_and_reports_binding_only(self):
        row, receipt = self.io.add_focus()
        snapshot = LiveSnapshot(scope=self.scope, cursor=StateCursor(epoch=uuid4(), revision=1),
            as_of=self.now, readiness="ready")
        runtime = SessionPerceptionRuntime(service=SimpleNamespace(snapshot=lambda: snapshot),
            canonical=self.canonical, bus=DomainEventBus(), clock=lambda: self.now, enabled=True)
        try:
            inputs = await runtime._read()
            selected = inputs[2]
            await runtime._apply(inputs)
            brief = runtime.bridge.composition.brief
            question, = brief.questions
            self.assertEqual(question, selected.question)
            self.assertEqual(set(brief.dependencies), set(selected.dependencies))
            self.assertEqual(len(brief.dependencies), 4)
            self.assertEqual(brief.durable_responsibility.status, "unavailable")
            self.assertEqual(question.check_ref.source_id, row["id"])
            self.assertEqual(question.episode_or_object_ref.revision, row["episode_revision"])
            self.assertEqual(question.focus_ref.generation, row["subject_generation"])
            self.assertIn(context_dependency(receipt), brief.dependencies)
            self.assertEqual(runtime.diagnostics()["visual_focus_check_episode_binding"],
                             "supported_with_canonical_receipt")
            self.assertIsNone(row["completed_at"])
            async with runtime.binder.guard(brief, runtime.bridge._binding):
                pass
            snapshot = snapshot.model_copy(update={"cursor": StateCursor(epoch=snapshot.cursor.epoch, revision=999)})
            async with runtime.binder.guard(brief, runtime.bridge._binding):
                pass
        finally:
            await runtime.close()

    async def test_resolved_find_once_is_omitted_but_monitor_keeps_tracking(self):
        for mode in ("find_once", "monitor"):
            with self.subTest(mode=mode):
                row, _ = self.io.add_focus(mode=mode)
                original = self.io.add_record(kind="observation", content={"scene": "The requested keys are visible."})
                episode = SourceDependency(source_namespace="episode", source_id=row["id"],
                    revision=row["episode_revision"])
                resolution = CheckResolution(basis="visual_observation", resolved_at=self.now,
                    applicable_episode_or_object_ref=episode, source_refs=(EvidenceReference(
                        source=context_dependency(original), captured_at=self.now,
                        episode_or_object_ref=episode, frame_id=uuid4(), visibility="visible"),))
                row["check_resolution"] = resolution.model_dump(mode="json")
                row["revision"] += 1
                self.io.receipt(row)
                before = deepcopy(row)
                snapshot = LiveSnapshot(scope=self.scope, cursor=StateCursor(epoch=uuid4(), revision=1),
                    as_of=self.now, readiness="ready")
                runtime = SessionPerceptionRuntime(service=SimpleNamespace(snapshot=lambda: snapshot),
                    canonical=self.canonical, bus=DomainEventBus(), clock=lambda: self.now, enabled=True)
                try:
                    inputs = await runtime._read()
                    selected = inputs[2]
                    self.assertIsNotNone(selected)
                    await runtime._apply(inputs)
                    brief = runtime.bridge.composition.brief
                    self.assertEqual(brief.durable_responsibility.status, "unavailable")
                    self.assertIsNone(runtime.focus_omission)
                    self.assertEqual(row, before)
                    if mode == "find_once":
                        self.assertEqual(brief.questions, ())
                        self.assertEqual(brief.dependencies, ())
                        self.assertIsNone(runtime._focus)
                    else:
                        self.assertEqual(brief.questions, (selected.question,))
                        self.assertEqual(set(brief.dependencies), set(selected.dependencies))
                        self.assertEqual(runtime._focus, selected)
                    async with runtime.binder.guard(brief, runtime.bridge._binding):
                        pass
                finally:
                    await runtime.close()

    async def test_expiry_between_read_and_compose_drops_question_and_its_dependencies(self):
        row, _ = self.io.add_focus()
        snapshot = LiveSnapshot(scope=self.scope, cursor=StateCursor(epoch=uuid4(), revision=1),
            as_of=self.now, readiness="ready")
        runtime = SessionPerceptionRuntime(service=SimpleNamespace(snapshot=lambda: snapshot),
            canonical=self.canonical, bus=DomainEventBus(), clock=lambda: self.now, enabled=True)
        try:
            inputs = await runtime._read()
            self.assertIsNotNone(inputs[2])
            self.now = row["expires_at"]
            await runtime._apply(inputs)
            brief = runtime.bridge.composition.brief
            self.assertEqual(brief.questions, ())
            self.assertEqual(brief.dependencies, ())
            self.assertEqual(brief.durable_responsibility.status, "unavailable")
            self.assertIsNone(row["completed_at"])
            self.assertIsNone(row["check_resolution"])
            async with runtime.binder.guard(brief, runtime.bridge._binding):
                pass
        finally:
            await runtime.close()
