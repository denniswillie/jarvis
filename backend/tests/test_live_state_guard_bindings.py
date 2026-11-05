"""Canonical owner and guarded-root compatibility; schemas do not authorize reads."""
from datetime import UTC, datetime
import unittest
from uuid import uuid4

from pydantic import ValidationError

from app.live_state.models import GuardedPayloadBinding, SelectedMemoryEntry, SessionScope, SourceDependency, StateCursor
from app.voice_routes.contracts import HistoryTurn, RouteIdentity, RouteStart, VoiceContextUpdate


NOW = datetime(2026, 10, 30, tzinfo=UTC)


class GuardBindingTests(unittest.TestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="fixture@example.test", session_id=uuid4(), area_id=uuid4())
        self.source = SourceDependency(source_namespace="knowledge_source", source_id=uuid4(), revision=1, generation=2)
        self.binding = GuardedPayloadBinding(scope=self.scope, payload_root_ids=(uuid4(),), dependencies=(self.source,))
        self.route = RouteIdentity(scope=self.scope, route_id=uuid4(), route_generation=1)

    def context_fields(self):
        return dict(route=self.route, update_id=uuid4(), cursor=StateCursor(epoch=uuid4(), revision=1),
                    mode="snapshot", text="Synthetic source-backed context", dependencies=(self.source,), created_at=NOW)

    def memory_fields(self):
        return dict(entry_id="memory", scope=self.scope, received_at=NOW, summary="Synthetic memory",
                    evidence_kind="user_reported", sources=(self.source,))

    def test_canonical_owner_domain_is_not_an_identifier_token(self):
        for owner in ("fixture@example.test", "fixture with spaces", "x" * 200, "fixtur\u00e9"):
            with self.subTest(owner_length=len(owner)):
                scope = SessionScope(owner_id=owner, session_id=uuid4())
                self.assertEqual(SessionScope.model_validate_json(scope.model_dump_json()), scope)
        for owner in ("", " leading", "trailing ", "x" * 201, 123, "bad\x00owner"):
            with self.subTest(owner=repr(owner)), self.assertRaises(ValidationError):
                SessionScope(owner_id=owner, session_id=uuid4())

    def test_roots_roundtrip_without_invented_payload_revisions(self):
        value = GuardedPayloadBinding.model_validate_json(self.binding.model_dump_json())
        self.assertEqual(value, self.binding)
        self.assertEqual(set(value.model_dump()), {"scope", "payload_root_ids", "dependencies"})
        self.assertEqual(value.dependencies[0].generation, 2)
        with self.assertRaises(ValidationError):
            GuardedPayloadBinding(**(self.binding.model_dump() | {"payload_revision": 1}))

    def test_roots_are_required_unique_and_bounded(self):
        for roots in ((), self.binding.payload_root_ids * 2, tuple(uuid4() for _ in range(65))):
            with self.subTest(count=len(roots)), self.assertRaises(ValidationError):
                GuardedPayloadBinding(scope=self.scope, payload_root_ids=roots)

    def test_entry_binding_requires_exact_scope_and_dependency_generation(self):
        entry = SelectedMemoryEntry(**self.memory_fields(), guard_binding=self.binding)
        self.assertEqual(entry.guard_binding.payload_root_ids, self.binding.payload_root_ids)
        other_scope = SessionScope(owner_id="other", session_id=self.scope.session_id, area_id=self.scope.area_id)
        stale = SourceDependency(**(self.source.model_dump() | {"generation": 1}))
        for binding in (
            GuardedPayloadBinding(scope=other_scope, payload_root_ids=(uuid4(),), dependencies=(self.source,)),
            GuardedPayloadBinding(scope=self.scope, payload_root_ids=(uuid4(),), dependencies=(stale,)),
        ):
            with self.assertRaises(ValidationError):
                SelectedMemoryEntry(**self.memory_fields(), guard_binding=binding)

    def test_unbound_selection_cannot_be_published_as_source_bearing_voice_context(self):
        self.assertIsNone(SelectedMemoryEntry(**self.memory_fields()).guard_binding)
        with self.assertRaises(ValidationError):
            VoiceContextUpdate(**self.context_fields())
        update = VoiceContextUpdate(**self.context_fields(), guard_binding=self.binding)
        self.assertEqual(VoiceContextUpdate.model_validate_json(update.model_dump_json()), update)
        with self.assertRaises(ValidationError):
            VoiceContextUpdate(**(self.context_fields() | {"dependencies": ()}), guard_binding=self.binding)

    def test_source_free_briefing_needs_no_invented_root(self):
        update = VoiceContextUpdate(**(self.context_fields() | {"dependencies": (), "text": "Context is warming."}))
        self.assertIsNone(update.guard_binding)

    def test_history_cannot_bypass_binding_or_route_scope(self):
        with self.assertRaises(ValidationError):
            HistoryTurn(role="user", text="Synthetic", dependencies=(self.source,))
        turn = HistoryTurn(role="user", text="Synthetic", dependencies=(self.source,), guard_binding=self.binding)
        context = VoiceContextUpdate(**self.context_fields(), guard_binding=self.binding)
        start_fields = dict(route=self.route, command_id=uuid4(), requested_at=NOW,
                            authorization_ref=uuid4(), initial_context=context, relevant_history=(turn,))
        self.assertEqual(RouteStart(**start_fields).relevant_history[0], turn)
        other = SessionScope(owner_id="other", session_id=uuid4())
        wrong = HistoryTurn(role="user", text="Synthetic", dependencies=(self.source,),
            guard_binding=GuardedPayloadBinding(scope=other, payload_root_ids=(uuid4(),), dependencies=(self.source,)))
        with self.assertRaises(ValidationError):
            RouteStart(**(start_fields | {"relevant_history": (wrong,)}))

    def test_binding_does_not_claim_current_availability_or_hearing(self):
        for invented in ({"currently_available": True}, {"human_heard": True}, {"authorization": "allow"}):
            with self.assertRaises(ValidationError):
                GuardedPayloadBinding(**(self.binding.model_dump() | invented))
