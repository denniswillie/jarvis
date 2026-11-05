"""P0 schema fixtures, not a claim that a reducer or source guard is implemented."""
from datetime import UTC, datetime, timedelta
import unittest
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from app.live_state.models import (
    AggregateRef, CapabilitiesChanged, CheckEntry, ConversationView, EntryRetracted,
    EntryUpserted, LiveCapabilities, LiveEvent, LiveSnapshot, ObservationEntry,
    ProducerCursor, RequestIdentity, SelectedMemoryEntry, SessionScope,
    SourceDependency, SourceStateChanged, SourceWatermark, StateCursor,
)


NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)


class LiveStateContractTests(unittest.TestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="fixture-owner", session_id=uuid4(), area_id=uuid4())
        self.source = SourceDependency(source_namespace="knowledge_source", source_id=uuid4(),
                                       revision=1, generation=1)
        self.producer = ProducerCursor(producer_id="fixture", producer_epoch=uuid4(),
            aggregate=AggregateRef(namespace="knowledge_source", aggregate_id=self.source.source_id,
                                   revision=1), source_generation=1)

    def observation(self, **changes):
        return ObservationEntry(**(dict(entry_id="observation-1", scope=self.scope,
            received_at=NOW, observed_at=NOW - timedelta(seconds=1), evidence_kind="observed",
            summary="Synthetic scene", sources=(self.source,)) | changes))

    def snapshot(self, **changes):
        return LiveSnapshot(**(dict(scope=self.scope, cursor=StateCursor(epoch=uuid4(), revision=0),
            as_of=NOW, readiness="warming", entries=(self.observation(),)) | changes))

    def event_fields(self):
        return dict(event_id=uuid4(), scope=self.scope, producer=self.producer, emitted_at=NOW)

    def test_json_roundtrip_retains_scope_tuples_and_source_generation(self):
        snapshot = self.snapshot()
        rebuilt = LiveSnapshot.model_validate_json(snapshot.model_dump_json())
        self.assertEqual(rebuilt, snapshot)
        self.assertIsInstance(rebuilt.entries, tuple)
        self.assertEqual(rebuilt.entries[0].sources[0].generation, 1)

    def test_nested_models_and_collections_are_immutable(self):
        snapshot = self.snapshot()
        for target, field, value in ((snapshot, "readiness", "ready"),
                                     (snapshot.entries[0], "summary", "changed"),
                                     (self.source, "generation", 2)):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                setattr(target, field, value)
        with self.assertRaises(TypeError):
            snapshot.entries[0] = self.observation()

    def test_same_content_revision_preserves_distinct_withdrawal_generation(self):
        withdrawn = SourceDependency(**(self.source.model_dump() | {"generation": 2}))
        self.assertEqual(withdrawn.revision, self.source.revision)
        self.assertNotEqual(withdrawn, self.source)
        event = SourceStateChanged(**self.event_fields(), watermark=SourceWatermark(
            source=withdrawn, availability="retracted", recorded_at=NOW))
        rebuilt = TypeAdapter(LiveEvent).validate_json(event.model_dump_json())
        self.assertEqual(rebuilt.watermark.source.generation, 2)
        self.assertEqual(rebuilt.watermark.availability, "retracted")

    def test_source_counter_is_not_a_process_epoch(self):
        self.assertIsNone(SourceDependency(source_namespace="fixture_without_lifecycle",
            source_id=uuid4(), revision=1).generation)
        with self.assertRaises(ValidationError):
            SourceDependency(**(self.source.model_dump() | {"generation": str(uuid4())}))

    def test_revisions_reject_bool_float_and_invalid_ranges(self):
        for value in (True, 1.5, "1", 0, -1):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                SourceDependency(**(self.source.model_dump() | {"revision": value}))
        self.assertEqual(StateCursor(epoch=uuid4(), revision=0).revision, 0)

    def test_no_cross_scope_snapshot_or_event(self):
        other = SessionScope(owner_id="other-owner", session_id=uuid4())
        entry = self.observation(scope=other)
        with self.assertRaises(ValidationError):
            self.snapshot(entries=(entry,))
        with self.assertRaises(ValidationError):
            EntryUpserted(**self.event_fields(), entry=entry)

    def test_duplicate_entries_and_conflicting_dependencies_are_rejected(self):
        with self.assertRaises(ValidationError):
            self.snapshot(entries=(self.observation(), self.observation()))
        newer = SourceDependency(**(self.source.model_dump() | {"generation": 2}))
        for dependencies in ((self.source, self.source), (self.source, newer)):
            with self.subTest(dependencies=dependencies), self.assertRaises(ValidationError):
                self.observation(sources=dependencies)

    def test_snapshot_and_text_bounds(self):
        with self.assertRaises(ValidationError):
            self.snapshot(entries=tuple(self.observation(entry_id=f"e-{i}") for i in range(129)))
        with self.assertRaises(ValidationError):
            self.observation(summary="x" * 2001)

    def test_freshness_cannot_outlive_expiry_or_use_naive_time(self):
        with self.assertRaises(ValidationError):
            self.observation(fresh_until=NOW + timedelta(seconds=2), expires_at=NOW)
        with self.assertRaises(ValidationError):
            self.observation(received_at=NOW.replace(tzinfo=None))

    def test_selected_memory_needs_provenance(self):
        fields = dict(entry_id="memory", scope=self.scope, received_at=NOW,
                      evidence_kind="user_reported", summary="Synthetic remembered report")
        with self.assertRaises(ValidationError):
            SelectedMemoryEntry(**fields, sources=())
        self.assertEqual(SelectedMemoryEntry(**fields, sources=(self.source,)).sources, (self.source,))

    def test_durable_check_needs_owner_and_expiry_is_not_completion(self):
        fields = dict(entry_id="check", scope=self.scope, received_at=NOW,
                      evidence_kind="canonical_record", check_id=uuid4(), check_revision=1,
                      lifecycle="expired", question="Is the item visible?")
        with self.assertRaises(ValidationError):
            CheckEntry(**fields, durable=True)
        owner = AggregateRef(namespace="responsibility", aggregate_id=uuid4(), revision=1)
        self.assertEqual(CheckEntry(**fields, durable=True, canonical_owner=owner).lifecycle, "expired")

    def test_capability_defaults_do_not_promise_persistent_tracking(self):
        self.assertEqual(self.snapshot().capabilities.durable_responsibilities, "unavailable")
        event = CapabilitiesChanged(**self.event_fields(), capabilities=LiveCapabilities(
            durable_responsibilities="disabled"))
        self.assertEqual(TypeAdapter(LiveEvent).validate_json(event.model_dump_json()), event)

    def test_event_union_roundtrip_and_extra_payload_rejection(self):
        adapter = TypeAdapter(LiveEvent)
        for event in (EntryUpserted(**self.event_fields(), entry=self.observation()),
                      EntryRetracted(**self.event_fields(), entry_id="observation-1", reason="expired")):
            self.assertEqual(adapter.validate_json(event.model_dump_json()), event)
        with self.assertRaises(ValidationError):
            self.observation(arbitrary_model_command="complete responsibility")

    def test_request_identity_is_canonical_not_provider_call_id(self):
        request = RequestIdentity(scope=self.scope, request_source=self.source,
                                  operation="delegate", slot="primary")
        self.assertEqual(request, RequestIdentity.model_validate_json(request.model_dump_json()))
        with self.assertRaises(ValidationError):
            RequestIdentity(**(request.model_dump() | {"provider_call_id": "call-1"}))

