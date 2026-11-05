"""Offline compact Live selection; no provider, database or media connections."""
import asyncio
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import UUID

from app.live_state.models import (
    AggregateRef, GuardedPayloadBinding, LiveCapabilities, LiveSnapshot,
    NotificationEntry, ObservationEntry, SessionScope, SourceDependency,
    StateCursor, TaskEntry,
)
from app.live_state.projection import prepare_voice_context, select_voice_context
from app.realtime.live_state_bridge import SessionLiveStateBridge
from app.voice_routes.bridge import ContextOfferResult, OfferDisposition
from app.voice_routes.contracts import RouteIdentity
from test_live_state_runtime import RuntimeFixtures, changed


NOW = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
SCOPE = SessionScope(owner_id="compact-fixture", session_id=UUID(int=1))
CURSOR = StateCursor(epoch=UUID(int=2), revision=12)
ROUTE = RouteIdentity(scope=SCOPE, route_id=UUID(int=3), route_generation=1)


def entry(number=1, *, summary="The number is 42.", kind="observation", **changes):
    source = SourceDependency(source_namespace="context_event", source_id=UUID(int=100 + number),
                              revision=1, generation=2)
    fields = dict(entry_id=f"entry-{number}", scope=SCOPE, received_at=NOW,
        observed_at=NOW, fresh_until=NOW + timedelta(seconds=15),
        expires_at=NOW + timedelta(seconds=30), evidence_kind="observed", summary=summary,
        sources=(source,), guard_binding=GuardedPayloadBinding(scope=SCOPE,
            payload_root_ids=(UUID(int=200 + number),), dependencies=(source,)))
    if kind == "task":
        fields.update(status="running", task=AggregateRef(namespace="task",
            aggregate_id=UUID(int=300 + number), revision=1))
        model = TaskEntry
    elif kind == "notification":
        fields.update(status="pending", intent_id=UUID(int=300 + number), intent_generation=1)
        model = NotificationEntry
    else:
        model = ObservationEntry
    return model(**(fields | changes))


def snapshot(*entries, **changes):
    return LiveSnapshot(**(dict(scope=SCOPE, cursor=CURSOR, as_of=NOW, readiness="ready",
                               entries=entries) | changes))


def select(value, *, budget=500, **kwargs):
    return select_voice_context(value, now=NOW, max_encoded_bytes=budget, **kwargs)


def encoded_size(value):
    # Independent wire fixture: LiveVoiceRoute._content and live_session's guard.
    wire = {"kind": "factual_context", "text": value.text, "mode": "snapshot"}
    return len(json.dumps(wire, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))


class CompactProjectionTests(unittest.TestCase):
    def test_default_realtime_text_and_metadata_are_unchanged(self):
        item = entry()
        value = snapshot(item)
        legacy = select_voice_context(value, now=NOW)
        self.assertEqual(legacy, select(value, budget=None))
        self.assertEqual(legacy.text,
            "[observation; observed; observed_at=2026-10-30T12:00:00+00:00] The number is 42.")
        self.assertEqual(legacy.dependencies, item.sources)
        self.assertEqual(legacy.parent_payload_root_ids, item.guard_binding.payload_root_ids)
        self.assertEqual(legacy.expires_at, item.fresh_until)
        self.assertEqual(select_voice_context(snapshot(), now=NOW).text,
                         "Context availability: no current source-backed observations.")
        self.assertEqual(select_voice_context(snapshot(readiness="warming"), now=NOW).text,
                         "Context availability: warming.")

    def test_exact_wire_budget_preserves_complete_negation_units_and_uncertainty(self):
        item = entry(summary="Not 42: the reading is 4.2 \u00b5A.",
                     unknowns=("Last digit is uncertain.",))
        complete = select(snapshot(item), budget=4096)
        result = select(snapshot(item), budget=encoded_size(complete))
        self.assertEqual(result, complete)
        self.assertIn(item.summary + " Uncertainties: " + item.unknowns[0], result.text)
        self.assertEqual(result.dependencies, item.sources)
        self.assertEqual(result.parent_payload_root_ids, item.guard_binding.payload_root_ids)
        self.assertEqual(result.expires_at, item.fresh_until)
        self.assertEqual(result.selected_entry_ids, (item.entry_id,))
        omitted = select(snapshot(item), budget=encoded_size(complete) - 1)
        self.assertEqual(omitted.selected_entry_ids, ())
        self.assertNotIn("4.2", omitted.text)
        self.assertNotIn("Last digit", omitted.text)

    def test_oversized_multibyte_entry_is_omitted_whole_and_next_entry_can_fit(self):
        huge = entry(summary="\U0001f50e\u6e2c\u8a66" * 300)
        small = entry(2, summary="The switch state is unknown.")
        result = select(snapshot(huge, small))
        self.assertLessEqual(encoded_size(result), 500)
        self.assertEqual(result.selected_entry_ids, (small.entry_id,))
        self.assertEqual(result.omitted_entry_ids, (huge.entry_id,))
        self.assertNotIn("\U0001f50e", result.text)
        self.assertIn(small.summary, result.text)
        self.assertEqual(result.dependencies, small.sources)
        self.assertEqual(result.parent_payload_root_ids, small.guard_binding.payload_root_ids)
        self.assertIn("omitted=1", result.text)
        self.assertIn("not absence.", result.text)

    def test_json_escaping_not_raw_utf8_is_the_budget(self):
        item = entry(summary='Not confirmed: "' + "\U0001f50e" * 18 + '"\\')
        complete = select(snapshot(item), budget=4096)
        wire = {"kind": "factual_context", "text": complete.text, "mode": "snapshot"}
        raw_utf8_size = len(json.dumps(wire, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        self.assertLess(raw_utf8_size, encoded_size(complete))
        result = select(snapshot(item), budget=raw_utf8_size)
        self.assertEqual(result.selected_entry_ids, ())
        self.assertLessEqual(encoded_size(result), raw_utf8_size)

    def test_oversized_unknowns_cannot_leave_a_confident_fragment(self):
        item = entry(summary="Possibly switched off.", unknowns=("Not confirmed. " * 15,))
        result = select(snapshot(item))
        self.assertNotIn(item.summary, result.text)
        self.assertEqual(result.omitted_entry_ids, (item.entry_id,))

    def test_mandatory_readiness_and_capability_gaps_survive_empty_selection(self):
        for readiness in ("ready", "warming", "degraded"):
            with self.subTest(readiness=readiness):
                value = snapshot(entry(summary="x" * 2000), readiness=readiness,
                    capabilities=LiveCapabilities(durable_responsibilities="disabled",
                                                  historical_retrieval="unavailable"))
                result = select(value)
                self.assertIn(f"Readiness={readiness}", result.text)
                self.assertIn("responsibilities=disabled", result.text)
                self.assertIn("history=unavailable", result.text)
                self.assertIn("omitted=1", result.text)
                self.assertIn("not absence", result.text)
                self.assertNotIn("no current", result.text)
                self.assertEqual(result.omitted_entry_ids, ("entry-1",))
                self.assertFalse(result.dependencies)
                self.assertFalse(result.parent_payload_root_ids)
                self.assertLessEqual(encoded_size(result), 500)

    def test_mandatory_notices_cannot_be_truncated_for_too_small_budget(self):
        value = snapshot(readiness="degraded")
        complete = select(value)
        self.assertEqual(select(value, budget=encoded_size(complete)), complete)
        with self.assertRaisesRegex(ValueError, "mandatory availability and omission"):
            select(value, budget=encoded_size(complete) - 1)

    def test_budget_must_be_an_explicit_positive_integer(self):
        for budget in (0, -1, True, 500.0, "500"):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                select(snapshot(), budget=budget)

    def test_priority_and_max_entries_still_apply_without_claiming_missing_tasks(self):
        notification = entry(3, kind="notification", summary="A result is ready.")
        observation = entry(2)
        task = entry(1, kind="task", summary="Still checking the code.")
        result = select(snapshot(task, observation, notification), budget=4096, max_entries=1)
        self.assertEqual(result.selected_entry_ids, (notification.entry_id,))
        self.assertEqual(result.omitted_entry_ids, (observation.entry_id, task.entry_id))
        self.assertIn("omitted=2", result.text)
        self.assertNotIn("Still checking", result.text)
        self.assertIn("not absence", result.text)

    def test_selected_task_keeps_its_status_and_complete_result(self):
        task = entry(kind="task", summary="Not finished; 3 of 5 files checked.")
        result = select(snapshot(task))
        self.assertIn("[task; running; observed_at=", result.text)
        self.assertIn(task.summary, result.text)
        self.assertEqual(result.dependencies, task.sources)

    def test_expired_retracted_and_unbound_entries_stay_omitted(self):
        expired = entry(1, fresh_until=NOW)
        retracted = entry(2, validity="retracted")
        unbound = entry(3, guard_binding=None)
        current = entry(4)
        result = select(snapshot(expired, retracted, unbound, current))
        self.assertEqual(result.selected_entry_ids, (current.entry_id,))
        self.assertEqual(result.omitted_entry_ids, (expired.entry_id, retracted.entry_id, unbound.entry_id))
        self.assertEqual(result.dependencies, current.sources)
        self.assertEqual(result.expires_at, current.fresh_until)

    def test_conflicting_source_version_is_not_merged_into_selection(self):
        first = entry(1)
        newer = first.sources[0].model_copy(update={"generation": 3})
        conflict = entry(2, sources=(newer,), guard_binding=GuardedPayloadBinding(
            scope=SCOPE, payload_root_ids=(UUID(int=222),), dependencies=(newer,)))
        result = select(snapshot(first, conflict), budget=4096)
        self.assertEqual(result.selected_entry_ids, (first.entry_id,))
        self.assertEqual(result.omitted_entry_ids, (conflict.entry_id,))
        self.assertEqual(result.dependencies, first.sources)

    def test_selection_is_deterministic_and_does_not_mutate_snapshot(self):
        value = snapshot(entry(2), entry(1, summary="\u6e2c" * 1000))
        before = value.model_dump()
        self.assertEqual(select(value), select(value))
        self.assertEqual(before, value.model_dump())
        self.assertEqual(select(value).cursor, value.cursor)

    def test_cursor_stays_local_and_char_bound_is_included(self):
        value = snapshot(entry(), cursor=CURSOR.model_copy(update={"revision": 10 ** 40}))
        complete = select(value, budget=4096)
        self.assertEqual(complete.cursor, value.cursor)
        self.assertNotIn(str(value.cursor.epoch), complete.text)
        self.assertEqual(encoded_size(complete), encoded_size(select(snapshot(entry()), budget=4096)))
        self.assertEqual(select(value, budget=encoded_size(complete)), complete)
        self.assertFalse(select(value, budget=encoded_size(complete) - 1).selected_entry_ids)
        with self.assertRaisesRegex(ValueError, "mandatory"):
            select(value, max_chars=128)


class CompactSourceInvalidationTests(RuntimeFixtures, unittest.TestCase):
    def test_authoritative_withdrawal_cannot_resurrect_compact_evidence(self):
        state = self.apply(self.active(), self.upsert(2))
        before = select_voice_context(changed(state.snapshot, readiness="ready"),
                                      now=NOW, max_encoded_bytes=500)
        self.assertEqual(before.selected_entry_ids, ("observation",))
        withdrawn = changed(self.source, generation=4)
        state = self.apply(state, self.watermark(3, withdrawn, "retracted"))
        after = select_voice_context(changed(state.snapshot, readiness="ready"),
                                     now=NOW, max_encoded_bytes=500)
        self.assertFalse(after.dependencies)
        self.assertFalse(after.parent_payload_root_ids)
        self.assertNotIn("42", after.text)
        from app.live_state.reducer import reduce_event
        self.assertEqual(reduce_event(state, self.upsert(4), now=NOW).reason,
                         "dependency_not_confirmed")


class CompactBridgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.current = snapshot(entry(1, summary="\U0001f50e" * 300), entry(2))
        self.bound = []

        async def bind(selection):
            self.bound.append(selection)
            return GuardedPayloadBinding(scope=SCOPE, payload_root_ids=(UUID(int=999),),
                                         dependencies=selection.dependencies)

        self.live = SimpleNamespace(requires_reconstruction=False,
            service=SimpleNamespace(snapshot=lambda: self.current), bind=AsyncMock(side_effect=bind))
        self.session = SimpleNamespace(_closed=asyncio.Event(), close=AsyncMock())
        self.route = SimpleNamespace(ready=True, request=SimpleNamespace(route=ROUTE),
            offer_context=Mock(return_value=ContextOfferResult(OfferDisposition.QUEUED)))
        self.bridge = SessionLiveStateBridge(self.session, self.live, self.route,
            clock=lambda: NOW, selection_max_encoded_bytes=500)

    async def test_both_selection_passes_use_budget_before_binding_and_publication(self):
        await self.bridge.publish_latest()
        self.live.bind.assert_awaited_once()
        self.route.offer_context.assert_called_once()
        selection = self.bound[0]
        update = self.route.offer_context.call_args.args[0]
        self.assertEqual(selection.selected_entry_ids, ("entry-2",))
        self.assertEqual(update.text, selection.text)
        self.assertEqual(update.dependencies, selection.dependencies)
        self.assertEqual(update.guard_binding.dependencies, selection.dependencies)
        self.assertLessEqual(encoded_size(selection), 500)
        self.assertIsNone(self.bridge.last_omission)

    async def test_default_bridge_still_binds_original_realtime_projection(self):
        bridge = SessionLiveStateBridge(self.session, self.live, self.route, clock=lambda: NOW)
        await bridge.publish_latest()
        self.assertEqual(self.bound[0], select_voice_context(self.current, now=NOW))
        self.assertEqual(self.bound[0].selected_entry_ids, ("entry-1", "entry-2"))

    async def test_selected_evidence_changes_during_binding_are_not_published(self):
        async def bind(selection):
            self.current = snapshot(entry(2, summary="Correction: not 42."),
                                    cursor=CURSOR.model_copy(update={"revision": 13}))
            return GuardedPayloadBinding(scope=SCOPE, payload_root_ids=(UUID(int=999),),
                                         dependencies=selection.dependencies)
        self.live.bind.side_effect = bind
        await self.bridge.publish_latest()
        self.route.offer_context.assert_not_called()
        self.assertEqual(self.bridge.last_omission, "selection_changed_during_binding")

    async def test_capability_gap_change_during_binding_requires_new_selection(self):
        async def bind(selection):
            self.current = self.current.model_copy(update={"capabilities": LiveCapabilities(
                durable_responsibilities="disabled", historical_retrieval="disabled")})
            return GuardedPayloadBinding(scope=SCOPE, payload_root_ids=(UUID(int=999),),
                                         dependencies=selection.dependencies)
        self.live.bind.side_effect = bind
        await self.bridge.publish_latest()
        self.route.offer_context.assert_not_called()
        self.assertEqual(self.bridge.last_omission, "selection_changed_during_binding")

    async def test_withdrawal_while_binding_drops_context_without_stripping_bound_text(self):
        async def bind(selection):
            self.current = snapshot(cursor=CURSOR.model_copy(update={"revision": 13}))
            return GuardedPayloadBinding(scope=SCOPE, payload_root_ids=(UUID(int=999),),
                                         dependencies=selection.dependencies)
        self.live.bind.side_effect = bind
        await self.bridge.publish_latest()
        self.route.offer_context.assert_not_called()
        self.assertEqual(self.bridge.last_omission, "selection_changed_during_binding")

    async def test_all_omitted_context_reports_gaps_without_binding_unselected_sources(self):
        self.current = snapshot(entry(summary="\u6e2c" * 2000))
        await self.bridge.publish_latest()
        self.live.bind.assert_not_awaited()
        update = self.route.offer_context.call_args.args[0]
        self.assertIsNone(update.guard_binding)
        self.assertFalse(update.dependencies)
        self.assertIn("omitted=1", update.text)
        self.assertIn("history=unavailable", update.text)

    async def test_unready_route_never_binds_or_publishes(self):
        self.route.ready = False
        await self.bridge.publish_latest()
        self.live.bind.assert_not_awaited()
        self.route.offer_context.assert_not_called()

    async def test_prepare_binds_exact_selected_document_and_expiry_still_applies(self):
        selection = select(self.current)
        update = await prepare_voice_context(selection, ROUTE, bind=self.live.bind, clock=lambda: NOW)
        self.assertEqual(self.bound, [selection])
        self.assertEqual(update.text, selection.text)
        self.assertEqual(update.expires_at, selection.expires_at)
        self.live.bind.reset_mock()
        expired = await prepare_voice_context(selection, ROUTE, bind=self.live.bind,
                                             clock=lambda: NOW + timedelta(seconds=15))
        self.assertIsNone(expired)
        self.live.bind.assert_not_awaited()
