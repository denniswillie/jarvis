"""Live freshness is historical evidence, not permission or remote erasure.

External provider/media/SQL are fixtures. These are not native acceptance tests.
"""
import asyncio
from datetime import datetime, timedelta, timezone
import json
import unittest
from uuid import uuid4

from pydantic import ValidationError

from app.knowledge.models import SourceError
from app.live_state.canonical import context_dependency
from app.live_state.models import GuardedPayloadBinding, ObservationEntry, SourceDependency
from app.live_state.projection import VoiceSelection, prepare_voice_context
from app.voice_routes.contracts import ObservationWindow, VoiceContextUpdate
import test_live_voice_adapter as adapter_fixtures
import test_live_session as session_fixtures


NOW = adapter_fixtures.NOW


def provenance(fixture, source=None):
    source = source or SourceDependency(source_namespace="context_event", source_id=uuid4(), revision=1)
    binding = GuardedPayloadBinding(scope=fixture.identity.scope,
        payload_root_ids=(uuid4(),), dependencies=(source,))
    window = ObservationWindow(observed_at=NOW - timedelta(seconds=5),
        fresh_until=NOW + timedelta(seconds=5), dependencies=(source,))
    return source, binding, window


def history(fixture, revision=0, source=None, **changes):
    source, binding, window = provenance(fixture, source)
    values = dict(text="Timestamped observation; after its freshness window the current view is unknown.",
        dependencies=(source,), guard_binding=binding, observation_history=(window,), expires_at=None)
    values.update(changes)
    return fixture.context(revision, **values)


class ObservationWindowContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = adapter_fixtures.LiveFixture()

    async def asyncTearDown(self):
        await self.fixture.close()

    def test_window_cannot_introduce_foreign_source_or_future_capture(self):
        update = history(self.fixture)
        foreign = update.observation_history[0].model_copy(update={"dependencies": (
            SourceDependency(source_namespace="context_event", source_id=uuid4(), revision=1),)})
        future = update.observation_history[0].model_copy(update={"observed_at": NOW + timedelta(seconds=1)})
        for window in (foreign, future):
            with self.subTest(window=window.observed_at):
                with self.assertRaises(ValidationError):
                    VoiceContextUpdate.model_validate(update.model_dump() | {"observation_history": [window]})

    def test_window_rejects_inverted_time_duplicate_versions_and_duplicate_windows(self):
        source, _, window = provenance(self.fixture)
        with self.assertRaises(ValidationError):
            ObservationWindow(observed_at=NOW, fresh_until=NOW, dependencies=(source,))
        with self.assertRaises(ValidationError):
            ObservationWindow(observed_at=NOW, fresh_until=NOW + timedelta(seconds=1),
                dependencies=(source, source.model_copy(update={"revision": 2})))
        update = history(self.fixture, source=source)
        with self.assertRaises(ValidationError):
            VoiceContextUpdate.model_validate(update.model_dump() | {"observation_history": [window, window]})

    async def test_prepare_default_legacy_expiry_and_live_hard_expiry_are_distinct(self):
        f = self.fixture
        source, binding, window = provenance(f)
        async def bind(selection):
            self.assertEqual(selection.dependencies, (source,))
            return binding
        common = dict(scope=f.identity.scope, cursor=f.cursor, text="Timestamped evidence.",
            dependencies=(source,), parent_payload_root_ids=binding.payload_root_ids,
            selected_entry_ids=("working_observation",), omitted_entry_ids=(),
            expires_at=window.fresh_until)
        legacy = await prepare_voice_context(VoiceSelection(**common), f.identity, bind=bind, clock=lambda: NOW)
        self.assertEqual(legacy.expires_at, window.fresh_until)
        self.assertEqual(legacy.observation_history, ())
        live = await prepare_voice_context(VoiceSelection(**common, observation_history=(window,)),
            f.identity, bind=bind, clock=lambda: NOW)
        self.assertIsNone(live.expires_at)
        self.assertEqual(live.observation_history, (window,))
        hard = NOW + timedelta(seconds=2)
        mixed = await prepare_voice_context(VoiceSelection(**(common | {"expires_at": hard}),
            observation_history=(window,), hard_expires_at=hard), f.identity, bind=bind, clock=lambda: NOW)
        self.assertEqual(mixed.expires_at, hard)

    async def test_new_preparation_does_not_publish_an_already_stale_selection(self):
        f = self.fixture
        source, binding, window = provenance(f)
        calls = []
        async def bind(selection):
            calls.append(selection)
            return binding
        selection = VoiceSelection(f.identity.scope, f.cursor, "Timestamped evidence.", (source,),
            binding.payload_root_ids, ("working_observation",), (), window.fresh_until, (window,))
        self.assertIsNone(await prepare_voice_context(selection, f.identity, bind=bind,
            clock=lambda: window.fresh_until))
        self.assertFalse(calls)


class LiveObservationHistoryAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = adapter_fixtures.LiveFixture()

    async def asyncTearDown(self):
        await self.fixture.close()

    async def start_history(self, **changes):
        initial = history(self.fixture, **changes)
        await self.fixture.start(self.fixture.start_request(initial_context=initial))
        return initial

    async def test_age_and_omission_keep_peer_but_retain_all_source_versions(self):
        f = self.fixture
        initial = await self.start_history()
        f.now = initial.observation_history[0].fresh_until + timedelta(seconds=1)
        await asyncio.sleep(.09)
        self.assertFalse(f.route.requires_fenced_replacement)
        unknown = f.context(1, text="Current visual state unknown; prior observations are historical.", expires_at=None)
        receipt = await f.route.publish_context(unknown)
        self.assertIn(receipt.status, {"written", "provider_acknowledged"})
        self.assertFalse(f.route.requires_fenced_replacement)
        self.assertEqual(f.route._context_sources, set())
        self.assertEqual(set(f.route._sources.values()), set(initial.dependencies))
        self.assertFalse(f.fences)
        sent = len(f.wire.socket.sent)
        await f.route.publish_context(unknown)
        self.assertEqual(len(f.wire.socket.sent), sent)

    async def test_new_observation_is_not_correction_of_an_older_observation(self):
        f = self.fixture
        initial = await self.start_history()
        newer = history(f, 1)
        result = await f.route.publish_context(newer)
        self.assertIn(result.status, {"written", "provider_acknowledged"})
        self.assertFalse(f.route.requires_fenced_replacement)
        self.assertEqual(set(f.route._sources.values()), set(initial.dependencies + newer.dependencies))

    async def test_same_source_correction_after_omission_still_fences(self):
        f = self.fixture
        initial = await self.start_history()
        await f.route.publish_context(f.context(1, text="Current view unknown.", expires_at=None))
        changed = initial.dependencies[0].model_copy(update={"revision": 2})
        receipt = await f.route.publish_context(history(f, 2, source=changed))
        self.assertEqual(receipt.status, "rejected")
        self.assertTrue(f.route.requires_fenced_replacement)
        await f.until(lambda: f.route._closed)

    async def test_omission_of_non_observational_source_still_fences(self):
        f = self.fixture
        source, _, window = provenance(f)
        task = SourceDependency(source_namespace="task_result", source_id=uuid4(), revision=1)
        binding = GuardedPayloadBinding(scope=f.identity.scope, payload_root_ids=(uuid4(),),
            dependencies=(source, task))
        initial = history(f, source=source, dependencies=(source, task), guard_binding=binding,
            observation_history=(window,))
        await f.start(f.start_request(initial_context=initial))
        receipt = await f.route.publish_context(f.context(1, expires_at=None))
        self.assertEqual(receipt.status, "rejected")
        self.assertTrue(f.route.requires_fenced_replacement)

    async def test_hard_expiry_still_fences_even_with_observation_history(self):
        f = self.fixture
        await self.start_history(expires_at=NOW + timedelta(seconds=1))
        f.now = NOW + timedelta(seconds=2)
        await f.until(lambda: f.route.requires_fenced_replacement)
        self.assertTrue(f.valid and f.authorized)

    async def test_cancelled_unknown_publication_is_not_replayed_or_reopened(self):
        f = self.fixture
        await self.start_history()
        update = history(f, 1)
        f.guard_block_id = update.update_id
        pending = asyncio.create_task(f.route.publish_context(update))
        try:
            await asyncio.wait_for(f.guard_entered.wait(), 1)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            self.assertTrue(f.route.requires_fenced_replacement)
            await f.until(lambda: f.route._closed)
            self.assertEqual(len(f.wire.requests), 1)
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    async def test_changed_scope_remains_rejected_without_provider_write(self):
        f = self.fixture
        await self.start_history()
        foreign = f.context(1, expires_at=None).model_copy(update={
            "route": f.identity.model_copy(update={"route_id": uuid4()})})
        before = len(f.wire.socket.sent)
        with self.assertRaisesRegex(adapter_fixtures.LiveRouteError, "^route_identity_mismatch$"):
            await f.route.publish_context(foreign)
        self.assertEqual(len(f.wire.socket.sent), before)


class LiveObservationHistoryCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = session_fixtures.LiveSessionTests(
            "test_actual_composition_gates_media_and_joins_single_readers")
        await self.fixture.asyncSetUp()

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def publish_observation(self):
        f = self.fixture
        session = await f.start()
        await f.until(lambda: session.live.service.snapshot().readiness == "ready")
        now = datetime.now(timezone.utc)
        source = f.db.event(f.context.owner_id, f.context.jarvis_session_id,
            at=now, kind="observation", trust="derived")
        record = f.db.record(source.request_event_id)
        summary = "A green cup was visible on the desk."
        binding = await session.live.binder.bind_record(record, summary)
        deadline = now + timedelta(seconds=.7)
        session.live.producer.publish(ObservationEntry(entry_id="working_observation", scope=session.live.scope,
            received_at=now, observed_at=now, fresh_until=deadline, expires_at=deadline,
            evidence_kind="observed", summary=summary, sources=(context_dependency(record),),
            guard_binding=binding))
        await f.until(lambda: session.adapter._context is not None
            and bool(session.adapter._context.observation_history))
        update = session.adapter._context
        self.assertIsNone(update.expires_at)
        self.assertEqual(update.observation_history[0].fresh_until, deadline)
        return session, source, update

    async def test_real_projection_expiry_sends_one_unknown_view_without_reconnect(self):
        session, source, update = await self.publish_observation()
        f = self.fixture
        await f.until(lambda: session.adapter._context.update_id != update.update_id
            and not session.adapter._context.observation_history)
        self.assertTrue(session.ready)
        self.assertIn("unknown", session.adapter._context.text.lower())
        self.assertGreater(session.adapter._context.cursor.revision, update.cursor.revision)
        self.assertTrue(session._source_context.has_sources)
        self.assertEqual(len(f.wire.requests), 1)
        self.assertFalse(any(m["type"] == "voice_route.close" for m in f.phone.messages))
        written = len(f.wire.socket.sent)
        await asyncio.sleep(.25)
        self.assertEqual(len(f.wire.socket.sent), written)

    async def test_expired_and_omitted_history_does_not_revive_deleted_canonical_source(self):
        session, source, update = await self.publish_observation()
        f = self.fixture
        await f.until(lambda: session.adapter._context.update_id != update.update_id
            and not session.adapter._context.observation_history)
        f.db.events[source.request_event_id].update(availability="deleted", content=None)
        await asyncio.wait_for(f.run_task, 3)
        self.assertTrue(session.adapter.requires_fenced_replacement)
        self.assertTrue(any(m["type"] == "voice_route.close" for m in f.phone.messages))
        self.assertEqual(len(f.wire.requests), 1)
        self.assertTrue(session.shutdown_confirmed)

    async def test_owner_loss_still_rejects_historical_context_guard(self):
        session, source, update = await self.publish_observation()
        session._connected = False
        with self.assertRaises(SourceError) as caught:
            async with session._source_guard(update):
                self.fail("Lost owner must not acquire a copy/write guard")
        self.assertEqual(caught.exception.code, "live_context_not_current")


if __name__ == "__main__":
    unittest.main()
