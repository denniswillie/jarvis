"""Actual shared derivation overlap, canonical writer fences and correction."""
from datetime import UTC, datetime, timedelta
import os
import unittest
from uuid import uuid4

from app.knowledge.models import CanonicalOwnerBusy, SourceError
from app.live_state.canonical import CanonicalContextBinder, memory_dependency
from app.live_state.models import LiveSnapshot, SessionScope, StateCursor
from app.live_state.perception_context import compose_perception_brief
from app.live_state.perception_runtime import CanonicalPerceptionBinder
from app.live_state.projection import VoiceSelection
from app.live_state.task_context import TaskContextProjector
from app.tasks.models import TaskCreate, TaskError, TaskVoiceSource
import test_second_brain_publication_integration as fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
                     and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested SQL fixture required")
class SharedDerivationSQLTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.SourcePublicationDatabaseTests.asyncSetUp
    ingress = fixture.SourcePublicationDatabaseTests.ingress
    mapped = fixture.SourcePublicationDatabaseTests.mapped

    async def canonical_counts(self):
        async with self.f.pool.acquire() as connection:
            return dict(await connection.fetchrow("""select
                (select count(*) from jarvis_context_events) as events,
                (select max(sequence) from jarvis_context_events) as event_sequence,
                (select count(*) from jarvis_knowledge_source_versions) as versions,
                (select count(*) from jarvis_task_events) as task_events,
                (select count(*) from jarvis_memory_outbox) as outbox"""))

    async def test_copy_creation_overlaps_readers_without_changing_canonical_state(self):
        owner = self.f.owner
        original = await self.context.append(owner, self.ingress())
        memory = await self.mapped(original)
        scope = SessionScope(owner_id=owner, session_id=self.session, area_id=self.f.area)
        binder = CanonicalContextBinder(self.f.sources, scope, clock=lambda: datetime.now(UTC))
        task = await self.tasks.create(owner, TaskCreate(request_id=uuid4(), session_id=self.session,
            knowledge_area_id=self.f.area, title="Synthetic task", requested_outcome="Fixture context"),
            model="fixture", source=TaskVoiceSource(original.id, original.revision, self.session), mode="background")
        before = await self.canonical_counts()
        async with self.f.pool.acquire(), self.f.pool.acquire():
            pass
        bindings = []
        # Hold one actual shared derivation transaction while independent API
        # calls use other leases. This is a lock test, not a native latency test.
        async with self.f.sources.derivation_transaction(owner) as first:
            await self.tasks.lock_owner_shared(first, owner)
            bindings.append(await binder.bind_record(original, "Exact synthetic request"))
            bindings.append(await binder.bind_memory(memory, "Exact synthetic memory"))
            selection = VoiceSelection(scope, StateCursor(epoch=uuid4(), revision=1),
                "Combined synthetic context", bindings[1].dependencies, bindings[1].payload_root_ids,
                ("memory",), (), None)
            bindings.append(await binder(selection))
            now = datetime.now(UTC)
            snapshot = LiveSnapshot(scope=scope, cursor=selection.cursor, as_of=now, readiness="ready")
            brief = compose_perception_brief(snapshot, now=now, scope_epoch=selection.cursor.epoch,
                context_dependencies=(memory_dependency(memory),), dependency_eligible=lambda source: True).brief
            bindings.append(await CanonicalPerceptionBinder(binder).bind(brief))
            entries = await TaskContextProjector(binder, self.tasks).project((task.id,),
                received_at=datetime.now(UTC), expires_at=datetime.now(UTC) + timedelta(seconds=20))
            self.assertEqual(len(entries), 1)
            bindings.append(entries[0].guard_binding)
            async with binder.guard(bindings[0], wait_for_owner=False):
                pass
            with self.assertRaises(CanonicalOwnerBusy):
                async with self.f.sources.transaction(owner, wait_for_owner=False):
                    self.fail("Canonical source mutation entered during derivation")
            async with self.tasks.transaction() as second:
                self.assertFalse(await self.tasks.try_lock_owner(second, owner))
        self.assertEqual(await self.canonical_counts(), before)
        corrected = await self.context.append(owner, self.ingress(
            text="Corrected synthetic request", finality="revised", expected_revision=original.revision))
        self.assertGreater(corrected.revision, original.revision)
        for binding in bindings:
            with self.subTest(root=str(binding.payload_root_ids[0])):
                with self.assertRaises((SourceError, TaskError)):
                    async with binder.guard(binding):
                        self.fail("A stale derived copy survived correction")
