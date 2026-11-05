"""Opt-in ledger proof, not production ingress or plugin-use enforcement.

The fixture explicitly supplies a synthetic revision-zero baseline and uses the
existing owner transaction/context repository. Core still owns real admission,
submission fences and installer/runtime composition.
"""
import asyncio
import os
from pathlib import Path
import unittest
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.plugins.models import PolicySnapshot
from app.plugins.policy import CAPABILITY, UI_NAMESPACE, PluginPolicyChange, changed_policy
from app.plugins.repository import PluginPolicyRepository
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Core's explicit manifested database lane is required")
class CuratedPluginPolicySQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_owner_ledger_replay_cas_rollback_and_withdrawal(self):
        fixture = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        tasks = TaskRepository(fixture.pool, TaskLimits(lease_seconds=300))
        contexts = ContextRepository(tasks)
        repository = PluginPolicyRepository(fixture.pool)
        baseline = PolicySnapshot(revision=0, entries=())

        async def read(owner=None, *, trusted_baseline=None):
            async with fixture.pool.acquire() as connection:
                return await repository.read(connection, owner or fixture.owner, baseline=trusted_baseline)

        async def counts():
            async with fixture.pool.acquire() as connection:
                return tuple([await connection.fetchval(
                    "select count(*) from " + table + " where owner_id=$1", fixture.owner)
                    for table in ("jarvis_context_events", "jarvis_context_payloads",
                                  "jarvis_context_ingress_receipts", "jarvis_capability_preferences")])

        async def apply(command, *, fail_after_append=False):
            # Test-only composition of existing internal seams. No route or
            # installer is invoked or substituted by this fixture.
            async with tasks.transaction(fixture.owner) as connection:
                event = await contexts.append_in_transaction(connection, fixture.owner, ContextIngress(
                    session_id=None, source_namespace=UI_NAMESPACE,
                    source_event_id=str(command.mutation_id), receipt_id=str(command.mutation_id),
                    kind="user_request", trust="user_input", finality="final",
                    expected_revision=0, payload=command.ingress_payload()))
                applied = await repository.applied(connection, fixture.owner, command, event)
                current = await repository.read(connection, fixture.owner, baseline=baseline, lock=True)
                if applied is not None:
                    return applied, current, event
                self.assertIsNotNone(current.policy)
                policy = changed_policy(current.policy, command, event.id)
                saved = await repository.append(connection, fixture.owner, command, event, current, policy)
                if fail_after_append:
                    raise RuntimeError("synthetic same-transaction failure")
                return saved.revision, saved, event

        self.assertIsNone((await read()).policy, "An absent ledger must remain unknown")
        first = PluginPolicyChange(mutation_id=uuid4(), expected_revision=0,
            enabled_changes=[dict(plugin_id="video_editor", enabled=True)],
            default_changes=[dict(category="video_editing", plugin_id="video_editor")])
        original_revision, original, original_event = await apply(first)
        self.assertEqual(original_revision, 1)
        self.assertEqual((await read()).policy, original.policy)
        self.assertIsNone((await read(fixture.other)).policy)
        async with fixture.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_sessions where owner_id=$1", fixture.owner), 0)

        disable = PluginPolicyChange(mutation_id=uuid4(), expected_revision=1,
            enabled_changes=[dict(plugin_id="video_editor", enabled=False)])
        _, disabled, _ = await apply(disable)
        self.assertFalse(disabled.policy.entries[0].desired_enabled)
        self.assertEqual(disabled.policy.defaults, original.policy.defaults)
        self.assertNotEqual(disabled.policy.entries[0].configuration_id, original.policy.entries[0].configuration_id)

        before = await counts()
        replay_revision, latest, replay_event = await apply(first)
        self.assertEqual((replay_revision, latest.revision, replay_event.id), (1, 2, original_event.id))
        self.assertEqual(await counts(), before, "Replay must not write another event, payload or policy")
        with self.assertRaises(TaskError) as conflict:
            await apply(first.model_copy(update={"expected_revision": 2}))
        self.assertEqual(conflict.exception.code, "context_receipt_conflict")
        self.assertEqual(await counts(), before)

        # Two explicit UI requests observe revision two; the existing owner lock
        # admits only one new policy head, with no orphan loser ingress.
        candidates = [PluginPolicyChange(mutation_id=uuid4(), expected_revision=2,
            enabled_changes=[dict(plugin_id="browser", enabled=enabled)]) for enabled in (True, False)]
        results = await asyncio.gather(*(apply(c) for c in candidates), return_exceptions=True)
        failures = [r for r in results if isinstance(r, JarvisHTTPError)]
        self.assertEqual(len(failures), 1, results)
        self.assertEqual(failures[0].code, "plugin_policy_revision_conflict")
        current = await read()
        self.assertEqual(current.revision, 3)
        video = next(x for x in current.policy.entries if x.plugin_id == "video_editor")
        self.assertEqual(video, disabled.policy.entries[0], "An unrelated plugin keeps its configuration identity")
        self.assertEqual(current.policy.defaults, disabled.policy.defaults)
        self.assertEqual((await counts())[0], 3, "Rejected CAS rolls back its uncommitted UI event")

        rollback = PluginPolicyChange(mutation_id=uuid4(), expected_revision=3,
            default_changes=[dict(category="video_editing", plugin_id=None)])
        before = await counts()
        with self.assertRaisesRegex(RuntimeError, "same-transaction"):
            await apply(rollback, fail_after_append=True)
        self.assertEqual(await counts(), before)
        self.assertEqual((await read()).policy, current.policy)
        _, cleared, _ = await apply(rollback)
        self.assertEqual(cleared.revision, 4)
        self.assertEqual(cleared.policy.defaults, ())
        self.assertEqual(cleared.policy.entries, current.policy.entries)

        # Canonical source withdrawal propagates through the policy dependency
        # chain. An explicitly offered fixture baseline cannot resurrect it.
        async with tasks.transaction(fixture.owner) as connection:
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", fixture.owner, original_event.payload_id)
        withdrawn = await read(trusted_baseline=baseline)
        self.assertEqual((withdrawn.revision, withdrawn.policy), (4, None))
        self.assertIsNone((await read(fixture.other)).policy)
        async with fixture.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_capability_preferences
                where owner_id=$1 and capability=$2""", fixture.owner, CAPABILITY), 4)


if __name__ == "__main__":
    unittest.main()
