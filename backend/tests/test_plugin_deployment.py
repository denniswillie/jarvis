from dataclasses import FrozenInstanceError, replace
import unittest
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.plugins.authority import ManagedPluginBinding, PluginAuthority
from app.plugins.deployment import CURATED_DEPLOYMENT_BASELINE_V1, PluginDeploymentBaseline
from app.plugins.policy import EnabledChange, PluginPolicyChange, PluginPolicyState, changed_policy


class Connection:
    def is_in_transaction(self):
        return True


class Repository:
    def __init__(self):
        self.state = None

    async def read(self, connection, owner_id, *, baseline=None, lock=False):
        return self.state if self.state is not None else PluginPolicyState(0, baseline)


class PluginDeploymentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.baseline = CURATED_DEPLOYMENT_BASELINE_V1
        self.repository = Repository()
        self.connection = Connection()
        self.authority = PluginAuthority(self.repository, None, None, deployment_baseline=self.baseline)

    async def select(self, authority=None):
        return await (authority or self.authority).select_new_work(owner_id="owner", category="video_editing",
                                                                 connection=self.connection)

    def test_baseline_is_explicit_immutable_and_repeatable(self):
        rebuilt = PluginDeploymentBaseline(self.baseline.version, self.baseline.policy)
        self.assertEqual(rebuilt.identity, self.baseline.identity)
        self.assertEqual(len({item.configuration_id for item in self.baseline.policy.entries}), 3)
        self.assertEqual({item.plugin_id for item in self.baseline.policy.entries}, {"video_editor", "browser", "files_data"})
        with self.assertRaises(FrozenInstanceError):
            self.baseline.version = "changed"
        with self.assertRaises(TypeError):
            PluginAuthority(self.repository, None, None, deployment_baseline=self.baseline.policy)
        with self.assertRaises(ValueError):
            PluginDeploymentBaseline("other", self.baseline.policy.model_copy(update={"revision": 1}))

    async def test_absent_account_selects_only_supplied_baseline(self):
        selected = await self.select()
        self.assertEqual(selected.policy_revision, 0)
        self.assertEqual(selected.deployment_baseline, self.baseline.identity)
        self.assertEqual(await self.authority.require_new_work(self.connection, owner_id="owner", binding=selected), selected)
        with self.assertRaises(JarvisHTTPError):
            await self.select(PluginAuthority(self.repository, None, None))

    async def test_bare_zero_or_unavailable_account_never_falls_back(self):
        self.repository.state = PluginPolicyState(0, self.baseline.policy)
        with self.assertRaises(JarvisHTTPError):
            await self.select(PluginAuthority(self.repository, None, None))
        for state in [PluginPolicyState(1, None, uuid4()), PluginPolicyState(0, self.baseline.policy, uuid4()),
                      PluginPolicyState(0, self.baseline.policy.model_copy(update={"entries": ()}))]:
            self.repository.state = state
            with self.assertRaises(JarvisHTTPError):
                await self.select()

    async def test_later_disable_blocks_baseline_selected_job(self):
        selected = await self.select()
        command = PluginPolicyChange(mutation_id=uuid4(), expected_revision=0,
            enabled_changes=(EnabledChange(plugin_id="video_editor", enabled=False),))
        policy = changed_policy(self.baseline.policy, command, uuid4())
        self.repository.state = PluginPolicyState(1, policy, uuid4())
        with self.assertRaises(JarvisHTTPError):
            await self.authority.require_new_work(self.connection, owner_id="owner", binding=selected)

    async def test_unrelated_first_change_keeps_unchanged_selection(self):
        selected = await self.select()
        command = PluginPolicyChange(mutation_id=uuid4(), expected_revision=0,
            enabled_changes=(EnabledChange(plugin_id="browser", enabled=False),))
        policy = changed_policy(self.baseline.policy, command, uuid4())
        self.repository.state = PluginPolicyState(1, policy, uuid4())
        await self.authority.require_new_work(self.connection, owner_id="owner", binding=selected)
        later = await self.select()
        self.assertEqual(later.policy_revision, 1)
        self.assertIsNone(later.deployment_baseline)

    async def test_baseline_binding_cannot_be_redefined_or_forged(self):
        selected = await self.select()
        with self.assertRaises(ValueError):
            ManagedPluginBinding("owner", "video_editor", selected.configuration_id, 0)
        with self.assertRaises(JarvisHTTPError):
            await self.authority.require_new_work(self.connection, owner_id="owner",
                binding=replace(selected, deployment_baseline="untrusted"))
        other = PluginAuthority(self.repository, None, None,
            deployment_baseline=PluginDeploymentBaseline("different-deployment", self.baseline.policy))
        with self.assertRaises(JarvisHTTPError):
            await other.require_new_work(self.connection, owner_id="owner", binding=selected)


if __name__ == "__main__":
    unittest.main()
