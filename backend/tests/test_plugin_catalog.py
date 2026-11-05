"""Pure projection only: no installer, database, grants or provider."""
from datetime import UTC, datetime, timedelta
import unittest
from uuid import UUID
from pydantic import ValidationError
from app.plugins.catalog import project_catalog
from app.plugins.models import InventoryObservation, LifecycleObservation, PolicySnapshot, ReadinessObservation

NOW = datetime(2026, 11, 1, 23, tzinfo=UTC)
CONFIG, OTHER = UUID(int=1), UUID(int=2)

def policy(enabled=True, revision=1, configuration=CONFIG, defaults=()):
    return PolicySnapshot(revision=revision, entries=[dict(plugin_id="video_editor", desired_enabled=enabled,
                                                         configuration_id=configuration)], defaults=defaults)

def observation(**values):
    return dict(plugin_id="video_editor", configuration_id=CONFIG, observed_at=NOW-timedelta(seconds=1),
                expires_at=NOW+timedelta(minutes=1)) | values

def installed(**values):
    return InventoryObservation(**(observation(state="installed", bundle_version="fixture-1", bundle_sha256="a"*64) | values))

def ready(**values):
    return ReadinessObservation(**(observation(state="ready", bundle_sha256="a"*64) | values))

def operation(**values):
    return LifecycleObservation(**(observation(operation_ref="operation:fixture-1", phase="installing") | values))

class PluginCatalogTests(unittest.TestCase):
    def project(self, **values):
        return project_catalog(**(dict(policy=policy(), inventory=[installed()], readiness=[ready()], now=NOW) | values))
    def card(self, **values):
        return self.project(**values).items[0]

    def test_current_exact_bundle_ready_but_no_mutation_adapter(self):
        result = self.project()
        self.assertEqual([x.plugin_id for x in result.items], ["video_editor", "browser", "files_data"])
        self.assertEqual((result.items[0].desired_enabled, result.items[0].installed_version,
                          result.items[0].readiness), (True, "fixture-1", "ready"))
        self.assertTrue(result.policy_available)
        self.assertFalse(any(x.can_change for x in result.items))

    def test_explicit_change_capability_requires_known_policy_choice(self):
        from pydantic import ValidationError
        from app.plugins.models import PluginCard
        result = self.project(changes_enabled=True)
        self.assertTrue(result.items[0].can_change)
        self.assertTrue(all(not card.can_change for card in result.items[1:]))
        self.assertFalse(any(card.can_change for card in self.project(policy=None, changes_enabled=True).items))
        self.assertTrue(self.card(policy=policy(False), changes_enabled=True).can_change,
                        "A confirmed disabled choice may be explicitly re-enabled")
        with self.assertRaises(TypeError):
            self.project(changes_enabled="true")
        for fields in (dict(can_change="true"), dict(can_change=True, desired_enabled=None),
                       dict(can_change=True, configuration_id=None)):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                PluginCard.model_validate(result.items[0].model_dump() | fields)

    def test_change_capability_does_not_invent_installation_or_readiness(self):
        result = self.project(changes_enabled=True, inventory=[], readiness=[])
        card = result.items[0]
        self.assertTrue(card.can_change)
        self.assertEqual((card.installed, card.readiness), ("unknown", "unknown"))
        self.assertIsNone(card.operation)
        self.assertIsNone(card.observations_valid_until)

    def test_missing_policy_never_becomes_enabled_default(self):
        result = self.project(policy=None)
        self.assertFalse(result.policy_available)
        self.assertIsNone(result.policy_revision)
        for card in result.items:
            self.assertIsNone(card.desired_enabled)
            self.assertIsNone(card.configuration_id)
            self.assertIsNone(card.installed_version)
            self.assertEqual((card.installed, card.readiness), ("unknown", "unknown"))
            self.assertFalse(card.selected_default)
            self.assertFalse(card.can_change)

    def test_explicit_zero_baseline_and_missing_choice_are_distinct(self):
        result = self.project(policy=policy(revision=0))
        self.assertEqual(result.policy_revision, 0)
        self.assertEqual(result.items[0].readiness, "ready")
        self.assertIsNone(result.items[1].desired_enabled)
        self.assertEqual(result.items[1].reason_code, "plugin_choice_unavailable")

    def test_disable_blocks_new_use_while_removing_old_installed_bundle(self):
        card = self.card(policy=policy(False), operations=[operation(phase="removing")])
        self.assertEqual((card.desired_enabled, card.installed, card.readiness), (False, "installed", "unavailable"))
        self.assertEqual(card.reason_code, "plugin_disabled")
        self.assertEqual(card.operation.phase, "removing")
        self.assertEqual(card.operation.operation_ref, "operation:fixture-1")

    def test_pending_install_is_not_ready_even_if_bundle_is_seen(self):
        card = self.card(operations=[operation()])
        self.assertEqual(card.readiness, "unavailable")
        self.assertEqual(card.reason_code, "plugin_change_pending")
        self.assertIsNone(card.operation.recovery)

    def test_exact_configuration_and_bundle_are_both_required(self):
        for changes in (dict(inventory=[installed(configuration_id=OTHER)]),
                        dict(readiness=[ready(configuration_id=OTHER)]),
                        dict(readiness=[ready(bundle_sha256="b"*64)])):
            with self.subTest(changes=changes):
                self.assertEqual(self.card(**changes).readiness, "unknown")
        self.assertEqual(self.card(readiness=[ready(bundle_sha256="b"*64)]).reason_code, "plugin_bundle_mismatch")

    def test_expired_and_future_observations_do_not_qualify(self):
        for window in (dict(observed_at=NOW-timedelta(minutes=2), expires_at=NOW),
                       dict(observed_at=NOW+timedelta(seconds=1), expires_at=NOW+timedelta(minutes=1))):
            with self.subTest(window=window):
                card = self.card(inventory=[installed(**window)])
                self.assertEqual((card.installed, card.readiness), ("unknown", "unknown"))
                self.assertIsNone(card.installed_version)
                self.assertEqual(self.card(readiness=[ready(**window)]).readiness, "unknown")

    def test_unrelated_policy_revision_preserves_config_but_new_intent_does_not(self):
        self.assertEqual(self.card(), self.card(policy=policy(revision=9)))
        self.assertEqual(self.card(policy=policy(configuration=OTHER)).readiness, "unknown")

    def test_old_lifecycle_phase_cannot_replace_new_intent(self):
        for phase in ("installing", "removing", "failed", "unknown"):
            with self.subTest(phase=phase):
                card = self.card(operations=[operation(configuration_id=OTHER, phase=phase)])
                self.assertIsNone(card.operation)
                self.assertEqual(card.readiness, "ready")
        card = self.card(policy=policy(configuration=OTHER), operations=[operation()])
        self.assertIsNone(card.operation)
        self.assertEqual(card.readiness, "unknown")

    def test_unknown_recheck_is_optional_and_must_be_supplied(self):
        self.assertIsNone(self.card(operations=[operation(phase="unknown")]).operation.recovery)
        recovery = dict(kind="recheck", reference="status:original")
        card = self.card(operations=[operation(phase="unknown", recovery=recovery)])
        self.assertEqual(card.readiness, "unknown")
        self.assertEqual(card.operation.recovery.model_dump(), recovery)
        self.assertEqual(self.card(policy=policy(False), operations=[operation(phase="unknown")]).readiness, "unavailable")

    def test_expired_operation_is_unknown_and_never_reuses_retry(self):
        card = self.card(operations=[operation(phase="failed", recovery=dict(kind="retry", reference="retry:original"),
                    observed_at=NOW-timedelta(minutes=2), expires_at=NOW)])
        self.assertEqual((card.operation.phase, card.operation.reason_code), ("unknown", "operation_status_stale"))
        self.assertIsNone(card.operation.recovery)
        self.assertEqual(card.readiness, "unknown")

    def test_recovery_is_phase_bound_and_not_inferred(self):
        self.assertIsNone(self.card(operations=[operation(phase="failed")]).operation.recovery)
        for kind in ("retry", "recheck"):
            recovery = dict(kind=kind, reference=kind+":original")
            self.assertEqual(self.card(operations=[operation(phase="failed", recovery=recovery)]).operation.recovery.kind, kind)
        for phase, kind in (("unknown", "retry"), ("installing", "recheck"), ("removing", "retry")):
            with self.subTest(phase=phase, kind=kind):
                with self.assertRaises(ValidationError):
                    operation(phase=phase, recovery=dict(kind=kind, reference="original"))

    def test_independent_plugin_does_not_wait_for_another_install(self):
        value = policy().model_dump()
        value["entries"] += (dict(plugin_id="browser", desired_enabled=True, configuration_id=OTHER),)
        result = self.project(policy=value, operations=[operation()],
            inventory=[installed(), installed(plugin_id="browser", configuration_id=OTHER)],
            readiness=[ready(), ready(plugin_id="browser", configuration_id=OTHER)])
        self.assertEqual([x.readiness for x in result.items], ["unavailable", "ready", "unknown"])

    def test_defaults_are_confirmed_compatible_and_remembered_while_disabled(self):
        selection = [dict(category="video_editing", plugin_id="video_editor")]
        self.assertTrue(self.card(policy=policy(defaults=selection)).selected_default)
        self.assertFalse(self.card().selected_default)
        disabled = self.card(policy=policy(False, defaults=selection))
        self.assertTrue(disabled.selected_default)
        self.assertFalse(disabled.desired_enabled)
        self.assertEqual((disabled.readiness, disabled.reason_code), ("unavailable", "plugin_disabled"))
        self.assertTrue(self.card(policy=policy(defaults=selection)).selected_default)
        for candidate in (policy(defaults=[dict(category="web_browsing", plugin_id="video_editor")]),
                          policy(defaults=[dict(category="web_browsing", plugin_id="browser")])):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    self.project(policy=candidate)

    def test_observation_expiry_is_bounded_without_expiring_preference(self):
        expires = NOW + timedelta(seconds=10)
        card = self.card(readiness=[ready(expires_at=expires)])
        self.assertEqual(card.observations_valid_until, expires)
        selection = [dict(category="video_editing", plugin_id="video_editor")]
        expired = self.card(policy=policy(defaults=selection), now=NOW+timedelta(minutes=2))
        self.assertIsNone(expired.observations_valid_until)
        self.assertEqual((expired.installed, expired.readiness), ("unknown", "unknown"))
        self.assertTrue(expired.selected_default)
        self.assertTrue(expired.desired_enabled)
        self.assertIsNone(self.card(policy=None).observations_valid_until)

    def test_absent_inventory_and_runtime_failure_are_not_ready(self):
        absent = installed(state="absent", bundle_version=None, bundle_sha256=None)
        self.assertEqual(self.card(inventory=[absent]).reason_code, "plugin_not_installed")
        self.assertEqual(self.card(readiness=[]).readiness, "unknown")
        card = self.card(readiness=[ready(state="unavailable", reason_code="worker_unavailable")])
        self.assertEqual((card.readiness, card.reason_code), ("unavailable", "worker_unavailable"))

    def test_invalid_ambiguous_or_secret_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.project(inventory=[installed(), installed()])
        with self.assertRaises(ValidationError):
            installed(state="unknown")
        with self.assertRaises(ValidationError):
            installed(bundle_sha256=None)
        with self.assertRaises(ValidationError):
            PolicySnapshot.model_validate(policy().model_dump() | dict(bearer_token="secret"))
        with self.assertRaises(ValidationError):
            operation(phase="unknown", recovery=dict(kind="recheck", reference="https://example.invalid"))
        with self.assertRaises(ValidationError):
            ready(observed_at=1234)
        with self.assertRaises(ValidationError):
            policy(revision=True)
        with self.assertRaises(ValidationError):
            policy(enabled="true")
        with self.assertRaises(ValueError):
            self.project(now=NOW.replace(tzinfo=None))

    def test_inputs_immutable_and_public_view_contains_no_authority_or_paths(self):
        original = policy()
        before = original.model_dump(mode="json")
        result = self.project(policy=original)
        self.assertEqual(original.model_dump(mode="json"), before)
        wire = result.model_dump_json()
        for field in ('"owner_id"', '"grant"', '"source_event"', '"command"', '"url"', '"path"'):
            self.assertNotIn(field, wire)
        self.assertIn('"can_change":false', wire)
        with self.assertRaises(ValidationError):
            result.items[0].can_change = True

if __name__ == "__main__":
    unittest.main()
