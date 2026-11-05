"""Policy intent/repository boundaries; SQL and enforcement need integration proof."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import UUID

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.errors import JarvisHTTPError
from app.plugins.authority import PluginMutationReceipt
from app.plugins.deployment import CURATED_DEPLOYMENT_BASELINE_V1
from app.plugins.models import PolicySnapshot
from app.plugins.policy import PluginPolicyChange, PluginPolicyState, StoredPluginPolicy, UI_NAMESPACE, changed_policy
from app.plugins.repository import PluginPolicyRepository
from app.plugins.router import router
from app.plugins.service import PluginPolicyService
from app.settings import get_settings
from app.tasks.context_events import ContextRecord


NOW = datetime(2026, 11, 2, 7, tzinfo=UTC)
MUTATION, EVENT, CONFIG, PAYLOAD = (UUID(int=i) for i in range(1, 5))


def policy(revision=7, enabled=True):
    return PolicySnapshot(revision=revision, entries=[
        dict(plugin_id="video_editor", desired_enabled=enabled, configuration_id=CONFIG),
        dict(plugin_id="browser", desired_enabled=True, configuration_id=UUID(int=8))],
        defaults=[dict(category="video_editing", plugin_id="video_editor")])


def command(**changes):
    return PluginPolicyChange(**(dict(mutation_id=MUTATION, expected_revision=7,
        enabled_changes=[dict(plugin_id="video_editor", enabled=False)]) | changes))


def source(change):
    return ContextRecord(id=EVENT, owner_id="owner", session_id=None, sequence=1, origin_sequence=1,
        source_namespace=UI_NAMESPACE, source_event_id=str(change.mutation_id), revision=1,
        supersedes_event_id=None, kind="user_request", trust="user_input", finality="final",
        source_time=None, recorded_at=NOW, payload_id=PAYLOAD, availability="available",
        content_digest=None, content=change.ingress_payload(), is_current=True)


class PluginPolicyTests(unittest.TestCase):
    def test_disable_remembers_default_and_preserves_other_configuration(self):
        before = policy()
        after = changed_policy(before, command(), EVENT)
        entries = {entry.plugin_id: entry for entry in after.entries}
        self.assertFalse(entries["video_editor"].desired_enabled)
        self.assertNotEqual(entries["video_editor"].configuration_id, CONFIG)
        self.assertEqual(entries["browser"], before.entries[1])
        self.assertEqual(after.defaults, before.defaults)
        self.assertEqual(after, changed_policy(before, command(), EVENT))

    def test_default_only_and_idempotent_enabled_choice_preserve_configuration(self):
        default = command(enabled_changes=[], default_changes=[dict(category="video_editing", plugin_id=None)])
        result = changed_policy(policy(), default, EVENT)
        self.assertEqual({x.plugin_id: x for x in result.entries}, {x.plugin_id: x for x in policy().entries})
        self.assertEqual(result.defaults, ())
        result = changed_policy(policy(enabled=False), command(), EVENT)
        self.assertEqual(next(x.configuration_id for x in result.entries if x.plugin_id == "video_editor"), CONFIG)

    def test_reenable_uses_new_admitted_identity_and_retains_default(self):
        disabled = changed_policy(policy(), command(), EVENT)
        enabled = changed_policy(disabled, command(expected_revision=8,
            enabled_changes=[dict(plugin_id="video_editor", enabled=True)]), UUID(int=99))
        self.assertEqual(enabled.defaults, disabled.defaults)
        self.assertNotEqual(enabled.entries, disabled.entries)

    def test_stale_revision_cannot_apply(self):
        with self.assertRaises(JarvisHTTPError) as caught:
            changed_policy(policy(revision=8), command(), EVENT)
        self.assertEqual(caught.exception.code, "plugin_policy_revision_conflict")

    def test_bounded_strict_intent_rejects_unknown_authority_fields_and_categories(self):
        for values in [dict(owner_id="other"), dict(session_id=str(EVENT)), dict(grant_id=str(EVENT)),
                       dict(expected_revision=True), dict(enabled_changes=[]),
                       dict(enabled_changes=[dict(plugin_id="video_editor", enabled="false")]),
                       dict(enabled_changes=[dict(plugin_id="browser", enabled=True)] * 2),
                       dict(default_changes=[dict(category="video_editing", plugin_id="browser")])]:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                command(**values)

    def test_order_is_not_a_different_intent_and_unknown_default_is_rejected(self):
        changes = [dict(plugin_id="browser", enabled=False), dict(plugin_id="video_editor", enabled=False)]
        self.assertEqual(command(enabled_changes=changes).canonical_intent(),
                         command(enabled_changes=list(reversed(changes))).canonical_intent())
        with self.assertRaises(JarvisHTTPError):
            changed_policy(policy(), command(enabled_changes=[], default_changes=[
                dict(category="files_data", plugin_id="files_data")]), EVENT)


class PluginRepositoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.connection = SimpleNamespace(fetchrow=AsyncMock(return_value=None),
            fetchval=AsyncMock(return_value="available"), executemany=AsyncMock(), execute=AsyncMock(),
            is_in_transaction=lambda: True)
        self.repository = PluginPolicyRepository(None)

    async def test_absent_policy_unknown_unless_core_explicitly_supplies_baseline(self):
        result = await self.repository.read(self.connection, "owner")
        self.assertIsNone(result.policy)
        result = await self.repository.read(self.connection, "owner", baseline=policy(revision=0))
        self.assertEqual(result.policy.revision, 0)
        with self.assertRaises(ValueError):
            await self.repository.read(self.connection, "owner", baseline=policy())

    async def test_withdrawn_or_corrupt_head_never_uses_baseline(self):
        for availability, content in [("erased", None), ("available", {}),
                                       ("available", {"schema_version": 2})]:
            self.connection.fetchrow.return_value = dict(revision=9, policy_payload_id=PAYLOAD,
                availability=availability, content=content)
            result = await self.repository.read(self.connection, "owner", baseline=policy(revision=0))
            self.assertEqual((result.revision, result.policy), (9, None))

    async def test_receipt_returns_original_revision_and_changed_intent_conflicts(self):
        applied = changed_policy(policy(), command(), EVENT)
        self.connection.fetchrow.return_value = dict(revision=8, availability="available",
            content=StoredPluginPolicy(intent=command(), policy=applied).model_dump(mode="json"))
        self.assertEqual(await self.repository.applied(self.connection, "owner", command(), source(command())), 8)
        altered = command(expected_revision=9)
        with self.assertRaises(JarvisHTTPError) as caught:
            await self.repository.applied(self.connection, "owner", altered, source(altered))
        self.assertEqual(caught.exception.code, "plugin_policy_mutation_conflict")
        self.connection.execute.assert_not_awaited()

    async def test_source_must_be_exact_final_account_ui_request_in_transaction(self):
        for changes in [dict(owner_id="other"), dict(session_id=EVENT), dict(is_current=False),
                        dict(trust="derived"), dict(source_namespace="voice_command"), dict(content={})]:
            with self.subTest(changes=changes), self.assertRaises(JarvisHTTPError):
                await self.repository.applied(self.connection, "owner", command(), source(command()).model_copy(update=changes))
        self.connection.fetchrow.assert_not_awaited()
        self.connection.is_in_transaction = lambda: False
        with self.assertRaises(RuntimeError):
            await self.repository.applied(self.connection, "owner", command(), source(command()))

    async def test_withdrawn_dependency_rejects_append_before_any_write(self):
        self.connection.fetchval.return_value = "erased"
        with self.assertRaises(JarvisHTTPError) as caught:
            await self.repository.append(self.connection, "owner", command(), source(command()),
                PluginPolicyState(7, policy(), UUID(int=90)), changed_policy(policy(), command(), EVENT))
        self.assertEqual(caught.exception.code, "plugin_policy_source_unavailable")
        self.connection.execute.assert_not_awaited()
        self.connection.executemany.assert_not_awaited()


class PluginRoutesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        connection = SimpleNamespace(fetchrow=AsyncMock(return_value=None))
        @asynccontextmanager
        async def acquire():
            yield connection
        self.connection = connection
        service = PluginPolicyService(PluginPolicyRepository(SimpleNamespace(acquire=acquire)), clock=lambda: NOW)
        self.service = service
        app = FastAPI()
        app.include_router(router)
        app.state.runtime_services = SimpleNamespace(plugin_policy_service=service)
        app.dependency_overrides[get_settings] = lambda: SimpleNamespace(
            JARVIS_STATIC_BEARER_TOKEN="synthetic-plugin-test", JARVIS_OWNER_ID="owner")
        @app.exception_handler(JarvisHTTPError)
        async def errors(request, error):
            return JSONResponse({"error": {"code": error.code, "message": error.message}}, status_code=error.status_code)
        self.client = await self.enterAsyncContext(httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
            base_url="http://fixture.invalid"))

    async def test_auth_required_and_owner_is_from_principal(self):
        self.assertEqual((await self.client.get("/plugins")).status_code, 401)
        self.connection.fetchrow.assert_not_awaited()
        response = await self.client.get("/plugins", headers={"Authorization": "Bearer synthetic-plugin-test"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("no-store", response.headers["cache-control"])
        result = response.json()
        self.assertFalse(result["policy_available"])
        self.assertTrue(all(not x["can_change"] and x["desired_enabled"] is None for x in result["items"]))
        self.assertEqual(self.connection.fetchrow.call_args.args[1], "owner")

    async def test_unintegrated_write_rejected_without_ledger_io(self):
        response = await self.client.put("/plugins/policy", json=command().model_dump(mode="json"),
            headers={"Authorization": "Bearer synthetic-plugin-test"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "plugin_policy_changes_unavailable")
        self.connection.fetchrow.assert_not_awaited()
        response = await self.client.put("/plugins/policy", json=command().model_dump(mode="json") | {"owner_id": "other"},
            headers={"Authorization": "Bearer synthetic-plugin-test"})
        self.assertEqual(response.status_code, 422)

    async def test_authority_presence_does_not_activate_changes(self):
        authority = SimpleNamespace(mutate=AsyncMock())
        self.service.authority = authority
        response = await self.client.put("/plugins/policy", json=command().model_dump(mode="json"),
            headers={"Authorization": "Bearer synthetic-plugin-test"})
        self.assertEqual(response.status_code, 503)
        authority.mutate.assert_not_awaited()
        self.connection.fetchrow.assert_not_awaited()

    async def test_activation_requires_authority_and_explicit_boolean(self):
        with self.assertRaises(ValueError):
            PluginPolicyService(self.service.repository, changes_enabled=True)
        with self.assertRaises(TypeError):
            PluginPolicyService(self.service.repository, changes_enabled="true")

    async def test_enabled_route_uses_authenticated_owner_and_committed_receipt(self):
        saved = changed_policy(policy(), command(), EVENT)
        authority = SimpleNamespace(mutate=AsyncMock(return_value=PluginMutationReceipt(
            8, False, PluginPolicyState(8, saved))))
        self.service.authority = authority
        self.service.changes_enabled = True  # Synthetic runtime, not production activation.
        response = await self.client.put("/plugins/policy", json=command().model_dump(mode="json"),
            headers={"Authorization": "Bearer synthetic-plugin-test"})
        self.assertEqual(response.status_code, 200)
        authority.mutate.assert_awaited_once_with(owner_id="owner", command=command())
        result = response.json()
        self.assertEqual((result["mutation_id"], result["applied_revision"], result["request_replayed"]),
                         (str(MUTATION), 8, False))
        self.assertIn("no-store", response.headers["cache-control"])
        cards = {x["plugin_id"]: x for x in result["catalog"]["items"]}
        self.assertFalse(cards["video_editor"]["desired_enabled"])
        self.assertTrue(cards["video_editor"]["selected_default"])
        self.assertTrue(cards["video_editor"]["can_change"])
        self.assertFalse(cards["files_data"]["can_change"])
        self.assertTrue(all(x["installed"] == "unknown" for x in cards.values()))
        self.connection.fetchrow.assert_not_awaited()

    async def test_replay_returns_original_revision_with_current_or_unavailable_catalog(self):
        authority = SimpleNamespace(mutate=AsyncMock())
        self.service.authority = authority
        self.service.changes_enabled = True
        for state in (PluginPolicyState(12, policy(revision=12)), PluginPolicyState(12, None)):
            authority.mutate.return_value = PluginMutationReceipt(8, True, state)
            response = await self.client.put("/plugins/policy", json=command().model_dump(mode="json"),
                headers={"Authorization": "Bearer synthetic-plugin-test"})
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertEqual((result["applied_revision"], result["request_replayed"]), (8, True))
            self.assertEqual(result["catalog"]["policy_revision"], 12 if state.policy else None)
            if state.policy is None:
                self.assertFalse(any(x["can_change"] for x in result["catalog"]["items"]))

    async def test_catalog_gate_requires_current_persisted_choices(self):
        self.service.authority = SimpleNamespace(mutate=AsyncMock())
        self.service.changes_enabled = True
        for revision, available, enabled in [(7, True, True), (7, False, False), (0, True, False)]:
            self.connection.fetchrow.return_value = dict(revision=revision, policy_payload_id=PAYLOAD,
                availability="available" if available else "deleted",
                content=StoredPluginPolicy(intent=command(), policy=policy(revision=revision)).model_dump(mode="json")
                    if available else None)
            response = await self.client.get("/plugins", headers={"Authorization": "Bearer synthetic-plugin-test"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["items"][0]["can_change"], enabled)
        self.service.authority.mutate.assert_not_awaited()

    async def test_explicit_deployment_defaults_are_read_only_until_runtime_activation(self):
        baseline = CURATED_DEPLOYMENT_BASELINE_V1
        self.service.authority = SimpleNamespace(deployment_baseline=baseline, mutate=AsyncMock())
        headers = {"Authorization": "Bearer synthetic-plugin-test"}
        for active in (False, True):
            self.service.changes_enabled = active
            response = await self.client.get("/plugins", headers=headers)
            self.assertEqual(response.status_code, 200)
            catalog = response.json()
            self.assertEqual(catalog["policy_revision"], 0)
            self.assertTrue(catalog["policy_available"])
            self.assertEqual({x["plugin_id"]: x["configuration_id"] for x in catalog["items"]},
                             {x.plugin_id: str(x.configuration_id) for x in baseline.policy.entries})
            self.assertTrue(all(x["desired_enabled"] and x["selected_default"] for x in catalog["items"]))
            self.assertTrue(all(x["can_change"] == active for x in catalog["items"]))
            self.assertTrue(all(x["installed"] == "unknown" and x["readiness"] == "unknown"
                                for x in catalog["items"]))
        self.service.authority.mutate.assert_not_awaited()

    async def test_existing_unavailable_or_corrupt_head_never_returns_deployment_defaults(self):
        self.service.authority = SimpleNamespace(
            deployment_baseline=CURATED_DEPLOYMENT_BASELINE_V1, mutate=AsyncMock())
        self.service.changes_enabled = True
        for availability, content in [("deleted", None), ("available", {}), ("available", {"schema_version": 2})]:
            self.connection.fetchrow.return_value = dict(revision=1, policy_payload_id=PAYLOAD,
                availability=availability, content=content)
            result = (await self.client.get("/plugins", headers={
                "Authorization": "Bearer synthetic-plugin-test"})).json()
            self.assertFalse(result["policy_available"])
            self.assertIsNone(result["policy_revision"])
            self.assertTrue(all(not x["can_change"] and x["desired_enabled"] is None for x in result["items"]))

    async def test_revision_zero_requires_exact_baseline_and_no_account_payload(self):
        baseline = CURATED_DEPLOYMENT_BASELINE_V1
        self.service.authority = SimpleNamespace(deployment_baseline=baseline, mutate=AsyncMock())
        self.service.changes_enabled = True
        for state in (PluginPolicyState(0, policy(revision=0)),
                      PluginPolicyState(0, baseline.policy, PAYLOAD),
                      PluginPolicyState(1, baseline.policy, PAYLOAD)):
            catalog = self.service._catalog(state)
            self.assertFalse(catalog.policy_available)
            self.assertTrue(all(not x.can_change for x in catalog.items))
        self.service.authority.deployment_baseline = baseline.policy
        with self.assertRaises(TypeError):
            await self.service.get(owner_id="owner")
        self.connection.fetchrow.assert_not_awaited()

    async def test_domain_conflicts_do_not_masquerade_as_account_authentication_loss(self):
        authority = SimpleNamespace(mutate=AsyncMock())
        self.service.authority = authority
        self.service.changes_enabled = True
        for original, status, expected in [
            ("context_receipt_conflict", 409, "plugin_policy_mutation_conflict"),
            ("plugin_request_unavailable", 403, "plugin_policy_source_unavailable"),
            ("context_source_removed", 403, "plugin_policy_source_unavailable"),
            ("plugin_policy_receipt_unavailable", 409, "plugin_policy_receipt_unavailable")]:
            authority.mutate.side_effect = JarvisHTTPError(status, original, "Synthetic conflict")
            response = await self.client.put("/plugins/policy", json=command().model_dump(mode="json"),
                headers={"Authorization": "Bearer synthetic-plugin-test"})
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["error"]["code"], expected)

    async def test_unauthenticated_write_never_reaches_enabled_authority(self):
        authority = SimpleNamespace(mutate=AsyncMock())
        self.service.authority = authority
        self.service.changes_enabled = True
        response = await self.client.put("/plugins/policy", json=command().model_dump(mode="json"))
        self.assertEqual(response.status_code, 401)
        authority.mutate.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
