"""Offline composition tests; no SQL, workers, providers or runtime startup."""
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
import unittest
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from app.errors import JarvisHTTPError
from app.plugins.authority import ManagedPluginBinding, PluginAuthority, SubmittedManagedJob
from app.plugins.models import DefaultChoice, PluginChoice, PolicySnapshot
from app.plugins.policy import (
    CATEGORIES, UI_NAMESPACE, PluginPolicyChange, PluginPolicyState, changed_policy,
)
from app.plugins.repository import PluginPolicyRepository
from app.tasks.context_events import ContextRecord
from app.tasks.models import TaskError


OWNER = " opaque account "
NOW = datetime(2026, 11, 2, tzinfo=timezone.utc)


def policy(revision=1):
    return PolicySnapshot(revision=revision,
        entries=tuple(PluginChoice(plugin_id=name, desired_enabled=True,
            configuration_id=uuid5(NAMESPACE_URL, "test-plugin:" + name)) for name in CATEGORIES),
        defaults=tuple(DefaultChoice(category=category, plugin_id=name) for name, category in CATEGORIES.items()))


def command(*, revision=1, mutation_id=None, plugin="browser", enabled=False, defaults=()):
    return PluginPolicyChange(mutation_id=mutation_id or uuid4(), expected_revision=revision,
        enabled_changes=({"plugin_id": plugin, "enabled": enabled},), default_changes=defaults)


class Connection:
    def __init__(self, owner=OWNER, active=True):
        self.owner, self.active = owner, active

    def is_in_transaction(self):
        return self.active


class Database:
    def __init__(self):
        initial = policy()
        self.data = {"policies": {OWNER: PluginPolicyState(1, initial, uuid4())},
                     "ingress": {}, "applied": {}, "appends": 0}
        self.calls = []
        self.fail_append = False
        self.fail_read = False

    def check(self, connection, owner):
        if not connection.active or connection.owner != owner:
            raise AssertionError("Wrong borrowed owner transaction")


class Tasks:
    def __init__(self, db):
        self.db = db
        self.transactions = 0

    @asynccontextmanager
    async def transaction(self, owner):
        self.transactions += 1
        connection = Connection(owner)
        previous = deepcopy(self.db.data)
        self.db.calls.append(("owner_transaction", connection, owner))
        try:
            yield connection
        except BaseException:
            self.db.data = previous
            raise
        finally:
            connection.active = False


class Contexts:
    publication = None

    def __init__(self, db):
        self.db = db

    async def append_in_transaction(self, connection, owner, event):
        self.db.check(connection, owner)
        self.db.calls.append(("ingress", connection, event))
        key = (owner, event.source_namespace, event.source_event_id, event.receipt_id)
        parameters = event.model_dump(mode="json")
        prior = self.db.data["ingress"].get(key)
        if prior:
            if prior[0] != parameters:
                raise TaskError("context_receipt_conflict", "Same source, different intent")
            return prior[1]
        record = ContextRecord(id=uuid4(), owner_id=owner, session_id=event.session_id,
            sequence=len(self.db.data["ingress"]) + 1, origin_sequence=len(self.db.data["ingress"]) + 1,
            source_namespace=event.source_namespace, source_event_id=event.source_event_id,
            revision=1, supersedes_event_id=None, kind=event.kind, trust=event.trust,
            finality=event.finality, source_time=event.source_time, recorded_at=NOW,
            payload_id=uuid4(), availability="available", content_digest=None,
            content=event.payload, is_current=True)
        self.db.data["ingress"][key] = (parameters, record)
        return record


class Policies(PluginPolicyRepository):
    """Substitute storage, retaining the production source-authority guard."""
    def __init__(self, db):
        self.db = db

    async def read(self, connection, owner, *, lock=False):
        self.db.check(connection, owner)
        self.db.calls.append(("read", connection, lock))
        if self.db.fail_read:
            raise RuntimeError("read unavailable")
        return self.db.data["policies"].get(owner, PluginPolicyState(0, None))

    async def applied(self, connection, owner, request, source):
        self.db.check(connection, owner)
        self.require_source(connection, owner, request, source)
        self.db.calls.append(("applied", connection, source))
        receipt = self.db.data["applied"].get((owner, source.id))
        if receipt is None:
            return None
        if receipt[0] != request.canonical_intent():
            raise JarvisHTTPError(409, "plugin_policy_mutation_conflict", "Different intent")
        return receipt[1]

    async def append(self, connection, owner, request, source, previous, changed):
        self.db.check(connection, owner)
        self.require_source(connection, owner, request, source)
        self.db.calls.append(("append", connection, changed))
        if self.db.fail_append:
            raise RuntimeError("append unavailable")
        if request.expected_revision != previous.revision or changed.revision != previous.revision + 1:
            raise AssertionError("Invalid CAS")
        current = PluginPolicyState(changed.revision, changed, uuid4())
        self.db.data["policies"][owner] = current
        self.db.data["applied"][(owner, source.id)] = (request.canonical_intent(), changed.revision)
        self.db.data["appends"] += 1
        return current


class PluginAuthorityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = Database()
        self.tasks = Tasks(self.db)
        self.contexts = Contexts(self.db)
        self.policies = Policies(self.db)
        self.authority = PluginAuthority(self.policies, self.contexts, self.tasks)

    async def assert_denied(self, code, operation):
        with self.assertRaises(JarvisHTTPError) as caught:
            await operation
        self.assertEqual(caught.exception.code, code)

    async def binding(self):
        return await self.authority.select_new_work(owner_id=OWNER, plugin_id="browser")

    async def test_mutation_constructs_exact_ui_ingress_under_one_owner_lock(self):
        request = command()
        result = await self.authority.mutate(owner_id=OWNER, command=request)
        self.assertEqual((result.applied_revision, result.replayed), (2, False))
        self.assertEqual(self.tasks.transactions, 1)
        event = next(item[2] for item in self.db.calls if item[0] == "ingress")
        self.assertEqual(event.source_namespace, "ui:plugins.policy")
        self.assertEqual(event.source_event_id, str(request.mutation_id))
        self.assertEqual(event.receipt_id, str(request.mutation_id))
        self.assertEqual(event.expected_revision, 0)
        self.assertEqual((event.kind, event.trust, event.finality), ("user_request", "user_input", "final"))
        self.assertIsNone(event.session_id)
        self.assertIsNone(event.source_time)
        self.assertEqual(event.payload, request.ingress_payload())
        self.assertEqual(len({id(item[1]) for item in self.db.calls}), 1)
        self.assertEqual([item[0] for item in self.db.calls],
                         ["owner_transaction", "ingress", "applied", "read", "append"])

    async def test_borrowed_connection_never_opens_a_nested_transaction(self):
        current = Connection()
        await self.authority.mutate(owner_id=OWNER, command=command(), connection=current)
        await self.authority.select_new_work(owner_id=OWNER, plugin_id="files_data", connection=current)
        self.assertEqual(self.tasks.transactions, 0)
        self.assertTrue(all(item[1] is current for item in self.db.calls))
        self.assertTrue(current.active)

    async def test_existing_publication_transaction_order_is_reused(self):
        calls = []
        tasks = self.tasks

        class Publication:
            @asynccontextmanager
            async def transaction(self, actual_tasks, owner):
                self_outer = actual_tasks
                calls.append(("source_before_task", owner, self_outer is tasks))
                async with actual_tasks.transaction(owner) as connection:
                    yield connection

        self.contexts.publication = Publication()
        await self.authority.mutate(owner_id=OWNER, command=command())
        self.assertEqual(calls, [("source_before_task", OWNER, True)])

    async def test_exact_replay_precedes_stale_cas_and_keeps_original_revision(self):
        original = command()
        await self.authority.mutate(owner_id=OWNER, command=original)
        await self.authority.mutate(owner_id=OWNER, command=command(revision=2, plugin="files_data"))
        self.db.calls.clear()
        replay = await self.authority.mutate(owner_id=OWNER, command=original)
        self.assertEqual((replay.applied_revision, replay.current.revision, replay.replayed), (2, 3, True))
        self.assertEqual([item[0] for item in self.db.calls], ["owner_transaction", "ingress", "applied", "read"])
        self.assertEqual(self.db.data["appends"], 2)

    async def test_equivalent_reordered_changes_replay_exactly(self):
        identity = uuid4()
        changes = ({"plugin_id": "browser", "enabled": False}, {"plugin_id": "files_data", "enabled": False})
        first = PluginPolicyChange(mutation_id=identity, expected_revision=1, enabled_changes=changes)
        second = PluginPolicyChange(mutation_id=identity, expected_revision=1, enabled_changes=changes[::-1])
        await self.authority.mutate(owner_id=OWNER, command=first)
        self.assertTrue((await self.authority.mutate(owner_id=OWNER, command=second)).replayed)
        self.assertEqual(self.db.data["appends"], 1)

    async def test_same_mutation_changed_intent_or_revision_conflicts(self):
        original = command()
        await self.authority.mutate(owner_id=OWNER, command=original)
        for different in (command(mutation_id=original.mutation_id, enabled=True),
                          command(mutation_id=original.mutation_id, revision=2)):
            with self.subTest(command=different):
                with self.assertRaises(TaskError) as caught:
                    await self.authority.mutate(owner_id=OWNER, command=different)
                self.assertEqual(caught.exception.code, "context_receipt_conflict")
        self.assertEqual(self.db.data["appends"], 1)

    async def test_stale_new_mutation_rolls_back_its_ingress(self):
        await self.authority.mutate(owner_id=OWNER, command=command())
        before = deepcopy(self.db.data)
        await self.assert_denied("plugin_policy_revision_conflict",
            self.authority.mutate(owner_id=OWNER, command=command()))
        self.assertEqual(self.db.data, before)

    async def test_append_failure_rolls_back_ingress_and_does_not_claim_success(self):
        self.db.fail_append = True
        before = deepcopy(self.db.data)
        with self.assertRaisesRegex(RuntimeError, "append unavailable"):
            await self.authority.mutate(owner_id=OWNER, command=command())
        self.assertEqual(self.db.data, before)

    async def test_absent_withdrawn_and_unproven_zero_baseline_do_not_bootstrap(self):
        for state in (PluginPolicyState(0, None), PluginPolicyState(8, None, uuid4()),
                      PluginPolicyState(0, policy(0))):
            with self.subTest(state=state):
                self.db.data["policies"][OWNER] = state
                await self.assert_denied("plugin_policy_unavailable",
                    self.authority.mutate(owner_id=OWNER, command=command(revision=state.revision)))
                await self.assert_denied("plugin_policy_unavailable", self.binding())
                self.assertEqual(self.db.data["appends"], 0)
                self.assertEqual(self.db.data["ingress"], {})

    async def test_exact_replay_can_report_original_receipt_with_current_unknown(self):
        request = command()
        await self.authority.mutate(owner_id=OWNER, command=request)
        self.db.data["policies"][OWNER] = PluginPolicyState(3, None, uuid4())
        result = await self.authority.mutate(owner_id=OWNER, command=request)
        self.assertTrue(result.replayed)
        self.assertEqual(result.applied_revision, 2)
        self.assertIsNone(result.current.policy)
        self.assertEqual(self.db.data["appends"], 1)

    async def test_withdrawn_ui_source_does_not_reauthorize_replay(self):
        request = command()
        await self.authority.mutate(owner_id=OWNER, command=request)
        key = (OWNER, UI_NAMESPACE, str(request.mutation_id), str(request.mutation_id))
        values, record = self.db.data["ingress"][key]
        self.db.data["ingress"][key] = (values, record.model_copy(update={"availability": "unavailable", "content": None}))
        await self.assert_denied("plugin_request_unavailable", self.authority.mutate(owner_id=OWNER, command=request))

    async def test_noop_enabled_choice_preserves_config_and_disable_changes_it(self):
        first = self.db.data["policies"][OWNER].policy
        old = {entry.plugin_id: entry.configuration_id for entry in first.entries}
        unchanged = await self.authority.mutate(owner_id=OWNER, command=command(enabled=True))
        self.assertEqual({entry.plugin_id: entry.configuration_id for entry in unchanged.current.policy.entries}, old)
        disabled = await self.authority.mutate(owner_id=OWNER, command=command(revision=2))
        current = {entry.plugin_id: entry for entry in disabled.current.policy.entries}
        self.assertNotEqual(current["browser"].configuration_id, old["browser"])
        self.assertEqual(current["files_data"].configuration_id, old["files_data"])
        self.assertEqual(
            {choice.category: choice.plugin_id for choice in disabled.current.policy.defaults},
            {choice.category: choice.plugin_id for choice in first.defaults},
        )

    async def test_policy_change_can_preserve_a_disabled_default(self):
        result = await self.authority.mutate(owner_id=OWNER, command=command(defaults=(
            {"category": "web_browsing", "plugin_id": "browser"},)))
        self.assertIn(DefaultChoice(category="web_browsing", plugin_id="browser"), result.current.policy.defaults)
        await self.assert_denied("plugin_disabled",
            self.authority.select_new_work(owner_id=OWNER, category="web_browsing"))

    async def test_owner_scope_is_opaque_and_never_normalized(self):
        binding = await self.binding()
        self.assertEqual(binding.owner_id, OWNER)
        await self.assert_denied("plugin_binding_owner_mismatch",
            self.authority.require_new_work(Connection(owner=OWNER.strip()), owner_id=OWNER.strip(), binding=binding))

    async def test_new_work_callback_checks_latest_policy_after_selection(self):
        binding = await self.binding()
        check = self.authority.new_work_check(owner_id=OWNER, binding=binding)
        self.assertEqual(await check(Connection()), binding)
        await self.authority.mutate(owner_id=OWNER, command=command())
        for route in ("new_job", "queued_unsubmitted", "selection", "reinstall", "preparation"):
            with self.subTest(route=route):
                await self.assert_denied("plugin_disabled", check(Connection()))

    async def test_unrelated_policy_revision_does_not_invalidate_binding(self):
        binding = await self.binding()
        await self.authority.mutate(owner_id=OWNER, command=command(plugin="files_data"))
        self.assertEqual(await self.authority.require_new_work(Connection(), owner_id=OWNER, binding=binding), binding)

    async def test_disable_reenable_cannot_revive_an_old_queued_configuration(self):
        binding = await self.binding()
        await self.authority.mutate(owner_id=OWNER, command=command())
        await self.authority.mutate(owner_id=OWNER, command=command(revision=2, enabled=True))
        await self.assert_denied("plugin_configuration_changed",
            self.authority.require_new_work(Connection(), owner_id=OWNER, binding=binding))
        current = await self.binding()
        self.assertNotEqual(current.configuration_id, binding.configuration_id)

    async def test_missing_binding_or_model_dictionary_cannot_downgrade_managed_work(self):
        for binding in (None, {"plugin_id": "browser"}, "browser"):
            with self.subTest(binding=binding):
                await self.assert_denied("plugin_binding_required",
                    self.authority.require_new_work(Connection(), owner_id=OWNER, binding=binding))
                with self.assertRaises(JarvisHTTPError):
                    self.authority.new_work_check(owner_id=OWNER, binding=binding)

    async def test_binding_rejects_future_revision_and_wrong_configuration(self):
        binding = await self.binding()
        for wrong in (replace(binding, policy_revision=2), replace(binding, configuration_id=uuid4())):
            await self.assert_denied("plugin_configuration_changed",
                self.authority.require_new_work(Connection(), owner_id=OWNER, binding=wrong))

    async def test_default_selection_is_exact_and_missing_default_is_not_fallback(self):
        chosen = await self.authority.select_new_work(owner_id=OWNER, category="web_browsing")
        self.assertEqual(chosen.plugin_id, "browser")
        snapshot = self.db.data["policies"][OWNER]
        self.db.data["policies"][OWNER] = replace(snapshot, policy=snapshot.policy.model_copy(update={"defaults": ()}))
        await self.assert_denied("plugin_default_unavailable",
            self.authority.select_new_work(owner_id=OWNER, category="web_browsing"))

    async def test_missing_choice_does_not_fall_back(self):
        snapshot = self.db.data["policies"][OWNER]
        current = PolicySnapshot(revision=1, entries=tuple(e for e in snapshot.policy.entries if e.plugin_id != "browser"))
        self.db.data["policies"][OWNER] = replace(snapshot, policy=current)
        await self.assert_denied("plugin_choice_unavailable", self.binding())

    async def test_selection_requires_exactly_one_known_selector(self):
        for fields in ({}, {"plugin_id": "browser", "category": "web_browsing"},
                       {"plugin_id": "terminal"}, {"category": "unknown"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                await self.authority.select_new_work(owner_id=OWNER, **fields)

    async def test_current_policy_revision_must_match_payload(self):
        self.db.data["policies"][OWNER] = PluginPolicyState(2, policy(1), uuid4())
        await self.assert_denied("plugin_policy_unavailable", self.binding())

    async def test_read_failure_propagates_without_enabling_or_fallback(self):
        self.db.fail_read = True
        with self.assertRaisesRegex(RuntimeError, "read unavailable"):
            await self.binding()

    async def test_nontransactional_connections_are_rejected(self):
        binding = await self.binding()
        for call in (
            self.authority.mutate(owner_id=OWNER, command=command(), connection=Connection(active=False)),
            self.authority.select_new_work(owner_id=OWNER, plugin_id="browser", connection=Connection(active=False)),
            self.authority.require_new_work(Connection(active=False), owner_id=OWNER, binding=binding),
        ):
            with self.assertRaises(RuntimeError):
                await call

    async def test_raw_ui_json_and_invalid_owner_are_not_authority(self):
        with self.assertRaises(TypeError):
            await self.authority.mutate(owner_id=OWNER, command=command().model_dump(mode="json"))
        for owner in (None, 42, " ", "owner\x00suffix"):
            with self.subTest(owner=owner), self.assertRaises(ValueError):
                await self.authority.mutate(owner_id=owner, command=command())

    async def test_submitted_job_keeps_exact_scope_after_disable_and_withdrawal(self):
        binding = await self.binding()
        pin = SubmittedManagedJob(binding, uuid4(), uuid4())
        await self.authority.mutate(owner_id=OWNER, command=command())
        calls = []

        async def read(connection, owner, job, token):
            calls.append((connection, owner, job, token))
            return pin

        for unavailable in (False, True):
            if unavailable:
                self.db.data["policies"][OWNER] = PluginPolicyState(3, None, uuid4())
            self.db.calls.clear()
            for purpose in ("continue", "status", "cleanup"):
                connection = Connection()
                returned = await self.authority.require_submitted_scope(connection, owner_id=OWNER,
                    binding=binding, job_id=pin.job_id, submission_token=pin.submission_token,
                    purpose=purpose, read_submitted=read)
                self.assertEqual(returned, pin)
                self.assertEqual(calls[-1], (connection, OWNER, pin.job_id, pin.submission_token))
            self.assertEqual(self.db.calls, [])

    async def test_queued_or_unknown_submission_cannot_use_submitted_exemption(self):
        binding = await self.binding()
        await self.authority.mutate(owner_id=OWNER, command=command())

        async def read(*args):
            return None

        await self.assert_denied("plugin_submitted_scope_unavailable",
            self.authority.require_submitted_scope(Connection(), owner_id=OWNER, binding=binding,
                job_id=uuid4(), submission_token=uuid4(), purpose="continue", read_submitted=read))

    async def test_submitted_reader_must_return_exact_owner_job_token_and_binding(self):
        binding = await self.binding()
        pin = SubmittedManagedJob(binding, uuid4(), uuid4())
        invalid = (None, {"submitted": True}, replace(pin, job_id=uuid4()),
            replace(pin, submission_token=uuid4()),
            replace(pin, binding=replace(binding, owner_id="another-owner")),
            replace(pin, binding=replace(binding, configuration_id=uuid4())),
            replace(pin, binding=replace(binding, policy_revision=2)))
        for value in invalid:
            async def read(*args):
                return value
            with self.subTest(value=value):
                await self.assert_denied("plugin_submitted_scope_unavailable",
                    self.authority.require_submitted_scope(Connection(), owner_id=OWNER, binding=binding,
                        job_id=pin.job_id, submission_token=pin.submission_token, purpose="status", read_submitted=read))

    async def test_submitted_pin_does_not_admit_new_selection_reinstall_or_preparation(self):
        binding = await self.binding()

        async def read(*args):
            self.fail("New work must not consult the submitted-job exemption")

        for purpose in ("selection", "reinstall", "preparation", "new_job"):
            await self.assert_denied("plugin_new_work_requires_admission",
                self.authority.require_submitted_scope(Connection(), owner_id=OWNER, binding=binding,
                    job_id=uuid4(), submission_token=uuid4(), purpose=purpose, read_submitted=read))

    async def test_submitted_reader_failure_is_not_treated_as_permission(self):
        binding = await self.binding()

        async def read(*args):
            raise RuntimeError("canonical receipt unavailable")

        with self.assertRaisesRegex(RuntimeError, "canonical receipt unavailable"):
            await self.authority.require_submitted_scope(Connection(), owner_id=OWNER, binding=binding,
                job_id=uuid4(), submission_token=uuid4(), purpose="cleanup", read_submitted=read)

    async def test_submitted_scope_requires_transaction_and_canonical_identity_types(self):
        binding = await self.binding()

        async def read(*args):
            self.fail("Invalid scopes must not reach the reader")

        with self.assertRaises(RuntimeError):
            await self.authority.require_submitted_scope(Connection(active=False), owner_id=OWNER,
                binding=binding, job_id=uuid4(), submission_token=uuid4(), purpose="status", read_submitted=read)
        await self.assert_denied("plugin_submitted_scope_unavailable",
            self.authority.require_submitted_scope(Connection(), owner_id=OWNER, binding=binding,
                job_id="not-a-uuid", submission_token=uuid4(), purpose="status", read_submitted=read))

    def test_bindings_are_immutable_and_validate_identity_shape(self):
        binding = ManagedPluginBinding(OWNER, "browser", uuid4(), 1)
        with self.assertRaises(FrozenInstanceError):
            binding.plugin_id = "files_data"
        for fields in ({"owner_id": " "}, {"plugin_id": "terminal"}, {"configuration_id": "not-a-uuid"},
                       {"policy_revision": True}, {"policy_revision": 0}):
            with self.subTest(fields=fields), self.assertRaises((ValueError, TypeError)):
                replace(binding, **fields)
        with self.assertRaises(TypeError):
            SubmittedManagedJob(binding, uuid4(), "not-a-token")


if __name__ == "__main__":
    unittest.main()
