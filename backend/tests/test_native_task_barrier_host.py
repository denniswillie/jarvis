"""Offline host-hook qualification with controlled delegates and canonical pool.

No SQL, provider, native app or worker is launched. The real tool definition,
invocation binding, witness mapper, canonical observer and file exchange run.
"""

import asyncio
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.tasks.models import TaskVoiceSource, digest
from app.tasks.voice_tools import BackgroundTaskToolModule
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.registry import JarvisToolRegistry, ToolDefinition
from evals.responsive_voice.native_task_barrier import BarrierError, decode_receipt
from evals.responsive_voice.native_task_barrier_host import NativeTaskBarrierHost
from test_native_task_barrier import NOW, Pool, case


class Repository:
    def __init__(self, pool):
        self.pool, self.saved, self.fail = pool, 0, None

    async def _save_request(self, connection, task, request_id, operation, parameters):
        self.saved += 1
        if self.fail is not None:
            raise self.fail
        return "original-save-result"


class ControlledModule(BackgroundTaskToolModule):
    def build_tools(self, context):
        return [self.service.definition]


@pytest.fixture
def setup(case):
    scope, witness, row = case
    params = {"voice_source": TaskVoiceSource(witness.request_event_id, witness.request_revision,
                                               scope.session_id).parameters(), "mode": "background"}
    witness = witness.model_copy(update={"parameters_digest": digest(params)})
    row["parameters_digest"] = witness.parameters_digest
    pool, task = Pool(scope, row), SimpleNamespace(id=witness.task_id, revision=1,
        owner_id=scope.owner_id, session_id=scope.session_id)
    repo = Repository(pool)
    service = SimpleNamespace(repository=repo,
                              provider_configured=lambda provider: provider == "frontier")
    async def original(arguments):
        value = await repo._save_request(None, task, witness.request_id, "create", params)
        assert value == "original-save-result"
        return {"original": arguments}
    service.definition = ToolDefinition("start_background_task", "unchanged-description",
        {"type": "object"}, original, mutates_state=True, audit_arguments=lambda x: x,
        audit_result=lambda x: x)
    module = ControlledModule(service)
    runtime = SimpleNamespace(db_pool=pool, background_task_service=service, tool_modules=(module,))
    return SimpleNamespace(scope=scope, witness=witness, pool=pool, task=task, params=params,
                           repo=repo, service=service, module=module, runtime=runtime)


def attach(s, **kwargs):
    return NativeTaskBarrierHost.attach(s.runtime, run_id=s.scope.run_id, barrier_id=s.scope.barrier_id,
        owner_id=s.scope.owner_id, database_schema=s.scope.database_schema, model=s.scope.model,
        host_expires_at=s.scope.host_expires_at, poll_interval_seconds=0.01,
        clock=lambda: NOW, **kwargs)


def readiness(s):
    return dict(status="ready", registry_ready=True, run_id=s.scope.run_id.hex,
        owner_id=s.scope.owner_id, database_schema=s.scope.database_schema, model=s.scope.model,
        worker_id=s.scope.worker_id, host_id=s.scope.runtime_host_id,
        host_expires_at=s.scope.host_expires_at.isoformat(), worker_release_root="DO_NOT_USE")


def definition(s, *, session_id=None):
    registry = JarvisToolRegistry()
    registry.register_many(s.runtime.tool_modules[0].build_tools(
        ToolBuildContext(None, s.scope.owner_id, session_id or s.scope.session_id)))
    return registry.get("start_background_task")


async def invoke(s, *, handler=None, call_id=None, event_id=None, revision=None):
    with bind_tool_invocation(provider_call_id=call_id or s.witness.provider_call_id,
        request_event_id=event_id or s.witness.request_event_id,
        request_revision=revision or s.witness.request_revision):
        return await (handler or definition(s).handler)({"sensitive": "not-in-receipts"})


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.002)


def test_attach_is_unconfigured_preserves_definition_and_restores_only_owned_hooks(setup):
    async def run():
        s = setup
        original = s.repo._save_request.__func__
        host = attach(s)
        d = definition(s)
        for name in ("name", "description", "input_schema", "mutates_state", "audit_arguments", "audit_result",
                     "image_resolver", "result_capture", "capture_after_commit"):
            assert getattr(d, name) == getattr(s.service.definition, name)
        assert host.poll_task is None and not host.status()["configured"]
        assert await invoke(s) == {"original": {"sensitive": "not-in-receipts"}}
        assert not s.pool.queries and host.scope is None
        other_module = object()
        s.runtime.tool_modules += (other_module,)
        closed = await host.close()
        assert closed["joined"] and closed["hooks_restored"] and not closed["worker_release_authorized"]
        assert s.runtime.tool_modules == (s.module, other_module)
        assert s.repo._save_request.__func__ is original
        assert (await host.close())["joined"]
    asyncio.run(run())


def test_late_configuration_uses_observed_session_committed_read_and_never_releases_worker(setup, tmp_path):
    async def run():
        s, host = setup, attach(setup)
        await invoke(s)
        assert host.exchange is None and not s.pool.queries
        tmp_path.chmod(0o700)
        await host.configure(worker_readiness=readiness(s), control_root=tmp_path)
        try:
            await until(lambda: host.reason == "active_claim_observed")
            receipt = decode_receipt(host.exchange.read())
            assert receipt.scope == s.scope and receipt.admission == s.witness
            assert receipt.claim.attempt_id == s.pool.row["attempt_id"]
            assert s.pool.transactions and all(t == dict(isolation="repeatable_read", readonly=True)
                                              for t in s.pool.transactions)
            assert "sensitive" not in host.exchange.read().decode()
            assert {p.name for p in tmp_path.iterdir()} == {"task-barrier.json", "task-barrier.json.lock"}
        finally:
            closed = await host.close()
        assert closed["joined"] and closed["receipt_withdrawal"] == "blocked_or_expired"
        assert decode_receipt((tmp_path / "task-barrier.json").read_bytes()).status == "blocked"
        count = len(s.pool.queries)
        await asyncio.sleep(0.02)
        assert len(s.pool.queries) == count
    asyncio.run(run())


@pytest.mark.parametrize("field,value", [
    ("status", "starting"), ("registry_ready", False), ("run_id", str(uuid4())),
    ("owner_id", "other"), ("database_schema", "public"), ("model", "other"),
    ("host_expires_at", (NOW + timedelta(seconds=50)).isoformat()),
    ("worker_id", ""), ("host_id", ""),
])
def test_readiness_mismatch_does_not_configure_or_write(setup, tmp_path, field, value):
    async def run():
        host = attach(setup)
        try:
            with pytest.raises(BarrierError, match="worker_readiness_mismatch"):
                await host.configure(worker_readiness={**readiness(setup), field: value}, control_root=tmp_path)
            assert host.poll_task is None and host.exchange is None and list(tmp_path.iterdir()) == []
        finally:
            await host.close()
    asyncio.run(run())


def test_no_call_or_missing_finalized_invocation_never_fabricates_receipt(setup, tmp_path):
    async def run():
        host = attach(setup)
        tmp_path.chmod(0o700)
        await host.configure(worker_readiness=readiness(setup), control_root=tmp_path)
        try:
            with bind_tool_invocation(provider_call_id="call-without-source"):
                assert await definition(setup).handler({}) == {"original": {}}
            await asyncio.sleep(0.03)
            assert host.reason == "missing_finalized_create" and host.admission is None
            assert not setup.pool.queries and list(tmp_path.iterdir()) == []
        finally:
            await host.close()
    asyncio.run(run())


def test_attempted_write_remains_blocked_until_independent_committed_read(setup, tmp_path):
    async def run():
        s, host = setup, attach(setup)
        committed = s.pool.row
        s.pool.row = None
        tmp_path.chmod(0o700)
        await host.configure(worker_readiness=readiness(s), control_root=tmp_path)
        try:
            await invoke(s)
            await until(lambda: host.reason == "admission_not_committed")
            assert decode_receipt(host.exchange.read()).status == "blocked"
            s.pool.row = committed
            await until(lambda: host.reason == "active_claim_observed")
        finally:
            assert (await host.close())["joined"]
    asyncio.run(run())


def test_original_handler_error_and_repository_error_are_preserved(setup):
    async def run():
        s = setup
        marker = RuntimeError("original business failure")
        async def fail(arguments):
            await s.repo._save_request(None, s.task, s.witness.request_id, "create", s.params)
            raise marker
        s.service.definition = replace_handler(s.service.definition, fail)
        host = attach(s)
        try:
            with pytest.raises(RuntimeError) as caught:
                await invoke(s)
            assert caught.value is marker and host.calls[0].done
            s.repo.fail = marker
            with pytest.raises(RuntimeError) as caught:
                await invoke(s)
            assert caught.value is marker and not host.calls[1].notes
        finally:
            await host.close()
    asyncio.run(run())


def replace_handler(d, handler):
    from dataclasses import replace
    return replace(d, handler=handler)


def test_duplicate_real_calls_reuse_one_admission_and_ambiguous_task_withdraws(setup, tmp_path):
    async def run():
        s, host = setup, attach(setup)
        tmp_path.chmod(0o700)
        await host.configure(worker_readiness=readiness(s), control_root=tmp_path)
        try:
            await invoke(s)
            await invoke(s)
            await invoke(s, call_id="independent-provider-retry")
            await until(lambda: host.reason == "active_claim_observed")
            pinned = host.admission
            assert len(host.calls) == 3 and host.admission.task_id == s.witness.task_id
            s.task.id = uuid4()
            await invoke(s, call_id="second-task-create")
            await until(lambda: host.reason == "ambiguous_admissions")
            assert host.admission == pinned
            assert decode_receipt(host.exchange.read()).status == "blocked"
            previous_sequence = host.sequence
            await asyncio.sleep(0.03)
            assert previous_sequence <= host.sequence <= previous_sequence + 10
            assert not host.poll_task.done()
        finally:
            await host.close()
    asyncio.run(run())


@pytest.mark.parametrize("field,value,reason", [
    ("revision", 2, "task_revision_replaced"),
    ("status", "cancel_requested", "task_not_active"),
    ("lease_until", NOW, "lease_expired"),
])
def test_canonical_change_withdraws_without_any_new_handler(setup, tmp_path, field, value, reason):
    async def run():
        s, host = setup, attach(setup)
        tmp_path.chmod(0o700)
        await host.configure(worker_readiness=readiness(s), control_root=tmp_path)
        try:
            await invoke(s)
            await until(lambda: host.reason == "active_claim_observed")
            s.pool.row[field] = value
            await until(lambda: host.reason == reason)
            assert decode_receipt(host.exchange.read()).status == "blocked" and s.repo.saved == 1
        finally:
            await host.close()
    asyncio.run(run())


def test_close_does_not_cancel_inflight_business_handler_or_old_registered_delegate(setup, tmp_path):
    async def run():
        s = setup
        entered, release = asyncio.Event(), asyncio.Event()
        async def waiting(arguments):
            await s.repo._save_request(None, s.task, s.witness.request_id, "create", s.params)
            entered.set()
            await release.wait()
            return "business-finished"
        s.service.definition = replace_handler(s.service.definition, waiting)
        host = attach(s)
        d = definition(s)
        tmp_path.chmod(0o700)
        await host.configure(worker_readiness=readiness(s), control_root=tmp_path)
        job = asyncio.create_task(invoke(s, handler=d.handler))
        await entered.wait()
        closed = await host.close()
        assert closed["joined"] and closed["inflight_handlers"] == 1 and not job.done()
        release.set()
        assert await job == "business-finished"
        assert await invoke(s, handler=d.handler) == "business-finished"
        assert len(host.calls) == 1 and not s.pool.queries
    asyncio.run(run())


def test_capacity_latches_missing_prerequisite_but_business_calls_continue(setup, tmp_path):
    async def run():
        host = attach(setup, max_calls=1)
        await invoke(setup)
        assert await invoke(setup, call_id="another") == {"original": {"sensitive": "not-in-receipts"}}
        tmp_path.chmod(0o700)
        await host.configure(worker_readiness=readiness(setup), control_root=tmp_path)
        try:
            await until(lambda: host.reason == "observation_capacity")
            assert host.reason == "observation_capacity"
            assert host.admission is None and host.scope is None and host.sequence == 0
            assert not setup.pool.queries and setup.repo.saved == 2
            with pytest.raises(BarrierError, match="receipt_missing"):
                host.exchange.read()
        finally:
            await host.close()
    asyncio.run(run())


def test_real_background_tool_validation_is_unchanged(setup):
    async def run():
        s = setup
        real = BackgroundTaskToolModule(s.service)
        s.runtime.tool_modules = (real,)
        original = next(d for d in real.build_tools(ToolBuildContext(None, s.scope.owner_id, s.scope.session_id))
                        if d.name == "start_background_task")
        expected = await original.handler({})
        host = attach(s)
        try:
            assert await definition(s).handler({}) == expected
            assert not s.repo.saved and not host.calls
        finally:
            await host.close()
    asyncio.run(run())


def test_query_timeout_withdraws_and_joins_owned_poll(setup, tmp_path):
    async def run():
        host = attach(setup, query_timeout_seconds=0.01)
        async def blocked(*args):
            await asyncio.Event().wait()
        setup.pool.fetchrow = blocked
        tmp_path.chmod(0o700)
        await host.configure(worker_readiness=readiness(setup), control_root=tmp_path)
        await invoke(setup)
        try:
            await until(lambda: host.reason == "canonical_poll_timeout")
            assert decode_receipt(host.exchange.read()).reason == "canonical_unavailable"
        finally:
            assert (await host.close())["joined"]
    asyncio.run(run())


def test_close_pending_keeps_same_poll_handle_until_it_joins(setup, tmp_path):
    async def run():
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        class SlowObserver:
            def __init__(self, *args, **kwargs):
                pass
            async def observe(self):
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                return None  # Closing path must not inspect or publish this result.
        host = attach(setup, observer_factory=SlowObserver, close_timeout_seconds=0.01)
        tmp_path.chmod(0o700)
        await host.configure(worker_readiness=readiness(setup), control_root=tmp_path)
        await invoke(setup)
        await entered.wait()
        actual = host.poll_task
        pending = await host.close()
        assert pending["status"] == "close_pending" and not pending["joined"] and cancelled.is_set()
        assert host.poll_task is actual and not actual.done()
        release.set()
        await until(actual.done)
        assert (await host.close())["joined"] and host.poll_task is actual
    asyncio.run(run())


def test_second_attach_and_reconfigure_fail_without_global_mutation(setup, tmp_path):
    async def run():
        host = attach(setup)
        try:
            with pytest.raises(BarrierError, match="runtime_hook_binding_unavailable"):
                attach(setup)
            tmp_path.chmod(0o700)
            await host.configure(worker_readiness=readiness(setup), control_root=tmp_path)
            with pytest.raises(BarrierError, match="host_already_configured_or_closed"):
                await host.configure(worker_readiness=readiness(setup), control_root=tmp_path)
        finally:
            await host.close()
        assert "_save_request" not in Repository.__dict__ or Repository._save_request.__name__ == "_save_request"
    asyncio.run(run())


def test_close_preserves_replacement_hook_and_does_not_claim_restoration(setup):
    async def run():
        host = attach(setup)
        async def replacement(*args, **kwargs):
            return "another-owner"
        setup.repo._save_request = replacement
        closed = await host.close()
        assert closed["joined"] and closed["hooks_restored"] is False
        assert setup.repo._save_request is replacement
        assert setup.runtime.tool_modules == (setup.module,)
        again = await host.close()
        assert again["hooks_restored"] is False and setup.repo._save_request is replacement
    asyncio.run(run())
