"""Separate real worker process: scoped intake, Hermes/E2B work, and cleanup-only restart."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from dotenv import dotenv_values

from app.runtime.feature_config import SandboxConfiguration, TaskLimits
from app.sandbox.allocations import SandboxAllocationRepository
from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.models import E2BConfiguration, SandboxPolicy, SandboxScope
from app.settings import JarvisSettings
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.models import TaskCreate, TaskError, TaskStatus, WorkerOutcome, json_object
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from evals.second_brain.e2b_probe import Probe
from evals.second_brain.hermes_sandbox_probe import REQUEST
from evals.second_brain.source_fixture import source_fixture


async def collect_resources(probe, fixture, allocations):
    """SQL receipts plus independent owner-scoped discovery for missed receipts."""
    try:
        async with fixture.pool.acquire() as connection:
            rows = await connection.fetch("select * from jarvis_sandbox_allocations where owner_id=$1", fixture.owner)
        for row in rows:
            scope, policy = allocations.scope(row), allocations.policy(row)
            if not any(old.allocation_id==scope.allocation_id for old, _ in probe.scopes):
                probe.scopes.append((scope, policy))
                probe.record["resources"].append({"scope":scope.model_dump(mode="json"),
                    "policy":policy.model_dump(mode="json"), "phase":row["phase"],
                    "sandbox_id":row["sandbox_id"], "cleanup_observed_ids":row["cleanup_observed_ids"]})
        probe.persist()
    finally:
        # The fixture owner is unique and recorded before the child starts.
        # Partial metadata lookup is for test containment, never task admission.
        sdk = probe.provider.sdk
        pages = sdk.AsyncSandbox.list(api_key=probe.provider.configuration.api_key.get_secret_value(),
            request_timeout=20, query=sdk.SandboxQuery(metadata={"jarvis_kind":"task-sandbox-v1",
                "jarvis_owner":hashlib.sha256(fixture.owner.encode()).hexdigest()},
                state=[sdk.SandboxState.RUNNING, sdk.SandboxState.PAUSED]))
        while pages.has_next:
            for item in await pages.next_items():
                meta = item.metadata
                scope = SandboxScope(owner_id=fixture.owner, task_id=meta["jarvis_task"],
                    task_revision=int(meta["jarvis_revision"]), attempt_id=meta["jarvis_attempt"],
                    allocation_id=meta["jarvis_allocation"])
                policy = SandboxPolicy(lifetime_seconds=180, command_seconds=30)
                await probe.provider.info(item.sandbox_id, scope, policy)
                if not any(old.allocation_id==scope.allocation_id for old, _ in probe.scopes):
                    probe.scopes.append((scope, policy))
                    probe.record["resources"].append({"scope":scope.model_dump(mode="json"),
                        "policy":policy.model_dump(mode="json"), "sandbox_id":item.sandbox_id,
                        "phase":"discovered_after_worker"})
        probe.persist()


async def exercise(probe, args, settings):
    async with source_fixture(settings.SUPABASE_DB_URL, args.manifest_dir, test_name="native-sandbox-worker") as fixture:
        probe.record.update(database_manifest=str(fixture.fixture.path), fixture_owner=fixture.owner)
        probe.persist()
        root = probe.path.parent/"worker"
        root.mkdir(mode=0o700)
        key_file = root/"e2b-key"
        with open(key_file, "x", opener=lambda path, flags:os.open(path, flags, 0o600)) as stream:
            stream.write(probe.provider.configuration.api_key.get_secret_value())
        async with fixture.pool.acquire() as connection:
            schema = await connection.fetchval("select current_schema()")
        url = urlsplit(settings.SUPABASE_DB_URL)
        query = dict(parse_qsl(url.query))
        query["search_path"] = schema+",pg_catalog,public,extensions"
        dsn = urlunsplit((url.scheme,url.netloc,url.path,urlencode(query),url.fragment))
        environment = {
            "PATH":"/usr/bin:/bin", "PYTHONPATH":os.pathsep.join(
                [str(Path(__file__).resolve().parents[2]), *([args.sdk_path] if args.sdk_path else [])]),
            "PYTHONUNBUFFERED":"1", "JARVIS_OWNER_ID":fixture.owner, "SUPABASE_DB_URL":dsn,
            "OPENAI_API_KEY":settings.OPENAI_API_KEY,
            "SECOND_BRAIN_ENABLED":"true", "SECOND_BRAIN_TASKS_ENABLED":"true",
            "SECOND_BRAIN_FRONTIER_ENABLED":"true", "SECOND_BRAIN_FRONTIER_MODEL":args.model,
            "SECOND_BRAIN_FRONTIER_REASONING_EFFORT":"medium",
            "SECOND_BRAIN_HERMES_PYTHON":str(args.hermes_python),
            "SECOND_BRAIN_HERMES_WORK_ROOT":str(root/"native"),
            "SECOND_BRAIN_WORKER_HOST_ID":"native-sandbox-worker-fixture",
            "SECOND_BRAIN_WORKER_HEARTBEAT_SECONDS":"1", "SECOND_BRAIN_POLL_SECONDS":"0.2",
            "SECOND_BRAIN_LEASE_SECONDS":"60", "SECOND_BRAIN_ACTIVE_SECONDS":"300",
            "SECOND_BRAIN_REQUEST_OUTPUT_TOKENS":"4096", "SECOND_BRAIN_KNOWLEDGE_ENABLED":"false",
            "SECOND_BRAIN_SANDBOX_ENABLED":"true", "SECOND_BRAIN_SANDBOX_LIFETIME_SECONDS":"180",
            "SECOND_BRAIN_SANDBOX_COMMAND_SECONDS":"30", "SECOND_BRAIN_SANDBOX_CLEANUP_POLL_SECONDS":"1",
            "JARVIS_FRONTIER_E2B_API_KEY_FILE":str(key_file),
        }
        tasks = TaskRepository(fixture.pool, TaskLimits(lease_seconds=60, active_seconds=300,
                                                       request_output_tokens=4096, poll_seconds=.2))
        registry = TaskWorkerRegistry(fixture.pool)
        providers = {"frontier":ConfiguredTaskProvider(args.model)}
        policy = SandboxPolicy(lifetime_seconds=180, command_seconds=30)
        configuration = SandboxConfiguration(policy=policy)
        service = TaskService(tasks, providers, registry=registry, sandbox_configuration=configuration)
        allocations = SandboxAllocationRepository(tasks)
        process = drain = None

        async def start(enabled):
            nonlocal process, drain
            process = await asyncio.create_subprocess_exec(sys.executable, "-m", "app.tasks.worker", "run",
                cwd=root, env=environment|{"SECOND_BRAIN_SANDBOX_ENABLED":str(enabled).lower()},
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
            drain = asyncio.create_task(process.communicate())
            probe.record.setdefault("workers", []).append({"pid":process.pid, "sandbox_enabled":enabled, "phase":"running"})
            probe.persist()
            for _ in range(180):
                if process.returncode is not None:
                    raise RuntimeError("worker_startup_failed")
                if await registry.ready(fixture.owner, "frontier", args.model,
                    required_capabilities=("sandbox.execute",) if enabled else ()):
                    return
                await asyncio.sleep(.25)
            raise TimeoutError("worker_readiness")

        async def stop():
            nonlocal process, drain
            if process is None:
                return
            if process.returncode is None:
                process.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(asyncio.shield(drain), 25)
                except TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = await drain
            safe = (stdout+stderr).decode(errors="replace")[-6000:]
            for value in (settings.OPENAI_API_KEY, settings.SUPABASE_DB_URL, dsn,
                          probe.provider.configuration.api_key.get_secret_value()):
                safe = safe.replace(value, "[redacted]")
            probe.record["workers"][-1].update(phase="stopped", exit_code=process.returncode, diagnostics=safe)
            probe.persist()
            process = drain = None

        async def wait_cleaned(identity=None):
            for _ in range(160):
                async with fixture.pool.acquire() as connection:
                    pending = await connection.fetchval("""select count(*) from jarvis_sandbox_allocations
                        where owner_id=$1 and ($2::uuid is null or id=$2) and phase<>'cleaned'""",
                        fixture.owner, identity)
                if not pending:
                    return
                await asyncio.sleep(.25)
            raise TimeoutError("automatic_cleanup")

        try:
            await start(True)
            probe.check("separate_worker_advertises_sandbox", await registry.ready(fixture.owner, "frontier",
                        args.model, required_capabilities=("sandbox.execute",)))
            task = await service.create(fixture.owner, TaskCreate(request_id=uuid4(),
                title="Native worker sandbox acceptance", requested_outcome=REQUEST,
                knowledge_area_id=fixture.area, execution_environment="offline_sandbox"))
            probe.record["task_id"] = str(task.id)
            probe.persist()
            for _ in range(800):
                current = await tasks.get(fixture.owner, task.id)
                if current.status in {TaskStatus.SUCCEEDED, TaskStatus.PAUSED, TaskStatus.FAILED, TaskStatus.WAITING}:
                    break
                if process.returncode is not None:
                    raise RuntimeError("worker_exited_during_task")
                await asyncio.sleep(.25)
            async with fixture.pool.acquire() as connection:
                result = await connection.fetchval("select content from jarvis_context_payloads where id=$1",
                                                  current.result_payload_id) if current.result_payload_id else None
                commands = await connection.fetch("""select c.phase,p.content from jarvis_sandbox_commands c
                    left join jarvis_context_payloads p on p.id=c.result_payload_id where c.task_id=$1""", task.id)
                grant_count = await connection.fetchval("select count(*) from jarvis_task_grants where task_id=$1", task.id)
            probe.record["task_result"] = {"state":current.status.value, "reason":current.reason,
                                          "result":json_object(result) if result else None}
            probe.persist()
            probe.check("automatic_source_bound_computation_grant", grant_count==1)
            probe.check("real_worker_hermes_task_succeeded", current.status==TaskStatus.SUCCEEDED)
            receipts = [json_object(row["content"]) for row in commands if row["content"] is not None]
            probe.check("two_successful_native_commands", len(receipts)>=2 and all(
                row["exit_code"]==0 and row["started"] and not row["timed_out"] for row in receipts))
            probe.check("native_readback_contains_totals", any("alpha" in row["stdout"] and "beta" in row["stdout"]
                                                            and "10" in row["stdout"] for row in receipts))
            await wait_cleaned()
            probe.check("automatic_cleanup_after_task", True)
            await stop()
            probe.check("worker_readiness_withdrawn", not await registry.ready(fixture.owner, "frontier", args.model))

            # Create a paused, owned resource while no worker is running. The
            # restart disables new execution but must still clean this resource.
            setup = TaskService(tasks, providers, sandbox_configuration=configuration)
            pending = await setup.create(fixture.owner, TaskCreate(request_id=uuid4(), title="Synthetic orphan cleanup",
                requested_outcome="Synthetic offline cleanup fixture.", execution_environment="offline_sandbox"))
            claim = await tasks.claim_next("native-fixture-setup", ("frontier",), owner_id=fixture.owner,
                                          worker_capabilities=("sandbox.execute",))
            async with fixture.pool.acquire() as connection:
                grant = await connection.fetchval("select grant_id from jarvis_task_grants where task_id=$1", pending.id)
            row = await allocations.reserve(claim, grant)
            token = await allocations.begin_creation(claim, row["id"])
            scope = allocations.scope(row)
            probe.scopes.append((scope, policy))
            item = {"scope":scope.model_dump(mode="json"),"policy":policy.model_dump(mode="json"),"phase":"planned"}
            probe.record["resources"].append(item)
            probe.persist()
            handle = await probe.provider.create(scope, policy)
            item.update(sandbox_id=handle.sandbox_id, phase="created")
            probe.persist()
            await allocations.record_created(fixture.owner, row["id"], token, handle)
            await probe.provider.prepare(handle)
            await allocations.ready(claim, row["id"])
            await probe.provider.pause(handle)
            await tasks.finish(claim, WorkerOutcome(TaskStatus.PAUSED, {"reason":"synthetic orphan for cleanup"}))
            await start(False)
            probe.check("disabled_worker_does_not_advertise_execution",
                not await registry.ready(fixture.owner, "frontier", args.model, required_capabilities=("sandbox.execute",)))
            try:
                await service.create(fixture.owner, TaskCreate(request_id=uuid4(), title="Should not be admitted",
                    requested_outcome="No worker has computation enabled.", execution_environment="offline_sandbox"))
            except TaskError as error:
                probe.check("disabled_worker_rejects_new_sandbox_intake", error.code=="task_worker_not_ready")
            else:
                raise AssertionError("disabled_sandbox_was_admitted")
            await wait_cleaned(row["id"])
            probe.check("cleanup_only_restart_removed_paused_resource", True)
        finally:
            try:
                await stop()
            finally:
                key_file.unlink(missing_ok=True)
                try:
                    await collect_resources(probe, fixture, allocations)
                finally:
                    if probe.scopes:
                        await probe.cleanup()
        probe.check("worker_secret_removed", not key_file.exists())
        probe.check("private_native_homes_removed", not any((root/"native").glob("*/home")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--hermes-python", type=Path, required=True)
    parser.add_argument("--sdk-path")
    parser.add_argument("--model", default="gpt-5.6-sol")
    args = parser.parse_args()
    probe = None
    try:
        provider = E2BProvider(E2BConfiguration(api_key=dotenv_values(args.credential_file).get("E2B_API_KEY")))
        probe = Probe(provider, Path("/tmp"))
        print(json.dumps({"manifest":str(probe.path)}), flush=True)
        asyncio.run(exercise(probe, args, JarvisSettings()))
    except Exception as error:
        if probe is not None:
            probe.record["failure"] = {"type":type(error).__name__, "code":getattr(error,"code",None)}
            probe.persist()
        print(json.dumps({"passed":False, "error":getattr(error,"code",type(error).__name__)}))
        return 1
    print(json.dumps({"passed":True,"checks":len(probe.record["checks"]),
                      "cleanup":probe.record.get("cleanup"),"manifest":str(probe.path)}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
