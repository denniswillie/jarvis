"""Real E2B plus isolated task-ledger cleanup; no live Jarvis task or model call."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values

from app.sandbox.cleanup import SandboxCleaner, SandboxCleanupRepository
from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.models import E2BConfiguration, SandboxError, SandboxPolicy
from app.settings import JarvisSettings
from app.tasks.models import TaskMutation
from evals.second_brain.e2b_probe import Probe
from evals.second_brain.sandbox_fixture import sandbox_task_fixture


async def exercise(probe, dsn, manifest_dir):
    for case in ("lost_receipt_paused","cancelled_running"):
        async with sandbox_task_fixture(dsn,manifest_dir,test_name="native-e2b-cleanup:"+case,
            policy=SandboxPolicy(lifetime_seconds=180,command_seconds=60)) as fixture:
            allocation = await fixture.allocations.reserve(fixture.claim,fixture.grant)
            token = await fixture.allocations.begin_creation(fixture.claim,allocation["id"])
            scope,policy = fixture.allocations.scope(allocation),fixture.policy
            probe.scopes.append((scope,policy))
            resource = {"scope":scope.model_dump(mode="json"),"policy":policy.model_dump(mode="json"),
                        "phase":"planned","case":case,"database_manifest":str(fixture.source.fixture.path)}
            probe.record["resources"].append(resource)
            probe.record["status"] = "running"
            probe.persist()
            try:
                handle = await probe.provider.create(scope,policy)
                resource.update(phase="created",sandbox_id=handle.sandbox_id,template_id=handle.template_id)
                probe.persist()
                await probe.provider.prepare(handle)
                if case=="lost_receipt_paused":
                    paused = await probe.provider.pause(handle)
                    probe.check(case+":native_paused",paused.state=="paused")
                    # Deliberately omit the DB receipt; the cleanup worker only
                    # has its durable pre-create allocation scope.
                    await fixture.allocations.creation_uncertain(fixture.source.owner,allocation["id"],token)
                else:
                    await fixture.allocations.record_created(fixture.source.owner,allocation["id"],token,handle)
                    await fixture.allocations.ready(fixture.claim,allocation["id"])
                    command = await probe.provider.start(handle,"sleep 60 & wait",operation_id=uuid4())
                    resource["native_pid"] = command.pid
                    await command.disconnect()
                    probe.check(case+":native_process_started",
                        command.pid in [process.pid for process in await probe.provider.processes(handle)])
                await fixture.tasks.mutate(fixture.source.owner,fixture.task.id,"cancel",
                    TaskMutation(request_id=uuid4(),expected_revision=1))
                cleaner = SandboxCleaner(SandboxCleanupRepository(fixture.allocations),probe.provider)
                result = await cleaner.run_once(fixture.source.owner,"native-cleanup-probe")
                probe.check(case+":durable_cleanup_complete",result is not None and result["state"]=="cleaned")
                async with fixture.source.pool.acquire() as connection:
                    row = await connection.fetchrow("select * from jarvis_sandbox_allocations where id=$1",allocation["id"])
                probe.check(case+":discovered_identity_retained",handle.sandbox_id in row["cleanup_observed_ids"])
                if case=="lost_receipt_paused":
                    probe.check(case+":no_primary_receipt_was_needed",row["sandbox_id"] is None)
                try:
                    await probe.provider.info(handle.sandbox_id,scope,policy)
                except SandboxError as error:
                    probe.check(case+":native_vm_absent",error.code=="sandbox_not_found")
                else:
                    raise AssertionError(case+":native_vm_still_present")
                resource["phase"] = "cleaned_by_task_worker"
                probe.persist()
            except BaseException as error:
                probe.record["failure"] = {"type":type(error).__name__,"code":getattr(error,"code",None),"case":case}
                probe.persist()
                raise
            finally:
                # Native cleanup precedes dropping its disposable SQL schema.
                # Private allocation manifests remain available after an error.
                await probe.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-db",action="store_true",required=True)
    parser.add_argument("--credential-file",type=Path,required=True)
    parser.add_argument("--manifest-dir",type=Path,required=True)
    parser.add_argument("--directory",type=Path,default=Path("/tmp"))
    args = parser.parse_args()
    try:
        provider = E2BProvider(E2BConfiguration(api_key=dotenv_values(args.credential_file).get("E2B_API_KEY")))
        probe = Probe(provider,args.directory)
        print(json.dumps({"manifest":str(probe.path)}),flush=True)
        asyncio.run(exercise(probe,JarvisSettings().SUPABASE_DB_URL,args.manifest_dir))
    except Exception as error:
        print(json.dumps({"passed":False,"error":getattr(error,"code",type(error).__name__)}))
        return 1
    print(json.dumps({"passed":True,"checks":len(probe.record["checks"]),"cleanup":probe.record["cleanup"],
                      "manifest":str(probe.path)}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
