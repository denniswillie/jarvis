"""Kill a real storage producer after upload, then reconcile without re-uploading."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
from uuid import UUID, uuid4

import asyncpg

from app.files.generated import ArtifactPolicy, ArtifactRequest
from app.files.storage import JarvisFileStorageClient
from app.files.tools import FileToolModule
from app.knowledge.files import SourceFileRepository
from app.knowledge.sources import SourceRepository
from app.migrations.runner import quote_schema
from app.runtime.feature_config import TaskLimits
from app.settings import JarvisSettings
from app.storage.supabase_config import get_supabase_storage_config
from app.tasks.file_recovery import TaskFileRecovery
from app.tasks.generated_artifacts import GeneratedArtifactPublisher
from app.tasks.memory_evidence import TaskMemoryEvidence
from app.tasks.models import Reconciliation, TaskMutation, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.tool_bridge import TaskToolBridge
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.e2b_probe import Probe
from evals.second_brain.sandbox_artifact_probe import TrackedStorage
from evals.second_brain.sandbox_fixture import sandbox_task_fixture

CONTENT = b'{"synthetic":true,"total_hours":10}\n'
REQUEST = ArtifactRequest(operation_key="crash-report", filename="crash-report.json",
                          format="json", description="Synthetic process-crash report.")


async def publisher(tasks, sources, storage, owner, task_id, parent):
    async with tasks.pool.acquire() as connection:
        task = await tasks.load(connection, owner, task_id)
        attempt = await connection.fetchrow("select * from jarvis_task_attempts where owner_id=$1 and id=$2",
                                           owner, task.current_attempt_id)
    context = TaskRunner(tasks, {}).context(tasks._claim(task, attempt))
    await WorkerContextFactory(tasks.pool)(context)
    evidence = TaskMemoryEvidence(context, sources, area_id=task.knowledge_area_id)
    files = SourceFileRepository(sources, storage, owner_id=owner, area_id=task.knowledge_area_id, task_evidence=evidence)
    bridge = await TaskToolBridge.from_modules(context, (FileToolModule(files, storage),),
        memory_evidence=evidence, receipt_revalidator=files.revalidate_receipt)
    return GeneratedArtifactPublisher(bridge, files, parent)


async def child():
    request = json.loads(sys.stdin.readline())
    settings = JarvisSettings()
    schema = request["schema"]
    if not schema.startswith("jarvis_sb_eval_") or request["owner"] == settings.JARVIS_OWNER_ID:
        raise ValueError("Only a synthetic isolated fixture is permitted")
    async def setup(connection):
        await connection.execute(f"set search_path to {quote_schema(schema)},pg_catalog,public,extensions")
    pool = await asyncpg.create_pool(settings.SUPABASE_DB_URL, min_size=1, max_size=3, setup=setup)
    probe = object.__new__(Probe)
    probe.path = Path(request["manifest"])
    probe.record = json.loads(probe.path.read_text())
    class CrashStorage(TrackedStorage):
        async def upload_bytes(self, **arguments):
            await super().upload_bytes(**arguments)
            os.kill(os.getpid(), signal.SIGKILL)
            raise AssertionError("SIGKILL returned")
    storage = CrashStorage(JarvisFileStorageClient(get_supabase_storage_config(settings)), probe)
    try:
        producer = await publisher(TaskRepository(pool, TaskLimits(lease_seconds=300)), SourceRepository(pool),
            storage, request["owner"], UUID(request["task_id"]), UUID(request["parent"]))
        await producer.publish(REQUEST, CONTENT, provider_call_id="fixture-native-killed-upload")
    finally:
        await storage.close()
        await pool.close()


async def exercise(probe, manifest_dir):
    settings = JarvisSettings()
    async with sandbox_task_fixture(settings.SUPABASE_DB_URL, manifest_dir,
                                    test_name="native-artifact-process-crash", with_session=True) as f:
        probe.record.update(database_manifest=str(f.source.fixture.path), fixture_owner=f.source.owner)
        probe.persist()
        async with f.source.pool.acquire() as connection:
            schema = await connection.fetchval("select current_schema()")
            scope = await f.tasks.payload(connection, f.source.owner, ArtifactPolicy().model_dump(mode="json"))
            parent = await connection.fetchval("""insert into jarvis_operation_grants
                (owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                values($1,'files.publish_generated',$2,$3,1,1,$4,clock_timestamp()+interval '1 hour') returning id""",
                f.source.owner, str(uuid4()), f.event, scope)
            await connection.execute("insert into jarvis_task_grants(owner_id,task_id,grant_id) values($1,$2,$3)",
                                     f.source.owner, f.task.id, parent)
        storage = TrackedStorage(JarvisFileStorageClient(get_supabase_storage_config(settings)), probe)
        files = SourceFileRepository(f.source.sources, storage, owner_id=f.source.owner, area_id=f.source.area)
        process = None
        try:
            process = await asyncio.create_subprocess_exec(sys.executable, "-m", "evals.second_brain.artifact_crash_probe", "--child",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            data = json.dumps({"schema": schema, "owner": f.source.owner, "task_id": str(f.task.id),
                               "parent": str(parent), "manifest": str(probe.path)}).encode()
            stdout, stderr = await asyncio.wait_for(process.communicate(data), 90)
            probe.record = json.loads(probe.path.read_text())
            probe.record["producer_exit"] = process.returncode
            probe.persist()
            probe.check("real_producer_sigkill", process.returncode == -signal.SIGKILL,
                        stderr_tail=stderr.decode(errors="replace")[-1000:])
            async with f.source.pool.acquire() as connection:
                action = await connection.fetchrow("select * from jarvis_task_actions where owner_id=$1", f.source.owner)
                phase = await connection.fetchval("select phase from jarvis_file_publications where owner_id=$1", f.source.owner)
                await connection.execute("update jarvis_task_attempts set lease_until=clock_timestamp() where id=$1", f.claim.attempt_id)
            probe.check("killed_after_upload_before_adoption", action["status"] == "submitting" and phase == "uploading"
                        and len(probe.record["storage_objects"]) == 1
                        and probe.record["storage_objects"][0]["phase"] == "uploaded")
            claim = await f.tasks.claim_recovery("fixture-post-crash", owner_id=f.source.owner)
            await f.tasks.resolve_recovery(claim, Reconciliation("not_started"))
            waiting = await f.tasks.get(f.source.owner, f.task.id)
            probe.check("unknown_effect_blocks_restart", waiting.status == TaskStatus.UNKNOWN and
                await f.tasks.claim_next("fixture-no-blind-retry", ("frontier",),
                    worker_capabilities=("sandbox.execute", "files.publish_generated")) is None)
            recovery = TaskFileRecovery(f.source.sources, f.tasks, storage,
                                       owner_id=f.source.owner, area_id=f.source.area)
            result = await recovery.reconcile_pending(owner_id=f.source.owner)
            probe.check("stored_original_reconciled", [item["phase"] for item in result] == ["available"], outcomes=result)
            paused = await f.tasks.get(f.source.owner, f.task.id)
            probe.check("barrier_settled_without_automatic_restart", paused.status == TaskStatus.PAUSED)
            await f.tasks.mutate(f.source.owner, f.task.id, "resume", TaskMutation(request_id=uuid4(),
                expected_revision=paused.revision, message="Continue the synthetic report."))
            resumed = await f.tasks.claim_next("fixture-resumed", ("frontier",),
                worker_capabilities=("sandbox.execute", "files.publish_generated"))
            producer = await publisher(f.tasks, f.source.sources, storage, f.source.owner, f.task.id, parent)
            receipt = await producer.publish(REQUEST, provider_call_id="fixture-post-crash-replay")
            original = await files.get_file(owner_id=f.source.owner, file_id=UUID(receipt["file"]["id"]))
            probe.check("fresh_attempt_reuses_original_receipt", resumed.attempt_id != f.claim.attempt_id
                and receipt["status"] == "success" and receipt["task_operation"]["replayed"]
                and original.metadata["task_artifact"]["attempt_id"] == str(f.claim.attempt_id))
            content = await files.read_content(owner_id=f.source.owner, file_id=UUID(receipt["file"]["id"]))
            probe.check("exact_stored_bytes", content == CONTENT)
            async with f.source.pool.acquire() as connection:
                count = await connection.fetchval("select count(*) from jarvis_task_actions where owner_id=$1", f.source.owner)
                unsettled = await connection.fetchval("""select count(*) from jarvis_task_reservations
                    where owner_id=$1 and status in ('submitted','unknown')""", f.source.owner)
            probe.check("one_upload_action_and_settled_budget", count == 1
                        and len(probe.record["storage_objects"]) == 1 and unsettled == 0)
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            probe.record = json.loads(probe.path.read_text())
            try:
                await storage.cleanup(files)
            finally:
                await storage.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--manifest-dir", type=Path)
    args = parser.parse_args()
    if args.child:
        asyncio.run(child())
        return 0
    if not args.live or args.manifest_dir is None:
        parser.error("Use --live and --manifest-dir for the isolated native test")
    probe = Probe(None, Path("/tmp"))
    probe.record["format"] = "jarvis.artifact-crash.probe.v1"
    probe.persist()
    print(json.dumps({"manifest": str(probe.path)}), flush=True)
    try:
        asyncio.run(exercise(probe, args.manifest_dir))
    except Exception as error:
        probe.record["failure"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
        probe.persist()
        print(json.dumps({"passed": False, "manifest": str(probe.path), "error": type(error).__name__}))
        return 1
    print(json.dumps({"passed": True, "checks": probe.record["checks"],
                      "storage_cleanup": probe.record.get("storage_cleanup"), "manifest": str(probe.path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
