"""Isolated native frontier deployment acceptance; never enables the live API."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from app.images import ImageReference
from app.runtime.feature_config import DEFAULT_FRONTIER_MODEL, TaskLimits
from app.settings import JarvisSettings
from app.tasks.catalog import ConfiguredTaskProvider, TaskWorkerRegistry
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from evals.second_brain.source_fixture import source_fixture
from evals.second_brain.stored_image_fixture import stored_image_fixture


async def run(root, image, *, stored_images=False):
    if not root.is_absolute() or not root.name.startswith("jarvis-s04-frontier-deployment-"):
        raise ValueError("Use a fresh absolute jarvis-s04-frontier-deployment- fixture directory")
    root.mkdir(mode=0o700)
    settings = JarvisSettings()
    project = "jarvis-s04-frontier-"+uuid4().hex[:16]
    model = os.environ.get("JARVIS_S04_MODEL", DEFAULT_FRONTIER_MODEL)
    secrets = root/"secrets"
    secrets.mkdir(mode=0o700)
    state = {"format":"jarvis.frontier_deployment_fixture.v1","project":project,"status":"starting",
             "secrets_removed":False,"image_id":None,"stored_images":stored_images}
    receipt = root/"deployment-resources.json"
    def record():
        receipt.write_text(json.dumps(state,indent=2)+"\n")
    record()
    repo = Path(__file__).resolve().parents[3]
    environment = os.environ | {
        "JARVIS_FRONTIER_SECRET_DIR":str(secrets),"JARVIS_FRONTIER_HOST_ID":project,
        "JARVIS_FRONTIER_MODEL":model,"JARVIS_FRONTIER_REASONING_EFFORT":"low",
    }
    override = root/"override.json"
    override.write_text(json.dumps({"services":{"jarvis-frontier":{
        "image":image,"environment":{"SECOND_BRAIN_WORKER_HEARTBEAT_SECONDS":"1",
                                     "SECOND_BRAIN_POLL_SECONDS":"0.2"},
    }}}))
    compose = ["docker","compose","-p",project,"-f",str(repo/"deployment/docker-compose.frontier.yml"),
               *([ "-f",str(repo/"deployment/docker-compose.frontier-images.yml")] if stored_images else []),
               "-f",str(override)]
    def command(arguments, *, timeout=90):
        result = subprocess.run(arguments,env=environment,cwd=repo,text=True,capture_output=True,timeout=timeout)
        if result.returncode:
            raise RuntimeError("Owned deployment command failed with exit code "+str(result.returncode))
        return result.stdout
    dsn = settings.SUPABASE_DB_URL
    attempted, cleaned = False, False
    try:
        async with source_fixture(dsn,root,test_name="native frontier container deployment") as fixture, AsyncExitStack() as resources:
            environment["JARVIS_FRONTIER_OWNER_ID"] = fixture.owner
            async with fixture.pool.acquire() as connection:
                schema = await connection.fetchval("select current_schema()")
            parsed = urlsplit(dsn)
            query = dict(parse_qsl(parsed.query))
            query["search_path"] = schema+",pg_catalog,public,extensions"
            fixture_dsn = urlunsplit((parsed.scheme,parsed.netloc,parsed.path,urlencode(query),parsed.fragment))
            secret_values = [("database-url",fixture_dsn),("openai-api-key",settings.OPENAI_API_KEY)]
            if stored_images:
                environment.update(JARVIS_FRONTIER_SUPABASE_URL=settings.SUPABASE_URL,
                                   JARVIS_FRONTIER_FRAME_BUCKET=settings.SUPABASE_STORAGE_BUCKET)
                secret_values.append(("supabase-storage-key",settings.SUPABASE_SERVICE_ROLE_KEY))
            for filename,value in secret_values:
                path = secrets/filename
                path.write_text(value)
                # Only this 0700 parent is traversable by the operator; Compose
                # binds each file separately read-only for the non-root worker.
                path.chmod(0o444)
            tasks = TaskRepository(fixture.pool,TaskLimits(request_output_tokens=4096))
            registry = TaskWorkerRegistry(fixture.pool)
            session, source, label = None, None, None
            if stored_images:
                from PIL import Image, ImageDraw, ImageFont
                label = "NIMBUS "+str(uuid4().int%90000+10000)
                image_data = Image.new("RGB",(800,240),"white")
                draw = ImageDraw.Draw(image_data)
                draw.text((24,35),label,font=ImageFont.load_default(size=38),fill="black")
                draw.text((24,115),"mid = low + high // 2",font=ImageFont.load_default(size=34),fill="black")
                output = BytesIO()
                image_data.save(output,format="JPEG",quality=95)
                async with fixture.pool.acquire() as connection:
                    session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id",
                                                         fixture.owner)
                frames = await resources.enter_async_context(stored_image_fixture(
                    fixture,root,settings,session_id=session,images=[output.getvalue()]))
                frame = frames[0]
                reference = ImageReference(frame_id=frame.id,owner_id=fixture.owner,session_id=session,
                    source="historical",captured_at=frame.captured_at,content_type=frame.content_type,
                    width=frame.width,height=frame.height,raw_image_expires_at=frame.raw_image_expires_at,
                    visual_session_id=frame.visual_session_id,frame_sequence=frame.frame_sequence,
                    server_received_at=frame.server_received_at)
                source = await ContextRepository(tasks).append(fixture.owner,ContextIngress(
                    session_id=session,source_namespace="fixture-container-image",source_event_id=str(uuid4()),
                    receipt_id=str(uuid4()),kind="observation",trust="sensor",
                    payload={"images":[reference.to_metadata()]}))
                state["image_source_id"],state["image_frame_id"] = str(source.id),str(frame.id)
                record()
            try:
                attempted = True
                command(compose+["up","-d","--no-build","jarvis-frontier"])
                state["status"] = "running"
                record()
                for _ in range(120):
                    if await registry.ready(fixture.owner,"frontier",model): break
                    await asyncio.sleep(.5)
                if not await registry.ready(fixture.owner,"frontier",model):
                    raise AssertionError("Container did not advertise native readiness")
                service = TaskService(tasks,{"frontier":ConfiguredTaskProvider(model)},registry=registry)
                task = await service.create(fixture.owner,TaskCreate(
                    request_id=uuid4(),title="Synthetic container code review",session_id=session,
                    requested_outcome=("Read the original stored image referenced by the session observation. "
                        "Quote its random label, identify the visible Python midpoint error, and give the corrected "
                        "assignment exactly. Cite the image source. Do not guess from metadata."
                        if stored_images else "In Python binary search, explain why mid = low + high // 2 is wrong. "
                        "Give the corrected assignment exactly and a short explanation."),
                ))
                for _ in range(240):
                    current = await tasks.get(fixture.owner,task.id)
                    if current.status in (TaskStatus.SUCCEEDED,TaskStatus.PAUSED,TaskStatus.WAITING,TaskStatus.FAILED): break
                    await asyncio.sleep(.5)
                state["task_id"],state["task_status"],state["task_reason"] = str(task.id),current.status,current.reason
                if current.status!=TaskStatus.SUCCEEDED:
                    raise AssertionError("Native container did not complete the delegated task")
                async with fixture.pool.acquire() as connection:
                    result = json.loads(await connection.fetchval(
                        "select content::text from jarvis_context_payloads where owner_id=$1 and id=$2",
                        fixture.owner,current.result_payload_id))
                if "mid = (low + high) // 2" not in result["answer"]:
                    raise AssertionError("Native answer did not contain the correct code")
                if stored_images:
                    if label not in result["answer"] or str(source.id) not in {e.get("event_id") for e in result["evidence"]}:
                        raise AssertionError("Container did not read and cite the original image label")
                    if not any(r["name"]=="read_task_context_image" and r["status"]=="success" for r in result["tool_receipts"]):
                        raise AssertionError("Container did not use its original-image tool")
                state["answer"] = result["answer"]
                health = json.loads(command(compose+["exec","-T","jarvis-frontier","python","-m","app.tasks.worker","check"]))
                if health.get("ready") is not True: raise AssertionError("Worker health command failed")
                identity = command(compose+["ps","-q","jarvis-frontier"]).strip()
                inspected = json.loads(command(["docker","inspect","--format",
                    '{"image":{{json .Image}},"user":{{json .Config.User}},"read_only":{{json .HostConfig.ReadonlyRootfs}},"ports":{{json .HostConfig.PortBindings}}}',
                    identity]))
                if inspected["user"]!="10001:10001" or not inspected["read_only"] or inspected["ports"]:
                    raise AssertionError("Deployment isolation differs from the declared role")
                state["image_id"],state["isolation"] = inspected["image"],inspected
                native = json.loads(command(compose+["exec","-T","jarvis-frontier","python","-c",
                    "import json,pathlib; r=pathlib.Path('/var/lib/jarvis/hermes'); "
                    "print(json.dumps([json.loads(p.read_text())|{'home_absent':not (p.parent/'home').exists()} "
                    "for p in r.glob('*/run.json')]))"]))
                if not native or any(row["phase"]!="exited" or not row["home_absent"] for row in native):
                    raise AssertionError("Native homes/lifetimes were not cleaned")
                state["native_runs"] = native
                command(compose+["stop","-t","20","jarvis-frontier"])
                if await registry.ready(fixture.owner,"frontier",model):
                    raise AssertionError("Stopped worker still advertises readiness")
                state["accepted"] = True
            finally:
                try:
                    logs = subprocess.run(compose+["logs","--no-color","--tail","80"],env=environment,cwd=repo,
                                          text=True,capture_output=True,timeout=20).stdout
                    for secret in (dsn,*(value for _,value in secret_values)):
                        if secret: logs = logs.replace(secret,"[redacted]")
                    (root/"worker.log").write_text(logs)
                except Exception:
                    state["log_capture"] = "failed"
                finally:
                    command(compose+["down","--volumes","--remove-orphans"],timeout=60)
                    cleaned = True
    finally:
        # This root is newly created and contains only the manifested fixture secrets.
        shutil.rmtree(secrets)
        state["secrets_removed"] = True
        state["status"] = "cleaned" if cleaned or not attempted else "cleanup_failed"
        record()
    print(json.dumps(state))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir",type=Path,required=True)
    parser.add_argument("--image",default="jarvis-frontier:s04-candidate")
    parser.add_argument("--stored-images",action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.artifact_dir,args.image,stored_images=args.stored_images))


if __name__=="__main__":
    main()
