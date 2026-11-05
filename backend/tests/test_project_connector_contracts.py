"""S10 cross-workstream contract gates, using isolated real schema and Mac processes."""
import asyncio
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from uuid import UUID, uuid4

import pytest

from app.knowledge.models import SourceError
from app.knowledge.runtime import KnowledgeRuntime
from app.projects.models import Completion, Inspection, Poll
from app.projects.service import ProjectService
from app.projects.tools import ProjectToolModule
from app.runtime.feature_config import DEFAULT_FRONTIER_MODEL, KnowledgeConfiguration, TaskLimits
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tasks.memory_evidence import TaskMemoryEvidence
from app.tasks.models import TaskCreate, TaskError, TaskStatus, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.worker import WorkerContextFactory
from app.tools.module import ToolBuildContext
from app.tools.provenance import ToolResultCapture, restore_source_result
from evals.second_brain.source_fixture import source_fixture
from test_project_connector_transport import REAL, cli, completion, enroll, running_backend


pytestmark = pytest.mark.skipif(not REAL, reason="Explicit isolated project contract acceptance required")


async def setup_session(fixture):
    tasks = TaskRepository(fixture.pool, TaskLimits(lease_seconds=60))
    # No provider request is needed for source-capture/transaction gates. The actual
    # native provider and semantic discovery are exercised by the separate native test.
    knowledge = await KnowledgeRuntime.create(pool=fixture.pool, owner_id=fixture.owner, tasks=tasks,
        configuration=KnowledgeConfiguration(base_url="http://127.0.0.1:1", api_key="synthetic-unused",
                                             area_id=fixture.area))
    async with fixture.pool.acquire() as db:
        session = await db.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
    return tasks, knowledge, session


async def accepted_read(service, fixture, registered, offer, *, text="value = 42\n"):
    pending = await service.submit(fixture.owner, fixture.area, Inspection(
        root_id=registered["roots"][0]["root_id"], request={"operation": "read", "path": "module.py"}))
    work = (await service.poll(registered["token"], offer))["request"]
    assert work["request_id"] == pending["request_id"]
    await service.complete(registered["token"], completion(work, text=text))
    return UUID(pending["request_id"])


@pytest.mark.parametrize("mutation", ["correction", "forget", "revoke"])
def test_foreground_result_capture_and_replay_obey_originals(tmp_path, mutation):
    async def exercise():
        async with source_fixture(JarvisSettings().SUPABASE_DB_URL, tmp_path / "fixtures",
                                  test_name="S10 foreground project evidence " + mutation) as fixture:
            tasks, knowledge, session = await setup_session(fixture)
            capture = SessionContextCapture(knowledge.context_repository, owner_id=fixture.owner,
                                            session_id=session, provider="openai", audio_session_id="s95-source")
            try:
                service = knowledge.projects
                registered, offer = await enroll(service, fixture)
                await service.poll(registered["token"], offer)
                request_id = await accepted_read(service, fixture, registered, offer)
                module = ProjectToolModule(service, fixture.owner, fixture.area,
                    selections=knowledge.area_selections, access_check=knowledge.context_for)
                tools = {tool.name: tool for tool in module.build_tools(
                    ToolBuildContext(settings=JarvisSettings(), owner_id=fixture.owner, session_id=session))}
                reader = tools["get_project_result"]
                result = await reader.handler({"request_id": str(request_id)})
                assert result["status"] == "success" and result["project_status"] == "complete"
                assert result["inspection"]["result"]["content"] == "value = 42\n"
                bound = await reader.result_capture(result, ToolResultCapture(
                    capture=capture, call_id="project-result", tool_name=reader.name))
                original_record = await capture.repository.get(fixture.owner, bound.event_id, session_id=session)
                assert original_record.content["result"]["inspection"]["result"]["content"] == "value = 42\n"
                replayed = await restore_source_result(capture, owner_id=fixture.owner, session_id=session,
                    call_id="project-result", name=reader.name, saved=bound.audit_receipt())
                assert replayed == bound
                # An existing result cannot be rebound to a different live voice session.
                async with fixture.pool.acquire() as db:
                    other_session = await db.fetchval(
                        "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
                wrong = SessionContextCapture(knowledge.context_repository, owner_id=fixture.owner,
                                              session_id=other_session, provider="openai", audio_session_id="other")
                try:
                    with pytest.raises(SourceError):
                        await reader.result_capture(result, ToolResultCapture(
                            capture=wrong, call_id="wrong-session", tool_name=reader.name))
                finally:
                    await wrong.close()
                source = result["sources"][0]
                if mutation == "correction":
                    await accepted_read(service, fixture, registered, offer, text="value = 43\n")
                elif mutation == "forget":
                    await fixture.sources.delete_source(fixture.owner, fixture.area, UUID(source["source_id"]),
                                                        expected_revision=source["revision"])
                else:
                    await service.revoke(fixture.owner, UUID(registered["connector_id"]))
                removed = await capture.repository.get(fixture.owner, bound.event_id, session_id=session)
                assert removed.availability != "available" and removed.content is None
                with pytest.raises(TaskError):
                    await restore_source_result(capture, owner_id=fixture.owner, session_id=session,
                        call_id="project-result", name=reader.name, saved=bound.audit_receipt())
                with pytest.raises(SourceError):
                    await reader.result_capture(result, ToolResultCapture(
                        capture=capture, call_id="late-copy", tool_name=reader.name))
                async with fixture.pool.acquire() as db:
                    assert await db.fetchval("""select count(*) from jarvis_context_payloads
                        where owner_id=$1 and availability='available'
                            and content->>'provider_call_id'='late-copy'""", fixture.owner) == 0
            finally:
                await capture.close()
                await knowledge.close()
    asyncio.run(exercise())


def test_worker_cannot_release_a_result_after_project_revocation(tmp_path):
    async def exercise():
        async with source_fixture(JarvisSettings().SUPABASE_DB_URL, tmp_path / "fixtures",
                                  test_name="S10 worker project release fence") as fixture:
            tasks, knowledge, session = await setup_session(fixture)
            try:
                service = knowledge.projects
                registered, offer = await enroll(service, fixture)
                await service.poll(registered["token"], offer)
                request_id = await accepted_read(service, fixture, registered, offer)
                task = await tasks.create(fixture.owner, TaskCreate(
                    request_id=uuid4(), title="Synthetic source-backed task",
                    requested_outcome="Read the approved project.", session_id=session,
                    knowledge_area_id=fixture.area), model=DEFAULT_FRONTIER_MODEL)
                claim = await tasks.claim_next("s95-project-fence", ("frontier",))
                assert claim is not None and claim.task.id == task.id
                context = TaskRunner(tasks, {}).context(claim)
                await WorkerContextFactory(fixture.pool, knowledge=knowledge)(context)
                evidence = TaskMemoryEvidence(context, fixture.sources, area_id=fixture.area,
                                              access_check=knowledge.context_for)
                module = ProjectToolModule(service, fixture.owner, fixture.area,
                                           access_check=knowledge.context_for, task_evidence=evidence)
                reader = next(tool for tool in module.build_tools(
                    ToolBuildContext(settings=JarvisSettings(), owner_id=fixture.owner, session_id=session))
                    if tool.name == "get_project_result")
                result = await reader.handler({"request_id": str(request_id)})
                assert result["status"] == "success"
                assert all(item["task_evidence"]["source_id"] == item["source_id"] for item in result["sources"])
                async with fixture.pool.acquire() as db:
                    assert await db.fetchval("select count(*) from jarvis_task_memory_sources where owner_id=$1",
                                             fixture.owner) >= 2
                await service.revoke(fixture.owner, UUID(registered["connector_id"]))
                sentinel = "S95_RESULT_MUST_NOT_BE_RELEASED_" + uuid4().hex
                await tasks.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED,
                    {"answer": sentinel, "speech": [{"text": sentinel}]}))
                async with fixture.pool.acquire() as db:
                    row = await db.fetchrow("""select status,result_payload_id,reason
                        from jarvis_background_tasks where owner_id=$1 and id=$2""", fixture.owner, task.id)
                    assert row["status"] != "succeeded" and row["result_payload_id"] is None and row["reason"]
                    assert await db.fetchval("""select count(*) from jarvis_context_payloads
                        where owner_id=$1 and content::text like $2""", fixture.owner, "%" + sentinel + "%") == 0
            finally:
                await knowledge.close()
    asyncio.run(exercise())


def test_inflight_server_revocation_and_actual_credential_expiry(tmp_path):
    async def exercise():
        async with source_fixture(JarvisSettings().SUPABASE_DB_URL, tmp_path / "fixtures",
                                  test_name="S10 in-flight project authority") as fixture:
            service = ProjectService(fixture.sources)
            registered, offer = await enroll(service, fixture)
            await service.poll(registered["token"], offer)
            root_id = UUID(registered["roots"][0]["root_id"])
            pending = await service.submit(fixture.owner, fixture.area,
                Inspection(root_id=root_id, request={"operation": "read", "path": "module.py"}))
            work = (await service.poll(registered["token"], offer))["request"]
            await service.revoke(fixture.owner, UUID(registered["connector_id"]))
            with pytest.raises(SourceError):
                await service.complete(registered["token"], completion(work))
            with pytest.raises(SourceError):
                await service.result(fixture.owner, fixture.area, UUID(pending["request_id"]))
            async with fixture.pool.acquire() as db:
                assert await db.fetchval("select count(*) from jarvis_knowledge_sources where owner_id=$1",
                                         fixture.owner) == 0

            # Seed a valid short-lived synthetic credential, without disabling or
            # changing production expiry guards. Wait for the actual database clock.
            secret = secrets.token_urlsafe(32)
            async with fixture.pool.acquire() as db:
                row = await db.fetchrow("""insert into jarvis_project_connectors
                    (owner_id,area_id,label,token_digest,expires_at)
                    values($1,$2,'Expiring synthetic Mac',$3,clock_timestamp()+interval '2 seconds')
                    returning id,expires_at""", fixture.owner, fixture.area, sha256(secret.encode()).hexdigest())
            token = str(row["id"]) + "." + secret
            assert (await service.poll(token, Poll(roots=[])))["request"] is None
            await asyncio.sleep(max(0, (row["expires_at"] - datetime.now(UTC)).total_seconds()) + .1)
            with pytest.raises(SourceError) as expired:
                await service.poll(token, Poll(roots=[]))
            assert expired.value.code == "project_connector_unavailable"
    asyncio.run(exercise())


def test_connected_typescript_navigation_dirty_source_and_unsupported_language(tmp_path):
    async def exercise():
        node = os.environ.get("JARVIS_CONNECTOR_TEST_NODE")
        typescript = os.environ.get("JARVIS_CONNECTOR_TEST_TYPESCRIPT")
        if not node or not typescript:
            pytest.skip("Connected TypeScript acceptance requires explicit maintained runtime paths")
        async with source_fixture(JarvisSettings().SUPABASE_DB_URL, tmp_path / "fixtures",
                                  test_name="S10 connected TypeScript") as fixture:
            service = ProjectService(fixture.sources)
            project = tmp_path / "Prism"
            project.mkdir()
            initial = "export class Controller {\n  move(): number { return 47; }\n}\n"
            (project / "controller.ts").write_text(initial)
            (project / "main.ts").write_text('import { Controller } from "./controller";\nconst robot = new Controller();\nrobot.move();\n')
            (project / "example.rs").write_text("fn example() -> i32 { 9 }\n")
            git = ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgSign=false",
                   "-c", "user.name=Synthetic acceptance", "-c", "user.email=synthetic@example.invalid"]
            subprocess.run(git + ["init", str(project)], check=True, capture_output=True)
            subprocess.run(git + ["-C", str(project), "add", "."], check=True, capture_output=True)
            subprocess.run(git + ["-C", str(project), "commit", "-m", "Synthetic original"], check=True, capture_output=True)
            edited = initial.replace("47", "48")
            (project / "controller.ts").write_text(edited)
            bootstrap = "synthetic-s95-bootstrap-" + uuid4().hex
            registry, credentials = tmp_path / "roots.json", tmp_path / "connection.json"
            base = [os.environ.get("JARVIS_MAC_TEST_PYTHON", sys.executable),
                    "-m", "jarvis_project_connector", "--registry", str(registry)]
            environment = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "LANG": "en_US.UTF-8",
                           "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "mac_connector"),
                           "JARVIS_PROJECT_BOOTSTRAP_TOKEN": bootstrap}
            await cli(base, "enroll", "--root", str(project), env=environment)
            async with running_backend(service, fixture.owner, bootstrap) as url:
                registered = await cli(base, "register", "--url", url, "--area-id", str(fixture.area),
                                       "--credentials", str(credentials), env=environment)
                serve = ["serve", "--credentials", str(credentials), "--node", node,
                         "--typescript", typescript, "--once"]
                await cli(base, *serve, env=environment)
                root_id = UUID(registered["roots"][0]["root_id"])
                async def query(read):
                    pending = await service.submit(fixture.owner, fixture.area, Inspection(root_id=root_id, request=read))
                    await cli(base, *serve, env=environment)
                    return await service.result(fixture.owner, fixture.area, UUID(pending["request_id"]))
                found = await query({"operation": "definition", "path": "main.ts", "line": 3, "column": 8})
                result = found["inspection"]["result"]
                assert not result["analysis_complete"]
                assert any(item["path"] == "controller.ts" and item["line"] == 2 for item in result["items"])
                assert result["snapshot"]["git"]["dirty"] is True
                controller = next(source for source in found["sources"] if source["source_type"] == "project_file")
                original = await fixture.sources.read(fixture.owner, fixture.area, UUID(controller["source_id"]))
                assert original.content["text"] == edited
                references = await query({"operation": "references", "path": "controller.ts", "line": 2, "column": 4})
                paths = {item["path"] for item in references["inspection"]["result"]["items"]}
                assert {"controller.ts", "main.ts"} <= paths
                changed = await query({"operation": "changes"})
                assert any(item["path"] == "controller.ts" for item in changed["inspection"]["result"]["items"])
                text = await query({"operation": "read", "path": "example.rs"})
                assert text["inspection"]["result"]["content"] == "fn example() -> i32 { 9 }\n"
                unsupported = await query({"operation": "definition", "path": "example.rs", "line": 1, "column": 5})
                assert unsupported["inspection"]["result"]["status"] == "unavailable"
                await service.revoke(fixture.owner, UUID(registered["connector_id"]))
            assert (project / "controller.ts").read_text() == edited
    asyncio.run(exercise())
