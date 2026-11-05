"""Versioned source/transport acceptance, with opt-in real isolated database/network."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import uvicorn

from app.auth.static_bearer import require_static_bearer
from app.errors import JarvisHTTPError
from app.knowledge.models import SourceError
from app.projects.models import Completion, Enrollment, Inspection, Original, Poll
from app.projects.router import router
from app.projects.service import ProjectService
from app.settings import JarvisSettings
from evals.second_brain.source_fixture import source_fixture


REAL = os.environ.get("JARVIS_TEST_REAL_PROJECTS") == "1"


@pytest.mark.parametrize("read_options", [
    {"operation": "exec", "command": "id"}, {"operation": "read", "path": "../secret"},
    {"operation": "read", "path": "/etc/passwd"}, {"operation": "read", "path": "a\\b"},
    {"operation": "files", "limit": 31}, {"operation": "read", "max_chars": True},
    {"operation": "read", "max_chars": 33000}, {"operation": "status", "shell": "pwd"},
])
def test_rejects_remote_shell_and_unbounded_requests(read_options):
    with pytest.raises(ValidationError):
        Inspection(root_id=uuid4(), request=read_options)


def test_original_requires_real_digest_and_canonical_path():
    with pytest.raises(ValidationError):
        Original(path="module.py", text="return 2", content_digest="0" * 64)
    with pytest.raises(ValidationError):
        Original(path="./module.py", text="", content_digest=sha256(b"").hexdigest())


def test_enrollment_rejects_duplicate_root_identity():
    root = {"local_root_id": uuid4(), "label": "Synthetic", "fingerprint": "a" * 64}
    with pytest.raises(ValidationError):
        Enrollment(area_id=uuid4(), label="Mac", roots=[root, root])


async def enroll(service, fixture):
    local = uuid4()
    result = await service.enroll(fixture.owner, Enrollment(area_id=fixture.area, label="Synthetic Mac",
        roots=[{"local_root_id": local, "label": "Project", "fingerprint": "a" * 64}]))
    offer = Poll(roots=[{"local_root_id": local, "label": "Project", "fingerprint": "a" * 64}])
    return result, offer


def completion(work, *, text="value = 42\n"):
    original = Original(path="module.py", text=text, content_digest=sha256(text.encode()).hexdigest())
    return Completion(request_id=work["request_id"], lease_id=work["lease_id"],
        captured_at=datetime.now(UTC), originals=[original],
        result={"status": "available", "root_id": work["local_root_id"], "operation": "read",
                "source": {"path": original.path, "content_digest": original.content_digest},
                "content": text, "git": {"available": False}})


@pytest.mark.skipif(not REAL, reason="Requires explicit isolated project database acceptance")
def test_durable_leases_publication_change_deletion_and_revocation(tmp_path):
    async def exercise():
        async with source_fixture(JarvisSettings().SUPABASE_DB_URL, tmp_path / "fixtures",
                                  test_name="S10 durable project transport") as fixture:
            service = ProjectService(fixture.sources)
            connection, offer = await enroll(service, fixture)
            token = connection["token"]
            root_id = UUID(connection["roots"][0]["root_id"])
            request = Inspection(root_id=root_id, request={"operation": "read", "path": "module.py"})
            with pytest.raises(SourceError, match="offline"):
                await service.submit(fixture.owner, fixture.area, request)
            await service.poll(token, offer)
            pending = await service.submit(fixture.owner, fixture.area, request)
            first = (await service.poll(token, offer))["request"]
            queued = await service.submit(fixture.owner, fixture.area, request)
            assert (await service.poll(token, offer))["request"] is None
            # Simulate a lost connection, not lost durable work.
            async with fixture.pool.acquire() as db:
                await db.execute("update jarvis_project_requests set lease_until=clock_timestamp()-interval '1 second' where id=$1",
                                 UUID(pending["request_id"]))
            replacement = (await ProjectService(fixture.sources).poll(token, offer))["request"]
            with pytest.raises(SourceError, match="lease"):
                await service.complete(token, completion(first))
            response = completion(replacement)
            assert (await service.complete(token, response))["accepted"]
            assert (await service.complete(token, response))["replayed"]
            next_work = (await service.poll(token, offer))["request"]
            assert next_work["request_id"] == queued["request_id"]
            await service.complete(token, completion(next_work))
            async with fixture.pool.acquire() as db:
                assert await db.fetchval("""select p.availability from jarvis_project_requests q
                    join jarvis_context_payloads p on p.owner_id=q.owner_id and p.id=q.request_payload_id
                    where q.id=$1""", UUID(pending["request_id"])) == "deleted"
            result = await service.result(fixture.owner, fixture.area, UUID(pending["request_id"]))
            original = result["sources"][0]
            assert original["source_type"] == "project_file"
            saved = await fixture.sources.read(fixture.owner, fixture.area, UUID(original["source_id"]))
            assert saved.content["text"] == "value = 42\n"
            assert saved.trust.value == "external_content"
            assert saved.ingestion_status == "pending"
            with pytest.raises(SourceError):
                await service.result(fixture.other, fixture.area, UUID(pending["request_id"]))
            with pytest.raises(SourceError):
                await service.poll(token[:-5] + "xxxxx", offer)
            # An identical read reuses the original revision; a changed file invalidates old receipts.
            same = await service.submit(fixture.owner, fixture.area, request)
            work = (await service.poll(token, offer))["request"]
            await service.complete(token, completion(work))
            again = await service.result(fixture.owner, fixture.area, UUID(same["request_id"]))
            assert again["sources"][0]["revision"] == original["revision"]
            changed = await service.submit(fixture.owner, fixture.area, request)
            work = (await service.poll(token, offer))["request"]
            await service.complete(token, completion(work, text="value = 43\n"))
            current = await service.result(fixture.owner, fixture.area, UUID(changed["request_id"]))
            assert current["sources"][0]["revision"] == original["revision"] + 1
            with pytest.raises(SourceError):
                await service.result(fixture.owner, fixture.area, UUID(pending["request_id"]))
            old = await fixture.sources.read(fixture.owner, fixture.area, UUID(original["source_id"]), revision=1)
            assert not old.is_current and old.content["text"] == "value = 42\n"
            # Explicit forgetting wins over a connector reread, even for the same still-present file.
            await fixture.sources.delete_source(fixture.owner, fixture.area, UUID(original["source_id"]), expected_revision=2)
            suppressed = await service.submit(fixture.owner, fixture.area, request)
            work = (await service.poll(token, offer))["request"]
            with pytest.raises(SourceError, match="suppressed|republished"):
                await service.complete(token, completion(work, text="value = 43\n"))
            await service.revoke(fixture.owner, UUID(connection["connector_id"]))
            with pytest.raises(SourceError):
                await service.poll(token, offer)
            with pytest.raises(SourceError):
                await service.result(fixture.owner, fixture.area, UUID(suppressed["request_id"]))
    asyncio.run(exercise())


@asynccontextmanager
async def running_backend(service, owner, token):
    app = FastAPI()
    app.state.runtime_services = SimpleNamespace(knowledge_runtime=SimpleNamespace(projects=service))
    settings_dependency = inspect.signature(require_static_bearer).parameters["settings"].default.dependency
    app.dependency_overrides[settings_dependency] = lambda: JarvisSettings(
        JARVIS_OWNER_ID=owner, JARVIS_STATIC_BEARER_TOKEN=token)
    @app.exception_handler(JarvisHTTPError)
    async def error_handler(request, error):
        return JSONResponse(status_code=error.status_code, content={"error": {"code": error.code, "message": error.message}})
    app.include_router(router)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    sock.setblocking(False)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off", access_log=False))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield "http://127.0.0.1:" + str(sock.getsockname()[1])
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            sock.close()


async def cli(base, *arguments, env):
    process = await asyncio.create_subprocess_exec(*base, *arguments, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        output, errors = await asyncio.wait_for(process.communicate(), timeout=30)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    assert process.returncode == 0, (output.decode(), errors.decode())
    return json.loads(output) if output.strip() else None


@pytest.mark.skipif(not REAL, reason="Requires explicit isolated database and Mac process acceptance")
def test_real_mac_cli_http_originals_navigation_and_local_revocation(tmp_path):
    async def exercise():
        async with source_fixture(JarvisSettings().SUPABASE_DB_URL, tmp_path / "fixtures",
                                  test_name="S10 Mac HTTP source flow") as fixture:
            service = ProjectService(fixture.sources)
            bootstrap = "synthetic-project-bootstrap-" + uuid4().hex
            project = tmp_path / "project"
            project.mkdir()
            text = "class Controller:\n    def move(self):\n        return 42\n"
            (project / "controller.py").write_text(text)
            (project / "main.py").write_text("from controller import Controller\nrobot = Controller()\nrobot.move()\n")
            second = tmp_path / "second"
            second.mkdir()
            (second / "controller.py").write_text("class Controller:\n    pass\n")
            registry = tmp_path / "roots.json"
            credentials = tmp_path / "connection.json"
            python = os.environ.get("JARVIS_MAC_TEST_PYTHON", sys.executable)
            base = [python, "-m", "jarvis_project_connector", "--registry", str(registry)]
            environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "mac_connector"),
                               JARVIS_PROJECT_BOOTSTRAP_TOKEN=bootstrap)
            # The subprocess has no production API/database keys.
            for key in tuple(environment):
                if any(word in key.upper() for word in ("OPENAI", "GEMINI", "SUPABASE", "DATABASE", "ANTHROPIC")):
                    del environment[key]
            await cli(base, "enroll", "--root", str(project), "--root", str(second), env=environment)
            async with running_backend(service, fixture.owner, bootstrap) as url:
                enrolled = await cli(base, "register", "--url", url, "--area-id", str(fixture.area),
                                     "--credentials", str(credentials), env=environment)
                assert "token" not in enrolled
                assert credentials.stat().st_mode & 0o077 == 0
                connector_token = json.loads(credentials.read_text())["token"]
                assert connector_token not in json.dumps(enrolled)
                async with httpx.AsyncClient(base_url=url, trust_env=False) as http:
                    # Scoped credentials cannot invoke the owner-only administration endpoint.
                    denied = await http.post("/projects/connectors", json={},
                                             headers={"Authorization": "Bearer " + connector_token})
                    assert denied.status_code in (401, 403)
                    await cli(base, "serve", "--credentials", str(credentials), "--once", env=environment)
                    roots = await service.roots(fixture.owner, fixture.area)
                    assert len(roots["roots"]) == 2 and all(root["online"] for root in roots["roots"])
                    root_id = UUID(next(root["root_id"] for root in enrolled["roots"] if root["label"] == "project"))
                    async def query(read):
                        pending = await service.submit(fixture.owner, fixture.area, Inspection(root_id=root_id, request=read))
                        await cli(base, "serve", "--credentials", str(credentials), "--once", env=environment)
                        return await service.result(fixture.owner, fixture.area, UUID(pending["request_id"]))
                    read = await query({"operation": "read", "path": "controller.py", "max_chars": 12})
                    assert read["inspection"]["result"]["content"] == text[:12]
                    original = await fixture.sources.read(fixture.owner, fixture.area, UUID(read["sources"][0]["source_id"]))
                    assert original.content["text"] == text  # Full original, not the 12-character page.
                    result = await query({"operation": "definition", "path": "main.py", "line": 3, "column": 8})
                    items = result["inspection"]["result"]["items"]
                    assert any(item["path"] == "controller.py" and item["line"] == 2 for item in items)
                    assert not result["inspection"]["result"]["analysis_complete"]
                    for source in result["sources"]:
                        assert source["source_id"] and source["content_digest"]
                    async def refresh():
                        async with fixture.pool.acquire() as db:
                            await db.execute("""update jarvis_project_file_watches
                                set checked_at=clock_timestamp()-interval '1 minute' where source_id=$1""",
                                original.source_id)
                        await cli(base, "serve", "--credentials", str(credentials), "--once", env=environment)
                        return await fixture.sources.read(fixture.owner, fixture.area, original.source_id)
                    (project / "controller.py").write_text(text.replace("42", "99"))
                    changed = await refresh()
                    assert changed.revision == original.revision + 1
                    assert "return 99" in changed.content["text"]
                    with pytest.raises(SourceError):
                        await service.result(fixture.owner, fixture.area, UUID(result["request_id"]))
                    (project / "controller.py").unlink()
                    direct_failure = await query({"operation": "read", "path": "controller.py"})
                    assert direct_failure["inspection"]["result"]["status"] == "unavailable"
                    immediate = await fixture.sources.read(fixture.owner, fixture.area, original.source_id)
                    assert immediate.content["availability"] == "unavailable"
                    assert "text" not in immediate.content
                    unavailable = await refresh()
                    assert unavailable.content["availability"] == "unavailable"
                    assert "text" not in unavailable.content
                    (project / "controller.py").write_bytes(b"binary" + bytes([0]) + b"source")
                    excluded = await refresh()
                    assert excluded.content["availability"] == "unavailable"
                    assert "text" not in excluded.content
                    (project / "controller.py").write_text(text)
                    restored = await refresh()
                    assert restored.content["text"] == text
                    # Local enrollment removal is propagated by the next outbound poll.
                    registry.write_text("[]")
                    await cli(base, "serve", "--credentials", str(credentials), "--once", env=environment)
                    with pytest.raises(SourceError):
                        await service.result(fixture.owner, fixture.area, UUID(result["request_id"]))
                    with pytest.raises(SourceError):
                        await fixture.sources.read(fixture.owner, fixture.area, original.source_id, revision=1)
            assert (project / "controller.py").read_text() == text
    asyncio.run(exercise())
