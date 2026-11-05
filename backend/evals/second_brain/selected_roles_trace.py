"""Actual API/worker images with synthetic canonical context and private Hindsight.

The API check uses its real lifespan and authenticated ASGI routes, not a voice
session. Only this driver is mounted; application code comes from the images.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID, uuid4


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")
    path.chmod(0o600)


def verify_results(memory, code, *, label, source_id, event_id):
    if label not in memory.get("answer", "") or "unfinished" not in memory["answer"].lower():
        raise AssertionError("Memory answer lost the reported location or current unfinished state")
    if not any(item.get("source_id") == str(source_id) for item in memory.get("evidence", [])):
        raise AssertionError("Memory answer did not cite the exact prior-session source")
    if not any(item.get("event_id") == str(event_id) for item in memory.get("evidence", [])):
        raise AssertionError("Memory answer did not cite the current conversation")
    if not any(item.get("name") == "search_memory" and item.get("status") == "success"
               for item in memory.get("tool_receipts", [])):
        raise AssertionError("Memory answer did not use the actual source-backed search tool")
    if "mid = (low + high) // 2" not in code.get("answer", ""):
        raise AssertionError("Independent code answer was incorrect")
    for result in (memory, code):
        if result.get("authorship", {}).get("kind") != "expert" or not result.get("speech"):
            raise AssertionError("Delegated result lacks expert authorship or direct speech")



async def read_task_results(tasks, owner, identities):
    values = []
    for identity in identities:
        task = await tasks.get(owner, UUID(identity))
        if task.result_payload_id is None:
            raise AssertionError("Completed task has no result payload")
        async with tasks.pool.acquire() as connection:
            encoded = await connection.fetchval(
                "select content::text from jarvis_context_payloads where owner_id=$1 and id=$2",
                owner, task.result_payload_id)
        if encoded is None:
            raise AssertionError("Task result payload is unavailable")
        values.append(json.loads(encoded))
    return values


def api_exercise(request_path, receipt_path):
    # Set mounted fixture credentials before any settings cache or app import.
    for variable, filename in (
        ("SUPABASE_DB_URL", "/run/secrets/jarvis_database_url"),
        ("JARVIS_STATIC_BEARER_TOKEN", "/run/secrets/fixture_bearer"),
    ):
        os.environ[variable] = Path(filename).read_text().strip()

    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.runtime.feature_config import MemoryAuthority
    from app.tasks.context_events import ContextIngress

    request = json.loads(request_path.read_text())
    owner = os.environ["JARVIS_OWNER_ID"]
    state = {"format": "jarvis.selected_roles_api.v1", "accepted": False,
             "voice_session_opened": False, "external_effect_requested": False}
    save(receipt_path, state)
    app = create_app()
    with TestClient(app) as client:
        services = app.state.runtime_services
        knowledge = services.knowledge_runtime
        if knowledge is None or services.memory_authority != MemoryAuthority.HINDSIGHT:
            raise AssertionError("API did not initialize selected memory")
        forbidden = ("memory_service", "memory_retrieval_service", "memory_projection_service",
                     "memory_job_runner", "memory_job_runner_task", "experience_consolidation_service")
        if any(getattr(services, key, None) is not None for key in forbidden):
            raise AssertionError("Legacy memory was constructed in the selected API")
        if client.get("/tasks").status_code != 401:
            raise AssertionError("Unauthenticated task route was not rejected")
        if client.get("/tasks", headers={"Authorization": "Bearer wrong-fixture-key"}).status_code != 403:
            raise AssertionError("Wrong task credential was not rejected")
        client.headers["Authorization"] = "Bearer " + os.environ["JARVIS_STATIC_BEARER_TOKEN"]
        if client.get("/health").status_code != 200:
            raise AssertionError("Enabled API is not healthy")
        status = client.get("/tasks/providers")
        status.raise_for_status()
        if not any(item.get("worker_ready") for item in status.json()):
            raise AssertionError("API cannot see the separate native worker")

        async def publish():
            historical = await knowledge.context_repository.append(owner, ContextIngress(
                session_id=UUID(request["historical_session"]), source_namespace="selected-roles-fixture",
                source_event_id=str(uuid4()), receipt_id=str(uuid4()), kind="user_transcript",
                trust="user_input", source_time=datetime.now(UTC) - timedelta(days=1),
                payload={"text": "I put the spare Kestrel workshop proximity sensor in the bin labelled "
                                 + request["label"] + ". This is my report, not camera verification."}))
            async with knowledge.sources.pool.acquire() as connection:
                source_id = await connection.fetchval(
                    "select source_id from jarvis_knowledge_context_sources where owner_id=$1 and event_id=$2",
                    owner, historical.id)
            if source_id is None:
                raise AssertionError("Canonical API capture did not publish a source")
            started = time.monotonic()
            while time.monotonic() - started < 150:
                source = await knowledge.sources.read(owner, knowledge.configuration.area_id, source_id)
                if source.ingestion_status == "searchable":
                    knowledge.client.verify_document(source, await knowledge.client.document(
                        source.bank_name, source.document_id))
                    break
                await asyncio.sleep(.5)
            else:
                raise AssertionError("Canonical source did not become natively searchable")
            current = await knowledge.context_repository.append(owner, ContextIngress(
                session_id=UUID(request["current_session"]), source_namespace="selected-roles-fixture",
                source_event_id=str(uuid4()), receipt_id=str(uuid4()), kind="user_transcript",
                trust="user_input", source_time=datetime.now(UTC),
                payload={"text": "I stopped Kestrel calibration for today. It is unfinished, not completed. "
                                 "Where did I report putting its spare proximity sensor?"}))
            return source, current, round(time.monotonic() - started, 3)

        source, current, ingestion_seconds = client.portal.call(publish)
        state.update(source=source.original_evidence(), current_event_id=str(current.id),
                     ingestion_seconds=ingestion_seconds, legacy_memory_absent=True,
                     authentication_checked=True)
        save(receipt_path, state)
        outcomes = (
            "Find the previously reported bin label for the spare Kestrel workshop proximity sensor "
            "using long-term memory, and state the current calibration status from this conversation. "
            "Quote the exact label and the word unfinished. Cite both the prior-session source and "
            "the current conversation event. Distinguish the user's report from camera verification.",
            "Explain the Python binary search error in mid = low + high // 2. "
            "Give the corrected assignment exactly and a brief explanation. No tools with side effects.",
        )
        ids = []
        for index, outcome in enumerate(outcomes):
            response = client.post("/tasks", json={
                "request_id": str(uuid4()), "title": "Selected roles fixture " + str(index + 1),
                "session_id": request["current_session"], "knowledge_area_id": str(knowledge.configuration.area_id),
                "requested_outcome": outcome,
            })
            response.raise_for_status()
            ids.append(response.json()["id"])
        state["task_ids"] = ids
        state["two_tasks_submitted_before_polling"] = True
        save(receipt_path, state)
        started, states = time.monotonic(), {}
        while time.monotonic() - started < 210:
            for identity in ids:
                response = client.get("/tasks/" + identity)
                response.raise_for_status()
                states[identity] = response.json()
            if all(item["status"] in {"succeeded", "failed", "paused", "waiting_for_input", "cancelled"}
                   for item in states.values()):
                break
            time.sleep(.5)
        state["task_statuses"] = {key: {"status": value["status"], "reason": value.get("reason")}
                                  for key, value in states.items()}
        state["task_wait_seconds"] = round(time.monotonic() - started, 3)
        save(receipt_path, state)
        if len(states) != 2 or any(item["status"] != "succeeded" for item in states.values()):
            raise AssertionError("Both native tasks must succeed; see the private receipt")

        memory, code = client.portal.call(
            read_task_results, knowledge.context_repository.tasks, owner, ids)
        state["results"] = [memory, code]
        save(receipt_path, state)
        verify_results(memory, code, label=request["label"], source_id=source.source_id, event_id=current.id)
        state.update(accepted=True, native_source_original_verified=True,
                     runtime_diagnostics=client.portal.call(knowledge.diagnostics))
    state["api_lifespan_closed"] = True
    save(receipt_path, state)
    print(json.dumps({"accepted": True, "task_ids": ids, "voice_tested": False}))


async def cleanup_memory(fixture, root, api_key):
    """Use normal source forgetting/outbox reconciliation before deleting owned banks."""
    from pydantic import SecretStr
    from app.knowledge.runtime import KnowledgeRuntime
    from app.runtime.feature_config import KnowledgeConfiguration

    runtime = await KnowledgeRuntime.create(pool=fixture.pool, owner_id=fixture.owner,
        configuration=KnowledgeConfiguration(base_url="http://127.0.0.1:8888",
            api_key=SecretStr(api_key), area_id=fixture.area, poll_seconds=.2))
    receipt = {"format": "jarvis.selected_roles_memory_cleanup.v1", "status": "preparing"}
    try:
        async with fixture.pool.acquire() as connection:
            banks = await connection.fetch(
                "select bank_name from jarvis_memory_banks where owner_id=$1", fixture.owner)
            sources = await connection.fetch(
                "select id,current_revision from jarvis_knowledge_sources where owner_id=$1 "
                "and area_id=$2 and status='active'", fixture.owner, fixture.area)
        receipt.update(banks=[row["bank_name"] for row in banks],
                       source_ids=[str(row["id"]) for row in sources])
        save(root / "memory-cleanup.json", receipt)
        for row in sources:
            await runtime.sources.delete_source(fixture.owner, fixture.area, row["id"],
                                                expected_revision=row["current_revision"])
        runtime.start()
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            async with fixture.pool.acquire() as connection:
                pending = await connection.fetchval(
                    "select count(*) from jarvis_memory_outbox where owner_id=$1 and operation='delete' "
                    "and status<>'done'", fixture.owner)
            if pending == 0:
                break
            await asyncio.sleep(.5)
        else:
            raise AssertionError("Owned native source cleanup did not settle")
        for bank in receipt["banks"]:
            await runtime.client._call("DELETE", runtime.client._bank(bank), missing=True)
        inventory = await runtime.client._call("GET", "/v1/default/banks")
        remaining = {item["bank_id"] for item in inventory["banks"]}
        if remaining.intersection(receipt["banks"]):
            raise AssertionError("Owned native bank remains after cleanup")
        receipt.update(status="cleaned", delete_intents_remaining=0, native_banks_absent=True)
    finally:
        await runtime.close()
        save(root / "memory-cleanup.json", receipt)


async def run(root, api_image, worker_image, knowledge_secrets):
    from app.settings import JarvisSettings
    from app.tasks.catalog import TaskWorkerRegistry
    from evals.second_brain.source_fixture import source_fixture

    if not root.is_absolute() or not root.name.startswith("jarvis-selected-roles-"):
        raise ValueError("Use a new absolute jarvis-selected-roles- fixture directory")
    root.mkdir(mode=0o700)
    secret_dir, output = root / "secrets", root / "output"
    secret_dir.mkdir(mode=0o700)
    output.mkdir(mode=0o700)
    settings = JarvisSettings()
    project = "jarvis-selected-" + uuid4().hex[:16]
    repo = Path(__file__).resolve().parents[3]
    state = {"format": "jarvis.selected_roles_deployment.v1", "project": project,
             "status": "preparing", "accepted": False, "live_api_replaced": False}
    environment = os.environ | {
        "JARVIS_FRONTIER_SECRET_DIR": str(secret_dir), "JARVIS_FRONTIER_HOST_ID": project,
        "JARVIS_FRONTIER_MODEL": "gpt-5.6-terra", "JARVIS_FRONTIER_REASONING_EFFORT": "medium",
        "JARVIS_HINDSIGHT_SECRET_DIR": str(knowledge_secrets), "JARVIS_KNOWLEDGE_ENABLED": "true",
    }
    sensitive = [settings.SUPABASE_DB_URL, settings.OPENAI_API_KEY]
    started = False
    compose = ["docker", "compose", "-p", project, "-f", str(repo / "deployment/docker-compose.frontier.yml"),
        "-f", str(repo / "deployment/docker-compose.frontier-knowledge.yml"),
        "-f", str(repo / "deployment/docker-compose.api-knowledge.yml"), "-f", str(root / "roles.json")]

    def command(args, *, timeout=90):
        completed = subprocess.run(args, cwd=repo, env=environment, capture_output=True, text=True, timeout=timeout)
        if completed.returncode:
            detail = completed.stdout + completed.stderr
            for value in sensitive:
                if value:
                    detail = detail.replace(value, "[redacted]")
            (root / "failed-command.log").write_text(detail)
            raise RuntimeError("Selected-role command failed; see private failed-command.log")
        return completed.stdout

    save(root / "deployment-resources.json", state)
    try:
        async with source_fixture(settings.SUPABASE_DB_URL, root, test_name="actual selected API and Hermes roles") as f:
            environment.update(JARVIS_FRONTIER_OWNER_ID=f.owner, JARVIS_KNOWLEDGE_AREA_ID=str(f.area))
            async with f.pool.acquire() as connection:
                schema = await connection.fetchval("select current_schema()")
                sessions = [await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", f.owner) for _ in range(2)]
            parsed = urlsplit(settings.SUPABASE_DB_URL)
            query = dict(parse_qsl(parsed.query))
            query["search_path"] = schema + ",pg_catalog,public,extensions"
            fixture_dsn = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))
            bearer = secrets.token_urlsafe(32)
            sensitive.extend((fixture_dsn, bearer))
            for name, value in (("database-url", fixture_dsn), ("openai-api-key", settings.OPENAI_API_KEY),
                                ("fixture-bearer", bearer)):
                path = secret_dir / name
                path.write_text(value)
                path.chmod(0o444)
            request = {"historical_session": str(sessions[0]), "current_session": str(sessions[1]),
                       "label": "AZURE " + str(uuid4().int % 90000 + 10000)}
            save(root / "request.json", request)
            shared = {"SECOND_BRAIN_ENABLED": "true", "SECOND_BRAIN_TASKS_ENABLED": "true",
                      "SECOND_BRAIN_FRONTIER_ENABLED": "true", "SECOND_BRAIN_MEMORY_AUTHORITY": "hindsight",
                      "SECOND_BRAIN_REQUEST_OUTPUT_TOKENS": "4096"}
            override = {"services": {
                "jarvis-frontier": {"image": worker_image, "environment": shared | {
                    "SECOND_BRAIN_WORKER_HEARTBEAT_SECONDS": "1"}},
                "jarvis-backend": {"image": api_image, "environment": shared | {
                    "APP_ENV": "production", "LOG_LEVEL": "WARNING", "JARVIS_OWNER_ID": f.owner,
                    "AUDIO_PROVIDER_DEFAULT": "openai", "GEMINI_LIVE_ENABLED": "false",
                    "REMINDER_SCHEDULER_ENABLED": "false", "SUPABASE_DB_POOL_MAX_SIZE": "3",
                    "SECOND_BRAIN_FRONTIER_MODEL": "gpt-5.6-terra"},
                    "secrets": ["jarvis_database_url", "fixture_bearer"],
                    "volumes": [
                        str(Path(__file__).resolve()) + ":/fixture/trace.py:ro",
                        str(root / "request.json") + ":/fixture/request.json:ro",
                        str(output) + ":/output"],
                    "command": ["python", "/fixture/trace.py", "api",
                                "--request", "/fixture/request.json", "--receipt", "/output/api-receipt.json"],
                    "init": True, "restart": "no"},
            }, "secrets": {"fixture_bearer": {"file": str(secret_dir / "fixture-bearer")}}}
            save(root / "roles.json", override)
            state.update(owner=f.owner, area_id=str(f.area), schema=schema,
                         images={name: command(["docker", "image", "inspect", "--format", "{{.Id}}", image]).strip()
                                 for name, image in (("api", api_image), ("frontier", worker_image))})
            save(root / "deployment-resources.json", state)
            try:
                started = True
                command(compose + ["up", "-d", "--no-build", "jarvis-frontier"])
                registry = TaskWorkerRegistry(f.pool)
                for _ in range(120):
                    if await registry.ready(f.owner, "frontier", "gpt-5.6-terra"):
                        break
                    await asyncio.sleep(.5)
                else:
                    raise AssertionError("Native selected worker never advertised readiness")
                state["status"] = "running"
                save(root / "deployment-resources.json", state)
                command(compose + ["run", "--rm", "--no-deps", "-T", "jarvis-backend"], timeout=420)
                result = json.loads((output / "api-receipt.json").read_text())
                if not result.get("accepted") or not result.get("api_lifespan_closed"):
                    raise AssertionError("Selected API check did not complete")
                health = json.loads(command(compose + ["exec", "-T", "jarvis-frontier",
                                                       "python", "-m", "app.tasks.worker", "check"]))
                if not health.get("ready"):
                    raise AssertionError("Worker health did not survive both tasks")
                state.update(accepted=True, api=result, worker_health=health)
            finally:
                try:
                    logs = command(compose + ["logs", "--no-color", "--tail", "120", "jarvis-frontier"], timeout=20)
                    for value in sensitive:
                        if value:
                            logs = logs.replace(value, "[redacted]")
                    (root / "worker.log").write_text(logs)
                finally:
                    command(compose + ["down", "--volumes", "--remove-orphans"], timeout=60)
                    state["role_resources_removed"] = True
                await cleanup_memory(f, root, (knowledge_secrets / "api-key").read_text().strip())
                state["memory_cleanup_complete"] = True
    finally:
        shutil.rmtree(secret_dir)
        state["secrets_removed"] = True
        state["status"] = ("cleaned" if (state.get("role_resources_removed") and state.get("memory_cleanup_complete"))
                           or not started else "cleanup_failed")
        save(root / "deployment-resources.json", state)
    if command(["docker", "ps", "-aq", "--filter", "label=com.docker.compose.project=" + project]).strip():
        raise AssertionError("Owned role container remains after cleanup")
    if command(["docker", "volume", "ls", "-q", "--filter", "label=com.docker.compose.project=" + project]).strip():
        raise AssertionError("Owned worker volume remains after cleanup")
    state["independent_role_cleanup_checked"] = True
    save(root / "deployment-resources.json", state)
    print(json.dumps({"accepted": state["accepted"], "cleanup": state["status"], "artifact_dir": str(root)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    api = sub.add_parser("api")
    api.add_argument("--request", type=Path, required=True)
    api.add_argument("--receipt", type=Path, required=True)
    host = sub.add_parser("run")
    host.add_argument("--artifact-dir", type=Path, required=True)
    host.add_argument("--api-image", default="jarvis-backend:s40-private-knowledge")
    host.add_argument("--worker-image", default="jarvis-frontier:s40-private-knowledge")
    host.add_argument("--knowledge-secrets", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "api":
        api_exercise(args.request, args.receipt)
    else:
        asyncio.run(run(args.artifact_dir, args.api_image, args.worker_image, args.knowledge_secrets))


if __name__ == "__main__":
    main()
