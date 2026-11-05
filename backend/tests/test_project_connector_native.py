"""Opt-in native Hermes -> backend -> separate Mac process source acceptance."""
import asyncio
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import signal
import time
from uuid import UUID, uuid4

import pytest

from app.knowledge.runtime import KnowledgeRuntime
from app.runtime.feature_config import DEFAULT_FRONTIER_MODEL, FrontierConfiguration, KnowledgeConfiguration, TaskLimits
from app.settings import JarvisSettings
from app.tasks.context_events import ContextIngress
from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider
from app.tasks.models import TaskCreate, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.hindsight_semantic_trace import drain
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.source_fixture import source_fixture
from test_project_connector_transport import cli, running_backend


def cites_progress_correction(citations, originals, correction):
    """Match either supported evidence representation to the actual saved event."""
    return any(item.get("event_id") == str(correction.id)
        and item.get("revision") == correction.revision for item in citations) or any(
        item.get("context_event_id") == str(correction.id)
        and item.get("context_event_revision") == correction.revision for item in originals)


@pytest.mark.skipif(os.environ.get("JARVIS_TEST_NATIVE_PROJECTS") != "1",
                    reason="Requires explicit native Hermes/Hindsight and isolated Mac acceptance")
def test_native_worker_uses_actual_scoped_mac_source(tmp_path):
    settings = JarvisSettings()
    runtime = Path(os.environ["JARVIS_NATIVE_PROJECT_RUNTIME"])
    root = Path("/tmp") / ("jarvis-m02-native.s94-" + uuid4().hex)
    (tmp_path / "native-root.txt").write_text(str(root))
    report = {"format": "jarvis.s10.native-project.v1", "worker_events": [], "model": DEFAULT_FRONTIER_MODEL,
              "checks": {}, "semantic_grade": "requires independent review"}
    with hindsight_service(root, runtime=runtime / "hindsight-env", api_key=settings.OPENAI_API_KEY) as service:
        async def exercise():
            async with source_fixture(settings.SUPABASE_DB_URL, root / "fixtures",
                                      test_name="S10 actual Mac native worker") as fixture:
                tasks = TaskRepository(fixture.pool, TaskLimits(lease_seconds=300))
                knowledge = await KnowledgeRuntime.create(pool=fixture.pool, owner_id=fixture.owner, tasks=tasks,
                    configuration=KnowledgeConfiguration(base_url=service["url"], api_key=service["token"],
                                                         area_id=fixture.area, poll_seconds=.2))
                process = None
                try:
                    async with fixture.pool.acquire() as db:
                        session = await db.fetchval(
                            "insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
                    await knowledge.context_repository.append(fixture.owner, ContextIngress(
                        session_id=session, source_namespace="voice:s10-prior-report", source_event_id="prior-note",
                        receipt_id=uuid4().hex, kind="user_transcript", trust="user_input",
                        source_time=datetime.now(UTC)-timedelta(days=1),
                        payload={"text": "PrismBridge is unfinished. Controller.move used to return 7; that may be outdated. "
                            "Calibration and hardware testing are still pending."}))
                    await drain(fixture, knowledge.worker, report, label="s10-old-note")
                    project, distractor = root / "PrismBridge", root / "CopperBridge"
                    project.mkdir()
                    distractor.mkdir()
                    source_text = "class Controller:\n    def move(self):\n        return (6 * 7) + 5\n"
                    (project / "controller.py").write_text(source_text)
                    (distractor / "controller.py").write_text("class Controller:\n    def move(self):\n        return 99\n")
                    registry, credential = root / "roots.json", root / "connector.json"
                    bootstrap = "synthetic-s10-bootstrap-" + uuid4().hex
                    environment = {"PATH": "/usr/bin:/bin", "HOME": str(root), "LANG": "en_US.UTF-8",
                                   "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "mac_connector"),
                                   "JARVIS_PROJECT_BOOTSTRAP_TOKEN": bootstrap}
                    base = [str(runtime / "jarvis-env/bin/python"), "-m", "jarvis_project_connector", "--registry", str(registry)]
                    await cli(base, "enroll", "--root", str(project), "--root", str(distractor), env=environment)
                    async with running_backend(knowledge.projects, fixture.owner, bootstrap) as url:
                        registered = await cli(base, "register", "--url", url, "--area-id", str(fixture.area),
                                               "--credentials", str(credential), env=environment)
                        with (root / "mac-process.log").open("wb") as log:
                            process = await asyncio.create_subprocess_exec(*base, "serve", "--credentials", str(credential),
                                env=environment, stdout=log, stderr=log)
                        report["mac_pid"] = process.pid
                        async with asyncio.timeout(10):
                            while not all(item["online"] for item in (await knowledge.projects.roots(
                                    fixture.owner, fixture.area))["roots"]):
                                if process.returncode is not None:
                                    raise RuntimeError("The actual Mac connector stopped")
                                await asyncio.sleep(.1)
                        query = ("For the enrolled PrismBridge project, inspect the current saved controller.py and tell me "
                                 "exactly what Controller.move returns and why. The older note may be wrong, and CopperBridge "
                                 "is a different project. Also recall where I left the unfinished project. "
                                 "Use real source access and retained progress notes, not the camera or remembered code. "
                                 "Do not modify files or infer project completion from a function working.")
                        task = await tasks.create(fixture.owner, TaskCreate(
                            request_id=uuid4(), title="S10 current Mac source acceptance", requested_outcome=query,
                            session_id=session, knowledge_area_id=fixture.area), model=DEFAULT_FRONTIER_MODEL)
                        claim = await tasks.claim_next("s10-native-acceptance", ("frontier",))
                        assert claim is not None and claim.task.id == task.id
                        context = TaskRunner(tasks, {}).context(claim)
                        provider = HermesTaskProvider(
                            FrontierConfiguration(model=DEFAULT_FRONTIER_MODEL, reasoning_effort="medium",
                                                  api_key=settings.OPENAI_API_KEY),
                            HermesRuntimeConfiguration(runtime / "hermes-env/bin/python", root / "hermes-state"),
                            prepare=WorkerContextFactory(fixture.pool, knowledge=knowledge))
                        started = time.monotonic()
                        outcome = await asyncio.wait_for(provider.execute(context), 240)
                        report["worker_seconds"] = time.monotonic() - started
                        assert await tasks.finish(claim, outcome)
                        report["outcome"] = {"status": outcome.status.value, "payload": outcome.payload}
                        report["usage"] = await context.budgets.usage(fixture.owner, task.id)
                        evidence = []
                        for citation in outcome.payload.get("evidence", []):
                            if "source_id" not in citation:
                                continue  # Context citations are validated by task-result acceptance.
                            source = await fixture.sources.read(fixture.owner, fixture.area, UUID(citation["source_id"]),
                                                                revision=citation["revision"])
                            evidence.append(source.original_evidence())
                        report["evidence"] = evidence
                        correct_root = next(item["root_id"] for item in registered["roots"] if item["label"] == "PrismBridge")
                        wrong_root = next(item["root_id"] for item in registered["roots"] if item["label"] == "CopperBridge")
                        receipts = {item["name"] for item in outcome.payload.get("tool_receipts", [])}
                        report["checks"].update({
                            "task_completed": outcome.status == TaskStatus.SUCCEEDED,
                            "actual_mac_tools": {"list_project_roots", "inspect_project"} <= receipts,
                            "correct_file_cited": any(source["source_type"] == "project_file"
                                and source["content"].get("root_id") == correct_root
                                and source["content"].get("text") == source_text for source in evidence),
                            "no_wrong_root_citation": not any(source["content"].get("root_id") == wrong_root for source in evidence),
                            "no_writes": (project / "controller.py").read_text() == source_text,
                        })
                        await drain(fixture, knowledge.worker, report, label="s10-project-originals")
                        report["recall"] = await knowledge.retrieval.recall(fixture.owner, fixture.area,
                            "PrismBridge controller.py Controller.move current saved implementation")
                        correct_sources = {source["source_id"] for source in evidence
                            if source["source_type"] == "project_file" and source["content"].get("root_id") == correct_root}
                        report["checks"]["published_source_discoverable"] = any(
                            source["source_id"] in correct_sources for source in report["recall"]["sources"])
                        correction = await knowledge.context_repository.append(fixture.owner, ContextIngress(
                            session_id=session, source_namespace="voice:focused-project-correction",
                            source_event_id="progress-correction", receipt_id=uuid4().hex,
                            kind="user_transcript", trust="user_input", source_time=datetime.now(UTC),
                            payload={"text": "Correction to my PrismBridge progress: calibration is done. "
                                "Hardware testing is still unfinished. I stopped for today, not finished the project."}))
                        await drain(fixture, knowledge.worker, report, label="focused-project-correction")
                        followup = await tasks.create(fixture.owner, TaskCreate(
                            request_id=uuid4(), title="Focused project continuity after correction",
                            requested_outcome="Resume my PrismBridge project after my latest progress correction. "
                                "Read the saved controller.py and recall my latest reported progress. "
                                "Tell me what is done and what remains. Cite the current file and the retained "
                                "progress evidence; do not assume the whole project is finished. Do not edit files.",
                            session_id=session, knowledge_area_id=fixture.area), model=DEFAULT_FRONTIER_MODEL)
                        follow_claim = await tasks.claim_next("focused-project-followup", ("frontier",))
                        assert follow_claim is not None and follow_claim.task.id == followup.id
                        follow_context = TaskRunner(tasks, {}).context(follow_claim)
                        follow_outcome = await asyncio.wait_for(provider.execute(follow_context), 240)
                        assert await tasks.finish(follow_claim, follow_outcome)
                        report["continuity"] = {"correction_event_id": str(correction.id),
                            "status": follow_outcome.status.value, "payload": follow_outcome.payload,
                            "semantic_review": "Verify calibration reported done, hardware testing still pending, "
                                "and no claim of whole-project completion."}
                        follow_sources = []
                        for citation in follow_outcome.payload.get("evidence", []):
                            if "source_id" not in citation:
                                continue  # Checked against the actual correction event below.
                            item = await fixture.sources.read(fixture.owner, fixture.area, UUID(citation["source_id"]),
                                revision=citation["revision"])
                            follow_sources.append(item.original_evidence())
                        report["continuity"]["evidence"] = follow_sources
                        report["checks"]["continuity_completed"] = follow_outcome.status == TaskStatus.SUCCEEDED
                        report["checks"]["continuity_current_file_cited"] = any(
                            item["source_type"] == "project_file" and item["content"].get("text") == source_text
                            for item in follow_sources)
                        report["checks"]["continuity_correction_cited"] = cites_progress_correction(
                            follow_outcome.payload.get("evidence", []), follow_sources, correction)
                        await knowledge.projects.revoke(fixture.owner, UUID(registered["connector_id"]))
                finally:
                    if process is not None and process.returncode is None:
                        process.send_signal(signal.SIGINT)
                        try:
                            await asyncio.wait_for(process.wait(), 10)
                        except TimeoutError:
                            process.kill()
                            await process.wait()
                    report["mac_process_terminal"] = process is None or process.returncode is not None
                    await knowledge.close()
        try:
            asyncio.run(exercise())
        finally:
            save(root / "project-result.json", report)
    assert report["checks"] and all(report["checks"].values()), report["checks"]
