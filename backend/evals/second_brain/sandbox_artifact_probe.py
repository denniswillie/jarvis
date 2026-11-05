"""Real Hermes/E2B/Supabase generated report with manifested synthetic cleanup."""
import argparse
import asyncio
from dataclasses import replace
import hashlib
import io
import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from dotenv import dotenv_values

from app.files.storage import JarvisFileStorageClient
from app.knowledge.files import SourceFileRepository
from app.runtime.feature_config import ArtifactConfiguration, DEFAULT_FRONTIER_MODEL, FrontierConfiguration, SandboxConfiguration, TaskLimits
from app.sandbox.allocations import SandboxAllocationRepository
from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.models import E2BConfiguration, SandboxPolicy
from app.settings import JarvisSettings
from app.storage.supabase_config import get_supabase_storage_config
from app.tasks.artifacts import result_artifacts
from app.tasks.catalog import ConfiguredTaskProvider
from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider
from app.tasks.hermes_result import inspect_expert_output
from app.tasks.models import TaskCreate, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.e2b_probe import Probe
from evals.second_brain.hermes_sandbox_probe import ManifestedProvider, ObservedBridge, cleanup_fixture
from evals.second_brain.source_fixture import source_fixture


REQUEST = """Create a durable downloadable summary.json for this synthetic CSV:
project,hours
alpha,2
beta,5
alpha,3
Use Python in the authorized offline sandbox to parse the CSV and compute an
object with by_project and total_hours. Do not substitute mental arithmetic.
Then publish the resulting summary.json using sandbox_publish_file with the
logical output key summary, filename summary.json, format json, and description
Synthetic project hours. Include the accepted tool operation reference in your
completed_operation_refs. Only the saved receipt proves publication. Do not
email anything. No long-term memory search is needed: all input is above."""


DOCUMENT_REQUEST = """Create two durable downloadable files from this synthetic CSV:
project,hours
alpha,2
beta,5
alpha,3
Use Linux commands and available programs/libraries in the authorized offline
sandbox to compute project totals and generate summary.xlsx and summary.pdf.
Choose your own implementation. Discover dependencies if necessary; do not
download packages in this offline scope or claim unavailable checks succeeded.

The workbook must have a Summary sheet: A1:B1 project,hours; rows 2 and 3 the
alphabetical project totals; A4 total_hours and B4 the Excel formula
=SUM(B2:B3). Use readable widths and number formatting. Preserve the formula;
do not replace it with a static total. Do not claim it was recalculated unless
you actually verified recalculation.
The one-page PDF must be readable, titled Synthetic project hours, and show
each computed total as a separate text line: project: hours, followed by
total_hours: hours. Include the supplied CSV as source context in both files.

Inspect the actual saved files with available sandbox tools, repair problems,
then publish summary.xlsx with operation_key spreadsheet and format xlsx, and
summary.pdf with operation_key report and format pdf. Include BOTH accepted
operation references in completed_operation_refs. Report any inspection limits.
Only publication receipts prove downloadable files. No email or memory search."""


NETWORK_DOCUMENT_REQUEST = DOCUMENT_REQUEST.replace(
    "Use Linux commands and available programs/libraries in the authorized offline\nsandbox",
    "Use Linux commands and available programs/libraries in the authorized internet-enabled\nsandbox"
).replace(
    "Choose your own implementation. Discover dependencies if necessary; do not\ndownload packages in this offline scope or claim unavailable checks succeeded.",
    "This is also a dependency-installation acceptance exercise. First discover whether\n"
    "openpyxl and reportlab are installed. Install the missing libraries from PyPI in a\n"
    "workspace .venv using ordinary shell commands, without root/sudo. Choose the\n"
    "installation commands yourself. In a SUBSEQUENT separate sandbox command, use\n"
    ".venv/bin/python and those libraries to generate both documents. Report actual\n"
    "versions and any failed installation or unavailable inspection honestly. Never\n"
    "claim that an install, check or recalculation succeeded without its result."
)


class DependencyProbeProvider(ManifestedProvider):
    """Independent before/after observations; never installs packages for Hermes."""

    async def prepare(self, handle):
        handle = await self.probe.provider.prepare(handle)
        code = """import importlib.util,json,os,urllib.request
with urllib.request.urlopen("https://pypi.org/simple/", timeout=10) as response:
    status = response.status
print(json.dumps({
    "missing": {name: importlib.util.find_spec(name) is None for name in ("openpyxl","reportlab")},
    "uid": os.getuid(), "pypi_status": status,
    "protected_records_readable": os.access("/var/lib/jarvis", os.R_OK),
    "credential_names_present": [name for name in ("OPENAI_API_KEY","E2B_API_KEY","SUPABASE_DB_URL",
        "SUPABASE_SERVICE_ROLE_KEY") if name in os.environ]
}))"""
        observation = await self.probe.python(handle, code)
        evidence = json.loads(observation.stdout) if observation.exit_code == 0 else {}
        self.probe.check("fresh_network_sandbox_has_missing_dependencies",
            observation.exit_code == 0 and all(evidence.get("missing", {}).get(name) for name in ("openpyxl", "reportlab"))
            and evidence.get("pypi_status") == 200 and evidence.get("uid", 0) != 0
            and evidence.get("protected_records_readable") is False
            and evidence.get("credential_names_present") == [], **evidence)
        return handle

    async def audit_installed(self, handle):
        code = """import importlib.metadata,json,os,sys,openpyxl,reportlab
print(json.dumps({"uid": os.getuid(), "prefix": sys.prefix,
    "versions": {name: importlib.metadata.version(name) for name in ("openpyxl","reportlab")},
    "module_paths": [openpyxl.__file__, reportlab.__file__]}))"""
        observation = await self.probe.run(handle, ".venv/bin/python -c " + shlex.quote(code))
        evidence = json.loads(observation.stdout) if observation.exit_code == 0 else {}
        paths = evidence.get("module_paths", [])
        self.probe.check("agent_installed_dependencies_persist_across_commands",
            observation.exit_code == 0 and evidence.get("uid", 0) != 0
            and len(paths) == 2 and all("/workspace/.venv/" in path for path in paths)
            and evidence.get("prefix", "").endswith("/workspace/.venv"), **evidence)


def document_evidence(filename, content):
    """Independent synthetic-fixture checks, never a production upload validator."""
    if filename == "summary.pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        assert len(reader.pages) == 1, "Expected a one-page report"
        text = reader.pages[0].extract_text()
        lines = {line.strip() for line in text.splitlines()}
        expected = {"Synthetic project hours", "alpha: 5", "beta: 5", "total_hours: 10"}
        assert expected <= lines, "PDF does not contain the independently expected totals"
        return {"page_count": 1, "text": text, "layout_reviewed": False}
    if filename == "summary.xlsx":
        from openpyxl import load_workbook
        workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=False)
        try:
            sheet = workbook["Summary"]
            rows = [list(row) for row in sheet.iter_rows(min_row=1, max_row=4, min_col=1, max_col=2, values_only=True)]
            assert rows[:3] == [["project", "hours"], ["alpha", 5], ["beta", 5]], "Unexpected spreadsheet totals"
            assert rows[3][0] == "total_hours", "Missing total label"
            assert isinstance(rows[3][1], str) and rows[3][1].replace(" ", "").upper() == "=SUM(B2:B3)", "Expected a live SUM formula"
        finally:
            workbook.close()
        cached = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        try:
            value = cached["Summary"]["B4"].value
            assert value is None or value == 10, "Saved formula cache contradicts the input totals"
        finally:
            cached.close()
        return {"rows": rows, "cached_total": value, "expected_total": 10,
                "formula_recalculation_verified": False, "layout_reviewed": False}
    raise ValueError("Unknown synthetic document")


class TrackedStorage:
    def __init__(self, storage, probe):
        self.storage, self.probe = storage, probe

    def __getattr__(self, name):
        return getattr(self.storage, name)

    async def upload_bytes(self, *, path, content, content_type):
        item = {"bucket": self.storage.bucket, "path": path, "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(), "phase": "planned"}
        self.probe.record.setdefault("storage_objects", []).append(item)
        self.probe.persist()
        try:
            receipt = await self.storage.upload_bytes(path=path, content=content, content_type=content_type)
        except BaseException:
            item["phase"] = "unknown"
            self.probe.persist()
            raise
        item["phase"] = "uploaded"
        self.probe.persist()
        return receipt

    async def cleanup(self, files):
        failures = []
        async with files.sources.pool.acquire() as connection:
            identities = await connection.fetch("select id from jarvis_files where owner_id=$1", files.owner_id)
        for row in identities:
            try:
                await files.delete_source_file(owner_id=files.owner_id, file_id=row["id"])
            except Exception as error:
                failures.append({"phase": "source_delete", "type": type(error).__name__})
        for item in self.probe.record.get("storage_objects", []):
            try:
                content = await self.storage.read_object(bucket=item["bucket"], path=item["path"], max_bytes=item["size"])
                if content is not None:
                    if hashlib.sha256(content).hexdigest() != item["sha256"]:
                        raise RuntimeError("Owned test object changed; do not delete unverified content")
                    await self.storage.delete_object(bucket=item["bucket"], path=item["path"])
                elif item["phase"] in {"unknown", "planned"}:
                    raise RuntimeError("Missing unknown upload is not proof it cannot arrive later")
                if await self.storage.read_object(bucket=item["bucket"], path=item["path"], max_bytes=item["size"]) is not None:
                    raise RuntimeError("Owned test object still exists")
                item["phase"] = "deleted"
                self.probe.persist()
            except Exception as error:
                failures.append({"phase": "object_cleanup", "type": type(error).__name__})
        self.probe.record["storage_cleanup"] = {"failures": failures}
        self.probe.persist()
        if failures:
            raise RuntimeError("Manifested storage cleanup remains unresolved")


async def exercise(probe, args, settings):
    frontier = FrontierConfiguration(model=args.model, reasoning_effort="medium", api_key=settings.OPENAI_API_KEY)
    networked = args.scenario == "network-documents"
    documents = args.scenario in {"documents", "network-documents"}
    outputs = {"summary.xlsx": "spreadsheet", "summary.pdf": "report"} if documents else {"summary.json": "summary"}
    probe.record["scenario"] = args.scenario
    probe.persist()
    async with source_fixture(settings.SUPABASE_DB_URL, args.manifest_dir, test_name="native-hermes-artifact") as source:
        probe.record.update(database_manifest=str(source.fixture.path), fixture_owner=source.owner)
        probe.persist()
        tasks = TaskRepository(source.pool, TaskLimits(lease_seconds=600 if networked else 300, active_seconds=600 if networked else 300, request_output_tokens=4096))
        configuration = ArtifactConfiguration(area_id=source.area)
        policy = SandboxPolicy(lifetime_seconds=600 if networked else 300,
                               command_seconds=60 if networked else 30, internet_access=networked)
        service = TaskService(tasks, {"frontier": ConfiguredTaskProvider(frontier.model)},
            sandbox_configuration=SandboxConfiguration(policy=policy), artifact_configuration=configuration)
        async with source.pool.acquire() as connection:
            session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", source.owner)
        task = await service.create(source.owner, TaskCreate(request_id=uuid4(), title="Synthetic durable report",
            requested_outcome=NETWORK_DOCUMENT_REQUEST if networked else DOCUMENT_REQUEST if documents else REQUEST,
            session_id=session, execution_environment="networked_sandbox" if networked else "offline_sandbox",
            publish_artifacts=True))
        claim = await tasks.claim_next("native-artifact-fixture", ("frontier",),
                                     worker_capabilities=policy.required_capabilities + ("files.publish_generated",))
        context = TaskRunner(tasks, {}).context(claim)
        fixture = SimpleNamespace(source=source, tasks=tasks, task=task, claim=claim,
                                  allocations=SandboxAllocationRepository(tasks))
        storage = TrackedStorage(JarvisFileStorageClient(get_supabase_storage_config(settings)), probe)
        files = SourceFileRepository(source.sources, storage, owner_id=source.owner, area_id=source.area)
        # Source originals and handlers are real. External semantic retrieval is
        # deliberately not part of this artifact scenario or claimed by it.
        knowledge = SimpleNamespace(sources=source.sources, retrieval=SimpleNamespace(sources=source.sources),
            configuration=SimpleNamespace(area_id=source.area), context_for=lambda owner: None)
        native = DependencyProbeProvider(probe) if networked else ManifestedProvider(probe)
        factory = WorkerContextFactory(source.pool, knowledge=knowledge, file_storage=storage,
            sandbox_provider=native, sandbox_policy=policy, artifact_configuration=configuration)
        prepared_runs = []
        async def prepare(ctx):
            prepared = await factory(ctx)
            prepared_runs.append(prepared)
            return replace(prepared, tools=ObservedBridge(prepared.tools, probe))
        root = probe.path.parent / "hermes"
        provider = HermesTaskProvider(frontier, HermesRuntimeConfiguration(args.hermes_python, root,
            host_id="isolated-artifact-probe", max_iterations=24 if networked else 16), prepare=prepare)
        probe.record["task_id"] = str(task.id)
        probe.persist()
        try:
            async with asyncio.timeout(540 if networked else 240):
                outcome = await provider.execute(context)
            probe.record["outcome"] = {"status": outcome.status.value, "reason": outcome.reason, "payload": outcome.payload}
            probe.persist()
            probe.check("frontier_generated_and_published_report", outcome.status == TaskStatus.SUCCEEDED)
            async with source.pool.acquire() as connection:
                rows = await connection.fetch("select * from jarvis_files where owner_id=$1", source.owner)
                native_count = await connection.fetchval("select count(*) from jarvis_sandbox_allocations where owner_id=$1", source.owner)
            probe.check("expected_reports_and_one_vm", len(rows) == len(outputs)
                        and {row["filename"] for row in rows} == set(outputs) and native_count == 1)
            by_name = {row["filename"]: row["id"] for row in rows}
            originals = {}
            output_dir = probe.root / "outputs"
            output_dir.mkdir(mode=0o700)
            for filename, file_id in by_name.items():
                content = await files.read_content(owner_id=source.owner, file_id=file_id)
                originals[filename] = content
                destination = output_dir / filename
                destination.write_bytes(content)
                destination.chmod(0o600)
                probe.record.setdefault("downloaded_artifacts", []).append({
                    "filename": filename, "file_id": str(file_id), "path": str(destination),
                    "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
                probe.persist()
                if documents:
                    try:
                        evidence = document_evidence(filename, content)
                    except (AssertionError, ValueError, KeyError) as error:
                        probe.check(filename + "_content", False, error=str(error))
                    probe.check(filename + "_content", True, **evidence)
                else:
                    probe.check("real_storage_computed_values", json.loads(content) ==
                                {"by_project": {"alpha": 5, "beta": 5}, "total_hours": 10})
            if networked:
                async with source.pool.acquire() as connection:
                    allocation = await connection.fetchrow("select * from jarvis_sandbox_allocations where owner_id=$1", source.owner)
                await native.audit_installed(fixture.allocations.handle(allocation))
            await cleanup_fixture(fixture, probe)
            for filename, file_id in by_name.items():
                after = await files.read_content(owner_id=source.owner, file_id=file_id)
                probe.check(filename + "_survives_vm_destruction", after == originals[filename])
                replay = await prepared_runs[-1].tools.invoke("sandbox_artifact_status",
                    {"operation_key": outputs[filename]}, provider_call_id="fixture-after-vm-deletion-" + filename)
                probe.check(filename + "_receipt_needs_no_vm", replay["status"] == "success"
                            and replay["file"]["id"] == str(file_id)
                            and replay["task_operation"]["replayed"])
            accepted = await tasks.finish(claim, outcome)
            final = await tasks.get(source.owner, task.id)
            async with source.pool.acquire() as connection:
                links = await result_artifacts(connection, tasks, final, inspect_expert_output(outcome.payload, task=final))
                uploads = await connection.fetchval("select count(*) from jarvis_file_publications where owner_id=$1", source.owner)
                unsettled = await connection.fetchval("""select count(*) from jarvis_task_reservations
                    where owner_id=$1 and status in ('reserved','submitted','unknown')""", source.owner)
            probe.check("accepted_task_inbox_links_exact_files", accepted and len(links) == len(outputs)
                        and {item["file_id"] for item in links} == {str(identity) for identity in by_name.values()})
            probe.check("one_upload_per_file_and_settled_budgets", uploads == len(outputs)
                        and len(probe.record["storage_objects"]) == len(outputs) and unsettled == 0)
            probe.check("expert_not_audio_model_authored_answer", outcome.payload.get("authorship", {}).get("kind") == "expert")
            probe.record["usage"] = await context.budgets.usage(source.owner, task.id)
            probe.persist()
            if getattr(args, "phone_fixture", False):
                from evals.second_brain.task_file_phone_fixture import serve_task_files
                await serve_task_files(probe, source, service, files, storage, task.id)
        finally:
            try:
                await cleanup_fixture(fixture, probe)
            finally:
                try:
                    await storage.cleanup(files)
                finally:
                    await storage.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--credential-file", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--hermes-python", type=Path, required=True)
    parser.add_argument("--directory", type=Path, default=Path("/tmp"))
    parser.add_argument("--model", default=DEFAULT_FRONTIER_MODEL)
    parser.add_argument("--scenario", choices=("json", "documents", "network-documents"), default="json")
    parser.add_argument("--phone-fixture", action="store_true",
        help="After native acceptance, serve these retained task files locally for simulator checks (20-minute limit).")
    args = parser.parse_args()
    probe = None
    try:
        if args.scenario in {"documents", "network-documents"}:
            # Evaluator dependencies only; publication and worker readiness do not import these.
            from importlib import import_module
            for module in ("pypdf", "openpyxl"):
                import_module(module)
        native = E2BProvider(E2BConfiguration(api_key=dotenv_values(args.credential_file).get("E2B_API_KEY")))
        probe = Probe(native, args.directory)
        print(json.dumps({"manifest": str(probe.path)}), flush=True)
        asyncio.run(exercise(probe, args, JarvisSettings()))
    except Exception as error:
        if probe is not None:
            probe.record["failure"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
            probe.persist()
        print(json.dumps({"passed": False, "error": getattr(error, "code", type(error).__name__)}))
        return 1
    print(json.dumps({"passed": True, "checks": probe.record["checks"], "cleanup": probe.record.get("cleanup"),
                      "storage_cleanup": probe.record.get("storage_cleanup"), "manifest": str(probe.path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
