"""Isolated, bounded memory lifecycle qualification; never starts a service.

Run from backend with the configured SECOND_BRAIN_HINDSIGHT_* credentials:
    python -m evals.responsive_voice.memory_lifecycle_qualification --actual \
        --report-dir /absolute/new/private/directory

The calendar labels describe synthetic source dates, NOT an advanced scheduler
clock. Fresh KnowledgeRuntime objects and sessions are real; OS process restarts,
frontend speech, reminders and model-authored answers are outside this runner.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import asyncpg

from app.knowledge.hindsight import HindsightClient
from app.knowledge.models import SourceError
from app.knowledge.runtime import KnowledgeRuntime
from app.migrations.runner import apply_migrations, quote_schema
from app.runtime.feature_config import Capability, SecondBrainSettings
from app.tasks.context_events import ContextIngress
from app.tasks.models import TaskError
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from evals.second_brain.fixtures import DatabaseFixture


ORIGINAL = "I keep my spare workshop key inside the blue toolbox."
REPLACEMENT = "I keep my spare workshop key inside the red canvas pouch."
INDEPENDENT = "My bicycle's frame serial is LARCH-492."
KEY_QUERY = "Where do I keep my spare workshop key?"
INDEPENDENT_QUERY = "What is my bicycle's frame serial?"
SOURCE_DAY_ZERO = datetime(2026, 1, 5, 12, tzinfo=UTC)
STAGES = (
    "day0_ingestion", "fresh_day1_recall", "day7_correction",
    "fresh_day30_current_and_historical", "day90_deletion",
    "fresh_day90_no_resurrection",
)


class QualificationFailure(RuntimeError):
    """Only static, credential-free codes may be included in reports."""


def require(condition, code):
    if not condition:
        raise QualificationFailure(code)


@dataclass(frozen=True)
class Bounds:
    drain_seconds: float = 180
    max_steps: int = 180
    lifecycle_seconds: float = 720
    cleanup_seconds: float = 240

    def __post_init__(self):
        if (type(self.max_steps) is not int or not 1 <= self.max_steps <= 360
                or not 0 < self.drain_seconds <= 300
                or not 0 < self.lifecycle_seconds <= 1200
                or not 0 < self.cleanup_seconds <= 600):
            raise ValueError("Invalid qualification bounds")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def failure(error):
    # Never serialize exception messages from HTTP, SQL, configuration or tools.
    value = {"type": type(error).__name__}
    if isinstance(error, QualificationFailure):
        value["check"] = str(error)
    return value


def new_report(mode):
    require(mode in {"actual_hindsight", "controlled_hindsight_real_postgres"}, "invalid_evidence_mode")
    return {
        "format": "jarvis.responsive_voice.memory_lifecycle.v1",
        "evidence_mode": mode,
        "status": "preparing", "accepted": False,
        "native_semantic_quality_qualified": False,
        "stages": {name: {"status": "unreached"} for name in STAGES},
        "runtime_instances": [], "recalls": [], "drains": [],
        "clock": {
            "source_day_zero": SOURCE_DAY_ZERO.isoformat(),
            "domain_day_labels": [0, 1, 7, 30, 90],
            "elapsed_clock": "time.monotonic",
            "scheduler_time_travel": False,
            "database_clock_changed": False,
        },
        "limitations": [
            "Synthetic typed ingress; no frontend, ASR, spoken answer or model-authored answer qualification.",
            "Fresh runtime/session reconstruction in one process, not an OS process restart.",
            "Calendar labels and original source timestamps do not test TTL, timers or real elapsed days.",
            "Historical evidence is an explicit local version read, not historical semantic search.",
            "No claim about backups, other tenants or independently supplied mentions outside the deletion group.",
            "No measured provider token cost; only bounded requests and real monotonic elapsed time.",
        ],
    }


def assert_recall(result, expected, *, absent=(), forbidden=()):
    require(result.get("status") == "available", "recall_unavailable")
    sources = result.get("sources", [])
    for original in expected:
        matches = [item for item in sources if item.get("source_id") == str(original.source_id)
                   and item.get("revision") == original.revision]
        require(len(matches) == 1, "expected_original_not_recalled")
        item = matches[0]
        require(item.get("is_current") is True, "recalled_source_not_current")
        require(item.get("content_digest") == original.content_digest, "recalled_digest_mismatch")
        require(item.get("content") == original.content, "recalled_original_mismatch")
        timestamp = item.get("source_time")
        try:
            observed = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else None
        except ValueError:
            observed = None
        require(observed is not None and observed.utcoffset() is not None
                and observed == original.source_time, "source_age_changed")
        require(item.get("trust") == original.trust, "source_trust_changed")
    forbidden_ids = {str(identity) for identity in absent}
    require(not any(item.get("source_id") in forbidden_ids for item in sources), "deleted_source_recalled")
    serialized = json.dumps(sources, sort_keys=True)
    require(not any(text in serialized for text in forbidden), "obsolete_content_recalled")


class LifecycleCase:
    """Reusable real-ingress seam for later longitudinal/frontend scenarios."""

    def __init__(self, fixture, pool, area, bank, configuration, report, path, bounds):
        self.fixture, self.pool, self.area, self.bank = fixture, pool, area, bank
        self.owner = fixture.owners[0]
        self.configuration = configuration.model_copy(update={"area_id": area})
        self.report, self.path, self.bounds = report, path, bounds
        self.runtime = None
        self.session = None

    def checkpoint(self):
        save(self.path, self.report)

    async def close_runtime(self):
        if self.runtime is not None:
            runtime, self.runtime = self.runtime, None
            self.report.setdefault("provider_requests_by_runtime", []).append(runtime.client.requests)
            await runtime.close()

    async def fresh(self, day):
        await self.close_runtime()
        async with self.pool.acquire() as connection:
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id,title) values($1,'Memory lifecycle fixture') returning id",
                self.owner)
        self.runtime = await KnowledgeRuntime.create(
            pool=self.pool, owner_id=self.owner, configuration=self.configuration)
        self.report["runtime_instances"].append({"session_id": str(self.session), "logical_day": day})
        self.checkpoint()

    async def capture(self, text, day, *, kind="user_transcript"):
        identity = uuid4().hex
        event = ContextIngress(
            session_id=self.session, source_namespace="qualification:memory-lifecycle",
            source_event_id=identity, receipt_id=identity, kind=kind, trust="user_input",
            finality="final", source_time=SOURCE_DAY_ZERO + timedelta(days=day),
            payload={"text": text, "synthetic_fixture": True}, expected_revision=0)
        record = await self.runtime.context_for(self.owner).append(self.owner, event)
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                "select source_id,source_revision from jarvis_knowledge_context_sources "
                "where owner_id=$1 and event_id=$2 and event_revision=$3",
                self.owner, record.id, record.revision)
        require(row is not None, "canonical_capture_not_published")
        source = await self.runtime.sources.read(self.owner, self.area, row["source_id"])
        return event, record, source

    async def invoke(self, name, arguments, record):
        tools = {tool.name: tool for tool in self.runtime.lifecycle_tool_module_for(self.owner).build_tools(
            ToolBuildContext(None, self.owner, self.session))}
        with bind_tool_invocation(provider_call_id=uuid4().hex,
                                  request_event_id=record.id, request_revision=record.revision):
            result = await tools[name].handler(arguments)
        require(result.get("status") == "success", "lifecycle_tool_failed")
        self.report.setdefault("commands", []).append({
            "tool": name, "request_event_id": str(record.id), "request_revision": record.revision,
            "status": result["status"],
        })

    async def drain(self):
        started = time.monotonic()
        steps = 0
        async with asyncio.timeout(self.bounds.drain_seconds):
            while True:
                diagnostic = await self.runtime.outbox.diagnostics(self.owner)
                require(not diagnostic["recorded_failures"], "outbox_has_failed_operations")
                if diagnostic["pending_operations"] == 0:
                    self.report["drains"].append({"steps": steps,
                        "elapsed_ms": round((time.monotonic() - started) * 1000, 3)})
                    return
                require(steps < self.bounds.max_steps, "outbox_step_budget_exhausted")
                await self.runtime.worker.step(self.owner, self.runtime.worker_id)
                steps += 1
                await asyncio.sleep(.2)

    async def recall(self, label, query, expected=(), *, absent=(), forbidden=()):
        started = time.monotonic()
        result = await self.runtime.retrieval.recall(self.owner, self.area, query)
        self.report["recalls"].append({
            "label": label, "status": result.get("status"),
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            "source_versions": [{"id": item.get("source_id"), "revision": item.get("revision")}
                                for item in result.get("sources", [])],
            "rejected_candidates": result.get("rejected_candidates"),
        })
        self.checkpoint()
        assert_recall(result, expected, absent=absent, forbidden=forbidden)

    async def unavailable(self, source, *, revision=None):
        try:
            await self.runtime.sources.read(self.owner, self.area, source.source_id, revision=revision)
        except SourceError:
            return
        raise QualificationFailure("deleted_original_still_readable")

    async def stage(self, name, operation):
        started = time.monotonic()
        self.report["stages"][name] = {"status": "running"}
        self.checkpoint()
        try:
            value = await operation()
        except BaseException as error:
            self.report["stages"][name].update(status="failed", failure=failure(error))
            raise
        else:
            self.report["stages"][name]["status"] = "passed"
            return value
        finally:
            self.report["stages"][name]["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
            self.checkpoint()

    async def exercise(self):
        async def ingest():
            await self.fresh(0)
            event, _, source = await self.capture(ORIGINAL, 0)
            _, _, independent = await self.capture(INDEPENDENT, 0)
            await self.drain()
            for item in (source, independent):
                current = await self.runtime.sources.read(self.owner, self.area, item.source_id)
                require(current.ingestion_status == "searchable", "ingestion_not_searchable")
            return event, source, independent
        original_event, original, independent = await self.stage("day0_ingestion", ingest)

        async def recall_original():
            await self.fresh(1)
            await self.recall("fresh_original", KEY_QUERY, (original,))
            await self.recall("fresh_independent", INDEPENDENT_QUERY, (independent,))
        await self.stage("fresh_day1_recall", recall_original)

        async def correct():
            await self.fresh(7)
            _, request, command_source = await self.capture(
                f"Correct memory source {original.source_id}, revision {original.revision}. "
                f"The replacement is: {REPLACEMENT}", 7, kind="user_request")
            await self.invoke("correct_memory_source", {
                "source_id": str(original.source_id), "expected_revision": original.revision,
                "replacement": REPLACEMENT,
            }, request)
            corrected = await self.runtime.sources.read(self.owner, self.area, original.source_id)
            require(corrected.revision == original.revision + 1, "correction_did_not_version_source")
            require(corrected.content.get("text") == REPLACEMENT, "correction_text_mismatch")
            require(corrected.trust == "derived", "correction_falsely_claims_observation")
            await self.drain()
            require(await self.runtime.client.document(self.bank, original.document_id) is None,
                    "old_native_document_remains_current")
            return corrected, command_source
        corrected, correction_request = await self.stage("day7_correction", correct)

        async def current_and_historical():
            await self.fresh(30)
            historical = await self.runtime.sources.read(
                self.owner, self.area, original.source_id, revision=original.revision)
            require(not historical.is_current and historical.content == original.content
                    and historical.source_time == original.source_time, "historical_original_changed")
            await self.recall("corrected_current", KEY_QUERY, (corrected,), forbidden=(ORIGINAL,))
            await self.recall("independent_after_correction", INDEPENDENT_QUERY, (independent,))
        await self.stage("fresh_day30_current_and_historical", current_and_historical)

        async def delete():
            await self.fresh(90)
            targets = (corrected, correction_request)
            _, request, _ = await self.capture(
                "Permanently delete both exact memory sources and their historical originals: "
                + "; ".join(f"{item.source_id} revision {item.revision}" for item in targets)
                + ". Keep every other fact.", 90, kind="user_request")
            await self.invoke("delete_memory_source_group", {"sources": [
                {"source_id": str(item.source_id), "expected_revision": item.revision} for item in targets
            ]}, request)
            await self.unavailable(corrected)
            await self.unavailable(original, revision=original.revision)
            await self.drain()
            for item in (original, corrected, correction_request):
                require(await self.runtime.client.document(self.bank, item.document_id) is None,
                        "deleted_native_document_still_exists")
        await self.stage("day90_deletion", delete)

        async def no_resurrection():
            await self.fresh(90)
            # Exact old ingress receipt, not a new user utterance or new authority.
            try:
                await self.runtime.context_for(self.owner).append(self.owner, original_event)
            except (SourceError, TaskError):
                self.report["original_receipt_replay"] = "rejected"
            else:
                self.report["original_receipt_replay"] = "returned_without_readable_original"
            await self.drain()
            await self.unavailable(corrected)
            await self.unavailable(original, revision=original.revision)
            await self.recall("deleted_topic", KEY_QUERY,
                absent=(original.source_id, correction_request.source_id), forbidden=(ORIGINAL, REPLACEMENT))
            await self.recall("independent_after_deletion", INDEPENDENT_QUERY, (independent,),
                absent=(original.source_id, correction_request.source_id))
        await self.stage("fresh_day90_no_resurrection", no_resurrection)

    async def cleanup_native(self):
        """Normal outbox cleanup first; never erase recovery records mid-flight."""
        if self.runtime is None:
            self.runtime = await KnowledgeRuntime.create(
                pool=self.pool, owner_id=self.owner, configuration=self.configuration)
        async with self.pool.acquire() as connection:
            rows = await connection.fetch(
                "select id,current_revision from jarvis_knowledge_sources "
                "where owner_id=$1 and area_id=$2 and status<>'deleted'", self.owner, self.area)
        require(len(rows) <= 32, "fixture_cleanup_exceeds_source_bound")
        targets = [(self.area, row["id"], row["current_revision"]) for row in rows]
        if len(targets) == 1:
            _, identity, revision = targets[0]
            await self.runtime.sources.delete_source(self.owner, self.area, identity, expected_revision=revision)
        elif targets:
            async with self.runtime.sources.transaction(self.owner) as connection:
                await self.runtime.sources.delete_sources_in_transaction(connection, self.owner, targets)
        await self.drain()
        await self.runtime.client._call("DELETE", self.runtime.client._bank(self.bank), missing=True)
        # Independent client, not the runtime client or transaction that deleted it.
        audit = HindsightClient(self.configuration.base_url, self.configuration.api_key.get_secret_value(),
            timeout=self.configuration.http_timeout_seconds,
            private_http_origin=self.configuration.private_http_origin)
        try:
            inventory = await audit._call("GET", "/v1/default/banks")
            require(isinstance(inventory.get("banks"), list), "native_cleanup_inventory_invalid")
            require(self.bank not in {item["bank_id"] for item in inventory["banks"]}, "native_bank_cleanup_residue")
        finally:
            await audit.close()
        return {"native_bank_absent": True, "independent_client_audit": True,
                "outbox_pending": 0, "outbox_failures": 0}


async def run_qualification(dsn, configuration, directory, *, mode="actual_hindsight", bounds=Bounds(),
                            case_factory=LifecycleCase):
    """Configuration is required; controlled-provider injection belongs only to tests."""
    directory = Path(directory)
    require(directory.is_absolute(), "report_directory_must_be_absolute")
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    report, path = new_report(mode), directory / "qualification.json"
    save(path, report)
    fixture = DatabaseFixture(dsn, directory, test_name="responsive-memory-lifecycle")
    bank = "jarvis_memory_eval_" + fixture.run_id
    report.update(fixture_manifest=str(fixture.path), bank=bank, owner_id=fixture.owners[0])
    pool = case = None
    bank_available = False
    native_safe = True
    started = time.monotonic()
    try:
        # Prove configured existing service access and fresh bank identity BEFORE DDL.
        client = HindsightClient(configuration.base_url, configuration.api_key.get_secret_value(),
            timeout=min(6, configuration.http_timeout_seconds),
            private_http_origin=configuration.private_http_origin)
        try:
            async with asyncio.timeout(8):
                inventory = await client._call("GET", "/v1/default/banks")
                require(isinstance(inventory.get("banks"), list), "service_inventory_invalid")
                require(bank not in {item["bank_id"] for item in inventory["banks"]}, "fixture_bank_already_exists")
            bank_available = True
            report["existing_service_reachable"] = True
        finally:
            await client.close()
        connection = await asyncpg.connect(dsn, timeout=8, command_timeout=300)
        try:
            schema = await fixture.create_schema(connection)
            await apply_migrations(connection, schema=schema, fresh=True)
        finally:
            await connection.close(timeout=5)

        async def setup(connection):
            await connection.execute(f"set search_path to {quote_schema(schema)},pg_catalog,extensions")

        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3,
            timeout=8, command_timeout=30, setup=setup)
        async with pool.acquire() as connection:
            area = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) "
                "values($1,'Lifecycle qualification','test') returning id", fixture.owners[0])
            await connection.execute("insert into jarvis_memory_banks(owner_id,area_id,bank_name) values($1,$2,$3)",
                                     fixture.owners[0], area, bank)
        report.update(area_id=str(area), schema=schema, status="running")
        save(path, report)
        case = case_factory(fixture, pool, area, bank, configuration, report, path, bounds)
        native_safe = False
        async with asyncio.timeout(bounds.lifecycle_seconds):
            await case.exercise()
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", failure=failure(error))
    finally:
        report["cleanup"] = {}
        if case is not None:
            try:
                async with asyncio.timeout(bounds.cleanup_seconds):
                    report["cleanup"]["memory"] = await case.cleanup_native()
                native_safe = True
            except Exception as error:
                report["cleanup"]["memory"] = {"status": "failed", "failure": failure(error)}
            finally:
                try:
                    await case.close_runtime()
                except Exception as error:
                    report["cleanup"]["runtime_close"] = failure(error)
        if pool is not None:
            try:
                await asyncio.wait_for(pool.close(), timeout=10)
            except Exception as error:
                pool.terminate()
                report["cleanup"]["pool_close"] = failure(error)
        if native_safe:
            try:
                report["cleanup"]["database"] = await fixture.cleanup()
            except Exception as error:
                report["cleanup"]["database"] = {"status": "failed", "failure": failure(error)}
        else:
            report["cleanup"]["database"] = {
                "status": "preserved_for_cleanup_retry",
                "reason": "Native operations unresolved; preserve their durable cleanup inventory.",
            }
        report["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
        report["existing_service_reachable"] = bank_available
        clean = (native_safe and report["cleanup"].get("database", {}).get("remaining_schemas") == 0
                 and report["cleanup"].get("database", {}).get("remaining_public_owner_rows") == 0
                 and "pool_close" not in report["cleanup"] and "runtime_close" not in report["cleanup"])
        report["accepted"] = report["status"] == "passed" and clean
        report["native_semantic_quality_qualified"] = report["accepted"] and mode == "actual_hindsight"
        save(path, report)
    return report


def configured_inputs():
    from dotenv import load_dotenv
    repo = Path(__file__).resolve().parents[3]
    for path in (repo / "backend/.env", repo / ".env", repo / "deployment/.env"):
        load_dotenv(path, override=False)
    dsn = os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") or os.getenv("SUPABASE_DB_URL")
    settings = SecondBrainSettings(_env_file=None).model_copy(update={
        "enabled": "true", "knowledge_enabled": "true", "knowledge_area_id": str(uuid4())})
    resolution = settings.resolve(Capability.KNOWLEDGE, database_url=dsn)
    return dsn, resolution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", required=True, action="store_true")
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    dsn, resolution = configured_inputs()
    if not resolution.configured:
        require(args.report_dir.is_absolute(), "report_directory_must_be_absolute")
        args.report_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        report = new_report("actual_hindsight")
        report.update(status="blocked", configuration_issue=resolution.issue,
            cleanup={"status": "no_resources_created"}, database_configured=bool(dsn))
        save(args.report_dir / "qualification.json", report)
    else:
        report = asyncio.run(run_qualification(dsn, resolution.configuration, args.report_dir))
    print(json.dumps({"status": report["status"], "accepted": report["accepted"],
        "report": str(args.report_dir / "qualification.json"),
        "configuration_issue": report.get("configuration_issue")}))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
