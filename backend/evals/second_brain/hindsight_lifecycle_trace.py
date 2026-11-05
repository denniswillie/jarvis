"""Read-only native derivative audit and owned service-restart experiment.

No vendor SQL mutation is used. The fixture's original source/outbox handlers
perform deletion; SQL only observes the run-owned private database.
"""
from __future__ import annotations

from app.knowledge.deletion import SourceDeletionService

import argparse
import asyncio
from contextlib import AsyncExitStack
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import time

from app.knowledge.current_context import CurrentSourceContext
from app.knowledge.hindsight import HindsightClient
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.outbox import OutboxRepository
from app.settings import JarvisSettings
from evals.second_brain.hindsight_semantic_trace import drain, shared_tools
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.hindsight_trace import require_check
from evals.second_brain.source_fixture import source_fixture


def sql_literal(value):
    return "'" + value.replace("'", "''") + "'"


def sql_identifier(value):
    return '"' + value.replace('"', '""') + '"'


def native_read(container, query):
    if not container.startswith("jarvis-m02-") or not query.lstrip().lower().startswith("select "):
        raise ValueError("Only an owned native fixture and read-only query are allowed")
    result = subprocess.run(
        ["docker", "exec", container, "psql", "-v", "ON_ERROR_STOP=1", "-U", "postgres",
         "-d", "hindsight", "-At", "-c", "BEGIN READ ONLY; " + query + "; COMMIT"],
        capture_output=True, text=True, timeout=25,
    )
    if result.returncode:
        raise RuntimeError("native_read_only_audit_failed")
    lines = [line for line in result.stdout.splitlines() if line not in {"BEGIN", "COMMIT"}]
    if len(lines) != 1:
        raise RuntimeError("native_read_only_audit_shape")
    return json.loads(lines[0])


def bank_snapshot(container, bank, target, control):
    """Enumerate every bank-scoped table, without exporting source text."""
    tables = native_read(container,
        "select coalesce(jsonb_agg(table_name order by table_name),'[]'::jsonb) "
        "from information_schema.columns where table_schema='public' and column_name='bank_id'")
    if not isinstance(tables, list) or not tables or len(tables) > 64:
        raise RuntimeError("native_bank_table_inventory_invalid")
    rows = {}
    for name in tables:
        target_match = "position(" + sql_literal(target.lower()) + " in lower(data::text)) > 0"
        control_match = "position(" + sql_literal(control.lower()) + " in lower(data::text)) > 0"
        rows[name] = native_read(container,
            "select jsonb_build_object('rows',count(*),'target_rows',count(*) filter(where " + target_match + "),"
            "'control_rows',count(*) filter(where " + control_match + "),"
            "'target_ids',coalesce(jsonb_agg(coalesce(data->>'operation_id',data->>'id',"
            "data->>'document_id',data->>'chunk_id','no_identity')) filter(where " + target_match + "),'[]'::jsonb)) "
            "from (select to_jsonb(t)-'embedding'-'search_vector' as data from public." + sql_identifier(name)
            + " t where bank_id=" + sql_literal(bank) + ") source_rows")
    # These tables lack bank_id. No unowned service is ever admitted to this probe.
    joins = {
        "unit_entities": "select to_jsonb(t) as data from unit_entities t join memory_units m on m.id=t.unit_id where m.bank_id=",
        "entity_cooccurrences": "select to_jsonb(t) as data from entity_cooccurrences t join entities e on e.id=t.entity_id_1 where e.bank_id=",
    }
    for name, query in joins.items():
        rows[name] = native_read(container, "select jsonb_build_object('rows',count(*)) from ("
                                 + query + sql_literal(bank) + ") linked_rows")
    return rows


def erasure_findings(before, after):
    """Absence without a positive marker is not promoted to erasure evidence."""
    positives = {name for name, row in before.items() if row.get("target_rows", 0)}
    remaining = {name: row["target_rows"] for name, row in after.items() if row.get("target_rows", 0)}
    return {
        "positively_exercised_tables": sorted(positives),
        "remaining_target_rows": remaining,
        "positive_surfaces_cleared": bool(positives) and not remaining,
        "no_positive_marker_tables": sorted(set(before)-positives),
    }


def background_idle(stats):
    counts = stats.get("operations_by_status")
    totals = ("pending_operations", "failed_operations", "pending_consolidation", "failed_consolidation")
    if (not isinstance(counts, dict) or not set(counts) <= {"pending", "processing", "completed", "failed", "cancelled"}
            or any(type(value) is not int or value < 0 for value in counts.values())
            or any(type(stats.get(key)) is not int or stats[key] < 0 for key in totals)):
        raise RuntimeError("native_background_status_invalid")
    if stats["failed_operations"] or stats["failed_consolidation"] or counts.get("failed", 0):
        raise RuntimeError("native_background_operation_failed")
    return not (stats["pending_operations"] or stats["pending_consolidation"]
                or counts.get("pending", 0) or counts.get("processing", 0))


async def quiescent(client, bank, *, timeout=120):
    deadline = time.monotonic() + timeout
    stable = 0
    while time.monotonic() < deadline:
        stats = await client._call("GET", client._bank(bank) + "/stats")
        if background_idle(stats):
            stable += 1
            if stable >= 3:
                return stats
        else:
            stable = 0
        await asyncio.sleep(1)
    raise TimeoutError("native_background_quiescence_deadline")


async def exercise(root, service, settings):
    report = {"stage_accepted": False, "synthetic_only": True, "checks": {}, "worker_events": [],
              "scope": "Native source-specific derivative observations, service outage and restart"}
    started = time.monotonic()
    target_marker, control_marker = "ORCHID-" + secrets.token_hex(6), "LANTERN-" + secrets.token_hex(6)
    report["marker_digests"] = [hashlib.sha256(value.encode()).hexdigest()
                                for value in (target_marker, control_marker)]
    try:
        async with AsyncExitStack() as stack:
            fixture = await stack.enter_async_context(source_fixture(
                settings.SUPABASE_DB_URL, root, test_name="MA02-native-derivatives-and-restart",
            ))
            client = HindsightClient(service["url"], service["token"], timeout=5)
            stack.push_async_callback(client.close)
            worker = IngestionWorker(OutboxRepository(fixture.sources, retry_seconds=1), client)
            target = await fixture.publish("Alex is building project " + target_marker + ", an ongoing robotics project. "
                                           "Alex stopped writing binary_search today. Both search-bound updates "
                                           "still remain; the function is not complete.",
                                           source_key="erasure-target")
            control = await fixture.publish("Alex runs " + control_marker + ", an unrelated cafe business, "
                                            "and has a supplier meeting on September 20.",
                                            source_key="unrelated-control")
            await drain(fixture, worker, report, label="lifecycle-positive-controls")
            report["stats_before"] = await quiescent(client, target.bank_name)
            for source in (target, control):
                client.verify_document(source, await client.document(source.bank_name, source.document_id))
            before = await asyncio.to_thread(bank_snapshot, service["container"], target.bank_name,
                                             target_marker, control_marker)
            report["before"] = before
            require_check(report, "original_and_derived_positive_controls",
                          before["documents"]["target_rows"] > 0 and before["memory_units"]["target_rows"] > 0
                          and before["async_operations"]["target_rows"] > 0
                          and before["documents"]["control_rows"] > 0)
            require_check(report, "unsupported_payload_histories_disabled",
                          all(before[name]["rows"] == 0 for name in
                              ("llm_requests", "audit_log", "observation_history", "mental_model_history")))
            require_check(report, "consolidation_really_exercised",
                          report["stats_before"].get("total_observations", 0) > 0)
            save(root / "lifecycle-result.json", report)

            # Stop only this fixture's API process; its private DB stays alive.
            await asyncio.to_thread(service["stop_owned"])
            pending = await fixture.publish("Synthetic current context remains available during a memory outage.",
                                            source_key="outage-pending")
            current_started = time.monotonic()
            current = await CurrentSourceContext(fixture.sources).describe(fixture.owner, fixture.area, include_content=True)
            report["outage_current_seconds"] = time.monotonic()-current_started
            report["outage_current"] = current
            require_check(report, "pending_current_context_survives_native_outage",
                          str(pending.source_id) in json.dumps(current)
                          and current.get("provider_contacted") is False)
            tools = shared_tools(fixture, client)
            failed = await tools["search_memory"].handler({"query": "What remains unfinished on Alex\u0027s robotics project?"})
            report["outage_search"] = failed
            require_check(report, "native_outage_is_not_successful_empty_recall", failed["status"] == "failed")
            await asyncio.to_thread(service["start_owned"])
            # Rebuild the application worker too. No process-local receipt is reused.
            worker = IngestionWorker(OutboxRepository(fixture.sources, retry_seconds=1), client)
            await drain(fixture, worker, report, label="restart-pending-ingestion")
            for source in (target, control, pending):
                client.verify_document(source, await client.document(source.bank_name, source.document_id))
            require_check(report, "restart_preserves_controls_and_ingests_pending_original", True)

            await fixture.sources.delete_source(fixture.owner, fixture.area, target.source_id, expected_revision=1)
            await drain(fixture, worker, report, label="source-specific-erasure")
            deletion = SourceDeletionService(fixture.sources)
            report["source_deletion_receipt"] = await deletion.inspect(
                fixture.owner, fixture.area, target.source_id)
            report["source_deletion_documents"] = await deletion.documents(
                fixture.owner, fixture.area, target.source_id)
            require_check(report, "source_deletion_receipt_records_only_verified_scope",
                          report["source_deletion_receipt"]["state"] == "scoped_cleanup_recorded"
                          and report["source_deletion_receipt"]["full_erasure_verified"] is False)
            report["stats_after"] = await quiescent(client, target.bank_name)
            after = await asyncio.to_thread(bank_snapshot, service["container"], target.bank_name,
                                            target_marker, control_marker)
            report["after"] = after
            report["erasure"] = erasure_findings(before, after)
            client.verify_document(control, await client.document(control.bank_name, control.document_id))
            require_check(report, "source_specific_deletion_preserves_control",
                          after["documents"]["control_rows"] > 0 and after["documents"]["target_rows"] == 0)
            # Keep going on a derivative finding to observe restart behavior too.
            report["checks"]["positive_native_surfaces_erased"] = report["erasure"]["positive_surfaces_cleared"]
            save(root / "lifecycle-result.json", report)

            await asyncio.to_thread(service["stop_owned"])
            await asyncio.to_thread(service["start_owned"])
            report["stats_after_erasure_restart"] = await quiescent(client, target.bank_name)
            restarted = await asyncio.to_thread(bank_snapshot, service["container"], target.bank_name,
                                                target_marker, control_marker)
            report["after_erasure_restart"] = restarted
            report["restart_erasure"] = erasure_findings(before, restarted)
            raw = await client.recall(target.bank_name, "What are Alex\u0027s current robotics and cafe projects?")
            report["checks"]["deleted_marker_absent_from_raw_native_recall"] = target_marker.lower() not in json.dumps(raw).lower()
            report["checks"]["restart_does_not_restore_deleted_native_surfaces"] = report["restart_erasure"]["positive_surfaces_cleared"]
            client.verify_document(control, await client.document(control.bank_name, control.document_id))
            require_check(report, "unrelated_control_survives_erasure_restart", True)
            report["log_marker_counts"] = {
                "target": (root/"service.log").read_text().lower().count(target_marker.lower()),
                "control": (root/"service.log").read_text().lower().count(control_marker.lower()),
            }
            for source in (control, pending):
                await fixture.sources.delete_source(fixture.owner, fixture.area, source.source_id, expected_revision=1)
            await drain(fixture, worker, report, label="remaining-fixture-erasure")
            report["provider_transport"] = client.diagnostics()
            report["scope_limits"] = [
                "Native SQL is read-only diagnostic evidence, not a production cleanup integration.",
                "Exact synthetic marker checks cannot detect every paraphrased or encoded derivative.",
                "Tables without positive markers are inventoried but do not establish erasure of populated copies.",
                "The isolated DB is tmpfs with no backup configured; no real backup was tested or deleted.",
                "Synthetic local reports/logs intentionally remain as evaluation evidence.",
                "No live memory, audio provider or personal source was changed.",
            ]
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        save(root/"lifecycle-result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("/tmp/jarvis-runtime-reuse.OZIsEM/hindsight-env"))
    args = parser.parse_args()
    settings = JarvisSettings()
    with hindsight_service(args.root, runtime=args.runtime, api_key=settings.OPENAI_API_KEY) as service:
        report = asyncio.run(exercise(args.root, service, settings))
    print(json.dumps({"root": str(args.root), "checks": report["checks"], "error": report.get("error"),
                      "erasure": report.get("erasure"), "restart_erasure": report.get("restart_erasure")}))
    return 0 if report["checks"] and all(report["checks"].values()) and "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
