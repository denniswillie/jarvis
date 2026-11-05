"""Native deployment/bank-policy gate using only owned synthetic sources."""
import argparse
import asyncio
from contextlib import AsyncExitStack
import json
from pathlib import Path

import httpx

from app.knowledge.hindsight import HindsightClient, HindsightError
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.outbox import OutboxRepository
from app.settings import JarvisSettings
from evals.second_brain.hindsight_semantic_trace import drain, queue_rows
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.hindsight_trace import require_check
from evals.second_brain.source_fixture import source_fixture


async def exercise(root, service, settings):
    report = {"stage_accepted": False, "synthetic_only": True, "checks": {}, "worker_events": []}
    try:
        async with AsyncExitStack() as stack:
            fixture = await stack.enter_async_context(source_fixture(
                settings.SUPABASE_DB_URL, root, test_name="M02-native-bank-policy",
            ))
            client = HindsightClient(service["url"], service["token"], timeout=10)
            stack.push_async_callback(client.close)
            outbox = OutboxRepository(fixture.sources, retry_seconds=1, max_in_flight=1)
            worker = IngestionWorker(outbox, client)
            source = await fixture.publish("Alex's synthetic robotics project still needs both binary-search bound updates.")
            await client.ensure_bank(source.bank_name)
            async with httpx.AsyncClient(timeout=5, trust_env=False) as unauthenticated:
                denied = await unauthenticated.get(service["url"]+client._bank(source.bank_name)+"/config")
            require_check(report, "native_memory_endpoint_requires_authentication", denied.status_code in {401, 403})
            safe = await client._call("GET", client._bank(source.bank_name)+"/config")
            report["bank_policy"] = {key: safe["config"].get(key) for key in ("audit_log_enabled", "store_document_text")}
            require_check(report, "effective_bank_policy_is_visible", report["bank_policy"] == {
                "audit_log_enabled": False, "store_document_text": True,
            })
            await client._call("PATCH", client._bank(source.bank_name)+"/config",
                               body={"updates": {"audit_log_enabled": True}})
            try:
                await client.recall(source.bank_name, "Synthetic policy query")
                recall_blocked = False
            except HindsightError as error:
                recall_blocked = error.code == "hindsight_bank_policy_not_satisfied"
            require_check(report, "unsafe_bank_recall_is_blocked", recall_blocked)
            status = await worker.step(fixture.owner, "unsafe-bank-admission")
            rows = await queue_rows(fixture)
            retained = next(row for row in rows if row["operation"] == "retain")
            require_check(report, "unsafe_bank_retain_stops_before_dispatch",
                          status == "failed" and retained["dispatch_count"] == 0
                          and retained["last_error"] == "hindsight_bank_policy_not_satisfied")
            require_check(report, "unsafe_bank_has_no_original_or_retain_operation",
                          await client.document(source.bank_name, source.document_id) is None
                          and await client.operation(source.bank_name, retained["id"]) is None)
            await client._call("PATCH", client._bank(source.bank_name)+"/config",
                               body={"updates": {"audit_log_enabled": False}})
            await client.check_bank_policy(source.bank_name)
            await fixture.sources.delete_source(fixture.owner, fixture.area, source.source_id, expected_revision=1)
            report["expected_rejected_retention_id"] = str(retained["id"])
            await drain(fixture, worker, report, label="unsafe-admission-cleanup", allowed_failed_ids={retained["id"]})
            require_check(report, "rejected_undispatched_source_cleanup_completes", True)

            control = await fixture.publish(
                "Alex has a supplier meeting for the Cedar Cafe business on September 20.",
                source_key="safe-retention-control",
            )
            first_status = await worker.step(fixture.owner, "bounded-native-admission")
            async with fixture.pool.acquire() as connection:
                await connection.execute(
                    "update jarvis_memory_outbox set next_attempt_at=clock_timestamp()+interval '1 hour' "
                    "where owner_id=$1 and document_id=$2 and operation='retain'", fixture.owner, control.document_id,
                )
            waiting = await fixture.publish(
                "Alex reports that the robotics prototype still needs a wheel alignment check.",
                source_key="local-admission-backlog",
            )
            blocked = await worker.step(fixture.owner, "bounded-native-admission")
            report["bounded_admission_queue"] = await outbox.diagnostics(fixture.owner)
            waiting_row = next(row for row in await queue_rows(fixture) if row["document_id"] == waiting.document_id)
            require_check(report, "native_admission_limit_keeps_extra_source_local",
                          first_status == "accepted" and blocked is None
                          and report["bounded_admission_queue"]["native_admissions"]["reserved"] == 1
                          and waiting_row["status"] == "pending" and waiting_row["dispatch_count"] == 0
                          and await client.document(waiting.bank_name, waiting.document_id) is None)
            current = await fixture.sources.read(fixture.owner, fixture.area, waiting.source_id)
            require_check(report, "bounded_backlog_original_remains_available",
                          current.content == waiting.content and current.ingestion_status == "pending")
            await fixture.sources.delete_source(fixture.owner, fixture.area, waiting.source_id, expected_revision=1)
            cleanup_status = await worker.step(fixture.owner, "cleanup-at-capacity")
            require_check(report, "cleanup_bypasses_full_native_admission_limit",
                          cleanup_status == "done" and (await outbox.diagnostics(fixture.owner))["native_admissions"]["reserved"] == 1)
            await fixture.ready_jobs()
            await drain(fixture, worker, report, label="safe-control", allowed_failed_ids={retained["id"]})
            client.verify_document(control, await client.document(control.bank_name, control.document_id))
            require_check(report, "normal_retention_after_policy_repair", True)
            await client._call("PATCH", client._bank(control.bank_name)+"/config",
                               body={"updates": {"store_document_text": False}})
            try:
                await client.check_bank_policy(control.bank_name)
                missing_original_blocked = False
            except HindsightError as error:
                missing_original_blocked = error.code == "hindsight_bank_policy_not_satisfied"
            require_check(report, "original_storage_cannot_be_disabled_silently", missing_original_blocked)
            await client._call("PATCH", client._bank(control.bank_name)+"/config",
                               body={"updates": {"store_document_text": True}})
            await fixture.sources.delete_source(fixture.owner, fixture.area, control.source_id, expected_revision=1)
            await drain(fixture, worker, report, label="safe-control-cleanup", allowed_failed_ids={retained["id"]})
            report["transport"] = client.diagnostics()
            report["scope_limits"] = [
                "Static privacy is enforced by the checked native launcher, not exposed by the bank API.",
                "The policy test deliberately enables audit logging only in its empty isolated bank.",
                "Rejected configuration requires deliberate repair; no production data or service was changed.",
            ]
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
    finally:
        save(root/"policy-result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("/tmp/jarvis-runtime-reuse.OZIsEM/hindsight-env"))
    args = parser.parse_args()
    settings = JarvisSettings()
    with hindsight_service(args.root, runtime=args.runtime, api_key=settings.OPENAI_API_KEY) as service:
        report = asyncio.run(exercise(args.root, service, settings))
    print(json.dumps({"root": str(args.root), "checks": report["checks"], "error": report.get("error")}))
    return 0 if report["checks"] and all(report["checks"].values()) and "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
