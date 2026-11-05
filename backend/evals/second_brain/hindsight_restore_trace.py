"""Actual older Hindsight database restore against current isolated Jarvis sources."""
import argparse
import asyncio
from contextlib import AsyncExitStack
import hashlib
import json
from pathlib import Path
import secrets
import time
import traceback

import asyncpg

from app.knowledge.hindsight import HindsightClient
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.lifecycle_journal import JournalCheckpoint, SourceLifecycleJournal
from app.knowledge.outbox import OutboxRepository
from app.knowledge.provider_recovery import ProviderRecoveryService
from app.knowledge.retrieval import SourceMemoryRetrieval
from app.settings import JarvisSettings
from evals.second_brain.hindsight_lifecycle_trace import bank_snapshot, erasure_findings, quiescent
from evals.second_brain.hindsight_semantic_trace import drain
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.source_fixture import source_fixture


class RecallTraceClient(HindsightClient):
    """Observe actual native candidates without changing the recall response."""
    async def recall(self, bank, query, *, max_tokens=2400):
        result = await super().recall(bank, query, max_tokens=max_tokens)
        self.last_recall_document_ids = {
            item["document_id"] for item in result["results"]
            if isinstance(item, dict) and isinstance(item.get("document_id"), str)
        }
        return result


async def exercise(root, service):
    report = {"checks": {}, "worker_events": [], "synthetic_only": True,
              "production_database_used": False, "application_release_authorized": False,
              "full_restore_accepted": False, "scope": "Older native Hindsight database, current canonical source database"}
    started = time.monotonic()
    marker, control_marker = "ORCHID-" + secrets.token_hex(6), "LANTERN-" + secrets.token_hex(6)
    report["marker_digests"] = [hashlib.sha256(value.encode()).hexdigest() for value in (marker, control_marker)]
    try:
        # Source DB is a different database in the same owned fixture cluster.
        # The native backup/restore affects only the hindsight database.
        setup = await asyncpg.connect(service["source_database_url"], timeout=10)
        try:
            await setup.execute("create extension if not exists pgcrypto with schema public; "
                                "create extension if not exists vector with schema public")
        finally:
            await setup.close()
        async with AsyncExitStack() as stack:
            source = await stack.enter_async_context(source_fixture(
                service["source_database_url"], root / "source-manifests", test_name="native-hindsight-restore"))
            client = RecallTraceClient(service["url"], service["token"], timeout=10)
            stack.push_async_callback(client.close)
            worker = IngestionWorker(OutboxRepository(source.sources, retry_seconds=1), client)
            target = await source.publish("Alex is building " + marker + ", a private robotics project.",
                                          source_key="delete-target")
            weight = await source.publish("Alex reports lifting 80 kg on the bench press.", source_key="weight")
            control = await source.publish("Alex runs " + control_marker + ", an unrelated cafe business.",
                                           source_key="control")
            await drain(source, worker, report, label="restore-positive-controls")
            await quiescent(client, target.bank_name)
            before = await asyncio.to_thread(bank_snapshot, service["container"], target.bank_name, marker, control_marker)
            assert before["documents"]["target_rows"] and before["memory_units"]["target_rows"]
            assert before["documents"]["control_rows"]
            report["before"] = before
            report["checks"]["original_and_derived_backup_positive_controls"] = True
            await asyncio.to_thread(service["stop_owned"])
            report["native_backup"] = await asyncio.to_thread(service["snapshot_owned"])
            await asyncio.to_thread(service["start_owned"])

            await source.sources.delete_source(source.owner, source.area, target.source_id, expected_revision=1)
            corrected = await source.publish("Alex corrects the bench press report: it was 75 kg, not 80 kg.",
                                             source_key="weight", expected_revision=1)
            newer = await source.publish("Alex has a new supplier meeting on September 22.", source_key="new")
            await drain(source, worker, report, label="pre-restore-current-state")
            await quiescent(client, target.bank_name)
            assert await client.document(target.bank_name, target.document_id) is None
            client.verify_document(corrected, await client.document(corrected.bank_name, corrected.document_id))
            client.verify_document(newer, await client.document(newer.bank_name, newer.document_id))
            async with source.pool.acquire() as connection:
                receipts = [dict(row) for row in await connection.fetch(
                    "select id,document_id,operation,dispatch_count,cleanup_operation_ids "
                    "from jarvis_memory_outbox order by id")]
            report["checks"]["deletion_correction_and_new_source_really_ingested_after_backup"] = True
            save(root / "restore-result.json", report)

            await asyncio.to_thread(service["stop_owned"])
            await asyncio.to_thread(service["restore_owned"])
            await asyncio.to_thread(service["start_owned"])
            await quiescent(client, target.bank_name)
            # Positive demonstration that the old native database really returned.
            client.verify_document(target, await client.document(target.bank_name, target.document_id))
            client.verify_document(weight, await client.document(weight.bank_name, weight.document_id))
            assert await client.document(newer.bank_name, newer.document_id) is None
            assert await client.document(corrected.bank_name, corrected.document_id) is None
            restored = await asyncio.to_thread(bank_snapshot, service["container"], target.bank_name, marker, control_marker)
            assert restored["documents"]["target_rows"] and restored["memory_units"]["target_rows"]
            report["restored_before_reconciliation"] = restored
            report["checks"]["actual_backup_restore_reintroduces_old_native_documents_and_derivatives"] = True

            retrieval = SourceMemoryRetrieval(source.sources, client, timeout_seconds=30)
            guarded = await retrieval.recall(source.owner, source.area,
                                             "What are Alex's robotics project, cafe business and bench press progress?")
            assert {target.document_id, weight.document_id} <= client.last_recall_document_ids
            assert guarded["status"] == "available" and guarded["rejected_candidates"] >= 1
            assert marker not in json.dumps(guarded) and all(
                item["document_id"] not in {target.document_id, weight.document_id} for item in guarded["sources"])
            report["checks"]["source_facade_rejects_restored_deleted_and_superseded_candidates"] = True
            report["guarded_recall"] = guarded
            report["native_candidates_at_guard"] = sorted(client.last_recall_document_ids)
            journal = SourceLifecycleJournal(source.pool)
            checkpoint = await journal.checkpoint(source.owner)
            recovery = ProviderRecoveryService(source.sources)
            cursor = None
            pages = []
            while True:
                page = await recovery.prepare_page(source.owner, source.area, checkpoint=checkpoint,
                                                    after_document_id=cursor, limit=2)
                pages.append(page)
                checkpoint = JournalCheckpoint.model_validate(page["checkpoint"])
                if page["complete"]:
                    break
                cursor = page["next_after_document_id"]
            report["preparation_pages"] = pages
            worker = IngestionWorker(OutboxRepository(source.sources, retry_seconds=1), client)
            await drain(source, worker, report, label="restored-native-reconciliation")
            await quiescent(client, target.bank_name)
            after = await asyncio.to_thread(bank_snapshot, service["container"], target.bank_name, marker, control_marker)
            report["after"] = after
            report["erasure"] = erasure_findings(restored, after)
            assert report["erasure"]["positive_surfaces_cleared"]
            assert await client.document(weight.bank_name, weight.document_id) is None
            for item in (control, corrected, newer):
                client.verify_document(item, await client.document(item.bank_name, item.document_id))
            async with source.pool.acquire() as connection:
                fresh = [dict(row) for row in await connection.fetch(
                    "select id,document_id,operation,dispatch_count,cleanup_operation_ids "
                    "from jarvis_memory_outbox order by id")]
            assert [row["id"] for row in receipts] == [row["id"] for row in fresh]
            assert [row["cleanup_operation_ids"] for row in receipts] == [row["cleanup_operation_ids"] for row in fresh]
            for old, now in zip(receipts, fresh, strict=True):
                assert now["dispatch_count"] >= old["dispatch_count"]
            report["checks"].update(
                restored_positive_native_surfaces_erased=True,
                corrected_and_new_originals_rebuilt_with_control_preserved=True,
                operation_identities_and_cleanup_inventories_preserved=True,
            )
            await asyncio.to_thread(service["stop_owned"])
            await asyncio.to_thread(service["start_owned"])
            await quiescent(client, target.bank_name)
            final = await asyncio.to_thread(bank_snapshot, service["container"], target.bank_name, marker, control_marker)
            assert erasure_findings(restored, final)["positive_surfaces_cleared"]
            for item in (control, corrected, newer):
                client.verify_document(item, await client.document(item.bank_name, item.document_id))
            report["checks"]["post_reconciliation_restart_does_not_resurrect_deleted_data"] = True
            final_recall = await retrieval.recall(source.owner, source.area,
                                                  "What is Alex's latest bench press report and supplier meeting?")
            assert final_recall["status"] == "available"
            assert {corrected.document_id, newer.document_id} <= {
                item["document_id"] for item in final_recall["sources"]}
            report["checks"]["final_source_backed_recall_contains_current_correction_and_new_source"] = True
            report["final_recall"] = final_recall
            report["provider_transport"] = client.diagnostics()
        report["source_fixture_cleanup_completed"] = True
    except Exception as error:
        report["error"] = {
            "type": type(error).__name__, "code": getattr(error, "code", None),
            "stack": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                      for frame in traceback.extract_tb(error.__traceback__)],
        }
    report["seconds"] = round(time.monotonic() - started, 2)
    save(root / "restore-result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path,
                        default=Path("/tmp/jarvis-runtime-reuse.V4dY8h/hindsight-env"))
    args = parser.parse_args()
    settings = JarvisSettings()
    with hindsight_service(args.root, runtime=args.runtime, api_key=settings.OPENAI_API_KEY, max_starts=4) as service:
        report = asyncio.run(exercise(args.root, service))
    print(json.dumps({"checks": report["checks"], "error": report.get("error"), "seconds": report["seconds"],
                      "source_fixture_cleanup_completed": report.get("source_fixture_cleanup_completed"),
                      "full_restore_accepted": False}))
    return 0 if report["checks"] and all(report["checks"].values()) and "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
