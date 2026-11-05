"""Native M02 source-backed ingestion trace, with explicit acceptance gaps."""
import argparse
import asyncio
from contextlib import AsyncExitStack
import json
from pathlib import Path
import time

import httpx

from app.knowledge.hindsight import HindsightClient, HindsightError
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.outbox import OutboxRepository
from app.knowledge.retrieval import SourceMemoryRetrieval
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.memory.tools import create_memory_tools
from app.settings import JarvisSettings
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.hindsight_privacy_trace import contains, operation_tree_with_payload
from evals.second_brain.source_fixture import source_fixture


PHRASE="COBALT-LANTERN-583"


class NativeTraceFailure(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def require_check(report, name, passed):
    report["checks"][name] = passed is True
    if passed is not True:
        raise NativeTraceFailure(name)


class LostCleanupAcknowledgementClient(HindsightClient):
    """Fault injection after one real native operation deletion succeeds."""
    cleanup_ack_lost = False

    async def delete_terminal_operation(self, bank, operation_id):
        deleted = await super().delete_terminal_operation(bank, operation_id)
        if deleted and not self.cleanup_ack_lost:
            self.cleanup_ack_lost = True
            raise HindsightError("native_fixture_lost_cleanup_ack", retryable=True)
        return deleted


async def exercise(root, configuration, settings, *, lose_cleanup_ack=False):
    report={"stage_accepted":False,"checks":{},"findings":[],"events":[]}
    started=time.monotonic()
    try:
        async with AsyncExitStack() as stack:
            f=await stack.enter_async_context(source_fixture(settings.SUPABASE_DB_URL,root,test_name="native-hindsight-trace"))
            client_type = LostCleanupAcknowledgementClient if lose_cleanup_ack else HindsightClient
            client=client_type(configuration["url"],configuration["token"],timeout=15)
            stack.push_async_callback(client.close)
            source=await f.publish("The fictional vault phrase is "+PHRASE+". This is a synthetic user report.")
            outbox=OutboxRepository(f.sources,retry_seconds=1)
            worker=IngestionWorker(outbox,client)
            async def step():
                status=await worker.step(f.owner,"native-fixture")
                report["events"].append({"seconds":time.monotonic()-started,"status":status})
                save(root/"result.json",report)
                return status
            status=await step()
            async with f.pool.acquire() as connection:
                receipt=await connection.fetchrow(
                    "select id,status,last_error,failure_count from jarvis_memory_outbox "
                    "where owner_id=$1 and document_id=$2 and operation='retain'", f.owner, source.document_id)
            report["outbox_receipt"] = (
                {"operation_id": str(receipt["id"]), "status": receipt["status"],
                 "last_error": receipt["last_error"], "failure_count": receipt["failure_count"]}
                if receipt is not None else None
            )
            require_check(report, "async_accepted_not_searchable",
                          status == "accepted" and receipt is not None and receipt["status"] == "accepted")
            operation_id=receipt["id"]
            replay=await client.retain(source,operation_id)
            require_check(report, "same_operation_replay", replay["operation_id"]==str(operation_id))
            deadline=time.monotonic()+240
            while status not in {"searchable","failed","cancelled"} and time.monotonic()<deadline:
                await asyncio.sleep(1.1)
                status=await step()
            require_check(report, "native_ingestion_complete", status=="searchable")
            retrieval=SourceMemoryRetrieval(f.sources,client,timeout_seconds=30)
            backend = SourceMemoryReadBackend(f.sources, retrieval, owner_id=f.owner, area_id=f.area)
            tools = {tool.name: tool for tool in create_memory_tools(owner_id=f.owner, read_backend=backend)}
            recall_started = time.monotonic()
            answer = await tools["search_memory"].handler({"query": "What is the fictional vault phrase?"})
            report["recall_seconds"] = time.monotonic() - recall_started
            require_check(report, "shared_search_handler", answer["status"] == "success")
            report["checks"]["source_backed_recall"]=any(
                PHRASE in json.dumps(s.get("content")) and s["trust"]=="user_input" for s in answer["results"])
            report["checks"]["source_identity"]=any(
                s["source_id"]==str(source.source_id) and s["memory_id"]==str(source.source_id)
                for s in answer["results"])
            expansion_started = time.monotonic()
            original = await tools["explain_memory"].handler({"memory_id": str(source.source_id)})
            report["expansion_seconds"] = time.monotonic() - expansion_started
            require_check(report, "shared_original_expansion",
                          original["status"] == "success"
                          and json.loads(original["content_page"]["text"]) == source.content
                          and original["source"]["content_digest"] == source.content_digest
                          and original["source"]["source_time"] == source.model_dump(mode="json")["source_time"]
                          and original["source"]["trust"] == "user_input"
                          and original["provider_contacted"] is False)
            async with httpx.AsyncClient(timeout=10,trust_env=False) as unauthenticated:
                response=await unauthenticated.get(configuration["url"]+"/v1/default/banks/"+source.bank_name+"/documents/"+source.document_id)
                report["checks"]["authentication_required"]=response.status_code in {401,403}
            operations=await operation_tree_with_payload(client,source.bank_name,operation_id)
            require_check(report, "operation_payload_positive_control", any(
                operation is not None and contains(operation.get("task_payload"), PHRASE)
                for operation in operations.values()
            ))
            report["operation_ids_checked_before_delete"]=list(operations)
            await f.sources.delete_source(f.owner,f.area,source.source_id,expected_revision=1)
            status=await step()
            deadline=time.monotonic()+90
            while status not in {"done","failed"} and time.monotonic()<deadline:
                await asyncio.sleep(1.1)
                status=await step()
            report["checks"]["native_document_removed"]=status=="done" and await client.document(source.bank_name,source.document_id) is None
            if lose_cleanup_ack:
                report["checks"]["lost_cleanup_ack_recovered"] = client.cleanup_ack_lost and status == "done"
            async with f.pool.acquire() as connection:
                cleanup = await connection.fetchrow(
                    "select status,cleanup_operation_ids,last_error from jarvis_memory_outbox "
                    "where owner_id=$1 and document_id=$2 and operation='delete'", f.owner, source.document_id,
                )
            report["cleanup_receipt"] = {
                "status": cleanup["status"],
                "operation_ids": [str(identity) for identity in cleanup["cleanup_operation_ids"]]
                                 if cleanup["cleanup_operation_ids"] is not None else None,
                "last_error": cleanup["last_error"],
            }
            report["checks"]["durable_cleanup_inventory"] = (
                cleanup["cleanup_operation_ids"] is not None
                and set(report["cleanup_receipt"]["operation_ids"]) == set(operations)
            )
            recalled = await tools["search_memory"].handler({"query": "What is the fictional vault phrase?"})
            report["checks"]["canonical_read_barrier"] = (
                recalled["status"] == "success" and not recalled["results"])
            denied = await tools["explain_memory"].handler({"memory_id": str(source.source_id)})
            report["checks"]["shared_expansion_delete_barrier"] = denied["status"] == "failed"
            remaining_operations=await operation_tree_with_payload(
                client,source.bank_name,operation_id,known_ids=operations)
            report["operation_ids_checked_after_delete"]=list(remaining_operations)
            report["checks"]["operation_payload_removed"]=not contains(remaining_operations,PHRASE)
            if not report["checks"]["operation_payload_removed"]:
                report["findings"].append("Native operation history still contains the original after document deletion.")
            report["diagnostics"]=client.diagnostics()
            report["source"]={"source_id":str(source.source_id),"document_id":source.document_id,"revision":source.revision,
                              "operation_id":str(operation_id),"bank_name":source.bank_name}
    except Exception as error:
        report["error"]={"type":type(error).__name__,"code":getattr(error,"code",None)}
    finally:
        report["elapsed_seconds"]=time.monotonic()-started
        save(root/"result.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--lose-cleanup-ack", action="store_true", help="Lose one acknowledgement after a real native deletion")
    parser.add_argument("--runtime",type=Path,default=Path("/tmp/jarvis-runtime-reuse.OZIsEM/hindsight-env"))
    args=parser.parse_args()
    settings=JarvisSettings()
    with hindsight_service(args.root,runtime=args.runtime,api_key=settings.OPENAI_API_KEY) as configuration:
        report=asyncio.run(exercise(args.root,configuration,settings,lose_cleanup_ack=args.lose_cleanup_ack))
    print(json.dumps({"root":str(args.root),"checks":report["checks"],"findings":report["findings"],"error":report.get("error")}))
    return 0 if report["checks"] and all(report["checks"].values()) and "error" not in report else 1


if __name__=="__main__":
    raise SystemExit(main())
