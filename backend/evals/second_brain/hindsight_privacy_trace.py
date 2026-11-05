"""Native document/operation erasure experiment, not full M02 acceptance.

This uses fresh explicit native submissions to establish positive controls.
It does not replace the standard ingestion trace or bypass its acceptance gate.
Only run-owned synthetic originals and an isolated authenticated service are used.
"""
import argparse
import asyncio
from contextlib import AsyncExitStack
import hashlib
import json
from pathlib import Path
import secrets
import time
from uuid import UUID, uuid4

from app.knowledge.hindsight import HindsightClient
from app.knowledge.models import SourceError
from app.settings import JarvisSettings
from evals.second_brain.hindsight_service import hindsight_service, save
from evals.second_brain.source_fixture import source_fixture


class PrivacyProbeFailure(RuntimeError):
    pass


def require_positive(values):
    if not values or not all(value is True for value in values.values()):
        raise PrivacyProbeFailure("Required positive control or deletion barrier is unproven")


def contains(value, marker):
    return marker in json.dumps(value, ensure_ascii=True, sort_keys=True)


def operation_identity(value):
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        raise PrivacyProbeFailure("Native operation identity is invalid") from None


async def operation_with_payload(client, bank, operation_id):
    identity = operation_identity(operation_id)
    result = await client._call(
        "GET", client._bank(bank) + "/operations/" + identity + "?include_payload=true",
        missing=True,
    )
    if result is not None:
        if result.get("operation_id") != identity or result.get("status") not in {
            "pending", "processing", "completed", "failed", "cancelled", "not_found",
        }:
            raise PrivacyProbeFailure("Native operation identity or status is unrecognized")
        if result["status"] == "not_found":
            return None
    return result


async def operation_tree_with_payload(client, bank, operation_id, *, known_ids=(), max_operations=64):
    """Follow maintained child references and retain their IDs across deletion."""
    pending = [operation_identity(operation_id), *(operation_identity(item) for item in known_ids)]
    result = {}
    while pending:
        identity = pending.pop(0)
        if identity in result:
            continue
        if len(result) >= max_operations:
            raise PrivacyProbeFailure("Native operation inventory exceeds the bounded probe")
        operation = await operation_with_payload(client, bank, identity)
        result[identity] = operation
        if operation is None:
            continue
        children = operation.get("child_operations")
        if children is None:
            continue
        if not isinstance(children, list) or len(children) > max_operations:
            raise PrivacyProbeFailure("Native child operation inventory is invalid")
        for child in children:
            if not isinstance(child, dict) or not isinstance(child.get("status"), str):
                raise PrivacyProbeFailure("Native child operation reference is invalid")
            pending.append(operation_identity(child.get("operation_id")))
    return result


async def inspect_deletion(client, source, control, operation_id, marker, control_marker, delete_original):
    original = await client.document(source.bank_name, source.document_id)
    control_original = await client.document(control.bank_name, control.document_id)
    client.verify_document(source, original)
    client.verify_document(control, control_original)
    operations = await operation_tree_with_payload(client, source.bank_name, operation_id)
    payload_owners = [
        identity for identity, operation in operations.items()
        if operation is not None and contains(operation.get("task_payload"), marker)
    ]
    positive = {
        "target_document_contains_marker": contains(original, marker),
        "control_document_contains_marker": contains(control_original, control_marker),
        "target_operation_contains_marker": bool(payload_owners),
        "target_operation_completed": bool(operations) and all(
            operation is not None and operation["status"] == "completed" for operation in operations.values()
        ),
    }
    # No negative deletion claim is made without all of these positive controls.
    require_positive(positive)
    local = await delete_original()
    require_positive(local)
    await client.delete_document(source.bank_name, source.document_id)
    absent = await client.document(source.bank_name, source.document_id) is None
    remaining_control = await client.document(control.bank_name, control.document_id)
    client.verify_document(control, remaining_control)
    remaining_operations = await operation_tree_with_payload(
        client, source.bank_name, operation_id, known_ids=operations,
    )
    remaining_operation = remaining_operations[operation_identity(operation_id)]
    scope = {
        "target_document_removed": absent,
        "unrelated_control_preserved": contains(remaining_control, control_marker),
        "operation_payload_removed": not contains(remaining_operations, marker),
    }
    return {
        "positive_controls": positive, "local_barrier": local, "checks": scope,
        "operation_status_after_delete": remaining_operation.get("status") if remaining_operation else "not_found",
        "operation_payload_evidence": {
            "positive_payload_owner_ids": payload_owners,
            "checked_before": list(operations), "checked_after": list(remaining_operations),
        },
        "document_operation_scope_passed": all(scope.values()),
        "full_privacy_acceptance": False,
        "scope_limits": [
            "Exact document, accepted operation and discovered child-operation payload checks only.",
            "Not an exhaustive derived-state, unrelated-operation, log, backup or resurrection audit.",
            "Fresh direct submissions do not establish acceptance of the standard Jarvis ingestion worker.",
        ],
    }


async def await_completed(client, source, operation_id, *, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        operation = await client.operation(source.bank_name, operation_id)
        if operation is not None and operation["status"] == "completed":
            return
        if operation is not None and operation["status"] in {"failed", "cancelled"}:
            raise PrivacyProbeFailure("Native positive-control ingestion did not complete")
        await asyncio.sleep(1)
    raise TimeoutError("Native positive-control completion deadline exceeded")


async def exercise(root, service, settings, report):
    started = time.monotonic()
    marker = "ORCHID-" + secrets.token_hex(6).upper()
    control_marker = "LANTERN-" + secrets.token_hex(6).upper()
    try:
        async with AsyncExitStack() as stack:
            fixture = await stack.enter_async_context(source_fixture(
                settings.SUPABASE_DB_URL, root, test_name="native-hindsight-privacy-positive-controls",
            ))
            client = HindsightClient(service["url"], service["token"], timeout=20)
            stack.push_async_callback(client.close)
            source = await fixture.publish("The synthetic locker access code is " + marker + ".")
            control = await fixture.publish(
                "The unrelated synthetic cabinet code is " + control_marker + ".", source_key="control-original",
            )
            report["marker_digest"] = hashlib.sha256(marker.encode()).hexdigest()
            report["phase"] = "native_positive_control_ingestion"
            save(root / "privacy-result.json", report)
            await client.ensure_bank(source.bank_name)
            operations = []
            for item in (source, control):
                operation_id = uuid4()
                receipt = await client.retain(item, operation_id)
                require_positive({
                    "native_receipt_matches": receipt.get("operation_id") == str(operation_id),
                    "native_async_receipt": receipt.get("async") is True,
                })
                operations.append(operation_id)
                await await_completed(client, item, operation_id)
            report["phase"] = "positive_control_and_deletion_checks"
            report["sources"] = [
                {"source_id": str(item.source_id), "document_id": item.document_id,
                 "operation_id": str(operation), "revision": item.revision}
                for item, operation in zip((source, control), operations, strict=True)
            ]
            save(root / "privacy-result.json", report)

            async def delete_original():
                receipt = await fixture.sources.delete_source(
                    fixture.owner, fixture.area, source.source_id, expected_revision=source.revision,
                )
                unavailable = False
                try:
                    await fixture.sources.read(fixture.owner, fixture.area, source.source_id)
                except SourceError as error:
                    unavailable = error.code == "knowledge_source_unavailable"
                return {
                    "canonical_read_barrier_deleted": receipt["read_barrier"] == "deleted",
                    "canonical_original_unavailable": unavailable,
                    "vendor_cleanup_not_prematurely_complete": receipt["vendor_cleanup"] == "pending",
                }

            report.update(await inspect_deletion(
                client, source, control, operations[0], marker, control_marker, delete_original,
            ))
            report["experiment_completed"] = True
            if not report["checks"]["operation_payload_removed"]:
                report["findings"].append(
                    "The accepted native operation still contains the synthetic original after document deletion."
                )
            # Inventory supported operation routes without executing cancellation,
            # retry or other administrative mutations.
            try:
                spec = await client._call("GET", "/openapi.json")
                report["operation_endpoints"] = {
                    path: {method: data.get("summary") for method, data in methods.items()
                           if method in {"get", "put", "post", "delete", "patch"}}
                    for path, methods in spec.get("paths", {}).items() if "/operations" in path
                }
            except Exception as error:
                report["endpoint_inventory_error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
            report["transport"] = client.diagnostics()
            report["phase"] = "observed"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save(root / "privacy-result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=Path("/tmp/jarvis-runtime-reuse.OZIsEM/hindsight-env"))
    args = parser.parse_args()
    root = args.root.resolve()
    if root.parent != Path("/tmp").resolve() or not root.name.startswith("jarvis-m02-native.") or root.exists():
        parser.error("Use a new private /tmp/jarvis-m02-native.* directory")
    report = {"experiment_completed": False, "stage_accepted": False, "findings": [], "phase": "service_startup"}
    try:
        settings = JarvisSettings()
        with hindsight_service(root, runtime=args.runtime, api_key=settings.OPENAI_API_KEY) as service:
            asyncio.run(exercise(root, service, settings, report))
        report["service_cleanup_completed"] = True
    except Exception as error:
        report["service_error"] = {"type": type(error).__name__}
    finally:
        if root.exists():
            save(root / "privacy-result.json", report)
    print(json.dumps({"root": str(root), **report}))
    if not report["experiment_completed"] or report.get("error") or report.get("service_error"):
        return 1
    return 0 if report["document_operation_scope_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
