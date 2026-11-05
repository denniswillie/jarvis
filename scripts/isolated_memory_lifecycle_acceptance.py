"""Explicit live acceptance in a fresh synthetic bank; never touches demo data.

Use PYTHONPATH=backend and the backend Python environment. No Hermes prompts,
service restarts, manual model refresh, or production invalidation callbacks.
Evidence and the empty synthetic bank are deliberately retained after success.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import quote

import httpx

from app.shared_memory.client import BankScope, MENTAL_MODELS, MemoryDocument, NativeMemory
from app.shared_memory.ingestion import CanonicalCaption
from app.shared_memory.intake import CaptionPage, LiveDialogueIntake


DEMO_BANK = "jarvis-native-d9098e00decdd8e3d75810bd89ebcbdf"
GENERATION = "isolated-memory-lifecycle-v1"
QUERY = "What is this synthetic persona's only active fictional project and daily routine?"


class Evidence:
    def __init__(self, root, *, resume=False):
        self.root = root
        self.path = root / "receipts.jsonl"
        if resume:
            assert root.is_dir() and self.path.is_file(), "Missing preserved evidence"
        else:
            root.mkdir(mode=0o700, parents=True, exist_ok=False)

    def record(self, event, **fields):
        row = {"at": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        with self.path.open("a") as out:
            out.write(json.dumps(row, sort_keys=True) + "\n")

    def progress(self, stage, **fields):
        self.record("stage", stage=stage, **fields)
        print(json.dumps({"stage": stage, **fields}), flush=True)


class FixtureSource:
    def __init__(self, owner_id):
        self.owner_id, self.rows = owner_id, []

    async def high_watermark(self):
        return len(self.rows)

    async def read_after(self, cursor, *, through=None, limit=128):
        ceiling = len(self.rows) if through is None else through
        latest = {}
        for row in self.rows:
            if cursor < row.sequence <= ceiling:
                latest[(row.source_namespace, row.source_event_id)] = row
        rows = sorted(latest.values(), key=lambda row: row.sequence)[:limit]
        return CaptionPage(tuple(rows), rows[-1].sequence if len(rows) == limit else ceiling)


class RecordedMemory(NativeMemory):
    def __init__(self, *, evidence, document_id, **kwargs):
        super().__init__(**kwargs)
        self.evidence, self.document_id = evidence, document_id
        self.operation_status = {}
        self.phase = "setup"

    async def _request(self, method, path="", *, body=None, missing_ok=False):
        assert self.scope.bank_id != DEMO_BANK
        assert self.scope.owner_id.startswith("isolated-memory-lifecycle:")
        if getattr(self, "resume_only", False):
            assert method == "GET" or method == "DELETE" or (method == "POST" and path == "/memories/recall"), "Resume must not resubmit retention or refresh"
        if method == "DELETE":
            assert path == "/documents/" + quote(self.document_id, safe="")
        if method == "POST" and path == "/memories":
            assert body["async"] is True
            assert {item["document_id"] for item in body["items"]} == {self.document_id}
        try:
            result = await super()._request(method, path, body=body, missing_ok=missing_ok)
        except Exception as exc:
            self.evidence.record("request_error", phase=self.phase, method=method,
                                 path=path, error=getattr(exc, "code", type(exc).__name__))
            raise
        self.evidence.record("request", phase=self.phase, method=method, path=path,
                             body=body, response=result)
        if method == "GET" and path.startswith("/operations/") and result:
            self.operation_status[path.rsplit("/", 1)[-1]] = result.get("status")
        return result


def fixture_caption(run_id, revision, token):
    observed = datetime.now(timezone.utc) - timedelta(seconds=40)
    return CanonicalCaption(
        sequence=revision, session_id="isolated-memory-lifecycle:" + run_id,
        source_namespace=GENERATION, source_event_id="only-project-caption",
        revision=revision, kind="user_transcript", source_time=observed,
        recorded_at=observed, availability="not_finalized",
        content={"voice_route": "live", "caption_id": "only-project-caption",
                 "caption_revision": revision, "speaker": "user",
                 "text": ("This is an entirely fictional synthetic acceptance persona, not a real user. "
                          f"My only active project is named {token}. Every morning I work on "
                          f"{token} for ten minutes. If I have spare time, I choose {token} "
                          "over games because careful practice is my stated priority. "
                          "This is my complete current project and routine record.")},
    )


async def wait_operations(memory, receipts, timeout=120):
    pending = set()
    for receipt in receipts:
        pending.update(receipt.get("operation_ids", []))
        if receipt.get("operation_id"):
            pending.add(receipt["operation_id"])
    assert pending, "No vendor operation receipt: submission is not completion"
    deadline = time.monotonic() + timeout
    while pending:
        for operation_id in tuple(pending):
            result = await memory._request("GET", "/operations/" + quote(operation_id, safe=""))
            status = result.get("status")
            if status == "completed":
                pending.remove(operation_id)
            elif status in {"failed", "cancelled"}:
                raise RuntimeError("vendor_operation_" + status)
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError("vendor_operations_not_completed")
            await asyncio.sleep(2)


async def model_snapshot(memory):
    return {model_id: await memory._request("GET", "/mental-models/" + model_id)
            for model_id in MENTAL_MODELS}


def model_text(snapshot):
    return "\n".join(str(model.get("content") or "") for model in snapshot.values())


def canary_models(snapshot, *tokens):
    return [key for key, model in snapshot.items()
            if any(token.lower() in str(model.get("content") or "").lower() for token in tokens)]


async def observe_cleanup(memory, *, baseline, old, new, forgetting, timeout):
    witnesses = canary_models(baseline, old, new)
    start = time.monotonic()
    while True:
        snapshot = await model_snapshot(memory)
        # Persist native background IDs even if this observer is interrupted.
        operations = await memory._request("GET", "/operations?limit=100")
        text = model_text(snapshot).lower()
        refreshed = bool(witnesses) and all(
            snapshot[key].get("last_refreshed_at") != baseline[key].get("last_refreshed_at")
            and snapshot[key].get("last_refreshed_at")
            and snapshot[key].get("is_stale") is not True
            for key in witnesses)
        clean = old.lower() not in text and (new.lower() not in text if forgetting else new.lower() in text)
        if clean and refreshed:
            return snapshot, {"qualified": True, "elapsed_seconds": round(time.monotonic() - start, 2),
                              "witness_models": witnesses}
        if time.monotonic() - start >= timeout:
            return snapshot, {"qualified": False, "elapsed_seconds": round(time.monotonic() - start, 2),
                              "witness_models": witnesses, "old_present": old.lower() in text,
                              "new_present": new.lower() in text, "witnesses_refreshed": bool(refreshed),
                              "operation_states": [{k: op.get(k) for k in ("id", "operation_type", "status", "next_retry_at")}
                                                   for op in operations.get("operations", [])]}
        await asyncio.sleep(min(10, max(0.1, timeout - (time.monotonic() - start))))


async def recall_check(memory, old, new):
    response = await memory.recall(QUERY, max_tokens=1600, budget="low")
    # Do not confuse echoed request text or query metadata with recalled content.
    results = response.get("results")
    assert isinstance(results, list), "Unrecognized recall response shape"
    text = json.dumps(results).lower()
    return {"result_count": len(results), "old_present": old.lower() in text,
            "new_present": new.lower() in text}


async def qualify(args, evidence, report):
    scope = BankScope.for_owner("isolated-memory-lifecycle:" + args.run_id, generation=GENERATION)
    assert scope.bank_id != DEMO_BANK
    suffix = hashlib.sha256(args.run_id.encode()).hexdigest()[:10]
    old, new = "cobalt-orchid-" + suffix, "amber-willow-" + suffix
    original = fixture_caption(args.run_id, 1, old)
    corrected = fixture_caption(args.run_id, 2, new)
    replay = fixture_caption(args.run_id, 3, old)
    assert original.document_id == corrected.document_id == replay.document_id
    report.update(bank_id=scope.bank_id, owner_id=scope.owner_id,
                  document_id=original.document_id, old_token=old, new_token=new,
                  checks={}, limitations=["Synthetic source and local invalidator; not serving SQL or real Live State/reminder cleanup.",
                                         "No native Hermes prompt; no demo bank mutation; no manual mental-model refresh."])
    evidence.progress("isolated_identity", bank_id=scope.bank_id, document_id=original.document_id)
    key = args.key_file.read_text().strip()
    async with httpx.AsyncClient(timeout=45, follow_redirects=False) as http:
        memory = RecordedMemory(scope=scope, base_url=args.url, api_key=key, http_client=http,
                                timeout=45, evidence=evidence, document_id=original.document_id)
        source = FixtureSource(scope.owner_id)
        invalidations = []

        async def invalidate(forget_scope):
            assert forget_scope.owner_id == scope.owner_id and forget_scope.bank_id == scope.bank_id
            assert forget_scope.document_ids == (original.document_id,)
            invalidations.append(asdict(forget_scope))
            evidence.record("fixture_only_local_invalidation", scope=asdict(forget_scope))

        intake = LiveDialogueIntake(memory, source, args.root / "intake.sqlite3",
                                    invalidate=invalidate, operation_timeout=120, poll_seconds=1)
        try:
            await memory.create_fresh_bank()
            await intake.initialize()  # Empty baseline: no historical backfill.
            memory.phase = "retain"
            source.rows.append(original)
            retained = await intake.tick()  # Waits for actual vendor completed status.
            assert retained["retained"] == 1
            first = await recall_check(memory, old, new)
            report["checks"]["original_recall"] = first["old_present"] and not first["new_present"]
            evidence.progress("retained_and_recalled", intake=retained, recall=first,
                              operation_status=memory.operation_status)

            memory.phase = "seed_models_after_original"
            seeds = await memory.seed_mental_models()
            await wait_operations(memory, seeds.values())
            baseline = await model_snapshot(memory)
            witnesses = canary_models(baseline, old)
            report["checks"]["nonvacuous_model_baseline"] = bool(witnesses)
            evidence.progress("models_seeded", old_canary_models=witnesses,
                              operation_status=memory.operation_status)

            memory.phase = "correction"
            source.rows.append(corrected)
            corrected_receipt = await intake.tick()
            assert corrected_receipt["retained"] == 1
            document = await memory.get_document(original.document_id)
            document_text = json.dumps(document).lower()
            report["checks"]["corrected_document"] = new in document_text and old not in document_text
            corrected_recall = await recall_check(memory, old, new)
            report["checks"]["corrected_recall"] = corrected_recall["new_present"] and not corrected_recall["old_present"]
            evidence.progress("correction_completed", intake=corrected_receipt, recall=corrected_recall,
                              operation_status=memory.operation_status)
            after_correction, correction_cleanup = await observe_cleanup(
                memory, baseline=baseline, old=old, new=new, forgetting=False, timeout=args.cleanup_timeout)
            report["checks"]["correction_model_cleanup"] = correction_cleanup["qualified"]
            report["correction_cleanup"] = correction_cleanup
            evidence.progress("correction_model_cleanup", **correction_cleanup)

            memory.phase = "forget"
            forgotten = await intake.forget([original.document_id])
            evidence.record("forget_receipt", receipt=asdict(forgotten))
            report["forget_receipt"] = asdict(forgotten)
            report["checks"]["source_delete_and_fixture_invalidation"] = forgotten.requested_deletions_applied and len(invalidations) == 1
            report["checks"]["document_absent"] = await memory.get_document(original.document_id) is None
            initial_deleted_recall = await recall_check(memory, old, new)
            evidence.progress("forget_source_response", requested_deletions_applied=forgotten.requested_deletions_applied,
                              derived_cleanup=forgotten.derived_cleanup, recall=initial_deleted_recall)
            _, forget_cleanup = await observe_cleanup(
                memory, baseline=after_correction, old=old, new=new, forgetting=True, timeout=args.cleanup_timeout)
            report["checks"]["forget_model_cleanup"] = forget_cleanup["qualified"]
            report["forget_cleanup"] = forget_cleanup

            memory.phase = "tombstone_replay"
            source.rows.append(replay)
            replayed = await intake.tick()
            report["checks"]["tombstone_suppresses_replay"] = replayed["retained"] == 0 and await memory.get_document(original.document_id) is None
            final_recall = await recall_check(memory, old, new)
            report["checks"]["forgotten_recall"] = not final_recall["old_present"] and not final_recall["new_present"]
            report["operation_status"] = memory.operation_status
            report["qualified"] = all(report["checks"].values())
            evidence.progress("finished", qualified=report["qualified"], checks=report["checks"],
                              final_recall=final_recall, forget_model_cleanup=forget_cleanup)
        finally:
            intake.abort()  # Preserve checkpoint/receipts; never submit implicit close work.


async def resume_qualification(args, evidence, report):
    """Continue this checkpoint only; never reconstruct remote retained state."""
    rows = [json.loads(line) for line in evidence.path.read_text().splitlines() if line.strip()]
    inspection = json.loads((args.root / "resume-inspection.json").read_text())
    scope = BankScope.for_owner("isolated-memory-lifecycle:" + args.run_id, generation=GENERATION)
    identity = next(row for row in rows if row.get("stage") == "isolated_identity")
    suffix = hashlib.sha256(args.run_id.encode()).hexdigest()[:10]
    old, new = "cobalt-orchid-" + suffix, "amber-willow-" + suffix
    original, corrected = fixture_caption(args.run_id, 1, old), fixture_caption(args.run_id, 2, new)
    assert scope.bank_id == identity["bank_id"] == inspection["identity"]["bank_id"]
    assert original.document_id == corrected.document_id == identity["document_id"]
    assert not any(row.get("method") == "DELETE" for row in rows), "Forget already attempted; reconcile rather than repeat"
    db = sqlite3.connect("file:" + str(args.root / "intake.sqlite3") + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        checkpoint = dict(db.execute("SELECT * FROM checkpoint WHERE id=1").fetchone())
        assert checkpoint["sequence"] == 2 and checkpoint["pending_operation"] is None
        assert checkpoint["owner_id"] == scope.owner_id and checkpoint["bank_id"] == scope.bank_id
        assert db.execute("SELECT count(*) FROM forgotten").fetchone()[0] == 0
    finally:
        db.close()
    first = next(row["recall"] for row in rows if row.get("stage") == "retained_and_recalled")
    previous_correction = next(row["recall"] for row in rows if row.get("stage") == "correction_completed")
    baseline = {row["path"].rsplit("/", 1)[-1]: row["response"] for row in rows
                if row.get("phase") == "seed_models_after_original" and row.get("method") == "GET"
                and row.get("path", "").startswith("/mental-models/") and row.get("response")}
    assert set(baseline) == set(MENTAL_MODELS)
    report.update(bank_id=scope.bank_id, document_id=original.document_id, resumed=True,
                  checks={"original_recall": first["old_present"] and not first["new_present"],
                          "original_correction_recall": previous_correction["new_present"] and not previous_correction["old_present"],
                          "nonvacuous_model_baseline": len(canary_models(baseline, old)) == 3},
                  limitations=["Synthetic source and invalidation callback; no serving SQL or real local projection cleanup.",
                               "Existing bank only; remote retain and refresh submissions are forbidden in resume mode."])
    async with httpx.AsyncClient(timeout=45, follow_redirects=False) as http:
        memory = RecordedMemory(scope=scope, base_url=args.url, api_key=args.key_file.read_text().strip(),
                                http_client=http, timeout=45, evidence=evidence, document_id=original.document_id)
        memory.resume_only = True
        memory.phase = "resume_saved_refresh"
        refresh_ids = list(inspection["refreshes"])
        assert refresh_ids, "No saved refresh IDs"
        try:
            await wait_operations(memory, [{"operation_ids": refresh_ids}], timeout=150)
            report["checks"]["correction_refresh_operations_completed"] = True
        except (TimeoutError, RuntimeError) as exc:
            report["checks"]["correction_refresh_operations_completed"] = False
            report["correction_refresh_outcome"] = type(exc).__name__
        after_correction, cleanup = await observe_cleanup(
            memory, baseline=baseline, old=old, new=new, forgetting=False, timeout=1)
        report["correction_cleanup"] = cleanup
        report["checks"]["correction_model_cleanup"] = cleanup["qualified"]
        document = await memory.get_document(original.document_id)
        assert document is not None, "Source already absent; reconcile before further deletion"
        document_text = json.dumps(document).lower()
        report["checks"]["corrected_document"] = new in document_text and old not in document_text
        recalled = await recall_check(memory, old, new)
        report["checks"]["corrected_recall"] = recalled["new_present"] and not recalled["old_present"]
        evidence.progress("resumed_correction_refresh", cleanup=cleanup, recall=recalled,
                          operation_status=memory.operation_status)
        source = FixtureSource(scope.owner_id)
        source.rows = [original, corrected]
        invalidations = []

        async def invalidate(forget_scope):
            assert forget_scope.owner_id == scope.owner_id and forget_scope.bank_id == scope.bank_id
            assert forget_scope.document_ids == (original.document_id,)
            invalidations.append(asdict(forget_scope))
            evidence.record("fixture_only_local_invalidation", scope=asdict(forget_scope))

        intake = LiveDialogueIntake(memory, source, args.root / "intake.sqlite3", invalidate=invalidate,
                                    operation_timeout=120, poll_seconds=1)
        try:
            await intake.initialize()
            memory.phase = "resumed_forget"
            forgotten = await intake.forget([original.document_id])
            evidence.record("forget_receipt", receipt=asdict(forgotten))
            report["forget_receipt"] = asdict(forgotten)
            report["checks"]["source_delete_and_fixture_invalidation"] = forgotten.requested_deletions_applied and len(invalidations) == 1
            report["checks"]["document_absent"] = await memory.get_document(original.document_id) is None
            await memory._request("GET", "/operations?limit=100")
            early_recall = await recall_check(memory, old, new)
            evidence.progress("resumed_forget_source_response", requested_deletions_applied=forgotten.requested_deletions_applied,
                              derived_cleanup=forgotten.derived_cleanup, recall=early_recall)
            _, cleanup = await observe_cleanup(memory, baseline=after_correction, old=old, new=new,
                                               forgetting=True, timeout=args.cleanup_timeout)
            report["forget_cleanup"] = cleanup
            report["checks"]["forget_model_cleanup"] = cleanup["qualified"]
            report["checks"]["durable_tombstone_suppresses_replay"] = False
            if forgotten.requested_deletions_applied:
                intake.abort()
                source.rows.append(fixture_caption(args.run_id, 3, old))
                memory.phase = "resumed_durable_tombstone_replay"
                intake = LiveDialogueIntake(memory, source, args.root / "intake.sqlite3", invalidate=invalidate,
                                            operation_timeout=120, poll_seconds=1)
                replay = await intake.tick()
                report["replay_receipt"] = replay
                report["checks"]["durable_tombstone_suppresses_replay"] = replay["retained"] == 0 and await memory.get_document(original.document_id) is None
            final_recall = await recall_check(memory, old, new)
            report["checks"]["forgotten_recall"] = not final_recall["old_present"] and not final_recall["new_present"]
            operations = await memory._request("GET", "/operations?limit=100")
            report["final_operations"] = [{k: op.get(k) for k in ("id", "operation_type", "status", "next_retry_at")}
                                          for op in operations.get("operations", [])]
            report["operation_status"] = memory.operation_status
            report["qualified"] = all(report["checks"].values())
            evidence.progress("resumed_finished", qualified=report["qualified"], checks=report["checks"],
                              final_recall=final_recall, forget_model_cleanup=cleanup)
        finally:
            intake.abort()


async def qualify_real_callback(args, evidence, report):
    """One real tool/callback invocation against fresh owner-scoped SQL rows."""
    from types import SimpleNamespace

    from app.db import create_pool
    from app.live_state.companion import CompanionState, STATE_KEY
    from app.live_state.companion_runtime import initialize_companion, connect_companion_tools
    from app.perception.repository import PerceptionRepository
    from app.reminders.repository import ReminderRepository
    from app.settings import JarvisSettings
    from app.shared_memory.configuration import initialize_shared_memory, close_shared_memory

    owner = "isolated-memory-callback:" + args.run_id
    generation = "isolated-memory-callback-v1"
    scope = BankScope.for_owner(owner, generation=generation)
    assert scope.bank_id != DEMO_BANK
    digest = hashlib.sha256(owner.encode()).hexdigest()
    document_id = "jarvis-callback-" + digest
    other_document_id = "jarvis-callback-unrelated-" + digest
    token = "violet-lantern-" + digest[:10]
    fact = "Synthetic callback fixture: the fictional observer carries a " + token + "."
    report.update(owner_id=owner, bank_id=scope.bank_id, document_id=document_id,
                  checks={}, limitations=["Direct real tool/callback and repositories; no voice or Hermes prompt.",
                                          "Reminder removal is persisted soft deletion before fixture cleanup."])
    evidence.progress("real_callback_identity", owner_id=owner, bank_id=scope.bank_id, document_id=document_id)
    env = {"JARVIS_COMPANION_STATE_ENABLED": "true", "JARVIS_NATIVE_MEMORY_ENABLED": "true",
           "JARVIS_HINDSIGHT_BANK_GENERATION": generation, "JARVIS_HINDSIGHT_URL": args.url,
           "JARVIS_HINDSIGHT_API_KEY_FILE": str(args.key_file.resolve()),
           "JARVIS_NATIVE_MEMORY_CHECKPOINT_FILE": str((args.root / "intake.sqlite3").resolve())}
    settings = JarvisSettings(_env_file=Path(__file__).resolve().parents[1] / "backend" / ".env",
                              JARVIS_OWNER_ID=owner, SUPABASE_DB_POOL_MIN_SIZE=1, SUPABASE_DB_POOL_MAX_SIZE=2)
    pool = await create_pool(settings)
    services = SimpleNamespace(settings=settings, db_pool=pool, realtime_session_manager=None,
                               shared_memory=None, shared_memory_intake=None, shared_memory_task=None,
                               shared_memory_stop=None, shared_memory_invalidator=None, shared_memory_issue=None)
    owns_rows = bank_created = False
    memory = None
    tables = ("jarvis_owner_working_memory", "jarvis_reminders", "jarvis_attention_visual_subjects")

    async def counts(connection):
        return {table: await connection.fetchval("SELECT count(*) FROM " + table + " WHERE owner_id=$1", owner)
                for table in tables}

    async def drain_operations(timeout=90):
        deadline, quiet = time.monotonic() + timeout, 0
        while True:
            listing = await memory._request("GET", "/operations?limit=100")
            operations = listing.get("operations", [])
            statuses = {op["id"]: op.get("status") for op in operations}
            report["vendor_operation_status"] = statuses
            if statuses and all(value == "completed" for value in statuses.values()):
                quiet += 1
                if quiet == 2:
                    return True
            else:
                quiet = 0
            if any(value in {"failed", "cancelled"} for value in statuses.values()) or time.monotonic() >= deadline:
                return False
            await asyncio.sleep(2)

    try:
        async with pool.acquire() as connection:
            before_counts = await counts(connection)
            assert all(value == 0 for value in before_counts.values()), "Synthetic owner already exists"
        owns_rows = True
        await initialize_companion(services, environ=env)
        await initialize_shared_memory(services, environ=env)
        connect_companion_tools(services)
        state, memory = services.companion_state, services.shared_memory
        assert memory.scope == scope and services.shared_memory_invalidator.__self__ is state
        assert services.shared_memory_invalidator.__func__ is CompanionState.invalidate_memory
        original_request = memory._request

        async def recorded_request(method, path="", *, body=None, missing_ok=False):
            assert memory.scope == scope and scope.bank_id != DEMO_BANK
            if method == "DELETE":
                assert path == "/documents/" + quote(document_id, safe="")
            if method == "POST" and path == "/memories":
                assert {item["document_id"] for item in body["items"]} == {document_id}
            response = await original_request(method, path, body=body, missing_ok=missing_ok)
            evidence.record("real_callback_native_request", method=method, path=path, body=body, response=response)
            return response

        memory._request = recorded_request  # Passive receipt capture; actual configured client executes every call.
        await memory.create_fresh_bank()
        bank_created = True
        accepted = await memory.retain([MemoryDocument(document_id, fact, datetime.now(timezone.utc),
                                        context="Isolated real-callback acceptance; fictional data only.",
                                        metadata={"fixture": args.run_id})], asynchronous=True)
        evidence.progress("real_callback_retain_submitted", receipt=accepted)
        await wait_operations(memory, [accepted], timeout=120)
        report["checks"]["retain_completed"] = True
        assert await memory.get_document(document_id) is not None
        generation_before = state.read()["human_generation"]
        assert await state.update_human({"fact": fact}, observed_at=datetime.now(timezone.utc),
                                        source_document_ids=(document_id,), generation=generation_before)
        linked = await state.manage_reminder(action="create", title="Synthetic linked lantern reminder",
                    condition="Only during this isolated fixture", source_document_ids=(document_id,))
        evidence.record("real_callback_linked_reminder_created", reminder=linked)
        unrelated = await state.manage_reminder(action="create", title="Synthetic unrelated control reminder",
                    condition="Only during a separate fictional fixture", source_document_ids=(other_document_id,))
        evidence.record("real_callback_unrelated_reminder_created", reminder=unrelated)
        report["reminder_ids"] = {"linked": linked["id"], "unrelated": unrelated["id"]}
        persisted_before = await state.working_memory.get_owner_working_memory(owner_id=owner)
        before_reminders = await state.reminders.list_reminders(owner_id=owner, status=None, limit=8)
        evidence.record("real_callback_persisted_before", revision=persisted_before.revision,
                        state=persisted_before.state, reminders=[row.model_dump(mode="json") for row in before_reminders])
        report["checks"]["persisted_source_linked_human"] = (
            persisted_before.state[STATE_KEY]["human"]["source_document_ids"] == [document_id]
            and persisted_before.state[STATE_KEY]["human"]["content"] == {"fact": fact})
        report["checks"]["both_reminders_initially_active"] = len(before_reminders) == 2 and all(row.status == "active" for row in before_reminders)

        # Empty explicit reminder IDs force source matching through the real invalidator.
        result = await services.companion_tools.invoke("forget_memory",
                    {"document_ids": [document_id], "reminder_ids": []}, owner_id=owner)
        evidence.record("real_callback_forget_tool_receipt", receipt=result)
        report["tool_receipt"] = result
        report["checks"]["real_tool_applied"] = result["status"] == "applied" and result["local_invalidated"]
        fresh = CompanionState(owner_id=owner, working_memory=PerceptionRepository(pool), reminders=ReminderRepository(pool))
        await fresh.initialize()
        persisted_after = await fresh.working_memory.get_owner_working_memory(owner_id=owner)
        after_reminders = await fresh.reminders.list_reminders(owner_id=owner, status=None, limit=8)
        after_by_id = {str(row.id): row.model_dump(mode="json") for row in after_reminders}
        evidence.record("real_callback_persisted_after", revision=persisted_after.revision,
                        state=persisted_after.state, reminders=list(after_by_id.values()))
        section = persisted_after.state[STATE_KEY]
        report["checks"]["persisted_human_invalidated"] = section["human"] is None
        report["checks"]["persisted_generation_advanced"] = section["human_generation"] == generation_before + 1
        report["checks"]["matching_reminder_persisted_deleted"] = after_by_id[linked["id"]]["status"] == "deleted"
        report["checks"]["unrelated_reminder_unchanged"] = after_by_id[unrelated["id"]] == unrelated
        report["checks"]["fresh_state_only_unrelated_active"] = [row["id"] for row in fresh.read()["system"]["reminders"]] == [unrelated["id"]]
        report["checks"]["stale_human_cannot_resurrect"] = not await fresh.update_human(
            {"fact": fact}, observed_at=datetime.now(timezone.utc), source_document_ids=(document_id,), generation=generation_before)
        report["checks"]["source_document_absent"] = await memory.get_document(document_id) is None
        report["checks"]["vendor_operations_completed"] = await drain_operations()
        recalled = await memory.recall("What does the synthetic callback observer carry?", max_tokens=1000)
        report["checks"]["forgotten_content_not_recalled"] = not recalled.get("results")
        evidence.progress("real_callback_persisted_verdict", checks=report["checks"],
                          derived_cleanup=result["derived_cleanup"], vendor_operation_status=report["vendor_operation_status"])
    finally:
        cleanup = {"owner_rows_removed": False, "fixture_document_absent": not bank_created}
        try:
            if bank_created and memory is not None:
                existing = await memory.get_document(document_id)
                if existing is not None:
                    await memory._request("DELETE", "/documents/" + quote(document_id, safe=""), missing_ok=True)
                    cleanup["vendor_cleanup_completed"] = await drain_operations()
                cleanup["fixture_document_absent"] = await memory.get_document(document_id) is None
        except Exception as exc:
            cleanup["native_cleanup_error"] = type(exc).__name__
        try:
            await close_shared_memory(services)
        except Exception as exc:
            cleanup["intake_close_error"] = type(exc).__name__
        try:
            if owns_rows:
                async with pool.acquire() as connection, connection.transaction():
                    cleanup["before"] = await counts(connection)
                    # No sessions, deliveries or occurrences are created by this fixture.
                    cleanup["deleted_reminders"] = await connection.execute("DELETE FROM jarvis_reminders WHERE owner_id=$1", owner)
                    cleanup["deleted_attention_rows"] = await connection.execute("DELETE FROM jarvis_attention_visual_subjects WHERE owner_id=$1", owner)
                    cleanup["deleted_working_memory"] = await connection.execute("DELETE FROM jarvis_owner_working_memory WHERE owner_id=$1", owner)
                    cleanup["after"] = await counts(connection)
                    cleanup["owner_rows_removed"] = all(value == 0 for value in cleanup["after"].values())
        except Exception as exc:
            cleanup["sql_cleanup_error"] = type(exc).__name__
        finally:
            await pool.close()
        report["cleanup"] = cleanup
        evidence.progress("real_callback_cleanup", **cleanup)
    report["qualified"] = bool(report["checks"]) and all(report["checks"].values()) and cleanup["owner_rows_removed"] and cleanup["fixture_document_absent"]
    evidence.progress("real_callback_finished", qualified=report["qualified"], checks=report["checks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8888")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--cleanup-timeout", type=float, default=360)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--real-callback", action="store_true")
    args = parser.parse_args()
    if args.resume and args.real_callback:
        parser.error("The real callback acceptance requires a fresh isolated owner")
    if not re.fullmatch(r"[a-z0-9-]{8,80}", args.run_id):
        parser.error("Use a unique deterministic lowercase run label")
    if args.url != "http://127.0.0.1:8888" or not 1 <= args.cleanup_timeout <= 360:
        parser.error("This acceptance is bounded to the local vendor and at most 360 seconds per cleanup")
    os.umask(0o077)
    evidence = Evidence(args.root, resume=args.resume)
    report = {"run_id": args.run_id, "qualified": False}
    try:
        run = qualify_real_callback if args.real_callback else resume_qualification if args.resume else qualify
        asyncio.run(asyncio.wait_for(run(args, evidence, report), timeout=1080))
    except Exception as exc:
        report["error"] = getattr(exc, "code", type(exc).__name__)
        evidence.progress("stopped", error=report["error"])
    finally:
        name = "real-callback-report.json" if args.real_callback else "resume-report.json" if args.resume else "report.json"
        (args.root / name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
