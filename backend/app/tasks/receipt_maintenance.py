"""Trusted operator maintenance for historical task receipt copies.

No provider calls, action replay or automatic application release occurs here.
Keep writers quiescent for a complete multi-page migration inventory.
"""
from __future__ import annotations

from uuid import UUID

from app.tasks.models import TaskError, digest, json_object


class ReceiptCopyMaintenance:
    def __init__(self, tasks):
        self.tasks = tasks

    @staticmethod
    def _bounds(owner_id, after_id, limit):
        if not isinstance(owner_id, str) or not owner_id or owner_id != owner_id.strip() or len(owner_id) > 200:
            raise ValueError("An exact owner is required")
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("Receipt maintenance pages must contain 1 to 128 rows")
        if after_id is not None and not isinstance(after_id, UUID):
            raise ValueError("An exact UUID cursor is required")

    async def _page(self, connection, owner_id, after_action_id, limit, *, write=False):
        if after_action_id is not None and not await connection.fetchval(
            "select exists(select 1 from jarvis_task_actions where owner_id=$1 and id=$2)", owner_id, after_action_id,
        ):
            raise TaskError("receipt_cursor_unavailable", "The action cursor is not available for this owner.")
        rows = await connection.fetch("""select a.* from jarvis_task_actions a
            where owner_id=$1 and ($2::uuid is null or id>$2) order by id limit $3""" +
            (" for update of a" if write else " for share of a"), owner_id, after_action_id, limit + 1)
        items, retained = [], []
        for action in rows[:limit]:
            item = {
                "action_id":str(action["id"]), "task_id":str(action["task_id"]), "state":action["status"],
                "parameters_payload_id":str(action["parameters_payload_id"]),
                "receipt_payload_id":str(action["receipt_payload_id"]) if action["receipt_payload_id"] else None,
                "receipt_digest":action["receipt_digest"],
                "identity_digest":digest({key:str(action[key]) for key in
                    ("id", "owner_id", "task_id", "task_revision", "operation_key", "grant_id", "submission_token")}),
                "issues":[], "operation":"no_receipt", "fingerprint_needed":False,
            }
            body = None
            if action["receipt_payload_id"] is not None:
                source = await connection.fetchrow("""select id,availability,created_at from jarvis_context_payloads
                    where owner_id=$1 and id=$2 for share""", owner_id, action["parameters_payload_id"])
                source_state = await connection.fetchval("select lock_jarvis_copy_source($1,$2)",
                                                         owner_id, action["parameters_payload_id"])
                body = await connection.fetchrow("""select id,availability,content,content_digest,created_at
                    from jarvis_context_payloads where owner_id=$1 and id=$2""" +
                    (" for update" if write else " for share"), owner_id, action["receipt_payload_id"])
                linked = await connection.fetchval("""select exists(select 1 from jarvis_context_payload_dependencies
                    where owner_id=$1 and source_payload_id=$2 and dependent_payload_id=$3)""",
                    owner_id, action["parameters_payload_id"], action["receipt_payload_id"])
                item.update(source_state=source_state, linked=linked,
                            receipt_availability=body["availability"] if body else None,
                            content_digest=body["content_digest"] if body else None)
                if source is None or body is None:
                    item["issues"].append("missing_owned_payload")
                elif body["availability"] == "not_finalized":
                    item["issues"].append("receipt_not_finalized")
                elif body["availability"] != "available":
                    item["operation"] = "already_removed"
                else:
                    fingerprint = digest(json_object(body["content"]))
                    if action["receipt_digest"] is not None and action["receipt_digest"] != fingerprint:
                        item["issues"].append("receipt_fingerprint_conflict")
                    item["fingerprint_needed"] = action["receipt_digest"] is None
                    if source_state == "available":
                        if not linked and (source["created_at"], source["id"]) >= (body["created_at"], body["id"]):
                            item["issues"].append("source_copy_order_unproved")
                        item["operation"] = "unchanged" if linked else "bind"
                    elif source_state in {"removed", "revoked"}:
                        item["operation"] = "redact"
                    else:
                        item["issues"].append("source_state_unproved")
                if item["issues"]:
                    item["operation"] = "unresolved"
            items.append(item)
            retained.append((action, body, item))
        page = {
            "format":"jarvis.task_receipt_audit.v1", "owner_id":owner_id,
            "after_action_id":str(after_action_id) if after_action_id else None, "limit":limit,
            "items":items, "has_more":len(rows) > limit,
            "next_after_action_id":str(rows[limit - 1]["id"]) if len(rows) > limit else None,
            "migration_accepted":False, "application_release_authorized":False,
        }
        page["snapshot_digest"] = digest(page)
        return page, retained

    async def audit_page(self, owner_id, *, after_action_id=None, limit=64):
        self._bounds(owner_id, after_action_id, limit)
        async with self.tasks.transaction(owner_id) as connection:
            page, _ = await self._page(connection, owner_id, after_action_id, limit)
            return page

    async def repair_page(self, owner_id, *, expected_digest, after_action_id=None, limit=64):
        self._bounds(owner_id, after_action_id, limit)
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ValueError("An audit snapshot digest is required")
        async with self.tasks.transaction(owner_id) as connection:
            page, retained = await self._page(connection, owner_id, after_action_id, limit, write=True)
            if page["snapshot_digest"] != expected_digest:
                raise TaskError("receipt_audit_stale", "Receipt inventory changed; audit this page again before repairing.")
            if any(item["issues"] for item in page["items"]):
                raise TaskError("receipt_inventory_unresolved", "This page contains unproved receipt mappings; no repair was applied.")
            counts = {"linked":0, "redacted":0, "fingerprints_added":0, "already_removed":0, "unchanged":0, "no_receipt":0}
            for action, body, item in retained:
                operation = item["operation"]
                if item["fingerprint_needed"]:
                    await connection.execute("update jarvis_task_actions set receipt_digest=$3 where owner_id=$1 and id=$2",
                                             owner_id, action["id"], digest(json_object(body["content"])))
                    counts["fingerprints_added"] += 1
                if operation == "bind":
                    await connection.execute("""insert into jarvis_context_payload_dependencies
                        (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
                        owner_id, action["parameters_payload_id"], action["receipt_payload_id"])
                    counts["linked"] += 1
                elif operation == "redact":
                    await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                        where owner_id=$1 and id=$2""", owner_id, action["receipt_payload_id"])
                    counts["redacted"] += 1
                else:
                    counts[operation] += 1
            return {
                "format":"jarvis.task_receipt_repair.v1", "owner_id":owner_id,
                "audit_snapshot_digest":expected_digest, "scanned":len(retained), **counts,
                "has_more":page["has_more"], "next_after_action_id":page["next_after_action_id"],
                "migration_accepted":False, "application_release_authorized":False,
            }

    async def unbound_history_page(self, owner_id, *, after_event_id=None, limit=64):
        """Report retained receipt events without a current pointer or explicit source edge."""
        self._bounds(owner_id, after_event_id, limit)
        async with self.tasks.read_transaction() as connection:
            if after_event_id is not None and not await connection.fetchval(
                "select exists(select 1 from jarvis_task_events where owner_id=$1 and id=$2)", owner_id, after_event_id,
            ):
                raise TaskError("receipt_cursor_unavailable", "The history cursor is not available for this owner.")
            rows = await connection.fetch("""select e.id,e.task_id,e.task_revision,e.event_type,e.payload_id
                from jarvis_task_events e join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
                where e.owner_id=$1 and ($2::uuid is null or e.id>$2) and p.availability='available'
                  and e.event_type in ('action_accepted','action_rejected','action_outcome_unknown','action_reconciled')
                  and not exists(select 1 from jarvis_task_actions a where a.owner_id=e.owner_id and a.receipt_payload_id=e.payload_id)
                  and not exists(select 1 from jarvis_context_payload_dependencies d
                                 where d.owner_id=e.owner_id and d.dependent_payload_id=e.payload_id)
                order by e.id limit $3""", owner_id, after_event_id, limit + 1)
            return {
                "format":"jarvis.task_receipt_unbound_history.v1", "owner_id":owner_id,
                "items":[{"event_id":str(row["id"]), "task_id":str(row["task_id"]),
                          "task_revision":row["task_revision"], "event_type":row["event_type"],
                          "payload_id":str(row["payload_id"]), "reason":"no_current_action_pointer_or_explicit_source"}
                         for row in rows[:limit]],
                "has_more":len(rows) > limit,
                "next_after_event_id":str(rows[limit - 1]["id"]) if len(rows) > limit else None,
                "migration_accepted":False, "application_release_authorized":False,
            }
