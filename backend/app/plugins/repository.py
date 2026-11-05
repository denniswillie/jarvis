"""Immutable plugin choices in the existing capability-preference ledger.

Core supplies the owner-locked transaction and canonical structured UI event.
This repository never creates a principal, event, grant, installer or baseline.
"""
from __future__ import annotations

from pydantic import ValidationError

from app.errors import JarvisHTTPError
from app.plugins.policy import CAPABILITY, UI_NAMESPACE, PluginPolicyChange, PluginPolicyState, StoredPluginPolicy, validate_policy
from app.tasks.context_events import ContextRecord
from app.tasks.models import json_object


class PluginPolicyRepository:
    def __init__(self, pool):
        self.pool = pool

    async def read(self, connection, owner_id: str, *, baseline=None, lock=False) -> PluginPolicyState:
        row = await connection.fetchrow("""select q.revision,q.policy_payload_id,p.availability,p.content
            from jarvis_capability_preferences q join jarvis_context_payloads p
              on p.owner_id=q.owner_id and p.id=q.policy_payload_id
            where q.owner_id=$1 and q.capability=$2 and q.knowledge_area_id is null
            order by q.revision desc limit 1""" + (" for share of p" if lock else ""), owner_id, CAPABILITY)
        if row is None:
            if baseline is not None:
                validate_policy(baseline)
                if baseline.revision != 0:
                    raise ValueError("A trusted initial baseline must have revision zero")
            return PluginPolicyState(0, baseline)
        policy = None
        if row["availability"] == "available" and row["content"] is not None:
            try:
                stored = StoredPluginPolicy.model_validate(json_object(row["content"]))
                if stored.policy.revision == row["revision"]:
                    policy = validate_policy(stored.policy)
            except (ValidationError, ValueError, TypeError):
                pass
        # A withdrawn/corrupt latest policy never falls back to an older row or
        # the deployment baseline. Preserve its revision internally for CAS.
        return PluginPolicyState(row["revision"], policy, row["policy_payload_id"])

    def require_source(self, connection, owner_id: str, command: PluginPolicyChange, source: ContextRecord):
        if not connection.is_in_transaction():
            raise RuntimeError("Plugin policy writes require the caller's owner transaction")
        if (source.owner_id != owner_id or source.session_id is not None or not source.authorizes_request
                or source.kind != "user_request" or source.source_namespace != UI_NAMESPACE
                or source.source_event_id != str(command.mutation_id) or source.content != command.ingress_payload()):
            raise JarvisHTTPError(403, "plugin_request_unavailable", "The authenticated plugin request is unavailable.")

    async def applied(self, connection, owner_id: str, command: PluginPolicyChange, source: ContextRecord) -> int | None:
        self.require_source(connection, owner_id, command, source)
        row = await connection.fetchrow("""select q.revision,p.availability,p.content
            from jarvis_capability_preferences q join jarvis_context_payloads p
              on p.owner_id=q.owner_id and p.id=q.policy_payload_id
            where q.owner_id=$1 and q.capability=$2 and q.knowledge_area_id is null
              and q.request_event_id=$3 and q.request_revision=$4
            order by q.revision desc limit 1 for share of p""", owner_id, CAPABILITY, source.id, source.revision)
        if row is None:
            return None
        try:
            if row["availability"] != "available" or row["content"] is None:
                raise ValueError("Unavailable receipt")
            stored = StoredPluginPolicy.model_validate(json_object(row["content"]))
            validate_policy(stored.policy)
            if stored.policy.revision != row["revision"]:
                raise ValueError("Mismatched receipt revision")
        except (ValidationError, ValueError, TypeError) as error:
            raise JarvisHTTPError(409, "plugin_policy_receipt_unavailable",
                "The original plugin change cannot be confirmed. Refresh its current status.") from error
        if stored.intent.canonical_intent() != command.canonical_intent():
            raise JarvisHTTPError(409, "plugin_policy_mutation_conflict", "This change identifier already refers to different choices.")
        return row["revision"]

    async def append(self, connection, owner_id: str, command: PluginPolicyChange,
                     source: ContextRecord, previous: PluginPolicyState, policy):
        self.require_source(connection, owner_id, command, source)
        validate_policy(policy)
        if previous.policy is None:
            raise JarvisHTTPError(409, "plugin_policy_unavailable", "Current plugin choices are unavailable. Refresh first.")
        if command.expected_revision != previous.revision or policy.revision != previous.revision + 1:
            raise ValueError("Plugin policy revision must advance exactly once")
        parents = {source.payload_id}
        if previous.payload_id is not None:
            parents.add(previous.payload_id)
        for parent in sorted(parents):
            if await connection.fetchval("select lock_jarvis_copy_source($1,$2)", owner_id, parent) != "available":
                raise JarvisHTTPError(409, "plugin_policy_source_unavailable", "The plugin choices changed or became unavailable. Refresh first.")
        body = StoredPluginPolicy(intent=command, policy=policy)
        payload = await connection.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
            values($1,'available',$2::jsonb) returning id""", owner_id, body.model_dump_json())
        await connection.executemany("""insert into jarvis_context_payload_dependencies
            (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
            [(owner_id, parent, payload) for parent in sorted(parents)])
        await connection.execute("""insert into jarvis_capability_preferences
            (owner_id,capability,revision,request_event_id,request_revision,policy_payload_id)
            values($1,$2,$3,$4,$5,$6)""", owner_id, CAPABILITY, policy.revision, source.id, source.revision, payload)
        return PluginPolicyState(policy.revision, policy, payload)
