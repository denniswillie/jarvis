"""Scoped SQL boundary fixture. No replacement task/admission/manifest APIs."""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from uuid import uuid4

from app.tasks.models import digest


class ReadonlyStore:
    def __init__(self):
        self.payloads, self.events, self.tasks, self.requests, self.ingress = {}, {}, {}, {}, {}
        self.parents, self.attempts, self.manifests, self.manifest_sources = {}, {}, {}, []
        self.task_events, self.sessions, self.grants = [], set(), set()
        self.sequence = 0
        self.depth = 0
        self.lock = asyncio.Lock()
        self.fragment_gate = None

    @asynccontextmanager
    async def acquire(self):
        yield self

    @asynccontextmanager
    async def transaction(self, **kwargs):
        async with self.lock:
            saved = deepcopy((self.payloads, self.events, self.tasks, self.requests, self.ingress,
                self.parents, self.attempts, self.manifests, self.manifest_sources, self.task_events, self.sequence))
            self.depth += 1
            try:
                yield
            except BaseException:
                (self.payloads, self.events, self.tasks, self.requests, self.ingress, self.parents,
                 self.attempts, self.manifests, self.manifest_sources, self.task_events, self.sequence) = saved
                raise
            finally:
                self.depth -= 1

    def is_in_transaction(self):
        return self.depth > 0

    def available(self, owner, identity):
        payload = self.payloads.get(identity)
        return bool(payload and payload["owner_id"] == owner and payload["availability"] == "available"
                    and all(self.available(owner, p) for p in self.parents.get(identity, ())))

    def record(self, identity):
        row = self.events.get(identity)
        if row is None:
            return None
        payload = self.payloads[row["payload_id"]]
        peers = [r for r in self.events.values() if (r["owner_id"], r["source_namespace"], r["source_event_id"])
                 == (row["owner_id"], row["source_namespace"], row["source_event_id"])]
        return deepcopy(row) | {"availability": payload["availability"] if all(
            self.available(row["owner_id"], parent)
            for parent in self.parents.get(row["payload_id"], ())) else "unavailable",
            "content": payload["content"], "content_digest": digest(payload["content"]),
            "origin_sequence": min(p["sequence"] for p in peers),
            "is_current": row["revision"] == max(p["revision"] for p in peers)}

    def rows(self, owner, sessions=None):
        return [self.record(i) for i, r in self.events.items() if r["owner_id"] == owner
                and (sessions is None or r["session_id"] in sessions)]

    async def fetchval(self, sql, *a):
        q = " ".join(sql.split()).lower()
        if "lock_jarvis_copy_source(" in q:
            return "available" if self.available(a[0], a[1]) else "removed"
        if "lock_jarvis_copy_sources(" in q:
            return bool(a[1]) and all(self.available(a[0], identity) for identity in a[1])
        if "insert into jarvis_context_payloads" in q:
            identity = uuid4()
            availability, content = (a[1], a[2]) if len(a) == 3 else ("available", a[1])
            if self.fragment_gate and json.loads(content).get("format") == "jarvis.live_provisional_fragment.v1":
                entered, release = self.fragment_gate
                entered.set()
                await release.wait()
            self.payloads[identity] = {"owner_id": a[0], "availability": availability, "content": json.loads(content)}
            return identity
        if "insert into jarvis_context_events" in q:
            identity = uuid4()
            self.sequence += 1
            owner, session, namespace, event, revision, previous, kind, finality, trust, at, payload = a
            self.events[identity] = dict(id=identity, owner_id=owner, session_id=session, source_namespace=namespace,
                source_event_id=event, revision=revision, supersedes_event_id=previous, kind=kind, finality=finality,
                trust=trust, source_time=at, payload_id=payload, sequence=self.sequence,
                recorded_at=datetime.now(timezone.utc))
            if previous:
                for dependency in self.manifest_sources:
                    if dependency[0] == owner and dependency[2] == previous and dependency[6] == "current":
                        self.manifests[dependency[1]]["status"] = "superseded"
            return identity
        if "insert into jarvis_background_tasks" in q:
            identity = uuid4()
            owner, session, area, backend, model, description, event, revision, watermark, budget = a
            self.tasks[identity] = dict(id=identity, owner_id=owner, session_id=session, knowledge_area_id=area,
                backend_kind=backend, model=model, description_payload_id=description, request_event_id=event,
                request_revision=revision, context_watermark=watermark, budget=budget, status="queued", revision=1,
                current_attempt_id=None, fence=0, attempt_count=0, active_seconds=0, result_payload_id=None,
                reason=None, pending_status=None, last_event_sequence=0,
                created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
            return identity
        if "last_event_sequence=last_event_sequence+1" in q:
            self.tasks[a[1]]["last_event_sequence"] += 1
            return self.tasks[a[1]]["last_event_sequence"]
        if "from jarvis_sessions" in q:
            return (a[0], a[1]) in self.sessions
        if "exists(select 1 from jarvis_task_grants" in q:
            return (a[0], a[1]) in self.grants
        if "from jarvis_context_streams" in q:
            return self.sequence
        if "update jarvis_task_attempts set charged_until" in q:
            return 0.0
        if "exists(select 1 from jarvis_task_actions" in q:
            return False
        if "count(*) from jarvis_background_tasks" in q:
            return sum(t["owner_id"] == a[0] and t["status"] in {"queued", "paused", "waiting_for_input"}
                       for t in self.tasks.values())
        if "count(*) from jarvis_context_events" in q:
            if "jarvis_context_pins" in q:
                return 0
            return len([r for r in self.rows(a[0], a[1]) if r["is_current"]])
        raise AssertionError((q, a))

    async def fetchrow(self, sql, *a):
        q = " ".join(sql.split()).lower()
        if "from jarvis_context_ingress_receipts" in q:
            return self.ingress.get(tuple(a))
        if "from jarvis_task_requests" in q:
            return self.requests.get(tuple(a))
        if q.startswith("select e.") and "from jarvis_context_events e" in q:
            if "e.id=$3" in q:
                row = self.record(a[2])
                return row if row and row["owner_id"] == a[0] and row["sequence"] <= a[3] and (
                    row["session_id"] in a[1] or row["id"] == a[4]) else None
            if "e.source_namespace=$2" in q:
                rows = [r for r in self.rows(a[0]) if r["source_namespace"] == a[1] and r["source_event_id"] == a[2]]
                return max(rows, key=lambda r: r["revision"]) if rows else None
            row = self.record(a[1])
            if row is None or row["owner_id"] != a[0]:
                return None
            if "e.revision=$3" in q and (row["revision"] != a[2] or not row["is_current"]):
                return None
            return row
        if "select t.*,p.content as description" in q:
            task = self.tasks.get(a[1])
            if task is None or task["owner_id"] != a[0]:
                return None
            return deepcopy(task) | {"description": deepcopy(self.payloads[task["description_payload_id"]]["content"])}
        if "from jarvis_task_attempts where" in q:
            attempt = self.attempts.get(a[1])
            return deepcopy(attempt) | {"lease_valid": True} if attempt else None
        if "select m.status from jarvis_task_attempts" in q:
            manifest = self.attempts[a[1]].get("context_manifest_id")
            return {"status": self.manifests[manifest]["status"]} if manifest else None
        if "insert into jarvis_task_context_manifests" in q:
            identity = uuid4()
            row = dict(id=identity, owner_id=a[0], task_id=a[1], task_revision=a[2], context_watermark=a[3],
                payload_id=a[4], status="active", created_at=datetime.now(timezone.utc))
            self.manifests[identity] = row
            return row
        if "select id,context_watermark,status,task_revision from jarvis_task_context_manifests" in q:
            row = self.manifests.get(a[2])
            return deepcopy(row) if row and row["owner_id"] == a[0] and row["task_id"] == a[1] else None
        if "from jarvis_task_reservations" in q and "coalesce(sum" in q:
            return {"input": 0, "output": 0, "steps": 0}
        if "select t.last_event_sequence" in q:
            task = self.tasks[a[1]]
            event = next((e for e in reversed(self.task_events) if e[1] == task["id"]
                          and e[0] == a[0] and e[3] == task["revision"] and e[5] == task["status"]
                          and e[6] == task["result_payload_id"] and e[4] is not None), None)
            manifest = self.manifests.get(self.attempts[event[4]]["context_manifest_id"]) if event else None
            return {"last_event_sequence": task["last_event_sequence"],
                "manifest_id": manifest["id"] if manifest else None,
                "manifest_payload_id": manifest["payload_id"] if manifest else None,
                "manifest_status": manifest["status"] if manifest else None,
                "result": self.payloads[task["result_payload_id"]]["content"] if manifest and manifest["status"] == "active" else None}
        raise AssertionError((q, a))

    async def fetch(self, sql, *a):
        q = " ".join(sql.split()).lower()
        if "from jarvis_task_grants" in q or "from jarvis_task_session_context" in q:
            return []
        if "from jarvis_context_events e" in q:
            rows = self.rows(a[0])
            if "e.id=any($2::uuid[])" in q:
                return [r for r in rows if r["id"] in a[1]]
            if "jarvis_context_pins" in q:
                return []
            return sorted([r for r in rows if r["session_id"] in a[1] and r["is_current"]],
                          key=lambda r: r["sequence"], reverse=True)[:a[2]]
        if "from jarvis_context_payloads" in q:
            return [{"id": identity} for identity in a[1] if identity in self.payloads]
        if "select id from jarvis_background_tasks" in q:
            return [{"id": t["id"]} for t in self.tasks.values() if t["owner_id"] == a[0]][:a[1]]
        raise AssertionError((q, a))

    async def execute(self, sql, *a):
        q = " ".join(sql.split()).lower()
        if "pg_advisory_xact_lock" in q:
            return
        if "insert into jarvis_context_ingress_receipts" in q:
            self.ingress[tuple(a[:4])] = {"parameters_digest": a[4], "event_id": a[5]}
        elif "insert into jarvis_task_requests" in q:
            assert a[3] in {"create", "update", "cancel", "resume"}
            self.requests[(a[0], a[1])] = {"task_id": a[2], "operation": a[3], "parameters_digest": a[4], "accepted_revision": a[5]}
        elif "insert into jarvis_task_events" in q:
            self.task_events.append(a)
        elif "insert into jarvis_context_payload_dependencies" in q:
            owner, parent, child = a
            assert self.available(owner, parent)
            self.parents.setdefault(child, set()).add(parent)
        elif "update jarvis_task_attempts set context_manifest_id" in q:
            self.attempts[a[1]]["context_manifest_id"] = a[2]
        elif "update jarvis_background_tasks set status=$3" in q:
            self.tasks[a[1]].update(status=a[2], reason=a[3])
        elif "update jarvis_background_tasks set revision=revision+1" in q:
            task = self.tasks[a[1]]
            task.update(revision=task["revision"] + 1, fence=task["fence"] + 1, pending_status=a[2], result_payload_id=None)
        elif "update jarvis_task_reservations" in q or "update jarvis_background_tasks set active_seconds" in q:
            pass  # This fixture submits no model requests or budgeted read calls.
        elif "update jarvis_task_attempts set status=$3,provider_state='terminal'" in q:
            self.attempts[a[1]].update(status=a[2], provider_state="terminal")
        elif "update jarvis_background_tasks set current_attempt_id=null,pending_status=null,result_payload_id=$3" in q:
            self.tasks[a[1]].update(current_attempt_id=None, pending_status=None, result_payload_id=a[2])
        else:
            raise AssertionError((q, a))

    async def executemany(self, sql, rows):
        for row in rows:
            if "jarvis_context_payload_dependencies" in sql:
                owner, parent, child = row
                assert self.available(owner, parent)
                self.parents.setdefault(child, set()).add(parent)
            elif "jarvis_task_context_sources" in sql:
                self.manifest_sources.append(row)
            else:
                raise AssertionError(sql)

    def running(self, task):
        identity = uuid4()
        self.tasks[task.id].update(status="running", current_attempt_id=identity, fence=1, attempt_count=1)
        self.attempts[identity] = dict(id=identity, task_id=task.id, task_revision=task.revision, fence=1,
            lease_until=datetime.now(timezone.utc) + timedelta(minutes=1), provider_state="not_started",
            provider_handle=None, checkpoint_payload_id=None, context_manifest_id=None, status="running")
        self.attempts[identity]["charged_until"] = datetime.now(timezone.utc)
        return identity


async def legacy_worker_context(session_id):
    """Real repository/context/claim for the legacy configuration-only test."""
    from app.runtime.feature_config import TaskLimits
    from app.tasks.context_events import ContextIngress
    from app.tasks.context_repository import ContextRepository
    from app.tasks.repository import TaskRepository
    from app.tasks.runner import TaskRunner
    db = ReadonlyStore()
    owner = "legacy-worker-fixture"
    if session_id is not None:
        db.sessions.add((owner, session_id))
    tasks = TaskRepository(db, TaskLimits())
    source = await ContextRepository(tasks).append(owner, ContextIngress(session_id=session_id,
        source_namespace="fixture-confirmed", source_event_id="request", receipt_id="request",
        kind="user_request", trust="user_input", payload={"text": "Reason about this fixture."}))
    async with tasks.transaction(owner) as connection:
        description = await tasks.payload(connection, owner, {"title": "Fixture", "requested_outcome": "Reason."},
                                          source_payload_ids=(source.payload_id,))
        identity = await connection.fetchval("insert into jarvis_background_tasks returning id", owner, session_id,
            None, "frontier", "fixture-model", description, source.id, source.revision, source.sequence,
            tasks.limits.model_dump_json())
    task = await tasks.get(owner, identity)
    attempt = db.running(task)
    task = await tasks.get(owner, identity)
    return db, TaskRunner(tasks, {}).context(tasks._claim(task, db.attempts[attempt]))
