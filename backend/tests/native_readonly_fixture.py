"""External SQL/model boundaries for actual native readonly qualification.

No agent loop or owner policy lives here. Production repositories interpret the
rows and enforce claims, budgets, source validity and recovery. This does not
establish PostgreSQL SQL/constraint compatibility or provider model quality.
"""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import json
from uuid import uuid4

from readonly_live_store import ReadonlyStore
from app.tasks.model_gateway import ModelEvent


def now():
    return datetime.now(UTC)


class NativeReadonlyStore(ReadonlyStore):
    def __init__(self):
        super().__init__()
        self.reservations = {}
        self.scheduler = None

    @asynccontextmanager
    async def transaction(self, **kwargs):
        async with super().transaction(**kwargs):
            saved = deepcopy((self.reservations, self.scheduler))
            try:
                yield
            except BaseException:
                self.reservations, self.scheduler = saved
                raise

    async def fetchval(self, sql, *a):
        q = " ".join(sql.split()).lower()
        if "count(*) from jarvis_task_attempts" in q:
            return sum(r["status"] in a[0] and (a[1] is None or r["owner_id"] == a[1])
                       for r in self.attempts.values())
        if "exists(select 1 from jarvis_task_reservations" in q:
            return any(r["owner_id"] == a[0] and r["attempt_id"] == a[1]
                       and r["is_provider"] and r["status"] in {"submitted", "unknown"}
                       for r in self.reservations.values())
        if "exists(select 1 from jarvis_task_attempts" in q:
            r = self.attempts.get(a[2])
            return bool(r and (r["owner_id"], r["task_id"]) == a[:2])
        if "update jarvis_task_attempts set charged_until" in q:
            r = self.attempts[a[1]]
            r["charged_until"] = min(now(), r["lease_until"])
            return max(0, (r["charged_until"] - a[2]).total_seconds())
        return await super().fetchval(sql, *a)

    async def fetchrow(self, sql, *a):
        q = " ".join(sql.split()).lower()
        if "from jarvis_task_scheduler_limits" in q:
            return deepcopy(self.scheduler)
        if q.startswith("select t.owner_id,t.id from jarvis_background_tasks"):
            models = json.loads(a[4]) if a[4] else None
            for t in self.tasks.values():
                active = sum(r["owner_id"] == t["owner_id"] and r["status"] in a[1]
                             for r in self.attempts.values())
                source = self.events[t["request_event_id"]]
                if (t["status"] == "queued" and t["current_attempt_id"] is None
                        and t["backend_kind"] in a[0] and (a[3] is None or t["owner_id"] == a[3])
                        and (models is None or models.get(t["backend_kind"]) == t["model"])
                        and active < a[2]
                        and (not source["source_namespace"].startswith("live-readonly:")
                             or a[5] is not None and "live.readonly.v1" in a[5])):
                    return {"owner_id":t["owner_id"], "id":t["id"]}
            return None
        if q.startswith("insert into jarvis_task_attempts"):
            identity = uuid4()
            self.attempts[identity] = dict(id=identity, owner_id=a[0], task_id=a[1], task_revision=a[2],
                attempt_number=a[3], fence=a[4], worker_id=a[5], lease_until=now()+timedelta(seconds=a[6]),
                charged_until=now(), status="running", provider_state="not_started", provider_handle=None,
                checkpoint_payload_id=None, context_manifest_id=None, runtime_host_id=None)
            return deepcopy(self.attempts[identity])
        if q.startswith("select a.owner_id,a.task_id,a.id from jarvis_task_attempts"):
            for r in self.attempts.values():
                t = self.tasks[r["task_id"]]
                if (r["status"] in a[0] and r["lease_until"] <= now() and t["current_attempt_id"] == r["id"]
                        and (a[1] is None or r["owner_id"] == a[1])
                        and (a[2] is None or r["runtime_host_id"] in {None, a[2]})
                        and (a[3] is None or t["backend_kind"] in a[3])):
                    return deepcopy(r)
            return None
        if q.startswith("update jarvis_task_attempts set status='recovering'"):
            r = self.attempts[a[1]]
            r.update(status="recovering", worker_id=a[2], fence=a[3], charged_until=now(),
                     lease_until=now()+timedelta(seconds=a[4]))
            return deepcopy(r)
        if "from jarvis_task_attempts where" in q:
            r = self.attempts.get(a[1])
            if r is None or r["owner_id"] != a[0]:
                return None
            return deepcopy(r) | {"lease_valid":r["lease_until"] > now(), "expired":r["lease_until"] <= now()}
        if q.startswith("select availability,content from jarvis_context_payloads"):
            r = self.payloads.get(a[1])
            return deepcopy(r) if r and r["owner_id"] == a[0] else None
        if "from jarvis_task_reservations" in q and "coalesce(sum" in q:
            rows = [r for r in self.reservations.values() if r["owner_id"] == a[0]
                    and r["status"] != "released" and (len(a) == 1 or r["task_id"] == a[1])]
            return {key:sum(r[("actual_" if r["status"] == "settled" else "")+field] for r in rows)
                    for key, field in (("input","input_tokens"), ("output","output_tokens"), ("steps","tool_steps"))}
        if q.startswith("insert into jarvis_task_reservations"):
            identity = uuid4()
            self.reservations[identity] = dict(id=identity, owner_id=a[0], task_id=a[1], attempt_id=a[2],
                operation_key=a[3], status="reserved", input_tokens=a[4], output_tokens=a[5], tool_steps=a[6],
                actual_input_tokens=None, actual_output_tokens=None, actual_tool_steps=None, is_provider=False,
                accounting_day=now().date())
            return deepcopy(self.reservations[identity])
        if q.startswith("update jarvis_task_reservations set status='reserved'"):
            r = self.reservations[a[1]]
            r.update(status="reserved", attempt_id=a[2], accounting_day=now().date())
            return deepcopy(r)
        if "from jarvis_task_reservations" in q:
            rows = [r for r in self.reservations.values() if r["owner_id"] == a[0] and r["task_id"] == a[1]]
            if "operation_key=$3" in q:
                rows = [r for r in rows if r["operation_key"] == a[2]]
            elif "attempt_id=$3 and id=$4" in q:
                rows = [r for r in rows if (r["attempt_id"], r["id"], r["status"]) == (a[2], a[3], "reserved")]
            else:
                rows = [r for r in rows if (r["id"], r["attempt_id"]) == (a[2], a[3])]
            return deepcopy(rows[0]) | {"current_day": rows[0]["accounting_day"] == now().date()} if rows else None
        return await super().fetchrow(sql, *a)

    async def execute(self, sql, *a):
        q = " ".join(sql.split()).lower()
        if q.startswith("insert into jarvis_task_scheduler_limits"):
            self.scheduler = self.scheduler or {"global_concurrency":a[0], "owner_concurrency":a[1]}
        elif q.startswith("insert into jarvis_task_owner_scheduling"):
            pass
        elif q.startswith("update jarvis_task_attempts set provider_state='live'"):
            self.attempts[a[1]].update(provider_state="live", provider_handle=a[2], checkpoint_payload_id=a[3], runtime_host_id=a[4])
        elif q.startswith("update jarvis_task_attempts set checkpoint_payload_id"):
            r = self.attempts[a[1]]
            r["checkpoint_payload_id"] = a[2]
            if a[3] is not None:
                r.update(provider_handle=a[3], provider_state="live")
        elif q.startswith("update jarvis_background_tasks set status='running'"):
            t = self.tasks[a[1]]
            t.update(status="running", reason=None, current_attempt_id=a[2],
                     attempt_count=t["attempt_count"]+1, fence=t["fence"]+1)
        elif q.startswith("update jarvis_background_tasks set active_seconds"):
            self.tasks[a[1]]["active_seconds"] += a[2]
        elif q.startswith("update jarvis_task_attempts set lease_until"):
            self.attempts[a[1]]["lease_until"] = now()+timedelta(seconds=a[2] if len(a)>2 else 0)
        elif q.startswith("update jarvis_task_attempts set status=$3,lease_until=clock_timestamp(),"):
            self.attempts[a[1]].update(status=a[2], lease_until=now())
        elif q.startswith("update jarvis_task_attempts set status=$3,lease_until=clock_timestamp()+"):
            self.attempts[a[1]].update(status=a[2], lease_until=now()+timedelta(seconds=a[3]))
        elif q.startswith("update jarvis_task_attempts set status='interrupted',provider_state='terminal'"):
            self.attempts[a[1]].update(status="interrupted", provider_state="terminal")
        elif q.startswith("update jarvis_background_tasks set fence=fence+1,current_attempt_id"):
            t = self.tasks[a[1]]
            t.update(fence=t["fence"]+1, current_attempt_id=a[2],
                pending_status="paused" if a[3] else None if not a[4] and not a[5] else t["pending_status"])
        elif q.startswith("update jarvis_background_tasks set fence=$3"):
            self.tasks[a[1]]["fence"] = a[2]
        elif q.startswith("update jarvis_background_tasks set pending_status=case"):
            t = self.tasks[a[1]]
            t.update(pending_status="cancelled" if t["pending_status"] == "cancelled" else "paused", reason=a[2])
        elif q.startswith("update jarvis_background_tasks set current_attempt_id=null, pending_status=case"):
            t = self.tasks[a[1]]
            t.update(current_attempt_id=None, pending_status=t["pending_status"] if a[2] else None)
        elif q.startswith("update jarvis_task_actions"):
            assert not self.grants  # This qualification has no actions or grants.
        elif q.startswith("update jarvis_task_reservations"):
            if "where owner_id=$1 and id=$2" in q:
                rows = [self.reservations[a[1]]]
            else:
                rows = [r for r in self.reservations.values() if r["owner_id"] == a[0] and r["attempt_id"] == a[1]]
                if "and status='reserved'" in q:
                    rows = [r for r in rows if r["status"] == "reserved"]
                elif "and status='submitted'" in q:
                    rows = [r for r in rows if r["status"] == "submitted"]
                elif "and is_provider" in q:
                    rows = [r for r in rows if r["is_provider"] and r["status"] in {"submitted", "unknown"}]
            for r in rows:
                if "set status='submitted'" in q:
                    r.update(status="submitted", is_provider=a[2], accounting_day=now().date())
                elif "set status='settled'" in q:
                    r.update(status="settled", actual_input_tokens=a[2], actual_output_tokens=a[3], actual_tool_steps=a[4])
                elif "set status='released'" in q:
                    r["status"] = "released"
                elif "set status='unknown'" in q:
                    r["status"] = "unknown"
                else:
                    raise AssertionError("unsupported reservation update")
        else:
            return await super().execute(sql, *a)


def frame(value):
    return ModelEvent(value, ("data: " + json.dumps(value) + "\n\n").encode())


class ControlledReadonlyUpstream:
    """Deterministic Responses events. No HTTP client or provider credential."""
    def __init__(self, answer, *, hold=False, budget_reject=False):
        self.answer = answer
        self.hold, self.budget_reject = hold, budget_reject
        self.count_entered, self.count_release = asyncio.Event(), asyncio.Event()
        self.stream_entered, self.stream_release = asyncio.Event(), asyncio.Event()
        self.requests, self.count_calls = [], []
        self.closed = False

    async def count(self, payload):
        self.count_calls.append(deepcopy(payload))
        self.count_entered.set()
        await self.count_release.wait()
        return 1_000_000_000 if self.budget_reject else 100

    async def events(self, payload):
        self.requests.append(deepcopy(payload))
        index = len(self.requests)
        if index > 3:
            raise AssertionError("unexpected native retry or loop")
        response_id = "resp_synthetic_" + str(index)
        yield frame({"type":"response.created", "response":{"id":response_id, "status":"in_progress"}})
        self.stream_entered.set()
        if self.hold:
            await self.stream_release.wait()
        if index == 1:
            item = {"type":"function_call", "id":"fc_synthetic", "call_id":"read_status_synthetic",
                    "name":"read_background_task_status", "arguments":"{}", "status":"completed"}
            yield frame({"type":"response.output_item.added", "output_index":0, "item":item | {"arguments":""}})
            yield frame({"type":"response.function_call_arguments.delta", "item_id":item["id"],
                         "output_index":0, "delta":"{}"})
            yield frame({"type":"response.function_call_arguments.done", "item_id":item["id"],
                         "output_index":0, "arguments":"{}"})
        else:
            text = json.dumps(self.answer)
            item = {"type":"message", "id":"msg_synthetic", "role":"assistant", "status":"completed",
                    "content":[{"type":"output_text", "text":text, "annotations":[]}]}
            yield frame({"type":"response.output_item.added", "output_index":0, "item":item | {"content":[]}})
            yield frame({"type":"response.output_text.delta", "item_id":item["id"], "output_index":0,
                         "content_index":0, "delta":text})
        yield frame({"type":"response.output_item.done", "output_index":0, "item":item})
        yield frame({"type":"response.completed", "response":{"id":response_id, "status":"completed",
                     "output":[item], "usage":{"input_tokens":11, "output_tokens":7}}})

    async def aclose(self):
        self.closed = True
