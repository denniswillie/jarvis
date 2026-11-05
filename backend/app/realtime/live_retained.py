"""Original dialogue into one retained master; native receipts own the work.

This is a transport adapter, not another task scheduler or conversation store.
Requested turns retain their own result watcher. Explicit background admissions
return only the Master's final answer, never raw events or child progress.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
import json
from uuid import NAMESPACE_URL, uuid4, uuid5

from app.hermes_sessions.bindings import identity
from app.hermes_sessions.client import HermesGatewayError
from app.hermes_sessions.service import InputReceipt
from app.tasks.models import TaskError
from app.voice_routes.bridge import utc_now
from app.voice_routes.live_results import LiveDelegationResult


PREFIX = "Original Jarvis User/Voice dialogue (data, with approximate timing):\n"


def _recent_results(messages):
    """Bound public answers, not tool output or private reasoning, newest first.

    Execution state is separate: an older answer does not mean the current turn
    has finished. Native rows remain authoritative; no Voice transcript cache.
    """
    candidates = []
    recovery_turn = False
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            recovery_turn = isinstance(content, str) and content.startswith(
                "[SYSTEM EVENT: backend_restarted; not a new user request]")
        # Recovery chatter is retained in native history, not promoted to a
        # user-work result where repeated restarts would evict the actual answer.
        if recovery_turn:
            continue
        if (message.get("role") != "assistant" or message.get("tool_calls")
                or (message.get("display_metadata") or {}).get("model_only")):
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part["text"] for part in content
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
                and isinstance(part.get("text"), str))
        if isinstance(content, str) and content.strip():
            candidates.append((message, content))
    results, remaining = [], 6000
    for message, content in list(reversed(candidates))[:3]:
        if remaining <= 2:
            break
        # Bound escaped JSON text, including non-ASCII, before Live framing.
        low, high = 0, min(len(content), remaining)
        while low < high:
            middle = (low + high + 1) // 2
            if len(json.dumps(content[:middle], ensure_ascii=True)) <= remaining:
                low = middle
            else:
                high = middle - 1
        if not low:
            break
        text = content[:low]
        remaining -= len(json.dumps(text, ensure_ascii=True))
        result = {key: message[key] for key in ("row_id", "timestamp", "archived") if key in message}
        result["text"] = text
        if low < len(content):
            result["text_truncated"] = True
        results.append(result)
    return results, len(candidates) - len(results)


class RetainedMasterNotices:
    """Drain explicit system/child-result admissions on the existing host pump.

    Voice admissions never enter this hook. Native receipts own results; this
    process keeps only delivery identities and one in-flight notice. No replay
    into a later voice session, new scheduler, or persisted transcript is added.
    """

    def __init__(self, runtime, session_manager, *, clock=utc_now):
        self.runtime, self.sessions, self.clock = runtime, session_manager, clock
        self.owner_id = runtime.bridge.service.bindings.owner_id
        self._seen, self._pending = set(), {}
        self._notice = None
        self._lock = asyncio.Lock()
        self.last_status = self.last_admission_id = None

    def admitted(self, receipt):
        binding = self.runtime.bridge.service.bindings.get(receipt.session_id)
        if not binding.is_master or receipt.admission_id in self._seen:
            return
        self._seen.add(receipt.admission_id)
        self._pending[receipt.admission_id] = receipt

    def notice_current(self, notice):
        return (self._notice is not None and notice == self._notice
                and self.clock().isoformat() < notice["expires_at"])

    async def pump(self, bridge, delivered):
        async with self._lock:
            for receipt in delivered:
                self.admitted(InputReceipt(**receipt))
            for admission_id, receipt in tuple(self._pending.items()):
                value = await bridge.service.receipt(receipt.session_id, admission_id)
                text = (value.get("result") or {}).get("final_response")
                if value.get("status") != "terminal":
                    continue
                if not isinstance(text, str) or not text.strip():
                    del self._pending[admission_id]
                    continue
                # Only an actual Master reply is speakable, never an admission
                # ACK or the child's answer. Stable IDs fence duplicate callbacks.
                notice_id = uuid5(NAMESPACE_URL, identity("hermes-completion", self.owner_id,
                    str(bridge.service.bindings.profile_home), receipt.session_id, admission_id))
                self._notice = {"id": str(notice_id), "kind": "hermes_completion", "text": text,
                    "expires_at": (self.clock() + timedelta(seconds=30)).isoformat()}
                self.last_admission_id = admission_id
                del self._pending[admission_id]
                try:
                    result = await self.sessions.send_companion_notice(self.owner_id, self._notice)
                    self.last_status = result.get("status", "unknown")
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # An uncertain write must not be repeated on a new route.
                    self.last_status = "unknown"
                finally:
                    self._notice = None


class RetainedLiveWork:
    retained = True

    def __init__(self, session, runtime):
        self.session, self.runtime = session, runtime
        self.service = runtime.bridge.service
        self.master = None
        self.seen = set()
        self.pending = None
        self.last_issue = None
        self.last_receipt = None
        self._lock = asyncio.Lock()
        self._jobs = set()
        self._watched_admissions = set()
        self._detached = False

    def route_followups(self):
        pass  # Only native delegations, never every incoming speech fragment.

    def offer_captions(self, parts):
        pass

    def router_directory(self):
        current = getattr(self.session.services, "companion_state", None)
        if current is not None:
            return current.directory()
        # Producers populate this RAM view in the Live State slice. No DB/model
        # call on the routing path, and missing inventory is not an empty account.
        return getattr(self.session.services, "live_target_directory", None)

    def snapshot(self):
        return {"master_session_id": self.master.logical_id if self.master else None,
                "last_admission": self.last_receipt, "last_issue": self.last_issue}

    async def read_snapshot(self):
        """Explicit Live State pull, including work predating this Voice session."""
        request = self.session.request
        if self._detached or request is None:
            raise TaskError("live_state_unavailable", "Retained work is unavailable for this Voice session.")
        if request.route.scope.owner_id != self.service.bindings.owner_id:
            raise TaskError("live_state_owner_mismatch", "Retained work belongs to another owner.")
        await self.runtime.wait_ready(timeout=2)
        # The ready runtime already owns the retained master. A state read must
        # not create one, admit a turn, or mark _connect's history as restored.
        master = self.master or getattr(self.runtime, "master", None)
        if master is None or not master.native_id:
            raise TaskError("live_state_unavailable", "The retained master is unavailable.")
        try:
            native = await self.service.client.snapshot(master.native_id)
        except (HermesGatewayError, OSError) as exc:
            raise TaskError("live_state_unavailable", "The retained work snapshot is unavailable.") from exc
        if self._detached or self.session.request is not request:
            raise TaskError("live_state_unavailable", "The Voice session changed during the work read.")
        if (not isinstance(native, dict) or native.get("session_id") != master.native_id
                or not isinstance(native.get("messages"), list) or native.get("messages_omitted")
                or type(native.get("running")) is not bool):
            raise TaskError("live_state_unavailable", "The retained work snapshot is incomplete.")
        results, omitted = _recent_results(native["messages"])
        return {"master_session_id": master.logical_id, "native_session_id": master.native_id,
                "observed_at": utc_now().isoformat(), "running": native["running"],
                "execution_generation": native.get("execution_generation"),
                "pending_count": len(native.get("pending", ())),
                "recent_result_status": "available" if results else "no_recorded_result",
                "recent_results": results, "recent_results_omitted": omitted,
                "history_truncated": any(row.get("archived_truncated") for row in native["messages"]),
                "last_admission": self.last_receipt, "last_issue": self.last_issue}

    async def _connect(self):
        await self.runtime.wait_ready(timeout=10)
        if self.master is not None:
            return
        master = await self.service.master()
        history = await self.service.client.snapshot(master.native_id)
        for message in history.get("messages", ()):
            content = message.get("content")
            if message.get("role") != "user" or not isinstance(content, str) or not content.startswith(PREFIX):
                continue
            try:
                envelope = json.loads(content[len(PREFIX):])
                if envelope.get("format") == "jarvis.original_dialogue.v1":
                    self.seen.update(part["context_event_id"] for part in envelope["dialogue"])
            except (ValueError, KeyError, TypeError):
                continue
        self.master = master

    async def _admit(self):
        pending = self.pending
        prepare = getattr(self.runtime, "prepare_voice_input", None)
        if prepare is not None:
            await prepare(session_id=self.master.logical_id, input_id=pending["input_id"],
                          jarvis_session_id=pending["jarvis_session_id"])
        elif hasattr(self.service, "bind_input_scope"):
            await self.service.bind_input_scope(self.master.logical_id,
                input_id=pending["input_id"], jarvis_session_id=pending["jarvis_session_id"],
                allowed_tools=getattr(self.runtime, "voice_allowed_tools", frozenset()))
        delivery = pending.get("delivery", "steer")
        redirected = None
        control = getattr(self.service, "control_live_input", None)
        # A raw session.steer can be accepted after the agent's last input
        # drain. Admit normal guidance first so native queued-steer settlement
        # can put unread input back in FIFO order when the running turn ends.
        if delivery == "redirect" and control is not None:
            try:
                redirected = await control(self.master.logical_id,
                    input_id=pending["input_id"], text=pending["text"],
                    jarvis_session_id=pending["jarvis_session_id"], delivery=delivery)
            except asyncio.CancelledError:
                # Unlike admission, native live controls have no replay identity.
                # Cancellation can race its delivery: never resubmit it later.
                self.seen.update(pending["ids"])
                self.pending = None
                self.last_issue = "hermes_guidance_delivery_unconfirmed"
                raise
        if redirected is not None:
            receipt, status = redirected
            if status == "unconfirmed":
                self.last_issue = "hermes_guidance_delivery_unconfirmed"
        else:
            receipt = await self.service.send(self.master.logical_id,
                input_id=pending["input_id"], text=pending["text"])
        self.seen.update(pending["ids"])
        self.pending = None
        self.last_receipt = {"admission_id": receipt.admission_id, "status": receipt.status,
                             "delivery": status if redirected is not None else
                                 ("queue" if control is not None else delivery)}
        steer = getattr(self.service, "steer_live_input", None)
        if delivery == "steer" and steer is not None:
            try:
                await steer(receipt, jarvis_session_id=pending["jarvis_session_id"])
            except asyncio.CancelledError:
                raise
            except Exception:
                # Admission already succeeded. Do not resend the dialogue or
                # lose its result watcher because a native control was uncertain.
                self.last_issue = "hermes_guidance_delivery_unconfirmed"
        return receipt

    async def __call__(self, packet, *, publish, is_current, delivery="steer"):
        if delivery not in ("steer", "queue", "redirect"):
            raise TaskError("live_work_delivery_invalid", "Unknown work delivery mode.")
        async with self._lock:
            if self._detached or not is_current():
                raise TaskError("live_work_unavailable", "This voice connection is no longer current.")
            if self.service.bindings.owner_id != packet.route.scope.owner_id:
                raise TaskError("live_work_owner_mismatch", "The retained master belongs to another owner.")
            await self._connect()
            # A lost admission acknowledgment reuses the same native input ID
            # and exact text rather than launching a replacement operation.
            if self.pending is not None:
                recovered = await self._admit()
                self._watch(recovered, packet, publish)
            original = packet.as_context()
            parts = [part for part in original["dialogue"] if part["context_event_id"] not in self.seen]
            if not parts:
                return
            payload = {"format": "jarvis.original_dialogue.v1",
                "jarvis_session_id": str(packet.route.scope.session_id),
                "delegation_id": packet.window.request.delegation_id,
                "window": original["window"],
                "dialogue": [{key: part[key] for key in
                    ("context_event_id", "sequence", "speaker", "delta", "start_ms", "end_ms") if key in part}
                    for part in parts]}
            self.pending = {"input_id": identity(str(packet.route.route_id), packet.window.request.delegation_id),
                "jarvis_session_id": packet.route.scope.session_id,
                "delivery": delivery,
                "text": PREFIX + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                "ids": {part["context_event_id"] for part in parts}}
            receipt = await self._admit()
            self._watch(receipt, packet, publish)

    def _watch(self, receipt, packet, publish):
        # Several redirects can join one turn; its result is still delivered once.
        if receipt.admission_id in self._watched_admissions:
            return
        self._watched_admissions.add(receipt.admission_id)
        job = asyncio.create_task(self._result(receipt, packet, publish), name="retained-hermes-reply")
        self._jobs.add(job)
        def finished(task):
            self._jobs.discard(task)
            if not task.cancelled() and task.exception() is not None:
                self.last_issue = getattr(task.exception(), "code", "hermes_reply_unavailable")
        job.add_done_callback(finished)

    async def _result(self, receipt, packet, publish):
        while not self._detached:
            value = await self.service.receipt(receipt.session_id, receipt.admission_id)
            if value.get("status") == "terminal":
                if value.get("outcome") == "steered":
                    # Guidance joined the running turn. Its original watcher
                    # delivers the answer; this transfer has no separate reply.
                    return
                result = value.get("result") or {}
                text = result.get("final_response")
                if not isinstance(text, str) or not text.strip():
                    text = "The background turn ended without an answer. Its result needs checking."
                if len(text.encode("utf-8")) > 10000:
                    # Do not silently cut code, numbers or instructions to fit voice.
                    text = "The background answer is too long for this voice reply. Ask for the relevant portion or a file."
                self.last_receipt = {"admission_id": receipt.admission_id, "status": "terminal",
                                     "outcome": value.get("outcome")}
                if self.session.ready and not self._detached:
                    output = LiveDelegationResult(packet.route, packet.window.request.delegation_id,
                        None, 0, uuid4(), text, utc_now() + timedelta(minutes=5), (),
                        kind="delegation_result", unsolicited=True)
                    await publish(output, guard=self._guard)
                return
            try:
                await self.service.client.wait_for_change(self.service.client.change_sequence, timeout=1)
            except TimeoutError:
                pass

    @asynccontextmanager
    async def _guard(self, result):
        if self._detached or not self.session.ready or result.route != self.session.request.route:
            raise TaskError("live_work_unavailable", "The voice connection changed.")
        current = getattr(self.session.services, "companion_state", None)
        if current is not None:
            if current.owner_id != result.route.scope.owner_id:
                raise TaskError("live_state_owner_mismatch", "Current state belongs to another owner.")
            async with current.delivery_guard():
                yield
            return
        async with self.session.live.scope_guard() as connection:
            yield connection

    def detach(self):
        self._detached = True
        for task in tuple(self._jobs):
            task.cancel()  # Detach response delivery; never cancel native work.

    async def close(self):
        self.detach()
        await asyncio.gather(*tuple(self._jobs), return_exceptions=True)
