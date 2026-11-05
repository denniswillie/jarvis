"""Backend-only foreground adapter; borrows the existing profile resource owner.

No binding store, scheduler, lifecycle authority or E2B client is created here.
Compose with the RetainedSessionService and RetainedSandboxService for the same
trusted profile. The socket server must run on the resource owner's event loop.
Attachments and replay fences are bounded, process-local connection state, not a
durable command ledger. Restart invalidates attachments and never replays work.
"""
import asyncio
from pathlib import Path, PurePosixPath
import secrets
from uuid import NAMESPACE_URL, uuid5
import json

from app.sandbox.models import CommandRequest, SandboxError
from .terminal_transport import WORKSPACE, validate_terminal_request


class RetainedTerminalAdapter:
    def __init__(self, *, sessions, sandboxes, callbacks=None):
        if sessions.bindings.owner_id != sandboxes.bindings.owner_id:
            raise ValueError("Session and sandbox owners differ")
        self.profile_home = Path(sessions.bindings.profile_home).resolve()
        self.sessions, self.sandboxes = sessions, sandboxes
        if callbacks is not None and callbacks.events.resources is not sandboxes.bindings:
            raise ValueError("Callback installer must use the same resource bindings")
        self.callbacks = callbacks
        self._attachments = {}
        self._seen = set()

    def close(self):
        """Invalidate proxies only; never pause, resume, kill or delete resources."""
        self._attachments.clear()

    @staticmethod
    def _error(code, *, unknown=False):
        return {"status": "unknown" if unknown else "refused", "error": {"code": code,
            "message": ("Command outcome unknown; the provider bound still applies. Do not resubmit blindly."
                        if unknown else "Foreground request refused; no command was submitted by this request.")}}

    async def invoke(self, *, operation, context, attachment, arguments):
        attempted = False
        try:
            validate_terminal_request(dict(version=1, channel="terminal", operation=operation,
                context=context, attachment=attachment, arguments=arguments))
            if operation == "detach":
                self._attachments.pop(attachment, None)
                return {"status": "detached"}
            binding = await self.sessions.caller(context["native_session_id"])
            if operation == "attach":
                if len(self._attachments) >= 256:
                    return self._error("terminal_attachment_limit")
                token = secrets.token_urlsafe(32)
                self._attachments[token] = binding.logical_id
                return {"status": "attached", "attachment": token}
            if attachment not in self._attachments:
                return self._error("terminal_attachment_expired")
            if self._attachments[attachment] != binding.logical_id:
                return self._error("terminal_attachment_mismatch")

            cwd = arguments["cwd"]
            if not isinstance(cwd, str) or not cwd or "\0" in cwd or ".." in PurePosixPath(cwd).parts:
                raise ValueError("Invalid workspace directory")
            path = PurePosixPath(cwd)
            if path.is_absolute():
                cwd = str(path.relative_to(WORKSPACE))
            timeout = arguments["timeout"]
            if type(timeout) is not int or not 1 <= timeout <= 300:
                raise ValueError("Invalid foreground observation bound")
            # Admission is deliberately distinct from turn and input, and is
            # retained in the command identity. Compaction tips resolve above.
            key = (binding.logical_id, context["admission_id"], context["input_id"], context["call_id"])
            operation_id = uuid5(NAMESPACE_URL, json.dumps(key, separators=(",", ":")))
            request = CommandRequest(operation_key=str(operation_id), command=arguments["command"], cwd=cwd)
            if key in self._seen:
                return self._error("terminal_call_already_attempted", unknown=True)
            if len(self._seen) >= 4096:
                return self._error("terminal_call_limit")
            # Fence before any await or provider effect, including a lost start ACK.
            self._seen.add(key)
            async with self.sandboxes.use(binding.logical_id) as handle:
                if self.callbacks is not None:
                    await self.callbacks.install(self.sandboxes.provider, handle)
                attempted = True
                command = await self.sandboxes.provider.start(handle, request.command,
                    operation_id=operation_id, cwd=request.cwd)
            # Do not hold the resource lifecycle mutex for the command lifetime.
            # Explicit maintenance can pause it; an interrupted observation is
            # unknown, never evidence that pause preserved or terminated a PID.
            result = await command.wait(timeout_seconds=min(timeout, handle.policy.command_seconds + 2))
            return {"status": "completed", "output": result.stdout + result.stderr,
                    "returncode": result.exit_code, "operation_id": str(operation_id),
                    "command_timeout_seconds": handle.policy.command_seconds}
        except asyncio.CancelledError:
            raise
        except SandboxError as exc:
            return self._error(exc.code, unknown=attempted)
        except Exception:
            return self._error("terminal_execution_uncertain" if attempted else "terminal_request_invalid",
                               unknown=attempted)
