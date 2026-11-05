"""Native in-process foreground provider: private RPC only, no resource ownership."""
import json
from pathlib import Path
import threading

from .domain_plugin import capture_native_context
from .terminal_transport import WORKSPACE, invoke_terminal


def _current_context():
    from tools.approval_context import _approval_session_id
    return capture_native_context(_approval_session_id.get())


def _failure(code, message):
    # Pinned native _run_foreground retries raised exceptions. Return a proxy
    # failure instead, never an invented successful command exit or a retry.
    return {"returncode": 125, "output": f"Retained terminal proxy failure ({code}). {message} "
            "125 is the proxy status, not an observed command exit status."}


class RetainedTerminalEnvironment:
    def __init__(self, profile_home, *, timeout=180):
        self.profile_home = Path(profile_home).resolve()
        self.cwd, self.timeout = WORKSPACE, timeout
        self._session_scoped = True
        self._attachment = None
        self._closed = False
        self._lock = threading.Lock()

    def execute(self, command, cwd="", *, timeout=None, stdin_data=None,
                rewrite_compound_background=True, bounded_capture=False, yield_handler=None):
        if (bounded_capture is not True or rewrite_compound_background is not True
                or stdin_data is not None or yield_handler is not None):
            return _failure("terminal_foreground_only",
                            "Background, polling, stdin, staging and internal execution are not qualified; nothing submitted.")
        with self._lock:
            try:
                if self._closed:
                    return _failure("terminal_detached", "This proxy was detached; nothing submitted.")
                context = _current_context()  # Fresh for EVERY execute, not captured at env creation.
                for attempt in range(2):
                    if self._attachment is None:
                        attached = invoke_terminal(self.profile_home, operation="attach", context=context)
                        if attached.get("status") != "attached":
                            return self._failure_result(attached)
                        self._attachment = attached["attachment"]
                    result = invoke_terminal(self.profile_home, operation="execute", context=context,
                        attachment=self._attachment, arguments={"command": command, "cwd": cwd or WORKSPACE,
                            "timeout": self.timeout if timeout is None else timeout})
                    if (result.get("status") == "refused"
                            and result.get("error", {}).get("code") == "terminal_attachment_expired"):
                        # The server rejected this request before submitting a
                        # command. Reconnect once; never retry unknown outcomes.
                        self._attachment = None
                        continue
                    break
                if result.get("status") != "completed":
                    return self._failure_result(result)
                # Cwd/shell variables do not persist implicitly in this bounded
                # provider path. Do not emit native cwd_observed without evidence.
                return {"output": result["output"], "returncode": result["returncode"]}
            except Exception:
                return _failure("terminal_context_or_transport_unavailable",
                                "No complete result; effects may have occurred. Do not resubmit blindly.")

    @staticmethod
    def _failure_result(result):
        error = result.get("error", {})
        return _failure(error.get("code", "terminal_unknown"),
                        error.get("message", "No complete result; do not resubmit blindly."))

    def cleanup(self):
        with self._lock:
            self._closed = True
            token, self._attachment = self._attachment, None
            if token is not None:
                try:
                    invoke_terminal(self.profile_home, operation="detach", attachment=token, timeout=2)
                except Exception:
                    pass  # Detach uncertainty never authorizes lifecycle operations.

    def get_temp_dir(self):
        # Native spawn_via_env asks for this before submitting its launch wrapper.
        raise RuntimeError("Retained terminal background/staging is not qualified")

    def fetch_file(self, *args, **kwargs):
        raise RuntimeError("Retained terminal file transfer is not qualified")


def register_terminal_provider(ctx):
    from hermes_constants import get_hermes_home
    home = Path(get_hermes_home()).resolve()
    try:
        config = json.loads((home / "jarvis-adapter.json").read_text())
    except FileNotFoundError:
        return
    if config.get("retained_terminal") is not True:
        return
    from agent.terminal_env_provider import TerminalEnvironmentProvider

    class RetainedTerminalProvider(TerminalEnvironmentProvider):
        name = "jarvis_e2b"
        session_isolated_when_nonpersistent = True
        supports_streams = False
        skip_container_guards = False
        default_cwd = WORKSPACE
        strip_env_keys = frozenset({"E2B_API_KEY"})
        env_description = (
            "a retained Jarvis workspace (foreground terminal calls; programs may launch detached "
            "children with Python subprocess.Popen and use the supplied wake helper; native "
            "background=true/process-manager and file-transfer integration are unavailable). "
            "For interactive browser work, use xdotool type --delay 0 and batch focus, input/action "
            "and screenshot in one command. No blanket multi-second sleeps: check readiness at "
            "short intervals with a deadline, stopping as soon as ready. Inspect the screenshot "
            "before claiming success; do not repeat a successful action."
        )

        def is_available(self):
            # Catalog availability means configured, not currently connected.
            # Native Hermes caches this at startup. The RPC reports an honest
            # transport error if the separately started backend is not ready.
            return True

        def create_environment(self, *, cwd, timeout, task_id="default", image=None,
                               container_config=None, **kwargs):
            # task_id/cwd/image are NOT identity or allocation authority.
            return RetainedTerminalEnvironment(home, timeout=timeout)

    ctx.register_terminal_environment_provider(RetainedTerminalProvider())
