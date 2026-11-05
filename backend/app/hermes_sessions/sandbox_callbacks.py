"""Install the small script callback helper using existing immutable file staging."""
import hashlib
import json
from pathlib import Path
import shlex
from urllib.parse import urlsplit
from uuid import uuid4

from app.sandbox.models import SandboxError
from .terminal_transport import WORKSPACE


BACKGROUND_INSTRUCTIONS = (
    "You may start ordinary background programs in your sandbox and save their logs/results there. "
    "Write the program to a file, then launch it from a short foreground Python command using "
    "subprocess.Popen with start_new_session=True, stdin=subprocess.DEVNULL and stdout/stderr "
    "redirected to log files. Do not use the native terminal background=true/process manager: "
    "that path is not supported by this retained backend. Shell nohup and trailing '&' are "
    "redirected by the native terminal to that unsupported path. Your program owns its work "
    "and calls the helper only when it needs to wake this conversation. "
    "To wake this same conversation, call: python3 " + WORKSPACE + "/.jarvis/wake_agent.py "
    "--event-id YOUR_STABLE_EVENT_ID --message 'A short result or path to saved output'. "
    "Use --message-file for a small text message instead. After uncertain delivery, retry the same "
    "event ID and unchanged message. Admission is not proof the agent processed the message. "
    "The adjacent wake.json is a private callback credential: do not print or share it. "
    "Programs continue independently of your current turn. On a restart notice, inspect existing "
    "processes and files before deciding how to continue; do not blindly relaunch them."
)


class SandboxCallbacks:
    def __init__(self, *, events, endpoint: str):
        url = urlsplit(endpoint)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.fragment or url.query or not url.path.endswith("/hermes/background-wake")):
            raise ValueError("A plain public HTTPS background-wake endpoint is required")
        self.events, self.endpoint = events, endpoint
        self._helper = Path(__file__).with_name("sandbox_wake_client.py").read_bytes()
        self._installed = set()

    async def install(self, provider, handle):
        """Caller holds the existing resource-use lock; no allocation/resume here.

        Credentials travel as file bytes, never command arguments or model tool
        output. Files are readable by the session's own sandbox programs. This
        is not a boundary against code already authorized to run in that sandbox.
        """
        session_id = handle.scope.logical_session_id
        resource = self.events._resource(session_id)
        if resource.scope != handle.scope or resource.sandbox_id != handle.sandbox_id:
            raise SandboxError("sandbox_callback_scope_mismatch")
        config = json.dumps({"endpoint": self.endpoint, "session_id": session_id,
                             "token": self.events.credential(session_id)}, sort_keys=True).encode()
        helper_id = hashlib.sha256(self._helper).hexdigest()
        config_id = hashlib.sha256(config).hexdigest()
        key = (handle.sandbox_id, helper_id, config_id)
        if key in self._installed:
            return
        helper = await provider.stage_input(handle, helper_id, self._helper)
        credential = await provider.stage_input(handle, config_id, config)
        # Names are in the existing sandbox workspace. Immutable staged sources
        # and idempotent symlink setup make a lost setup ACK safe to retry.
        command = ("set -eu; mkdir -p .jarvis; "
                   "ln -sfnT " + shlex.quote(helper["path"]) + " .jarvis/wake_agent.py; "
                   "ln -sfnT " + shlex.quote(credential["path"]) + " .jarvis/wake.json; "
                   "printf jarvis.callback.ready.v1")
        setup = await provider.start(handle, command, operation_id=uuid4())
        result = await setup.wait(timeout_seconds=20)
        if result.exit_code != 0 or result.stdout != "jarvis.callback.ready.v1":
            raise SandboxError("sandbox_callback_setup_failed")
        self._installed.add(key)
