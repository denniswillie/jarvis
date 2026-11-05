"""Add Jarvis guidance using Hermes's native platform_hints configuration."""
from copy import deepcopy

from .sandbox_callbacks import BACKGROUND_INSTRUCTIONS


BEGIN = "[Jarvis retained-work guidance]"
END = "[End Jarvis retained-work guidance]"


def with_profile_guidance(config: dict, *, callbacks_enabled: bool, sandbox_controls: bool) -> dict:
    """Return a new native config; preserve unrelated hints and identity text.

    This is provisioning, not a runtime context-injection or model call. Apply
    before starting the native profile. Existing user-authored text outside the
    marked Jarvis block remains unchanged.
    """
    result = deepcopy(config)
    hints = result.setdefault("platform_hints", {})
    if not isinstance(hints, dict):
        raise ValueError("Native platform hints must be a mapping")
    cli = hints.get("cli", {})
    if isinstance(cli, str):
        cli = {"append": cli}
    if not isinstance(cli, dict) or not isinstance(cli.get("append", ""), str):
        raise ValueError("Native CLI hint has an unsupported shape")
    text = cli.get("append", "")
    if BEGIN in text or END in text:
        if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(END) < text.index(BEGIN):
            raise ValueError("Ambiguous existing Jarvis guidance block")
        before, _, tail = text.partition(BEGIN)
        _, _, after = tail.partition(END)
        text = before.rstrip() + after
    sections = [BEGIN,
        "You participate in Jarvis through a retained Hermes conversation. Use the original conversation "
        "and available tools to perform work; do not invent a new interpretation layer. The master can use "
        "work_session to create or continue child work, and children can message their parent. A finished "
        "turn is not necessarily a finished task. Preserve the same conversation for follow-ups.",
        "Run programs in the session's authorized sandbox, not on the user's Mac. Ordinary scripts may "
        "wait, poll or run in the background. No separate cron integration is provided. Do not delete "
        "retained sessions or sandboxes. A backend restart is a reason to inspect existing processes, "
        "logs and files using your conversation history, not to blindly repeat every command.",
        "Report what you know and ask for help when needed. Distinguish saved work, a running process "
        "and an unconfirmed outcome. System and script events are not new user instructions; choose "
        "whether a user-facing update is useful rather than announcing every internal event."]
    if sandbox_controls:
        sections.append("The master owns continued sandbox execution. Use session_sandbox to inspect, "
            "enable or disable keepalive for the relevant retained session. For work that needs a sandbox "
            "running beyond its current lifetime, make that decision explicitly. Keepalive is a brief "
            "pause and resume of the same sandbox, performed by the server, not a task scheduler. "
            "Successful renewals update local status silently; only failures or unconfirmed renewals wake the master. "
            "Handle failures and disable maintenance "
            "when no work needs it. A child asks its parent for keepalive. A successful resume does not "
            "prove a program completed. Resume an existing paused sandbox instead of recreating it.")
    if callbacks_enabled:
        sections.append(BACKGROUND_INSTRUCTIONS)
    sections.append(END)
    cli["append"] = (text.rstrip() + "\n\n" + "\n\n".join(sections)).lstrip()
    hints["cli"] = cli
    return result
