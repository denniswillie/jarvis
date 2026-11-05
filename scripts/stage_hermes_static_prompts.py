#!/usr/bin/env python3
"""Stage one pinned Hermes 0.22.6 / Hindsight 1.2.3 compatibility patch.

Default: read two vendor files, save private originals and patched copies, and
write a hash manifest. Never touch the live files. Main may later use --apply
--offline at a coordinated gateway stop, then launch at the SAME source/profile
paths with JARVIS_STATIC_INSTRUCTIONS_ONLY=1. This is our flag, not upstream's.

No launcher, configuration, reservation, database or history is read or changed.
Explicit caller policy/SOUL and static capability guidance remain authoritative;
this is not a sanitizer for arbitrary operator-authored policy/plugin prose.
Existing cached prompts are not rewritten. Disable the flag for legacy behavior;
restore the two originals while offline to remove the patch entirely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


FLAG = "JARVIS_STATIC_INSTRUCTIONS_ONLY"
REVISION = "hermes-0.22.6-hindsight-1.2.3-static-instructions-v2-clock"
REPO = Path(__file__).resolve().parents[1]
TARGETS = {
    "hermes": (
        ".jarvis-dev/hermes-session-h0.Ld4B6Z/source/agent/system_prompt.py",
        "8bdab617accd5ae04335c6588638336d377174a468f6ff5d7100cc8098c68b20",
    ),
    "hindsight": (
        ".jarvis-dev/integration-demo/profile/plugins/hindsight/__init__.py",
        "1b1436b80e271be8915a743e696fd63e590ab0938ffff5fb581fb1882cf1a10a",
    ),
}


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("Pinned compatibility anchor no longer matches exactly once")
    return source.replace(old, new, 1)


def patch_hermes(source: str) -> str:
    source = _once(source, "logger = logging.getLogger(__name__)\n", '''logger = logging.getLogger(__name__)


def _jarvis_static_instructions_only() -> bool:
    # Repo-owned, opt-in compatibility for this pinned release only.
    return os.environ.get("JARVIS_STATIC_INSTRUCTIONS_ONLY") == "1"

''')
    source = _once(source, "    _agent_home_path = _agent_home(agent)\n", '''    if _jarvis_static_instructions_only():
        return (
            "Each Hermes profile has separate skills, plugins, cron and memories. "
            "Do not modify another profile's data unless the user explicitly directs you to."
        )
    _agent_home_path = _agent_home(agent)
''')
    source = _once(source,
        "    from hermes_time import get_timezone as _hermes_tz, now as _hermes_now, safe_strftime\n    now = _hermes_now()\n",
        '''    from hermes_time import get_timezone as _hermes_tz, now as _hermes_now, safe_strftime
    now = _hermes_now()
    if _jarvis_static_instructions_only():
        return f"Clock (as of prompt build): {now.isoformat(timespec='seconds')} ({_hermes_tz()})"
''')
    source = _once(source, '    if agent.provider != "alibaba":\n',
        '    if _jarvis_static_instructions_only() or agent.provider != "alibaba":\n')
    source = _once(source,
        '    if getattr(agent, "_environment_probe", True):\n',
        '    if not _jarvis_static_instructions_only() and getattr(agent, "_environment_probe", True):\n')
    source = _once(source, '            parts.append(epoch_line(_agent_home(agent)))\n',
        '''            if not _jarvis_static_instructions_only():
                parts.append(epoch_line(_agent_home(agent)))
''')
    source = _once(source,
        "        cwd = resolve_context_cwd()\n        cwd_key = _workspace_pin_key()\n",
        '''        cwd = resolve_context_cwd()
        if _jarvis_static_instructions_only():
            # Empty replay bypasses workspace probing, without changing the pin/history.
            return coding_system_prompt_parts(
                platform=agent.platform, cwd=cwd, model=agent.model,
                valid_tool_names=agent.valid_tool_names, workspace_block="",
            )
        cwd_key = _workspace_pin_key()
''')
    source = _once(source, "    if agent.skip_context_files:\n",
        "    if _jarvis_static_instructions_only() or agent.skip_context_files:\n")
    source = _once(source,
        '    if agent._memory_store:\n        for enabled, kind in ((agent._memory_enabled, "memory"), (agent._user_profile_enabled, "user")):\n',
        '    if not _jarvis_static_instructions_only() and agent._memory_store:\n        for enabled, kind in ((agent._memory_enabled, "memory"), (agent._user_profile_enabled, "user")):\n')
    return _once(source, "    environment_hints = _pb.build_environment_hints()\n",
        '    environment_hints = "" if _jarvis_static_instructions_only() else _pb.build_environment_hints()\n')


def patch_hindsight(source: str) -> str:
    source = _once(source,
        '        return f"# Hindsight Memory\\nActive{label}. Bank: {self._bank_id}, budget: {self._budget}.\\n{tail}"\n',
        '''        if os.environ.get("JARVIS_STATIC_INSTRUCTIONS_ONLY") == "1":
            return f"# Hindsight Memory\\n{tail}"
        return f"# Hindsight Memory\\nActive{label}. Bank: {self._bank_id}, budget: {self._budget}.\\n{tail}"
''')
    return _once(source,
        '        kwargs: dict = {"query": query, "budget": self._budget, "max_tokens": self._recall_max_tokens}\n',
        '''        kwargs: dict = {"query": query, "budget": self._budget, "max_tokens": self._recall_max_tokens}
        if os.environ.get("JARVIS_STATIC_INSTRUCTIONS_ONLY") == "1":
            # One current UTC anchor per query, shared across banks and retries.
            # Pinned SDK arecall supports this API field; it is never policy text.
            kwargs["query_timestamp"] = datetime.now(timezone.utc).isoformat()
''')


def patched(name: str, original: bytes) -> bytes:
    if digest(original) != TARGETS[name][1]:
        raise ValueError(f"{name}: original hash does not match the pinned release")
    patch = patch_hermes if name == "hermes" else patch_hindsight
    return patch(original.decode("utf-8")).encode("utf-8")


def _write_private(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(content)


def stage(repo: Path, output: Path) -> dict:
    originals = {name: (repo / path).read_bytes() for name, (path, _) in TARGETS.items()}
    replacements = {name: patched(name, content) for name, content in originals.items()}
    # Both preimages and every replacement anchor must pass before writing anything.
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (output / "originals").mkdir(mode=0o700)
    (output / "patched").mkdir(mode=0o700)
    files = []
    for name, (path, before) in TARGETS.items():
        _write_private(output / "originals" / f"{name}.py", originals[name])
        _write_private(output / "patched" / f"{name}.py", replacements[name])
        files.append({"name": name, "target": path, "before_sha256": before,
                      "after_sha256": digest(replacements[name])})
    manifest = {"revision": REVISION, "status": "staged_not_applied",
                "launch_env": {FLAG: "1"}, "files": files}
    _write_private(output / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


def apply_offline(repo: Path, output: Path, *, offline: bool = False) -> dict:
    if not offline:
        raise ValueError("Apply requires --offline: Main must first stop the gateway")
    pending = []
    for name, (relative, before) in TARGETS.items():
        original = (output / "originals" / f"{name}.py").read_bytes()
        expected = patched(name, original)
        if (output / "patched" / f"{name}.py").read_bytes() != expected:
            raise ValueError(f"{name}: staged copy does not match this pinned patch")
        target = repo / relative
        current = digest(target.read_bytes())
        if current not in (before, digest(expected)):
            raise ValueError(f"{name}: live file drifted; neither original nor this patch")
        pending.append((name, target, expected, current == before))
    # Fixed targets only; never trust paths from the manifest. Both files preflighted.
    receipt = {"revision": REVISION, "status": "partial", "replaced": [],
               "already_applied": [name for name, _, _, needed in pending if not needed]}
    try:
        for name, target, content, needed in pending:
            if not needed:
                continue
            temp = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                    temp = Path(stream.name)
                    os.chmod(temp, target.stat().st_mode & 0o777)
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, target)
                receipt["replaced"].append(name)
            finally:
                if temp is not None:
                    temp.unlink(missing_ok=True)
        receipt["status"] = "applied_offline"
    finally:
        receipt_path = output / "apply-receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        receipt_path.chmod(0o600)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Operator confirms gateway is stopped")
    args = parser.parse_args(argv)
    try:
        receipt = apply_offline(REPO, args.output, offline=args.offline) if args.apply else stage(REPO, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
