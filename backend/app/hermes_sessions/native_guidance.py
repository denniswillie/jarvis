"""Narrow v0.22.6 compatibility for native queued steering of our master.

Hermes stores programmatically created CLI sessions as ``local`` although the
frozen launch policy remains ``cli``. Its queued-control eligibility reads only
the stored label. Do not broaden all local sessions: our explicitly opted-in
consumer understands transferred admissions and does not expect a second result.
Native admission, authorization, delivery and settlement remain unchanged.
"""
import json
from pathlib import Path

from .bindings import BindingError, SessionBindings


def enable_live_guidance(home: Path) -> None:
    config = json.loads((home / "jarvis-adapter.json").read_text())
    if config.get("native_live_guidance") is not True:
        return
    from gateway import session_queued_controls as native
    from gateway.session_policy import policy_for_source

    if getattr(native.queued_controls, "_jarvis_live_guidance", False):
        return
    bindings = SessionBindings(home / "jarvis" / "bindings.sqlite",
        owner_id=config["owner_id"], profile_home=home)
    original = native.queued_controls

    def queued_controls(authority, row):
        controls = original(authority, row)
        if controls:
            return controls
        payload = row.get("payload") or {}
        text = payload.get("text")
        if (row.get("status") != "queued" or set(payload) - native._TEXT_ONLY
                or not isinstance(text, str) or not text.strip()
                or (payload.get("display_v1") or {}).get("kind") == "hidden"):
            return []
        session_id = row["target_session_id"]
        try:
            binding = bindings.by_native(session_id)
        except BindingError:
            return []
        live = authority.sessions.get(session_id)
        if not binding.is_master or live is None or live.source is None:
            return []
        policy = policy_for_source(authority.runner, live.source)
        return ["steer"] if policy is not None and policy.source == "cli" else []

    queued_controls._jarvis_live_guidance = True
    native.queued_controls = queued_controls
