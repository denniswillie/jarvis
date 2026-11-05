"""Pure section selection and metadata cleanup; never evicts requested facts."""
from copy import deepcopy
import json


SECTIONS = ("human", "system", "all")
# Advisory only: exceeding this target must never remove facts or action IDs.
# It measures encoded bytes, not model tokens; transport limits are unchanged.
DEFAULT_FRAMED_BYTES = 880
_REPOSITORY_FIELDS = {"owner_id", "created_at", "storage_bucket", "storage_path"}


def framed_reply_bytes(value):
    """Measure the full escaped single-envelope representation used by Live.

    The fixed UUID has the same encoded length as every actual result UUID.
    Tests also run the real result packer; no protocol ceiling changes here.
    """
    text = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    frame = {"result_id": "00000000-0000-0000-0000-000000000000",
             "kind": "delegation_result", "part": 1, "complete": False, "text": text}
    return len(json.dumps(frame, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))


def _clean(value, *, system=False):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if system and key in _REPOSITORY_FIELDS:
                continue
            if key == "availability" and item == "open":
                continue
            item = _clean(item, system=system)
            if item is None or item == "" or item == [] or item == {}:
                continue
            if key in {"sources", "source_document_ids"} and isinstance(item, list):
                item = list(dict.fromkeys(item))
            result[key] = item
        return result
    if isinstance(value, list):
        return [_clean(item, system=system) for item in value]
    return deepcopy(value)


def compact_live_state_reply(snapshot, section="all", *, max_framed_bytes=DEFAULT_FRAMED_BYTES):
    """Select one RAM snapshot; keep facts, timestamps and identities verbatim.

    The upstream snapshot is already bounded. This layer never applies a second
    semantic eviction policy: every requested fact, qualifier and actionable ID
    survives even when more packets are required. Size is advisory only.
    Unknown legacy snapshot shapes pass through only for ``all``; a section
    selector cannot safely interpret their opaque text.
    """
    if section not in SECTIONS or type(max_framed_bytes) is not int or max_framed_bytes < 115:
        raise ValueError("Invalid Live State section or encoded-byte target")
    if not isinstance(snapshot, dict):
        raise ValueError("Live State must be a JSON object")
    if "human" not in snapshot or "system" not in snapshot:
        if section != "all":
            raise ValueError("This Live State snapshot does not expose selectable sections")
        return deepcopy(snapshot)
    reply = _clean({key: value for key, value in snapshot.items() if key not in {"human", "system", "location"}})
    if section in ("human", "all") and snapshot.get("location") is not None:
        cleaned_location = _clean(snapshot["location"], system=False)
        if cleaned_location:
            reply["location"] = cleaned_location
    for name in ("human", "system"):
        if section not in (name, "all"):
            continue
        value = snapshot[name]
        if value is not None and not isinstance(value, dict):
            raise ValueError("Live State sections must be objects or explicitly unavailable")
        reply[name] = _clean(value, system=name == "system")
        # None is unavailability, not a proven empty state.
        if isinstance(value, dict) and name == "system":
            # Work precedes connection metadata. Remove only an identical
            # duplicate; preserve canonical IDs and nonidentical action views.
            if "work" in reply[name]:
                reply[name] = {"work": reply[name]["work"],
                               **{key: item for key, item in reply[name].items() if key != "work"}}
            app = value.get("app")
            if (isinstance(app, dict) and value.get("sandboxes") is not None
                    and app.get("sandboxes") == value["sandboxes"]):
                reply[name].get("app", {}).pop("sandboxes", None)
                if not reply[name].get("app"):
                    reply[name].pop("app", None)
            for key in ("app", "sandboxes"):
                if key in value and value[key] is None:
                    reply[name][key] = None
    if framed_reply_bytes(reply) > max_framed_bytes:
        reply["over_budget"] = True
    return reply
