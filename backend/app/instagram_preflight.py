"""Read-only discovery for Meta's Facebook Login Instagram route.

This does not connect/store credentials, qualify publishing permissions, select
an account, or publish. It deliberately does not guess an Instagram Login
endpoint or reuse that route's token on Facebook. The caller supplies the
explicit API version and a Facebook User token obtained through its authorized
connection flow. Official request:
https://www.postman.com/meta/instagram/documentation/6yqw8pt/instagram-api
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
import re
from typing import Literal

import httpx
from pydantic import SecretStr


_ID = re.compile(r"[0-9]{1,32}\Z")
_VERSION = re.compile(r"v[1-9][0-9]?\.0\Z")
_MAX_BODY = 256 * 1024
_MAX_PAGES = 3


@dataclass(frozen=True, slots=True)
class InstagramCandidate:
    page_id: str
    page_name: str
    instagram_user_id: str
    # A Page task observation is not an OAuth publishing permission.
    page_tasks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InstagramDiscovery:
    route: Literal["facebook_login"]
    outcome: Literal["connection_required", "observed", "unknown"]
    reason: str
    candidates: tuple[InstagramCandidate, ...] = ()
    enumeration_complete: bool = False
    # Must remain unknown until a separate exact-route permission check.
    publication_readiness: Literal["unknown"] = "unknown"


class _UnreadableResponse(Exception):
    """Private; never contains provider text or request credentials."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _UnreadableResponse()
        result[key] = value
    return result


def _candidate(row: object) -> InstagramCandidate | None:
    if not isinstance(row, dict):
        raise _UnreadableResponse()
    linked = row.get("instagram_business_account")
    if linked is None:
        return None
    if not isinstance(linked, dict):
        raise _UnreadableResponse()
    page_id, ig_id, name = row.get("id"), linked.get("id"), row.get("name")
    tasks = row.get("tasks", [])
    if (not isinstance(page_id, str) or not _ID.fullmatch(page_id)
            or not isinstance(ig_id, str) or not _ID.fullmatch(ig_id)
            or not isinstance(name, str) or len(name) > 512
            or not isinstance(tasks, list) or len(tasks) > 32
            or any(not isinstance(task, str) or not re.fullmatch(r"[A-Z_]{1,64}", task)
                   for task in tasks)):
        raise _UnreadableResponse()
    return InstagramCandidate(page_id, name, ig_id, tuple(sorted(set(tasks))))


def _unknown(reason: str) -> InstagramDiscovery:
    return InstagramDiscovery("facebook_login", "unknown", reason)


async def discover_facebook_linked_instagram(
    *, api_version: str, user_token: SecretStr | None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> InstagramDiscovery:
    """Inspect at most three pages/12 seconds; no mutation or automatic retry.

    The optional transport is for the existing application's HTTP wiring and
    isolated tests. It must be trusted, like the caller supplying the token.
    No raw provider response, error message, URL, cursor or credential escapes
    into the result. Pagination rebuilds the fixed endpoint from an opaque
    cursor; it never follows a provider-supplied URL.
    """
    if not isinstance(api_version, str) or not _VERSION.fullmatch(api_version):
        raise ValueError("An explicit Graph API version such as v22.0 is required")
    if user_token is None:
        return InstagramDiscovery("facebook_login", "connection_required", "token_missing")
    if not isinstance(user_token, SecretStr):
        raise TypeError("Pass a SecretStr from the authorized Facebook Login connection")
    token = user_token.get_secret_value()
    if not token:
        return InstagramDiscovery("facebook_login", "connection_required", "token_missing")
    if len(token) > 8192 or any(ord(char) < 33 or ord(char) > 126 for char in token):
        raise ValueError("The supplied credential has an invalid format")

    endpoint = f"https://graph.facebook.com/{api_version}/me/accounts"
    params = {"fields": "id,name,tasks,instagram_business_account", "limit": "100"}
    candidates: dict[str, InstagramCandidate] = {}
    cursors: set[str] = set()
    try:
        async with asyncio.timeout(12):
            async with httpx.AsyncClient(
                transport=transport, trust_env=False, follow_redirects=False,
                timeout=httpx.Timeout(5), limits=httpx.Limits(max_connections=1),
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            ) as client:
                for _ in range(_MAX_PAGES):
                    async with client.stream("GET", endpoint, params=params) as response:
                        if response.status_code != 200:
                            # Even a 400/401 may be token, scope, version or route
                            # trouble. It says nothing conclusive about account type.
                            return _unknown("provider_request_rejected")
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > _MAX_BODY:
                                return _unknown("response_limit_exceeded")
                    payload = json.loads(body, object_pairs_hook=_unique_object)
                    if not isinstance(payload, dict) or "error" in payload:
                        raise _UnreadableResponse()
                    rows = payload.get("data")
                    if not isinstance(rows, list) or len(rows) > 100:
                        raise _UnreadableResponse()
                    for row in rows:
                        candidate = _candidate(row)
                        if candidate is not None:
                            prior = candidates.get(candidate.page_id)
                            if prior is not None and prior != candidate:
                                return _unknown("conflicting_account_observations")
                            candidates[candidate.page_id] = candidate
                    paging = payload.get("paging", {})
                    if not isinstance(paging, dict):
                        raise _UnreadableResponse()
                    if not paging.get("next"):
                        return InstagramDiscovery(
                            "facebook_login", "observed",
                            "linked_accounts_found" if candidates else "no_linked_accounts_observed",
                            tuple(candidates.values()), True,
                        )
                    page_cursors = paging.get("cursors")
                    cursor = page_cursors.get("after") if isinstance(page_cursors, dict) else None
                    if not isinstance(cursor, str) or not 1 <= len(cursor) <= 2048:
                        raise _UnreadableResponse()
                    if cursor in cursors:
                        return _unknown("pagination_did_not_advance")
                    cursors.add(cursor)
                    params["after"] = cursor
                return InstagramDiscovery("facebook_login", "observed", "page_limit_reached",
                                          tuple(candidates.values()), False)
    except (httpx.HTTPError, TimeoutError):
        return _unknown("provider_unavailable")
    except (ValueError, UnicodeError, RecursionError, _UnreadableResponse):
        return _unknown("provider_response_unreadable")


def main() -> None:
    """Operator preflight only; credentials are read privately from the environment."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-version", required=True)
    parser.add_argument("--token-env", default="JARVIS_INSTAGRAM_FACEBOOK_USER_TOKEN")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", args.token_env):
        parser.error("Use an environment variable name, not a token")
    raw = os.environ.get(args.token_env)
    result = asyncio.run(discover_facebook_linked_instagram(
        api_version=args.api_version, user_token=SecretStr(raw) if raw else None))
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
