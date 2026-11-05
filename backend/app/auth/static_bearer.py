from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header

from app.errors import JarvisHTTPError
from app.settings import JarvisSettings, get_settings, is_configured_secret


@dataclass(frozen=True)
class JarvisPrincipal:
    owner_id: str


def authenticate_static_bearer(
    *,
    authorization: str | None,
    settings: JarvisSettings,
) -> JarvisPrincipal:
    expected_token = settings.JARVIS_STATIC_BEARER_TOKEN
    if not is_configured_secret(expected_token):
        raise JarvisHTTPError(
            503,
            "jarvis_auth_not_configured",
            "Jarvis static bearer token is not configured.",
        )

    if not authorization:
        raise JarvisHTTPError(
            401,
            "jarvis_auth_required",
            "Authorization bearer token is required.",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise JarvisHTTPError(
            401,
            "jarvis_auth_required",
            "Authorization bearer token is required.",
        )

    if not secrets.compare_digest(token, expected_token):
        raise JarvisHTTPError(
            403,
            "jarvis_auth_invalid",
            "Authorization bearer token is invalid.",
        )

    return JarvisPrincipal(owner_id=settings.JARVIS_OWNER_ID)


def require_static_bearer(
    authorization: Annotated[str | None, Header()] = None,
    settings: JarvisSettings = Depends(get_settings),
) -> JarvisPrincipal:
    return authenticate_static_bearer(
        authorization=authorization,
        settings=settings,
    )
