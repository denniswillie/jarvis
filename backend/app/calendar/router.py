from __future__ import annotations

from typing import Annotated

import asyncpg
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.calendar.google_oauth import GoogleCalendarError, GoogleOAuthService
from app.calendar.models import GoogleCalendarStatusResponse, GoogleOAuthStartResponse
from app.calendar.repository import GoogleOAuthTokenRepository
from app.db import get_pool
from app.errors import JarvisHTTPError
from app.settings import JarvisSettings, get_settings


router = APIRouter(prefix="/oauth/google", tags=["google-calendar"])


def get_google_token_repository(
    pool: Annotated[asyncpg.Pool, Depends(get_pool)],
) -> GoogleOAuthTokenRepository:
    return GoogleOAuthTokenRepository(pool)


@router.get("/start", response_model=GoogleOAuthStartResponse)
async def start_google_oauth(
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    repository: Annotated[GoogleOAuthTokenRepository, Depends(get_google_token_repository)],
    settings: Annotated[JarvisSettings, Depends(get_settings)],
) -> GoogleOAuthStartResponse:
    try:
        authorization_url, expires_in_seconds = GoogleOAuthService(
            settings=settings,
            repository=repository,
        ).authorization_url(owner_id=principal.owner_id)
    except GoogleCalendarError as error:
        raise JarvisHTTPError(503, error.code, error.message) from error
    return GoogleOAuthStartResponse(
        authorization_url=authorization_url,
        expires_in_seconds=expires_in_seconds,
    )


@router.get("/status", response_model=GoogleCalendarStatusResponse)
async def get_google_oauth_status(
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    repository: Annotated[GoogleOAuthTokenRepository, Depends(get_google_token_repository)],
) -> GoogleCalendarStatusResponse:
    token = await repository.get_token(owner_id=principal.owner_id)
    if token is None:
        return GoogleCalendarStatusResponse(connected=False)
    return GoogleCalendarStatusResponse(
        connected=True,
        expires_at=token.expires_at,
        scopes=token.scopes,
    )


@router.get("/callback", response_model=None)
async def google_oauth_callback(
    repository: Annotated[GoogleOAuthTokenRepository, Depends(get_google_token_repository)],
    settings: Annotated[JarvisSettings, Depends(get_settings)],
    state: Annotated[str | None, Query()] = None,
    code: Annotated[str | None, Query()] = None,
    error: Annotated[str | None, Query()] = None,
) -> RedirectResponse | HTMLResponse:
    if error or not state or not code:
        return _oauth_error_page("Google Calendar connection was cancelled or incomplete.")
    try:
        await GoogleOAuthService(settings=settings, repository=repository).exchange_callback(
            state=state,
            code=code,
        )
    except GoogleCalendarError as oauth_error:
        return _oauth_error_page(oauth_error.message)
    return RedirectResponse("jarvis://oauth/google/connected", status_code=302)


def _oauth_error_page(message: str) -> HTMLResponse:
    safe_message = (
        message
        if message
        in {
            "Google Calendar authorization could not be verified.",
            "Google Calendar authorization expired. Start the connection again.",
            "Google Calendar authorization could not be completed.",
            "Google Calendar authorization is not configured.",
        }
        else "Google Calendar could not be connected. Return to Jarvis and try again."
    )
    return HTMLResponse(
        status_code=400,
        content=f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Jarvis Calendar</title><style>
body{{margin:0;background:#171713;color:#f4eedf;font-family:ui-rounded,system-ui;display:grid;min-height:100vh;place-items:center}}
main{{max-width:32rem;padding:2rem}}h1{{font-size:2.6rem;margin:0 0 1rem}}p{{color:#c7c1b2;line-height:1.5}}
</style></head><body><main><h1>Jarvis</h1><p>{safe_message}</p></main></body></html>""",
    )
