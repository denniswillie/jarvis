from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

from app.calendar.models import GoogleOAuthToken
from app.calendar.repository import GoogleOAuthTokenRepository
from app.settings import JarvisSettings, is_configured_secret


GOOGLE_OAUTH_STATE_PURPOSE = "google_calendar_oauth"
GOOGLE_OAUTH_STATE_TTL_SECONDS = 600
GOOGLE_TOKEN_REFRESH_WINDOW_SECONDS = 60
GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"


class GoogleCalendarError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class GoogleOAuthStateCodec:
    def __init__(self, secret: str, *, ttl_seconds: int = GOOGLE_OAUTH_STATE_TTL_SECONDS) -> None:
        self._secret = secret.encode("utf-8")
        self._ttl_seconds = ttl_seconds

    def encode(self, *, owner_id: str, now: int | None = None) -> str:
        payload = {
            "owner_id": owner_id,
            "issued_at": int(time.time()) if now is None else now,
            "nonce": secrets.token_urlsafe(18),
            "purpose": GOOGLE_OAUTH_STATE_PURPOSE,
        }
        encoded_payload = _urlsafe_encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        signature = hmac.new(self._secret, encoded_payload.encode("ascii"), hashlib.sha256).digest()
        return f"{encoded_payload}.{_urlsafe_encode(signature)}"

    def decode(self, token: str, *, now: int | None = None) -> str:
        try:
            encoded_payload, encoded_signature = token.split(".", 1)
            expected_signature = hmac.new(
                self._secret,
                encoded_payload.encode("ascii"),
                hashlib.sha256,
            ).digest()
            supplied_signature = _urlsafe_decode(encoded_signature)
            if not hmac.compare_digest(expected_signature, supplied_signature):
                raise ValueError("signature mismatch")
            payload = json.loads(_urlsafe_decode(encoded_payload))
            owner_id = payload["owner_id"]
            issued_at = payload["issued_at"]
            purpose = payload["purpose"]
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            binascii.Error,
            json.JSONDecodeError,
        ) as error:
            raise GoogleCalendarError(
                "google_oauth_state_invalid",
                "Google Calendar authorization could not be verified.",
            ) from error

        current_time = int(time.time()) if now is None else now
        if not isinstance(owner_id, str) or not owner_id:
            raise GoogleCalendarError(
                "google_oauth_state_invalid",
                "Google Calendar authorization could not be verified.",
            )
        if purpose != GOOGLE_OAUTH_STATE_PURPOSE or not isinstance(issued_at, int):
            raise GoogleCalendarError(
                "google_oauth_state_invalid",
                "Google Calendar authorization could not be verified.",
            )
        if issued_at > current_time + 60 or current_time - issued_at > self._ttl_seconds:
            raise GoogleCalendarError(
                "google_oauth_state_expired",
                "Google Calendar authorization expired. Start the connection again.",
            )
        return owner_id


class GoogleOAuthService:
    def __init__(
        self,
        *,
        settings: JarvisSettings,
        repository: GoogleOAuthTokenRepository,
    ) -> None:
        self._settings = settings
        self._repository = repository

    def authorization_url(self, *, owner_id: str) -> tuple[str, int]:
        self._require_configured()
        state = self._state_codec().encode(owner_id=owner_id)
        flow = self._flow(state=state)
        authorization_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            state=state,
        )
        return authorization_url, GOOGLE_OAUTH_STATE_TTL_SECONDS

    async def exchange_callback(self, *, state: str, code: str) -> GoogleOAuthToken:
        self._require_configured()
        owner_id = self._state_codec().decode(state)
        flow = self._flow(state=state)
        try:
            await asyncio.to_thread(flow.fetch_token, code=code)
        except Exception as error:
            raise GoogleCalendarError(
                "google_oauth_exchange_failed",
                "Google Calendar authorization could not be completed.",
            ) from error

        credentials = flow.credentials
        token_data: dict[str, Any] = dict(flow.oauth2session.token)
        scopes = _normalize_scopes(token_data.get("scope"), credentials.scopes)
        return await self._repository.upsert_token(
            owner_id=owner_id,
            access_token=credentials.token,
            refresh_token=credentials.refresh_token,
            token_type=_optional_text(token_data.get("token_type")),
            scopes=scopes or self._settings.GOOGLE_CALENDAR_SCOPES,
            expires_at=_aware_datetime(credentials.expiry),
        )

    def _state_codec(self) -> GoogleOAuthStateCodec:
        secret = self._settings.GOOGLE_OAUTH_STATE_SECRET
        if not is_configured_secret(secret):
            raise GoogleCalendarError(
                "google_calendar_not_configured",
                "Google Calendar authorization is not configured.",
            )
        return GoogleOAuthStateCodec(secret)

    def _flow(self, *, state: str | None = None) -> Flow:
        flow = Flow.from_client_config(
            _client_config(self._settings),
            scopes=self._settings.GOOGLE_CALENDAR_SCOPES,
            state=state,
        )
        flow.redirect_uri = self._settings.GOOGLE_OAUTH_REDIRECT_URI
        return flow

    def _require_configured(self) -> None:
        if not all(
            is_configured_secret(value)
            for value in (
                self._settings.GOOGLE_OAUTH_CLIENT_ID,
                self._settings.GOOGLE_OAUTH_CLIENT_SECRET,
                self._settings.GOOGLE_OAUTH_STATE_SECRET,
            )
        ):
            raise GoogleCalendarError(
                "google_calendar_not_configured",
                "Google Calendar authorization is not configured.",
            )


class GoogleCredentialService:
    def __init__(
        self,
        *,
        settings: JarvisSettings,
        repository: GoogleOAuthTokenRepository,
    ) -> None:
        self._settings = settings
        self._repository = repository

    async def credentials_for_owner(self, *, owner_id: str) -> Credentials:
        token = await self._repository.get_token(owner_id=owner_id)
        if token is None:
            raise GoogleCalendarError(
                "google_calendar_not_connected",
                "Google Calendar is not connected. Connect it from Jarvis Settings.",
            )
        credentials = Credentials(
            token=token.access_token,
            refresh_token=token.refresh_token,
            token_uri=GOOGLE_TOKEN_URI,
            client_id=self._settings.GOOGLE_OAUTH_CLIENT_ID,
            client_secret=self._settings.GOOGLE_OAUTH_CLIENT_SECRET,
            scopes=token.scopes or self._settings.GOOGLE_CALENDAR_SCOPES,
        )
        credentials.expiry = _aware_datetime(token.expires_at)
        if not _needs_refresh(token.expires_at):
            return credentials
        if not token.refresh_token:
            raise GoogleCalendarError(
                "google_calendar_reauthorization_required",
                "Google Calendar must be reconnected from Jarvis Settings.",
            )
        try:
            await asyncio.to_thread(credentials.refresh, GoogleAuthRequest())
        except RefreshError as error:
            raise GoogleCalendarError(
                "google_calendar_reauthorization_required",
                "Google Calendar must be reconnected from Jarvis Settings.",
            ) from error
        await self._repository.upsert_token(
            owner_id=owner_id,
            access_token=credentials.token,
            refresh_token=credentials.refresh_token,
            token_type="Bearer",
            scopes=list(credentials.scopes or token.scopes),
            expires_at=_aware_datetime(credentials.expiry),
        )
        return credentials


def _client_config(settings: JarvisSettings) -> dict[str, Any]:
    return {
        "web": {
            "client_id": settings.GOOGLE_OAUTH_CLIENT_ID,
            "client_secret": settings.GOOGLE_OAUTH_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": GOOGLE_TOKEN_URI,
            "redirect_uris": [settings.GOOGLE_OAUTH_REDIRECT_URI],
        }
    }


def _normalize_scopes(value: Any, fallback: list[str] | None) -> list[str]:
    if isinstance(value, str):
        return [scope for scope in value.split(" ") if scope]
    if isinstance(value, list):
        return [str(scope) for scope in value if str(scope)]
    return list(fallback or [])


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _aware_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _needs_refresh(expires_at: datetime | None) -> bool:
    if expires_at is None:
        return False
    aware_expiry = _aware_datetime(expires_at)
    assert aware_expiry is not None
    return aware_expiry <= datetime.now(timezone.utc) + timedelta(
        seconds=GOOGLE_TOKEN_REFRESH_WINDOW_SECONDS
    )
