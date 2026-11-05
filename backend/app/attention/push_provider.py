"""Small APNs HTTP/2 adapter. No assistant content enters a lock-screen payload."""
from datetime import UTC, datetime
import logging
import time
from uuid import UUID

import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from app.attention.push_models import PushAuthorization, PushOutcome
from app.runtime.http_privacy import (
    PRIVATE_HTTP_ACTIVE as _APNS_REQUEST_ACTIVE,
    TRANSPORT_PRIVACY_FILTER as _APNS_TRANSPORT_PRIVACY_FILTER,
    TRANSPORT_LOGGERS as _APNS_TRANSPORT_LOGGERS,
    TransportPrivacyFilter as _APNsTransportPrivacyFilter,
)


class APNsProvider:
    def __init__(self, configuration, *, client=None, clock=time.time):
        for name in _APNS_TRANSPORT_LOGGERS:
            # Logger.addFilter is idempotent for this shared filter instance.
            logging.getLogger(name).addFilter(_APNS_TRANSPORT_PRIVACY_FILTER)
        self.configuration = configuration
        self.clock = clock
        self._key = serialization.load_pem_private_key(
            configuration.private_key.get_secret_value().encode(), password=None)
        if not isinstance(self._key, ec.EllipticCurvePrivateKey) or not isinstance(self._key.curve, ec.SECP256R1):
            raise ValueError("An APNs P-256 private key is required")
        self.client = client or httpx.AsyncClient(http2=True, timeout=configuration.http_timeout_seconds,
                                                 follow_redirects=False, trust_env=False)
        self._token = None
        self._issued = 0

    def _authorization(self):
        now = int(self.clock())
        if self._token is None or not 0 <= now - self._issued < 3000:
            self._token = jwt.encode({"iss": self.configuration.team_id, "iat": now},
                                     self._key, algorithm="ES256",
                                     headers={"kid": self.configuration.key_id})
            self._issued = now
        return "bearer " + self._token

    async def send(self, request, *, authorize=None):
        if request.environment != self.configuration.environment or request.topic != self.configuration.topic:
            return PushOutcome("failed", "push_registration_mismatch")
        host = "api.sandbox.push.apple.com" if request.environment == "sandbox" else "api.push.apple.com"
        headers = {
            "authorization": self._authorization(), "apns-push-type": "alert",
            "apns-topic": request.topic, "apns-id": str(request.claim.attempt_id),
            "apns-collapse-id": str(request.claim.intent_id),
            # No store-and-forward for a potentially changed/deleted source.
            "apns-expiration": "0", "apns-priority": "10" if request.urgency == "high" else "5",
        }
        payload = {"aps": {"alert": {"title": "Jarvis", "body": "An update is available in your inbox."}, "sound": "default"},
                   "jarvis": {"version": 1, "intent_id": str(request.claim.intent_id)}}
        if authorize is None:
            return PushOutcome("revoked", "push_handoff_authorization_required")
        authorization = await authorize(request)
        if not isinstance(authorization, PushAuthorization) or authorization.claim != request.claim:
            return PushOutcome("revoked", "push_handoff_authorization_invalid")
        if authorization.allowed is not True:
            return PushOutcome("revoked", authorization.reason)
        # The callback commits and releases its database transaction before
        # returning. No database/voice lock spans HTTP I/O. The durable handoff
        # is deliberately conservative: cancellation, a socket queue or a later
        # quiet change cannot prove that no bytes were sent, nor recall them.
        logging_context = _APNS_REQUEST_ACTIVE.set(True)
        try:
            response = await self.client.post("https://" + host + "/3/device/" + request.token,
                                              headers=headers, json=payload)
        except httpx.RequestError:
            # A timeout/reset does not establish that APNs rejected the request.
            return PushOutcome("unknown", "apns_transport_uncertain")
        finally:
            _APNS_REQUEST_ACTIVE.reset(logging_context)
        try:
            provider_id = UUID(response.headers.get("apns-id", str(request.claim.attempt_id)))
        except ValueError:
            provider_id = request.claim.attempt_id
        if response.status_code == 200:
            return PushOutcome("accepted", provider_status=200, provider_id=provider_id)
        data = {}
        try:
            data = response.json()
            reason = data.get("reason") if isinstance(data, dict) else None
        except ValueError:
            reason = None
        if not isinstance(reason, str):
            reason = None
        # Only known protocol codes, never arbitrary provider bodies or tokens.
        permanent = {"BadDeviceToken", "DeviceTokenNotForTopic", "Forbidden", "ExpiredToken",
                     "Unregistered", "PayloadTooLarge"}
        known = permanent | {"BadTopic", "MissingTopic", "BadExpirationDate", "BadPriority",
                             "ExpiredProviderToken", "InvalidProviderToken", "MissingProviderToken",
                             "TooManyProviderTokenUpdates", "TooManyRequests", "InternalServerError",
                             "ServiceUnavailable", "Shutdown"}
        safe_reason = reason if reason in known else "apns_rejected"
        invalidated = None
        if response.status_code == 410 and isinstance(data, dict):
            stamp = data.get("timestamp")
            if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
                try:
                    invalidated = datetime.fromtimestamp(stamp / 1000, UTC)
                except (OverflowError, OSError, ValueError):
                    pass
        invalidate = reason in {"BadDeviceToken", "DeviceTokenNotForTopic"} or (
            response.status_code == 410 and invalidated is not None)
        retry = 900 if response.status_code >= 500 else (60 if reason == "TooManyRequests" else None)
        if reason in permanent:
            retry = None
        return PushOutcome("failed", safe_reason, response.status_code, provider_id,
                           retry_seconds=retry, invalidate_token=invalidate, invalidated_at=invalidated)

    async def close(self):
        await self.client.aclose()
