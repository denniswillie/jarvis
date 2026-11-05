"""Deployment policy shared by the native launcher and Jarvis adapter.

Pure standard library: importing the audio backend never imports Hindsight.
"""
from __future__ import annotations

import math
from urllib.parse import urlsplit


HINDSIGHT_VERSION = "0.9.2"
POLICY_ID = "jarvis.source-memory.v1"
PRIVACY_ENV = {
    "HINDSIGHT_API_LLM_TRACE_ENABLED": "false",
    "HINDSIGHT_API_AUDIT_LOG_ENABLED": "false",
    "HINDSIGHT_API_OTEL_TRACES_ENABLED": "false",
    "HINDSIGHT_API_LLM_DEBUG_DUMP_4XX": "false",
    "HINDSIGHT_API_ENABLE_OBSERVATION_HISTORY": "false",
    "HINDSIGHT_API_ENABLE_MENTAL_MODEL_HISTORY": "false",
}
FALSE_FIELDS = (
    "llm_trace_enabled", "audit_log_enabled", "otel_traces_enabled", "llm_debug_dump_4xx",
    "enable_observation_history", "enable_mental_model_history",
)
BANK_SETTINGS = {"audit_log_enabled": False, "store_document_text": True}
TENANT_EXTENSION = "hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension"


class ServicePolicyError(RuntimeError):
    """Only a fixed code may reach startup output, never native config values."""


def validate_bank_configuration(payload, bank):
    if (not isinstance(payload, dict) or payload.get("bank_id") != bank
            or not isinstance(payload.get("config"), dict)
            or any(payload["config"].get(key) is not expected for key, expected in BANK_SETTINGS.items())):
        raise ServicePolicyError("hindsight_bank_policy_not_satisfied")


def validate_service_configuration(config, environment, version):
    if version != HINDSIGHT_VERSION:
        raise ServicePolicyError("hindsight_version_not_accepted")
    if any(getattr(config, key, None) is not False for key in FALSE_FIELDS):
        raise ServicePolicyError("hindsight_payload_logging_or_history_enabled")
    if (environment.get("HINDSIGHT_API_TENANT_EXTENSION") != TENANT_EXTENSION
            or not 24 <= len(environment.get("HINDSIGHT_API_TENANT_API_KEY", "").strip()) <= 4096):
        raise ServicePolicyError("hindsight_private_authentication_required")
    # Native extension configuration is loaded from the environment, not from
    # HindsightConfig. Do not mistake a missing dataclass field for verification.
    try:
        database = urlsplit(config.database_url)
        if database.scheme not in {"postgres", "postgresql"} or not database.hostname or not database.path.strip("/"):
            raise ValueError
    except (AttributeError, TypeError, ValueError):
        raise ServicePolicyError("hindsight_dedicated_postgres_required") from None
    slots = getattr(config, "worker_max_slots", None)
    reservations = getattr(config, "worker_slot_reservations", None)
    concurrency = getattr(config, "llm_max_concurrent", None)
    if (type(slots) is not int or not 1 <= slots <= 8 or not isinstance(reservations, dict)
            or any(type(value) is not int or value < 0 for value in reservations.values())
            or slots - sum(reservations.values()) < 1
            or type(concurrency) is not int or not 1 <= concurrency <= 8):
        raise ServicePolicyError("hindsight_capacity_not_bounded")
    timeout = getattr(config, "llm_timeout", None)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 180:
        raise ServicePolicyError("hindsight_request_deadline_not_bounded")
    return {
        "policy": POLICY_ID, "version": version, "privacy_defaults_verified": True,
        "bank_overrides": "must_be_checked_by_adapter", "worker_slots": slots,
        "shared_worker_slots": slots-sum(reservations.values()), "llm_concurrency": concurrency,
    }
