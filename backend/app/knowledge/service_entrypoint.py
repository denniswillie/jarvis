"""Checked launcher for the separately installed, maintained Hindsight server."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

from app.knowledge.service_policy import ServicePolicyError, validate_service_configuration


SECRET_FIELDS = (
    "HINDSIGHT_API_LLM_API_KEY", "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY",
    "HINDSIGHT_API_TENANT_API_KEY",
)


def secret_file(path):
    location = Path(path)
    if not location.is_absolute():
        raise ServicePolicyError("hindsight_secret_path_must_be_absolute")
    try:
        with location.open("r", encoding="utf-8") as handle:
            value = handle.read(8194)
    except (OSError, UnicodeError):
        raise ServicePolicyError("hindsight_secret_file_unavailable") from None
    # Read one character beyond the maximum key plus one trailing newline.
    # Never accept a truncated prefix of a larger mounted secret.
    if len(value) > 8193:
        raise ServicePolicyError("hindsight_secret_file_invalid")
    value = value.strip()
    if not value or len(value) > 8192 or "\n" in value or "\r" in value or "\x00" in value:
        raise ServicePolicyError("hindsight_secret_file_invalid")
    return value


def deployment_environment(environment):
    """Docker secrets are optional; native fixtures can use explicit env keys."""
    result = dict(environment)
    for name in SECRET_FIELDS:
        path = result.get(name + "_FILE")
        if path:
            if result.get(name):
                raise ServicePolicyError("hindsight_secret_configuration_ambiguous")
            result[name] = secret_file(path)
            result.pop(name + "_FILE", None)
    password_file = result.get("JARVIS_HINDSIGHT_DB_PASSWORD_FILE")
    if password_file:
        if result.get("HINDSIGHT_API_DATABASE_URL"):
            raise ServicePolicyError("hindsight_database_configuration_ambiguous")
        password = secret_file(password_file)
        host = result.get("JARVIS_HINDSIGHT_DB_HOST", "hindsight-db")
        if not host or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for character in host):
            raise ServicePolicyError("hindsight_database_host_invalid")
        result["HINDSIGHT_API_DATABASE_URL"] = (
            "postgresql://hindsight:" + quote(password, safe="") + "@" + host + ":5432/hindsight"
        )
        result.pop("JARVIS_HINDSIGHT_DB_PASSWORD_FILE", None)
    return result


def native_configuration():
    # Lazy import keeps Hindsight out of the audio backend's dependency graph.
    from hindsight_api.config import HindsightConfig
    return HindsightConfig.from_env(), version("hindsight-api-slim")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    try:
        environment = deployment_environment(os.environ)
        os.environ.clear()
        os.environ.update(environment)
        config, installed_version = native_configuration()
        receipt = validate_service_configuration(config, environment, installed_version)
    except Exception as error:
        code = str(error) if isinstance(error, ServicePolicyError) else "hindsight_startup_policy_unavailable"
        print(json.dumps({"ready": False, "code": code}), file=sys.stderr)
        return 1
    print(json.dumps({"policy_verified": True, **receipt}), flush=True)
    if args.check_only:
        return 0
    # Call the maintained CLI in this process. No alternate agent loop, HTTP
    # proxy, native table mutation or extra server implementation is introduced.
    from hindsight_api.main import main as native_main
    sys.argv = ["hindsight-api", "--host", args.host, "--port", str(args.port)]
    native_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
