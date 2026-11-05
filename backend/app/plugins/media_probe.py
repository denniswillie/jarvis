"""Explicit operator command: two read-only provider checks, never a render or VM.

No API startup, admission or worker heartbeat invokes this command. Qualification is
immutable release evidence; fresh observations belong to the explicit deployment
owner and expire in five minutes. Actual VM inventory/decode checks still run.
"""
import argparse
import asyncio
from datetime import UTC, datetime, timedelta
import json
import logging
import os
from pathlib import Path
import tempfile

from app.plugins.media_deployment import (
    DEPLOYMENT_ENV, MAX_OBSERVATION_SECONDS, MediaObservations, OperatorMediaObserver,
    read_deployment, read_operator_bytes,
)


LOGGER = logging.getLogger(__name__)
REFRESH_INTERVAL_SECONDS = 120


class WorkerMediaReadinessRefresh:
    """One worker-owned optional child; no task/voice/database work is done here."""
    def __init__(self, path, *, owner_id, api_key):
        self.path, self.owner_id, self.api_key = path, owner_id, api_key

    async def __call__(self, stop):
        while not stop.is_set():
            try:
                await refresh_observations(self.path, owner_id=self.owner_id,
                    api_key=self.api_key.get_secret_value())
            except Exception:
                # Preserve the original file and expiry on failure. The normal
                # managed admission observer denies new work when it is stale.
                # No exception/provider text or key may enter ordinary logs.
                LOGGER.warning("Managed editor readiness refresh unavailable; existing evidence was not renewed")
            if stop.is_set():
                return
            try:
                await asyncio.wait_for(stop.wait(), REFRESH_INTERVAL_SECONDS)
            except TimeoutError:
                pass


def worker_media_refresh(managed_plugins, sandbox_runtime, *, environ=None):
    observer = getattr(managed_plugins, "observe", None)
    # Ordinary/test providers are never opt-ins. Only our validated deployment
    # carrier may select the child; the default path does not inspect credentials.
    if not isinstance(observer, OperatorMediaObserver) or observer.deployment.readiness_refresh != "worker":
        return None
    environment = os.environ if environ is None else environ
    key = getattr(getattr(getattr(sandbox_runtime, "provider", None), "configuration", None), "api_key", None)
    if not getattr(sandbox_runtime, "enabled", False) or not callable(getattr(key, "get_secret_value", None)):
        raise ValueError("Selected editor refresh requires the existing sandbox worker credential")
    path = environment.get(DEPLOYMENT_ENV)
    if path is None:
        raise ValueError("Selected editor refresh requires the operator deployment path")
    return WorkerMediaReadinessRefresh(path, owner_id=observer.deployment.owner_id, api_key=key)


async def provider_state(deployment, receipt, api_key):
    # Same SDK requests used by the executed managed qualification; lazy imports
    # keep ordinary API startup and the offline --check free of SDK/provider work.
    from e2b import AsyncTemplate
    from e2b.template.types import BuildInfo
    from e2b.api.client_async import get_api_client
    from e2b.connection_config import ConnectionConfig
    from e2b.api.client.api.templates import get_templates_template_id
    from app.runtime.http_privacy import private_http_logs

    renderer = deployment.bundle.renderer
    options = {"api_key": api_key, "request_timeout": 10, "retries": 0, "debug": False}
    info = BuildInfo(template_id=renderer.physical_template_id, build_id=renderer.build_id,
                     name=receipt["name"], alias=receipt["name"])
    with private_http_logs():
        status = await AsyncTemplate.get_build_status(info, logs_offset=0, **options)
        async with get_api_client(ConnectionConfig(**options)) as client:
            response = await get_templates_template_id.asyncio_detailed(renderer.physical_template_id, client=client)
            if response.status_code != 200:
                raise ValueError("Current template metadata unavailable")
            metadata = response.parsed.to_dict()
    return (status.template_id, status.build_id, status.status.value), metadata


async def refresh_observations(path, *, owner_id, api_key):
    deployment, receipt = read_deployment(path, owner_id=owner_id)
    renderer, bundle = deployment.bundle.renderer, deployment.bundle
    async with asyncio.timeout(25):
        status, metadata = await provider_state(deployment, receipt, api_key)
    expected = (renderer.physical_template_id, renderer.build_id, "ready")
    if status != expected or (metadata.get("templateID"), metadata.get("buildID"), metadata.get("buildStatus")) != expected:
        raise ValueError("Provider does not currently expose the exact qualified immutable image")
    # Timestamp only after both real responses have matched. Never extend an old
    # fixture observation or infer readiness merely from configuration presence.
    observed = datetime.now(UTC)
    fields = dict(plugin_id="video_editor", configuration_id=bundle.configuration_id,
        observed_at=observed, expires_at=observed + timedelta(seconds=MAX_OBSERVATION_SECONDS),
        bundle_sha256=bundle.bundle_sha256)
    observations = MediaObservations(schema_version=1, owner_id=owner_id, renderer=renderer,
        inventory={**fields, "state": "installed", "bundle_version": bundle.bundle_version},
        readiness={**fields, "state": "ready"})
    destination = Path(deployment.observations_path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, prefix=".media-observations-",
                                          delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(observations.model_dump_json(indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--owner", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Local static evidence only; not readiness")
    mode.add_argument("--refresh", action="store_true", help="Execute two provider reads and publish a bounded observation")
    parser.add_argument("--key-file", help="Existing private E2B credential; read only for --refresh")
    args = parser.parse_args()
    try:
        if args.check:
            read_deployment(args.deployment, owner_id=args.owner)
            print(json.dumps({"configuration_valid": True, "readiness_probed": False}))
        else:
            if not args.key_file:
                raise ValueError("An explicit credential file is required")
            key = read_operator_bytes(args.key_file, limit=16384).decode().strip()
            if not key:
                raise ValueError("An explicit credential is required")
            observation = asyncio.run(refresh_observations(args.deployment, owner_id=args.owner, api_key=key))
            print(json.dumps({"readiness_probed": True, "provider_read_calls": 2, "vm_created": False,
                "expires_at": observation.readiness.expires_at.isoformat()}))
        return 0
    except Exception:
        # No provider messages, credential contents or operator paths in output.
        print(json.dumps({"readiness_probed": False, "error": "managed_media_preparation_failed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
