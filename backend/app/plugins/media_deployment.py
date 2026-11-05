"""Explicit operator-owned media composition, with no probes or implicit readiness.

The immutable qualification proves this exact image has rendered successfully.
Separate, externally produced observations describe current owner-scoped readiness.
Neither loading configuration nor restarting a process refreshes those observations.
"""
from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Literal

from pydantic import Field, field_validator

from app.plugins.managed import ManagedPluginRuntime
from app.plugins.deployment import CURATED_DEPLOYMENT_BASELINE_V1
from app.plugins.media import MediaRendererBundle, MediaRendererEvidence
from app.plugins.models import Digest, FrozenRecord, InventoryObservation, ReadinessObservation
from app.runtime.feature_config import ArtifactConfiguration, SandboxConfiguration
from app.tasks.models import TaskError
from app.workspace.media_render import RendererBinding


DEPLOYMENT_ENV = "JARVIS_MANAGED_MEDIA_DEPLOYMENT_FILE"
MAX_OBSERVATION_SECONDS = 300


class EvidenceFile(FrozenRecord):
    path: str
    sha256: Digest

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value):
        if not Path(value).is_absolute():
            raise ValueError("Operator evidence requires an absolute path")
        return value


class MediaDeployment(FrozenRecord):
    schema_version: Literal[1]
    owner_id: str = Field(strict=True, min_length=1, max_length=200)
    bundle: MediaRendererBundle
    build_receipt: EvidenceFile
    qualification: EvidenceFile
    observations_path: str
    readiness_refresh: Literal["worker"] | None = None

    @field_validator("observations_path")
    @classmethod
    def absolute_observations(cls, value):
        return EvidenceFile.absolute_path(value)


class MediaObservations(FrozenRecord):
    schema_version: Literal[1]
    owner_id: str = Field(strict=True, min_length=1, max_length=200)
    renderer: RendererBinding
    inventory: InventoryObservation
    readiness: ReadinessObservation


def read_operator_bytes(path, *, limit=65536):
    """Bounded regular local files writable only by their operator owner."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Operator configuration requires an absolute path")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_mode & 0o022 or info.st_size > limit):
            raise ValueError("Operator evidence must be a bounded, owner-controlled regular file")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Operator evidence exceeds the byte limit")
    return data


def _read_json(path, *, digest=None, limit=65536):
    data = read_operator_bytes(path, limit=limit)
    if digest is not None and hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Operator evidence bytes do not match")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Operator evidence requires a JSON object")
    return value


def _require_qualification(deployment):
    """Read the existing real managed-run receipt; never turn it into a new lease."""
    proof = _read_json(deployment.qualification.path,
                       digest=deployment.qualification.sha256, limit=2 * 1024 * 1024)
    renderer = deployment.bundle.renderer
    decoded = proof.get("independent_output_inspection", {})
    output = proof.get("accepted_output", {})
    command = proof.get("fixed_command_receipt", {}).get("result", {})
    inputs = proof.get("input_decodes", [])
    if (proof.get("format") != "jarvis.managed-media.qualification.v1"
            or proof.get("qualification", {}).get("passed") is not True
            or RendererBinding.model_validate(proof.get("renderer")) != renderer
            or not inputs or any(item.get("software_sha256") != renderer.software_sha256
                or item.get("video_decoded") is not True
                or (item.get("audio_streams", 0) and item.get("audio_decoded") is not True)
                for item in inputs)
            or command.get("started") is not True or type(command.get("exit_code")) is not int
            or command.get("exit_code") != 0
            or command.get("timed_out") is not False or command.get("runner_error") is not None
            or decoded.get("full_video_audio_decode") is not True
            or not output.get("file", {}).get("content_sha256")
            or decoded.get("content_sha256") != output["file"]["content_sha256"]
            or type(output.get("size_bytes")) is not int or output["size_bytes"] <= 0
            or decoded.get("size_bytes") != output["size_bytes"]):
        raise ValueError("The exact renderer requires a passing managed render/decode qualification")


def read_deployment(path, *, owner_id):
    """Validate static release evidence; this does not establish current readiness."""
    deployment = MediaDeployment.model_validate(_read_json(path))
    bundle = deployment.bundle
    renderer = bundle.renderer
    expected_id = next(entry.configuration_id for entry in CURATED_DEPLOYMENT_BASELINE_V1.policy.entries
                       if entry.plugin_id == "video_editor")
    if (deployment.owner_id != owner_id or bundle.configuration_id != expected_id
            or renderer.template.id != "jarvis.clip.v1"
            or bundle.bundle_sha256 != renderer.software_sha256
            or bundle.bundle_sha256 != renderer.template.bundle_sha256
            or bundle.bundle_version != renderer.template.version
            or deployment.build_receipt.sha256 != renderer.build_receipt_sha256):
        raise ValueError("The exact curated owner/image mapping is required")
    receipt = _read_json(deployment.build_receipt.path, digest=deployment.build_receipt.sha256)
    if ((receipt.get("template_id"), receipt.get("build_id"), receipt.get("status"))
            != (renderer.physical_template_id, renderer.build_id, "ready")
            or not isinstance(receipt.get("name"), str) or not receipt["name"]):
        raise ValueError("A matching finalized build receipt is required")
    _require_qualification(deployment)
    return deployment, receipt


class OperatorMediaObserver:
    def __init__(self, deployment, *, clock):
        self.deployment, self.clock = deployment, clock

    def read(self, owner_id, bundle):
        deployment = self.deployment
        if owner_id != deployment.owner_id or bundle != deployment.bundle:
            raise ValueError("The observation is scoped to a different owner or renderer")
        # The operator may atomically replace observations after a real new probe.
        # No provider/file mutation runs inside the task-owner transaction.
        value = MediaObservations.model_validate(_read_json(deployment.observations_path))
        now = self.clock()
        if value.owner_id != owner_id or value.renderer != bundle.renderer:
            raise ValueError("Owner and renderer observations must match exactly")
        inventory, readiness = value.inventory, value.readiness
        if (inventory.state != "installed" or readiness.state != "ready"
                or inventory.bundle_version != bundle.bundle_version
                or any(item.plugin_id != "video_editor" or item.configuration_id != bundle.configuration_id
                    or item.bundle_sha256 != bundle.bundle_sha256 or not item.current_at(now)
                    or (item.expires_at - item.observed_at).total_seconds() > MAX_OBSERVATION_SECONDS
                    for item in (inventory, readiness))):
            raise ValueError("Current exact inventory and readiness observations are required")
        return MediaRendererEvidence(owner_id, value.renderer, inventory, readiness)

    async def __call__(self, connection, owner_id, bundle):
        try:
            return self.read(owner_id, bundle)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            raise TaskError("managed_bundle_not_ready",
                "Current qualified editor readiness is unavailable; no new edit was submitted.", 503) from None


def load_managed_media(authority, *, owner_id, sandbox_configuration, artifact_configuration,
                       environ=None, clock=None):
    """No opt-in => no media runtime. Invalid selected config => sanitized error.

    API composition catches this error without disabling unrelated services.
    A worker explicitly selecting invalid media configuration fails startup.
    """
    environment = os.environ if environ is None else environ
    path = environment.get(DEPLOYMENT_ENV)
    if path is None:
        return None
    clock = clock or (lambda: datetime.now(UTC))
    try:
        deployment, _ = read_deployment(path, owner_id=owner_id)
        bundle = deployment.bundle
        renderer = bundle.renderer
        if (authority is None or not isinstance(sandbox_configuration, SandboxConfiguration)
                or sandbox_configuration.policy.template != "base"
                or not isinstance(artifact_configuration, ArtifactConfiguration)
                or not {"json", "mp4"}.issubset(artifact_configuration.policy.formats)
                or artifact_configuration.policy.max_files < 2
                or artifact_configuration.policy.limit_for("mp4") < renderer.max_output_bytes
                or artifact_configuration.policy.limit_for("json") < renderer.policy.max_recipe_bytes
                or artifact_configuration.policy.max_total_bytes < renderer.max_output_bytes + renderer.policy.max_recipe_bytes):
            raise ValueError("Media dependencies do not authorize this exact deployment")
        observer = OperatorMediaObserver(deployment, clock=clock)
        observer.read(owner_id, bundle)
        return ManagedPluginRuntime(authority, bundles=(bundle,), observe=observer, clock=clock)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        raise TaskError("managed_media_configuration_invalid",
            "Selected editor configuration or qualified readiness is unavailable.", 503) from None
