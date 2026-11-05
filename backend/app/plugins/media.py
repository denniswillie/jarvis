"""Trusted renderer mapping for the existing managed task lifecycle.

These records are composition inputs, not installation or readiness evidence.
The task runtime checks independently supplied observations before submission.
"""
from dataclasses import asdict, dataclass
from typing import ClassVar
from uuid import UUID

from app.plugins.authority import ManagedPluginBinding
from app.plugins.models import BundleVersion, Digest, FrozenRecord, InventoryObservation, ReadinessObservation
from app.tasks.models import TaskError
from app.workspace.media_render import MediaRenderPlan, RendererBinding


MEDIA_ROUTE = "video_editing"
MEDIA_PLUGIN = "video_editor"
MEDIA_CAPABILITY = "plugins.video_editor"
MEDIA_GRANT_PREFIX = "managed-video-editor:"
MEDIA_FORMAT = "jarvis.managed-media.v1"


class MediaRendererBundle(FrozenRecord):
    configuration_id: UUID
    bundle_version: BundleVersion
    bundle_sha256: Digest
    renderer: RendererBinding

    @property
    def sandbox_template_id(self):
        return self.renderer.physical_template_id

    def require_image(self, handle):
        if handle.template_id != self.sandbox_template_id:
            raise TaskError("managed_bundle_image_mismatch", "The actual sandbox image is not the qualified renderer.", 503)

    def require_command(self, request, *, plan=None):
        if (not isinstance(plan, MediaRenderPlan) or plan.binding != self.renderer
                or request != plan.command()):
            raise TaskError("managed_media_plan_required", "Rendering requires the exact prepared plan and pinned launcher.", 403)


@dataclass(frozen=True)
class MediaRendererEvidence:
    owner_id: str
    renderer: RendererBinding
    inventory: InventoryObservation
    readiness: ReadinessObservation


@dataclass(frozen=True)
class ManagedMediaJob:
    binding: ManagedPluginBinding
    bundle: MediaRendererBundle
    route: ClassVar[str] = MEDIA_ROUTE
    grant_prefix: ClassVar[str] = MEDIA_GRANT_PREFIX

    def __post_init__(self):
        if (not isinstance(self.binding, ManagedPluginBinding) or self.binding.plugin_id != MEDIA_PLUGIN
                or not isinstance(self.bundle, MediaRendererBundle)
                or self.binding.configuration_id != self.bundle.configuration_id):
            raise ValueError("An exact managed video-editor binding and renderer mapping are required")

    def payload(self):
        binding = asdict(self.binding)
        binding["configuration_id"] = str(self.binding.configuration_id)
        return {"format": MEDIA_FORMAT, "binding": binding, "bundle": self.bundle.model_dump(mode="json")}

    @classmethod
    def from_payload(cls, value, owner_id):
        try:
            if not isinstance(value, dict) or set(value) != {"format", "binding", "bundle"} or value["format"] != MEDIA_FORMAT:
                raise ValueError("Invalid managed media scope")
            binding = dict(value["binding"])
            binding["configuration_id"] = UUID(binding["configuration_id"])
            job = cls(ManagedPluginBinding(**binding), MediaRendererBundle.model_validate(value["bundle"]))
            if job.binding.owner_id != owner_id:
                raise ValueError("Wrong owner")
            return job
        except (ValueError, TypeError, KeyError, AttributeError):
            raise TaskError("managed_scope_unavailable", "The original managed media scope is unavailable.", 403) from None
