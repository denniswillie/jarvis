"""Bound renderer in the actual tool/context assembly; SQL/provider are doubled."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.files.generated import ArtifactPolicy
from app.sandbox.media import ManagedMediaTools, RenderMedia
from app.sandbox.models import SandboxPolicy
from app.sandbox.tools import SandboxToolBridge
from app.tasks.hermes import HermesContextFactory
from test_managed_media import media_scope


class RuntimeTemplateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.job, _ = media_scope()
        self.media = object.__new__(ManagedMediaTools)
        self.media.job = self.job
        self.base = SimpleNamespace(catalog=lambda: [], authorized_operations=lambda: [])
        self.bridge = object.__new__(SandboxToolBridge)
        self.bridge.base = self.base
        self.bridge.policy = SandboxPolicy()
        self.bridge.managed_job = self.job
        self.bridge.media = self.media
        self.bridge.definitions = self.media.definitions()
        self.bridge.artifacts = SimpleNamespace(policy=ArtifactPolicy(
            formats=("json", "mp4"), mp4_max_file_bytes=self.job.bundle.renderer.max_output_bytes))

    def test_agent_catalog_omits_deployment_pin_but_canonical_recipe_keeps_it(self):
        tool = next(item for item in self.bridge.catalog() if item["name"] == "media_render")
        definitions = tool["parameters"]["$defs"]
        recipe = definitions["MediaEditRecipeV1"]
        self.assertNotIn("template", recipe["properties"])
        self.assertNotIn("template", recipe["required"])
        self.assertNotIn("TemplatePin", definitions)
        self.assertFalse(recipe["additionalProperties"])
        self.assertEqual(set(recipe["required"]), set(recipe["properties"]))
        self.assertIn("template", RenderMedia.model_json_schema()["$defs"]["MediaEditRecipeV1"]["required"])
        self.assertIn("runtime supplies", tool["description"])

    def test_descriptor_is_exact_task_binding_and_does_not_mutate_it(self):
        operation = self.bridge.authorized_operations()[0]
        self.assertEqual(operation["renderer"]["template"], self.job.bundle.renderer.template.model_dump(mode="json"))
        self.assertIn("not a user/model selection", operation["renderer"]["binding"])
        operation["renderer"]["template"]["bundle_sha256"] = "0" * 64
        self.assertEqual(self.bridge.authorized_operations()[0]["renderer"]["template"]["bundle_sha256"],
                         self.job.bundle.renderer.template.bundle_sha256)

    def test_nonmedia_task_does_not_gain_a_renderer(self):
        self.bridge.media = None
        self.bridge.managed_job = None
        self.bridge.artifacts = None
        self.assertNotIn("renderer", self.bridge.authorized_operations()[0])

    async def test_normal_hermes_preparation_carries_renderer_and_creative_catalog(self):
        task = SimpleNamespace(owner_id=self.job.binding.owner_id, id=uuid4(), session_id=uuid4(),
            revision=1, title="Edit my clip", requested_outcome="Change the caption and preserve the cuts.",
            request_event_id=uuid4(), request_revision=1)
        context = SimpleNamespace(task=task, tasks=SimpleNamespace(pool=object()),
                                  claim=SimpleNamespace(task=task, task_revision=1))
        self.base.context = context
        self.bridge.context = context
        repository = SimpleNamespace(snapshot=AsyncMock(return_value={
            "watermark": 1, "pinned": [], "recent": [], "selected_pinned_ids": set(),
            "pinned_total": 0, "total": 0}),
            store_manifest=AsyncMock(side_effect=lambda claim, **kw: SimpleNamespace(content=kw["content"])))
        async def bound_bridge(base):
            self.assertIs(base, self.base)
            return self.bridge
        # Real HermesContextFactory and TaskContextBuilder assemble the final
        # model-facing manifest. Only database reads/writes and base modules
        # are substituted, with no native/model/VM execution.
        with patch("app.tasks.hermes.ContextRepository", return_value=repository), \
                patch("app.tasks.hermes.TaskToolBridge.from_modules", AsyncMock(return_value=self.base)):
            prepared = await HermesContextFactory(bridge_factory=bound_bridge)(context)
        operations = prepared.manifest.content["shared_context"]["authorized_operations"]
        renderer = next(item["renderer"] for item in operations if item["tool"] == "media_render")
        self.assertEqual(renderer["template"], self.job.bundle.renderer.template.model_dump(mode="json"))
        self.assertEqual(prepared.manifest.content["task"]["requested_outcome"], task.requested_outcome)
        catalog = next(item for item in prepared.tools.catalog() if item["name"] == "media_render")
        self.assertNotIn("template", catalog["parameters"]["$defs"]["MediaEditRecipeV1"]["properties"])


if __name__ == "__main__":
    unittest.main()
