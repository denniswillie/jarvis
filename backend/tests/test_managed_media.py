"""Managed media dispatch contracts with explicit SQL/provider doubles.

These checks do not qualify a cloud image, render pixels, or prove SQL behavior.
"""
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.files.generated import ArtifactPolicy
from app.plugins.authority import ManagedPluginBinding
from app.plugins.managed import (
    CAPABILITY, GRANT_PREFIX, ManagedPluginRuntime, decode_managed_job,
    load_managed_task, require_request, submission_receipt,
)
from app.plugins.media import (
    MEDIA_CAPABILITY, MEDIA_FORMAT, MEDIA_GRANT_PREFIX, MEDIA_ROUTE,
    ManagedMediaJob, MediaRendererBundle, MediaRendererEvidence,
)
from app.plugins.models import InventoryObservation, ReadinessObservation
from app.plugins.task_tools import MediaTaskToolModule
from app.sandbox.broker import SandboxCommandBroker
from app.sandbox.models import CommandRequest, SandboxError, SandboxPolicy
from app.tasks.models import TaskCreate, TaskError
from app.tasks.service import TaskService
from app.tasks.worker import WorkerContextFactory
from app.workspace.media_recipe import (
    MediaRecipePolicy, SourceInspection, TemplatePin, validate_recipe,
)
from app.workspace.media_render import AuthorizedRenderInput, RendererBinding, prepare_render
from test_managed_files_data import Connection, NOW, OWNER, Tasks, fixture_scope


def media_scope():
    binding = ManagedPluginBinding(OWNER, "video_editor", uuid4(), 2)
    renderer = RendererBinding(
        template=TemplatePin(id="jarvis.clip.v1", version="fixture-v1", bundle_sha256="b" * 64),
        physical_template_id="synthetic-render-image", build_id="synthetic-render-build",
        software_sha256="c" * 64, build_receipt_sha256="d" * 64,
        policy=MediaRecipePolicy(max_source_bytes=1024 * 1024, max_source_duration_ms=10000,
                                max_output_frames=240), max_output_bytes=1024 * 1024)
    bundle = MediaRendererBundle(configuration_id=binding.configuration_id, bundle_version="fixture-v1",
                                 bundle_sha256="a" * 64, renderer=renderer)
    fields = dict(plugin_id="video_editor", configuration_id=binding.configuration_id,
                  observed_at=NOW - timedelta(seconds=1), expires_at=NOW + timedelta(seconds=60),
                  bundle_sha256=bundle.bundle_sha256)
    evidence = MediaRendererEvidence(OWNER, renderer,
        InventoryObservation(**fields, state="installed", bundle_version=bundle.bundle_version),
        ReadinessObservation(**fields, state="ready"))
    return ManagedMediaJob(binding, bundle), evidence


def render_plan(job, task):
    from app.workspace.media_recipe import MediaEditRecipeV1, TemplateInventoryEntry
    source = dict(file_id=str(uuid4()), version_id="opaque-fixture-version", content_sha256="e" * 64)
    recipe = MediaEditRecipeV1.model_validate(dict(schema_version=1, edit_id=str(uuid4()),
        previous_recipe=None, template=job.bundle.renderer.template.model_dump(),
        sources=[dict(source_id="original", file=source)],
        segments=[dict(segment_id="cut", source_id="original", in_ms=0, out_ms=1000)],
        captions=[], output_profile="portrait_720p24_v1", fit="contain", audio="preserve"))
    inspection = SourceInspection(file=recipe.sources[0].file, duration_ms=10000,
                                  size_bytes=100, video_decodable=True, audio_streams=1, audio_decodable=True)
    checked = validate_recipe(recipe, inspections=[inspection], policy=job.bundle.renderer.policy,
        inventory=[TemplateInventoryEntry(template=job.bundle.renderer.template,
                                          output_profile="portrait_720p24_v1", ready=True)])
    return prepare_render(owner_id=task.owner_id, task_id=task.id, operation_key="original-render",
        recipe=checked, source=AuthorizedRenderInput(task.owner_id, task.id, "fixture-vm",
            "/home/jarvis/inputs/" + "f" * 64, inspection), binding=job.bundle.renderer)


class MediaConnection(Connection):
    async def fetch(self, sql, *args):
        if "managed-files-data:%" in sql:
            return [dict(id=key, operation_key=row["operation_key"]) for key, row in self.grants.items()
                    if row["operation_key"].startswith((GRANT_PREFIX, MEDIA_GRANT_PREFIX))]
        return await super().fetch(sql, *args)


class ManagedMediaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.job, self.evidence = media_scope()
        self.authority = SimpleNamespace(select_new_work=AsyncMock(return_value=self.job.binding),
                                         require_new_work=AsyncMock())
        self.observe = AsyncMock(side_effect=lambda *_: self.evidence)
        self.runtime = ManagedPluginRuntime(self.authority, bundles=(self.job.bundle,),
                                            observe=self.observe, clock=lambda: NOW)
        self.tasks = Tasks()
        self.tasks.connection = MediaConnection(self.tasks)
        self.tasks.task.session_id = uuid4()
        self.tasks.task.knowledge_area_id = uuid4()
        self.configuration = SimpleNamespace(policy=ArtifactPolicy(formats=("json", "mp4"),
            mp4_max_file_bytes=1024 * 1024), area_id=self.tasks.task.knowledge_area_id,
            required_capabilities=("files.publish_generated", "files.publish_generated_mp4"), grant_seconds=120)
        self.registry = SimpleNamespace(require=AsyncMock())
        self.service = TaskService(self.tasks, {"frontier": SimpleNamespace(model="fixture-model")},
            registry=self.registry, sandbox_configuration=SimpleNamespace(policy=SandboxPolicy(), grant_seconds=120),
            artifact_configuration=self.configuration, managed_plugins=self.runtime)
        self.request = TaskCreate(request_id=uuid4(), title="Edit retained clip", requested_outcome="Keep the first second.",
            session_id=self.tasks.task.session_id, knowledge_area_id=self.tasks.task.knowledge_area_id,
            execution_environment="offline_sandbox", publish_artifacts=True)
        self.context = SimpleNamespace(tasks=self.tasks, task=self.tasks.task, claim=self.tasks.claim)
        self.budget = patch("app.tasks.actions.TaskBudgetRepository.reserve_locked",
                            new=AsyncMock(return_value={"id": uuid4(), "status": "reserved"}))
        self.budget.start()
        self.addCleanup(self.budget.stop)

    async def create(self):
        await self.service.create_media(OWNER, self.request)
        return next(key for key, row in self.tasks.connection.grants.items() if row["capability"] == "sandbox.execute")

    async def test_exact_mapping_round_trip_and_no_cross_route_decode(self):
        self.assertEqual(decode_managed_job(self.job.payload(), OWNER, route=MEDIA_ROUTE), self.job)
        for owner, route in (("other-owner", MEDIA_ROUTE), (OWNER, "files_data")):
            with self.subTest(owner=owner, route=route), self.assertRaises(TaskError):
                decode_managed_job(self.job.payload(), owner, route=route)
        payload = self.job.payload()
        payload["format"] = "jarvis.managed-files-data.v1"
        with self.assertRaises(TaskError):
            decode_managed_job(payload, OWNER)

    async def test_constructor_mapping_is_not_readiness(self):
        self.observe.return_value = None
        self.observe.side_effect = None
        with self.assertRaises(TaskError):
            await self.create()
        self.assertFalse(self.tasks.requests)
        self.assertFalse(self.tasks.connection.grants)

    async def test_owner_image_build_and_observation_freshness_are_exact(self):
        original = self.evidence
        variants = [replace(original, owner_id="other-owner"),
                    replace(original, renderer=original.renderer.model_copy(update={"build_id": "different-build"})),
                    replace(original, renderer=original.renderer.model_copy(update={"physical_template_id": "other-image"})),
                    replace(original, readiness=original.readiness.model_copy(update={"state": "unknown"})),
                    replace(original, readiness=original.readiness.model_copy(update={"expires_at": NOW})),
                    replace(original, inventory=original.inventory.model_copy(update={"bundle_sha256": "f" * 64}))]
        for evidence in variants:
            with self.subTest(evidence=evidence), self.assertRaises(TaskError):
                self.evidence = evidence
                await self.create()
        self.assertFalse(self.tasks.connection.grants)

    async def test_media_route_persists_exact_job_and_separate_publication_grant(self):
        grant = await self.create()
        self.assertEqual(self.tasks.connection.grants[grant]["operation_key"], MEDIA_GRANT_PREFIX + str(self.tasks.task.id))
        self.assertEqual(self.tasks.connection.payloads[self.tasks.task.description_payload_id]["content"]["managed_plugin"],
                         self.job.payload())
        self.assertEqual(len(self.tasks.connection.grants), 2)
        self.assertFalse(self.tasks.connection.actions)
        self.assertIn(MEDIA_CAPABILITY, self.registry.require.call_args.kwargs["required_capabilities"])
        self.assertNotIn(CAPABILITY, self.registry.require.call_args.kwargs["required_capabilities"])
        self.assertEqual(self.authority.select_new_work.call_args.kwargs["category"], MEDIA_ROUTE)

    async def test_replay_does_not_reselect_or_require_present_mapping(self):
        await self.create()
        self.service.managed_plugins = None
        self.service.artifact_configuration = None
        self.registry.require.side_effect = AssertionError("Replay cannot admit new work")
        self.assertIs(await self.service.create_media(OWNER, self.request), self.tasks.task)
        self.observe.assert_awaited_once()
        with self.assertRaises(TaskError):
            await self.service.create_files_data(OWNER, self.request)

    async def test_media_requires_recipe_and_video_allowance(self):
        self.service.artifact_configuration = SimpleNamespace(
            **(vars(self.configuration) | {"policy": ArtifactPolicy()}))
        with self.assertRaises(TaskError) as error:
            await self.create()
        self.assertEqual(error.exception.code, "managed_media_publication_unavailable")
        self.assertFalse(self.tasks.connection.grants)

    async def test_incompatible_request_cannot_select_renderer(self):
        for changes in ({"publish_artifacts": False}, {"execution_environment": "networked_sandbox"},
                        {"grant_ids": (uuid4(),)}, {"edit_memory": True}):
            with self.subTest(changes=changes), self.assertRaises(TaskError):
                require_request(self.request.model_copy(update=changes), MEDIA_ROUTE)
        self.authority.select_new_work.assert_not_awaited()

    async def test_missing_payload_or_wrong_variant_marker_cannot_become_general_work(self):
        grant = await self.create()
        self.tasks.connection.grants[grant]["operation_key"] = GRANT_PREFIX + str(self.tasks.task.id)
        with self.assertRaises(TaskError):
            await load_managed_task(self.tasks.connection, self.tasks.task)
        self.tasks.connection.grants[grant]["operation_key"] = MEDIA_GRANT_PREFIX + str(self.tasks.task.id)
        self.tasks.connection.payloads[self.tasks.task.description_payload_id]["availability"] = "deleted"
        with self.assertRaises(TaskError):
            await load_managed_task(self.tasks.connection, self.tasks.task)

    async def test_existing_submission_pin_is_idempotent_and_survives_plugin_disable(self):
        grant = await self.create()
        self.assertEqual(await self.runtime.ensure_submitted(self.context, grant), self.job)
        self.assertEqual(len(self.tasks.connection.actions), 1)
        action = next(iter(self.tasks.connection.actions.values()))
        receipt = self.tasks.connection.payloads[action["receipt_payload_id"]]["content"]
        self.assertEqual(receipt, submission_receipt(self.tasks.task, self.job, action["submission_token"]))
        self.assertEqual(receipt["format"], "jarvis.managed-media.submission.v1")
        self.authority.require_new_work.side_effect = TaskError("plugin_disabled", "Disabled")
        self.runtime.bundles.clear()
        self.assertEqual(await self.runtime.ensure_submitted(self.context, grant), self.job)
        self.assertEqual(len(self.tasks.connection.actions), 1)
        self.authority.require_new_work.assert_awaited_once()

    async def test_disabled_before_submission_has_no_accepted_pin(self):
        grant = await self.create()
        self.authority.require_new_work.side_effect = TaskError("plugin_disabled", "Disabled")
        with self.assertRaises(TaskError):
            await self.runtime.ensure_submitted(self.context, grant)
        self.assertFalse(any(action["submission_token"] is not None for action in self.tasks.connection.actions.values()))

    async def test_worker_capabilities_follow_supplied_mapping_not_installed_claims(self):
        self.assertEqual(self.runtime.capabilities, (MEDIA_CAPABILITY,))
        file_job, _ = fixture_scope()
        mixed = ManagedPluginRuntime(self.authority, bundles=(file_job.bundle, self.job.bundle),
                                     observe=self.observe, clock=lambda: NOW)
        self.assertEqual(mixed.capabilities, (CAPABILITY, MEDIA_CAPABILITY))
        with self.assertRaises(ValueError):
            WorkerContextFactory(None, sandbox_provider=SimpleNamespace(), managed_plugins=self.runtime)

    async def test_foreground_tool_requests_background_work_not_recipe_construction(self):
        tools = MediaTaskToolModule(self.service).build_tools(SimpleNamespace(session_id=uuid4(), owner_id=OWNER))
        self.assertEqual([tool.name for tool in tools], ["start_media_task"])
        self.assertNotIn("recipe", tools[0].input_schema["properties"])
        self.assertTrue(tools[0].input_schema["properties"]["publish_artifacts"]["default"])

    async def test_media_worker_cannot_advertise_without_actual_decoder_interface(self):
        from app.files.transfer import FileTransferPolicy
        provider = SimpleNamespace(materialize_media_render=AsyncMock())
        options = dict(knowledge=object(), file_storage=object(), sandbox_provider=provider,
            artifact_configuration=self.configuration, media_policy=FileTransferPolicy("mp4_output_v1", 1024 * 1024),
            media_inspector=AsyncMock(), managed_plugins=self.runtime)
        with self.assertRaisesRegex(ValueError, "decoder"):
            WorkerContextFactory(None, **options)
        provider.decode_media_render_input = AsyncMock()
        factory = WorkerContextFactory(None, **options)
        self.assertIn(MEDIA_CAPABILITY, factory.capabilities)
        self.assertNotIn(CAPABILITY, factory.capabilities)
        provider.decode_media_render_input.assert_not_awaited()
        provider.materialize_media_render.assert_not_awaited()

    async def test_broker_rejects_general_shell_and_wrong_owner_before_allocation(self):
        broker = SandboxCommandBroker(self.context, SimpleNamespace(materialize_media_render=AsyncMock()), uuid4(),
                                      managed_plugins=self.runtime)
        broker._managed = AsyncMock(return_value=self.job)
        broker.allocation = AsyncMock(side_effect=AssertionError("No allocation may occur"))
        with self.assertRaises(TaskError):
            await broker.execute(CommandRequest(operation_key="bypass", command="echo bypass"))
        plan = render_plan(self.job, self.tasks.task)
        with self.assertRaises(SandboxError):
            await broker.execute_media(replace(plan, owner_id="other-owner"))
        broker.allocation.assert_not_awaited()

    async def test_exact_plan_is_required_and_files_data_cannot_execute_renderer(self):
        plan = render_plan(self.job, self.tasks.task)
        self.job.bundle.require_command(plan.command(), plan=plan)
        with self.assertRaises(TaskError):
            self.job.bundle.require_command(plan.command())
        with self.assertRaises(TaskError):
            self.job.bundle.require_command(plan.command().model_copy(update={"command": "echo bypass"}), plan=plan)
        broker = SandboxCommandBroker(self.context, SimpleNamespace(), uuid4(), managed_plugins=self.runtime)
        broker._managed = AsyncMock(return_value=fixture_scope()[0])
        with self.assertRaises(SandboxError):
            await broker.execute_media(plan)

    async def test_materialization_precedes_submission_and_is_not_repeated_for_unknown(self):
        plan = render_plan(self.job, self.tasks.task)
        handle = SimpleNamespace(sandbox_id=plan.sandbox_id, template_id=plan.binding.physical_template_id)
        events = []
        async def materialize(*_):
            events.append("materialize")
        async def begin(*_):
            events.append("submit")
            return uuid4()
        async def start(*_, **kwargs):
            events.append("start")
        provider = SimpleNamespace(materialize_media_render=AsyncMock(side_effect=materialize),
                                   start_managed=AsyncMock(side_effect=start))
        broker = SandboxCommandBroker(self.context, provider, uuid4(), managed_plugins=self.runtime)
        broker._managed = AsyncMock(return_value=self.job)
        allocation = dict(id=uuid4())
        command = dict(id=uuid4(), phase="prepared", request_digest="a" * 64)
        broker.allocation = AsyncMock(return_value=allocation)
        broker._ready = AsyncMock(return_value=allocation)
        broker.allocations = SimpleNamespace(handle=lambda _: handle)
        broker.commands = SimpleNamespace(prepare=AsyncMock(return_value=command),
            begin_submission=AsyncMock(side_effect=begin), uncertain=AsyncMock())
        broker.observe = AsyncMock(return_value={"state": "running"})
        self.assertEqual(await broker.execute_media(plan), {"state": "running"})
        self.assertEqual(events, ["materialize", "submit", "start"])
        command["phase"] = "uncertain"
        await broker.execute_media(plan)
        self.assertEqual(events, ["materialize", "submit", "start"])
        broker.observe.assert_awaited_with(command["id"])


if __name__ == "__main__":
    unittest.main()
