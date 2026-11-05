"""Desktop admission and bounded image transport, independent of any application."""
import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from PIL import Image
import pytest

from app.images import ImageUnavailableError
from app.runtime.feature_config import Capability, SandboxConfiguration, SecondBrainSettings
from app.sandbox.images import SandboxImageTools
from app.sandbox.inputs import SandboxInputService
from app.sandbox.models import SandboxPolicy
from app.sandbox.tools import SandboxToolBridge
from app.tasks.models import TaskError
from app.tasks.service import TaskService


def pixels(format="PNG"):
    output = BytesIO()
    Image.new("RGB", (48, 32), "orange").save(output, format=format)
    return output.getvalue()


def image_fixture():
    context = SimpleNamespace(task=SimpleNamespace(owner_id="fixture", id=uuid4()),
        claim=SimpleNamespace(attempt_id=uuid4()), reserve=AsyncMock(return_value={"id": uuid4()}),
        budgets=SimpleNamespace(submit=AsyncMock()), settle=AsyncMock())
    bridge = SimpleNamespace(context=context, _check=AsyncMock())
    allocation = {"id": uuid4()}
    provider = SimpleNamespace(read_workspace_file=AsyncMock(return_value=pixels()))
    broker = SimpleNamespace(_ready=AsyncMock(return_value=allocation), allocation=AsyncMock(return_value=allocation),
        allocations=SimpleNamespace(current=AsyncMock(return_value=allocation), handle=lambda _: object()),
        provider=provider)
    inputs = SimpleNamespace(revalidate_staged=AsyncMock())
    return SandboxImageTools(bridge, broker, inputs), bridge, provider


@pytest.mark.parametrize("environment,template,network", [
    ("offline_sandbox", "base", False), ("networked_sandbox", "base", True),
    ("desktop_sandbox", "desktop", False), ("networked_desktop_sandbox", "desktop", True),
])
def test_ceiling_never_implicitly_upgrades_request(environment, template, network):
    service = TaskService(None, {}, sandbox_configuration=SandboxConfiguration(
        policy=SandboxPolicy(template="desktop", internet_access=True)))
    policy = service.execution_policy(environment)
    assert policy.template == template and policy.internet_access is network


def test_desktop_admission_is_independent_of_network():
    service = TaskService(None, {}, sandbox_configuration=SandboxConfiguration(
        policy=SandboxPolicy(internet_access=True)))
    with pytest.raises(TaskError) as error:
        service.execution_policy("desktop_sandbox")
    assert error.value.code == "sandbox_desktop_disabled"


def test_desktop_flag_defaults_off_and_invalid_value_fails_closed():
    for selected, configured, template in [("false", True, "base"), ("true", True, "desktop"), ("maybe", False, None)]:
        flags = SecondBrainSettings(_env_file=None, enabled="true", sandbox_enabled="true",
                                    sandbox_desktop_enabled=selected)
        result = flags.resolve(Capability.SANDBOX, database_url="configured")
        assert result.configured is configured
        if configured:
            assert result.configuration.policy.template == template


@pytest.mark.parametrize("format,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_image_formats_keep_exact_pixels_and_dimensions(format, mime):
    assert SandboxImageTools.inspect(pixels(format)) == (mime, 48, 32)


@pytest.mark.parametrize("content", [b"", b"not an image", pixels("GIF"), pixels()[:50]])
def test_invalid_image_is_not_forwarded(content):
    with pytest.raises(ImageUnavailableError):
        SandboxImageTools.inspect(content)


@pytest.mark.asyncio
async def test_workspace_snapshot_has_real_provenance_not_invented_camera_metadata():
    images, bridge, provider = image_fixture()
    selected = await images.select({"path": "screen.png"})
    resolved = await images.resolve(selected)
    assert resolved.image_bytes == pixels()
    assert selected["image"]["source"] == "sandbox_workspace"
    assert "observed_at" in selected["image"]
    assert not {"frame_id", "captured_at", "session_id"} & selected["image"].keys()
    assert selected["image"]["task_id"] == str(bridge.context.task.id)
    bridge.context.settle.assert_awaited_once()
    provider.read_workspace_file.assert_awaited_once()
    await resolved.revalidate()
    assert provider.read_workspace_file.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/etc/passwd", "../screen.png", "", "a/../../screen.png"])
async def test_image_path_rejected_before_native_io(path):
    images, _, provider = image_fixture()
    assert (await images.select({"path": path}))["status"] == "failed"
    provider.read_workspace_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_selection_cannot_be_forged_or_moved_to_another_attempt():
    images, _, _ = image_fixture()
    selected = await images.select({"path": "screen.png"})
    other, _, _ = image_fixture()
    for reader, result in [(other, selected), (images, selected | {"image": {}}),
                           (images, selected | {"selection_id": []})]:
        with pytest.raises(ImageUnavailableError):
            await reader.resolve(result)


@pytest.mark.asyncio
async def test_revoked_scope_blocks_already_resolved_pixels():
    images, _, _ = image_fixture()
    selected = await images.select({"path": "screen.png"})
    resolved = await images.resolve(selected)
    images.broker.allocations.current.side_effect = TaskError("revoked", "Revoked")
    with pytest.raises(TaskError):
        await resolved.revalidate()


@pytest.mark.asyncio
async def test_changed_staged_input_blocks_derived_image_transport():
    images, _, _ = image_fixture()
    selected = await images.select({"path": "screen.png"})
    images.inputs.revalidate_staged.side_effect = TaskError("source_changed", "Changed")
    with pytest.raises(TaskError):
        await images.resolve(selected)


@pytest.mark.asyncio
async def test_concurrent_selections_cannot_exceed_count_or_byte_budget():
    images, _, provider = image_fixture()
    images.MAX_SELECTIONS = 1
    results = await asyncio.gather(*(images.select({"path": "screen.png"}) for _ in range(2)),
                                   return_exceptions=True)
    assert sum(isinstance(result, TaskError) for result in results) == 1
    provider.read_workspace_file.assert_awaited_once()
    images, _, provider = image_fixture()
    images.MAX_TOTAL_BYTES = len(pixels()) - 1
    with pytest.raises(TaskError):
        await images.select({"path": "screen.png"})
    assert not images.selections and images.total_bytes == 0


@pytest.mark.asyncio
async def test_staged_source_callbacks_are_rechecked():
    inputs = SandboxInputService(SimpleNamespace(), object())
    check = AsyncMock()
    inputs._staged_revalidators.append(check)
    await inputs.revalidate_staged()
    check.assert_awaited_once()


@pytest.mark.asyncio
async def test_composed_bridge_routes_workspace_images_without_replacing_original_image_tools():
    images, _, _ = image_fixture()
    selected = await images.select({"path": "screen.png"})
    bridge = object.__new__(SandboxToolBridge)
    bridge.images = images
    bridge.base = SimpleNamespace(resolve_image=AsyncMock(return_value="original-camera-reader"))
    assert (await bridge.resolve_image("sandbox_view_image", selected)).image_bytes == pixels()
    assert await bridge.resolve_image("read_task_context_image", {}) == "original-camera-reader"
