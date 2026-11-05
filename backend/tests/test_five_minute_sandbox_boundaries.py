"""Pure/mocked source-input contracts; no allocation, process or media decode.

Large sizes are metadata only. Embedded programs run as definitions with their
entrypoints disabled; installer admission is observed before any filesystem I/O.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest.mock import Mock
from uuid import UUID

from app.files.transfer import CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS
from app.sandbox.file_transfer import INSTALL_INPUT_FILE, MAX_TRANSFER_BYTES
from app.sandbox.media_decode import (
    DECODE_MEDIA_INPUT, MediaDecodeReceipt, MediaInspectionTiming,
    decode_request, parse_decode_receipt,
)
from app.sandbox.models import SandboxInfo, SandboxPolicy, SandboxScope
from app.workspace.media_recipe import (
    MediaRecipeError, SourceInspection, TemplateInventoryEntry, validate_recipe,
)
from app.workspace.media_render import RendererBinding, prepare_render
from app.workspace.media_staging import INSTALL_RENDER_INPUTS, materialization_request
from tests.test_workspace_media_render import inputs


CAPTURE_BYTES = 256 * 1024**2
CAPTURE_MS = 300_000
OUTPUT_BYTES = 32 * 1024**2


def definitions(program):
    scope = {"__name__": "offline_boundary_checks"}
    exec(compile(program, "<trusted-fixed-program>", "exec"), scope)
    return scope


def five_minute_inputs():
    args = inputs()
    binding = args["binding"].model_dump()
    binding["policy"].update(max_source_bytes=CAPTURE_BYTES,
                             max_source_duration_ms=CAPTURE_MS, max_output_frames=720)
    args["binding"] = RendererBinding.model_validate(binding)
    measured = SourceInspection.model_validate(args["source"].inspection.model_dump() | {
        "size_bytes": CAPTURE_BYTES, "duration_ms": CAPTURE_MS})
    args["source"] = replace(args["source"], inspection=measured)
    recipe = args["recipe"].recipe.model_dump(mode="json")
    # A short edit near the END of a five-minute source must remain selectable.
    recipe["segments"] = [{"segment_id": "last", "source_id": "original",
                            "in_ms": 295_000, "out_ms": 300_000}]
    args["recipe"] = validate_recipe(recipe, inspections=[measured], policy=args["binding"].policy,
        inventory=[TemplateInventoryEntry(template=recipe["template"],
                                          output_profile=recipe["output_profile"], ready=True)])
    return args


def handle_for(args):
    scope = SandboxScope(owner_id=args["owner_id"], task_id=args["task_id"], task_revision=1,
                         attempt_id=UUID(int=600), allocation_id=UUID(int=601))
    return SandboxInfo(sandbox_id=args["source"].sandbox_id, scope=scope, policy=SandboxPolicy(),
        state="running", expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        template_id=args["binding"].physical_template_id)


def request_for(args, **changes):
    parameters = dict(input_path=args["source"].path,
        content_sha256=args["source"].inspection.file.content_sha256,
        size_bytes=args["source"].inspection.size_bytes, binding=args["binding"], budget_seconds=57)
    return decode_request(handle_for(args), **(parameters | changes))


def receipt_for(request, **changes):
    return {"format": "jarvis.media-decode.v1", **{key: request[key] for key in
        ("sandbox_id", "allocation_id", "input_path", "content_sha256", "size_bytes", "software_sha256")},
        "duration_ms": CAPTURE_MS, "video_streams": 1, "audio_streams": 1,
        "video_decoded": True, "audio_decoded": True, "video_frames": 7200,
        "audio_duration_us": CAPTURE_MS * 1000,
        "inspection_timing": {"inventory_ms": 10, "metadata_ms": 20, "video_decode_ms": 30,
                              "audio_decode_ms": 40, "total_ms": 100}, **changes}


class FiveMinuteSandboxBoundaries(unittest.TestCase):
    def test_short_late_edit_of_five_minute_source_keeps_output_and_manifest_exact(self):
        self.assertEqual((CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS), (CAPTURE_BYTES, CAPTURE_MS))
        plan = prepare_render(**five_minute_inputs())
        envelope = json.loads(plan.envelope_bytes)
        self.assertEqual(envelope["sources"][0]["durationMs"], CAPTURE_MS)
        self.assertEqual(envelope["recipe"]["segments"][0]["toFrame"], 7200)
        self.assertEqual(plan.recipe.duration_in_frames, 120)
        self.assertEqual(plan.staging()["size_bytes"], CAPTURE_BYTES)
        self.assertEqual(plan.binding.max_output_bytes, 20_000_000)
        self.assertEqual(plan.binding.policy.max_output_frames, 720)

    def test_explicit_smaller_source_grants_are_not_widened(self):
        args = inputs()  # Existing explicit 20 MB / 30 s operation allowance.
        request = request_for(args, size_bytes=20_000_000)
        self.assertEqual(request["max_duration_ms"], 30_000)
        with self.assertRaises(ValueError):
            request_for(args, size_bytes=20_000_001)
        for changes in ({"size_bytes": 20_000_001}, {"duration_ms": 30_001}):
            measured = SourceInspection.model_validate(args["source"].inspection.model_dump() | changes)
            with self.subTest(changes=changes), self.assertRaises(MediaRecipeError):
                prepare_render(**(args | {"source": replace(args["source"], inspection=measured)}))

    def test_host_and_embedded_decoder_accept_exact_source_boundary(self):
        request = request_for(five_minute_inputs())
        program = definitions(DECODE_MEDIA_INPUT)
        program["validate"](request)
        self.assertEqual((request["size_bytes"], request["max_duration_ms"], request["budget_seconds"]),
                         (CAPTURE_BYTES, CAPTURE_MS, 57))
        measured = parse_decode_receipt(json.dumps(receipt_for(request)), request)
        self.assertEqual((measured.size_bytes, measured.duration_ms), (CAPTURE_BYTES, CAPTURE_MS))

    def test_decoder_global_bounds_are_positive_and_do_not_extend_wall_time(self):
        args = five_minute_inputs()
        request = request_for(args)
        program = definitions(DECODE_MEDIA_INPUT)
        for changes in ({"size_bytes": CAPTURE_BYTES + 1}, {"max_duration_ms": CAPTURE_MS + 1},
                        {"size_bytes": 0}, {"max_duration_ms": 0}, {"size_bytes": True},
                        {"budget_seconds": 58}, {"budget_seconds": 0}):
            with self.subTest(remote=changes), self.assertRaises(ValueError):
                program["validate"](request | changes)
        for changes in ({"size_bytes": CAPTURE_BYTES + 1}, {"size_bytes": 0},
                        {"budget_seconds": 58}, {"budget_seconds": 0}):
            with self.subTest(host=changes), self.assertRaises(ValueError):
                request_for(args, **changes)
        with self.assertRaises(ValueError):
            MediaInspectionTiming(inventory_ms=0, metadata_ms=0, video_decode_ms=0,
                                  audio_decode_ms=0, total_ms=60_001)
        too_slow = receipt_for(request, inspection_timing={"inventory_ms": 0, "metadata_ms": 0,
            "video_decode_ms": 57_001, "audio_decode_ms": 0, "total_ms": 57_001})
        with self.assertRaises(ValueError):
            parse_decode_receipt(json.dumps(too_slow), request)

    def test_decode_receipt_cannot_claim_overlong_or_different_input(self):
        request = request_for(five_minute_inputs())
        for changes in ({"duration_ms": CAPTURE_MS + 1}, {"size_bytes": CAPTURE_BYTES + 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                MediaDecodeReceipt.model_validate(receipt_for(request, **changes))
        for changes in ({"content_sha256": "f" * 64}, {"size_bytes": CAPTURE_BYTES - 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parse_decode_receipt(json.dumps(receipt_for(request, **changes)), request)
        with self.assertRaises(ValueError):
            parse_decode_receipt(json.dumps(receipt_for(request)), request | {"max_duration_ms": CAPTURE_MS - 1})

    def test_input_installer_preserves_capture_output_and_smaller_byte_allowances(self):
        program = definitions(INSTALL_INPUT_FILE)
        class ReachedFilesystemBoundary(Exception):
            pass
        boundary = Mock(side_effect=ReachedFilesystemBoundary)
        program["protected_directory"] = boundary
        base = {"identity": "a" * 64, "stage_id": "b" * 32, "sha256": "c" * 64}
        for policy, limit in (("mp4_capture_v1", CAPTURE_BYTES),
                              ("mp4_capture_v1", 20_000_000), ("mp4_output_v1", OUTPUT_BYTES)):
            boundary.reset_mock()
            request = base | {"policy": policy, "limit": limit, "size": limit}
            with self.subTest(policy=policy, limit=limit), self.assertRaises(ReachedFilesystemBoundary):
                program["install"](request)
            boundary.assert_called_once_with("/var/lib/jarvis/incoming")
            boundary.reset_mock()
            with self.assertRaises(ValueError):
                program["install"](request | {"size": limit + 1})
            boundary.assert_not_called()
        for policy, limit in (("mp4_capture_v1", CAPTURE_BYTES + 1),
                              ("mp4_output_v1", OUTPUT_BYTES + 1), ("mp4_capture_v1", 0),
                              ("mp4_capture_v1", True), ("document_v1", 8 * 1024**2)):
            boundary.reset_mock()
            with self.subTest(policy=policy, limit=limit), self.assertRaises(ValueError):
                program["install"](base | {"policy": policy, "limit": limit, "size": 1})
            boundary.assert_not_called()
        self.assertEqual(MAX_TRANSFER_BYTES, 8 * 1024**2)

    def test_render_materialization_validates_new_bound_before_any_io(self):
        plan = prepare_render(**five_minute_inputs())
        request = materialization_request(plan)
        program = definitions(INSTALL_RENDER_INPUTS)
        self.assertEqual(program["validate"](request), plan.envelope_bytes)
        smaller = request | {"limit_bytes": 20_000_000, "size_bytes": 20_000_000}
        self.assertEqual(program["validate"](smaller), plan.envelope_bytes)
        for changed in ({"limit_bytes": CAPTURE_BYTES + 1}, {"size_bytes": CAPTURE_BYTES + 1},
                        {"limit_bytes": 0}, {"size_bytes": 0}, {"limit_bytes": True}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                program["validate"](request | changed)
        with self.assertRaises(ValueError):
            program["validate"](smaller | {"size_bytes": 20_000_001})


if __name__ == "__main__":
    unittest.main()
