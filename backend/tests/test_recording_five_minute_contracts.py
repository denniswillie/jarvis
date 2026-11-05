"""Offline capture/inspection/delivery contracts; no media bytes are allocated.

The ffprobe-shaped metadata below is synthetic input to the pure parser. These
checks establish coherent limits, not actual capture, decoding or cloud storage.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import unittest
from uuid import UUID

from app.files.media_inspection import MediaInspectionError, _parse_metadata
from app.files.recording_ingress import RecordingClientReport, RecordingGap, RecordingUpload
from app.files.transfer import (
    CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS, DOCUMENT_POLICY, FileManifest,
    FileTransferPolicy, MP4_CAPTURE_POLICY, MP4Inspection, MP4_OUTPUT_POLICY,
)
from app.workspace.presentation import PreparedContent, PresentationRequest
from app.recordings.models import RecordingScope, FINALIZATION_FORMAT, freeze_manifest


# Independent expected values pin the accepted contract; a coordinated accidental
# widening of all imported models must not make the regression disappear.
CAPTURE_BYTES = 1024**3
CAPTURE_MILLISECONDS = 1_800_000
OUTPUT_BYTES = 32 * 1024**2
DOCUMENT_BYTES = 8 * 1024**2
SHA256 = "a" * 64
FILE_ID = UUID("bc000000-0000-4000-8000-000000000001")
GENERATION_ID = UUID("bc000000-0000-4000-8000-000000000002")
REQUEST_ID = UUID("bc000000-0000-4000-8000-000000000003")


def report(**changes):
    return RecordingClientReport(**({
        "generation_id": GENERATION_ID,
        "timeline_end_ms": CAPTURE_MILLISECONDS,
        "video_frames": 30 * 60 * 24,
        "source_audio_frames": 30 * 60 * 48_000,
        "padded_audio_frames": 0,
    } | changes))


def upload(**changes):
    return RecordingUpload(**({
        "filename": "thirty-minute-recording.mp4",
        "description": "Synthetic thirty-minute contract fixture",
        "size_bytes": CAPTURE_BYTES,
        "content_sha256": SHA256,
        "client_report": report(),
    } | changes))


def inspection(**changes):
    return MP4Inspection(**({
        "content_sha256": SHA256,
        "size_bytes": CAPTURE_BYTES,
        "duration_ms": CAPTURE_MILLISECONDS,
        "width": 1920,
        "height": 1080,
        "video_codec": "h264",
        "audio_codec": "aac",
    } | changes))


def probe_metadata(*, size=CAPTURE_BYTES, duration="1800.000000"):
    return {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": duration,
                   "size": str(size), "tags": {"major_brand": "isom"}},
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1920,
             "height": 1080, "duration": duration, "disposition": {"attached_pic": 0}},
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "duration": duration},
        ],
    }


def delivery_fields(count, policy=None, limit=None):
    fields = {"file_id": FILE_ID, "byte_count": count, "content_sha256": SHA256}
    if policy is not None:
        fields.update(transfer_policy=policy, transfer_limit_bytes=limit)
    return fields


def ready_request(fields):
    now = datetime(2026, 11, 2, tzinfo=timezone.utc)
    return PresentationRequest(**({
        "request_id": REQUEST_ID, "action": "download", "artifact_kind": "file",
        "artifact_id": FILE_ID, "version_id": "recording:exact-version",
        "content_ready": True, "created_at": now, "expires_at": now + timedelta(minutes=5),
    } | fields))


class ThirtyMinuteRecordingContracts(unittest.TestCase):
    def setUp(self):
        self.capture = FileTransferPolicy(MP4_CAPTURE_POLICY, CAPTURE_BYTES)
        self.output = FileTransferPolicy(MP4_OUTPUT_POLICY, OUTPUT_BYTES)

    def test_exact_thirty_minute_capture_crosses_descriptor_inspection_and_delivery(self):
        self.assertEqual((CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS),
                         (CAPTURE_BYTES, CAPTURE_MILLISECONDS))
        declared = upload()
        declared = RecordingUpload.model_validate_json(declared.model_dump_json())
        manifest = FileManifest(size_bytes=declared.size_bytes, content_sha256=declared.content_sha256)
        measured = _parse_metadata(json.dumps(probe_metadata()).encode(), manifest, self.capture)
        measured.validate_file(manifest, self.capture)
        prepared = PreparedContent(**delivery_fields(manifest.size_bytes, MP4_CAPTURE_POLICY, CAPTURE_BYTES))
        request = ready_request(prepared.model_dump())
        self.assertEqual(measured.duration_ms, declared.client_report.timeline_end_ms)
        self.assertEqual(measured.audio_codec, "aac")
        self.assertEqual((request.byte_count, request.content_sha256),
                         (measured.size_bytes, measured.content_sha256))
        self.assertEqual(request.transfer_limit_bytes, CAPTURE_BYTES)
        self.assertEqual(request.state, "pending")
        self.assertIsNone(request.local_receipt)

    def test_capture_metadata_cannot_exceed_byte_ceiling_or_accept_invalid_counts(self):
        for size in (CAPTURE_BYTES + 1, -1, True):
            with self.subTest(model="manifest", size=size), self.assertRaises(ValueError):
                FileManifest(size_bytes=size, content_sha256=SHA256)
        self.assertEqual(FileManifest(size_bytes=0, content_sha256=SHA256).size_bytes, 0)
        for make in (upload, inspection):
            for size in (CAPTURE_BYTES + 1, 0, -1, True):
                with self.subTest(model=make.__name__, size=size), self.assertRaises(ValueError):
                    make(size_bytes=size)
        # A larger deployment value cannot silently widen the selected profile.
        self.assertEqual(FileTransferPolicy(MP4_CAPTURE_POLICY, CAPTURE_BYTES + 1).max_bytes,
                         CAPTURE_BYTES)

    def test_inspection_duration_and_output_policy_remain_separate(self):
        for duration in (1_800_101, 0, -1, True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                inspection(duration_ms=duration)
        exact_output = inspection(size_bytes=OUTPUT_BYTES, duration_ms=30_000)
        exact_output.validate_file(FileManifest(size_bytes=OUTPUT_BYTES, content_sha256=SHA256), self.output)
        for size, duration in ((OUTPUT_BYTES + 1, 30_000), (OUTPUT_BYTES, 30_001)):
            measured = inspection(size_bytes=size, duration_ms=duration)
            manifest = FileManifest(size_bytes=size, content_sha256=SHA256)
            measured.validate_file(manifest, self.capture)
            with self.subTest(size=size, duration=duration), self.assertRaises(ValueError):
                measured.validate_file(manifest, self.output)

    def test_measured_capture_respects_narrower_deployment_bytes_and_exact_manifest(self):
        measured = inspection()
        exact = FileManifest(size_bytes=CAPTURE_BYTES, content_sha256=SHA256)
        with self.assertRaises(ValueError):
            measured.validate_file(exact, FileTransferPolicy(MP4_CAPTURE_POLICY, CAPTURE_BYTES - 1))
        for changed in ({"size_bytes": CAPTURE_BYTES - 1}, {"content_sha256": "b" * 64}):
            manifest = FileManifest(**(exact.model_dump() | changed))
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                measured.validate_file(manifest, self.capture)

    def test_synthetic_probe_rejects_any_track_past_encoding_allowance(self):
        original = probe_metadata()
        manifest = FileManifest(size_bytes=CAPTURE_BYTES, content_sha256=SHA256)
        for target in ("format", 0, 1):
            data = deepcopy(original)
            item = data["format"] if target == "format" else data["streams"][target]
            item["duration"] = "1800.100001"
            with self.subTest(target=target), self.assertRaises(MediaInspectionError) as raised:
                _parse_metadata(json.dumps(data).encode(), manifest, self.capture)
            self.assertEqual(raised.exception.code, "media_duration_exceeded")

    def test_synthetic_probe_does_not_promote_long_capture_to_generated_output(self):
        manifest = FileManifest(size_bytes=OUTPUT_BYTES, content_sha256=SHA256)
        raw = json.dumps(probe_metadata(size=OUTPUT_BYTES, duration="1800.000000")).encode()
        self.assertEqual(_parse_metadata(raw, manifest, self.capture).duration_ms, CAPTURE_MILLISECONDS)
        with self.assertRaises(MediaInspectionError) as raised:
            _parse_metadata(raw, manifest, self.output)
        self.assertEqual(raised.exception.code, "media_duration_exceeded")
        short = json.dumps(probe_metadata(size=OUTPUT_BYTES, duration="30.000000")).encode()
        self.assertEqual(_parse_metadata(short, manifest, self.output).duration_ms, 30_000)

    def test_codec_tail_is_separate_from_capture_admission_and_shorter_timeline(self):
        scope = RecordingScope(recording_id=FILE_ID, owner_id="fixture-owner", session_id=GENERATION_ID,
            area_id=REQUEST_ID, include_jarvis_audio=True, max_bytes=CAPTURE_BYTES,
            max_duration_ms=1_800_000, admission_event_id=FILE_ID, admission_payload_id=REQUEST_ID)
        self.assertEqual(self.capture.max_duration_ms, 1_800_000)
        self.assertEqual(self.output.max_encoded_duration_ms, 30_000)
        for duration in ("1800.086666", "1800.100000"):
            declared = upload()
            measured = _parse_metadata(json.dumps(probe_metadata(duration=duration)).encode(),
                FileManifest(size_bytes=CAPTURE_BYTES, content_sha256=SHA256), self.capture)
            manifest = {"format": FINALIZATION_FORMAT, **declared.model_dump(mode="json"),
                "duration_ms": measured.duration_ms, "media_inspection": measured.model_dump(mode="json")}
            self.assertEqual(freeze_manifest(scope, manifest).manifest["client_report"]["timeline_end_ms"], 1_800_000)
        for duration, timeline in ((1_800_101, 1_800_000), (1_800_100, 1_799_999), (1_800_001, 1_800_001)):
            with self.subTest(duration=duration, timeline=timeline), self.assertRaises(ValueError):
                freeze_manifest(scope, manifest | {"duration_ms": duration,
                    "client_report": {"timeline_end_ms": timeline}})
        with self.assertRaises(ValueError):
            report(timeline_end_ms=1_800_001)
        short_manifest = manifest | {"duration_ms": 1100, "client_report": {"timeline_end_ms": 1000}}
        freeze_manifest(scope, short_manifest)
        with self.assertRaises(ValueError):
            freeze_manifest(scope, short_manifest | {"duration_ms": 1101})

    def test_recording_gap_uses_microseconds_and_must_fit_its_report_timeline(self):
        last_gap = RecordingGap(track="audio", start_us=1_799_999_000, end_us=1_800_000_000,
                                reason="missingInput")
        self.assertEqual(report(missing_intervals=(last_gap,)).missing_intervals, (last_gap,))
        for changed in ({"start_us": 1_800_000_001, "end_us": 1_800_000_001},
                        {"end_us": 1_800_000_001}, {"start_us": 1_800_000_000, "end_us": 1_799_999_999},
                        {"start_us": -1}, {"end_us": True}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                RecordingGap(**(last_gap.model_dump() | changed))
        outside_short_report = RecordingGap(track="audio", start_us=999_000, end_us=1_001_000,
                                            reason="missingInput")
        with self.assertRaises(ValueError):
            report(timeline_end_ms=1000, source_audio_frames=48_000,
                   missing_intervals=(outside_short_report,))

    def test_client_report_rejects_overlong_timeline_and_keeps_video_counter_bounded(self):
        for timeline in (1_800_101, 0, -1, True):
            with self.subTest(timeline=timeline), self.assertRaises(ValueError):
                report(timeline_end_ms=timeline)
        self.assertEqual(report(video_frames=100_000).video_frames, 100_000)
        for frames in (100_001, 0, -1, True):
            with self.subTest(video_frames=frames), self.assertRaises(ValueError):
                report(video_frames=frames)

    def test_audio_counters_allow_thirty_minutes_but_not_double_counted_timeline(self):
        total = 30 * 60 * 48_000
        for source, padded in ((total, 0), (0, total), (total // 2, total // 2)):
            value = report(source_audio_frames=source, padded_audio_frames=padded)
            self.assertEqual(value.source_audio_frames + value.padded_audio_frames, total)
        for source, padded in ((total + 1, 0), (0, total + 1), (total, 1), (True, 0), (-1, 0)):
            with self.subTest(source=source, padded=padded), self.assertRaises(ValueError):
                report(source_audio_frames=source, padded_audio_frames=padded)
        self.assertEqual(report(timeline_end_ms=1000, source_audio_frames=48_000).source_audio_frames, 48_000)
        with self.assertRaises(ValueError):
            report(timeline_end_ms=1000, source_audio_frames=48_000, padded_audio_frames=1)

    def test_delivery_limits_are_capture_1024_output_32_document_8_mib(self):
        for policy, ceiling in ((MP4_CAPTURE_POLICY, CAPTURE_BYTES),
                                (MP4_OUTPUT_POLICY, OUTPUT_BYTES), (None, DOCUMENT_BYTES)):
            exact = delivery_fields(ceiling, policy, ceiling)
            with self.subTest(policy=policy):
                prepared = PreparedContent(**exact)
                request = ready_request(prepared.model_dump())
                self.assertEqual(request.byte_count, ceiling)
                if policy is None:
                    self.assertNotIn("transfer_policy", request.wire())
                    self.assertNotIn("transfer_limit_bytes", prepared.model_dump())
                excessive = exact | {"byte_count": ceiling + 1}
                with self.assertRaises(ValueError):
                    PreparedContent(**excessive)
                with self.assertRaises(ValueError):
                    ready_request(excessive)
        self.assertEqual(FileTransferPolicy(DOCUMENT_POLICY).max_bytes, DOCUMENT_BYTES)

    def test_delivery_cannot_widen_or_relabel_an_advertised_media_allowance(self):
        capture = delivery_fields(CAPTURE_BYTES, MP4_CAPTURE_POLICY, CAPTURE_BYTES)
        invalid = (
            capture | {"transfer_policy": MP4_OUTPUT_POLICY},
            capture | {"transfer_policy": None},
            delivery_fields(1, MP4_CAPTURE_POLICY, CAPTURE_BYTES + 1),
            delivery_fields(1, MP4_OUTPUT_POLICY, OUTPUT_BYTES + 1),
            capture | {"transfer_limit_bytes": None},
        )
        for fields in invalid:
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    PreparedContent(**fields)
                with self.assertRaises(ValueError):
                    ready_request(fields)

    def test_smaller_media_delivery_allowance_is_enforced_by_both_models(self):
        for policy, limit in ((MP4_CAPTURE_POLICY, CAPTURE_BYTES - 1),
                              (MP4_OUTPUT_POLICY, OUTPUT_BYTES - 1)):
            exact = delivery_fields(limit, policy, limit)
            with self.subTest(policy=policy):
                self.assertEqual(PreparedContent(**exact).byte_count, limit)
                self.assertEqual(ready_request(exact).byte_count, limit)
                with self.assertRaises(ValueError):
                    PreparedContent(**(exact | {"byte_count": limit + 1}))
                with self.assertRaises(ValueError):
                    ready_request(exact | {"byte_count": limit + 1})


if __name__ == "__main__":
    unittest.main()
