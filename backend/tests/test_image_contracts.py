from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.images import ImageReference, ImageUnavailableError, ResolvedImage


class ImageContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)
        self.reference = ImageReference(
            frame_id=uuid4(), owner_id="private-owner", session_id=uuid4(),
            source="historical", captured_at=self.now,
            content_type="image/jpeg", width=48, height=32,
        )

    def _current(self) -> ImageReference:
        return replace(
            self.reference,
            source="current", visual_session_id=uuid4(), frame_sequence=0,
            stream_connection_id=uuid4(), server_received_at=self.now,
            sampled_at=self.now, sampled_monotonic=100.0,
            raw_image_expires_at=self.now + timedelta(seconds=60),
        )

    def test_historical_reference_does_not_require_live_stream_fields(self) -> None:
        self.assertEqual(self.reference.source, "historical")
        self.assertIsNone(self.reference.stream_connection_id)

    def test_current_reference_requires_complete_source_identity_and_timing(self) -> None:
        current = self._current()
        for name in (
            "visual_session_id", "frame_sequence", "stream_connection_id",
            "server_received_at", "sampled_at", "sampled_monotonic", "raw_image_expires_at",
        ):
            with self.subTest(field=name):
                with self.assertRaises(ValueError):
                    replace(current, **{name: None})

    def test_current_freshness_uses_original_monotonic_sample(self) -> None:
        freshness = self._current().freshness(
            now=self.now, monotonic_now=111.0, max_age_seconds=10,
        )
        self.assertEqual(freshness.status, "stale")
        self.assertEqual(freshness.age_seconds, 11.0)

    def test_historical_image_cannot_certify_current_view(self) -> None:
        with self.assertRaises(ValueError):
            self.reference.freshness(now=self.now, monotonic_now=100.0, max_age_seconds=10)

    def test_reference_metadata_is_json_serializable_and_metadata_only(self) -> None:
        metadata = self._current().to_metadata()
        serialized = json.dumps(metadata)
        self.assertEqual(json.loads(serialized), metadata)
        self.assertEqual(metadata["captured_at"], self.now.isoformat())
        self.assertEqual(metadata["frame_sequence"], 0)
        self.assertEqual(metadata["sampled_monotonic"], 100.0)
        self.assertNotIn("private-owner", serialized)
        self.assertTrue({
            "owner_id", "image_bytes", "image_url", "signed_url", "storage_bucket",
            "storage_path", "description", "confidence", "provider_item_id", "image_status",
        }.isdisjoint(metadata))

    def test_historical_metadata_retains_evidence_relationship(self) -> None:
        record_id = uuid4()
        metadata = replace(
            self.reference, record_type="experience_event", record_id=record_id,
        ).to_metadata()
        self.assertEqual(metadata["record_type"], "experience_event")
        self.assertEqual(metadata["record_id"], str(record_id))
        self.assertNotIn("sampled_monotonic", metadata)

    def test_evidence_record_type_and_id_must_be_paired(self) -> None:
        for changes in ({"record_type": "experience_event"}, {"record_id": uuid4()}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(self.reference, **changes)

    def test_invalid_reference_metadata_is_rejected(self) -> None:
        for changes in (
            {"owner_id": " "}, {"source": "unknown"}, {"content_type": "image/png"},
            {"width": 0}, {"height": -1}, {"frame_sequence": -1},
            {"sampled_monotonic": float("nan")}, {"sampled_monotonic": float("inf")},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(self.reference, **changes)

    def test_naive_timestamps_are_rejected(self) -> None:
        for field in ("captured_at", "raw_image_expires_at", "server_received_at", "sampled_at"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    replace(self.reference, **{field: self.now.replace(tzinfo=None)})

    def test_reference_and_resolved_image_are_frozen(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.reference.width = 96
        resolved = ResolvedImage(reference=self.reference, image_bytes=b"private-pixel-payload")
        with self.assertRaises(FrozenInstanceError):
            resolved.reference = self._current()

    def test_resolved_bytes_are_excluded_from_repr_and_not_json_serializable(self) -> None:
        resolved = ResolvedImage(reference=self.reference, image_bytes=b"private-pixel-payload")
        self.assertNotIn("private-pixel-payload", repr(resolved))
        with self.assertRaises(TypeError):
            json.dumps(resolved)

    def test_unavailable_error_preserves_stable_reason_and_message(self) -> None:
        error = ImageUnavailableError("source_changed", "The camera stream has changed.")
        self.assertEqual(error.code, "source_changed")
        self.assertEqual(error.message, str(error))
        self.assertEqual(error.args, (error.message,))


if __name__ == "__main__":
    unittest.main()
