from datetime import UTC, datetime
import unittest
from uuid import uuid4

from pydantic import ValidationError

from app.knowledge.models import SourcePublication


class MemorySourceContractTests(unittest.TestCase):
    def request(self, **changes):
        return SourcePublication(**(dict(
            area_id=uuid4(), namespace="fixture", source_key="original-one", payload_id=uuid4(),
            source_type="observation", trust="sensor", source_time=datetime.now(UTC),
            time_basis="original camera timestamp", expected_revision=0,
        ) | changes))

    def test_source_metadata_preserves_unknown_time_and_assistant_origin(self):
        request = self.request(source_time=None, time_basis="not supplied", trust="assistant_output")
        self.assertIsNone(request.source_time)
        self.assertEqual(request.trust, "assistant_output")

    def test_publication_is_immutable_and_rejects_provider_selected_routing(self):
        request = self.request()
        with self.assertRaises(ValidationError):
            request.payload_id = uuid4()
        for field in ("bank_name", "provider_url", "owner_id", "content_digest"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                self.request(**{field: "model-selected"})

    def test_naive_time_invalid_revision_and_blank_identity_are_rejected(self):
        for change in (
            {"source_time": datetime.now()}, {"expected_revision": True},
            {"expected_revision": -1}, {"source_key": " "}, {"namespace": "\x00"},
        ):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                self.request(**change)
