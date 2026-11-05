"""Opaque browse positions reject malformed values without granting access."""
import base64
from datetime import datetime, timezone
import json
import unittest
from uuid import uuid4

from app.files.pagination import decode_cursor, encode_cursor
from app.knowledge.models import SourceError


class FileCursorTests(unittest.TestCase):
    def test_round_trip_preserves_exact_position(self):
        timestamp, identity = datetime.now(timezone.utc), uuid4()
        self.assertEqual(decode_cursor(encode_cursor(timestamp, identity)), (timestamp, identity))
        self.assertEqual(decode_cursor(None), (None, None))

    def test_untrusted_positions_are_bounded_and_well_formed(self):
        def encoded(value):
            return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
        for value in ("", "a" * 201, "?", "a", encoded({}), encoded(["2026-11-02", str(uuid4())]),
                      encoded([datetime.now(timezone.utc).isoformat(), "not-an-id"]), encoded([1, 2])):
            with self.subTest(value=value), self.assertRaises(SourceError):
                decode_cursor(value)
