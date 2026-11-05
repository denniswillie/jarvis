from __future__ import annotations

import unittest

from app.calendar.google_oauth import GoogleCalendarError, GoogleOAuthStateCodec


class GoogleOAuthStateCodecTests(unittest.TestCase):
    def test_round_trip_returns_bound_owner(self) -> None:
        codec = GoogleOAuthStateCodec("test-secret", ttl_seconds=600)
        token = codec.encode(owner_id="owner", now=1_000)

        self.assertEqual(codec.decode(token, now=1_100), "owner")

    def test_tampered_state_is_rejected(self) -> None:
        codec = GoogleOAuthStateCodec("test-secret", ttl_seconds=600)
        token = codec.encode(owner_id="owner", now=1_000)
        payload, signature = token.split(".", 1)

        with self.assertRaises(GoogleCalendarError) as context:
            codec.decode(f"{payload}x.{signature}", now=1_100)

        self.assertEqual(context.exception.code, "google_oauth_state_invalid")

    def test_expired_state_is_rejected(self) -> None:
        codec = GoogleOAuthStateCodec("test-secret", ttl_seconds=600)
        token = codec.encode(owner_id="owner", now=1_000)

        with self.assertRaises(GoogleCalendarError) as context:
            codec.decode(token, now=1_601)

        self.assertEqual(context.exception.code, "google_oauth_state_expired")


if __name__ == "__main__":
    unittest.main()
