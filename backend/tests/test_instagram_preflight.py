"""HTTP-boundary tests using a fake Meta transport; no account/provider calls."""
import json
import unittest

import httpx
from pydantic import SecretStr

from app.instagram_preflight import discover_facebook_linked_instagram


TOKEN = "private-test-token-do-not-return"


def row(page="10", ig="20", name="Demo Page"):
    return {"id": page, "name": name, "tasks": ["CREATE_CONTENT"],
            "instagram_business_account": {"id": ig}, "access_token": "unsolicited-secret"}


class InstagramPreflightTests(unittest.IsolatedAsyncioTestCase):
    async def check(self, handler, token=SecretStr(TOKEN)):
        calls = []
        def handle(request):
            calls.append(request)
            return handler(request, len(calls))
        result = await discover_facebook_linked_instagram(
            api_version="v22.0", user_token=token, transport=httpx.MockTransport(handle))
        serialized = repr(result)
        self.assertNotIn(TOKEN, serialized)
        self.assertNotIn("unsolicited-secret", serialized)
        self.assertEqual(result.publication_readiness, "unknown")
        for request in calls:
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.url.host, "graph.facebook.com")
            self.assertEqual(request.url.path, "/v22.0/me/accounts")
            self.assertNotIn(TOKEN, str(request.url))
            self.assertNotIn("access_token", request.url.params["fields"])
            self.assertEqual(request.headers["Authorization"], "Bearer " + TOKEN)
        return result, calls

    async def test_missing_connection_never_calls_provider(self):
        result, calls = await self.check(lambda *_: self.fail("network attempted"), token=None)
        self.assertEqual(result.outcome, "connection_required")
        self.assertEqual(calls, [])

    async def test_discovers_multiple_accounts_without_selecting_or_claiming_publish_ready(self):
        result, calls = await self.check(lambda *_: httpx.Response(200, json={"data": [row(), row("11", "21")]}))
        self.assertEqual([x.instagram_user_id for x in result.candidates], ["20", "21"])
        self.assertTrue(result.enumeration_complete)
        self.assertEqual(len(calls), 1)

    async def test_empty_result_does_not_diagnose_personal_account(self):
        result, _ = await self.check(lambda *_: httpx.Response(200, json={"data": []}))
        self.assertEqual(result.reason, "no_linked_accounts_observed")
        self.assertEqual(result.candidates, ())

    async def test_pagination_ignores_url_and_uses_only_cursor_on_fixed_host(self):
        def handler(request, n):
            if n == 1:
                return httpx.Response(200, json={"data": [row()], "paging": {
                    "next": "https://untrusted.invalid/steal?access_token=" + TOKEN,
                    "cursors": {"after": "next-cursor"}}})
            self.assertEqual(request.url.params["after"], "next-cursor")
            return httpx.Response(200, json={"data": [row("11", "21")]} )
        result, calls = await self.check(handler)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result.candidates), 2)

    async def test_redirect_is_not_followed_or_treated_as_account_type(self):
        result, calls = await self.check(lambda *_: httpx.Response(302, headers={"location": "https://untrusted.invalid"}))
        self.assertEqual(result.outcome, "unknown")
        self.assertEqual(len(calls), 1)

    async def test_rejected_or_provider_error_never_returns_error_body(self):
        for status in (400, 401, 403, 429, 500, 200):
            with self.subTest(status=status):
                result, calls = await self.check(lambda *_: httpx.Response(status, json={
                    "error": {"message": TOKEN, "access_token": "unsolicited-secret"}}))
                self.assertEqual(result.outcome, "unknown")
                self.assertEqual(len(calls), 1)

    async def test_timeout_has_no_retry_or_exception_secret(self):
        def handler(request, _):
            raise httpx.ReadTimeout(TOKEN, request=request)
        result, calls = await self.check(handler)
        self.assertEqual(result.reason, "provider_unavailable")
        self.assertEqual(len(calls), 1)

    async def test_conflicting_page_identity_discards_partial_discovery(self):
        result, _ = await self.check(lambda *_: httpx.Response(200, json={"data": [row(), row(ig="99")]}))
        self.assertEqual(result.reason, "conflicting_account_observations")
        self.assertEqual(result.candidates, ())

    async def test_truncated_enumeration_is_explicit(self):
        result, calls = await self.check(lambda _, n: httpx.Response(200, json={
            "data": [row(str(n), str(n+10))], "paging": {
                "next": "unused", "cursors": {"after": str(n)}}}))
        self.assertEqual(len(calls), 3)
        self.assertFalse(result.enumeration_complete)
        self.assertEqual(result.reason, "page_limit_reached")

    async def test_repeated_cursor_is_not_an_infinite_loop(self):
        result, calls = await self.check(lambda *_: httpx.Response(200, json={"data": [],
            "paging": {"next": "unused", "cursors": {"after": "same"}}}))
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.reason, "pagination_did_not_advance")

    async def test_malformed_oversized_and_duplicate_key_responses(self):
        for content in (b'{"data":[],"data":[{}]}', b'{"data":{}}', b'not-json',
                        json.dumps({"data": [row(ig="../../other")]}).encode(),
                        b' ' * (256 * 1024 + 1)):
            with self.subTest(length=len(content)):
                result, _ = await self.check(lambda *_: httpx.Response(200, content=content))
                self.assertEqual(result.outcome, "unknown")
                self.assertEqual(result.candidates, ())

    async def test_version_and_token_types_are_validated_before_network(self):
        for kwargs in ({"api_version": "v22.0/../../evil", "user_token": SecretStr(TOKEN)},
                       {"api_version": "v22.0", "user_token": TOKEN},
                       {"api_version": "v22.0", "user_token": SecretStr("bad\nheader")}):
            with self.assertRaises((TypeError, ValueError)):
                await discover_facebook_linked_instagram(**kwargs)
