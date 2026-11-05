from __future__ import annotations

import unittest
import asyncio
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import asyncpg

from app.memory.prospective.repository import ProspectiveRepository
from evals.gemini_live.interruption import DatabaseFixtures


class ProspectiveDeliveryConcurrencyContractTests(unittest.TestCase):
    def test_schema_has_durable_claims_and_occurrence_uniqueness(self) -> None:
        schema = (Path(__file__).parents[1] / "app" / "schema.sql").read_text()
        self.assertIn("trigger_claim_token uuid", schema)
        self.assertIn("jarvis_prospective_focus_occurrence_unique_idx", schema)
        self.assertIn("jarvis_prospective_reminder_occurrence_unique_idx", schema)


@unittest.skipUnless(
    os.getenv("JARVIS_TEST_DB_URL"),
    "JARVIS_TEST_DB_URL is required for prospective-delivery integration tests",
)
class ProspectiveDeliveryConcurrencyIntegrationTests(
    unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(self) -> None:
        output = Path(tempfile.mkdtemp(
            prefix="prospective-stage3-", dir=os.getenv("JARVIS_TEST_ARTIFACT_DIR"),
        ))
        self.fixtures = DatabaseFixtures(
            os.environ["JARVIS_TEST_DB_URL"], output,
        )
        # Registered before connecting or writing. The shared fixture records
        # its unique owner in an atomic manifest before its first DB mutation.
        self.addAsyncCleanup(self._cleanup_fixture)
        await self.fixtures.connect()
        self.owner_id, self.session_id = await self.fixtures.create("prospective-stage3")
        self.pool = self.fixtures.pool
        self.repository = ProspectiveRepository(self.pool)

    async def _cleanup_fixture(self) -> None:
        # Includes trigger-created jobs/profile state, not only our three
        # directly written tables. Cleanup fails closed on foreign references
        # and independently counts every owner-scoped table before returning.
        await self.fixtures.cleanup()
        self.assertEqual(self.fixtures.cleanup_result["remaining_rows"], 0)

    async def test_late_scoped_stop_preserves_replacement_and_is_idempotent(self):
        old = await self._create_focus()
        replacement = await self._create_focus()
        for _ in range(2):
            self.assertIsNone(await self.repository.clear_active_visual_focus(
                owner_id=self.owner_id, expected_focus_id=old.id))
        self.assertEqual((await self.repository.get_active_visual_focus(owner_id=self.owner_id)).id,
                         replacement.id)

    async def test_scoped_stop_clears_exact_active_focus(self):
        focus = await self._create_focus()
        cleared = await self.repository.clear_active_visual_focus(
            owner_id=self.owner_id, expected_focus_id=focus.id)
        self.assertEqual(cleared.id, focus.id)
        self.assertIsNotNone(cleared.cleared_at)
        self.assertIsNone(await self.repository.get_active_visual_focus(owner_id=self.owner_id))

    async def test_scoped_stop_and_replacement_are_atomic_across_connections(self):
        old = await self._create_focus()
        _, replacement = await asyncio.gather(
            self.repository.clear_active_visual_focus(owner_id=self.owner_id, expected_focus_id=old.id),
            self._create_focus(),
        )
        self.assertEqual((await self.repository.get_active_visual_focus(owner_id=self.owner_id)).id,
                         replacement.id)

    async def test_scoped_stop_rejects_another_owners_identity(self):
        focus = await self._create_focus()
        foreign, _ = await self.fixtures.create("foreign-scoped-stop")
        self.assertIsNone(await self.repository.clear_active_visual_focus(
            owner_id=foreign, expected_focus_id=focus.id))
        self.assertEqual((await self.repository.get_active_visual_focus(owner_id=self.owner_id)).id, focus.id)

    async def test_exact_expired_focus_can_still_be_cancelled(self):
        focus = await self._create_focus()
        async with self.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_visual_focuses set created_at = now() - interval '5 minutes', "
                "expires_at = now() - interval '1 second' where owner_id=$1 and id=$2",
                self.owner_id, focus.id,
            )
        self.assertIsNone(await self.repository.get_active_visual_focus(owner_id=self.owner_id))
        cleared = await self.repository.clear_active_visual_focus(
            owner_id=self.owner_id, expected_focus_id=focus.id)
        self.assertEqual(cleared.id, focus.id)
        self.assertIsNotNone(cleared.cleared_at)

    async def test_concurrent_claim_and_occurrence_reservation_have_one_winner(
        self,
    ) -> None:
        focus = await self._create_focus()
        claims = await asyncio.gather(
            *(
                self.repository.try_claim_subject(
                    owner_id=self.owner_id,
                    subject_type="visual_focus",
                    subject_id=focus.id,
                    claim_timeout_seconds=120,
                )
                for _ in range(4)
            )
        )
        winners = [claim for claim in claims if claim is not None]
        self.assertEqual(len(winners), 1)
        claim_token = winners[0]

        events = await asyncio.gather(
            *(
                self.repository.reserve_delivery_event(
                    owner_id=self.owner_id,
                    session_id=self.session_id,
                    subject_type="visual_focus",
                    subject_id=focus.id,
                    claim_token=claim_token,
                    occurrence_key="same-occurrence",
                    message="The target is visible.",
                    evidence={"supporting_frame_ids": [str(uuid4())]},
                    dedupe_window_seconds=10,
                )
                for _ in range(4)
            )
        )
        self.assertEqual(sum(event is not None for event in events), 1)

    async def test_sliding_window_dedupes_distinct_frame_occurrences(self) -> None:
        focus = await self._create_focus()
        first_claim = await self.repository.try_claim_subject(
            owner_id=self.owner_id,
            subject_type="visual_focus",
            subject_id=focus.id,
            claim_timeout_seconds=120,
        )
        assert first_claim is not None
        first_event = await self.repository.reserve_delivery_event(
            owner_id=self.owner_id,
            session_id=self.session_id,
            subject_type="visual_focus",
            subject_id=focus.id,
            claim_token=first_claim,
            occurrence_key="first-frame-window",
            message="The target is visible.",
            evidence={"supporting_frame_ids": [str(uuid4())]},
            dedupe_window_seconds=10,
        )
        self.assertIsNotNone(first_event)
        await self.repository.release_subject_claim(
            owner_id=self.owner_id,
            subject_type="visual_focus",
            subject_id=focus.id,
            claim_token=first_claim,
        )

        second_claim = await self.repository.try_claim_subject(
            owner_id=self.owner_id,
            subject_type="visual_focus",
            subject_id=focus.id,
            claim_timeout_seconds=120,
        )
        assert second_claim is not None
        duplicate = await self.repository.reserve_delivery_event(
            owner_id=self.owner_id,
            session_id=self.session_id,
            subject_type="visual_focus",
            subject_id=focus.id,
            claim_token=second_claim,
            occurrence_key="different-frame-window",
            message="The target is still visible.",
            evidence={"supporting_frame_ids": [str(uuid4())]},
            dedupe_window_seconds=10,
        )

        self.assertIsNone(duplicate)

    async def test_claim_release_requires_token_and_stale_claim_recovers(self) -> None:
        focus = await self._create_focus()
        claim_token = await self.repository.try_claim_subject(
            owner_id=self.owner_id,
            subject_type="visual_focus",
            subject_id=focus.id,
            claim_timeout_seconds=120,
        )
        self.assertIsNotNone(claim_token)
        assert claim_token is not None

        await self.repository.release_subject_claim(
            owner_id=self.owner_id,
            subject_type="visual_focus",
            subject_id=focus.id,
            claim_token=uuid4(),
        )
        self.assertIsNone(
            await self.repository.try_claim_subject(
                owner_id=self.owner_id,
                subject_type="visual_focus",
                subject_id=focus.id,
                claim_timeout_seconds=120,
            )
        )

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                update jarvis_visual_focuses
                set trigger_claimed_at = now() - interval '5 minutes'
                where owner_id = $1 and id = $2
                """,
                self.owner_id,
                focus.id,
            )
        recovered = await self.repository.try_claim_subject(
            owner_id=self.owner_id,
            subject_type="visual_focus",
            subject_id=focus.id,
            claim_timeout_seconds=120,
        )
        self.assertIsNotNone(recovered)
        self.assertNotEqual(recovered, claim_token)

    async def test_concurrent_first_set_and_replacement_leave_exactly_one_open_focus(self):
        focuses = await asyncio.gather(*(self._create_focus() for _ in range(3)))
        self.assertEqual(len({focus.id for focus in focuses}), 3)
        async with self.pool.acquire() as conn:
            active = await conn.fetch(
                "select id from jarvis_visual_focuses where owner_id = $1 "
                "and completed_at is null and cleared_at is null",
                self.owner_id,
            )
        self.assertEqual(len(active), 1)
        self.assertIn(active[0]["id"], {focus.id for focus in focuses})

    async def test_dispatch_preflight_is_owner_scoped_and_single_use(self):
        focus = await self._create_focus()
        event = await self._reserve(focus)
        foreign, _ = await self.fixtures.create("foreign-disposable-owner")
        self.assertFalse(await self.repository.begin_delivery(
            owner_id=foreign, event_id=event.id, session_id=self.session_id,
        ))
        self.assertTrue(await self.repository.begin_delivery(
            owner_id=self.owner_id, event_id=event.id, session_id=self.session_id,
        ))
        self.assertFalse(await self.repository.begin_delivery(
            owner_id=self.owner_id, event_id=event.id, session_id=self.session_id,
        ))

    async def test_cancelled_queued_event_cannot_start_and_late_outcome_cannot_finish_replacement(self):
        focus = await self._create_focus(mode="find_once")
        event = await self._reserve(focus)
        replacement = await self._create_focus(mode="find_once")
        replacement_claim = await self.repository.try_claim_subject(
            owner_id=self.owner_id, subject_type="visual_focus", subject_id=replacement.id,
            claim_timeout_seconds=120,
        )
        self.assertFalse(await self.repository.begin_delivery(
            owner_id=self.owner_id, event_id=event.id, session_id=self.session_id,
        ))
        await self.repository.resolve_delivery_event(
            owner_id=self.owner_id, event_id=event.id, outcome="delivered",
            provider_response_id="late-old-response",
        )
        active = await self.repository.get_active_visual_focus(owner_id=self.owner_id)
        self.assertEqual(active.id, replacement.id)
        self.assertIsNone(active.completed_at)
        self.assertEqual(active.trigger_claim_token, replacement_claim)

    async def test_unknown_dispatch_cannot_be_replayed_by_lease_expiry(self):
        focus = await self._create_focus()
        event = await self._reserve(focus)
        self.assertTrue(await self.repository.begin_delivery(
            owner_id=self.owner_id, event_id=event.id, session_id=self.session_id,
        ))
        active = await self.repository.get_active_visual_focus(owner_id=self.owner_id)
        self.assertEqual(active.delivery_status["status"], "awaiting_outcome")
        self.assertEqual(active.as_prompt_payload()["delivery_status"]["playback_status"], "unconfirmed")
        async with self.pool.acquire() as conn:
            await conn.execute(
                "update jarvis_visual_focuses set trigger_claimed_at = now() - interval '5 minutes' "
                "where owner_id = $1 and id = $2", self.owner_id, focus.id,
            )
        self.assertIsNone(await self.repository.try_claim_subject(
            owner_id=self.owner_id, subject_type="visual_focus", subject_id=focus.id,
            claim_timeout_seconds=120,
        ))
        await self.repository.resolve_delivery_event(
            owner_id=self.owner_id, event_id=event.id, outcome="interrupted",
            provider_response_id="cancelled-response",
        )
        active = await self.repository.get_active_visual_focus(owner_id=self.owner_id)
        self.assertEqual(active.delivery_status["status"], "interrupted")
        self.assertIsNotNone(await self.repository.try_claim_subject(
            owner_id=self.owner_id, subject_type="visual_focus", subject_id=focus.id,
            claim_timeout_seconds=120,
        ))

    async def test_delivered_find_once_finishes_but_monitor_remains_active(self):
        for mode in ("find_once", "monitor"):
            with self.subTest(mode=mode):
                focus = await self._create_focus(mode=mode)
                event = await self._reserve(focus)
                self.assertTrue(await self.repository.begin_delivery(
                    owner_id=self.owner_id, event_id=event.id, session_id=self.session_id,
                ))
                await self.repository.resolve_delivery_event(
                    owner_id=self.owner_id, event_id=event.id, outcome="delivered",
                    provider_response_id="completed-response",
                    metadata={"reason": "completed_forwarded_answer", "playback_status": "unconfirmed"},
                )
                # A delayed fallback cannot overwrite the authoritative result.
                late = await self.repository.resolve_delivery_event(
                    owner_id=self.owner_id, event_id=event.id, outcome="missed",
                    provider_response_id=None, metadata={"reason": "late-fallback"},
                )
                self.assertEqual(late.event_type, "delivered_to_session")
                self.assertEqual(late.provider_response_id, "completed-response")
                active = await self.repository.get_active_visual_focus(owner_id=self.owner_id)
                if mode == "find_once":
                    self.assertIsNone(active)
                    self.assertIsNone(await self.repository.try_claim_subject(
                        owner_id=self.owner_id, subject_type="visual_focus", subject_id=focus.id,
                        claim_timeout_seconds=120,
                    ))
                else:
                    self.assertEqual(active.id, focus.id)
                    self.assertIsNone(active.trigger_claim_token)

    async def test_expired_focus_cannot_begin_queued_delivery(self):
        focus = await self._create_focus()
        event = await self._reserve(focus)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "update jarvis_visual_focuses set created_at = now() - interval '5 minutes', "
                "expires_at = now() - interval '1 second' where owner_id = $1 and id = $2",
                self.owner_id, focus.id,
            )
        self.assertFalse(await self.repository.begin_delivery(
            owner_id=self.owner_id, event_id=event.id, session_id=self.session_id,
        ))
        self.assertIsNone(await self.repository.get_active_visual_focus(owner_id=self.owner_id))

    async def _reserve(self, focus):
        claim = await self.repository.try_claim_subject(
            owner_id=self.owner_id, subject_type="visual_focus", subject_id=focus.id,
            claim_timeout_seconds=120,
        )
        self.assertIsNotNone(claim)
        event = await self.repository.reserve_delivery_event(
            owner_id=self.owner_id, session_id=self.session_id,
            subject_type="visual_focus", subject_id=focus.id, claim_token=claim,
            occurrence_key=uuid4().hex, message="The target is here.",
            evidence={"supporting_frame_ids": [str(uuid4())]}, dedupe_window_seconds=10,
        )
        self.assertIsNotNone(event)
        return event

    async def _create_focus(self, *, mode="monitor"):
        return await self.repository.set_visual_focus(
            owner_id=self.owner_id,
            target="keys",
            description="Silver keys with a black fob",
            purpose="Tell me where they are",
            mode=mode,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
