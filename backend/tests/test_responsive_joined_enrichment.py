"""P2 joined slow-history / native-control / canonical-perception acceptance.

The selected route, actual direct-session handlers, ordered provider writer,
Live State factory/enrichment, voice bridge, composer, analyzer and frame
pipeline execute together. Provider/SQL/model endpoints remain controlled.
"""
import asyncio
import json
import unittest

from app.knowledge.models import SourceError
from app.live_state.canonical import context_dependency, memory_dependency
from responsive_joined_enrichment_fixtures import A_MARKER, B_MARKER, JoinedEnrichmentFixture


class JoinedHistoricalVoicePerceptionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = JoinedEnrichmentFixture(self)
        self.addAsyncCleanup(self.f.close)
        await self.f.setup()

    async def reach_bound_B(self):
        f = self.f
        self.assertEqual(f.live.service.snapshot().readiness, "ready")
        self.assertFalse(f.release_a.is_set())
        self.assertTrue(f.route.ready)
        self.assertIs(f.lane.pipeline.live_brief_for(f.scope.owner_id, f.scope.session_id),
                      f.live.perception_brief_bridge)
        self.assertIsNotNone(f.live.perception_brief_bridge.composition)
        self.assertNotIn(A_MARKER, f.brief_text())

        # A is still blocked when a real frame crosses capture/analyzer/storage
        # and a real direct-session turn reaches the ordered provider writer.
        frame_a = await asyncio.wait_for(f.frame(1), 1)
        self.assertFalse(f.lane.harness.visual_repository.failed)
        self.assertEqual(len(f.db.observations), 1)
        self.assertIn(frame_a.id, f.db.frames)
        self.assertTrue(f.db.applied)
        self.assertIsNotNone(f.lane.requests[-1].perception_brief_input_digest)
        self.assertEqual(f.lane.requests[-1].camera_connection_id, f.lease.connection_id)
        await asyncio.wait_for(f.ask("while-A-blocked"), 1)
        await f.until(lambda: bool(f.wire.responses) and bool(f.session._active_responses))
        self.assertFalse(f.release_a.is_set())
        self.assertEqual(len(f.calls), 1)
        self.assertEqual(f.active, 1)
        self.assertIsNotNone(f.session._source_context.source_for(f.wire.responses[0]["event"]["event_id"]))

        await asyncio.wait_for(f.session._handle_client_event({"type": "response.cancel"}), 1)
        await f.until(lambda: any(event["type"] == "response.cancel" for event in f.wire.events))
        await f.until(lambda: any(event["type"] == "voice_route.event"
            and event["payload"].get("operation") == "interrupt_speech"
            and event["payload"].get("status") == "completed" for event in f.client.messages))
        self.assertFalse(f.release_a.is_set())

        # Change both canonical user intent and actually analyzed activity,
        # coalescing them while the old lookup has not returned.
        f.intent_b = f.intent("Help with experiment B")
        await f.until(lambda: f.live.perception_brief_bridge.composition.brief.intent_ref == context_dependency(f.intent_b))
        f.activity = "leaving"
        await asyncio.wait_for(f.frame(2), 1)
        await f.until(lambda: f.live.enrichment._wanted is not None
            and "experiment B" in f.live.enrichment._wanted.query and "leaving" in f.live.enrichment._wanted.query)
        self.assertEqual(len(f.calls), 1)
        f.release_a.set()
        await f.until(lambda: B_MARKER in f.brief_text() and B_MARKER in f.voice_text())
        self.assertEqual(len(f.calls), 2)
        self.assertEqual(f.max_active, 1)
        self.assertEqual(f.live.enrichment.stale_results, 1)
        self.assertIn("experiment B", f.calls[1])
        self.assertIn("leaving", f.calls[1])
        memories = [entry for entry in f.live.service.snapshot().entries if entry.kind == "selected_memory"]
        self.assertEqual(len(memories), 1)
        selected = memories[0]
        self.assertIn(memory_dependency(f.b), selected.sources)
        self.assertIn(context_dependency(f.intent_b), selected.sources)
        self.assertNotIn(memory_dependency(f.a), selected.sources)
        self.assertEqual(selected.observed_at, f.b.source_time)
        self.assertTrue(await f.live.binder.available(selected.guard_binding))
        f.sources.current_snapshot.assert_not_awaited()

        # B is not merely visible in a local snapshot: a subsequent actual
        # model input and recorded observation retain its removable lineage.
        await asyncio.wait_for(f.frame(3), 1)
        self.assertFalse(f.lane.harness.visual_repository.failed)
        self.assertEqual(len(f.db.observations), 3)
        request, receipt = f.lane.requests[-1], f.lane.captured[-1]
        self.assertEqual(request.perception_brief.intent_ref, context_dependency(f.intent_b))
        self.assertIn(memory_dependency(f.b), request.perception_brief.dependencies)
        self.assertIn(B_MARKER, json.dumps(f.lane.model.generate_temporal_json.await_args_list[-1].kwargs, default=str))
        self.assertIn(f.b.payload_id, f.db.ancestors(receipt.payload_id))
        self.assertNotIn(f.a.payload_id, f.db.ancestors(receipt.payload_id))
        observation = next(item for item in f.db.observations.values()
                           if item.primary_visual_frame_id == request.primary_frame.frame_id)
        self.assertIn(f.b.payload_id, f.db.ancestors(f.db.observation_roots[observation.id]))
        async with f.live.perception_runtime.binder.guard(request.perception_brief, request.perception_brief_binding):
            pass

        self.assertNotIn(A_MARKER, f.voice_text())
        self.assertFalse(any(A_MARKER in json.dumps(item.perception_brief.model_dump(mode="json"))
                             for item in f.lane.requests if item.perception_brief))
        self.assertFalse(any(A_MARKER in json.dumps(row["content"], default=str)
            for identity, row in f.db.copies.items() if identity != f.a.payload_id))
        self.assertIsNone(f.attachment.failure)
        self.assertTrue(f.route.ready)
        return request, selected

    async def test_blocked_A_does_not_block_voice_control_frames_and_only_B_is_delivered(self):
        await self.reach_bound_B()

    async def test_withdrawn_B_revokes_actual_voice_and_fixed_perception_evidence(self):
        request, selected = await self.reach_bound_B()
        f = self.f
        before_responses = len(f.wire.responses)
        f.db.copies[f.b.payload_id]["content"] = None
        f.db.originals.pop(f.b.source_id)
        self.assertFalse(await f.live.binder.available(selected.guard_binding))
        with self.assertRaises(SourceError):
            async with f.live.perception_runtime.binder.guard(request.perception_brief, request.perception_brief_binding):
                self.fail("Withdrawn B authorized an existing visual assignment")
        # Do not fabricate a clean replacement remote history. The session's
        # real accumulated-source guard must reject further response creation.
        await f.ask("after-B-withdrawal")
        await f.until(lambda: f.session._context_revoked or f.session._closed.is_set())
        await f.until(lambda: not f.session._direct_tasks)
        self.assertEqual(len(f.wire.responses), before_responses)
        self.assertFalse(f.route.ready)
        self.assertTrue(f.session._context_revoked or f.route.failure is not None)
        self.assertNotIn(A_MARKER, f.voice_text())


if __name__ == "__main__":
    unittest.main()
