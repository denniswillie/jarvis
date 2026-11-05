"""Actual prepared Live append bodies for the five retained 073d observations.

No providers or SQL: existing transport/media fixtures return controlled ACKs.
"""
from datetime import datetime, timedelta
import json
import unittest
from uuid import uuid4

from app.live_state.models import (
    GuardedPayloadBinding, LiveCapabilities, LiveSnapshot, ObservationEntry, SourceDependency,
)
from app.live_state.projection import prepare_voice_context, select_voice_context
from app.voice_routes.live_context_content import encode_live_context_content
import test_live_voice_adapter as fixtures


CASES = json.loads("[{\"summary\":\"A yellow bell pepper is being sliced on a wooden cutting board. A portion of the pepper has already been cut into strips, and a piece remains to the side along with a kitchen knife.\",\"observed_at\":\"2026-10-31T17:30:11.712+00:00\",\"prepared_at\":\"2026-10-31T17:30:18.26506+00:00\",\"revision\":24,\"expected_bytes\":443},{\"summary\":\"A person is slicing a yellow bell pepper on a wooden cutting board. Between frame:0 and frame:3, the person uses their hand to stabilize the pepper strips while holding a knife, then moves their hand away from the strips.\",\"observed_at\":\"2026-10-31T17:30:17.712+00:00\",\"prepared_at\":\"2026-10-31T17:30:23.634469+00:00\",\"revision\":31,\"expected_bytes\":483},{\"summary\":\"The user is slicing a yellow bell pepper on a wooden cutting board. Throughout the sequence, the user's left hand stabilizes the pepper strips while the right hand holds a knife, occasionally adjusting the position of the pepper strips.\",\"observed_at\":\"2026-10-31T17:30:21.712+00:00\",\"prepared_at\":\"2026-10-31T17:30:28.466724+00:00\",\"revision\":38,\"expected_bytes\":498},{\"summary\":\"The user is continuing to slice a yellow bell pepper into strips on a wooden cutting board. The user's left hand stabilizes the pepper while the right hand uses a knife to cut through the vegetable.\",\"observed_at\":\"2026-10-31T17:30:25.712+00:00\",\"prepared_at\":\"2026-10-31T17:30:34.746637+00:00\",\"revision\":43,\"expected_bytes\":460},{\"summary\":\"The user continues to slice a yellow bell pepper into strips on a wooden cutting board.\",\"observed_at\":\"2026-10-31T17:30:33.712+00:00\",\"prepared_at\":\"2026-10-31T17:30:40.319852+00:00\",\"revision\":50,\"expected_bytes\":349}]")


class PreparedLiveObservationWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_actual_observations_fit_after_prepare_and_actual_append(self):
        for case in CASES:
            with self.subTest(revision=case["revision"]):
                fixture = fixtures.LiveFixture()
                try:
                    observed = datetime.fromisoformat(case["observed_at"])
                    prepared = datetime.fromisoformat(case["prepared_at"])
                    fixture.now = prepared
                    source = SourceDependency(source_namespace="context_event", source_id=uuid4(), revision=1)
                    binding = GuardedPayloadBinding(scope=fixture.identity.scope,
                        payload_root_ids=(uuid4(),), dependencies=(source,))
                    entry = ObservationEntry(entry_id="working_observation", scope=fixture.identity.scope,
                        received_at=prepared, observed_at=observed, evidence_kind="observed",
                        fresh_until=observed+timedelta(seconds=10), expires_at=observed+timedelta(seconds=10),
                        sources=(source,), guard_binding=binding, summary=case["summary"])
                    snapshot = LiveSnapshot(scope=fixture.identity.scope,
                        cursor=fixture.cursor.model_copy(update={"revision":case["revision"]}),
                        as_of=prepared, readiness="ready",
                        capabilities=LiveCapabilities(durable_responsibilities="disabled",
                            historical_retrieval="unavailable"), entries=(entry,))
                    selection = select_voice_context(snapshot, now=prepared,
                        max_encoded_bytes=500, preserve_observation_history=True)
                    self.assertEqual(selection.selected_entry_ids, ("working_observation",))
                    bound = []
                    async def bind(selected):
                        bound.append(selected)
                        return binding
                    update = await prepare_voice_context(selection, fixture.identity,
                        bind=bind, clock=lambda:fixture.now)
                    self.assertIsNotNone(update)
                    self.assertEqual(bound, [selection])
                    self.assertEqual(update.text, selection.text)
                    self.assertEqual(update.cursor, snapshot.cursor)
                    self.assertEqual(update.guard_binding, binding)
                    self.assertEqual(update.dependencies, (source,))
                    self.assertEqual(update.observation_history, selection.observation_history)
                    self.assertEqual(len(update.observation_history), 1)
                    self.assertEqual(update.observation_history[0].observed_at, observed)
                    self.assertEqual(update.observation_history[0].fresh_until, entry.fresh_until)
                    self.assertIsNone(update.expires_at)
                    expected = encode_live_context_content(update.text, mode=update.mode)
                    self.assertEqual(fixture.route._content(update), expected)
                    await fixture.start(fixture.start_request(
                        initial_context=update, requested_at=prepared))
                    sent = [json.loads(x) if isinstance(x,str) else x for x in fixture.wire.socket.sent]
                    appends = [x for x in sent if x.get("type")=="session.thinking.append"]
                    self.assertEqual(len(appends), 1)
                    self.assertEqual(appends[0]["content"], expected)
                    document = json.loads(appends[0]["content"])
                    self.assertEqual(set(document), {"kind","text","mode"})
                    self.assertIn(case["summary"], document["text"])
                    size = len(appends[0]["content"].encode("utf-8"))
                    self.assertEqual(size, case["expected_bytes"])
                    self.assertLessEqual(size, 500)
                    self.assertEqual(len(fixture.wire.requests), 1)
                    self.assertFalse(fixture.route.requires_fenced_replacement)
                finally:
                    await fixture.close()


if __name__ == "__main__":
    unittest.main()

