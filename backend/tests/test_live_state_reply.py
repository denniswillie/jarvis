"""Representative offline state, not a reconstruction of an unrecorded reply."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import unittest
from uuid import UUID

from app.realtime.live_state_reply import compact_live_state_reply, framed_reply_bytes
from app.voice_routes.live_results import LiveDelegationResult


AT = "2026-11-04T04:26:36.779686+00:00"
SOURCE = "c9510a4f-e34b-47df-a94e-900b06d9fc65"


def fact(text="The card in the current camera view shows 47.", at=AT):
    return {"description": text, "evidence_at": at, "basis": "observed", "sources": [SOURCE, SOURCE]}


def current_view_fixture():
    return {
        "ready": True, "as_of": "2026-11-04T04:26:42.000000+00:00", "revision": 29, "human_generation": 0,
        "human": {"revision": 3, "last_observed_at": AT, "generated_at": AT, "stale": False,
            "source_document_ids": [], "content": {"setting": None, "activities": [],
                "stated_intentions": [], "relevant_details": [fact()], "uncertainties": [], "availability": "open"}},
        "system": {
            "reminders": [{"id": str(UUID(int=index + 1)), "owner_id": "fixture-owner",
                "title": "Bring the umbrella when leaving", "status": "active", "remind_at": None,
                "condition": "When I am about to leave, remind me to bring the umbrella before going outside.",
                "repeat_until_confirmed": False, "created_at": AT, "updated_at": AT,
                "metadata": {"source_document_ids": ["jarvis-live-caption-" + str(index) * 64]}}
                for index in range(3)],
            "sandboxes": {"ready": True, "has_more": False, "next_cursor": None,
                "items": [{"allocation_id": str(UUID(int=50 + index)), "generation": 2,
                    "title": "Retained media workspace", "state": "active", "stale": False,
                    "expires_at": "2026-11-04T05:00:00+00:00", "can_view": True, "can_resume_and_view": False}
                    for index in range(2)]},
            "app": None,
            "work": {"master_session_id": "jarvis-" + "8c0a" * 16,
                "last_admission": {"admission_id": str(UUID(int=99)), "status": "accepted"}, "last_issue": None},
        },
    }


def failing_current_view_db_fixture():
    """Captured current-view03 DB human fields; RAM system state was not captured."""
    caption = "jarvis-live-caption-2e7a262444cbd06ae3f87ad295b2391d7e1af51f80b31773b0e64a67ad52a965"
    return {
        "as_of": "2026-11-04T04:37:06.086543+00:00", "revision": 27, "human_generation": 0,
        "human": {
            "content": {
                "setting": None,
                "activities": [{"basis": "user_report", "status": "active",
                    "sources": ["dialogue:" + caption + ":15"], "description": "User is at their desk",
                    "evidence_at": "2026-11-04T04:27:46.568628Z"}],
                "availability": "do_not_disturb", "uncertainties": [],
                "relevant_details": [{"basis": "observed", "sources": ["frame:" + SOURCE],
                    "description": "The number on the card in front of the user is 47",
                    "evidence_at": "2026-11-04T04:26:36.779686Z"}],
                "stated_intentions": [{"basis": "user_report", "sources": ["dialogue:" + caption + ":10"],
                    "description": "User wants to be reminded to sit up straight when they say they are at their desk",
                    "evidence_at": "2026-11-04T04:27:33.167800Z"}],
            },
            "revision": 15, "generated_at": "2026-11-04T04:28:24.044017+00:00",
            "last_observed_at": "2026-11-04T04:28:21.370355+00:00", "stale": True,
            "source_document_ids": [caption,
                "jarvis-observation-9fa2738f684f1fc1b2da731878ddfe2bc8164a6b77499eb75f769c43a0f8859f"],
        },
        "system": None,
    }


def encoded_result(value):
    text = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    result = LiveDelegationResult(None, "fixture-delegation", None, 0, UUID(int=1), text,
        datetime(2027, 1, 1, tzinfo=timezone.utc), (), kind="delegation_result")
    frames = result.chunks(lambda encoded: len(encoded.encode("utf-8")))
    return {"json_bytes": len(text.encode("utf-8")), "framed_full_bytes": framed_reply_bytes(value),
            "packets": len(frames), "packet_bytes": [len(frame.encode("utf-8")) for frame in frames],
            "wire_content_bytes": sum(len(frame.encode("utf-8")) for frame in frames)}, frames


class LiveStateReplyTests(unittest.TestCase):
    def test_human_selection_preserves_fact_time_freshness_and_revisions_without_mutation(self):
        state = current_view_fixture()
        original = deepcopy(state)
        reply = compact_live_state_reply(state, "human")
        self.assertEqual(state, original)
        self.assertNotIn("system", reply)
        self.assertEqual(reply["human"]["content"]["relevant_details"], [dict(fact(), sources=[SOURCE])])
        for key in ("revision", "last_observed_at", "generated_at", "stale"):
            self.assertEqual(reply["human"][key], state["human"][key])
        self.assertEqual(reply["human_generation"], 0)
        self.assertNotIn("availability", reply["human"]["content"])
        self.assertNotIn("source_document_ids", reply["human"])

    def test_system_selection_keeps_exact_action_ids_and_false_flags(self):
        state = current_view_fixture()
        reply = compact_live_state_reply(state, "system", max_framed_bytes=12000)
        self.assertNotIn("human", reply)
        for before, after in zip(state["system"]["reminders"], reply["system"]["reminders"]):
            for key in ("id", "condition", "updated_at", "repeat_until_confirmed"):
                self.assertEqual(after[key], before[key])
            self.assertNotIn("owner_id", after)
            self.assertNotIn("created_at", after)
        self.assertEqual(reply["system"]["sandboxes"]["items"], state["system"]["sandboxes"]["items"])
        self.assertEqual(reply["system"]["work"]["master_session_id"], state["system"]["work"]["master_session_id"])
        self.assertNotIn("quiet_mode", reply["system"])
        self.assertIsNone(reply["system"]["app"])

    def test_budget_is_advisory_and_keeps_old_and_new_facts_in_order(self):
        state = current_view_fixture()
        old = fact("An earlier observation. " * 20, "2026-11-03T00:00:00+00:00")
        state["human"]["content"]["relevant_details"].insert(0, old)
        reply = compact_live_state_reply(state, "human")
        self.assertNotIn("omitted", reply)
        self.assertTrue(reply["over_budget"])
        self.assertEqual(reply["human"]["content"]["relevant_details"],
                         [dict(old, sources=[SOURCE]), dict(fact(), sources=[SOURCE])])
        self.assertEqual(state["human"]["content"]["relevant_details"][0], old)

    def test_qualifying_uncertainty_and_provenance_are_never_silently_removed(self):
        state = current_view_fixture()
        uncertainty = fact("The view is partially obstructed; do not treat the reading as certain. " * 8)
        state["human"]["content"]["uncertainties"] = [uncertainty]
        state["human"]["source_document_ids"] = ["exact-document", "exact-document"]
        reply = compact_live_state_reply(state, "human", max_framed_bytes=500)
        self.assertTrue(reply["over_budget"])
        self.assertEqual(reply["human"]["content"]["uncertainties"][0]["description"], uncertainty["description"])
        self.assertEqual(reply["human"]["source_document_ids"], ["exact-document"])
        self.assertNotIn("omitted", reply)
        self.assertEqual(reply["human"]["content"]["relevant_details"], [dict(fact(), sources=[SOURCE])])

    def test_default_all_keeps_every_actionable_id_despite_small_target(self):
        state = current_view_fixture()
        original = deepcopy(state)
        reply = compact_live_state_reply(state, max_framed_bytes=500)
        self.assertTrue(reply["over_budget"])
        self.assertNotIn("omitted", reply)
        self.assertEqual(state, original)
        self.assertEqual(reply["system"]["sandboxes"]["items"], state["system"]["sandboxes"]["items"])
        for before, after in zip(state["system"]["reminders"], reply["system"]["reminders"]):
            for key in ("id", "condition", "updated_at", "repeat_until_confirmed"):
                self.assertEqual(after[key], before[key])
        self.assertEqual(len(reply["system"]["reminders"]), len(state["system"]["reminders"]))
        self.assertEqual(reply["system"]["work"]["master_session_id"], state["system"]["work"]["master_session_id"])
        self.assertEqual(reply["system"]["work"]["last_admission"], state["system"]["work"]["last_admission"])

    def test_actual_failing_db_shape_retains_47_all_facts_and_stale_evidence(self):
        state = failing_current_view_db_fixture()
        original = deepcopy(state)
        measurements = {}
        for section in ("human", "all"):
            with self.subTest(section=section):
                reply = compact_live_state_reply(state, section)
                self.assertTrue(reply["over_budget"])
                self.assertNotIn("omitted", reply)
                for field in ("activities", "stated_intentions", "relevant_details", "availability"):
                    self.assertEqual(reply["human"]["content"][field], state["human"]["content"][field])
                for field in ("revision", "last_observed_at", "generated_at", "source_document_ids", "stale"):
                    self.assertEqual(reply["human"][field], state["human"][field])
                measured, frames = encoded_result(reply)
                self.assertLessEqual(measured["packets"], 4)
                self.assertGreaterEqual(measured["packets"], 3)
                self.assertTrue(all(size <= 500 for size in measured["packet_bytes"]))
                self.assertEqual(json.loads("".join(json.loads(frame)["text"] for frame in frames)), reply)
                measurements[section] = measured
        self.assertEqual(state, original)
        print("CURRENT_VIEW03_DB_REGRESSION " + json.dumps(measurements, sort_keys=True))

    def test_hard_transport_limit_still_rejects_without_semantic_eviction(self):
        state = current_view_fixture()
        state["human"]["content"]["relevant_details"] = [fact("x" * 600) for _ in range(12)]
        state["human"]["content"]["activities"] = [dict(fact("y" * 600), status="active") for _ in range(8)]
        reply = compact_live_state_reply(state, "human")
        self.assertEqual(len(reply["human"]["content"]["relevant_details"]), 12)
        self.assertEqual(len(reply["human"]["content"]["activities"]), 8)
        with self.assertRaisesRegex(ValueError, "live_result_size_unavailable"):
            encoded_result(reply)

    def test_absent_section_remains_unavailable_not_fabricated_empty(self):
        reply = compact_live_state_reply({"human": None, "system": None, "ready": False})
        self.assertEqual(reply, {"human": None, "system": None, "ready": False})

    def test_legacy_all_is_lossless_but_section_selection_is_explicitly_unsupported(self):
        legacy = {"text": "Earlier source-bound state", "expires_at": AT}
        self.assertEqual(compact_live_state_reply(legacy), legacy)
        for section in ("human", "system", "unknown", None):
            with self.subTest(section=section), self.assertRaises(ValueError):
                compact_live_state_reply(legacy, section)

    def test_actual_packer_before_after_and_lossless_unicode_framing(self):
        state = current_view_fixture()
        before, _ = encoded_result(state)
        reply = compact_live_state_reply(state, "human")
        after, frames = encoded_result(reply)
        self.assertGreater(before["packets"], after["packets"])
        self.assertLessEqual(after["packets"], 2)
        self.assertNotIn("omitted", reply)
        self.assertTrue(all(size <= 500 for size in after["packet_bytes"]))
        self.assertEqual(json.loads("".join(json.loads(frame)["text"] for frame in frames)), reply)
        value = {"text": "Quoted \"reading\": \u00e9 \U0001f441"}
        measured, frames = encoded_result(value)
        self.assertEqual(measured["framed_full_bytes"], len(frames[0].encode("utf-8")) + 1)
        print("CURRENT_VIEW_REPRESENTATIVE " + json.dumps({"before": before, "after_human": after}, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
