from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from evals.glasses_reliability.reference_probe import aggregate, trial
from test_perception_partial_results import focused_request, handle_payload
from test_fast_perception_contract import _payload


class ReferenceProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_probe_uses_real_analyzer_parser_without_an_extra_model_call(self):
        request = focused_request()
        model = SimpleNamespace(provider="fixture", model="fixture",
                                generate_temporal_json=AsyncMock(return_value=json.dumps(handle_payload(request))))
        result = await trial(model, request, "short_references")
        model.generate_temporal_json.assert_awaited_once()
        self.assertEqual(result["outcome"], "accepted")
        self.assertEqual(result["accepted_matches"], 1)
        self.assertEqual(result["reference_rejections"], 0)
        self.assertIn("frame:0", model.generate_temporal_json.await_args.kwargs["images"][0].label)

    async def test_legacy_rejection_is_not_reported_as_a_delivered_match(self):
        request = focused_request()
        payload = _payload(request)
        payload["prospective_matches"] = [{"subject_type": "visual_focus", "subject_id": "bad", "confidence": .9,
                                           "observed_condition": "visible", "supporting_frame_ids": [str(request.primary_frame.frame_id)],
                                           "suggested_message": "visible"}]
        model = SimpleNamespace(provider="fixture", model="fixture",
                                generate_temporal_json=AsyncMock(return_value=json.dumps(payload)))
        result = await trial(model, request, "legacy_uuid")
        self.assertEqual(result["outcome"], "contract_rejected")
        self.assertEqual(result["accepted_matches"], 0)
        self.assertEqual(result["reference_rejections"], 1)
        self.assertTrue(result["core_usable"])

    async def test_provider_failure_is_neither_zero_reference_error_nor_measured_success(self):
        model = SimpleNamespace(provider="fixture", model="fixture", generate_temporal_json=AsyncMock(side_effect=RuntimeError("SECRET")))
        result = await trial(model, focused_request(), "short_references")
        report = aggregate([result])["short_references"]
        self.assertEqual(report["outcomes"], {"provider_failed": 1})
        self.assertIsNone(report["reference_error_rate"])
        self.assertIsNone(report["median_analysis_ms"])
        self.assertNotIn("SECRET", str(result))

    async def test_legacy_metrics_do_not_accept_handles_that_legacy_prompt_never_supplied(self):
        request = focused_request()
        model = SimpleNamespace(provider="fixture", model="fixture",
                                generate_temporal_json=AsyncMock(return_value=json.dumps(handle_payload(request))))
        result = await trial(model, request, "legacy_uuid")
        self.assertEqual(result["outcome"], "contract_rejected")
        self.assertEqual(result["reference_rejections"], 1)
        self.assertFalse(result["core_usable"])

    async def test_harness_bug_is_not_blame_assigned_to_the_provider(self):
        model = SimpleNamespace(provider="fixture", model="fixture", generate_temporal_json=AsyncMock())
        with patch("evals.glasses_reliability.reference_probe.FastPerceptionAnalyzer", side_effect=RuntimeError("SECRET")):
            result = await trial(model, focused_request(), "short_references")
        self.assertEqual(result["outcome"], "harness_failed")
        self.assertFalse(result["model_returned"])
        self.assertIsNone(aggregate([result])["short_references"]["reference_error_rate"])
        model.generate_temporal_json.assert_not_awaited()
        self.assertNotIn("SECRET", str(result))
