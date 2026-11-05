from __future__ import annotations

import unittest

from app.assistant.base_prompt import build_base_prompt


class AgentMemoryToolSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt = build_base_prompt()

    def test_current_view_routes_to_direct_image_retrieval_when_needed(self) -> None:
        self.assertIn(
            "Request another current view when needed pixels are absent, stale, "
            "of uncertain freshness, or do not reflect a relevant scene change",
            self.prompt,
        )
        self.assertIn(
            "Do not use historical memory as the first choice for an exact current-view request",
            self.prompt,
        )
        self.assertIn("answer directly from those pixels", self.prompt)
        self.assertIn("tool does not supply an analysis", self.prompt)

    def test_sufficient_fresh_images_precede_another_camera_request(self) -> None:
        for policy in (
            "first use adequate fresh images already attached to this response",
            "Assess the images acknowledged in this response's live_view cutoff",
            "their capture times and the supplied freshness and age metadata",
            "When they are fresh, relevant, and sufficiently clear for the request, "
            "answer directly from those pixels",
            "Do not call get_current_camera_view merely because the question is visual or to "
            "reconfirm freshness already established by this response's metadata",
            "Degraded or unsynchronized live_view metadata does not certify a fresh or exact image set",
        ):
            with self.subTest(policy=policy):
                self.assertIn(policy, self.prompt)

    def test_reported_change_checks_whether_attached_images_already_show_it(self) -> None:
        self.assertIn(
            "If Dennis reports a change or adjusts his view, assess whether the images already attached "
            "show the relevant updated state before requesting another image",
            self.prompt,
        )

    def test_historical_recall_routes_to_hierarchical_memory(self) -> None:
        self.assertIn(
            "Use search_memory for historical recall across experiences and durable memory",
            self.prompt,
        )
        self.assertIn(
            "Do not use current-frame capture as a substitute for historical search",
            self.prompt,
        )

    def test_evidence_questions_route_to_explanation_or_direct_historical_pixels(self) -> None:
        self.assertIn("Use explain_memory", self.prompt)
        self.assertIn("use get_memory_image", self.prompt)
        self.assertNotIn("reinspection", self.prompt)
        self.assertIn("textual memory result", self.prompt)
        self.assertIn("Historical images depict their capture time, not the present", self.prompt)
        self.assertIn("do not substitute a current frame", self.prompt)

    def test_explicit_memory_and_forgetting_have_separate_routes(self) -> None:
        self.assertIn("Use remember only when the user explicitly asks", self.prompt)
        self.assertIn("Use forget_memory only for a specific derived memory", self.prompt)

    def test_capture_and_domain_tools_are_not_generic_memory(self) -> None:
        self.assertIn("Use capture_latest_visual_frame only when", self.prompt)
        for capability in (
            "notes",
            "preferences",
            "reminders",
            "calendar",
            "files",
            "behavior instructions",
            "web research",
        ):
            self.assertIn(capability, self.prompt)
        self.assertIn("Use each dedicated tool", self.prompt)


if __name__ == "__main__":
    unittest.main()
