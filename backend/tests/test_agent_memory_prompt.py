from __future__ import annotations

import unittest

from app.assistant.base_prompt import build_base_prompt
from app.realtime.instructions import (
    active_prompt_context_payload,
    build_realtime_instructions,
)


class _LegacyCompatibleContext:
    def as_prompt_payload(self) -> dict[str, object]:
        return {
            "current_utc_time": "2026-08-28T12:00:00+00:00",
            "working_memory": {
                "available": True,
                "session": {"revision": 7, "state": {"activity": "walking"}},
            },
            "recent_visual_context": [
                {"description": "Legacy individual frame caption."}
            ],
            "recent_visual_interrupts": [],
        }


class AgentMemoryPromptTests(unittest.TestCase):
    def test_base_prompt_uses_structured_memory_policy(self) -> None:
        prompt = build_base_prompt()

        self.assertIn("structured current-state estimate", prompt)
        self.assertIn("does not give you direct access to frame pixels", prompt)
        self.assertIn("get_current_camera_view", prompt)
        self.assertNotIn("get_current_image", prompt)
        self.assertIn("Automatic image delivery and on-request camera access are separate", prompt)
        self.assertIn("not that the camera is off or that a frame cannot be requested", prompt)
        self.assertIn("before saying you cannot see", prompt)
        self.assertIn("Let the tool report when the camera or a fresh image is unavailable", prompt)
        self.assertIn(
            "Dennis may refer to material in front of him without mentioning the camera",
            prompt,
        )
        self.assertIn("Past-tense recall takes precedence over this camera default", prompt)
        self.assertIn("instead of asking him to paste or describe the material", prompt)
        self.assertIn(
            "the absence of pasted text does not mean that material is missing from attached images",
            prompt,
        )
        self.assertIn("An earlier turn's image_status attached is not proof", prompt)
        self.assertIn("call get_current_camera_view again before answering", prompt)
        self.assertNotIn("get_current_visual_context", prompt)
        self.assertIn("search_memory", prompt)
        self.assertIn("get_memory_image", prompt)
        self.assertNotIn("reinspection", prompt)
        self.assertIn("explain_memory", prompt)
        self.assertIn("remember", prompt)
        self.assertIn("forget_memory", prompt)
        self.assertIn("capture_latest_visual_frame", prompt)
        self.assertNotIn("search_visual_memory", prompt)

    def test_situated_guidance_preserves_tool_boundaries_and_permissions(self) -> None:
        prompt = build_base_prompt()

        self.assertIn("Call it with no arguments", prompt)
        self.assertIn("Interpret the attached image yourself", prompt)
        self.assertIn("selected alone does not mean the pixels reached you", prompt)
        self.assertIn("Never claim to see details from a textual summary alone", prompt)
        self.assertIn("Do not use historical memory as the first choice", prompt)
        self.assertIn("Use remember only when the user explicitly asks", prompt)
        self.assertIn("it does not inspect the image or search the past", prompt)
        self.assertIn("Use each dedicated tool", prompt)
        self.assertIn(
            "may create, update, or delete Jarvis-owned and connected data without asking for confirmation",
            prompt,
        )

    def test_active_context_strips_legacy_caption_list_without_mutating_source(self) -> None:
        context = _LegacyCompatibleContext()
        original = context.as_prompt_payload()

        active = active_prompt_context_payload(context)  # type: ignore[arg-type]

        self.assertNotIn("recent_visual_context", active)
        self.assertIn("recent_visual_context", original)
        self.assertEqual(active["working_memory"], original["working_memory"])

    def test_realtime_instructions_include_working_memory_not_frame_captions(self) -> None:
        instructions = build_realtime_instructions(  # type: ignore[arg-type]
            context=_LegacyCompatibleContext()
        )

        self.assertIn('"working_memory"', instructions)
        self.assertIn('"revision": 7', instructions)
        self.assertIn('"current_utc_time": "2026-08-28T12:00:00+00:00"', instructions)
        self.assertIn("next useful action", instructions)
        self.assertIn("does not give you direct access to frame pixels", instructions)
        self.assertNotIn("recent_visual_context", instructions)
        self.assertNotIn("Legacy individual frame caption", instructions)


if __name__ == "__main__":
    unittest.main()
