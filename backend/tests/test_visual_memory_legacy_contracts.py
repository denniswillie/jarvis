from __future__ import annotations

import ast
import base64
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from app.realtime.router import router as audio_router
from app.sessions.router import router as session_router
from app.sessions.models import SessionStartResponse
from app.visual_memory.frame_capture_tool import create_capture_latest_visual_frame_tool
from app.visual_memory.router import router as visual_router
from app.visual_memory.validation import (
    VisualFrameValidationError,
    validate_visual_frame_event,
)


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (BACKEND_ROOT / relative_path).read_text(encoding="utf-8")


def _visual_event_names(relative_path: str) -> set[str]:
    tree = ast.parse(_source(relative_path))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("visual.")
    }


def _swift_function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:index]
    raise AssertionError(f"unterminated Swift function: {signature}")


class VisualMemoryIngressContractTests(unittest.TestCase):
    def test_session_start_response_shape_is_frozen(self) -> None:
        properties = SessionStartResponse.model_json_schema()["properties"]
        self.assertEqual(set(properties), {"session", "realtime"})

    def test_visual_frame_capture_shape_is_frozen(self) -> None:
        fixture_path = (
            BACKEND_ROOT
            / "evals/perception_memory/fixtures/images/desk_static.jpg"
        )
        event = {
            "type": "visual.frame.capture",
            "source": "glasses",
            "content_type": "image/jpeg",
            "frame_sequence": 1,
            "captured_at": "2026-08-15T10:00:00Z",
            "image_base64": base64.b64encode(fixture_path.read_bytes()).decode("ascii"),
            "width": 640,
            "height": 360,
        }
        settings = SimpleNamespace(
            VISUAL_FRAME_MAX_BYTES=2_000_000,
            VISUAL_FRAME_MAX_WIDTH=1920,
            VISUAL_FRAME_MAX_HEIGHT=1920,
        )
        validate_visual_frame_event(
            event,
            settings=settings,
            allowed_sources={"glasses"},
        )
        for required_field in (
            "type",
            "source",
            "content_type",
            "frame_sequence",
            "captured_at",
            "image_base64",
        ):
            incomplete = dict(event)
            incomplete.pop(required_field)
            with self.assertRaises(VisualFrameValidationError):
                validate_visual_frame_event(
                    incomplete,
                    settings=settings,
                    allowed_sources={"glasses"},
                )

    def test_realtime_paths_and_auth_inputs_are_frozen(self) -> None:
        self.assertIn("/sessions/start", {route.path for route in session_router.routes})
        self.assertIn("/realtime/audio/ws", {route.path for route in audio_router.routes})
        self.assertIn("/realtime/visual/ws", {route.path for route in visual_router.routes})
        audio_source = _source("app/realtime/router.py")
        visual_source = _source("app/visual_memory/router.py")
        for token in ("authorization", "session_id", "client_kind"):
            self.assertIn(token, audio_source.casefold())
        for token in ("authorization", "session_id", "visual_session_id", "audio_session_id"):
            self.assertIn(token, visual_source.casefold())

    def test_visual_and_interrupt_event_names_are_frozen(self) -> None:
        event_names: set[str] = set()
        for path in (
            "app/visual_memory/router.py",
            "app/visual_memory/pipeline.py",
            "app/realtime/session.py",
        ):
            event_names.update(_visual_event_names(path))
        expected = {
            "visual.session.created",
            "visual.frame.accepted",
            "visual.frame.dropped",
            "visual.frame.analysis_skipped",
            "visual.frame.analyzed",
            "visual.frame.analysis_failed",
            "visual.interrupt.started",
            "visual.interrupt.response_created",
            "visual.interrupt.done",
        }
        self.assertTrue(expected.issubset(event_names))

    def test_capture_latest_visual_frame_contract_is_metadata_only(self) -> None:
        tool = create_capture_latest_visual_frame_tool(
            settings=SimpleNamespace(VISUAL_LATEST_CAPTURE_MAX_AGE_SECONDS=5),
            latest_frame_buffer=SimpleNamespace(),
            visual_repository=SimpleNamespace(),
            file_repository=SimpleNamespace(),
            file_storage=SimpleNamespace(),
            owner_id="owner",
            session_id=uuid4(),
        )
        self.assertEqual(tool.name, "capture_latest_visual_frame")
        self.assertEqual(set(tool.input_schema["properties"]), {"filename", "description"})
        self.assertEqual(tool.input_schema["required"], ["description"])
        self.assertIn("metadata only", tool.description)
        self.assertIn("not image bytes or signed urls", tool.description.casefold())

    def test_sensory_buffering_preserves_storage_and_acceptance_order(self) -> None:
        source = _source("app/visual_memory/router.py")
        storage_available = source.index("mark_frame_available")
        sensory_buffer = source.index("sensory_frame_window.put")
        accepted_event = source.index('"type": "visual.frame.accepted"')
        analyzer_enqueue = source.index("pipeline.enqueue")
        self.assertLess(storage_available, sensory_buffer)
        self.assertLess(sensory_buffer, accepted_event)
        self.assertLess(accepted_event, analyzer_enqueue)

    def test_durable_session_close_clears_sensory_memory(self) -> None:
        source = _source("app/sessions/service.py")
        close_method = source[source.index("async def close_session"):]
        durable_close = close_method.index("self._repository.close_session")
        sensory_clear = close_method.index("self._sensory_frame_window.clear")
        response = close_method.index("return SessionDetailResponse")
        self.assertLess(durable_close, sensory_clear)
        self.assertLess(sensory_clear, response)

    def test_ios_primary_control_delegates_startup_and_coordinates_shutdown(self) -> None:
        source = (
            BACKEND_ROOT.parent
            / "ios/Jarvis/Jarvis/ViewModels/JarvisAudioSessionViewModel.swift"
        ).read_text(encoding="utf-8")
        start = _swift_function_body(source, "func startJarvis()")
        session_start = _swift_function_body(source, "func startSession()")
        stop = _swift_function_body(source, "func stopJarvis()")
        self.assertLess(start.index("isUnifiedJarvisSession = true"), start.index("startSession()"))
        self.assertNotIn("startLocalVisualStream()", start)
        self.assertLess(session_start.index("await apiClient.startSession("),
                        session_start.index("activeSession = response.session"))
        self.assertLess(session_start.index("activeSession = response.session"),
                        session_start.index("realtimeClient.connect("))
        self.assertLess(stop.index("isUnifiedJarvisSession = false"), stop.index("stopLocalVisualStream()"))
        self.assertLess(stop.index("stopLocalVisualStream()"), stop.index("stopSession()"))

    def test_reminder_lifecycle_remains_available(self) -> None:
        reminder_models = _source("app/reminders/models.py")
        reminder_scheduler = _source("app/reminders/scheduler.py")
        for token in ("status", "remind_at", "trigger"):
            self.assertIn(token, (reminder_models + reminder_scheduler).casefold())


if __name__ == "__main__":
    unittest.main()
