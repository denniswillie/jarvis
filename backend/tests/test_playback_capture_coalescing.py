from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import patch
from uuid import uuid4

import pytest

from app.tasks.context_capture import SessionContextCapture


@pytest.mark.asyncio
async def test_pending_progress_coalesces_without_losing_terminal_boundaries():
    entered, release = asyncio.Event(), asyncio.Event()
    events = []

    class Repository:
        async def append(self, owner_id, event):
            entered.set()
            await release.wait()
            events.append(event)

    capture = SessionContextCapture(Repository(), owner_id="fixture", session_id=uuid4(),
        provider="openai", audio_session_id="fixture-audio", max_pending=8)
    capture.expert_speech(response_id="r", item_id="i", text="Synthetic fixture.",
        samples=100000, complete=False, interrupted=False, provenance={})
    await asyncio.wait_for(entered.wait(), 1)
    assert capture.playback_receipt("i", response_id="r", samples=100, state="started")
    for samples in range(1000, 100000, 1000):
        assert capture.playback_receipt("i", response_id="r", samples=samples, state="progress")
    capture.expert_speech(response_id="r", item_id="i", text="Synthetic fixture.",
        samples=100000, complete=True, interrupted=False, provenance={})
    assert capture.playback_receipt("i", response_id="r", samples=100000, state="completed")
    assert capture.diagnostics()["pending_sources"] <= 4
    assert capture.issue is None
    release.set()
    assert await capture.flush()
    boundaries = [event.payload for event in events if event.kind == "played_output_boundary"]
    assert [(event["state"], event["played_samples"]) for event in boundaries] == [
        ("started", 100), ("progress", 99000), ("completed", 100000)]
    assert all(event["heard_transcript"] is None for event in boundaries)
    assert all(event["word_alignment_available"] is False for event in boundaries)
    await capture.close()


@pytest.mark.asyncio
async def test_progress_revision_uses_its_own_receipt_time_and_duplicate_is_noop():
    events = []

    class Repository:
        async def append(self, owner_id, event):
            events.append(event)

    capture = SessionContextCapture(Repository(), owner_id="fixture", session_id=uuid4(),
        provider="openai", audio_session_id="fixture-audio")
    capture.expert_speech(response_id="r", item_id="i", text="Synthetic fixture.",
        samples=1000, complete=False, interrupted=False, provenance={})
    first = datetime(2026, 9, 15, 10, 0, tzinfo=UTC)
    later = datetime(2026, 9, 15, 10, 1, tzinfo=UTC)
    with patch("app.tasks.context_capture.datetime") as clock:
        clock.now.return_value = first
        assert capture.playback_receipt("i", response_id="r", samples=100, state="progress")
        assert await capture.flush()
        clock.now.return_value = later
        assert capture.playback_receipt("i", response_id="r", samples=200, state="progress")
        assert await capture.flush()
        count = len(events)
        assert capture.playback_receipt("i", response_id="r", samples=200, state="progress")
        assert await capture.flush()
    assert len(events) == count
    boundaries = [event for event in events if event.kind == "played_output_boundary"]
    assert [event.source_time for event in boundaries] == [first, later]
    assert len({event.source_event_id for event in boundaries}) == 1
    await capture.close()
