"""Offline image receipt identity tests; fake persistence is not SQL proof."""
from dataclasses import replace
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.images import ImageReference, ImageUnavailableError, ResolvedImage
from app.knowledge.images import SourceImageService
from app.knowledge.models import SourceError
from app.tasks.context import TaskContextBuilder
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextRecord
from app.tasks.context_images import TaskContextImages
from test_second_brain_capture import CaptureRepository


def reference(owner, session):
    now = datetime.now(UTC)
    return ImageReference(frame_id=uuid4(), owner_id=owner, session_id=session, source="current",
        captured_at=now, content_type="image/jpeg", width=48, height=32,
        raw_image_expires_at=now + timedelta(minutes=5), visual_session_id=uuid4(), frame_sequence=2,
        stream_connection_id=uuid4(), server_received_at=now, sampled_at=now, sampled_monotonic=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", ["socket_written", "write_outcome_unknown", "not_written"])
async def test_ack_is_additive_immutable_receipt_not_visual_revision(initial):
    repository = CaptureRepository()
    capture = SessionContextCapture(repository, owner_id="fixture", session_id=uuid4(),
                                    provider="openai", audio_session_id="fixture-audio")
    image = reference(capture.owner_id, capture.session_id)
    try:
        capture.image_delivery(image, initial, "exact-delivery")
        assert await capture.flush()
        before = repository.of_kind("observation")[0]
        original = dict(before.content)
        for _ in range(3):
            capture.image_delivery(image, "provider_acknowledged", "exact-delivery")
            assert await capture.flush()
        rows = repository.of_kind("observation")
        assert len(rows) == 2 and len(repository.events) == 2
        assert before.content == original and before.revision == 1
        assert rows[1].revision == 1 and rows[1].source_event_id != before.source_event_id
        assert rows[1].content["images"] == before.content["images"] == [image.to_metadata()]
        assert all(event.source_time == image.captured_at for event in repository.events)
        assert rows[1].content["provider_acknowledged"] is True
        assert before.content["provider_acknowledged"] is False
        assert all(row.content["current_view_verified"] is False for row in rows)
        assert all(row.content["image_view"] == "captured_snapshot" for row in rows)
        assert all(row.content["delivery_state_scope"] == "immutable_receipt" for row in rows)
        assert all(row.content["model_understanding"] == "unknown" for row in rows)
        assert capture.issue is None
    finally:
        await capture.close()


@pytest.mark.asyncio
async def test_queued_write_and_ack_do_not_coalesce_away_the_original_receipt():
    repository = CaptureRepository()
    capture = SessionContextCapture(repository, owner_id="fixture", session_id=uuid4(),
                                    provider="openai", audio_session_id="fixture-audio")
    try:
        image = reference(capture.owner_id, capture.session_id)
        capture.image_delivery(image, "socket_written", "image")
        capture.image_delivery(image, "provider_acknowledged", "image")
        assert await capture.flush()
        assert [r.content["delivery_state"] for r in repository.of_kind("observation")] == [
            "socket_written", "provider_acknowledged"]
    finally:
        await capture.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["owner", "frame", "captured_at"])
async def test_receipt_identity_never_authorizes_reference_substitution(change):
    repository = CaptureRepository()
    capture = SessionContextCapture(repository, owner_id="fixture", session_id=uuid4(),
                                    provider="openai", audio_session_id="fixture-audio")
    try:
        image = reference(capture.owner_id, capture.session_id)
        capture.image_delivery(image, "socket_written", "image")
        assert await capture.flush()
        changed = replace(image, **{"owner_id": "other"} if change == "owner" else
                          {"frame_id": uuid4()} if change == "frame" else
                          {"captured_at": image.captured_at + timedelta(seconds=1)})
        capture.image_delivery(changed, "provider_acknowledged", "image")
        await capture.flush()
        assert capture.issue in {"context_image_evidence_invalid", "context_image_identity_conflict"}
        rows = repository.of_kind("observation")
        assert len(rows) == 1 and rows[0].content["images"] == [image.to_metadata()]
    finally:
        await capture.close()


@pytest.mark.asyncio
async def test_capacity_is_reported_not_silent_overwrite_or_unbounded_growth():
    repository = CaptureRepository()
    capture = SessionContextCapture(repository, owner_id="fixture", session_id=uuid4(),
                                    provider="openai", audio_session_id="fixture-audio", max_pending=2)
    try:
        image = reference(capture.owner_id, capture.session_id)
        capture.image_delivery(image, "socket_written", "image")
        capture.image_delivery(image, "provider_acknowledged", "image")
        assert not await capture.flush()
        assert capture.issue == "context_capture_queue_full"
        assert len(repository.of_kind("observation")) == 1
        assert repository.of_kind("observation")[0].content["delivery_state"] == "socket_written"
        assert len(repository.of_kind("context_gap")) == 1
    finally:
        await capture.close()


@pytest.mark.asyncio
async def test_real_builder_keeps_current_dependency_and_explicit_snapshot_meaning():
    repository = CaptureRepository()
    capture = SessionContextCapture(repository, owner_id="fixture", session_id=uuid4(),
                                    provider="openai", audio_session_id="fixture-audio")
    try:
        image = reference(capture.owner_id, capture.session_id)
        capture.image_delivery(image, "socket_written", "image")
        assert await capture.flush()
        event, row = repository.events[0], repository.of_kind("observation")[0]
        record = ContextRecord(id=row.id, owner_id="fixture", session_id=capture.session_id,
            sequence=1, origin_sequence=1, source_namespace=event.source_namespace,
            source_event_id=event.source_event_id, revision=1, supersedes_event_id=None,
            kind=event.kind, trust=event.trust, finality=event.finality, source_time=event.source_time,
            recorded_at=datetime.now(UTC), payload_id=uuid4(), availability="available",
            content_digest="a" * 64, content=event.payload, is_current=True)
        snapshots = SimpleNamespace(snapshot=AsyncMock(return_value={"pinned": [], "recent": [record],
            "selected_pinned_ids": set(), "pinned_total": 0, "total": 1, "watermark": 1}),
            store_manifest=AsyncMock())
        task = SimpleNamespace(id=uuid4(), owner_id="fixture", session_id=capture.session_id,
            revision=1, title="Read captured evidence", requested_outcome="Reason about the past snapshot",
            request_event_id=uuid4(), request_revision=1)
        await TaskContextBuilder(snapshots).build(SimpleNamespace(task=task))
        args = snapshots.store_manifest.await_args.kwargs
        assert args["records"] == [record] and record.dependency_usage == "current"
        message = args["content"]["messages"][0]
        assert datetime.fromisoformat(message["source_time"]) == image.captured_at
        assert message["content"]["current_view_verified"] is False
        assert "not the current world" in " ".join(args["content"]["evidence_rules"])
        assert not any(key in json.dumps(message) for key in ("image_bytes", "storage_path", "signed_url"))
    finally:
        await capture.close()


@pytest.mark.asyncio
async def test_stored_snapshot_selection_and_queued_permission_revalidation_remain_fenced():
    image = reference("fixture", uuid4())
    record = SimpleNamespace(id=uuid4(), session_id=image.session_id, revision=1, availability="available",
        kind="observation", trust="sensor", content={"images": [image.to_metadata()]})
    claim = SimpleNamespace(task=SimpleNamespace(owner_id="fixture"))
    repository = SimpleNamespace(source_for_claim=AsyncMock(return_value=record))
    permitted = True

    async def check():
        if not permitted:
            raise ImageUnavailableError("frame_unavailable", "Controlled permission withdrawal")

    async def resolve_image(*, owner_id, reference):
        await check()
        return ResolvedImage(reference, b"controlled-pixels", check)

    images = TaskContextImages(repository, stored=SimpleNamespace(resolve_image=resolve_image))
    selection = await images.select(claim, record.id)
    assert selection["image_view"] == "captured_snapshot" and not selection["current_view_verified"]
    assert selection["image"]["source"] == "historical"
    assert "stream_connection_id" not in selection["image"]
    resolved = await images.resolve(claim, selection)
    permitted = False
    with pytest.raises(ImageUnavailableError):
        await resolved.revalidate()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
async def test_source_image_validation_uses_existing_batched_provenance_api(changed):
    area = uuid4()
    connection = SimpleNamespace(fetchval=AsyncMock(return_value="available"),
                                 fetch=AsyncMock(return_value=[{"area_id": area}]))

    @asynccontextmanager
    async def transaction(owner):
        assert owner == "fixture"
        yield connection

    sources = SimpleNamespace(pool=None, transaction=transaction)
    with patch("app.knowledge.images.SavedFileImageService", return_value=None):
        service = SourceImageService(sources, SimpleNamespace(), owner_id="fixture", area_id=area)
    image = replace(reference("fixture", uuid4()), source="historical",
                    record_type="source_evidence", record_id=uuid4())
    parents, originals = [object()], [object()]
    service._parents = AsyncMock(return_value=parents)
    service._direct = AsyncMock(return_value=[service._snapshot(image)])
    service.provenance._resolve_many = AsyncMock(return_value=originals,
        side_effect=SourceError("knowledge_read_evidence_changed", "Controlled stale source") if changed else None)
    assert not hasattr(service.provenance, "_resolve")
    if changed:
        with pytest.raises(SourceError) as error:
            await service._validate("fixture", image)
        assert error.value.code == "knowledge_read_evidence_changed"
        service._direct.assert_not_awaited()
    else:
        await service._validate("fixture", image)
        service._direct.assert_awaited_once_with(connection, originals[0])
    service.provenance._resolve_many.assert_awaited_once_with(connection, "fixture", area, parents)


@pytest.mark.asyncio
async def test_source_image_parent_traversal_uses_existing_batched_provenance_api():
    area, root, dependency = uuid4(), uuid4(), object()
    connection = object()

    @asynccontextmanager
    async def transaction(owner):
        assert owner == "fixture"
        yield connection

    def source():
        identity = uuid4()
        return SimpleNamespace(source_id=identity, revision=1, payload_id=uuid4(),
            original_evidence=lambda: {"source_id": str(identity), "revision": 1, "content": {}})

    first, parent = source(), source()
    sources = SimpleNamespace(pool=None, transaction=transaction, read=AsyncMock(return_value=first))
    stored = SimpleNamespace(timeout_seconds=1, _read=AsyncMock())
    with patch("app.knowledge.images.SavedFileImageService", return_value=None):
        service = SourceImageService(sources, stored, owner_id="fixture", area_id=area)
    image = service._snapshot(reference("fixture", uuid4()))
    service._direct = AsyncMock(side_effect=[[], [image]])
    service._parents = AsyncMock(return_value=[dependency])
    service.provenance._resolve_many = AsyncMock(return_value=[parent])
    service.provenance.bind_context = AsyncMock(return_value=(root,))
    service._validate = AsyncMock()
    selected = await service.select_image(owner_id="fixture", area_id=area, source_id=first.source_id)
    assert selected.frame_id == image.frame_id and selected.record_id == root
    service.provenance._resolve_many.assert_awaited_once_with(connection, "fixture", area, [dependency])
    service._validate.assert_awaited_once_with("fixture", selected)
