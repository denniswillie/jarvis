"""Authenticated recording ingress, mounted only by trusted runtime.

The direct demo route signs one target and adopts verified cloud bytes. The
legacy raw-MP4 route retains its existing original/reconciliation contract.
"""
import asyncio
import json
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError
from app.evidence.models import FrozenRecord
from app.files.local_file import LocalFileError
from app.files.recording_ingress import RecordingIngress, RecordingUpload


router = APIRouter(prefix="/recordings", tags=["recordings"])
Principal = Annotated[JarvisPrincipal, Depends(require_static_bearer)]
HEADERS = {"Cache-Control": "private, no-store", "Vary": "Authorization"}
MAX_DESCRIPTOR_BYTES = 8192
UPLOAD_SECONDS = 120
# Phone metadata calls have a 120s deadline; retain bounded verification.
FINALIZE_SECONDS = 80


class RecordingFinalizeUpload(FrozenRecord):
    upload_id: UUID
    descriptor: RecordingUpload


def _ingress(request, owner_id):
    services = getattr(request.app.state, "runtime_services", None)
    ingress = getattr(services, "recording_ingress", None)
    if not isinstance(ingress, RecordingIngress):
        raise JarvisHTTPError(503, "recording_ingress_unavailable", "Recording retention is not configured.")
    ingress.admissions._authorize(owner_id)
    return ingress


def _descriptor(request):
    text = request.headers.get("x-jarvis-recording-manifest", "")
    try:
        encoded = text.encode("ascii")
        if not 1 <= len(encoded) <= MAX_DESCRIPTOR_BYTES:
            raise ValueError()
        upload = RecordingUpload.model_validate(json.loads(text))
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "video/mp4":
            raise ValueError()
        length = request.headers.get("content-length")
        if length is not None and (not length.isascii() or not length.isdecimal() or int(length) != upload.size_bytes):
            raise ValueError()
        return upload
    except (ValueError, TypeError, ValidationError):
        raise JarvisHTTPError(422, "recording_manifest_invalid",
            "An ASCII JSON recording descriptor up to8192bytes and matching raw MP4 body are required.") from None


async def _finalization_descriptor(request):
    try:
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise ValueError()
        data = bytearray()
        async for block in request.stream():
            if len(data) + len(block) > MAX_DESCRIPTOR_BYTES + 256:
                raise ValueError()
            data.extend(block)
        return RecordingFinalizeUpload.model_validate_json(bytes(data))
    except (ValueError, TypeError, ValidationError):
        raise JarvisHTTPError(422, "recording_manifest_invalid",
            "A bounded JSON upload identity and recording descriptor are required.") from None


@router.post("/{recording_id}/upload-target")
async def recording_upload_target(recording_id: UUID, session_id: UUID, request: Request, principal: Principal):
    result = await _ingress(request, principal.owner_id).create_upload_target(
        owner_id=principal.owner_id, session_id=session_id, recording_id=recording_id)
    return JSONResponse(result, headers=HEADERS)


@router.post("/{recording_id}/finalize-upload")
async def finalize_recording_upload(recording_id: UUID, session_id: UUID, request: Request, principal: Principal):
    ingress = _ingress(request, principal.owner_id)
    try:
        async with asyncio.timeout(FINALIZE_SECONDS):
            body = await _finalization_descriptor(request)
            result = await ingress.finalize_upload(owner_id=principal.owner_id, session_id=session_id,
                recording_id=recording_id, upload_id=body.upload_id, upload=body.descriptor)
    except TimeoutError:
        raise JarvisHTTPError(408, "recording_finalization_timeout",
            "The final cloud recording could not be verified in time; no new saved receipt is confirmed.") from None
    except LocalFileError as error:
        raise JarvisHTTPError(422, error.code, "The cloud recording does not match its exact byte manifest.") from None
    return JSONResponse(result, status_code=200 if result["outcome"] == "saved" else 202, headers=HEADERS)


@router.put("/{recording_id}/content")
async def upload_recording(recording_id: UUID, session_id: UUID, request: Request, principal: Principal):
    ingress = _ingress(request, principal.owner_id)
    upload = _descriptor(request)

    try:
        # Bound the whole attempt, without leaving a timeout context suspended
        # across generator yields. Existing publication tracks uncertain effects;
        # the private staging sink removes incomplete bytes on cancellation.
        async with asyncio.timeout(UPLOAD_SECONDS):
            result = await ingress.finalize(owner_id=principal.owner_id, session_id=session_id,
                recording_id=recording_id, upload=upload, chunks=request.stream())
    except TimeoutError:
        raise JarvisHTTPError(408, "recording_transfer_timeout",
            "Retention did not finish in time; reconcile this recording before retrying and retain its original.") from None
    except LocalFileError as error:
        raise JarvisHTTPError(422, error.code, "The uploaded recording does not match its exact byte manifest.") from None
    return JSONResponse(result, status_code=200 if result["outcome"] == "saved" else 202, headers=HEADERS)


@router.post("/{recording_id}/reconcile")
async def reconcile_recording(recording_id: UUID, session_id: UUID, request: Request, principal: Principal):
    result = await _ingress(request, principal.owner_id).reconcile(owner_id=principal.owner_id,
        session_id=session_id, recording_id=recording_id)
    return JSONResponse(result, status_code=200 if result["outcome"] == "saved" else 202, headers=HEADERS)
