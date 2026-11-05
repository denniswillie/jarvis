"""Bounded MP4 metadata inspection of trusted, privately staged exact bytes.

The caller supplies a provisioned, trusted executable and owns the local file.
This does not establish source permission, playback, full decode, heard audio or
audio/video synchronization. No executable discovery or installation occurs.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal, ROUND_CEILING, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re

from app.files.local_file import VerifiedLocalFile
from app.files.transfer import FileManifest, FileTransferPolicy, MP4Inspection, MP4_CAPTURE_POLICY


MAX_TIMEOUT_SECONDS = 30.0
CLEANUP_TIMEOUT_SECONDS = 2.0
MAX_STDOUT_BYTES = 64 * 1024
MAX_STDERR_BYTES = 16 * 1024
_READ_BYTES = 8192
_MP4_BRANDS = frozenset({"isom", "iso2", "iso3", "iso4", "iso5", "iso6", "mp41", "mp42", "avc1"})
_FORMAT_NAMES = frozenset({"mp4", "mov,mp4,m4a,3gp,3g2,mj2"})
_ENTRIES = ("format=format_name,duration,size:format_tags=major_brand:"
            "stream=index,codec_type,codec_name,width,height,duration:stream_disposition=attached_pic")


class MediaInspectionError(ValueError):
    """Content-free failure: never includes a file path or process output."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


async def _verify_file(local_file: VerifiedLocalFile, policy: FileTransferPolicy) -> FileManifest:
    measured = hashlib.sha256()
    size = 0
    async for block in local_file.iter_chunks():
        size += len(block)
        if size > policy.max_bytes:
            raise MediaInspectionError("media_size_exceeded")
        measured.update(block)
    if size != local_file.size_bytes or measured.hexdigest() != local_file.content_sha256:
        raise MediaInspectionError("media_manifest_mismatch")
    return FileManifest(size_bytes=size, content_sha256=measured.hexdigest())


async def _read_bounded(stream, limit: int) -> bytes:
    content = bytearray()
    while block := await stream.read(min(_READ_BYTES, limit + 1 - len(content))):
        content.extend(block)
        if len(content) > limit:
            raise MediaInspectionError("media_probe_output_exceeded")
    return bytes(content)


async def _discard(stream):
    while await stream.read(_READ_BYTES):
        pass


async def _cleanup(process, readers):
    # ffprobe is invoked directly (no shell or child-producing command). Kill
    # before draining so an overflowing writer cannot keep this loop alive.
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    for task in readers:
        task.cancel()
    await asyncio.gather(*readers, return_exceptions=True)
    try:
        async with asyncio.timeout(CLEANUP_TIMEOUT_SECONDS):
            results = await asyncio.gather(_discard(process.stdout), _discard(process.stderr),
                                           process.wait(), return_exceptions=True)
            # A pipe can be broken already. Still await the process instead of
            # letting that pipe error abandon kill/reap cleanup.
            if isinstance(results[-1], BaseException):
                raise MediaInspectionError("media_probe_cleanup_failed")
    except TimeoutError:
        raise MediaInspectionError("media_probe_cleanup_failed") from None


async def _read_packet_end(stream) -> Decimal:
    """Measure presentation end without retaining a recording's packet list."""
    end, count = Decimal(0), 0
    while line := await stream.readline():
        if not line.strip():
            continue
        count += 1
        fields = line.strip().split(b",")
        if count > 1_000_000 or len(fields) != 2 or any(
                not 1 <= len(value) <= 64 or re.fullmatch(rb"-?[0-9]+(?:\.[0-9]+)?", value) is None
                for value in fields):
            raise MediaInspectionError("media_packet_timing_invalid")
        start, duration = (Decimal(value.decode("ascii")) for value in fields)
        if duration <= 0:
            raise MediaInspectionError("media_packet_timing_invalid")
        with localcontext() as context:
            context.prec = 80
            end = max(end, start + duration)
    if not count or end <= 0:
        raise MediaInspectionError("media_packet_timing_invalid")
    return end


async def _probe(executable: str, local_path: Path, *, packet_timing=False) -> bytes | Decimal:
    # The forced MOV/MP4 demuxer cannot open arbitrary playlist formats. Disable
    # MOV external data references as well as all non-file protocols.
    try:
        process = await asyncio.create_subprocess_exec(
            executable, "-v", "error", "-protocol_whitelist", "file", "-f", "mov",
            "-enable_drefs", "0", "-use_absolute_path", "0", "-show_entries",
            "packet=pts_time,duration_time:packet_side_data=" if packet_timing else _ENTRIES,
            "-of", "csv=p=0" if packet_timing else "json", "-i", str(local_path), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=_READ_BYTES)
    except OSError:
        raise MediaInspectionError("media_probe_start_failed") from None
    readers = [asyncio.create_task(_read_packet_end(process.stdout) if packet_timing
                                  else _read_bounded(process.stdout, MAX_STDOUT_BYTES)),
               asyncio.create_task(_read_bounded(process.stderr, MAX_STDERR_BYTES))]
    waiter = asyncio.create_task(process.wait())
    try:
        stdout, stderr, returncode = await asyncio.gather(*readers, waiter)
        if returncode != 0 or stderr.strip():
            raise MediaInspectionError("media_probe_failed")
        return stdout
    except BaseException as error:
        # An independent, shielded cleanup survives cancellation, including a
        # repeated cancel during cleanup. Its own deadline remains bounded.
        cleanup = asyncio.create_task(_cleanup(process, readers))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        cleanup.result()
        if isinstance(error, Exception) and not isinstance(error, MediaInspectionError):
            raise MediaInspectionError("media_probe_failed") from None
        raise
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise MediaInspectionError("media_probe_invalid_metadata")
        result[key] = value
    return result


def _invalid_constant(_):
    raise MediaInspectionError("media_probe_invalid_metadata")


def _duration(value, maximum_ms: int | None) -> Decimal:
    # ffprobe emits decimal seconds as strings. Bound conversion before Decimal
    # arithmetic; do not accept NaN/Infinity, unavailable or negative durations.
    if (not isinstance(value, str) or not 1 <= len(value) <= 64
            or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value) is None):
        raise MediaInspectionError("media_duration_invalid")
    duration = Decimal(value)
    if not duration.is_finite() or duration <= 0:
        raise MediaInspectionError("media_duration_invalid")
    if maximum_ms is not None and duration > Decimal(maximum_ms) / 1000:
        raise MediaInspectionError("media_duration_exceeded")
    return duration


def _parse_metadata(raw: bytes, manifest: FileManifest, policy: FileTransferPolicy, *,
                    packet_end: Decimal | None = None) -> MP4Inspection:
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise MediaInspectionError("media_probe_invalid_metadata") from None
    if not isinstance(data, dict) or not isinstance(data.get("format"), dict):
        raise MediaInspectionError("media_probe_invalid_metadata")
    format_info = data["format"]
    tags = format_info.get("tags")
    if (not isinstance(format_info.get("format_name"), str)
            or format_info["format_name"] not in _FORMAT_NAMES or not isinstance(tags, dict)
            or not isinstance(tags.get("major_brand"), str) or tags["major_brand"] not in _MP4_BRANDS):
        raise MediaInspectionError("media_container_unsupported")
    size = format_info.get("size")
    if (not isinstance(size, str) or not size.isascii() or not size.isdecimal()
            or len(size) > 10 or int(size) != manifest.size_bytes):
        raise MediaInspectionError("media_manifest_mismatch")
    streams = data.get("streams")
    if not isinstance(streams, list) or not 1 <= len(streams) <= 2:
        raise MediaInspectionError("media_streams_unsupported")
    video, audio, indices = None, None, set()
    metadata_limit = policy.max_encoded_duration_ms if packet_end is None else None
    durations = [_duration(format_info.get("duration"), metadata_limit)]
    for stream in streams:
        if not isinstance(stream, dict):
            raise MediaInspectionError("media_streams_unsupported")
        index = stream.get("index")
        if type(index) is not int or index < 0 or index in indices:
            raise MediaInspectionError("media_streams_unsupported")
        indices.add(index)
        durations.append(_duration(stream.get("duration"), metadata_limit))
        if stream.get("codec_type") == "video" and stream.get("codec_name") == "h264" and video is None:
            disposition = stream.get("disposition")
            if (not isinstance(disposition, dict) or type(disposition.get("attached_pic")) is not int
                    or disposition["attached_pic"] != 0):
                raise MediaInspectionError("media_streams_unsupported")
            video = stream
        elif stream.get("codec_type") == "audio" and stream.get("codec_name") == "aac" and audio is None:
            audio = stream
        else:
            raise MediaInspectionError("media_streams_unsupported")
    if video is None:
        raise MediaInspectionError("media_streams_unsupported")
    if any(type(video.get(key)) is not int or not 1 <= video[key] <= 7680 for key in ("width", "height")):
        raise MediaInspectionError("media_dimensions_invalid")
    with localcontext() as context:
        context.prec = 80  # Preserve every accepted decimal digit before rounding upward.
        measured_end = (max(durations) if packet_end is None else
                        _duration(str(packet_end), policy.max_encoded_duration_ms))
        duration_ms = int((measured_end * 1000).to_integral_value(rounding=ROUND_CEILING))
    inspection = MP4Inspection(content_sha256=manifest.content_sha256, size_bytes=manifest.size_bytes,
        duration_ms=duration_ms, width=video["width"], height=video["height"], video_codec="h264",
        audio_codec="aac" if audio is not None else None)
    inspection.validate_file(manifest, policy)
    return inspection


async def inspect_mp4_file(local_file: VerifiedLocalFile, policy: FileTransferPolicy, *,
                           ffprobe_path: str | Path, timeout: float = 10.0) -> MP4Inspection:
    """Inspect exact staged bytes using trusted local ffprobe, with no retry.

    Timeout (0 < seconds <= 30) covers hashing, subprocess and metadata parsing;
    failure cleanup may additionally take at most two seconds. This first bounded
    profile accepts one H.264 video, zero/one AAC stream, known MP4 brands and
    measured finite duration on every stream. Captures additionally measure the
    last presentation endpoint: fragmented MP4 header estimates can count the
    initial video offset twice. This scan is streamed within the same deadline;
    it does not decode, rewrite media, or widen capture limits. Extra tracks are rejected,
    never silently discarded. Provisioning/version qualification is external.
    """
    if not isinstance(local_file, VerifiedLocalFile) or not isinstance(policy, FileTransferPolicy):
        raise MediaInspectionError("media_inspection_invalid_input")
    if not policy.is_media or not 0 < local_file.size_bytes <= policy.max_bytes:
        raise MediaInspectionError("media_size_exceeded")
    if (type(timeout) not in (int, float) or not math.isfinite(timeout)
            or not 0 < timeout <= MAX_TIMEOUT_SECONDS):
        raise MediaInspectionError("media_timeout_invalid")
    if not isinstance(ffprobe_path, (str, Path)):
        raise MediaInspectionError("media_executable_invalid")
    try:
        executable = Path(ffprobe_path)
        if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
            raise MediaInspectionError("media_executable_invalid")
    except (OSError, ValueError):
        raise MediaInspectionError("media_executable_invalid") from None
    try:
        async with asyncio.timeout(timeout):
            before = await _verify_file(local_file, policy)
            raw = await _probe(str(executable), local_file.path)
            packet_end = (await _probe(str(executable), local_file.path, packet_timing=True)
                          if policy.policy_id == MP4_CAPTURE_POLICY else None)
            after = await _verify_file(local_file, policy)
            if before != after:
                raise MediaInspectionError("media_manifest_mismatch")
            return _parse_metadata(raw, after, policy, packet_end=packet_end)
    except TimeoutError:
        raise MediaInspectionError("media_probe_timeout") from None
