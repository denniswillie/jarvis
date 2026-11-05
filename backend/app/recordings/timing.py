"""Optional saved-file picture delay; never owns capture, saving, or publication.

The caller retains its original VerifiedLocalFile. Only a successful correction
returns another owned snapshot. Offsets are calibrated configuration, not an
inference from arbitrary footage. This component is not enabled by importing it.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import tempfile

from app.files.local_file import VerifiedLocalFile, snapshot_file
from app.files.transfer import CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS


@dataclass(frozen=True)
class RecordingTimingOptions:
    video_delay_ms: int = 0
    max_bytes: int = CAPTURE_MAX_BYTES
    max_duration_ms: int = CAPTURE_MAX_DURATION_MS
    timeout_seconds: float = 120.0
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"

    def __post_init__(self):
        for value, maximum in ((self.video_delay_ms, 2000),
                               (self.max_bytes, CAPTURE_MAX_BYTES),
                               (self.max_duration_ms, CAPTURE_MAX_DURATION_MS)):
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError("Invalid recording timing bound")
        if self.max_bytes == 0 or self.max_duration_ms == 0:
            raise ValueError("Recording timing limits must be positive")
        if (type(self.timeout_seconds) not in (int, float)
                or not math.isfinite(self.timeout_seconds)
                or not 0 < self.timeout_seconds <= 600):
            raise ValueError("A finite bounded correction timeout is required")
        if any(type(p) is not str or not p or "\x00" in p for p in (self.ffmpeg, self.ffprobe)):
            raise ValueError("Explicit executable names or paths are required")


@dataclass(frozen=True)
class RecordingTimingResult:
    status: str
    reason: str
    corrected: VerifiedLocalFile | None = None
    metadata: dict | None = None


class _CorrectionError(Exception):
    pass


async def _run(arguments: list[str]) -> bytes:
    # No shell, inherited stdin, remote input protocols, or unbounded stderr.
    process = await asyncio.create_subprocess_exec(
        *arguments, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        output, _ = await process.communicate()
    except BaseException:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
        raise
    if process.returncode != 0:
        raise _CorrectionError("media_process_failed")
    # Commands emit selected metadata, a short frame-timestamp tail, or a hash.
    if len(output) > 64 * 1024:
        raise _CorrectionError("media_metadata_too_large")
    return output


async def _probe(path: Path, options: RecordingTimingOptions) -> dict:
    return json.loads(await _run([
        options.ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe",
        "-count_packets", "-show_entries",
        "stream=codec_type,codec_name,width,height,r_frame_rate,time_base,"
        "start_time,duration,sample_rate,channels,nb_read_packets:format=duration",
        "-of", "json", str(path)]))


def _tracks(data: dict) -> tuple[dict, dict]:
    streams = data.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if (len(streams) != 2 or len(video) != 1 or len(audio) != 1
            or video[0].get("codec_name") != "h264"
            or audio[0].get("codec_name") != "aac"):
        raise _CorrectionError("unsupported_recording_tracks")
    return video[0], audio[0]


def _seconds(value) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise _CorrectionError("invalid_media_time")
    return result


async def _audio_hash(path: Path, options: RecordingTimingOptions) -> str:
    value = (await _run([
        options.ffmpeg, "-v", "error", "-xerror", "-nostdin",
        "-protocol_whitelist", "file,pipe", "-i", str(path),
        "-map", "0:a:0", "-c:a", "copy", "-f", "hash", "-hash", "sha256", "-"])).strip()
    if not re.fullmatch(rb"SHA256=[0-9a-f]{64}", value):
        raise _CorrectionError("audio_hash_unavailable")
    return value.decode().split("=", 1)[1]


async def _last_video_timestamp(
    path: Path, end_hint: float, options: RecordingTimingOptions,
) -> float:
    # Header duration is only a seek hint, not evidence of the last frame.
    # ffprobe seeks to a preceding keyframe and decodes the remaining tail.
    data = json.loads(await _run([
        options.ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe",
        "-select_streams", "v:0", "-read_intervals", f"{max(0, end_hint - 5):.6f}%",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)]))
    frames = data.get("frames", [])
    if not frames:
        raise _CorrectionError("video_frame_timestamp_unavailable")
    return max(_seconds(frame["best_effort_timestamp_time"]) for frame in frames)


async def correct_recording_timing(
    original: VerifiedLocalFile, options: RecordingTimingOptions,
) -> RecordingTimingResult:
    """Return a separate corrected snapshot or a reason to keep the original.

    A successful caller owns ``result.corrected.close()``. This function never
    closes/deletes the original, publishes an object, or edits a save receipt.
    Cancellation propagates after terminating the child process and cleaning up.
    """
    if not isinstance(original, VerifiedLocalFile) or not isinstance(options, RecordingTimingOptions):
        raise TypeError("A verified original and timing options are required")
    if options.video_delay_ms == 0:
        return RecordingTimingResult("disabled", "zero_offset")
    if original.size_bytes > options.max_bytes:
        return RecordingTimingResult("skipped", "input_size_limit")
    corrected = None
    try:
        original._check()
        async with asyncio.timeout(options.timeout_seconds):
            before = await _probe(original.path, options)
            video, audio = _tracks(before)
            delay = options.video_delay_ms / 1000
            duration = _seconds(before["format"]["duration"])
            video_start = _seconds(video["start_time"])
            audio_start = _seconds(audio["start_time"])
            rate = Fraction(video["r_frame_rate"])
            time_base = Fraction(video["time_base"])
            if (not 1 <= rate <= 120 or time_base <= 0
                    or time_base.numerator != 1 or time_base.denominator > 1_000_000):
                raise _CorrectionError("unsupported_video_clock")
            if int(video["width"]) * int(video["height"]) > 3840 * 2160:
                return RecordingTimingResult("skipped", "video_dimensions_limit")
            if duration * 1000 + options.video_delay_ms > options.max_duration_ms:
                return RecordingTimingResult("skipped", "corrected_duration_limit")
            original_audio_hash = await _audio_hash(original.path, options)
            with tempfile.TemporaryDirectory(prefix="jarvis-recording-timing-") as temporary:
                output = Path(temporary).resolve() / "corrected.mp4"
                graph = (
                    "[0:v:0]split=2[first][main];"
                    "[first]trim=end_frame=1,setpts=0[hold];"
                    f"[main]setpts=PTS+{options.video_delay_ms}/(1000*TB)[shifted];"
                    "[hold][shifted]interleave=nb_inputs=2:duration=longest[v]")
                await _run([
                    options.ffmpeg, "-v", "error", "-xerror", "-nostdin", "-n",
                    "-copyts", "-protocol_whitelist", "file,pipe", "-i", str(original.path),
                    "-filter_complex", graph, "-map", "[v]", "-map", "0:a:0",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-bf", "0",
                    "-pix_fmt", "yuv420p", "-fps_mode:v", "passthrough",
                    "-enc_time_base:v", str(time_base), "-c:a", "copy",
                    "-video_track_timescale", str(time_base.denominator),
                    "-movie_timescale", "48000", "-avoid_negative_ts", "disabled",
                    "-movflags", "+faststart", "-fs", str(options.max_bytes), str(output)])
                if not 0 < output.stat().st_size <= options.max_bytes:
                    raise _CorrectionError("output_size_limit")
                after = await _probe(output, options)
                new_video, new_audio = _tracks(after)
                after_duration = _seconds(after["format"]["duration"])
                if after_duration * 1000 > options.max_duration_ms:
                    raise _CorrectionError("output_duration_limit")
                if (int(new_video["nb_read_packets"]) != int(video["nb_read_packets"]) + 1
                        or int(new_audio["nb_read_packets"]) != int(audio["nb_read_packets"])
                        or (new_video["width"], new_video["height"]) != (video["width"], video["height"])
                        or (new_audio["sample_rate"], new_audio["channels"]) != (audio["sample_rate"], audio["channels"])):
                    raise _CorrectionError("media_content_changed")
                sample_tolerance = 1 / int(audio["sample_rate"])
                if (abs(_seconds(new_audio["start_time"]) - audio_start) > sample_tolerance
                        or abs(_seconds(new_audio["duration"]) - _seconds(audio["duration"])) > sample_tolerance
                        or await _audio_hash(output, options) != original_audio_hash):
                    raise _CorrectionError("audio_changed")
                original_last_frame = await _last_video_timestamp(
                    original.path, video_start + _seconds(video["duration"]), options)
                corrected_last_frame = await _last_video_timestamp(
                    output, _seconds(new_video["start_time"]) + _seconds(new_video["duration"]), options)
                if (abs(_seconds(new_video["start_time"])) > float(time_base)
                        or abs(corrected_last_frame - original_last_frame - delay)
                        > float(time_base) + 0.000002):
                    raise _CorrectionError("video_timeline_changed")
                await _run([options.ffmpeg, "-v", "error", "-xerror", "-nostdin",
                            "-protocol_whitelist", "file,pipe", "-i", str(output),
                            "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"])
                original._check()
                corrected = await snapshot_file(output, max_bytes=options.max_bytes)
                metadata = {
                    "format": "jarvis.recording_timing.v1", "video_delay_ms": options.video_delay_ms,
                    "original_sha256": original.content_sha256,
                    "corrected_sha256": corrected.content_sha256,
                    "audio_payload_sha256": original_audio_hash, "audio_reencoded": False,
                    "video_reencoded": True, "initial_picture_hold_seconds": video_start + delay,
                    "tail_without_audio_seconds": max(0, after_duration - audio_start - _seconds(audio["duration"])),
                    "duration_seconds": after_duration, "calibration": "configured_not_inferred",
                    "original_last_frame_seconds": original_last_frame,
                    "corrected_last_frame_seconds": corrected_last_frame,
                }
                return RecordingTimingResult("corrected", "validated", corrected, metadata)
    except asyncio.CancelledError:
        if corrected is not None:
            corrected.close()
        raise
    except Exception as error:
        if corrected is not None:
            corrected.close()
        reason = ("timeout" if isinstance(error, TimeoutError) else
                  str(error) if isinstance(error, _CorrectionError) else
                  "executable_unavailable" if isinstance(error, FileNotFoundError) else
                  "correction_failed")
        return RecordingTimingResult("failed", reason)
