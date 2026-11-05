"""Extract bounded local, permission-cleared clips into chronological replay assets.

No downloader, provider client or inference is involved. Source frame timestamps
come from ffprobe, not the time extraction happens. Audio is deliberately omitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from evals.responsive_voice.video_replay import FrameAsset, PermissionRecord, ReplayInputs


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def _execute(arguments, *, timeout=45):
    result = subprocess.run(arguments, stdin=subprocess.DEVNULL, capture_output=True,
                            timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError("video_extraction_command_failed")
    if len(result.stdout) > 8 * 1024 * 1024:
        raise ValueError("video_metadata_too_large")
    return result.stdout


def _number(raw):
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("video_timestamp_not_finite")
    return value


def extract_clip(*, video: Path, output: Path, permission: PermissionRecord,
                 expected_sha256: str, sample_interval_ms=1000):
    if not permission.local_evaluation or not permission.retain_derived_results:
        raise ValueError("video_permission_unresolved")
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
        raise ValueError("video_checksum_required")
    if type(sample_interval_ms) is not int or not 100 <= sample_interval_ms <= 5000:
        raise ValueError("video_sampling_interval_invalid")
    video = Path(video).resolve(strict=True)
    output = Path(output)
    if not video.is_file() or not 0 < video.stat().st_size <= 512 * 1024 * 1024:
        raise ValueError("video_source_size_invalid")
    if sha256_file(video) != expected_sha256:
        raise ValueError("video_source_checksum_mismatch")
    if output.exists():
        raise ValueError("video_output_already_exists")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("video_decoder_unavailable")
    metadata = json.loads(_execute([ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe",
        "-select_streams", "v:0", "-show_entries",
        "stream=width,height,duration,start_time:format=duration", "-of", "json", str(video)], timeout=15))
    streams = metadata.get("streams", [])
    if len(streams) != 1:
        raise ValueError("video_stream_unavailable")
    stream = streams[0]
    duration = _number(stream.get("duration", metadata.get("format", {}).get("duration")))
    if not 0 < duration <= 180:
        raise ValueError("video_clip_duration_invalid")
    width, height = int(stream["width"]), int(stream["height"])
    if not 0 < width <= 8192 or not 0 < height <= 8192:
        raise ValueError("video_dimensions_invalid")
    raw = json.loads(_execute([ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe",
        "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "json", str(video)]))
    frames = raw.get("frames", [])
    if not 0 < len(frames) <= 50000:
        raise ValueError("video_frame_count_invalid")
    timestamps = [_number(frame["best_effort_timestamp_time"]) for frame in frames]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("video_frame_timestamps_not_increasing")
    origin, selected, target = timestamps[0], [], timestamps[0]
    for index, timestamp in enumerate(timestamps):
        if timestamp + 1e-7 >= target:
            selected.append((index, timestamp))
            target = timestamp + sample_interval_ms / 1000
    if len(selected) > 360:
        raise ValueError("video_sample_budget_exceeded")
    offsets = [round((timestamp - origin) * 1000) for _, timestamp in selected]
    duration_ms = math.ceil(duration * 1000)
    if offsets[-1] > duration_ms or len(set(offsets)) != len(offsets):
        raise ValueError("video_frame_offset_invalid")
    output.mkdir(parents=True, exist_ok=False)
    frame_dir = output / "frames"
    frame_dir.mkdir()
    expression = "+".join(f"eq(n\\,{index})" for index, _ in selected)
    # Integer frame selection preserves source timestamps and does not invent
    # repeated frames to satisfy a requested output frame rate.
    _execute([ffmpeg, "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
        "-i", str(video), "-map", "0:v:0", "-an", "-vf", f"select={expression}",
        "-fps_mode", "vfr", "-frames:v", str(len(selected)), "-q:v", "3",
        str(frame_dir / "%06d.jpg")], timeout=60)
    assets = []
    for number, offset in enumerate(offsets, start=1):
        path = frame_dir / f"{number:06d}.jpg"
        if not path.is_file() or not 0 < path.stat().st_size <= 4 * 1024 * 1024:
            raise ValueError("video_extracted_frame_missing_or_oversized")
        assets.append(FrameAsset(offset_ms=offset, relative_path=f"frames/{number:06d}.jpg",
            sha256=sha256_file(path), width=width, height=height))
    inputs = ReplayInputs(duration_ms=duration_ms, frames=tuple(assets))
    manifest = {"format": "jarvis.video_assets.v1", "source_sha256": expected_sha256,
        "permission": permission.model_dump(), "source_first_pts_seconds": origin,
        "sample_interval_ms": sample_interval_ms, "sampling": "minimum_source_frame_interval",
        "source_frame_indices": [index for index, _ in selected],
        "source_pts_seconds": [timestamp for _, timestamp in selected],
        "source_audio_included": False, "inputs": inputs.model_dump(mode="json"),
        "quality_qualified": False}
    (output / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permission-record", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--sample-interval-ms", type=int, default=1000)
    args = parser.parse_args()
    permission = PermissionRecord.model_validate_json(args.permission_record.read_text(encoding="utf-8"))
    manifest = extract_clip(video=args.video, output=args.output, permission=permission,
        expected_sha256=args.source_sha256, sample_interval_ms=args.sample_interval_ms)
    print(json.dumps({"extracted_frames": len(manifest["inputs"]["frames"]),
                      "quality_qualified": False, "source_audio_included": False}))


if __name__ == "__main__":
    main()
