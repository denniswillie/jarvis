#!/bin/bash
set -euo pipefail
phone_test_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
phone_repo_root="$(cd -- "${phone_test_dir}/../../.." && pwd)"
phone_source_root="${phone_repo_root}/ios/Jarvis/Jarvis"
phone_output="${PHONE_TEST_OUTPUT_DIR:-${phone_repo_root}/.jarvis-dev/phone-tests/media-preview-20261102}"
mkdir -p "${phone_output}"
# Synthetic silent footage only; no camera, microphone, device or provider.
ffmpeg -hide_banner -loglevel error -f lavfi -i 'testsrc2=size=1280x720:rate=30' -t 3 \
    -c:v libx264 -threads 2 -preset ultrafast -b:v 30M -minrate 30M -maxrate 30M -bufsize 30M \
    -x264-params 'nal-hrd=cbr:force-cfr=1' -an -movflags +faststart -y "${phone_output}/synthetic.mp4"
ffprobe -v error -show_entries format=duration,size:stream=codec_name,codec_type,width,height \
    -of json "${phone_output}/synthetic.mp4" > "${phone_output}/inspection.json"
swiftc -module-cache-path "${phone_output}/module-cache" \
    "${phone_source_root}/Models/JarvisModels.swift" \
    "${phone_source_root}/Models/JarvisMemoryModels.swift" \
    "${phone_source_root}/Models/JarvisFileModels.swift" \
    "${phone_source_root}/Models/JarvisMediaFileModels.swift" \
    "${phone_source_root}/Models/JarvisWorkspaceModels.swift" \
    "${phone_source_root}/Services/JarvisWorkspaceLocalStore.swift" \
    "${phone_source_root}/Services/JarvisVerifiedTemporaryFile.swift" \
    "${phone_source_root}/ViewModels/JarvisMediaPreviewViewModel.swift" \
    "${phone_test_dir}/PhoneMediaPreviewRegression.swift" \
    -o "${phone_output}/media-preview"
"${phone_output}/media-preview" "${phone_output}/synthetic.mp4" | tee "${phone_output}/tests.log"
