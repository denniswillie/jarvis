#!/bin/bash
set -euo pipefail
phone_test_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
phone_repo_root="$(cd -- "${phone_test_dir}/../../.." && pwd)"
phone_source_root="${phone_repo_root}/ios/Jarvis/Jarvis"
phone_output="${PHONE_TEST_OUTPUT_DIR:?Use an isolated output directory}"
phone_fixture="${PHONE_MEDIA_FIXTURE:-${phone_repo_root}/.jarvis-dev/phone-tests/media-preview-20261102/synthetic.mp4}"
test -f "${phone_fixture}"
mkdir -p "${phone_output}"
phone_sources=(
    Models/JarvisResourceSelectionContext.swift Services/JarvisAudiovisualThumbnailLoader.swift
    Models/JarvisModels.swift Models/JarvisMemoryModels.swift Models/JarvisFileModels.swift
    Models/JarvisWorkModels.swift Models/JarvisWorkspaceModels.swift Models/JarvisMediaFileModels.swift
    Models/JarvisCalendarModels.swift Models/JarvisReminderModels.swift Models/JarvisRealtimeModels.swift
    Services/JarvisWorkspaceLocalStore.swift Services/JarvisVerifiedTemporaryFile.swift
    Services/JarvisAPIClient.swift Services/JarvisAPIClient+Media.swift Services/JarvisSessionFilesClient.swift
    ViewModels/JarvisMediaPreviewViewModel.swift ViewModels/JarvisFileMediaPreviewModel.swift ViewModels/JarvisFilesViewModel.swift
)
phone_inputs=()
for phone_path in "${phone_sources[@]}"; do phone_inputs+=("${phone_source_root}/${phone_path}"); done
for phone_case in PhoneFilesRegression PhoneFilesMediaRegression AudiovisualResourceChecks PhoneFilesLocalPreviewRegression; do
    swiftc -module-cache-path "${phone_output}/module-cache" "${phone_inputs[@]}" \
        "${phone_test_dir}/${phone_case}.swift" -o "${phone_output}/${phone_case}"
    "${phone_output}/${phone_case}" "${phone_fixture}" | tee "${phone_output}/${phone_case}.log"
done
