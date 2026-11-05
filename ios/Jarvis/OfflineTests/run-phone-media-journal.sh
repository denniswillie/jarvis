#!/bin/bash
set -euo pipefail
phone_test_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
phone_repo_root="$(cd -- "${phone_test_dir}/../../.." && pwd)"
phone_source_root="${phone_repo_root}/ios/Jarvis/Jarvis"
phone_output="${PHONE_TEST_OUTPUT_DIR:-${phone_repo_root}/.jarvis-dev/phone-tests/media-offline-save-20261102}"
phone_fixture="${PHONE_MEDIA_FIXTURE:-${phone_repo_root}/.jarvis-dev/phone-tests/media-preview-20261102/synthetic.mp4}"
test -f "${phone_fixture}"
mkdir -p "${phone_output}"
swiftc -module-cache-path "${phone_output}/module-cache" \
    "${phone_source_root}/Models/JarvisModels.swift" \
    "${phone_source_root}/Models/JarvisWorkspaceModels.swift" \
    "${phone_source_root}/Services/JarvisWorkspaceLocalStore.swift" \
    "${phone_test_dir}/PhoneMediaJournalRegression.swift" \
    -o "${phone_output}/media-journal"
"${phone_output}/media-journal" "${phone_fixture}" | tee "${phone_output}/journal.log"

swiftc -module-cache-path "${phone_output}/module-cache" \
    "${phone_source_root}/Models/JarvisModels.swift" \
    "${phone_source_root}/Models/JarvisWorkspaceModels.swift" \
    "${phone_source_root}/Models/JarvisWorkModels.swift" \
    "${phone_source_root}/Models/JarvisMemoryModels.swift" \
    "${phone_source_root}/Models/JarvisFileModels.swift" \
    "${phone_source_root}/Models/JarvisResourceSelectionContext.swift" \
    "${phone_source_root}/Models/JarvisMediaFileModels.swift" \
    "${phone_source_root}/Services/JarvisWorkspaceLocalStore.swift" \
    "${phone_source_root}/Services/JarvisVerifiedTemporaryFile.swift" \
    "${phone_source_root}/ViewModels/JarvisMediaPreviewViewModel.swift" \
    "${phone_source_root}/Models/JarvisSandboxModels.swift" \
    "${phone_source_root}/Models/JarvisCalendarModels.swift" \
    "${phone_source_root}/Models/JarvisReminderModels.swift" \
    "${phone_source_root}/Models/JarvisRealtimeModels.swift" \
    "${phone_source_root}/Services/JarvisAPIClient.swift" \
    "${phone_source_root}/ViewModels/JarvisSandboxViewModel.swift" \
    "${phone_source_root}/ViewModels/JarvisWorkspaceViewModel.swift" \
    "${phone_test_dir}/PhoneMediaWorkspaceRegression.swift" \
    -o "${phone_output}/media-workspace"
"${phone_output}/media-workspace" "${phone_fixture}" | tee "${phone_output}/workspace.log"
