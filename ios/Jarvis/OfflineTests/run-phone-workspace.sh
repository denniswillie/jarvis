#!/bin/bash
set -euo pipefail

# Host-only product checks. No Xcode, app, simulator, provider or database.
phone_test_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
phone_repo_root="$(cd -- "${phone_test_dir}/../../.." && pwd)"
phone_source_root="${phone_repo_root}/ios/Jarvis/Jarvis"
phone_test_output="${PHONE_TEST_OUTPUT_DIR:-${phone_repo_root}/.jarvis-dev/phone-tests}"
mkdir -p "${phone_test_output}"

swiftc -module-cache-path "${phone_test_output}/module-cache" \
    "${phone_source_root}/Models/JarvisModels.swift" \
    "${phone_source_root}/Models/JarvisWorkModels.swift" \
    "${phone_source_root}/Models/JarvisWorkspaceModels.swift" \
    "${phone_source_root}/Services/JarvisWorkspaceLocalStore.swift" \
    "${phone_source_root}/Models/JarvisMemoryModels.swift" \
    "${phone_source_root}/Models/JarvisFileModels.swift" \
    "${phone_source_root}/Models/JarvisMediaFileModels.swift" \
    "${phone_source_root}/Services/JarvisVerifiedTemporaryFile.swift" \
    "${phone_source_root}/ViewModels/JarvisMediaPreviewViewModel.swift" \
    "${phone_source_root}/Models/JarvisSandboxModels.swift" \
    "${phone_source_root}/Models/JarvisCalendarModels.swift" \
    "${phone_source_root}/Models/JarvisReminderModels.swift" \
    "${phone_source_root}/Models/JarvisRealtimeModels.swift" \
    "${phone_source_root}/Services/JarvisAPIClient.swift" \
    "${phone_source_root}/ViewModels/JarvisSandboxViewModel.swift" \
    "${phone_source_root}/ViewModels/JarvisWorkspaceViewModel.swift" \
    "${phone_test_dir}/PhoneWorkspaceRegression.swift" \
    -o "${phone_test_output}/phone-workspace-regression"
"${phone_test_output}/phone-workspace-regression" | tee "${phone_test_output}/product-state.log"

swiftc -module-cache-path "${phone_test_output}/module-cache" \
    "${phone_source_root}/Models/JarvisModels.swift" \
    "${phone_source_root}/Models/JarvisFileModels.swift" \
    "${phone_source_root}/Models/JarvisWorkModels.swift" \
    "${phone_source_root}/Models/JarvisMemoryModels.swift" \
    "${phone_source_root}/Models/JarvisCalendarModels.swift" \
    "${phone_source_root}/Models/JarvisReminderModels.swift" \
    "${phone_source_root}/Models/JarvisRealtimeModels.swift" \
    "${phone_source_root}/Models/JarvisWorkspaceModels.swift" \
    "${phone_source_root}/Services/JarvisAPIClient.swift" \
    "${phone_source_root}/Models/JarvisAccountProductModels.swift" \
    "${phone_source_root}/Services/JarvisAPIClient+Workspace.swift" \
    "${phone_test_dir}/PhoneWorkspaceTransportRegression.swift" \
    -o "${phone_test_output}/phone-workspace-transport-regression"
"${phone_test_output}/phone-workspace-transport-regression" | tee "${phone_test_output}/transport.log"

swiftc -module-cache-path "${phone_test_output}/module-cache" \
    "${phone_source_root}/Models/JarvisModels.swift" \
    "${phone_source_root}/Models/JarvisMemoryModels.swift" \
    "${phone_source_root}/Models/JarvisFileModels.swift" \
    "${phone_source_root}/Models/JarvisWorkModels.swift" \
    "${phone_source_root}/ViewModels/JarvisWorkInboxViewModel.swift" \
    "${phone_test_dir}/PhoneWorkInboxRegression.swift" \
    -o "${phone_test_output}/phone-work-inbox-regression"
"${phone_test_output}/phone-work-inbox-regression" | tee "${phone_test_output}/work-inbox.log"

swiftc -module-cache-path "${phone_test_output}/module-cache" \
    "${phone_source_root}/Models/JarvisPluginPresentation.swift" \
    "${phone_test_dir}/PhonePluginPresentationRegression.swift" \
    -o "${phone_test_output}/phone-plugin-presentation-regression"
"${phone_test_output}/phone-plugin-presentation-regression" | tee "${phone_test_output}/plugin-presentation.log"

swiftc -module-cache-path "${phone_test_output}/module-cache" \
    "${phone_source_root}/Models/JarvisModels.swift" \
    "${phone_source_root}/Models/JarvisMemoryModels.swift" \
    "${phone_source_root}/Models/JarvisFileModels.swift" \
    "${phone_source_root}/Models/JarvisWorkspaceModels.swift" \
    "${phone_source_root}/Models/JarvisMediaFileModels.swift" \
    "${phone_source_root}/Services/JarvisWorkspaceLocalStore.swift" \
    "${phone_source_root}/Services/JarvisVerifiedTemporaryFile.swift" \
    "${phone_source_root}/ViewModels/JarvisMediaPreviewViewModel.swift" \
    "${phone_source_root}/ViewModels/JarvisFileMediaPreviewModel.swift" \
    "${phone_source_root}/Models/JarvisResourceSelectionContext.swift" \
    "${phone_source_root}/Services/JarvisAudiovisualThumbnailLoader.swift" \
    "${phone_source_root}/ViewModels/JarvisFilesViewModel.swift" \
    "${phone_test_dir}/PhoneFilesRegression.swift" \
    -o "${phone_test_output}/phone-files-regression"
"${phone_test_output}/phone-files-regression" | tee "${phone_test_output}/files.log"

swiftc -module-cache-path "${phone_test_output}/module-cache" \
    "${phone_source_root}/Models/JarvisPluginPresentation.swift" \
    "${phone_source_root}/Models/JarvisPluginCatalogModels.swift" \
    "${phone_test_dir}/PhonePluginCatalogRegression.swift" \
    -o "${phone_test_output}/phone-plugin-catalog-regression"
"${phone_test_output}/phone-plugin-catalog-regression" "${phone_test_dir}/fixtures/plugin-catalog-v1.json" \
    | tee "${phone_test_output}/plugin-catalog.log"
