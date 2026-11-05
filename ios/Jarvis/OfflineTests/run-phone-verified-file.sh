#!/bin/bash
set -euo pipefail
phone_test_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
phone_repo_root="$(cd -- "${phone_test_dir}/../../.." && pwd)"
phone_source_root="${phone_repo_root}/ios/Jarvis/Jarvis"
phone_output="${PHONE_TEST_OUTPUT_DIR:-${phone_repo_root}/.jarvis-dev/phone-tests/verified-file-20261102}"
mkdir -p "${phone_output}"
swiftc -module-cache-path "${phone_output}/module-cache" \
    "${phone_source_root}/Models/JarvisModels.swift" \
    "${phone_source_root}/Models/JarvisMemoryModels.swift" \
    "${phone_source_root}/Models/JarvisFileModels.swift" \
    "${phone_source_root}/Models/JarvisResourceSelectionContext.swift" \
    "${phone_source_root}/Models/JarvisMediaFileModels.swift" \
    "${phone_source_root}/Models/JarvisWorkspaceModels.swift" \
    "${phone_source_root}/Services/JarvisWorkspaceLocalStore.swift" \
    "${phone_source_root}/Services/JarvisVerifiedTemporaryFile.swift" \
    "${phone_test_dir}/PhoneVerifiedFileRegression.swift" \
    "${phone_test_dir}/PhoneMediaContractRegression.swift" \
    -o "${phone_output}/verified-file"
"${phone_output}/verified-file" | tee "${phone_output}/tests.log"

swiftc -module-cache-path "${phone_output}/module-cache" \
    "${phone_source_root}/Models/JarvisModels.swift" \
    "${phone_source_root}/Models/JarvisFileModels.swift" \
    "${phone_source_root}/Models/JarvisResourceSelectionContext.swift" \
    "${phone_source_root}/Models/JarvisMediaFileModels.swift" \
    "${phone_source_root}/Models/JarvisWorkModels.swift" \
    "${phone_source_root}/Models/JarvisMemoryModels.swift" \
    "${phone_source_root}/Models/JarvisCalendarModels.swift" \
    "${phone_source_root}/Models/JarvisReminderModels.swift" \
    "${phone_source_root}/Models/JarvisRealtimeModels.swift" \
    "${phone_source_root}/Models/JarvisWorkspaceModels.swift" \
    "${phone_source_root}/Services/JarvisWorkspaceLocalStore.swift" \
    "${phone_source_root}/Services/JarvisVerifiedTemporaryFile.swift" \
    "${phone_source_root}/Services/JarvisAPIClient.swift" \
    "${phone_source_root}/Services/JarvisAPIClient+Media.swift" \
    "${phone_test_dir}/PhoneMediaTransportRegression.swift" \
    -o "${phone_output}/media-transport"
"${phone_output}/media-transport" | tee "${phone_output}/transport.log"
