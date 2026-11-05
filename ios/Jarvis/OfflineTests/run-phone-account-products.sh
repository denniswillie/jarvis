#!/bin/bash
set -euo pipefail
phone_test_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
phone_repo_root="$(cd -- "${phone_test_dir}/../../.." && pwd)"
phone_source_root="${phone_repo_root}/ios/Jarvis/Jarvis"
phone_output="${PHONE_TEST_OUTPUT_DIR:-${phone_repo_root}/.jarvis-dev/phone-tests/account-products-20261102}"
mkdir -p "${phone_output}"

swiftc -module-cache-path "${phone_output}/module-cache" \
    "${phone_source_root}/Models/JarvisModels.swift" \
    "${phone_source_root}/Models/JarvisWorkspaceModels.swift" \
    "${phone_source_root}/Models/JarvisRealtimeModels.swift" \
    "${phone_source_root}/Models/JarvisAccountProductModels.swift" \
    "${phone_source_root}/ViewModels/JarvisPreferencesViewModel.swift" \
    "${phone_test_dir}/PhoneAccountProductsRegression.swift" \
    -o "${phone_output}/account-products"
"${phone_output}/account-products" | tee "${phone_output}/models.log"

swiftc -module-cache-path "${phone_output}/module-cache" \
    "${phone_source_root}/Models/JarvisModels.swift" \
    "${phone_source_root}/Models/JarvisFileModels.swift" \
    "${phone_source_root}/Models/JarvisWorkModels.swift" \
    "${phone_source_root}/Models/JarvisMemoryModels.swift" \
    "${phone_source_root}/Models/JarvisCalendarModels.swift" \
    "${phone_source_root}/Models/JarvisReminderModels.swift" \
    "${phone_source_root}/Models/JarvisRealtimeModels.swift" \
    "${phone_source_root}/Models/JarvisWorkspaceModels.swift" \
    "${phone_source_root}/Models/JarvisAccountProductModels.swift" \
    "${phone_source_root}/Services/JarvisAPIClient.swift" \
    "${phone_source_root}/Services/JarvisAPIClient+Workspace.swift" \
    "${phone_test_dir}/PhoneAccountProductTransportRegression.swift" \
    -o "${phone_output}/account-transport"
"${phone_output}/account-transport" | tee "${phone_output}/transport.log"
