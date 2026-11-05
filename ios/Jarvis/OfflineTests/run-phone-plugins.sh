#!/bin/bash
set -euo pipefail
phone_test_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
phone_repo_root="$(cd -- "${phone_test_dir}/../../.." && pwd)"
phone_source_root="${phone_repo_root}/ios/Jarvis/Jarvis"
phone_output="${PHONE_TEST_OUTPUT_DIR:?Use an isolated output directory}"
mkdir -p "${phone_output}"
phone_sources=(
    Models/JarvisModels.swift Models/JarvisFileModels.swift Models/JarvisWorkModels.swift
    Models/JarvisMemoryModels.swift Models/JarvisCalendarModels.swift Models/JarvisReminderModels.swift
    Models/JarvisRealtimeModels.swift Models/JarvisWorkspaceModels.swift
    Models/JarvisPluginPresentation.swift Models/JarvisPluginCatalogModels.swift
    Services/JarvisAPIClient.swift Services/JarvisAPIClient+Plugins.swift
    ViewModels/JarvisPluginsViewModel.swift
)
phone_inputs=()
for phone_path in "${phone_sources[@]}"; do phone_inputs+=("${phone_source_root}/${phone_path}"); done
for phone_case in PhonePluginCatalogRegression PhonePluginsRegression; do
    swiftc -module-cache-path "${phone_output}/module-cache" "${phone_inputs[@]}" \
        "${phone_test_dir}/${phone_case}.swift" -o "${phone_output}/${phone_case}"
    "${phone_output}/${phone_case}" "${phone_test_dir}/fixtures/plugin-catalog-v1.json" | tee "${phone_output}/${phone_case}.log"
done
