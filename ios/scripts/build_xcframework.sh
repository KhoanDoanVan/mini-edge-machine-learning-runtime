#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_ROOT="${1:-${PROJECT_ROOT}/build/ios}"
DEPLOYMENT_TARGET="${MINI_ORT_IOS_DEPOYMENT_TARGET:-15.0}"

if [[ -z "${OUTPUT_ROOT}" || "${OUTPUT_ROOT}" == "/" ]]; then 
    echo "Refusing to use an unsafe iOS output directory" >&2
    exit 1
fi


DEVICE_BUILD="${OUTPUT_ROOT}/iphoneos"
SIMULATOR_BUILD="${OUTPUT_ROOT}/iphonesimulator"
DEVICE_INSTALL="${DEVICE_BUILD}/install"
SIMULATOR_INSTALL="${SIMULATOR_BUILD}/install"
XCFRAMEWORK="${OUTPUT_ROOT}/MiniOrt.xcframework"
NEXT_XCFRAMEWORK="${OUTPUT_ROOT}/MiniOrt.next.xcframework"

COMMON_OPTIONS=(
    -DCMAKE_SYSTEM_NAME=iOS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_OSX_DEPLOYMENT_TARGET="${DEPLOYMENT_TARGET}"
    -DMINI_ORT_LIBRARY_TYPE=STATIC
    -DMINI_ORT_BUILD_EXAMPLES=OFF
    -DMINI_ORT_BUILD_BENCHMARKS=OFF
)

mkdir -p "${OUTPUT_ROOT}"

cmake -S "${PROJECT_ROOT}/cpp" -B "${DEVICE_BUILD}" \
    "${COMMON_OPTIONS[@]}" \
    -DCMAKE_OSX_SYSROOT=iphoneos \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_INSTALL_PREFIX="${DEVICE_INSTALL}"
cmake --build "${DEVICE_BUILD}" --target mini_ort --parallel
cmake --install "${DEVICE_BUILD}"

SIMULATOR_SDK="$(xcrun --sdk iphonesimulator --show-sdk-path)"

cmake -S "${PROJECT_ROOT}/cpp" -B "${SIMULATOR_BUILD}" \
    "${COMMON_OPTIONS[@]}" \
    -DCMAKE_OSX_SYSROOT="${SIMULATOR_SDK}" \
    '-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64' \
    -DCMAKE_INSTALL_PREFIX="${SIMULATOR_INSTALL}"
cmake --build "${SIMULATOR_BUILD}" --target mini_ort --parallel
cmake --install "${SIMULATOR_BUILD}"

cp "${PROJECT_ROOT}/ios/ModuleMap/module.modulemap" \
    "${DEVICE_INSTALL}/include/module.modulemap"

cp "${PROJECT_ROOT}/ios/ModuleMap/module.modulemap" \
    "${SIMULATOR_INSTALL}/include/module.modulemap"

rm -rf "${NEXT_XCFRAMEWORK}"

xcodebuild -create-xcframework \
    -library "${DEVICE_INSTALL}/lib/libmini_ort.a" \
    -headers "${DEVICE_INSTALL}/include" \
    -library "${SIMULATOR_INSTALL}/lib/libmini_ort.a" \
    -headers "${SIMULATOR_INSTALL}/include" \
    -output "${NEXT_XCFRAMEWORK}"

rm -rf "${XCFRAMEWORK}"

mv "${NEXT_XCFRAMEWORK}" "${XCFRAMEWORK}"

echo "Created ${XCFRAMEWORK}"