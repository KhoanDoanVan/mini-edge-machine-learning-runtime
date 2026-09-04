#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IOS_OUTPUT="${PROJECT_ROOT}/build/ios"
MODEL_OUTPUT="${PROJECT_ROOT}/build/models/tiny_mlp.mer"


"${SCRIPT_DIR}/build_xcframework.sh" "${IOS_OUTPUT}"

mkdir -p "$(dirname "${MODEL_OUTPUT}")"

cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}" python3 -c \
  'from examples.mlp import build_model; build_model().save("build/models/tiny_mlp.mer")'

xcodebuild \
    -project "${PROJECT_ROOT}/demo/iOS/MiniOrtDemo/MiniOrtDemo.xcodeproj" \
    -scheme MiniOrtDemo \
    -configuration Debug \
    -sdk iphonesimulator \
    -destination 'generic/platform=iOS Simulator' \
    -derivedDataPath "${IOS_OUTPUT}/DerivedData" \
    CODE_SIGNING_ALLOWED=NO \
    build

echo "Built ${IOS_OUTPUT}/DerivedData/Build/Products/Debug-iphonesimulator/MiniOrtDemo.app"