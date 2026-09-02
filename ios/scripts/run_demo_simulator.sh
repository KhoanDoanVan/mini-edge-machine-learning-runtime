#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEVICE="${1:-booted}"
APP_PATH="${PROJECT_ROOT}/build/ios/DerivedData/Build/Products/Debug-iphonesimulator/MiniOrtDemo.app"

if [[ ! -d "${APP_PATH}" ]]; then
    echo "MiniOrtDemo.app is missing; run ios/scripts/build_demo.sh first" >&2
    exit 1
fi

xcrun simctl install "${DEVICE}" "${APP_PATH}"
xcrun simctl launch "${DEVICE}" dev.miniedgeruntime.demo