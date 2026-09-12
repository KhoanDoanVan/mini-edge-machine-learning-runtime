<div align="center">
  <img src="assets/icon.png" alt="TinyONNX — Lightweight AI Inference" width="720" />

  <h1>TinyONNX · Mini Edge Runtime</h1>

  <p>
    <strong>A tiny, dependency-free machine learning runtime built from scratch for learning and edge inference.</strong>
  </p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
    <img src="https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white" alt="C++20" />
    <img src="https://img.shields.io/badge/runtime_dependencies-0-7AC70C" alt="Zero runtime dependencies" />
    <img src="https://img.shields.io/badge/license-MIT-8A2BE2" alt="MIT license" />
    <img src="https://img.shields.io/badge/status-experimental-F59E0B" alt="Experimental status" />
  </p>

  <p>
    <a href="#overview">Overview</a> ·
    <a href="#hardware-support">Hardware</a> ·
    <a href="#quick-start">Quick start</a> ·
    <a href="#architecture">Architecture</a> ·
    <a href="#supported-operations">Supported operations</a> ·
    <a href="#development">Development</a>
  </p>
</div>

---

## Overview

TinyONNX is a compact implementation of an ONNX-inspired model format and inference runtime for mobile, edge, and on-device AI. The project keeps the entire execution path visible: tensors, graph validation, operator schemas, planning, kernels, and native execution.

It does **not** depend on ONNX, ONNX Runtime, NumPy, or another machine learning framework.

> [!IMPORTANT]
> TinyONNX is an educational, experimental runtime. Its JSON model format is inspired by ONNX architecture, but it is not the official ONNX wire format and is not intended for production workloads yet.

## Hardware support

| Board | Image | Progress | Documentation / example |
|---|---|---|---|
| ESP32-CAM | <img src="assets/models/esp32_cam.jpg" alt="ESP32-CAM board" width="110" /> | Planned; not yet validated. | Not available yet. |
| ESP32 DevKit | <img src="assets/models/esp32_dev_kit.jpg" alt="ESP32 DevKit board" width="110" /> | **Complete reference:** ESP-IDF build and flash verified. | [Build guide](docs/ESP32/ESP32_DEVKIT_BUILD_GUIDE.md) · [Firmware example](examples/esp32_generated_mlp/main/app_main.c) |
| ESP32-S3 | <img src="assets/models/esp32_s3.jpg" alt="ESP32-S3 board" width="110" /> | Planned; not yet validated. | Not available yet. |

## License

Distributed under the MIT License.

<div align="center">
  <sub>Built to make edge inference internals small enough to understand.</sub>
</div>
