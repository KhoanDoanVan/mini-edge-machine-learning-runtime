# Porting MiniOrt to ESP32

This document describes the recommended path from the working desktop/iOS
MiniOrt runtime to an embedded ESP32 runtime. The first target should be an
ESP32-S3 DevKit. The original ESP32 DevKit can be used for compatibility tests,
and ESP32-CAM should be added after the tensor runtime works without a camera.

## Why ESP32-S3 first?

The ESP32-S3 provides dual Xtensa LX7 cores, Wi-Fi/BLE, USB, and peripherals
useful for embedded ML applications. ESP-IDF is Espressif's official framework
and supplies the compiler, CMake-based build, flashing, and serial-monitor
workflow:

<https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/index.html>

The three boards should be treated as different hardware targets:

```text
ESP32-S3 DevKit  -> primary runtime and performance target
ESP32 DevKit    -> lower-resource compatibility target
ESP32-CAM       -> later camera/pre-processing application target
```

Do not start with the camera. Camera frame buffers and image preprocessing can
consume substantially more RAM than the small MLP model.

## Target architecture

Keep the existing portable MiniOrt core and add an ESP32 platform layer:

```text
mini_ort core
  ├── desktop backend
  ├── iOS backend
  └── ESP32 embedded backend
        ├── ESP-IDF integration
        ├── static arena allocator
        ├── reference CPU kernels
        └── optional ESP32-optimized kernels
```

The embedded build should not depend on Python, `std::filesystem`, dynamic
libraries, or model files on disk.

## Phase 0: Define the hardware contract

Choose one exact board and record:

```text
chip model
flash size
internal RAM
PSRAM availability
clock frequency
maximum latency
maximum model size
power budget
```

Start with batch size `1` and static tensor shapes. Define a golden input and
output from `examples/mlp.py`; the ESP32 result must match the desktop result
within a documented tolerance.

## Phase 1: Freeze the embedded operator set

The first embedded graph should contain only:

```text
Input -> Linear -> ReLU -> Linear -> Output
```

Initially support `Linear`/`MatMul`, bias `Add`, and `ReLU`. Reject dynamic
shapes, unsupported operators, and variable batch sizes during model
conversion rather than at runtime.

## Phase 2: Use static memory

Microcontroller runtimes commonly use a preallocated arena because heap usage
and virtual memory are unsuitable for deterministic inference. This is a
central design principle of TensorFlow Lite Micro:

<https://arxiv.org/abs/2010.08678>

Separate memory into:

```text
Flash:       model weights and immutable model bytes
Persistent:  execution metadata and operator state
Arena:       intermediate activations and scratch buffers
```

The embedded `Run()` path must not call `new`, `malloc`, filesystem APIs, or
grow containers. The planner should calculate peak arena usage before startup
and fail if it exceeds the configured arena.

## Phase 3: Store the model in firmware

Desktop MiniOrt loads a `.mer` file from a path. ESP32 firmware should instead
embed the model as a constant byte array:

```text
tiny_mlp.mer
    -> host conversion tool
    -> tiny_mlp_model.cc/.h
    -> const model bytes in flash
```

The embedded API should accept:

```text
model pointer
model byte count
arena pointer
arena byte count
```

The memory-backed model reader must be independent of the filesystem reader.

## Phase 4: Create an ESP-IDF component

ESP-IDF applications are composed of components that are compiled and linked
into the firmware:

<https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/build-system.html>

Recommended layout:

```text
esp32_app/
├── CMakeLists.txt
├── sdkconfig
├── main/
│   ├── CMakeLists.txt
│   └── app_main.cpp
└── components/
    └── mini_ort/
        ├── CMakeLists.txt
        ├── include/
        ├── src/
        └── model/
```

The component should build MiniOrt as a static library. The application should
only call the small embedded C API, for example create-session-from-buffer,
run, read output, and destroy-session.

## Phase 5: Build and flash a no-ML ESP32 application

After installing ESP-IDF, activate its environment and select the target:

```bash
source ~/esp/esp-idf/export.sh
cd esp32_app
idf.py set-target esp32s3
idf.py build
idf.py -p YOUR_SERIAL_PORT flash monitor
```

The official ESP-IDF workflow documents project creation, build, flash, and
monitor commands:

<https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/start-project.html>

First verify serial output and reset behavior before adding MiniOrt.

## Phase 6: Run the float32 reference model

Add the generated model array and the MiniOrt component. Run the same fixed
input used by the Python example and print:

```text
output values
input/output shapes
inference latency
arena bytes used
free heap before and after initialization
```

The first success criterion is numerical equivalence with the desktop backend,
not speed.

## Phase 7: Add int8 quantization

Use this order:

```text
float32 reference
    -> int8 weights
    -> int8 activations
    -> int8 bias and kernels
```

Store scale and zero-point for every quantized tensor. Use representative
calibration data and compare accuracy against the float32 model. The integer
scale/zero-point approach is described in:

<https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf>

Start with int8; defer int4 until the complete int8 path is correct and
measured.

## Phase 8: Fuse and optimize operators

Fuse common sequences:

```text
Linear + Bias
Linear + Bias + ReLU
```

This reduces temporary buffers and memory traffic. Then optimize in this order:

1. Correctness tests.
2. Peak arena size.
3. Int8 arithmetic.
4. Operator fusion.
5. Aligned buffers and loop layout.
6. ESP32-specific kernels.
7. Link-time optimization and dead-code removal.
8. Latency and energy measurement.

Do not write hardware-specific assembly before a portable reference kernel and
golden tests exist.

## Phase 9: Consider ahead-of-time code generation

An interpreter is useful when models change. For one fixed production model,
generate C/C++ containing the operator sequence, weights, tensor offsets, and
arena plan:

```text
model -> generated C/C++ -> ESP-IDF component -> firmware
```

This removes generic graph traversal and parser overhead. MicroTVM describes
this style of compiling model operators for bare-metal targets:

<https://arxiv.org/abs/2304.04842>

MCUNet also demonstrates co-design between the model and a memory-aware,
specialized runtime:

<https://arxiv.org/abs/2007.10319>

Keep the generic MiniOrt backend as the reference implementation for validating
generated code.

## Phase 10: Verify every target build

Record these values for each model and board:

```text
firmware flash bytes
model flash bytes
static RAM bytes
arena bytes
stack usage
heap usage
initialization time
worst-case inference latency
energy per inference
maximum output error
```

Use the linker map file and serial diagnostics. TinyML benchmarking should
consider memory, latency, and energy—not accuracy alone:

<https://arxiv.org/abs/2112.01319>

## Recommended milestone sequence

```text
1. ESP32-S3 hello-world ESP-IDF application
2. Static MiniOrt component
3. Embedded model byte array
4. Float32 Linear/ReLU inference
5. Static arena with no allocation during Run()
6. Int8 quantization
7. Fused Linear+Bias+ReLU kernel
8. ESP32 DevKit compatibility test
9. AOT code generation
10. ESP32-CAM camera pipeline
```

The ESP32-CAM should therefore be the final application milestone, not the
first runtime-port milestone.
