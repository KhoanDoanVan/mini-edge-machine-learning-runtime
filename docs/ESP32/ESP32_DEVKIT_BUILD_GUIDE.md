# ESP32 DevKit Build and Run Guide

This guide builds the current generated FP32 MLP, flashes it to a classic
ESP32 DevKit, and verifies inference through the serial monitor. It covers the
complete path from the host `.mer` artifact to execution on the physical
device.

## 1. What this pipeline deploys

The microcontroller does not load or parse a `.mer` file. Model parsing,
validation, scheduling, and memory planning happen on the development
computer. The generated C model is then compiled into the firmware.

```text
Python model or fixture
  -> build/models/micro_mlp_f32.mer
  -> tools/mer_codegen.py
  -> micro/generated/tiny_mlp_f32/model.c + model.h
  -> ESP-IDF components + app_main.c
  -> bootloader.bin + partition-table.bin + application.bin
  -> ESP32 flash
  -> app_main() invokes the model
  -> inference result appears over UART
```

The current validation model has this contract:

```text
Input:  float32[3] = [1.0, 2.0, 3.0]
Graph:  FullyConnected -> ReLU -> FullyConnected
Output: float32[2] = [5.50, 5.50]
```

## 2. Source layout

The relevant source files are separated by responsibility:

```text
examples/export_micro_fixture.py
  Creates the deterministic .mer validation model on the host.

tools/mer_codegen.py
python/mini_ort/compiler/
  Validate .mer, plan memory, and emit static C code.

micro/generated/tiny_mlp_f32/
  Generated model.c, model.h, and model_report.json.

micro/src/
micro/kernels/reference/
micro/include/mini_ort_micro/
  Portable C11 runtime, tensor rules, and FP32 kernels.

micro/platform/esp_idf/components/
  ESP-IDF adapters for the runtime, generated model, and memory probe.

examples/esp32_generated_mlp/
  Reusable ESP-IDF application that calls the generated model.
```

`tools/mer_codegen.py` generates the files under `micro/generated`; it does
not generate the ESP-IDF example directory. The example is handwritten
firmware scaffolding and can be reused for later compatible generated models.

## 3. Requirements

Hardware:

- a classic ESP32 DevKit;
- a USB data cable, not a charge-only cable;
- a macOS USB port or powered USB hub;
- no other program holding the board's serial port.

Host software:

- Git;
- CMake;
- Ninja;
- Python 3;
- ESP-IDF 5.3 with the `esp32` toolchain;
- this repository.

The physical pipeline documented here was verified with:

```text
Board chip: ESP32-D0WD-V3 revision 3.1
USB UART:   Silicon Labs CP2102
ESP-IDF:    v5.3
Serial:     115200 baud
```

## 4. Install ESP-IDF once

Skip this section when ESP-IDF 5.3 is already installed correctly.

```bash
mkdir -p ~/esp
cd ~/esp
git clone --branch v5.3 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh esp32
```

Load ESP-IDF into every new terminal that will run `idf.py`:

```bash
source ~/esp/esp-idf/export.sh
```

Verify the installation:

```bash
idf.py --version
xtensa-esp32-elf-gcc --version
```

Expected major results:

```text
ESP-IDF v5.3
xtensa-esp32-elf-gcc ...
```

### Python environment pin used on the tested Mac

The tested Mac uses multiple Python versions through `pyenv`. ESP-IDF had an
existing Python 3.10 environment, while one project shell selected Python
3.13. That mismatch produced a nonexistent `idf5.3_py3.13_env` path.

When the same error appears, pin the installed IDF environment before loading
ESP-IDF:

```bash
export IDF_PYTHON_ENV_PATH=/Users/doanvankhoan/.espressif/python_env/idf5.3_py3.10_env
source /Users/doanvankhoan/esp/esp-idf/export.sh
```

This pin is specific to the current Mac installation. On another machine,
prefer running `./install.sh esp32` so ESP-IDF creates an environment for that
machine's active Python version.

## 5. Start from the repository root

```bash
cd "/Users/doanvankhoan/Documents/mini-edge-runtime 2"
```

All host export and code-generation commands below assume this working
directory.

## 6. Export the `.mer` model

Create the deterministic microcontroller validation fixture:

```bash
python3 examples/export_micro_fixture.py \
  build/models/micro_mlp_f32.mer
```

Expected output:

```text
wrote build/models/micro_mlp_f32.mer (240 bytes)
```

This step runs on the Mac. The `.mer` file is an intermediate host artifact
and is not copied to the ESP32.

## 7. Generate the static C model

```bash
python3 tools/mer_codegen.py \
  build/models/micro_mlp_f32.mer \
  --output-dir micro/generated/tiny_mlp_f32
```

Verify that code generation produced:

```text
micro/generated/tiny_mlp_f32/model.c
micro/generated/tiny_mlp_f32/include/mini_ort_generated_model/model.h
micro/generated/tiny_mlp_f32/model_report.json
```

The generated header declares the model input/output sizes, alignment, arena
size, scratch size, and invocation function. The generated C file stores the
weights as immutable constants and contains the fixed sequence of kernel
calls.

The ESP-IDF component currently points to this exact output directory in:

```text
micro/platform/esp_idf/components/mini_ort_generated_model/CMakeLists.txt
```

If code is generated into a differently named directory, update
`MER_GENERATED_MODEL_DIR` in that component before building firmware.

## 8. Validate the generated model on the Mac

Host validation catches code-generation or numerical problems before a board
is involved:

```bash
cmake -S micro -B build/micro -DCMAKE_BUILD_TYPE=Release
cmake --build build/micro
./build/micro/mini_ort_generated_mlp
```

Expected result:

```text
generated_output=[5.50, 5.50] arena_bytes=16 scratch_bytes=0
```

Do not flash when this result is incorrect. A host/device mismatch is harder
to diagnose than a host-only failure.

## 9. Connect and identify the ESP32

Connect the board with its USB data cable, then list serial devices:

```bash
ls /dev/cu.*
```

A CP2102-based DevKit commonly appears as:

```text
/dev/cu.usbserial-0001
```

Other common names include:

```text
/dev/cu.SLAB_USBtoUART
/dev/cu.wchusbserial-*
/dev/cu.usbmodem*
```

Store the detected port in a task-specific variable:

```bash
export MER_ESP32_PORT=/dev/cu.usbserial-0001
```

Confirm that the board responds:

```bash
esptool --port "$MER_ESP32_PORT" chip-id
```

The tested board reported:

```text
Chip type: ESP32-D0WD-V3 (revision v3.1)
Features: Wi-Fi, BT, Dual Core, 240MHz
Crystal frequency: 40MHz
```

`esptool` resets the board after identification. It does not erase the board
for this command.

## 10. Open the ESP-IDF application

Load ESP-IDF if the current terminal has not loaded it yet:

```bash
export IDF_PYTHON_ENV_PATH=/Users/doanvankhoan/.espressif/python_env/idf5.3_py3.10_env
source /Users/doanvankhoan/esp/esp-idf/export.sh
```

Enter the firmware project:

```bash
cd "/Users/doanvankhoan/Documents/mini-edge-runtime 2/examples/esp32_generated_mlp"
```

The top-level `CMakeLists.txt` adds the project's ESP-IDF component directory.
The application component depends on:

- `mini_ort_generated_model`;
- `mini_ort_micro` through the generated-model component;
- `mini_ort_platform` for the ESP32 memory probe.

## 11. Select the correct target

For the classic ESP32 DevKit:

```bash
idf.py set-target esp32
```

Run this command for the first configuration or when changing from another
chip family. It generates `sdkconfig` and recreates the CMake build files.

Do not use `esp32s3` for a classic ESP32 DevKit. An ESP32-S3 board requires a
separate build configured with:

```bash
idf.py set-target esp32s3
```

## 12. Build the firmware

```bash
idf.py build
```

The build must compile these project-owned units:

```text
main/app_main.c
micro/generated/tiny_mlp_f32/model.c
micro/src/runtime.c
micro/src/tensor_view.c
micro/kernels/reference/fully_connected_f32.c
micro/kernels/reference/relu_f32.c
micro/platform/esp_idf/components/mini_ort_platform/src/esp_memory_probe.c
```

Successful output ends with:

```text
Project build complete.
```

Important generated artifacts:

```text
build/bootloader/bootloader.bin
build/partition_table/partition-table.bin
build/mini_ort_generated_mlp.bin
build/mini_ort_generated_mlp.elf
build/mini_ort_generated_mlp.map
```

The tested firmware produced an application binary of approximately 178 KiB
and left 83 percent of the 1 MiB application partition free.

## 13. Inspect firmware memory and size

```bash
idf.py size
```

The tested build reported approximately:

```text
Flash code:  82,302 bytes
Flash data:  38,812 bytes
IRAM:        51,850 / 131,072 bytes
DRAM:        11,220 / 180,736 bytes
Total image: 181,920 bytes before binary padding
```

For component-level detail:

```bash
idf.py size-components
```

Use these reports to separate firmware overhead from generated model weights
and runtime kernels as the project grows.

## 14. Flash the firmware

Make sure `MER_ESP32_PORT` still names the connected board:

```bash
export MER_ESP32_PORT=/dev/cu.usbserial-0001
```

Flash the bootloader, partition table, and application:

```bash
idf.py -p "$MER_ESP32_PORT" flash
```

The default layout writes:

```text
0x1000  bootloader.bin
0x8000  partition-table.bin
0x10000 mini_ort_generated_mlp.bin
```

Successful flashing includes a verified hash for each image and ends with:

```text
Hard resetting via RTS pin...
Done
```

## 15. Monitor and verify inference

Open the serial monitor:

```bash
idf.py -p "$MER_ESP32_PORT" monitor
```

Press the board's `EN` or `RESET` button if the application output was emitted
before the monitor attached.

The key success line is:

```text
mini_ort_mlp: output=[5.50, 5.50] arena_bytes=16 scratch_bytes=0
```

The memory probe should also print device and heap information around the
invocation. The tested board reported:

```text
target=esp32 revision=301 cores=2 idf=v5.3 cpu_mhz=160
spiram mapped=no bytes=0
before_runtime_init free=305088
after_inference free=304864
```

Press `Ctrl+]` to close the monitor.

## 16. Build, flash, and monitor in one command

After the first successful target configuration, the common development loop
is:

```bash
idf.py -p "$MER_ESP32_PORT" flash monitor
```

`flash` rebuilds changed files before writing the firmware, so a separate
`idf.py build` is optional during normal iteration.

## 17. What happens on the ESP32

`app_main.c` owns the application-level flow:

1. Create the fixed FP32 input `[1.0, 2.0, 3.0]`.
2. Read the generated model's buffer requirements.
3. Initialize `MerMicroContext` with the caller-owned static arena.
4. Invoke `mer_model_invoke_f32()`.
5. Execute generated Fully Connected, ReLU, and Fully Connected calls.
6. Print output and memory statistics.
7. Return from `app_main()`.

The current model uses:

```text
Arena:   16 bytes
Scratch: 0 bytes
```

There is no model parser, filesystem access, dynamic graph construction, C++
runtime, Python interpreter, ONNX dependency, or invocation-time model loading
on the ESP32.

## 18. Deploy a changed compatible MLP

When the `.mer` model changes but remains supported by the current compiler:

```bash
cd "/Users/doanvankhoan/Documents/mini-edge-runtime 2"

python3 tools/mer_codegen.py \
  build/models/micro_mlp_f32.mer \
  --output-dir micro/generated/tiny_mlp_f32

cmake --build build/micro
./build/micro/mini_ort_generated_mlp

export IDF_PYTHON_ENV_PATH=/Users/doanvankhoan/.espressif/python_env/idf5.3_py3.10_env
source /Users/doanvankhoan/esp/esp-idf/export.sh
export MER_ESP32_PORT=/dev/cu.usbserial-0001

cd examples/esp32_generated_mlp
idf.py -p "$MER_ESP32_PORT" flash monitor
```

The current `app_main.c` contains a three-element test input and formats two
output elements. Update that application input and output handling if the new
model contract changes.

The current compiler/runtime vertical slice supports FP32 Linear and ReLU.
Code generation should reject unsupported operators instead of producing
firmware with incomplete behavior.

## 19. Clean and reproducible rebuilds

Use ESP-IDF's clean command when configuration or generated dependencies have
changed significantly:

```bash
idf.py fullclean
idf.py set-target esp32
idf.py build
```

`fullclean` removes generated firmware artifacts, not the model or runtime
source.

If a terminated configuration created a directory that ESP-IDF says is not a
CMake build directory, preserve it for inspection and recreate the build:

```bash
mv build build.failed
idf.py set-target esp32
```

Delete `build.failed` later only after confirming that it contains no source
files.

## 20. Troubleshooting

### `idf.py: command not found`

ESP-IDF has not been loaded into the terminal:

```bash
source ~/esp/esp-idf/export.sh
```

If `~/esp/esp-idf/export.sh` does not exist, install or restore the ESP-IDF
source tree as described in section 4.

### A nonexistent `idf5.3_py3.13_env` is selected

Pin the existing environment on the tested Mac:

```bash
export IDF_PYTHON_ENV_PATH=/Users/doanvankhoan/.espressif/python_env/idf5.3_py3.10_env
source /Users/doanvankhoan/esp/esp-idf/export.sh
```

Alternatively, run `./install.sh esp32` from ESP-IDF using the desired active
Python version to create a new matching environment.

### No `/dev/cu.usb*` device appears

1. Replace the USB cable with a known data cable.
2. Try another USB port or powered hub.
3. Inspect System Information -> USB for CP2102, CH340, or FTDI hardware.
4. Install the vendor driver only when macOS does not already expose a serial
   device.
5. Disconnect and reconnect the board.

### Serial port is busy

Find the process using it:

```bash
lsof "$MER_ESP32_PORT"
```

Close Arduino Serial Monitor, PlatformIO Monitor, another `idf.py monitor`, or
any terminal program holding the port. Only one process can own the serial
device at a time.

### `Connecting...` repeats and flashing fails

Most DevKit boards enter download mode automatically. If automatic reset
fails:

1. Hold the `BOOT` button.
2. Press and release `EN` or `RESET`.
3. Release `BOOT` when writing begins.
4. Retry the flash command.

Also confirm that `idf.py set-target esp32` was used for a classic ESP32.

### Serial output is unreadable

The application uses 115200 baud. Prefer `idf.py monitor`, which reads the
configured baud automatically:

```bash
idf.py -p "$MER_ESP32_PORT" monitor
```

A short burst of unreadable characters while reset changes UART state is not
an inference failure. Press `EN` and inspect the subsequent boot log.

### `model.h` is missing

Regenerate the model from the repository root:

```bash
python3 tools/mer_codegen.py \
  build/models/micro_mlp_f32.mer \
  --output-dir micro/generated/tiny_mlp_f32
```

### Generated model symbols are not linked

Check that this component still points to the generated directory:

```text
micro/platform/esp_idf/components/mini_ort_generated_model/CMakeLists.txt
```

Then run a clean ESP-IDF build.

### Device output differs from host output

1. Run `./build/micro/mini_ort_generated_mlp` on the host.
2. Confirm that host output is `[5.50, 5.50]`.
3. Confirm the firmware rebuilt after `model.c` changed.
4. Confirm `app_main.c` uses `[1.0, 2.0, 3.0]`.
5. Run `idf.py fullclean`, rebuild, and flash again.

### Flash-size warning: detected 4 MB but image header uses 2 MB

The tested DevKit physically has 4 MB flash, while the current default image
uses a conservative 2 MB header. This is safe for the current partition table
and does not prevent inference.

To use the full physical flash for a future larger partition table:

```bash
idf.py menuconfig
```

Select `Serial flasher config -> Flash size -> 4 MB`, save, rebuild, and flash.
Do not select 4 MB without confirming the physical device size.

### `ninja: warning: premature end of file; recovering`

Ninja can recover its generated dependency database automatically. If it
continues on every build, use a clean generated build:

```bash
idf.py fullclean
idf.py build
```

This warning does not indicate a corrupted generated model when the final
image builds, flashes with verified hashes, and produces the expected output.

## 21. Success checklist

The complete pipeline is successful only when all items pass:

- [ ] `micro_mlp_f32.mer` is exported successfully.
- [ ] `model.c`, `model.h`, and `model_report.json` are generated.
- [ ] Host generated-model inference prints `[5.50, 5.50]`.
- [ ] `esptool` identifies a classic ESP32.
- [ ] `idf.py set-target esp32` completes.
- [ ] `idf.py build` ends with `Project build complete`.
- [ ] Flash writes and verifies all three images.
- [ ] The ESP32 boots the `mini_ort_generated_mlp` application.
- [ ] UART prints `output=[5.50, 5.50]`.
- [ ] Arena and scratch requirements fit the generated static buffers.
- [ ] Memory headroom is recorded for later model growth.

## 22. Short repeat-run reference

After one complete successful setup:

```bash
cd "/Users/doanvankhoan/Documents/mini-edge-runtime 2"

python3 examples/export_micro_fixture.py build/models/micro_mlp_f32.mer
python3 tools/mer_codegen.py \
  build/models/micro_mlp_f32.mer \
  --output-dir micro/generated/tiny_mlp_f32

cmake -S micro -B build/micro -DCMAKE_BUILD_TYPE=Release
cmake --build build/micro
./build/micro/mini_ort_generated_mlp

export IDF_PYTHON_ENV_PATH=/Users/doanvankhoan/.espressif/python_env/idf5.3_py3.10_env
source /Users/doanvankhoan/esp/esp-idf/export.sh
export MER_ESP32_PORT=/dev/cu.usbserial-0001

cd examples/esp32_generated_mlp
idf.py -p "$MER_ESP32_PORT" flash monitor
```

Expected final line:

```text
mini_ort_mlp: output=[5.50, 5.50] arena_bytes=16 scratch_bytes=0
```
