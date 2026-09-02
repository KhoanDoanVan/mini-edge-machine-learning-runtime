# iOS Build Pipeline

## Flow

```text
Python model
  -> tiny_mlp.mer
  -> MiniOrt.xcframework
  -> Swift application
  -> C API
  -> C++ runtime
  -> CPU inference
```

## Requirements

- Python 3
- CMake
- Xcode with an iOS SDK and Simulator runtime

Run commands from the project root.

## 1. Verify the native runtime

```bash
cmake -S cpp -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --parallel
PYTHONPATH=python:. python3 examples/mlp.py
```

Expected output:

```text
output=[4.75, 4.5]
```

## 2. Export the model

```bash
PYTHONPATH=python:. python3 -c \
  'from examples.mlp import build_model; build_model().save("build/models/tiny_mlp.mer")'
```

## 3. Build the XCFramework

```bash
ios/scripts/build_xcframework.sh
```

Output:

```text
build/ios/MiniOrt.xcframework
```

## 4. Build the iOS application

```bash
ios/scripts/build_demo.sh
```

## 5. Run on a Simulator

Boot an iPhone Simulator, then run:

```bash
ios/scripts/run_demo_simulator.sh
```

Alternatively:

```bash
open ios/MiniOrtDemo/MiniOrtDemo.xcodeproj
```

Select an iPhone Simulator and press Run.

Expected inference result:

```text
Input:  [1.0, 2.0, 3.0]
Output: [4.75, 4.50]
```

## 6. Run on a physical iPhone

1. Open the Xcode project.
2. Select an Apple development team under Signing & Capabilities.
3. Select the connected iPhone.
4. Press Run.

## Rebuild after changing the model

Edit `build_model()` in `examples/mlp.py`, then run:

```bash
ios/scripts/build_demo.sh
```

Update the Swift input in `ios/MiniOrtDemo/Sources/ContentView.swift` when the model input size changes.
