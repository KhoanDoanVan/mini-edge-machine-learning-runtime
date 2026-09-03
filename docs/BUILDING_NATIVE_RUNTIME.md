# Building the Native MiniOrt Runtime

This project has a Python API and a C++ inference backend. Python does not
compile or execute the C++ source files directly. The C++ backend must first be
compiled into a dynamic library, which Python loads through `ctypes`.

## Source and build directories

```text
cpp/                         C++ source and headers
python/                      Python API and FFI bindings
build/native/                Generated CMake files and native build output
build/native/libmini_ort.dylib  macOS dynamic library loaded by Python
```

The `build/` directory is generated output. It should not be committed to Git;
the repository `.gitignore` excludes it. It is safe to remove and regenerate.

## CMake has two phases

### Configure

Run this from the repository root:

```bash
cmake -S cpp -B build/native
```

The -S option selects the source directory. The `-B` option selects where
CMake writes generated build files. Configuration detects the compiler and
creates the build system; it does not compile the runtime yet.

### Build

```bash
cmake --build build/native --target mini_ort
```

The build phase compiles each `.cc` file into an object file and links those
objects into `libmini_ort.dylib`. A successful build ends with:

```text
[100%] Built target mini_ort
```

Then verify the library exists:

```bash
ls -l build/native/libmini_ort.dylib
```

## Run the Python example

After the native target builds successfully:

```bash
PYTHONPATH=python:. python3 examples/mlp.py
```

The Python FFI searches `build/native` automatically. If the library is in a
different location, set it explicitly:

```bash
MINI_ORT_LIBRARY=/absolute/path/to/libmini_ort.dylib \
PYTHONPATH=python:. python3 examples/mlp.py
```

## Why run commands from the project root?

The commands use relative paths such as `cpp`, `build/native`, and `python`.
Running them from `mini-edge-machine-learning-runtime/` ensures those paths
resolve correctly. The build directory could technically be elsewhere, but
the Python loader expects `build/native` unless `MINI_ORT_LIBRARY` is set.

## Common failures

| Message | Meaning | Action |
|---|---|---|
| `native mini_ort library was not found` | The C++ library has not been built or is not in the expected path. | Run the native build or set `MINI_ORT_LIBRARY`. |
| `Cannot find source file` during configure | `CMakeLists.txt` references a missing C++ source file. | Remove the stale target or restore the missing source. |
| `error:` while compiling a `.cc` file | A C++ source or header error stopped compilation. | Fix the reported source error, then rerun the build. |
| `Undefined symbols` while linking | A function is declared or called but has no linked definition. | Add the definition or include its source file in `CMakeLists.txt`. |

Configuration only needs to be rerun when CMake inputs change. After ordinary
C++ source edits, rerunning `cmake --build build/native --target mini_ort` is
normally sufficient.
