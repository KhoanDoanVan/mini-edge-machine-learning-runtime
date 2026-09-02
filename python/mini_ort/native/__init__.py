"""Native C++ backend exposed through the stable C ABI."""

"""build/native is the directory where CMake places generated build files for the native C++
  runtime. """

from .backend import NativeBackend

__all__ = ["NativeBackend"]