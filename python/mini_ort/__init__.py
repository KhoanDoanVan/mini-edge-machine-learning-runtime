"""Small inference runtime for mini_onnx models."""

from .session import InferenceSession
from .tensor import Tensor

__all__ = ["InferenceSession", "Tensor"]