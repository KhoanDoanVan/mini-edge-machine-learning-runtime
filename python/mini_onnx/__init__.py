"""Small, dependency-free model IR inspired by ONNX's archiecture"""

from .checker import ModelValidationError, check_model
from .codec import load_model, save_model
from .ir import DType, Graph, Model, Node, TensorProto, ValueInfo

__all__ = [
    "DType",
    "Graph",
    "Model",
    "ModelValidationError",
    "Node",
    "TensorProto",
    "ValueInfo",
    "check_model",
    "load_model",
    "save_model"
]