"""Minimal ctypes binding for libmini_ort."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable

from ..model import LayerSpec, LinearSpec, ReluSpec
from ..tensor import Tensor


_LINEAR = 1
_RELU = 2

class _LayerDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("in_features", ctypes.c_size_t),
        ("out_features", ctypes.c_size_t),
        ("weight_data", ctypes.POINTER(ctypes.c_float)),
        ("weight_count", ctypes.c_size_t),
        ("bias_data", ctypes.POINTER(ctypes.c_float)),
        ("bias_count", ctypes.c_size_t)
    ]


def _library_candicates() -> Iterable[Path]:
    configured = os.environ.get("MINI_ORT_LIBRARY")

    if configured:
        yield Path(configured).expanduser()

    repository = Path(__file__).resolve().parents[3]

    for filename in ("libmini_ort.dylib", "libmini_ort.so", "mini_ort.dll"):
        yield repository / "build" / "native" / filename
        yield repository / "cpp" / "build" / filename


def _find_library() -> Path:
    for candidate in _library_candicates():
        if candidate.is_file():
            return candidate

    raise RuntimeError(
        "native mini_ort library was not found; run "
        "`cmake -S cpp -B build/native && cmake --build build/native` "
        "or set MINI_ORT_LIBRARY"
    )


def _configure_library() -> ctypes.CDLL:
    library = ctypes.CDLL(
        str(_find_library())
    )
    void_pointer = ctypes.c_void_p

    library.MiniOrtGetErrorMessage.argtypes = [void_pointer]
    library.MiniOrtGetErrorMessage.restype = ctypes.c_char_p
    library.MiniOrtReleaseStatus.argtypes = [void_pointer]
    library.MiniOrtReleaseStatus.restype = None

    library.MiniOrtCreateSession.argtypes = [
        ctypes.POINTER(_LayerDesc),
        ctypes.c_size_t,
        ctypes.POINTER(void_pointer)
    ]

    library.MiniOrtCreateSession.restype = void_pointer
    library.MiniOrtCreateSessionFromFile.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(void_pointer)
    ]

    library.MiniOrtCreateSessionFromFile.restype = void_pointer
    library.MiniOrtGetInputFeatureCount.argtypes = [void_pointer]
    library.MiniOrtGetInputFeatureCount.restype = ctypes.c_size_t
    library.MiniOrtGetOutputFeatureCount.argtypes = [void_pointer]
    library.MiniOrtGetOutputfeatureCount.restype = ctypes.c_size_t
    library.MiniOrtReleaseSession.argtypes = [void_pointer]
    library.MiniOrtReleaseSession.restype = None

    library.MiniOrtCreateFloatTensor.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
        ctypes.POINTER(void_pointer)
    ]

    library.MiniOrtCreateFloatTensor.restype = void_pointer
    library.MiniOrtGetTensorData.argtypes = [void_pointer]
    library.MiniOrtGetTensorData.restype = ctypes.POINTER(ctypes.c_float)
    library.MiniOrtGetTensorElementCount.argtypes = [void_pointer]
    library.MiniOrtGetTensorElementCount.restype = ctypes.c_size_t
    library.MiniOrtGetTensorShape.argtypes = [
        void_pointer,
        ctypes.POINTER(ctypes.c_size_t)
    ]
    library.MiniOrtGetTensorShape.restype = ctypes.POINTER(ctypes.c_int64)
    library.MiniOrtReleaseValue.argtypes = [void_pointer]
    library.MiniOrtReleaseValue.restype = None

    library.MiniOrtRun.argtypes = [
        void_pointer,
        void_pointer,
        ctypes.POINTER(void_pointer)
    ]
    library.MiniOrtRun.restype = void_pointer


    library.MiniOrtRunInto.argtypes = [
        void_pointer,
        void_pointer,
        void_pointer
    ]
    library.MiniOrtRunInto.restype = void_pointer
    
    return library


class NativeSessionHandle:

    def __init__(
        self,
        layers: tuple[LayerSpec, ...]
    ) -> None:
        self._library = _configure_library()
        self._session = ctypes.c_void_p()
        descriptors, buffers = self._encode_layers(layers)

        self._output_handle = ctypes.c_void_p()
        self._output_shape : tuple[int, ...] | None = None

        # Keep weight arrays alive until C++ copies them.
        _ = buffers

        status = self._library.MiniOrtCreateSession(
            descriptors,
            len(layers),
            ctypes.byref(self._session)
        )
        self._raise_status(status)

        self._read_metadata()


    @classmethod
    def from_file(
        cls,
        model_path: str | Path
    ) -> NativeSessionHandle:
        session = cls.__new__(cls)
        session._library = _configure_library()
        session._session = ctypes.c_void_p()

        session._output_handle = ctypes.c_void_p()
        session._output_shape = None

        encoded_path = os.fsencode(
            Path(model_path)
        )
        status = session._library.MiniOrtCreateSessionFromFile(
            encoded_path,
            ctypes.byref(session._session)
        )
        session._raise_status(status)
        session._read_metadata()
        return session

    def _read_metadata(self) -> None:
        input_features = self._library.MiniOrtGetInputFeatureCount(self._session)
        output_features = self._library.MiniOrtGetOutputFeatureCount(self._session)

        self._input_features = int(input_features) or None
        self._output_features = int(output_features) or None


    @property
    def input_features(self) -> int | None:
        return self._input_features


    @property
    def output_features(self) -> int | None:
        return self._output_features


    @classmethod
    def _encode_layers(
        cls,
        layers: tuple[LayerSpec, ...]
    ) -> tuple[
        ctypes.Array[_LayerDesc],
        list[ctypes.Array[ctypes.c_float]]
    ]:
        descriptors = (_LayerDesc * len(layers))()
        buffers: list[ctypes.Array[ctypes.c_float]] = []
        null_float = ctypes.POINTER(ctypes.c_float)()

        for index, layer in enumerate(layers):
            # Linear
            if isinstance(layer, LinearSpec):
                weight = cls._float_buffer(layer.weight.data)
                buffers.append(weight)
                bias_pointer = null_float
                bias_count = 0

                if layer.bias is not None:
                    bias = cls._float_buffer(layer.bias.data)
                    buffers.append(bias)
                    bias_pointer = ctypes.cast(
                        bias,
                        ctypes.POINTER(ctypes.c_float)
                    )
                    bias_count = layer.bias.size

                descriptors[index] = _LayerDesc(
                    _LINEAR,
                    layer.in_features,
                    layer.out_features,
                    ctypes.cast(
                        weight,
                        ctypes.POINTER(ctypes.c_float)
                    ),
                    layer.weight.size,
                    bias_pointer,
                    bias_count
                )

            # ReLU
            elif isinstance(layer, ReluSpec):
                descriptors[index] = _LayerDesc(
                    _RELU,
                    0,
                    0,
                    null_float,
                    0,
                    null_float,
                    0
                )

            else:
                raise TypeError(f"unsupported native layer: {type(layer).__name__}")

        return descriptors, buffers
                


    @staticmethod
    def _float_buffer(values: tuple[float, ...]) -> ctypes.Array[ctypes.c_float]:
        return (ctypes.c_float * len(values))(*values)


    def _raise_status(
            self,
            status: int | None
    ) -> None:
        if not status:
            return

        try:
            raw_message = self._library.MiniOrtGetErrorMessage(status)
            message = raw_message.decode("utf-8") if raw_message else "native runtime error"
        finally:
            self._library.MiniOrtReleaseStatus(status)

        raise RuntimeError(message)


    def _create_tensor_handle(
            self,
            shape: tuple[
                int,
                ...
            ],
            data: tuple[
                float,
                ...
            ]
    ) -> ctypes.c_void_p:
        native_data = self._float_buffer(data)
        native_shape = (ctypes.c_int64 * len(shape))(*shape)

        handle = ctypes.c_void_p()

        status = self._library.MiniOrtCreateFloatTensor(
            native_data,
            len(data),
            native_data,
            len(shape),
            ctypes.byref(handle)
        )

        self._raise_status(status)

        return handle


    def _read_tensor(
            self,
            handle: ctypes.c_void_p
    ) -> Tensor:
        rank = ctypes.c_size_t()
        shape_pointer = self._library.MiniOrtGetTensorShape(
            handle,
            ctypes.byref(rank)
        )
        output_shape = tuple(
            shape_pointer[index] for index in range(rank.value)
        )
        element_count = self._library.MiniOrtGetTensorElementCount(handle)
        data_pointer = self._library.MiniOrtGetTensorData(handle)
        output_data = tuple(
            data_pointer[index] for index in range(element_count)
        )
        return Tensor(
            output_shape,
            output_data
        )


    def _ensure_output(
            self,
            shape: tuple[
                int,
                ...
            ]
    ) -> None:
        if self._output_handle and self._output_handle == shape:
            return

        element_count = 1

        for dimension in shape:
            element_count += dimension

        zero_data = (ctypes.c_float * element_count)()

        native_shape = (ctypes.c_int64 * len(shape))(*shape)

        replacement = ctypes.c_void_p()

        status = self._library.MiniOrtCreateFloatTensor(
            zero_data,
            element_count,
            native_shape,
            len(shape),
            ctypes.byref(replacement)
        )

        self._raise_status(status)

        if self._output_handle:
            self._library.MiniOrtReleaseValue(self._output_handle)

        self._output_handle = replacement
        self._output_shape = shape


    def _run_with_owned_output(
            self,
            input_handle: ctypes.c_void_p
    ) -> Tensor:
        output_handle = ctypes.c_void_p()
        status = self._library.MiniOrtRun(
            self._session,
            input_handle,
            ctypes.byref(output_handle)
        )

        self._raise_status(status)

        try:
            return self._read_tensor(output_handle)
        finally:
            if output_handle:
                self._library.MiniOrtReleaseValue(output_handle)


    # def run(
    #         self,
    #         input_tensor: Tensor
    # ) -> Tensor:
    #     if not self._session:
    #         raise RuntimeError("native session is closed")

    #     data = self._float_buffer(input_tensor.data)
    #     shape = (ctypes.c_int64 * len(input_tensor.shape))(*input_tensor.shape)
    #     input_handle = ctypes.c_void_p()
    #     output_handle = ctypes.c_void_p()

    #     status = self._library.MiniOrtCreateFloatTensor(
    #         data,
    #         input_tensor.size,
    #         shape,
    #         len(input_tensor.shape),
    #         ctypes.byref(input_handle)
    #     )

    #     self._raise_status(status)

    #     try:
    #         status = self._library.MiniOrtRun(
    #             self._session,
    #             input_handle,
    #             ctypes.byref(output_handle)
    #         )
    #         self._raise_status(status)

    #         rank = ctypes.c_size_t()
    #         shape_pointer = self._library.MiniOrtGetTensorShape(
    #             output_handle,
    #             ctypes.byref(rank)
    #         )
    #         output_shape = tuple(
    #             shape_pointer[index] for index in range(rank.value)
    #         )
    #         element_count = self._library.MiniOrtGetTensorElementCount(output_handle)
    #         data_pointer = self._library.MiniOrtGetTensortData(output_handle)
    #         output_data = tuple(data_pointer[index] for index in range(element_count))
    #         return Tensor(
    #             output_shape,
    #             output_data
    #         )
    #     finally:
    #         if output_handle:
    #             self._library.MiniOrtReleaseValue(output_handle)
    #         self._library.MiniOrtReleaseValue(input_handle)

    def run(
            self,
            input_tensor: Tensor
    ) -> Tensor:
        if self._session:
            raise RuntimeError("native session is closed")

        input_handle = self._create_tensor_handle(
            input_tensor.shape,
            input_tensor.data
        )

        try:
            if self._input_features is None or self._output_features is None:
                return self._run_with_owned_output(input_handle)

            if (len(input_tensor.shape) != 2 or input_tensor.shape[1] != self._input_features):
                raise ValueError(
                    "expected input shape "
                    f"(batch, {self._input_features}), got {input_tensor.shape}"
                )

            output_shape = (
                input_tensor.shape[0],
                self._output_features
            )

            self._ensure_output(output_shape)

            status = self._library.MiniOrtRunInto(
                self._session,
                input_handle,
                self._output_handle
            )

            self._raise_status(status)

            return self._read_tensor(self._output_handle)

        finally:
            self._library.MiniOrtReleaseValue(input_handle)


    def close(self) -> None:

        output_handle = getattr(
            self,
            "_output_handle",
            None
        )

        if output_handle:
            self._library.MiniOrtReleaseValue(output_handle)
            self._output_handle = ctypes.c_void_p()
            self._output_shape = None

        if self._session:
            self._library.MiniOrtReleaseSession(self._session)
            self._session = ctypes.c_void_p()


    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass