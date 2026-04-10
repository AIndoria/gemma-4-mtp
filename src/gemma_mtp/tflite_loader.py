from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import tflite


_TENSOR_TYPE_TO_DTYPE = {
    tflite.TensorType.FLOAT32: np.float32,
    tflite.TensorType.FLOAT16: np.float16,
    tflite.TensorType.INT32: np.int32,
    tflite.TensorType.INT16: np.int16,
    tflite.TensorType.INT8: np.int8,
    tflite.TensorType.UINT8: np.uint8,
    tflite.TensorType.BOOL: np.bool_,
}


@dataclass(frozen=True)
class QuantizationInfo:
    scale: np.ndarray
    zero_point: np.ndarray
    quantized_dimension: int


@dataclass(frozen=True)
class TensorInfo:
    index: int
    name: str
    shape: tuple[int, ...]
    tensor_type: int
    buffer_index: int
    quantization: QuantizationInfo | None


def _shape_of(tensor: Any) -> tuple[int, ...]:
    return tuple(int(tensor.Shape(i)) for i in range(tensor.ShapeLength()))


def _quantization_of(tensor: Any) -> QuantizationInfo | None:
    quant = tensor.Quantization()
    if quant is None:
        return None

    scale = np.array([quant.Scale(i) for i in range(quant.ScaleLength())], dtype=np.float32)
    zero_point = np.array(
        [quant.ZeroPoint(i) for i in range(quant.ZeroPointLength())], dtype=np.int64
    )
    if scale.size == 0 and zero_point.size == 0:
        return None

    return QuantizationInfo(
        scale=scale,
        zero_point=zero_point,
        quantized_dimension=int(quant.QuantizedDimension()),
    )


def _unpack_int4(buffer: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    flat = buffer.astype(np.uint8, copy=False).reshape(-1)
    low = flat & 0x0F
    high = flat >> 4

    packed = np.empty(flat.size * 2, dtype=np.int8)
    packed[0::2] = low.astype(np.int8)
    packed[1::2] = high.astype(np.int8)

    packed = np.where(packed >= 8, packed - 16, packed).astype(np.int8)
    needed = int(np.prod(shape))
    return packed[:needed].reshape(shape)


class TFLiteModelReader:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.buffer = self.model_path.read_bytes()
        self.model = tflite.Model.GetRootAsModel(self.buffer, 0)
        self.main_subgraph = self.model.Subgraphs(0)

    @cached_property
    def name_to_index(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for index in range(self.main_subgraph.TensorsLength()):
            mapping[self.main_subgraph.Tensors(index).Name().decode()] = index
        return mapping

    def tensor_info(self, name_or_index: str | int) -> TensorInfo:
        if isinstance(name_or_index, str):
            index = self.name_to_index[name_or_index]
        else:
            index = int(name_or_index)

        tensor = self.main_subgraph.Tensors(index)
        return TensorInfo(
            index=index,
            name=tensor.Name().decode(),
            shape=_shape_of(tensor),
            tensor_type=int(tensor.Type()),
            buffer_index=int(tensor.Buffer()),
            quantization=_quantization_of(tensor),
        )

    def _tensor(self, name_or_index: str | int) -> Any:
        if isinstance(name_or_index, str):
            name_or_index = self.name_to_index[name_or_index]
        return self.main_subgraph.Tensors(int(name_or_index))

    def read_raw(self, name_or_index: str | int) -> np.ndarray:
        tensor = self._tensor(name_or_index)
        info = self.tensor_info(int(self.name_to_index[tensor.Name().decode()]) if isinstance(name_or_index, str) else name_or_index)
        buffer = self.model.Buffers(info.buffer_index).DataAsNumpy()

        if info.tensor_type == tflite.TensorType.INT4:
            return _unpack_int4(buffer, info.shape)

        dtype = _TENSOR_TYPE_TO_DTYPE.get(info.tensor_type)
        if dtype is None:
            raise NotImplementedError(f"Unsupported tensor type: {info.tensor_type}")

        raw = np.frombuffer(buffer.tobytes(), dtype=dtype)
        return raw.reshape(info.shape)

    def read_dequantized(self, name_or_index: str | int) -> np.ndarray:
        info = self.tensor_info(name_or_index)
        raw = self.read_raw(name_or_index)
        quant = info.quantization
        if quant is None:
            return raw.astype(np.float32, copy=False)

        if quant.scale.size == 1:
            zero = quant.zero_point[0] if quant.zero_point.size else 0
            return (raw.astype(np.float32) - float(zero)) * float(quant.scale[0])

        axis = quant.quantized_dimension
        reshape = [1] * raw.ndim
        reshape[axis] = quant.scale.size
        scale = quant.scale.reshape(reshape).astype(np.float32)
        if quant.zero_point.size:
            zero_point = quant.zero_point.reshape(reshape).astype(np.float32)
        else:
            zero_point = 0.0
        return (raw.astype(np.float32) - zero_point) * scale
