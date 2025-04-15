import numpy as np
from typing import Union, Sequence, Optional
from DataTypes import DataType, resolve_datatype


NestedList = Union[float, int, bool, Sequence["NestedList"]]
TensorInput = Union[NestedList, np.ndarray]


class Tensor:
    def __init__(
        self,
        data: TensorInput,
        dtype: DataType = DataType.bfloat16,
        _children: Optional[list["Tensor"]] = None,
        _operator: Optional[str] = None,
    ):
        # for now just utilize numpy liberally for debugging purposes
        self.data = np.array(data, dtype=dtype.value)
        self.shape = tuple(self.data.shape)

        self.datatype = dtype

        self._children = _children
        self._operator = _operator

    def __repr__(self):
        return f"Tensor({self.data.tolist()}, dtype={self.datatype.name}, shape={self.shape})"

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Tensors can only operate with other tensors")

        result = Tensor(
            self.data + other.data,
            dtype=resolve_datatype(self.datatype, other.datatype),
            _children=[self, other],
            _operator="+",
        )

        return result

    def __neg__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Tensors can only operate with other tensors")

        result = Tensor(
            self.data - other.data,
            dtype=resolve_datatype(self.datatype, other.datatype),
            _children=[self, other],
            _operator="-",
        )

        return result

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Tensors can only operate with other tensors")

        result = Tensor(
            self.data * other.data,
            dtype=resolve_datatype(self.datatype, other.datatype),
            _children=[self, other],
            _operator="*",
        )

        return result

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Tensors can only operate with other tensors")

        result = Tensor(
            self.data / other.data,
            dtype=resolve_datatype(self.datatype, other.datatype),
            _children=[self, other],
            _operator="/",
        )

        return result

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Tensors can only operate with other tensors")

        result = Tensor(
            self.data @ other.data,
            dtype=resolve_datatype(self.datatype, other.datatype),
            _children=[self, other],
            _operator="@",
        )

        return result
