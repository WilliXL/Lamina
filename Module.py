from DataTypes import DataType
from Tensor import Tensor
import numpy as np


class Module:
    def parameters(self):
        return []

    def __call__(self):
        raise NotImplementedError

    def compile(
        self, input_shapes: list[tuple[int]], dtype: DataType = DataType.bfloat16
    ):
        fake_inputs = [
            Tensor(data=np.zeros(shape), dtype=dtype) for shape in input_shapes
        ]
        output = self(*fake_inputs)
        return output


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Tensor(
            np.random.randn(in_features, out_features), dtype=DataType.bfloat16
        )
        if bias:
            self.bias = Tensor(np.zeros(out_features), dtype=DataType.bfloat16)
        else:
            self.bias = None

    def __call__(self, x: Tensor):
        if self.bias is not None:
            return (x @ self.weight) + self.bias
        else:
            return x @ self.weight

    def parameters(self):
        return [self.weight, self.bias]

    def __repr__(self):
        return (
            f"Linear(in_features={self.in_features}, out_features={self.out_features})"
        )
