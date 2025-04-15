import numpy as np
from typing import Union, Sequence, Optional, List, Tuple
from DataTypes import DataType, resolve_datatype

# Recursive type for nested lists (N-D arrays)
NestedList = Union[float, int, bool, Sequence["NestedList"]]
TensorInput = Union[NestedList, np.ndarray]


class Tensor:
    def __init__(
        self,
        data: Optional[TensorInput] = None,
        dtype: DataType = DataType.bfloat16,
        _children: Optional[List["Tensor"]] = None,
        _operator: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ):
        """
        In normal mode, data must be provided and is converted to a NumPy array.
        In compile-only mode, data is ignored and shape must be provided.
        """
        self.datatype = dtype
        self._children = _children if _children is not None else []
        self._operator = _operator

        self.data = np.array(data, dtype=dtype.value) if data is not None else None
        self.shape = tuple(self.data.shape) if self.data is not None else shape

    def __repr__(self):
        data_repr = self.data.tolist() if self.data is not None else "None"
        return f"Tensor({data_repr}, dtype={self.datatype.name}, shape={self.shape})"

    # Operator overloads:
    def __add__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise TypeError("Tensors can only operate with other tensors")

        return Tensor(
            data=self.data + other.data,
            dtype=resolve_datatype(self.datatype, other.datatype),
            _children=[self, other],
            _operator="+",
        )

    def __mul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise TypeError("Tensors can only operate with other tensors")

        return Tensor(
            data=self.data * other.data,
            dtype=resolve_datatype(self.datatype, other.datatype),
            _children=[self, other],
            _operator="*",
        )

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise TypeError("Tensors can only operate with other tensors")

        return Tensor(
            data=self.data @ other.data,
            dtype=resolve_datatype(self.datatype, other.datatype),
            _children=[self, other],
            _operator="@",
        )

    def __truediv__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise TypeError("Tensors can only operate with other tensors")

        return Tensor(
            data=self.data / other.data,
            dtype=resolve_datatype(self.datatype, other.datatype),
            _children=[self, other],
            _operator="/",
        )

    def __neg__(self) -> "Tensor":
        return Tensor(
            data=-self.data,
            dtype=self.datatype,
            _children=[self],
            _operator="neg",
        )


# -- End of Tensor definition --


# Now, define a base Module to mimic a container for layers
class Module:
    def __init__(self):
        self._modules = {}  # to hold child modules

    def parameters(self) -> List[Tensor]:
        """Recursively collect parameters (Tensors) from modules."""
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor):
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Tensor):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def __call__(self):
        raise NotImplementedError

    def compile(
        self, input_shapes: List[Tuple[int, ...]], dtype: DataType = DataType.bfloat16
    ) -> Tensor:
        """
        Simulate a forward pass with fake Tensors (compile-only mode) to generate an execution graph.
        input_shapes: list of tuples defining the shape of each input.
        """
        fake_inputs = [
            Tensor(
                data=np.zeros(shape),
                dtype=dtype,
                _children=[],
                _operator=None,
                compile_only=True,
                shape=shape,
            )
            for shape in input_shapes
        ]
        output = self(*fake_inputs)
        compile(output)


def compile(final_tensor: Tensor):
    breakpoint()
    print(final_tensor)


# Define a Linear layer as a Module
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Tensor(
            np.random.randn(in_features, out_features), dtype=DataType.bfloat16
        )
        if bias:
            self.bias = Tensor(np.zeros(out_features), dtype=DataType.bfloat16)
        else:
            self.bias = None

    def __call__(self, x: Tensor) -> Tensor:
        if self.bias is not None:
            return (x @ self.weight) + self.bias
        else:
            return x @ self.weight

    def parameters(self) -> List[Tensor]:
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, out_features={self.out_features})"
        )


# Define a ReLU activation as a functional module
def relu(x: Tensor) -> Tensor:
    if x.compile_only:
        # In compile mode, we assume shape remains the same
        return Tensor(
            data=None,
            dtype=x.datatype,
            _children=[x],
            _operator="ReLU",
            compile_only=True,
            shape=x.shape,
        )
    else:
        return Tensor(
            np.maximum(x.data, 0), dtype=x.datatype, _children=[x], _operator="ReLU"
        )


# Define a basic MLP using the above layers
class MLP(Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc3 = Linear(hidden_dim, output_dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --- Example usage ---

if __name__ == "__main__":
    # Create a sample MLP with dimensions: input 10, hidden 20, output 5
    mlp = MLP(input_dim=1000, hidden_dim=4096, output_dim=32)

    # For actual inference, you would pass a real tensor:
    real_input = Tensor(np.random.randn(1, 1000), dtype=DataType.bfloat16)
    output = mlp(real_input)
    print("Real output:", output)

    # For compile (graph tracing) mode, assume batch size 1 and input shape (1, 10)
    traced_output = mlp.compile(input_shapes=[(1, 1000)], dtype=DataType.bfloat16)
