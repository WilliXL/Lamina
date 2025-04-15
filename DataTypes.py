from enum import Enum
import numpy as np


class DataType(Enum):
    """Maps between "native" Tensor type and the numpy representation that will be used when doing eager calculations
    on the CPU
    """

    bfloat16 = np.float32
    float32 = np.float32


def resolve_datatype(datatype_1: DataType, datatype_2: DataType) -> DataType:
    if datatype_1 == datatype_2:
        return datatype_1

    if DataType.float32 in (datatype_1, datatype_2):
        return DataType.float32
    else:
        return DataType.bfloat16


def sizeof(datatype: DataType) -> int:
    if datatype == DataType.float32:
        return 4
    else:
        return 2
