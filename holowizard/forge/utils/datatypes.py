# standard libraries
from typing import TypeVar, Tuple
from pathlib import Path

# third party libraries
import torch
import numpy as np

# local libraries


__all__ = [
    "TensorComplex128",
    "TensorComplex64",
    "TensorFloat32",
    "Tensorlike",
    "Hologram",
    "Numeric",
    "Pathlike",
    "is_pathlike",
    "Range",
]


T = TypeVar("T", int, float)

TensorComplex128 = torch.Tensor
TensorComplex64 = torch.Tensor
TensorFloat32 = torch.Tensor
Tensorlike = torch.Tensor | np.ndarray

Hologram = TensorFloat32
Numeric = int | float
Range = Tuple[T, T]
Pathlike = str | Path
is_pathlike = lambda x: isinstance(x, Pathlike)
