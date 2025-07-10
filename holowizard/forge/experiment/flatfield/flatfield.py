# standard libraries
from pathlib import Path
from typing import Tuple, List
from abc import ABC, abstractmethod

# third party libraries
import torch
import numpy as np

# local libraries
from holowizard.forge.utils.torch_settings import get_torch_device
from holowizard.forge.utils.datatypes import Pathlike, TensorFloat32, Tensorlike
from holowizard.forge.utils import fileIO


__all__ = [
    "BaseFlatField",
    "FlatField",
]


class BaseFlatField(ABC):
    @property
    @abstractmethod
    def flatfield(self) -> TensorFloat32:
        """2D tensor."""
        pass

    @property
    @abstractmethod
    def condition(self) -> TensorFloat32:
        """3D Tensor with flat-fields (2D Tensors) used as conditions NN training."""
        pass

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.flatfield.shape

    @property
    def size(self) -> int:
        """Size of the first dimension, often representing height."""
        return self.shape[0]


class FlatField(BaseFlatField):
    def __init__(self, flatfield: Tensorlike | Pathlike) -> None:
        match type(flatfield):
            case t if t in [str, Path]:
                flatfield = torch.Tensor(fileIO.load_img(flatfield))
            case torch.Tensor:
                pass
            case np.ndarray:
                flatfield = torch.tensor(flatfield)
            case _:
                raise ValueError(f"Ivalid input for flatfield. Type: {type(flatfield)}")

        self.flatfield = flatfield.to(device=get_torch_device())
        self.condition = None

    def __add__(self, other):
        if isinstance(other, FlatField):
            new_flatfield = self.flatfield + other.flatfield
            return FlatField(new_flatfield)
        elif isinstance(other, TensorFloat32):
            return self.flatfield + other
        else:
            raise ValueError("Unsupported operand type for '+'.")

    def __iadd__(self, other):
        if isinstance(other, FlatField):
            self.flatfield += other.flatfield
            return self
        elif isinstance(other, TensorFloat32):
            self.flatfield += other
            return self
        else:
            raise ValueError("Unsupported operand type for '+='.")

    def __mul__(self, other):
        if isinstance(other, FlatField):
            new_flatfield = self.flatfield * other.flatfield
            return FlatField(new_flatfield)
        elif isinstance(other, TensorFloat32):
            return self.flatfield * other
        else:
            raise ValueError("Unsupported operand type for '*'.")

    def __imul__(self, other):
        if isinstance(other, FlatField):
            self.flatfield *= other.flatfield
            return self
        elif isinstance(other, TensorFloat32):
            self.flatfield *= other
            return self
        else:
            raise ValueError("Unsupported operand type for '*='.")

    @property
    def flatfield(self) -> TensorFloat32:
        """2D tensor."""
        return self._flatfield

    @flatfield.setter
    def flatfield(self, val: TensorFloat32) -> None:
        """2D tensor."""
        if not isinstance(val, TensorFloat32):
            raise AttributeError(f"Invalid datatype for a flatfield: {type(val)}")
        self._flatfield = val

    @property
    def condition(self) -> TensorFloat32:
        """3D Tensor with flat-fields (2D Tensors)."""
        return self._condition

    @condition.setter
    def condition(self, val: List[TensorFloat32]) -> None:
        """3D Tensor with flat-fields (2D Tensors)."""
        if val is not None:
            if not isinstance(val, list):
                raise AttributeError(f"Invalid datatype for a flatfield: {type(val)}]")
            if not all([isinstance(v, TensorFloat32) for v in val]):
                raise AttributeError(f"Invalid datatype for flatfield stack elements: {type(val[0])}]")
        self._condition = val
