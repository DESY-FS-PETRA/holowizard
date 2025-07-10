# standard libraries
from pathlib import Path
from typing import List

# third party libraries
import torch
import numpy as np

# local libraries
from holowizard.forge.utils.torch_settings import get_torch_device
from holowizard.forge.utils.datatypes import Pathlike, Tensorlike, TensorFloat32
from holowizard.forge.utils import fileIO
from holowizard.forge.experiment.flatfield import BaseFlatField


__all__ = [
    "MultiChannelFlatField",
]


class MultiChannelFlatField(BaseFlatField):
    def __init__(self, flatfield: Tensorlike | Pathlike, conditions: List[Tensorlike | Pathlike]) -> None:
        assert isinstance(conditions, list) and conditions != []
        flatfield_stack = []
        for ff in [flatfield] + conditions:
            match type(ff):
                case t if t in [str, Path]:
                    flatfield = torch.tensor(fileIO.load_img(ff))
                case torch.Tensor:
                    flatfield = ff
                case np.ndarray:
                    flatfield = torch.tensor(ff)
                case _:
                    raise ValueError(f"Invalid input for flatfield. Type: {type(ff)}")
            flatfield_stack.append(flatfield)

        self.condition = torch.stack(flatfield_stack[1:], dim=0).to(get_torch_device())
        self.flatfield = flatfield_stack[0].to(get_torch_device())

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
    def condition(self, val: TensorFloat32) -> None:
        """3D Tensor with flat-fields (2D Tensors)."""
        if not isinstance(val, TensorFloat32):
            raise AttributeError(f"Invalid datatype for a flatfield stack: {type(val)}]")

        if not val.ndim == 3:
            raise AttributeError(f"Invalid number of dimensions. Should be 3, but it {val.ndim}]")

        self._condition = val
