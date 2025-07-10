# standard libraries
from typing import Tuple

# third party libraries
import torch

# local libraries
from .shape import Shape
from holowizard.forge.utils.torch_settings import get_torch_device


__all__ = [
    "Rectangle",
]


class Rectangle(Shape):
    def __init__(self, size: Tuple[int, int]) -> None:
        # FIXME: currently it is a np.ndarray
        assert len(size) == 2, f"Invalid shape: {size}."
        shape = torch.ones(size.tolist(), device=get_torch_device(), dtype=torch.float)
        super().__init__(shape)
