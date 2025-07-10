# standard libraries
from typing import List

# third party libraries
import torch
from skimage.draw import polygon

# local libraries
from .shape import Shape
from holowizard.forge.utils.torch_settings import get_torch_device


__all__ = [
    "Polygon",
]


class Polygon(Shape):
    def __init__(self, r: List[int], c: List[int]) -> None:
        if len(r) != len(c):
            raise ValueError("r and c must have equal number of points.")

        self.r, self.c = r, c
        self.rr, self.cc = polygon(r, c)

        shape = torch.zeros((max(r) + 1, max(c) + 1), device=get_torch_device(), dtype=torch.float)
        shape[self.rr, self.cc] = 1
        cropped_shape = shape[min(r) : max(r) + 1, min(c) : max(c) + 1]
        super().__init__(cropped_shape)
