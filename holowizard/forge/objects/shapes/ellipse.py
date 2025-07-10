# standard libraries
from typing import List

# third party libraries
import numpy as np
import torch
from skimage.draw import ellipse

# local libraries
from .shape import Shape
from holowizard.forge.utils.torch_settings import get_torch_device


__all__ = [
    "Ellipse",
]


class Ellipse(Shape):
    def __init__(self, radius_r: int, radius_c: int, cuts: List[float] = [0] * 4) -> None:
        """TODO

        Args:
            radius_r (int): Radius in vertical direction.
            radius_c (int): Radius in horizontal direction.
            cuts (List[float], optional): [left, right, top, bottom]. Defaults to [0]*4.
        """
        self.cuts = cuts
        self.radius = (radius_r, radius_c)
        rr, cc = ellipse(*self.radius, *self.radius)

        mask = self._get_mask(rr, cc)
        self.rr = rr[mask]
        self.cc = cc[mask]

        shape = torch.zeros((2 * radius_r, 2 * radius_c), device=get_torch_device(), dtype=torch.float)
        shape[self.rr, self.cc] = 1
        super().__init__(shape)

    def _get_mask(self, rr, cc) -> np.array:
        mask_rr_t = self._cut_mask(self.cuts[0], "t", rr, 2 * self.radius[0])
        mask_rr_b = self._cut_mask(self.cuts[1], "b", rr, 2 * self.radius[0])
        mask_cc_l = self._cut_mask(self.cuts[2], "l", cc, 2 * self.radius[1])
        mask_cc_r = self._cut_mask(self.cuts[3], "r", cc, 2 * self.radius[1])
        return mask_rr_t & mask_rr_b & mask_cc_l & mask_cc_r

    def _cut_mask(self, cut: float, side: str, vec: np.array, size: int) -> np.array:
        if side.lower() in "lt":  # left or top
            return vec >= round(cut * size)
        elif side.lower() in "rb":  # right or bottom
            return vec < round(size - cut * size)
        else:
            raise ValueError(f"Invalid side. Must be one of [l]eft, [r]ight, [t]op or [b]ottom. Actual: {side}.")
