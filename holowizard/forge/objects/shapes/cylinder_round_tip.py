# standard libraries

# third party libraries
import torch

# local libraries
from .shape import Shape
from .ball import Ball
from .cylinder import Cylinder


__all__ = [
    "CylinderRoundTip",
]


class CylinderRoundTip(Shape):
    def __init__(self, radius: int, height: int) -> None:
        cylinder_round_tip = self._create_shape(radius, height)
        super().__init__(cylinder_round_tip)  # cut away zero-columns in first (and last) column

    def _create_shape(self, radius: int, height: int) -> torch.Tensor:
        cylinder = Cylinder(radius, height).shape
        cylinder_tip = self._create_tip(radius)
        cylinder[:radius, :] = cylinder_tip
        return cylinder

    def _create_tip(self, radius: int) -> torch.Tensor:
        ball = Ball(radius)
        cylinder_tip = ball.shape[:radius, :]
        return cylinder_tip
