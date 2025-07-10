# standard libraries

# third party libraries
import torch

# local libraries
from .shape import Shape
from holowizard.forge.utils.torch_settings import get_torch_device


__all__ = [
    "Ball",
]


class Ball(Shape):
    def __init__(self, radius: int) -> None:
        ball = self._create_ball(radius)
        super().__init__(ball)

    def _create_ball(self, radius: int) -> torch.Tensor:
        diameter = 2 * radius
        lspace = torch.linspace(-1, 1, diameter, device=get_torch_device())
        x, y = torch.meshgrid(lspace, lspace, indexing="xy")
        temp = x**2 + y**2
        return torch.sqrt(torch.clamp(1 - temp, min=1e-8, max=1))
