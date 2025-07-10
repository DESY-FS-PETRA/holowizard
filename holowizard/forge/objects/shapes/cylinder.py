# standard libraries

# third party libraries
import torch

# local libraries
from .shape import Shape
from holowizard.forge.utils.torch_settings import get_torch_device


__all__ = [
    "Cylinder",
]


class Cylinder(Shape):
    def __init__(self, radius: int, height: int) -> None:
        cylinder = self._create_shape(radius, height)
        super().__init__(cylinder)

    def _create_shape(self, radius: int, height: int) -> torch.Tensor:
        diameter_cylinder = 2 * radius
        lspace_x = torch.linspace(-1, 1, diameter_cylinder, device=get_torch_device())
        lspace_y = torch.linspace(-1, 1, height, device=get_torch_device())
        x, y = torch.meshgrid(lspace_x, lspace_y, indexing="xy")
        return torch.sqrt(torch.clamp(1 - x**2, min=1e-8, max=1))
