# standard libraries
from typing import Dict, Any, List

# third party libraries
import numpy as np

# local libraries
import holowizard.forge.objects.shape_builder as module_shape_builder
from holowizard.forge.objects.shape_builder import ShapeBuilder
import holowizard.forge.experiment as experiment
from holowizard.forge.configs.parse_config import init_obj
from .shapes import Shape


__all__ = [
    "ShapeSampler",
]


class ShapeSampler:
    def __init__(self, shapes: List[str], rotate: bool, **kwargs) -> None:
        self.rotate = rotate
        kwargs = self._fix_kwargs(kwargs)
        self.shape_builders: List[ShapeBuilder] = []
        for shape in shapes:
            builder = init_obj(f"{shape}Builder", module_shape_builder, module_args=kwargs)
            self.shape_builders.append(builder)
        assert len(self.shape_builders) >= 1

    def get_shape(self) -> Shape:
        idx = np.random.randint(len(self.shape_builders))
        shape = self.shape_builders[idx].build()
        if self.rotate:
            angle = np.random.randint(360)
            shape.rotate(angle)

        return shape

    def _fix_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Fix shape sizes based on the downsample factor of the experiment setup.

        Args:
            kwargs (Dict[str, Any]): Keyword arguments for the shape builders.

        Returns:
            Dict[str, Any]: Keyword arguments adapted to possible resizing.
        """
        if (downsample := experiment.GLOBAL_EXP_SETUP.downsample_factor) == 1:
            return kwargs
        kwargs["size_range"] = (
            max(1, int(kwargs["size_range"][0] / downsample)),
            int(kwargs["size_range"][1] / downsample),
        )
        kwargs["radius_range"] = (
            max(1, int(kwargs["radius_range"][0] / downsample)),
            int(kwargs["radius_range"][1] / downsample),
        )
        return kwargs
