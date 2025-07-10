# standard libraries
from abc import ABC, abstractmethod

# third party libraries
import numpy as np

# local libraries
from .shapes import *
from holowizard.forge.utils.utilities import randint_from_range
from holowizard.forge.utils.datatypes import Range


__all__ = [
    "EllipseBuilder",
    "BallBuilder",
    "PolygonBuilder",
    "RectangleBuilder",
    "CylinderBuilder",
    "CylinderRoundTipBuilder",
]


class ShapeBuilder(ABC):
    @abstractmethod
    def build(self) -> Shape:
        raise NotImplementedError("build-method need to be implemented in child class.")


class EllipseBuilder(ShapeBuilder):
    def __init__(self, radius_range: Range, ellipse_max_cut: float, **kwargs) -> None:
        self.radius_range = radius_range
        self.ellipse_max_cut = ellipse_max_cut

    def build(self) -> Ellipse:
        r_r, r_c = randint_from_range(self.radius_range, size=2)
        # include cuts with probability of p = 0.25
        cut_mask = np.random.binomial(size=4, n=1, p=0.25)
        cuts = np.random.uniform(0, self.ellipse_max_cut, size=4) * cut_mask
        return Ellipse(r_r, r_c, cuts.tolist())


class BallBuilder(ShapeBuilder):
    def __init__(self, radius_range: Range, **kwargs) -> None:
        self.radius_range = radius_range

    def build(self) -> Ball:
        radius = randint_from_range(self.radius_range)
        return Ball(radius)


class PolygonBuilder(ShapeBuilder):
    def __init__(self, polygon_max_corners: int, size_range: Range, **kwargs) -> None:
        self.polygon_max_corners = polygon_max_corners
        self.size_range = size_range

    def build(self) -> Polygon:
        num_corners = randint_from_range((3, self.polygon_max_corners))
        coord_range = (0, self.size_range[1] - 1)
        r, c = randint_from_range(coord_range, size=(2, num_corners))
        return Polygon(r, c)


class RectangleBuilder(ShapeBuilder):
    def __init__(self, size_range: Range, **kwargs) -> None:
        self.size_range = size_range

    def build(self) -> Rectangle:
        size = randint_from_range(self.size_range, size=2)
        return Rectangle(size)


class CylinderBuilder(ShapeBuilder):
    def __init__(self, radius_range: Range, size_range: Range, **kwargs) -> None:
        self.radius_range = radius_range
        self.size_range = size_range

    def build(self) -> Cylinder:
        radius = randint_from_range(self.radius_range)
        height = randint_from_range(self.size_range)
        return Cylinder(radius, height)


class CylinderRoundTipBuilder(ShapeBuilder):
    def __init__(self, radius_range: Range, size_range: Range, **kwargs) -> None:
        self.radius_range = radius_range
        self.size_range = size_range

    def build(self) -> CylinderRoundTip:
        radius = randint_from_range(self.radius_range)
        height = randint_from_range(self.size_range)
        timeout_tick = 0
        while height < radius and timeout_tick < 1000:
            height = randint_from_range(self.size_range)
            timeout_tick += 1

        if timeout_tick == 1000:
            height = radius
        return CylinderRoundTip(radius, height)
