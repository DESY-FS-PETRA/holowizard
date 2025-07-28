# standard libraries
from typing import List

# third party libraries
import matplotlib.pyplot as plt
import numpy as np

# local libraries
from holowizard.forge.objects.shapes import *


__all__ = [
    "create_shapes",
    "run_shape_tests",
    "shapes_plot_test",
    "shape_rotate_test",
    "shape_standardized_test",
]


def create_shapes() -> List[Shape]:
    shapes = []
    shapes.append(Ball(50))
    shapes.append(Polygon([0, 20, 40, 50], [0, 30, 40, 100]))
    shapes.append(Ellipse(50, 100))
    shapes.append(Ellipse(50, 100, cuts=[0.05, 0.0, 0.02, 0.1]))
    shapes.append(Rectangle((100, 50)))
    shapes.append(Cylinder(50, 256))
    shapes.append(CylinderRoundTip(50, 256))
    return shapes


def run_shape_tests() -> None:
    shapes_plot_test()
    shape_rotate_test()
    shape_standardized_test()


def shapes_plot_test() -> None:
    for shape in create_shapes():
        shape.plot()


def shape_rotate_test() -> None:
    rotated_shapes = [shape.rotate(45) for shape in create_shapes()]
    for shape in rotated_shapes:
        plt.imshow(shape)
        plt.show()


def shape_standardized_test() -> None:
    for shape in create_shapes():
        assert np.all((shape.shape >= 0) & (shape.shape <= 1))
    print("Test - Standardized:\tPASSED")
