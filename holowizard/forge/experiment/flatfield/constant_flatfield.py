# standard libraries

# third party libraries
import torch

# local libraries
from .flatfield import FlatField


__all__ = [
    "ConstantFlatField",
]


class ConstantFlatField(FlatField):
    def __init__(self, val: float, size: int) -> None:
        if val < 0:
            raise ValueError("Flatfield value must be >= 0.")
        flatfield = torch.ones((size, size), dtype=torch.float32) * val
        super(ConstantFlatField, self).__init__(flatfield, size)
