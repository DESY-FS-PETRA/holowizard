# standard libraries
from abc import ABC, abstractmethod

# third party libraries

# local packages
from holowizard.forge.experiment.setup import Setup
from holowizard.forge.utils.datatypes import TensorFloat32


__all__ = [
    "Simulation",
]


class Simulation(ABC):
    def __init__(self, setup: Setup):
        self.setup = setup

    def __call__(self, *args, **kwargs) -> TensorFloat32:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self) -> TensorFloat32:
        raise NotImplementedError("Forward simulation needs to implemented in child class.")

    @property
    @abstractmethod
    def gt_hologram(self) -> TensorFloat32:
        pass
