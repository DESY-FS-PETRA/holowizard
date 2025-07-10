# standard libraries
from abc import ABC, abstractmethod

# third party libraries

# local packages
import holowizard.forge.utils.random as random
from holowizard.forge.utils.datatypes import Tensorlike


__all__ = [
    "Noise",
    "GaussianNoise",
    "Gaussian",
    "PoissonNoise",
    "Poisson",
]


class Noise(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x: Tensorlike) -> Tensorlike:
        return x + self.get_noise()

    @abstractmethod
    def get_noise(self) -> Tensorlike:
        raise NotImplementedError("Needs to be implemented in child classes.")

    @property
    def noise(self) -> Tensorlike:
        return self.get_noise()


class GaussianNoise(Noise):
    def __init__(self, mean: float, std: float, intensity: float, *, size: int):
        self.mean = mean
        self.std = std
        self.intensity = intensity
        self.size = size

    def get_noise(self) -> Tensorlike:
        return self.intensity * random.normal(self.mean, self.std, size=(self.size, self.size))


class Gaussian(GaussianNoise):
    pass


class PoissonNoise(Noise):
    def __init__(self, lam: float, intensity: float, *, size: int):
        self.lam = lam
        self.intensity = intensity
        self.size = size

    def get_noise(self) -> Tensorlike:
        return self.intensity * random.poisson(self.lam, size=(self.size, self.size))


class Poisson(PoissonNoise):
    pass
