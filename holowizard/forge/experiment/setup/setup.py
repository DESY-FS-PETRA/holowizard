# standard libraries
from abc import ABC
from typing import Dict, Any

# third party libraries

# local package


__all__ = [
    "Setup",
]


class Setup(ABC):
    """Abstract class for all kinds of experimental setups in X-ray nano-holography."""

    def __init__(
        self,
        detector_size: int,
        detector_px_size: int,
        padding_factor: int,
        downsample_factor: int,
        energy: float,
    ):
        self._detector_size = int(detector_size / downsample_factor)
        self._detector_px_size = detector_px_size * downsample_factor
        self._padding_factor = padding_factor
        self._downsample_factor = downsample_factor
        self._energy = energy
        self._probe_size = self.detector_size * padding_factor

    def as_dict(self) -> Dict[str, Any]:
        d = dict(
            detector_size=self.detector_size,
            detector_px_size=self.detector_px_size,
            padding_factor=self.padding_factor,
            downsample_factor=self.downsample_factor,
            energy=self.energy,
            probe_size=self.probe_size,
        )
        return d

    @property
    def detector_size(self) -> int:
        return self._detector_size

    @property
    def detector_px_size(self) -> float:
        return self._detector_px_size

    @property
    def padding_factor(self) -> float:
        return self._padding_factor

    @property
    def downsample_factor(self) -> float:
        return self._downsample_factor

    @property
    def energy(self) -> float:
        return self._energy

    @property
    def probe_size(self) -> float:
        return self._probe_size
