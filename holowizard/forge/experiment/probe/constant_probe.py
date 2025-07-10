# standard libraries

# third party libraries

# local libraries
from .polynomial_probe import PolynomialProbe
from .beam_config import BeamConfig


__all__ = [
    "ConstantProbe",
]


class ConstantProbe(PolynomialProbe):
    def __init__(self, intensity: float, size: int) -> None:
        if intensity < 0:
            raise ValueError("Probe value must be >= 0.")
        beam_cfg = BeamConfig(intensity)
        super().__init__(beam_cfg=beam_cfg, size=size)
