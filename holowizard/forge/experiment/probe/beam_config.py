# standard libraries
from typing import Dict, List, Tuple
from dataclasses import dataclass

# third party libraries
import numpy as np

# local libraries
import holowizard.forge.utils.random as random


__all__ = [
    "BeamConfig",
    "get_random_beam_config",
]


@dataclass(frozen=True)
class BeamConfig:  # TODO: ProbeConfig (or even PolynomialConfig) instead of BeamConfig -> also need to change config
    C: float = 1.0  # constant
    x: float = 0.0  # coeff before X
    y: float = 0.0  # coeff before Y
    xx: float = 0.0  # coeff before X^2
    xy: float = 0.0  # coeff before X*Y
    yy: float = 0.0  # coeff before Y^2

    def as_dict(self) -> Dict[str, float]:
        d = {
            "c": self.C,
            "x": self.x,
            "y": self.y,
            "xx": self.xx,
            "xy": self.xy,
            "yy": self.yy,
        }
        return d

    def get_labels(self) -> List[float]:
        return list(self.as_dict().values())

    def get_annotation(self) -> List[str]:
        return list(self.as_dict().keys())


def get_random_beam_config(offset: float | Tuple[float, float], linear: float = 0.0, square: float = 0.0) -> BeamConfig:
    """Randomly samples values for the xray-beam.

    Returns:
        BeamConfig: Configuration object from which the xray can be constructed.
    """

    def symmetric_uniform(val: float) -> float:
        val = abs(val)
        return float(random.uniform(-val, val))

    if type(offset) in [float, int]:
        C = float(offset)
    else:
        C = random.uniform(*offset)
    x = symmetric_uniform(linear)
    y = symmetric_uniform(linear)
    xx = symmetric_uniform(square)
    xy = symmetric_uniform(square)
    yy = symmetric_uniform(square)
    coeffs = [C, x, y, xx, xy, yy]

    beam_decimals = 5
    round_decimals = lambda x: -np.floor(np.log10(x)).astype(int) + beam_decimals

    round_m = round_decimals(linear) if linear else 0
    round_sq = round_decimals(square) if square else 0
    roundings = [beam_decimals, round_m, round_m, round_sq, round_sq, round_sq]
    coeffs = [round(x, y) for x, y in zip(coeffs, roundings)]

    return BeamConfig(*coeffs)
