# standard libraries
from dataclasses import dataclass

# third party libraries

# local libraries

__all__ = [
    "PhysicalProperties",
]


@dataclass(frozen=True)
class PhysicalProperties:
    material: str
    energy: float
    thickness: int
    delta: float
    beta: float
    absorption: float
    phaseshift: float
