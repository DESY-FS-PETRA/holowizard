"""Unit conversions to match other repositories or general functionality.

Convert units for values used in the configuration files or generally in this repository, so in case the units
that are used in this repository are changed, the conversions only need to be done here.

Example:
    The energy used in the livereco_server-repository is keV, whereas here it is in eV, due to the henke Interface which
    only accepts the energy in eV.
"""

# standard libraries

# third party libraries

# local libraries


__all__ = [
    "convert_energy",
    "convert_z01",
    "convert_z02",
    "convert_det_px_size",
]


def convert_energy(energy: int) -> float:
    return energy


def convert_z01(z01: float) -> float:
    return round(z01, 3) * 1e6


def convert_z02(z02: float) -> float:
    return round(z02, 3) * 1e6

def convert_det_px_size(det_px_size: float) -> float:
    return det_px_size * 1e6
