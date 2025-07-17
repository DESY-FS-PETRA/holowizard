# standard libraries
from typing import Dict, List, Any

# third party libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# local libraries
from holowizard.forge.utils.datatypes import Range, Pathlike
from holowizard.forge.utils import fileIO
from holowizard.forge.utils.unit_conversions import (
    convert_z01,
    convert_z02,
    convert_det_pixel_size,
    convert_energy,
)


__all__ = [
    "append_df",
    "append_row_df",
    "calc_Fr",
    "crop_center",
    "convert_energy",
    "map_str",
    "randint_from_range",
    "show_phantom",
]


def calc_Fr(energy: int, z01: float, z02: float, det_px_size: int) -> float:
    """Calculates the Fresnel number for a specific setup.

    Args:
        energy (int): The beam's energy (in [eV]).
        z01 (float): The distance from the focal point to the object (in [cm] = 1e7 nm).
        z02 (float): Distance from the focal point to the detector (in [m] = 1e9 nm).
        det_px_size (int): Pixel size of detector (in [nm]).

    Returns:
        float: Fresnel number.
    """
    lam = 1.2398 / convert_energy(energy)  # h*c / E[keV]
    M = convert_z02(z02) / convert_z01(z01)
    Fr = convert_det_px_size(det_px_size)**2 / (lam * (convert_z02(z02) - convert_z01(z01)) * M)
    return Fr


def crop_center(img, crop: tuple):
    cropx = crop[0]
    cropy = crop[1]

    x, y = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    if startx < 0:
        startx = 0
        cropx = img.shape[0]
    if starty < 0:
        starty = 0
        cropy = img.shape[1]
    return img[startx : (startx + cropx), starty : (starty + cropy)]


def append_df(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([df_1, df_2])
    return df


def append_row_df(df: pd.DataFrame, row: Dict[str, float]) -> pd.DataFrame:
    df_2 = pd.DataFrame([row])
    return append_df(df, df_2)


def map_str(l: List[Any]) -> List[str]:
    return list(map(str, l))


def randint_from_range(r: Range, size: int | None = None, incl_high: bool = True) -> np.ndarray | int:
    """Wrapper to get random integer values in the given Range with the option to include the upper bound.

    Args:
        r (Range): Draw from the interval (lower and upper bound).
        size (int | None, optional): Number of values to be drawn. Defaults to None.
        incl_high (bool, optional): If True, the upper bound is included. Defaults to True.

    Returns:
        np.ndarray | int: Randomly drawn integers of given size.
    """
    low, high = r
    high += incl_high
    return np.random.randint(low, high, size=size)


def show_phantom(path: Pathlike) -> None:
    phaseshift, absorption = fileIO.load_phantom(path)
    plt.subplot(1, 2, 1)
    # Use 'gray' colormap for grayscale images
    plt.imshow(phaseshift, cmap="gray")
    plt.title("Phaseshift")

    plt.subplot(1, 2, 2)
    plt.imshow(absorption, cmap="gray")
    plt.title("Absorption")
    plt.show()
