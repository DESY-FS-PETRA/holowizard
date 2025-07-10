# standard libraries
from pathlib import Path
from typing import Optional

# third party libraries
import numpy as np

# local libraries
from .datatypes import Pathlike
from . import fileIO


__all__ = [
    "tiff_to_showable",
    "phantom_to_showable",
    "convert_all_tiffs",
]


def tiff_to_showable(path: Pathlike, format: str = "png", target: Optional[Pathlike] = None) -> None:
    path = Path(path)
    filename = f"{path.stem}.{format}"
    output_folder = Path(target) if target is not None else path.parent
    output_folder.mkdir(parents=True, exist_ok=True)
    img = fileIO.load_img(path)
    fileIO.save_img(img, output_folder / filename)


def phantom_to_showable(path: Pathlike, format: str = "png", target: Optional[Pathlike] = None) -> None:
    path = Path(path)
    filename = f"{path.stem}.{format}"
    output_folder = Path(target) if target is not None else path.parent
    output_folder.mkdir(parents=True, exist_ok=True)
    phantom = fileIO.load_phantom(path)
    fileIO.save_img(np.abs(phantom), output_folder / filename)


def phantom_pkl_to_tiff(path: Pathlike, target: Optional[Pathlike] = None) -> None:
    path = Path(path)
    output_folder = Path(target) if target is not None else path.parent
    output_folder.mkdir(parents=True, exist_ok=True)
    phantom = fileIO.load_phantom(path)
    fileIO.save_img(phantom.phaseshift, output_folder / f"{path.stem}_phaseshift.tiff")
    fileIO.save_img(phantom.absorption, output_folder / f"{path.stem}_absorption.tiff")


def convert_all_tiffs(path: Pathlike, format: str = "png", target: Optional[Pathlike] = None) -> None:
    """Convert all tiff-files in a folder into showables.

    Args:
        path (Pathlike): Folder in which to look and convert tiff-files.
        format (str, optional): Target format. Defaults to 'png'.
        target (Optional[Pathlike], optional): If given, store showables in that folder. Defaults to None.
    """
    path = Path(path)
    for file in path.iterdir():
        if file.suffix == ".tiff":
            tiff_to_showable(file, format=format, target=target)
