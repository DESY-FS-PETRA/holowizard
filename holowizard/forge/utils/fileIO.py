# standard libraries
from typing import List, TextIO, Dict, Any
from pathlib import Path
import json
import pickle

# third party libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import io
from PIL import Image

# local libraries
from .datatypes import Pathlike, Hologram
from holowizard.forge.objects.phantom import Phantom


__all__ = [
    "add_line_csv",
    "load_hologram",
    "read_json",
    "load_single_line",
    "load_phantom",
    "load_img",
    "read_csv",
    "save_hologram",
    "save_phantom",
    "save_img",
    "write_csv",
    "write_json",
]


def add_line_csv(f: TextIO, line: List[str]) -> None:
    """Append the elements in `line` to a csv file.

    Args:
        f (TextIO): File descriptor of the csv-file in which the data should be written. Should be a csv-file.
        line (List[str]): Elements that will be added to line and separated by comma.
    """
    line = ",".join(line) + "\n"
    f.write(line)


def read_csv(file: Pathlike, header: bool = True) -> pd.DataFrame:
    header = 0 if header else None
    df = pd.read_csv(file, header=header)
    return df


def write_csv(
    df: pd.DataFrame,
    path: Pathlike,
    index: bool = False,
    header: bool | List[str] | None = None,
) -> None:
    df.to_csv(path, index=index, header=header)


def load_single_line(path: Pathlike, idx: int, header: Dict[str, Any] | bool) -> List[int | str | float]:
    """Load a single line based on an index from csv-file.

    Args:
        path (Pathlike): Path to either the csv-file.
        idx (int): Index of the line to be loaded.
        header (Dict[str, Any] | bool): Dictionary containing the information about the csv-file header (defined
                in the config.json file) or just a boolean value.

    Returns:
        List[int | str | float]: The desired line.
    """
    has_header = header if type(header) is bool else all([i != v for i, v in enumerate(header.values())])
    line = read_csv(path, header=has_header).iloc[idx]
    return line


def load_img(path: Pathlike, expand_dim: bool = False, convert_to_float32: bool = True) -> np.ndarray:
    img = io.imread(path)
    if expand_dim and img.ndim == 2:
        img = np.expand_dims(img, axis=0)
    if convert_to_float32:
        img = img.astype(np.float32)
    return img


def save_img(img: np.ndarray, path: Pathlike) -> None:
    path = Path(path)
    if path.suffix.lower() == ".tiff":
        img = Image.fromarray(img)
        img.save(path)
    else:
        plt.imsave(path, img)


def load_hologram(path: Pathlike) -> np.ndarray:
    return load_img(path)


def save_hologram(path: Pathlike, holo: Hologram, idx: int, format: str, fill_zeros: int = 6) -> str:
    """Save hologram.

    Args:
        path (Pathlike): Outputfolder, where to save the hologram
        holo (Hologram): Hologram will be saved.
        idx (int): Index for next file. Defaults to 6.
        format (str): In which format the hologram will be stored.
        fill_zeros (int, optional): Defines how long the file-index should at least be,
                e.g. `idx = 2, fill_zeros = 3` => '002'; `idx = 2323, fill_zeros = 3` => '2323' Defaults to 0.

    Returns:
        str: Filename of the saved hologram.
    """
    path = Path(path)
    file = f"hologram_{str(idx).zfill(fill_zeros)}.{format}"
    filename = path / file
    save_img(holo, filename)
    return filename.name


def save_phantom(path: Pathlike, phantom: Phantom | None, idx: int, fill_zeros: int = 6) -> str | None:
    """Save Phantom as a pickle file.

    Args:
        path (Pathlike): Outputfolder, where to save the phantom.
        phantom (Phantom): Complex array of the phantom. Might be `None`.
        fill_zeros (int, optional): Defines how long the file-index should at least be,
                e.g. `idx = 2, fill_zeros = 3` => '002'; `idx = 2323, fill_zeros = 3` => '2323' Defaults to 6.

    Returns:
        str | None: Filename of the saved phantom. None, if the phantom is None.
    """
    if phantom is None:
        return None

    path = Path(path)
    file = f"phantom_{str(idx).zfill(fill_zeros)}.pkl"
    filename = path / file
    with open(filename, "wb") as f:
        pickle.dump({"phantom": phantom}, f)
    return filename.name


def load_phantom(path: Pathlike) -> Phantom:
    with open(path, "rb") as f:
        data = pickle.load(f)
    phantom = data["phantom"]
    return phantom


def read_json(path: Pathlike) -> Dict[str, Any]:
    """Load data from a JSON file.

    Args:
        path (Pathlike): The path to the JSON file.

    Returns:
        Dict[str, Any]: The loaded data from the JSON file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find file {str(path)}.")
    with open(path, "r") as file:
        data = json.load(file)
    return data


def write_json(data: Dict[str, Any], path: Pathlike) -> None:
    """Write data to a JSON file.

    Args:
        data (Dict[str, Any]): The data to be written to the JSON file.
        path (Pathlike): The path to the JSON file.
    """
    with open(path, "w") as file:
        json.dump(data, file, indent=4)
