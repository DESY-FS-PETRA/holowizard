# standard libraries
from pathlib import Path
import glob
from typing import List, Dict, Any

# third party libraries

# local libraries
import holowizard.forge.utils.fileIO as fileIO
from holowizard.forge.utils.datatypes import Pathlike, TensorFloat32
from .base_dataset import BaseDataset


__all__ = [
    "TIFFDataset",
]


class TIFFDataset(BaseDataset):
    """Dataset for tiff-files."""

    def __init__(
        self,
        dirs: Pathlike | List[Pathlike],
        patterns: str | List[str],
        recursive: bool,
        remove_outliers: bool,
        augmentations: List[str | Dict[str, Any]] | None = [],
    ) -> None:
        """
        Args:
            dirs (Pathlike | List[Pathlike]): Directory with all the flat fields.
            patterns (str | List[str]): Pattern or list of patterns to identify a flat-field; used for pattern matching.
                        If patterns == "" or patterns == [] then all files from the directories are used.
            recursive (bool): If True, search recursively for flat fields for each given directory in `dirs`.
            remove_outliers (bool): If True, apply `RemoveOutliers` as first transformation.
            augmentations (List[str], optional): List of transformations applied on the loaded data. Available
                    Defaults to [].
        """
        super().__init__(remove_outliers, as_tuple=False, augmentations=augmentations)
        # use dict with None values as ordered set
        self.dirs = list(dict.fromkeys(dirs)) if type(dirs) == list else [dirs]
        for dir in self.dirs:
            assert Path(dir).is_dir(), f"{dir=} does not exist"
            is_empty = not any(Path(dir).iterdir())
            assert not is_empty, f"{dir=} is empty."
            # no check whether there ar actually TIFF-files

        match patterns:
            case p if p in ["", []]:
                self.patterns = [""]  # need at least one pattern for the loop below
            case s if type(s) == str:
                self.patterns = [s]
            case l if type(l) == list:
                self.patterns = l
            case _:
                raise ValueError(f"Invalid patterns: {patterns}")

        self.dataset = []
        for dir in list(self.dirs):
            for pattern in self.patterns:
                self.dataset += [
                    file
                    for file in glob.glob(str(Path(dir) / f"*.tif*"), recursive=recursive)
                    if pattern in Path(file).name
                ]
        self.length = len(self.dataset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> TensorFloat32:
        sample = fileIO.load_img(self.dataset[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample
