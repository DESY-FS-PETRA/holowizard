# standard librarie
from typing import List, Dict, Any

# third party libraries

# local libraries
import holowizard.forge.utils.fileIO as fileIO
from holowizard.forge.utils.random import get_random_idx
from holowizard.forge.utils.datatypes import Pathlike, TensorFloat32
from holowizard.forge.datasets import TIFFDataset


__all__ = [
    "TIFFUniformByPattern",
]


class TIFFUniformByPattern(TIFFDataset):
    """Dataset for tiff-files."""

    def __init__(
        self,
        dirs: Pathlike | List[Pathlike],
        patterns: str | List[str],
        recursive: bool,
        remove_outliers: bool,
        augmentations: List[str | Dict[str, Any]] | None = [],
    ) -> None:
        """Normal `TIFFDataset`, but the files are divided into subsets based on given pattern(s).
        Args:
            dirs (Pathlike | List[Pathlike]): Directory with all the flat fields.
            patterns (str | List[str]): Pattern or list of patterns to identify a flat-field. If patterns == "" or
                            patterns == [] then all files from the directories are used.
            recursive (bool): If True, search recursively for flat fields for each given directory in `dirs`.
            remove_outliers (bool): If True, apply `RemoveOutliers` as first transformation.
            augmentations (List[str], optional): List of transformations applied on the loaded data. Available
                    Defaults to [].
        """
        super().__init__(dirs, patterns, recursive, remove_outliers, augmentations=augmentations)

        dataset_sorted = {}
        for pattern in self.patterns.copy():
            files = [file for file in self.dataset if pattern in file]
            if len(files) == 0:  # no match for pattern found
                self.patterns.remove(pattern)
                continue
            dataset_sorted[pattern] = [file for file in self.dataset if pattern in file]
        self.dataset = dataset_sorted
        self.length = len(self.dataset)  # only considers the keys
        assert self.length > 0, f"No match for patterns found: {self.patterns}."

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> TensorFloat32:
        """Load a random flatfield.

        Args:
            idx (int): Index of the filters.

        Returns:
            TensorFloat32: Loaded image.
        """
        idx = get_random_idx(subset := self.dataset[self.patterns[idx]])
        sample = fileIO.load_img(subset[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample
