# standard librarie
from typing import List, Dict, Any, Tuple

# third party libraries

# local libraries
import holowizard.forge.utils.fileIO as fileIO
from holowizard.forge.utils.random import get_random_idx
from holowizard.forge.utils.datatypes import Pathlike, TensorFloat32
from .tiff_by_pattern import TIFFUniformByPattern


__all__ = [
    "TIFFMultiChannel",
]


class TIFFMultiChannel(TIFFUniformByPattern):
    """Dataset for tiff-files."""

    def __init__(
        self,
        dirs: Pathlike | List[Pathlike],
        patterns: str | List[str],
        recursive: bool,
        remove_outliers: bool,
        augmentations: List[str | Dict[str, Any]] | None = [],
        num_conditions: int = 1,
    ) -> None:
        """A `TIFFUniformByPattern`, which allows to get a stack of flatfields.
        Args:
            dirs (Pathlike | List[Pathlike]): Directory with all the flat fields.
            patterns (str | List[str]): Pattern or list of patterns to identify a flat-field. If patterns == "" or
                            patterns == [] then all files from the directories are used.
            recursive (bool): If True, search recursively for flat fields for each given directory in `dirs`.
            remove_outliers (bool): If True, apply `RemoveOutliers` as first transformation.
            augmentations (List[str], optional): List of transformations applied on the loaded data. Available
                    Defaults to [].
            num_conditions (int, optional): Number of flat-fields to be loaded. The first channels corresponds to the
                        multiplicative flatfield in the forward pass. Every further channel can be used for conditioning
                        the training of neural networks. Defaults to 1.

        """
        assert num_conditions >= 1
        super().__init__(dirs, patterns, recursive, remove_outliers, augmentations=augmentations)
        self.num_conditions = num_conditions

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[TensorFloat32, List[TensorFloat32]]:
        """Load a random flat-field and other flat-fields similar to the first.

        Args:
            idx (int): Index of the filters.

        Returns:
            Tuple[TensorFloat32, List[TensorFloat32]]: Loaded image and a stack of further (similar) images.
        """
        idx = get_random_idx(subset := self.dataset[self.patterns[idx]])
        sample = fileIO.load_img(subset[idx])
        conditions = []
        for _ in range(self.num_conditions):
            idx = get_random_idx(subset)
            conditions.append(fileIO.load_img(subset[idx]))
        if self.transform:
            transformed = self.transform(sample, *conditions)
            sample, conditions = transformed[0], transformed[1:]
        return sample, list(conditions)
