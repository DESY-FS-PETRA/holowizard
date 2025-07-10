# standard libraries
from typing import List, Dict, Any

# third party libraries
from torch.utils.data import Dataset
import torchvision.transforms as torch_transforms

# local libraries
from holowizard.forge.configs.parse_config import init_obj
import holowizard.forge.utils.transforms as custom_transforms


__all__ = [
    "BaseDataset",
]


class BaseDataset(Dataset):
    """Base dataset."""

    def __init__(
        self, remove_outliers: bool, *, as_tuple: bool, augmentations: List[Dict[str, Any] | str] | None = None
    ) -> None:
        """Initialize class.

        Args:
            remove_outliers (bool): If True, first remove outliers from the image, before running any other transforms.
            as_tuple (bool): Only relevant when only a single image will be transformed. Flag to determine whether the
                        `FlexCompose` should return the transformed images as a tuple-type in the form `(img,)` or as
                        the single image `img`. Must be explicitly passed by the child class, since this is
                        class-specific.
            augmentations (List[Dict[str, Any] | str] | None, optional): List containing either the class name of a
                        tranform or a dictionary of the form {"type": <class name>, "args": {}}, where "args" allow
                        keyword parametric initialization of the transform with name <class name>. If only a name is
                        given, i.e. string, the class is initialized without any arguments. The transform can either be
                        a custom transform from the module `holowizard.forge.utils.transform` or a pytorch
                        implementation in the moduole `torchvision.transforms`. Note, that the former will be searched
                        first for a match. If `augmenations` is `[]` or `None` no transformations are applied, except
                        `RemoveOutliers` if the flag is set correspondingly. `Defaults to None.
        """
        super().__init__()
        self.transform = self._load_transform(remove_outliers, as_tuple=as_tuple, augmentations=augmentations)

    def _load_transform(self, remove_outliers: bool, as_tuple: bool, augmentations: List[Dict[str, Any] | str] | None):
        if not remove_outliers and not augmentations:
            return None

        if augmentations is None:
            augmentations = []
        elif len(augmentations) >= 1 and "To2DTensor" not in augmentations:
            # ensure to have 2D output, regardless of all previous transforms
            augmentations.append("To2DTensor")

        if remove_outliers and "RemoveOutliers" not in augmentations:
            augmentations = ["RemoveOutliers"] + augmentations

        transforms = []
        for augmentation in augmentations:
            try:
                name = augmentation["type"]
                args = augmentation.get("args", {})  # no arguments given
            except:
                name = augmentation
                args = {}

            try:
                transform = init_obj(name, custom_transforms, module_args=args)
            except:
                t = init_obj(name, torch_transforms, module_args=args)
                transform = custom_transforms.GenericFlexTransform(t)
            finally:
                transforms.append(transform)
        return custom_transforms.FlexCompose(transforms, as_tuple=as_tuple)
