# standard libraries
from typing import List, Tuple, Any, Callable
from abc import ABC, abstractmethod
import math

# third party libraries
import torch
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch.nn.functional as F_torch  # used in `MedianPool2d``
import torch.nn as nn
from torch.nn.modules.utils import _pair, _quadruple
import numpy as np

# local libraries
from holowizard.forge.utils.datatypes import Tensorlike
import holowizard.forge.experiment as experiment
import holowizard.forge.utils.random as random


__all__ = [
    "FlexCompose",
    "FlexTransform",
    "GenericFlexTransform",
    "HorizontalFlip",
    "VerticalFlip",
    "MedianPool2d",
    "NumpyExpandDim",
    "NumpyToTensor",
    "TensorToNumpy",
    "RandomCropAndResize",
    "RandomFixedRotation",
    "AdjustBrightness",
    "AdjustContrast",
    "RemoveOutliers",
    "Resize",
    "To2DTensor",
]


class FlexTransform(ABC):
    @abstractmethod
    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[Tensorlike, ...]:
        raise NotImplementedError("Must be implemented in child class.")

    def transform(self, t: callable, *imgs) -> Tensorlike | Tuple[Tensorlike, ...]:
        """Apply the same (i.e. deterministic) transform on all given images.

        Args:
            t (callable): Lambda expression for the transform.
            *imgs: Images on which to apply the given transform `t`.

        Returns:
            Tensorlike | Tuple[Tensor, ...]: Transformed images.
        """
        assert len(imgs) >= 1, f"Must have at least 1 image in `imgs`, but got {len(imgs)}."
        return tuple([t(img) for img in imgs])


class FlexCompose(T.Compose):
    def __init__(self, *args, as_tuple: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.as_tuple = as_tuple

    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        for t in self.transforms:
            imgs = t(*imgs)
        return imgs[0] if not self.as_tuple and len(imgs) == 1 else imgs

    def __getitem__(self, idx: int) -> FlexTransform:
        return self.transforms[idx]

    def __add__(self, other):
        if not isinstance(other, FlexCompose):
            raise ValueError(f"type {type(other)} is not allowed for {type(self)}")
        return FlexCompose(self.transforms + other.transforms)


class GenericFlexTransform(FlexTransform):
    def __init__(self, t: Any) -> None:
        """Generic template for deterministic torchvision transforms to be applied to multiple images at once.

        Args:
            t (Any): Torchvision transform.
        """
        self.t = t

    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        return self.transform(lambda a: self.t(a), *imgs)


class Resize(FlexTransform):
    def __init__(self, *args, **kwargs) -> None:
        size = experiment.GLOBAL_EXP_SETUP.detector_size
        self.t = T.Resize(size=(size, size), *args, **kwargs)

    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        return self.transform(lambda a: self.t(a), *imgs)


class RandomCropAndResize(FlexTransform):
    def __init__(self, min_size: float = 0.7):
        self.max_size = experiment.GLOBAL_EXP_SETUP.detector_size
        assert 0.5 <= min_size < 1, f"Min size should be in [0.5, 1.0), not {min_size}"
        self.min_size = min_size
        self.resize = Resize()

    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        output_size = int(self.max_size * random.uniform(self.min_size, 1))
        params = T.RandomCrop.get_params(imgs[0], output_size=(output_size, output_size))
        return self.resize(*self.transform(lambda a: F.crop(a, *params), *imgs))


class ExpandDim(FlexTransform):
    """Add a third dimension of a numpy array of shape (H, W) to (1, H, W)."""

    def __init__(self, axis: int = 0):
        self.axis = axis

    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[Tensorlike, ...]:
        def expand_dim(img: Tensorlike) -> Tensorlike:
            match t := type(img):
                case np.ndarray:
                    return np.expand_dims(img, axis=self.axis)
                case torch.Tensor:
                    return img.unsqueeze(self.axis)
                case _:
                    raise ValueError(f"Expecting either np.ndarray or torch.Tensor, but got: {t}")

        return self.transform(expand_dim, *imgs)


class NumpyToTensor(FlexTransform):
    def __init__(self, expand_dim: bool = True):
        self.expand_dim = ExpandDim(axis=0) if expand_dim else None

    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        if self.expand_dim is not None:
            imgs = self.expand_dim(*imgs)
        return self.transform(lambda a: torch.tensor(a), *imgs)


class TensorToNumpy(FlexTransform):
    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[np.ndarray, ...]:
        return self.transform(lambda a: a.numpy(), *imgs)


class To2DTensor(FlexTransform):
    def __init__(self):
        def t(img: Tensorlike) -> Tensorlike:
            while len(img.shape) > 2:
                img = img[0]
            return img

        self.t_wrapper = t

    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        return self.transform(self.t_wrapper, *imgs)


class RandomFixedRotation(FlexTransform):
    """Rotate by one of the given angles."""

    def __init__(self, angles: List[int] = [90, 180, 270]):
        self.angles = angles

    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        angle = np.random.choice(self.angles)
        return self.transform(lambda a: F.rotate(a, angle), *imgs)


class AdjustBrightness(FlexTransform):
    def __init__(self, tol: float | Tuple[float, float] = 0.2) -> None:
        if isinstance(tol, tuple | list):
            self.lower, self.upper = tol
        else:
            self.lower, self.upper = 1 - tol, 1 + tol

    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """Wrapper for PyTorch implementation of `adjust_brightness`.

        `F.adjust_brightness` only accepts only: torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64. All
        other values will be capped at 1, which is the case for torch.float32 (or torch.uint16, however, the promotion
        to uin16 is not supported). To circumvent this problem, the images are first scaled w.r.t. uin16, i.e. divided
        by 65535, and afterwards rescaled by multiplying by that value.

        Returns:
            Tuple[Tensor, ...]: Images with adjusted brightness.
        """
        # scale by 2*16-1 (UInt15 max value), since adjust brightness handles int-values weirdly and sets the maximum value to one
        brightness_factor = np.random.uniform(self.lower, self.upper)
        return self.transform(
            lambda a: F.adjust_brightness(a / 65535, brightness_factor=brightness_factor) * 65535, *imgs
        )


class AdjustContrast(FlexTransform):
    def __init__(self, tol: float | Tuple[float, float] = 0.2) -> None:
        if isinstance(tol, tuple | list):
            self.lower, self.upper = tol
        else:
            self.lower, self.upper = 1 - tol, 1 + tol

    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """Wrapper for PyTorch implementation of `adjust_contrast`.

        `F.adjust_contrast` only accepts only: torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64. All
        other values will be capped at 1, which is the case for torch.float32 (or torch.uint16, however, the promotion
        to uin16 is not supported). To circumvent this problem, the images are first scaled w.r.t. uin16, i.e. divided
        by 65535, and afterwards rescaled by multiplying by that value.

        Returns:
            Tuple[Tensor, ...]: Images with adjusted contrast.
        """
        contrast_factor = np.random.uniform(self.lower, self.upper)
        return self.transform(lambda a: F.adjust_contrast(a / 65535, contrast_factor=contrast_factor) * 65535, *imgs)


class VerticalFlip(FlexTransform):
    """Randomly flip the image vertically with a probability."""

    def __init__(self, prob: float = 0.5):
        """Initialize with the probability of applying the flip.

        Args:
            prob (float, optional): The probability of flipping the image. Default is 0.5.
        """
        self.prob = prob

    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """Randomly flip the image vertically with the given probability.

        Args:
            x (Tensor): Input tensor (image).
            *args: Additional arguments.

        Returns:
            Tensor: Flipped (or unchanged) tensor.
        """
        return imgs if np.random.rand() < self.prob else self.transform(F.vflip, *imgs)


class HorizontalFlip(FlexTransform):
    """Randomly flip the image horizontally with a probability."""

    def __init__(self, prob: float = 0.5):
        """Initialize with the probability of applying the flip.

        Args:
            prob (float, optional): The probability of flipping the image. Default is 0.5.
        """
        self.prob = prob

    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """Randomly flip the image horizontally with the given probability.

        Args:
            x (Tensor): Input tensor (image).
            *args: Additional arguments.

        Returns:
            Tensor: Flipped (or unchanged) tensor.
        """
        return imgs if np.random.rand() < self.prob else self.transform(F.hflip, *imgs)


class RemoveOutliers(FlexTransform):
    def __init__(self, threshold: float = 1, filter_size: int = 5) -> None:
        self.threshold = threshold
        self.median_pool = MedianPool2d(kernel_size=filter_size, padding=int(math.floor(filter_size / 2)))
        self.t_wrapper = self.get_transform()

    def get_transform(self) -> Callable[[Tensorlike], Tensor]:
        def transform(img: Tensorlike) -> Tensor:
            """Remove outliers from image.

            NOTE: The image must be a 2D Tensor.

            Args:
                img (Tensorlike): Image which will be filtered.

            Returns:
                Tensor: Filtered image.
            """
            img = torch.tensor(img) if type(img) != torch.Tensor else img.clone()
            assert img.ndim == 2
            filtered_img = self.median_pool(img[None, None, :, :])[0, 0, :, :]
            diff_img = img - filtered_img
            std_dev_value = torch.std(diff_img)
            pixels_to_correct = torch.where(abs(diff_img) > (self.threshold * std_dev_value))
            img[pixels_to_correct] = filtered_img[pixels_to_correct]
            return img

        return transform

    def __call__(self, *imgs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        return self.transform(self.t_wrapper, *imgs)


# taken from livereco-server
class MedianPool2d(nn.Module):
    """Median pool (usable as median filter when stride=1) module.

    Args:
        kernel_size: size of pooling kernel, int or 2-tuple
        stride: pool stride, int or 2-tuple
        padding: pool padding, int or 4-tuple (l, r, t, ^) as in pytorch F.pad
        same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        # NOTE: When using F, the option "replace" and the option "mode" do not exist. Instead use
        #       `padding_mode=reflect`. WTF.
        #       The definition below is the same as in the livereco server for convinience
        x = F_torch.pad(x, self._padding(x), mode="replicate")
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
