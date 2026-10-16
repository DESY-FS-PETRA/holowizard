import logging
import torch
import torch.nn as nn
import torchvision.transforms as ttf
import math
from torchvision.transforms import InterpolationMode

from holowizard.core.src.misc.transform import crop_center
from holowizard.core.src.misc.transform import pad_to_size

from holowizard.core.src.parameters.data_dimensions import DataDimensions
from holowizard.core.src.parameters.padding import Padding


def flip_and_pad(image, padding_options: Padding, data_dimensions: DataDimensions, shift_phases=False, padval=0.0):
    fov_size = data_dimensions.fov_size

    fov = crop_center(image, fov_size).clone()

    if shift_phases:
        max_shift = torch.max(fov.real)
        if max_shift > 0:
            pass
        else:
            fov.real = fov.real + torch.abs(max_shift)

    if padding_options.padding_mode.value == Padding.PaddingMode.MIRROR_ALL.value:
        padding_size = min(fov.shape) - 1
        m = nn.ReflectionPad2d(padding_size)
    elif padding_options.padding_mode.value == Padding.PaddingMode.MIRROR_LEFT.value:
        m = nn.ReflectionPad2d((fov_size[0] - 1, 0, 0, 0))
    elif padding_options.padding_mode.value == Padding.PaddingMode.CONSTANT.value:
        def m(x):
            return x
    else:
        raise RuntimeError("Padding mode " + str(padding_options.padding_mode.value) + " not implemented")

    fov = fov[None, None, :, :]
    mirrored_image = m(fov)[0, 0, :, :]

    if mirrored_image.shape[0] < data_dimensions.total_size[0]:
        mirrored_image = pad_to_size(mirrored_image, data_dimensions.total_size, pad_mode="constant", padval=padval)
    elif mirrored_image.shape[0] > data_dimensions.total_size[0]:
        mirrored_image = crop_center(mirrored_image, data_dimensions.total_size)
    else:
        pass

    if padding_options.padding_mode.value == Padding.PaddingMode.MIRROR_LEFT.value:
        mirrored_image = torch.roll(mirrored_image, -int(fov_size[0] / 2), 1)

    return mirrored_image


def flip_and_pad_and_fade(image, padding_options: Padding, data_dimensions: DataDimensions, padval, index=0):
    image = flip_and_pad(image, padding_options, data_dimensions, padval=padval)

    logging.image_debug(
        "image_unmasked_" + str(index),
        image.real.cpu().numpy(),
    )

    constant_shift = padval * torch.ones(image.shape, device=image.device)
    image = (image - constant_shift) * data_dimensions.window + constant_shift

    logging.image_debug(
        "image_masked_" + str(index),
        image.real.cpu().numpy(),
    )

    return image


def resize_and_flip_and_pad_and_fade(image, padding_options: Padding, data_dimensions: DataDimensions, padval, index=0):
    if image is None:
        return None

    original_size = image.shape
    image = image[
        padding_options.cutting_band : original_size[0] - padding_options.cutting_band,
        padding_options.cutting_band : original_size[1] - padding_options.cutting_band,
    ]

    if padding_options.down_sampling_factor > 1:
        down_sampled_size = tuple([math.ceil(x / padding_options.down_sampling_factor) for x in image.shape])
        image = ttf.Resize(down_sampled_size, interpolation=InterpolationMode.BILINEAR, antialias=True)(
            image[None, None, :, :]
        )[0, 0, :, :]

    logging.image_debug("image_resized_" + str(index), image.cpu().numpy())

    image = flip_and_pad_and_fade(image, padding_options, data_dimensions, padval, index)

    return image
