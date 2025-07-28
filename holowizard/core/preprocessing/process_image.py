import logging
import torch
import torch.nn as nn
import torchvision.transforms as ttf
import math
from torchvision.transforms import InterpolationMode

from holowizard.core.utils.transform import crop_center
from holowizard.core.utils.transform import pad_to_size

from holowizard.core.parameters.data_dimensions import DataDimensions
from holowizard.core.parameters.padding import Padding
from holowizard.core.reconstruction.constraints.window_2d import get_2d_window

from .boundaries import Boundaries


def flip_and_pad(image, data_dimensions: DataDimensions, padding_options: Padding, shift_phases=False):
    orig_size = data_dimensions.fov_size

    fov = crop_center(image, orig_size).clone()

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
        m = nn.ReflectionPad2d((orig_size[0] - 1, 0, 0, 0))
    else:
        return image

    fov = fov[None, None, :, :]
    mirrored_image = m(fov)[0, 0, :, :]

    mirrored_image = pad_to_size(mirrored_image, image.shape, padMode="constant", padval=padding_options.a0)

    if padding_options.padding_mode.value == Padding.PaddingMode.MIRROR_LEFT.value:
        mirrored_image = torch.roll(mirrored_image, -int(orig_size[0] / 2), 1)

    return mirrored_image


def process_image(
    image,
    padding_options: Padding,
    data_dimensions: DataDimensions,
    index,
    padding_val=1.0,
):
    if image == None:
        return None

    image = torch.rot90(image)

    if padding_options.padding_factor < 4:
        padded_size = tuple([math.ceil(x * 4) for x in data_dimensions.fov_size])
    else:
        padded_size = data_dimensions.total_size
    padded_size_extern = data_dimensions.total_size

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
    cropped_size = image.shape

    # Pad image
    logging.debug("Pad to a total size of " + str(padded_size))
    image = pad_to_size(image, padded_size, padval=padding_val)

    logging.image_debug(
        "image_padded_" + str(index),
        crop_center(image, padded_size_extern).cpu().numpy(),
    )

    window_width_x = (
        int(data_dimensions.fading_width[0][0] // padding_options.down_sampling_factor),
        int(data_dimensions.fading_width[0][1] // padding_options.down_sampling_factor),
    )
    window_width_y = (
        int(data_dimensions.fading_width[1][0] // padding_options.down_sampling_factor),
        int(data_dimensions.fading_width[1][1] // padding_options.down_sampling_factor),
    )

    boundaries = Boundaries(padded_size, cropped_size)
    image = flip_and_pad(image, data_dimensions, padding_options)

    if padding_options.padding_mode.value == Padding.PaddingMode.MIRROR_ALL.value:
        window = get_2d_window(
            image.shape,
            [
                (boundaries.start_top_x, boundaries.end_bottom_x),
                (boundaries.start_left_y, boundaries.end_right_y),
            ],
            [window_width_x, window_width_y],
            data_dimensions.window_type,
            image.device,
        )

    elif padding_options.padding_mode.value == Padding.PaddingMode.MIRROR_LEFT.value:
        window = get_2d_window(
            image.shape,
            [
                (boundaries.start_middle_x, boundaries.end_middle_x),
                (boundaries.start_left_y, boundaries.end_middle_y),
            ],
            [window_width_x, window_width_y],
            data_dimensions.window_type,
            image.device,
        )

    elif padding_options.padding_mode.value == Padding.PaddingMode.CONSTANT.value:
        window = get_2d_window(
            image.shape,
            [
                (boundaries.start_middle_x, boundaries.end_middle_x),
                (boundaries.start_middle_y, boundaries.end_middle_y),
            ],
            (window_width_x, window_width_y),
            data_dimensions.window_type,
            image.device,
        )

        # window = torch.ones(image.shape,device=image.device)
    else:
        raise RuntimeError("Padding mode not implemented")

    data_dimensions.window = window

    if padded_size_extern < padded_size:
        image = crop_center(image, padded_size_extern)
        data_dimensions.window = crop_center(data_dimensions.window, padded_size_extern)

    logging.image_debug(
        "window_object_" + str(index),
        data_dimensions.window.cpu().numpy(),
    )
    logging.image_debug(
        "image_unmasked_" + str(index),
        image.cpu().numpy(),
    )

    constant_shift = padding_val * torch.ones(image.shape, device=image.device)
    image = (image - constant_shift) * data_dimensions.window + constant_shift

    logging.image_debug(
        "image_masked_" + str(index),
        image.cpu().numpy(),
    )

    return image


def process_measurement(image, padding_options: Padding, data_dimensions: DataDimensions, index):
    return process_image(image / padding_options.a0, padding_options, data_dimensions, index)
