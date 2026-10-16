import logging

import holowizard.core
from holowizard.core.src.parameters.data_dimensions import DataDimensions
from holowizard.core.src.parameters.padding import Padding
from holowizard.core.src.methods.preprocessing.window_2d import get_2d_window

from .boundaries import Boundaries


def get_window_from_padding_options(
    data_dimensions, padding_options, torch_device=holowizard.core.torch_running_device
):
    window_width_x = (
        int(data_dimensions.fading_width[0][0] // padding_options.down_sampling_factor),
        int(data_dimensions.fading_width[0][1] // padding_options.down_sampling_factor),
    )
    window_width_y = (
        int(data_dimensions.fading_width[1][0] // padding_options.down_sampling_factor),
        int(data_dimensions.fading_width[1][1] // padding_options.down_sampling_factor),
    )

    boundaries = Boundaries(data_dimensions.total_size, data_dimensions.fov_size)

    if padding_options.padding_mode.value == Padding.PaddingMode.MIRROR_ALL.value:
        window = get_2d_window(
            data_dimensions.total_size,
            [
                (boundaries.start_top_x, boundaries.end_bottom_x),
                (boundaries.start_left_y, boundaries.end_right_y),
            ],
            [window_width_x, window_width_y],
            data_dimensions.window_type,
            torch_device,
        )

    elif padding_options.padding_mode.value == Padding.PaddingMode.MIRROR_LEFT.value:
        window = get_2d_window(
            data_dimensions.total_size,
            [
                (boundaries.start_middle_x, boundaries.end_middle_x),
                (boundaries.start_left_y, boundaries.end_middle_y),
            ],
            [window_width_x, window_width_y],
            data_dimensions.window_type,
            torch_device,
        )

    elif padding_options.padding_mode.value == Padding.PaddingMode.CONSTANT.value:
        window = get_2d_window(
            data_dimensions.total_size,
            [
                (boundaries.start_middle_x, boundaries.end_middle_x),
                (boundaries.start_middle_y, boundaries.end_middle_y),
            ],
            (window_width_x, window_width_y),
            data_dimensions.window_type,
            torch_device,
        )

    else:
        raise RuntimeError("Padding mode not implemented")

    return window


def process_data_dimensions(
    data_dimensions: DataDimensions,
    padding_options: Padding,
    init_window=True,
    torch_device=holowizard.core.torch_running_device,
):
    support_range = data_dimensions.fov_range_raw
    total_size_padded = tuple([round(x * padding_options.padding_factor) for x in data_dimensions.total_size])

    data_dimensions.fov_size = (
        data_dimensions.fov_size[0] - 2 * padding_options.cutting_band,
        data_dimensions.fov_size[1] - 2 * padding_options.cutting_band,
    )

    boundaries = Boundaries(total_size_padded, data_dimensions.fov_size)

    support_range = [
        (
            boundaries.start_middle_x + support_range[0][0],
            boundaries.start_middle_x + support_range[0][1],
        ),
        (
            boundaries.start_middle_y + support_range[1][0],
            boundaries.start_middle_y + support_range[1][1],
        ),
    ]

    support_shift = [round((padding_options.padding_factor - 1) * x / 2) for x in data_dimensions.total_size]
    data_dimensions.fov_range = [
        tuple([x + support_shift[0] for x in support_range[0]]),
        tuple([x + support_shift[1] for x in support_range[1]]),
    ]
    data_dimensions.total_size = tuple(
        [
            round(x * padding_options.padding_factor / padding_options.down_sampling_factor)
            for x in data_dimensions.total_size
        ]
    )
    data_dimensions.fov_size = tuple(
        [round(x / padding_options.down_sampling_factor) for x in data_dimensions.fov_size]
    )
    data_dimensions.fov_range = [
        tuple([round(x / padding_options.down_sampling_factor) for x in support_range[0]]),
        tuple([round(x / padding_options.down_sampling_factor) for x in support_range[1]]),
    ]

    if init_window:
        data_dimensions.window = get_window_from_padding_options(
            data_dimensions, padding_options, torch_device=torch_device
        )

    # logging.image_debug(
    # "window",
    # data_dimensions.window.cpu().numpy(),
    # )

    return data_dimensions
