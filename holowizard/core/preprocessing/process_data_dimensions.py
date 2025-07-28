from holowizard.core.parameters.data_dimensions import DataDimensions
from holowizard.core.parameters.padding import Padding

from .boundaries import Boundaries


def process_data_dimensions(data_dimensions: DataDimensions, padding_options: Padding):
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
    return data_dimensions
