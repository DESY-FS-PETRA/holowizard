import logging
import holowizard.core
import numpy as np
from typing import List
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as ttf
import PIL
import math
from torchvision.transforms import InterpolationMode
from scipy.ndimage import fourier_gaussian

from holowizard.interface.parameters.beam_setup import BeamSetup
from holowizard.interface.parameters.data_dimensions import DataDimensions
from holowizard.interface.parameters.measurement import Measurement
from holowizard.core.utils.transform import pad_to_size
from holowizard.core.utils.transform import crop_center
from holowizard.core.reconstruction.constraints.window_2d import (
    get_2d_window_from_function,
)
from holowizard.core.utils.remove_outliers import remove_outliers
from holowizard.interface.parameters.padding import Padding

from .process_data_dimensions import process_data_dimensions
from .process_image import process_image, process_measurement


def blackman(x, width):
    return (
        0.42
        - 0.5 * np.cos(x * 2 * np.pi / (width - 1))
        + 0.08 * np.cos(x * 4 * np.pi / (width - 1))
    )


def process_padding_options(
    measurements: List[Measurement],
    beam_setup: BeamSetup,
    data_dimensions: DataDimensions,
    padding_options: Padding,
):
    measurements = deepcopy(measurements)
    padding_options = deepcopy(padding_options)
    beam_setup = deepcopy(beam_setup)
    data_dimensions = deepcopy(data_dimensions)

    data_dimensions = process_data_dimensions(data_dimensions, padding_options)

    beam_setup = process_beam_setup(beam_setup, padding_options, data_dimensions)

    for i in range(len(measurements)):
        measurements[i].data = process_measurement(
            measurements[i].data, padding_options, data_dimensions, i
        )

    return measurements, beam_setup, data_dimensions


def process_beam_setup(
    beam_setup: BeamSetup, padding_options: Padding, data_dimensions: DataDimensions
):
    beam_setup.px_size = beam_setup.px_size * padding_options.down_sampling_factor
    beam_setup.flat_field = process_image(
        beam_setup.flat_field, padding_options, data_dimensions, 0
    )
    beam_setup.probe = process_image(
        beam_setup.probe, padding_options, data_dimensions, 0
    )
    return beam_setup
