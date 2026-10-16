import torch
import numpy as np
from typing import List
from copy import deepcopy

from holowizard.core.src.parameters.beam_setup import BeamSetup
from holowizard.core.src.parameters.data_dimensions import DataDimensions
from holowizard.core.src.parameters.measurement import Measurement
from holowizard.core.src.parameters.padding import Padding
from holowizard.core.src.misc.transform import resize_arrays_to_size

from .process_data_dimensions import process_data_dimensions
from .process_measurement import process_measurement, process_measurement_noflipandpad
from .process_beam_setup import process_beam_setup, process_beam_setup_noflipandpad


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
        measurements[i].data = process_measurement(measurements[i].data, padding_options, data_dimensions, i)

    return measurements, beam_setup, data_dimensions


def process_padding_options_noflipandpad(
    measurements: List[Measurement],
    beam_setup: BeamSetup,
    data_dimensions: DataDimensions,
    padding_options: Padding,
):
    measurements = deepcopy(measurements)
    padding_options = deepcopy(padding_options)
    beam_setup = deepcopy(beam_setup)
    data_dimensions = deepcopy(data_dimensions)

    data_dimensions = process_data_dimensions(data_dimensions, padding_options, init_window=False)

    beam_setup = process_beam_setup_noflipandpad(beam_setup, padding_options, data_dimensions)

    for i in range(len(measurements)):
        measurements[i].data = process_measurement_noflipandpad(
            measurements[i].data, padding_options, data_dimensions, i
        )

    return measurements, beam_setup, data_dimensions
