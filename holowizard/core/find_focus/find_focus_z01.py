import logging

import numpy
import scipy
import sys
from typing import List

from holowizard.core.logging.logger import Logger
from holowizard.core.reconstruction.single_projection.reconstruct_multistage import (
    reconstruct,
)
from holowizard.core.parameters.options import Options
from holowizard.core.parameters.measurement import Measurement
from holowizard.core.parameters.data_dimensions import DataDimensions
from holowizard.core.parameters.beam_setup import BeamSetup
from holowizard.core.utils.transform import crop_center


z01_values_history = []
loss_values_history = []


def check_history(z01):
    for i in range(len(z01_values_history)):
        if z01_values_history[i] == z01:
            return True, loss_values_history[i]

    return False, 0


def get_loss_reconstruction(
    z01,
    measurement: Measurement,
    beam_setup: BeamSetup,
    options: List[Options],
    data_dimensions: DataDimensions,
    viewer,
):
    global z01_values_history
    global loss_values_history

    measurement.z01 = z01[0]

    found, loss_local = check_history(measurement.z01)

    if not found:
        current_result, loss_se_all, fov_size = reconstruct(
            [measurement], beam_setup, options, data_dimensions, viewer
        )
        loss_local = loss_se_all[-1]
        z01_values_history.append(measurement.z01)
        loss_values_history.append(loss_local.cpu().numpy())
        loss = loss_local
        if Logger.current_log_level <= Logger.level_num_image_final:
            logging.image_final(
                "focus_series_" + str(z01[0]),
                crop_center(current_result.real, fov_size).cpu().numpy(),
            )

    else:
        # Fix for buggy nelder-mead implementation when bounds are used and nelder-mead reflects to outside of bound interval. Punish this out-of-bound case
        loss = sys.float_info.max

    logging.loss(f"{'Find Focus - '}{'Loss: '}{loss_local: < 25}{'z01: '}{z01[0]}")

    return float(loss)


def find_focus(
    measurement: Measurement,
    beam_setup: BeamSetup,
    options: List[Options],
    data_dimensions: DataDimensions,
    viewer,
):
    global z01_values_history
    global loss_values_history

    z01_values_history = []
    loss_values_history = []

    z01_guess = measurement.z01

    bounds = measurement.z01_bounds
    bounds_array = numpy.zeros((2, 1))
    bounds_array[0, 0] = bounds[0]
    bounds_array[1, 0] = bounds[1]

    logging.info("Using z01 guess " + str(z01_guess))
    logging.info("Using z01 bounds " + str(bounds))

    logging.debug("Starting optimizer (squared error)")
    found_z01 = scipy.optimize.minimize(
        get_loss_reconstruction,
        z01_guess,
        args=(measurement, beam_setup, options, data_dimensions, viewer),
        method="Nelder-Mead",
        bounds=[bounds],
        options={
            "xatol": options[-1].z01_tol,
            "fatol": 1000000.0,
            "initial_simplex": bounds_array,
        },
    )
    logging.debug("Found z01=" + str(found_z01.x[0]))
    logging.debug("find_focus finished")

    return found_z01.x[0], z01_values_history, loss_values_history
