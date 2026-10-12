import logging

import numpy
import scipy
import sys
from typing import List

from holowizard.core.reconstruction.single_projection.reconstruct_multistage import (
    reconstruct,
)
from holowizard.core.parameters.options import Options
from holowizard.core.parameters.measurement import Measurement
from holowizard.core.parameters.data_dimensions import DataDimensions
from holowizard.core.parameters.beam_setup import BeamSetup

values_history = []
loss_values_history = []


def check_history(value):
    try:
        if value in values_history:
            return True, loss_values_history[values_history.index(value)]
    except:
        pass

    return False, 0


def get_loss_reconstruction(
    values,
    measurement: Measurement,
    beam_setup: BeamSetup,
    options: List[Options],
    data_dimensions: DataDimensions,
    viewer,
):
    global values_history
    global loss_values_history

    measurement.z01 = values[0]
    for option in options:
        option.padding.a0 = values[1]

    found, loss_local = check_history(values)

    if not found:
        current_result, loss_se_all, fov_size = reconstruct([measurement], beam_setup, options, data_dimensions, viewer)
        loss_local = loss_se_all[-1]
        values_history.append(values)
        loss_values_history.append(loss_local.cpu().numpy())
        loss = loss_local
    else:
        # Fix for buggy nelder-mead implementation when bounds are used and nelder-mead reflects to outside of bound interval. Punish this out-of-bound case
        loss = sys.float_info.max

    logging.loss(f"{'Find Focus - '}{'Loss: '}{loss_local: < 25}{'z01: '}{values[0]}{' a0: '}{values[1]}")

    return float(loss)


def get_loss_reconstruction_z01(
    z01,
    measurement: Measurement,
    beam_setup: BeamSetup,
    options: List[Options],
    data_dimensions: DataDimensions,
    viewer,
):
    return get_loss_reconstruction(
        [z01[0], options[0].padding.a0],
        measurement,
        beam_setup,
        options,
        data_dimensions,
        viewer,
    )


def get_loss_reconstruction_a0(
    a0,
    measurement: Measurement,
    beam_setup: BeamSetup,
    options: List[Options],
    data_dimensions: DataDimensions,
    viewer,
):
    return get_loss_reconstruction(
        [measurement.z01, a0[0]],
        measurement,
        beam_setup,
        options,
        data_dimensions,
        viewer,
    )


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

    for option in options:
        option.padding.a0 = 1.0

    for i in range(1):
        z01_guess = measurement.z01
        z01_bounds = measurement.z01_bounds
        z01_bounds_array = numpy.zeros((2, 1))
        z01_bounds_array[0, 0] = z01_bounds[0]
        z01_bounds_array[1, 0] = z01_bounds[1]

        a0_guess = options[0].padding.a0
        a0_bounds = [0.3, 1.7]
        a0_bounds_array = numpy.zeros((2, 1))
        a0_bounds_array[0, 0] = a0_bounds[0]
        a0_bounds_array[1, 0] = a0_bounds[1]
        a0_initial_simplex = numpy.zeros((2, 1))
        a0_initial_simplex[0, 0] = 0.6
        a0_initial_simplex[1, 0] = 1.4

        logging.info("Using a0 bounds " + str(a0_bounds))
        logging.info("Using z01 bounds " + str(z01_bounds))

        logging.debug("Starting optimizer")

        a0_found = scipy.optimize.minimize(
            get_loss_reconstruction_a0,
            a0_guess,
            args=(measurement, beam_setup, options, data_dimensions, viewer),
            method="Nelder-Mead",
            bounds=[a0_bounds],
            options={
                "xatol": 0.01,
                "fatol": 1000000.0,
                "initial_simplex": a0_initial_simplex,
            },
        )

        for option in options:
            option.padding.a0 = a0_found.x[0]

        z01_found = scipy.optimize.minimize(
            get_loss_reconstruction_z01,
            z01_guess,
            args=(measurement, beam_setup, options, data_dimensions, viewer),
            method="Nelder-Mead",
            bounds=[z01_bounds],
            options={
                "xatol": options[-1].z01_tol,
                "fatol": 1000000.0,
                "initial_simplex": z01_bounds_array,
            },
        )

        measurement.z01 = z01_found.x[0]

        logging.info("Found z01=" + str(z01_found.x[0]))
        logging.info("Found a0=" + str(a0_found.x[0]))

    z01_history, a0_history = list(zip(*values_history))
    z01_history = list(z01_history)
    a0_history = list(a0_history)

    logging.info("find_focus finished after " + str(len(loss_values_history)) + " iterations")

    return z01_found.x[0], z01_history, a0_found.x[0], a0_history, loss_values_history
