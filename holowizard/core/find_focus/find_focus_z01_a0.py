import logging

import numpy as np
from scipy import optimize
import sys
from typing import List
import math

from holowizard.core.reconstruction.single_projection.reconstruct_multistage import (
    reconstruct,
)
from holowizard.interface.parameters.options import Options
from holowizard.interface.parameters.measurement import Measurement
from holowizard.interface.parameters.data_dimensions import DataDimensions
from holowizard.interface.parameters.beam_setup import BeamSetup

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
    z01_order,
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
        option.padding.a0 = values[1] / z01_order

    found, loss_local = check_history(values)

    if not found:
        current_result, loss_se_all, fov_size = reconstruct(
            [measurement], beam_setup, options, data_dimensions, viewer
        )
        loss_local = loss_se_all[-1]
        values_history.append(values)
        loss_values_history.append(loss_local.cpu().numpy())
        loss = loss_local
    else:
        # Fix for buggy nelder-mead implementation when bounds are used and nelder-mead reflects to outside of bound interval. Punish this out-of-bound case
        loss = sys.float_info.max

    logging.loss(
        f"{'Find Focus - '}{'Loss: '}{loss_local: < 25}{'z01: '}{values[0]}{' a0: '}{values[1]/z01_order}"
    )

    return float(loss)


def find_focus(
    measurement: Measurement,
    beam_setup: BeamSetup,
    options: List[Options],
    data_dimensions: DataDimensions,
    viewer,
):
    global values_history
    global loss_values_history

    values_history = []
    loss_values_history = []

    z01_guess = measurement.z01
    z01_order = 10 ** math.floor(math.log10(z01_guess))

    a0_guess = options[0].padding.a0
    bounds_z01 = measurement.z01_bounds
    bounds_a0 = ((0.3) * z01_order, (1.8) * z01_order)
    a0_guess = a0_guess * z01_order

    initial_simplex = np.zeros((3, 2))
    initial_simplex[0] = [
        z01_guess + (bounds_z01[0] - z01_guess) / 2,
        (1.4) * z01_order,
    ]
    initial_simplex[1] = [
        z01_guess + (bounds_z01[1] - z01_guess) / 2,
        (0.6) * z01_order,
    ]
    initial_simplex[2] = [
        z01_guess + (bounds_z01[1] - z01_guess) / 2,
        (1.4) * z01_order,
    ]

    logging.info("Using a0 bounds " + str(bounds_a0))
    logging.info("Using z01 bounds " + str(bounds_z01))

    logging.debug("Starting optimizer (squared error)")
    found_values = optimize.minimize(
        get_loss_reconstruction,
        np.array([z01_guess, a0_guess]),
        args=(z01_order, measurement, beam_setup, options, data_dimensions, viewer),
        method="Nelder-Mead",
        bounds=[bounds_z01, bounds_a0],
        options={
            "xatol": options[-1].z01_tol,
            "fatol": 1000000.0,
            "initial_simplex": initial_simplex,
            "adaptive": True,
        },
    )

    z01_history, a0_history = list(zip(*values_history))
    z01_history = list(z01_history)
    a0_history = list(a0_history)

    for i in range(len(a0_history)):
        a0_history[i] = a0_history[i] / z01_order

    found_values.x[1] = found_values.x[1] / z01_order

    logging.info("Found z01=" + str(found_values.x[0]))
    logging.info("Found a0=" + str(found_values.x[1]))
    logging.info("find_focus finished")

    return (
        found_values.x[0],
        z01_history,
        found_values.x[1],
        a0_history,
        loss_values_history,
    )
