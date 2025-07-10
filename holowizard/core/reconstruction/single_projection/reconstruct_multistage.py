import torch
from typing import List

from holowizard.interface.parameters.measurement import Measurement
from holowizard.interface.parameters.beam_setup import BeamSetup
from holowizard.interface.parameters.options import Options
from holowizard.interface.parameters.data_dimensions import DataDimensions
from holowizard.core.reconstruction.logging import *
from holowizard.core.reconstruction.single_projection.context import Context
from holowizard.core.reconstruction.viewer import Viewer
from holowizard.core.reconstruction.single_projection.reconstruct import (
    reconstruct as reconstruct_stage,
)
import holowizard.core

if "cuda" in holowizard.core.torch_running_device_name:
    import cupy as cp


def reconstruct(
    measurements: List[Measurement],
    beam_setup: BeamSetup,
    options: List[Options],
    data_dimensions: DataDimensions,
    viewer: List[Viewer],
):
    log_input(measurements)

    logging.comment("Initialization")

    log_params(measurements, beam_setup, options, data_dimensions)

    context = Context(
        viewer=viewer,
        options=options,
        measurements=measurements,
        beam_setup=beam_setup,
        data_dimensions=data_dimensions,
    )

    for options_index in range(len(options)):
        context.set_stage(options_index)

        logging.info(
            f"{'Downsampling':<17}{str(context.current_options.padding.down_sampling_factor)}"
        )

        reconstruct_stage(context)

        context.current_iter_offset = (
            context.current_iter_offset
            + context.current_options.regularization_object.iterations
        )

        if "cuda" in holowizard.core.torch_running_device_name:
            torch.cuda.empty_cache()  # should empty device cache
            cp.get_default_memory_pool().free_all_blocks()

        options_index < (len(options) - 1) and log_results(
            "snapshot_x" + str(context.current_options.padding.down_sampling_factor),
            [context.oref_predicted],
            context.data_dimensions,
        )

    log_results(
        "result_x" + str(options[-1].padding.down_sampling_factor),
        [context.oref_predicted],
        context.data_dimensions,
    )

    return (
        context.oref_predicted,
        context.se_losses_all,
        context.data_dimensions.fov_size,
    )
