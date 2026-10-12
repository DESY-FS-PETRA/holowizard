import torch
from typing import List
import logging

from holowizard.core.reconstruction.utils.logging import log_input, log_params, log_results
from holowizard.core.reconstruction.controller.controller_projection_multistage import ControllerProjectionMultistage
from holowizard.core.reconstruction.viewer import Viewer
from holowizard.core.reconstruction.methods.asrm.reconstruct_single_stage import reconstruct_single_stage
import holowizard.core

if "cuda" in holowizard.core.torch_running_device_name:
    import cupy as cp


def reconstruct_asrm(
    reco_params,
    viewer: List[Viewer],
):
    log_input(reco_params.measurements)

    logging.comment("Initialization")

    log_params(reco_params.measurements, reco_params.beam_setup, reco_params.reco_options, reco_params.data_dimensions)

    controller = ControllerProjectionMultistage()
    context_multi_stage = controller.create_multistage_context(reco_params, viewer=viewer)

    for options_index in range(len(reco_params.reco_options)):
        controller.set_stage(context_multi_stage, options_index)

        logging.info(
            f"{'Downsampling':<17}{str(context_multi_stage.current_projection_singlestage.options.padding.down_sampling_factor)}"
        )

        controller.prepare_next_reco_block(context_multi_stage)
        reconstruct_single_stage(context_multi_stage.current_projection_singlestage)

        if "cuda" in holowizard.core.torch_running_device_name:
            torch.cuda.empty_cache()  # should empty device cache
            cp.get_default_memory_pool().free_all_blocks()

        options_index < (len(reco_params.reco_options) - 1) and log_results(
            "snapshot_x" + str(context_multi_stage.current_projection_singlestage.options.padding.down_sampling_factor),
            [context_multi_stage],
        )

    log_results("result_x" + str(reco_params.reco_options[-1].padding.down_sampling_factor), [context_multi_stage])

    return (
        context_multi_stage.current_projection_singlestage.oref_predicted,
        context_multi_stage.current_projection_singlestage.se_losses_all,
        context_multi_stage.current_projection_singlestage.data_dimensions.fov_size,
    )
