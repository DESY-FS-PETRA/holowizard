import logging

import torch
import gc

from holowizard.core.src.logging.logger import Logger
from holowizard.core.src.parameters.dask_options import DaskOptions
from holowizard.core.src.methods.reconstruction.modules.contexts.context_projection_singlestage import (
    ContextProjectionSingleStage,
)
from holowizard.core.src.methods.reconstruction.methods.asrm.reconstruct_single_stage import reconstruct_single_stage
from holowizard.core.src.api.viewer.loss_viewer import LossViewer
from holowizard.core.src.methods.reconstruction.utils.logging import log_params
from holowizard.core.src.misc.transform import crop_center


def task_run_asrm(dask_options_serialized: str, context_single_stage_serialized: str):
    dask_options: DaskOptions = DaskOptions.from_json(dask_options_serialized)
    context = ContextProjectionSingleStage.from_json(context_single_stage_serialized)
    context.nesterov_vt = torch.zeros(
        context.data_dimensions.total_size,
        device=context.torch_device,
        dtype=torch.cfloat,
    )

    Logger.current_log_level = Logger.level_num_image_info
    Logger.configure(dask_options.working_dir / "logs/", "asrm_" + context.key)

    logging.info("task_run_asrm for key " + context.key)

    log_params(context.measurements, context.beam_setup, [context.options], context.data_dimensions)

    context.viewer = [LossViewer()]

    reconstruct_single_stage(context)

    context.nesterov_vt = None
    context.oref_predicted = crop_center(context.oref_predicted, context.data_dimensions.fov_size)

    gc.collect()
    torch.cuda.empty_cache()
    return context.to_json()
