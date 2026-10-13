import logging
import torch
from typing import List

from holowizard.core.src.parameters.reco_params import RecoParams
from holowizard.core.src.methods.reconstruction.algorithms.asrm.reconstruct_asrm import reconstruct_asrm
from holowizard.core.src.misc.transform import crop_center
from holowizard.core.src.visualization.viewer import Viewer


def reconstruct(reco_params: RecoParams, viewer: List[Viewer] = []):
    for i in range(len(reco_params.measurements)):
        reco_params.measurements[i].data = torch.sqrt(reco_params.measurements[i].data)

    x_predicted, se_losses_all, fov = reconstruct_asrm(reco_params, viewer)
    x_predicted = crop_center(x_predicted, fov)

    logging.image_info(
        "result_phaseshift_cropped",
        crop_center(x_predicted.real.cpu().numpy(), reco_params.data_dimensions.fov_size),
    )
    logging.image_info(
        "result_absorption_cropped",
        crop_center(x_predicted.imag.cpu().numpy(), reco_params.data_dimensions.fov_size),
    )

    return x_predicted, se_losses_all
