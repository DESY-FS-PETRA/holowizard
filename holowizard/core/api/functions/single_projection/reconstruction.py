import logging
import torch
from typing import List

from holowizard.core.parameters.reco_params import RecoParams

from holowizard.core.reconstruction.single_projection.reconstruct_multistage import (
    reconstruct as reco_multistage,
)
from holowizard.core.utils.transform import crop_center
from holowizard.core.reconstruction.viewer import Viewer


def reconstruct(reco_params: RecoParams, viewer: List[Viewer] = []):
    for i in range(len(reco_params.measurements)):
        reco_params.measurements[i].data = torch.sqrt(reco_params.measurements[i].data)

    x_predicted, se_losses_all, fov = reco_multistage(
        measurements=reco_params.measurements,
        beam_setup=reco_params.beam_setup,
        options=reco_params.reco_options,
        data_dimensions=reco_params.data_dimensions,
        viewer=viewer,
    )

    x_predicted = crop_center(x_predicted, fov)

    logging.image_info(
        "result_phaseshift_cropped",
        crop_center(
            x_predicted.real.cpu().numpy(), reco_params.data_dimensions.fov_size
        ),
    )
    logging.image_info(
        "result_absorption_cropped",
        crop_center(
            x_predicted.imag.cpu().numpy(), reco_params.data_dimensions.fov_size
        ),
    )

    return x_predicted, se_losses_all
