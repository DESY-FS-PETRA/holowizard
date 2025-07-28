import logging
import torch
from typing import List

from holowizard.core.parameters.reco_params import RecoParams
from holowizard.core.parameters.dask_options import DaskOptions
from holowizard.core.reconstruction.probe_retrieval.host_reconstruction import reconstruct as reconstruct_base
from holowizard.core.utils.transform import crop_center
from holowizard.core.reconstruction.viewer import Viewer


def reconstruct(reco_params: RecoParams, dask_options: DaskOptions, viewer: List[Viewer] = None):
    logging.info("Reconstructing " + str(len(reco_params.measurements)) + " holograms.")

    for i in range(len(reco_params.measurements)):
        reco_params.measurements[i].data = torch.sqrt(reco_params.measurements[i].data)

    x_predicted, se_losses_all, fov = reconstruct_base(
        measurements=reco_params.measurements,
        beam_setup=reco_params.beam_setup,
        options=reco_params.reco_options,
        data_dimensions=reco_params.data_dimensions,
        viewer=viewer,
        dask_options=dask_options,
    )

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
