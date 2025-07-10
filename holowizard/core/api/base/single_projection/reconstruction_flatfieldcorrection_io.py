import numpy as np
import logging
from typing import List

from holowizard.interface.parameters import RecoParams
from holowizard.interface.parameters import FlatfieldCorrectionParams
from holowizard.core.utils.transform import crop_center
import holowizard.core.utils.fileio as fileio
from holowizard.core.api.viewer import Viewer

from .reconstruction_flatfieldcorrection_i import (
    reconstruct as reconstruct_ffc_i,
)
from holowizard.core.api.base.default_load_data_callback import (
    default_load_data_callback,
)


def reconstruct(
    flatfield_correction_params: FlatfieldCorrectionParams,
    reco_params: RecoParams,
    glob_data_path: str,
    image_index: int,
    load_data_callback=default_load_data_callback,
    viewer: List[Viewer] = None,
):

    x_predicted, se_losses = reconstruct_ffc_i(
        flatfield_correction_params=flatfield_correction_params,
        reco_params=reco_params,
        glob_data_path=glob_data_path,
        image_index=image_index,
        load_data_callback=default_load_data_callback,
        viewer=viewer,
    )

    result_phaseshift = np.rot90(np.float32(np.real(x_predicted.cpu().numpy())))
    # result_absorption = np.float32(np.imag(x_predicted.cpu().numpy()))
    fileio.write_img_data(
        reco_params.output_path.split(".")[0] + ".tiff",
        crop_center(result_phaseshift, reco_params.data_dimensions.fov_size),
    )
    # fileio.write_img_data(reco_params.output_path.split(".")[0] + "_absorption.tiff",
    #                       crop_center(result_absorption,
    #                                   reco_params.data_dimensions.fov_size))
