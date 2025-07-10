import numpy as np
from typing import List

from holowizard.core.parameters.reco_params import RecoParams
from holowizard.core.reconstruction.viewer import Viewer
import holowizard.core.utils.fileio as fileio

from holowizard.core.api.functions.default_load_data_callback import (
    default_load_data_callback,
)
from holowizard.core.api.functions.single_projection.reconstruction import (
    reconstruct as reconstruct_base,
)


def single_reconstruction(
    reco_params: RecoParams,
    glob_data_path: str,
    image_index: int,
    load_data_callback=default_load_data_callback,
    viewer: List[Viewer] = None,
):

    if image_index is None:
        for i in range(len(reco_params.measurements)):
            data = load_data_callback(reco_params.measurements[i].data_path)
            reco_params.measurements[i].data = data
    else:
        data_path_loaded, data = load_data_callback(glob_data_path, image_index)
        reco_params.measurements[0].data_path = data_path_loaded
        reco_params.measurements[0].data = data

    x_predicted, se_losses_all = reconstruct_base(reco_params, viewer=viewer)

    result_phaseshift = np.float32(np.real(x_predicted.cpu().numpy()))
    result_absorption = np.float32(np.imag(x_predicted.cpu().numpy()))
    fileio.write_img_data(
        reco_params.output_path.split(".")[0] + "_phaseshift.tiff", result_phaseshift
    )
    fileio.write_img_data(
        reco_params.output_path.split(".")[0] + "_absorption.tiff", result_absorption
    )
