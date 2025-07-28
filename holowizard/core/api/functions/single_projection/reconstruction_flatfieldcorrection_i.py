import logging
import pickle
from typing import List

from holowizard.core.parameters.reco_params import RecoParams
from holowizard.core.parameters.flatfield_correction_params import FlatfieldCorrectionParams
from holowizard.core.preprocessing.correct_flatfield import correct_flatfield
from holowizard.core.reconstruction.viewer import Viewer
from holowizard.core.api.functions.single_projection.reconstruction import reconstruct as reconstruct_base
from holowizard.core.api.functions.default_load_data_callback import default_load_data_callback


def reconstruct(
    flatfield_correction_params: FlatfieldCorrectionParams,
    reco_params: RecoParams,
    glob_data_path=None,
    image_index=None,
    load_data_callback=default_load_data_callback,
    viewer: List[Viewer] = None,
):
    if load_data_callback is None:
        raise RuntimeError(
            "Try to load image with index " + str(image_index),
            " but data callback is None",
        )

    if image_index is None:
        for i in range(len(reco_params.measurements)):
            data = load_data_callback(reco_params.measurements[i].data_path)
            reco_params.measurements[i].data = data
    else:
        data_path_loaded, data = load_data_callback(glob_data_path, image_index)
        reco_params.measurements[0].data_path = data_path_loaded
        reco_params.measurements[0].data = data

    logging.info("Load components from " + flatfield_correction_params.components_path)
    with open(flatfield_correction_params.components_path, "rb") as file:
        components_model = pickle.load(file)

    for i in range(len(reco_params.measurements)):
        logging.info("raw_" + str(i), reco_params.measurements[i].data.cpu().numpy())

        logging.info("Correct flatfield Nr." + str(i))
        corrected_image = correct_flatfield(reco_params.measurements[i].data.float(), components_model)

        logging.info("flatfield_corrected_" + str(i), corrected_image.cpu().numpy())

        reco_params.measurements[i].data = corrected_image

    x_predicted, se_losses_all = reconstruct_base(reco_params, viewer=viewer)

    return x_predicted, se_losses_all
