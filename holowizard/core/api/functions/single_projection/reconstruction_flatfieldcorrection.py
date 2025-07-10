import logging
import pickle
from typing import List

from holowizard.core.parameters import RecoParams
from holowizard.core.parameters import FlatfieldCorrectionParams
from holowizard.core.preprocessing.correct_flatfield import correct_flatfield
from holowizard.core.api.viewer import Viewer

from holowizard.core.api.functions.single_projection.reconstruction import (
    reconstruct as reconstruct_base,
)


def reconstruct(
    flatfield_correction_params: FlatfieldCorrectionParams,
    reco_params: RecoParams,
    viewer: List[Viewer] = None,
):
    logging.info("Load components from " + flatfield_correction_params.components_path)
    with open(flatfield_correction_params.components_path, "rb") as file:
        components_model = pickle.load(file)

    for i in range(len(reco_params.measurements)):
        logging.image_info(
            "raw_" + str(i), reco_params.measurements[i].data.cpu().numpy()
        )

        logging.info("Correct flatfield Nr." + str(i))
        corrected_image = correct_flatfield(
            reco_params.measurements[i].data.float(), components_model
        )

        logging.image_info(
            "flatfield_corrected_" + str(i), corrected_image.cpu().numpy()
        )

        reco_params.measurements[i].data = corrected_image

    x_predicted, se_losses_all = reconstruct_base(reco_params, viewer=viewer)

    return x_predicted, se_losses_all
