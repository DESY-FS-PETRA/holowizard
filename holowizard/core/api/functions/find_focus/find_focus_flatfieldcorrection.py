import logging
import pickle
from typing import List

from holowizard.core.parameters import RecoParams
from holowizard.core.parameters import FlatfieldCorrectionParams

from holowizard.core.preprocessing.correct_flatfield import correct_flatfield
from holowizard.core.api.viewer import Viewer
from holowizard.core.api.functions.find_focus.find_focus import (
    find_focus as find_focus_internal,
)


def find_focus(
    flatfield_correction_params: FlatfieldCorrectionParams,
    reco_params: RecoParams,
    viewer: List[Viewer] = None,
):
    logging.info("Load components from " + flatfield_correction_params.components_path)
    with open(flatfield_correction_params.components_path, "rb") as file:
        components_model = pickle.load(file)

    logging.info("raw", reco_params.measurements[0].data.cpu().numpy())

    logging.info("Correct flatfield")
    corrected_image = correct_flatfield(
        reco_params.measurements[0].data.float(), components_model
    )

    logging.info("flatfield_corrected", corrected_image.cpu().numpy())

    reco_params.measurements[0].data = corrected_image

    z01_guess, z01_values_history, loss_values_history = find_focus_internal(
        reco_params, viewer
    )

    return z01_guess, z01_values_history, loss_values_history
