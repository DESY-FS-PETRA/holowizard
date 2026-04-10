import logging
import pickle
from typing import List

from holowizard.core.parameters.reco_params import RecoParams
from holowizard.core.parameters.flatfield_correction_params import FlatfieldCorrectionParams
from holowizard.core.preprocessing.correct_flatfield import correct_flatfield
from holowizard.core.reconstruction.viewer import Viewer
from holowizard.core.api.functions.find_focus.find_focus import find_focus as find_focus_internal
from holowizard.core.reconstruction.plotter import Plotter


def find_focus(
    flatfield_correction_params: FlatfieldCorrectionParams,
    reco_params: RecoParams,
    viewer: List[Viewer] = None,
    plotter: List[Plotter] = None,
):
    logging.info("Load components from " + flatfield_correction_params.components_path)
    with open(flatfield_correction_params.components_path, "rb") as file:
        components_model = pickle.load(file)

    logging.image_debug("raw", reco_params.measurements[0].data.cpu().numpy())

    logging.info("Correct flatfield")
    corrected_image = correct_flatfield(reco_params.measurements[0].data.float(), components_model)

    logging.image_debug("flatfield_corrected", corrected_image.cpu().numpy())

    reco_params.measurements[0].data = corrected_image

    z01_guess, z01_values_history, loss_values_history = find_focus_internal(reco_params, viewer, plotter)

    return z01_guess, z01_values_history, loss_values_history
