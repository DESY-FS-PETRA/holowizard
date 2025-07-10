from typing import List
import logging
import pickle

from holowizard.core.parameters.reco_params import RecoParams
from holowizard.core.parameters.flatfield_correction_params import FlatfieldCorrectionParams
from holowizard.core.find_focus.find_focus_z01_a0_orthogonal_search import (
    find_focus as find_focus_internal,
)
from holowizard.core.reconstruction.viewer import Viewer
from holowizard.core.preprocessing.correct_flatfield import correct_flatfield


def find_focus(
    flatfield_correction_params: FlatfieldCorrectionParams,
    reco_params: RecoParams,
    viewer: List[Viewer] = None,
):
    logging.info("Load components from " + flatfield_correction_params.components_path)
    with open(flatfield_correction_params.components_path, "rb") as file:
        components_model = pickle.load(file)

    logging.image_info("raw", reco_params.measurements[0].data.cpu().numpy())

    logging.info("Correct flatfield")
    corrected_image = correct_flatfield(
        reco_params.measurements[0].data.float(), components_model
    )

    logging.image_info("flatfield_corrected", corrected_image.cpu().numpy())

    for i in range(len(reco_params.measurements)):
        reco_params.measurements[i].data = torch.sqrt(reco_params.measurements[i].data)

    (
        z01_guess,
        z01_values_history,
        a0_guess,
        a0_values_history,
        loss_values_history,
    ) = find_focus_internal(
        reco_params.measurements[0],
        reco_params.beam_setup,
        reco_params.reco_options,
        reco_params.data_dimensions,
        viewer,
    )

    return (
        z01_guess,
        z01_values_history,
        a0_guess,
        a0_values_history,
        loss_values_history,
    )
