import logging
from typing import List

from holowizard.core.parameters import RecoParams
from holowizard.core.parameters import FlatfieldCorrectionParams

from holowizard.core.api.viewer import Viewer
from holowizard.core.api.functions.find_focus.find_focus_flatfieldcorrection import (
    find_focus as find_focus_internal,
)
from holowizard.core.api.functions.default_load_data_callback import (
    default_load_data_callback,
)


def find_focus(
    glob_data_path,
    flatfield_correction_params: FlatfieldCorrectionParams,
    reco_params: RecoParams,
    image_index,
    load_data_callback=default_load_data_callback,
    viewer: List[Viewer] = None,
):
    data_path_loaded, data = load_data_callback(glob_data_path, image_index)

    reco_params.measurements[0].data_path = data_path_loaded
    reco_params.measurements[0].data = data

    logging.debug("loaded", data)

    z01_guess, z01_values_history, loss_values_history = find_focus_internal(
        flatfield_correction_params, reco_params, viewer
    )

    return z01_guess, z01_values_history, loss_values_history
