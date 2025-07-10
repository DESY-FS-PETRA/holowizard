import torch
from typing import List

from holowizard.core.parameters import RecoParams
from holowizard.core.find_focus.find_focus_z01 import find_focus as find_focus_internal
from holowizard.core.api.viewer import Viewer


def find_focus(reco_params: RecoParams, viewer: List[Viewer] = None):
    for i in range(len(reco_params.measurements)):
        reco_params.measurements[i].data = torch.sqrt(reco_params.measurements[i].data)

    z01_guess, z01_values_history, loss_values_history = find_focus_internal(
        reco_params.measurements[0],
        reco_params.beam_setup,
        reco_params.reco_options,
        reco_params.data_dimensions,
        viewer,
    )

    return z01_guess, z01_values_history, loss_values_history
