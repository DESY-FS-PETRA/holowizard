import torch
from typing import List

from holowizard.core.parameters.reco_params import RecoParams
from holowizard.core.find_focus.find_focus_z01_a0_orthogonal_search import (
    find_focus as find_focus_internal,
)
from holowizard.core.reconstruction.viewer import Viewer


def find_focus(reco_params: RecoParams, viewer: List[Viewer] = None):
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
