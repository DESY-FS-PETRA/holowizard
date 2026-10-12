import torch
from dataclasses import dataclass
from typing import List
from abc import ABC

from holowizard.core.src.parameters.reco_params import RecoParams
from holowizard.core.src.parameters.beam_setup import BeamSetup
from holowizard.core.src.parameters.options import Options
from holowizard.core.src.parameters.data_dimensions import DataDimensions
from holowizard.core.src.methods.reconstruction.viewer import Viewer


@dataclass(init=True)
class AbstractContextMultistage(ABC):
    key: str
    reco_params: RecoParams
    viewer: List[Viewer]
    current_options: Options
    current_global_beam_setup: BeamSetup
    current_data_dimensions: DataDimensions

    current_stage: int
    current_iter_offset: int
    torch_device: torch.device

    se_losses_all: torch.Tensor
