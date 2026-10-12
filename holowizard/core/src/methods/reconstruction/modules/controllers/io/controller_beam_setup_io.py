from typing import List
import os
import torch
from pathlib import Path
import traceback

import holowizard.core.src
from holowizard.core.src.parameters.beam_setup import BeamSetup
from holowizard.core.src.utils.fileio import check_file_exists
from holowizard.core.src.utils.torchio import write_torch_array

class ControllerBeamSetupIO:
    def __init__(self, default_postfix=None):
        self._default_postfix = Path("probe_refractive.pt")

        return
    
    def get_default_probe_path(self, root:Path):
        return root / self._default_postfix

    def write_probe_refractive(self, beam_setup: BeamSetup):
        write_torch_array(beam_setup.probe_refractive, beam_setup.probe_path)
        
    def swap_probe_refractive_from_ram_to_file(self,beam_setup: BeamSetup):
        write_torch_array(beam_setup.probe_refractive, beam_setup.probe_path)
        beam_setup.probe_refractive = None

    def swap_probe_refractive_from_file_to_ram(self,beam_setup: BeamSetup):
        file_path = beam_setup.probe_path
        
        try:
            check_file_exists(file_path=file_path)
            beam_setup.probe_refractive = torch.load(file_path).to(holowizard.core.src.torch_running_device)
        except Exception as exception:
            tb = traceback.format_exc()
            raise RuntimeError(f"Could not read probe file with name {file_path}: {tb}") from exception