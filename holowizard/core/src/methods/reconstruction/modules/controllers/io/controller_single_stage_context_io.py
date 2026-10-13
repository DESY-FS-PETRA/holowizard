from typing import List
import os
import torch
from pathlib import Path
import traceback
import time

import holowizard.core.src
from holowizard.core.src.misc.fileio import check_file_exists
from holowizard.core.src.misc.torchio import write_torch_array
from holowizard.core.src.methods.reconstruction.modules.contexts.context_projection_singlestage import ContextProjectionSingleStage
from holowizard.core.src.methods.reconstruction.controller.io.controller_beam_setup_io import ControllerBeamSetupIO

class ControllerSingleStageContextIO:
    def __init__(self):
        self.controller_beam_setup_io = ControllerBeamSetupIO()
        return

    def read_file_with_timeout(self,file_name,tries=5,pause_between_tries=1):
        data = None
        for i in range(tries):
            try:
                check_file_exists(file_path=file_name)
                data = torch.load(file_name, map_location=holowizard.core.torch_running_device)
                break
            except Exception as exception:
                if (i+1) <tries:
                    time.sleep(pause_between_tries)
                    continue
                tb = traceback.format_exc()
                raise RuntimeError(f"Could not read file with name {file_name}: {tb}") from exception

        return data

    def define_io_paths(self,context_single_stage: ContextProjectionSingleStage, root:Path):
        context_single_stage.beam_setup.probe_path = self.controller_beam_setup_io.get_default_probe_path(root)
        context_single_stage.oref_predicted_path = root / Path("oref_predicted_" + context_single_stage.key + ".pt")

        for i in range(len(context_single_stage.measurements)):
            context_single_stage.measurements[i].data_path = root / Path("measurement_" + str(i) + "_" + context_single_stage.key + ".pt")

    def swap_arrays_from_ram_to_file(self,context_single_stage: ContextProjectionSingleStage):
        self.swap_oref_predicted_from_ram_to_file(context_single_stage)
        self.swap_probe_refractive_from_ram_to_file(context_single_stage)
        self.swap_measurements_from_ram_to_file(context_single_stage)

    def swap_arrays_from_file_to_ram(self,context_single_stage: ContextProjectionSingleStage):
        self.swap_oref_predicted_from_file_to_ram(context_single_stage)
        self.swap_probe_refractive_from_file_to_ram(context_single_stage)
        self.swap_measurements_from_file_to_ram(context_single_stage)

    def swap_probe_refractive_from_ram_to_file(self,context_single_stage: ContextProjectionSingleStage):
        self.controller_beam_setup_io.swap_probe_refractive_from_ram_to_file(context_single_stage.beam_setup)

    def swap_oref_predicted_from_ram_to_file(self,context_single_stage: ContextProjectionSingleStage):
        write_torch_array(context_single_stage.oref_predicted, context_single_stage.oref_predicted_path)
        context_single_stage.oref_predicted = None
        
    def swap_measurements_from_ram_to_file(self,context_single_stage: ContextProjectionSingleStage):
        for measurement in context_single_stage.measurements:
            write_torch_array(measurement.data, measurement.data_path)
            measurement.data = None

    def swap_probe_refractive_from_file_to_ram(self,context_single_stage: ContextProjectionSingleStage):
        self.controller_beam_setup_io.swap_probe_refractive_from_file_to_ram(context_single_stage.beam_setup)
    
    def swap_oref_predicted_from_file_to_ram(self,context_single_stage: ContextProjectionSingleStage):
        file_name = context_single_stage.oref_predicted_path
        context_single_stage.oref_predicted = self.read_file_with_timeout(file_name)
        
        
    def swap_measurements_from_file_to_ram(self,context_single_stage: ContextProjectionSingleStage):
        for measurement in context_single_stage.measurements:
            file_name = measurement.data_path 
            measurement.data = self.read_file_with_timeout(file_name)
            
    def clean_arrays(self,context_single_stage: ContextProjectionSingleStage):
        for measurement in context_single_stage.measurements:
            measurement.data = None
        context_single_stage.oref_predicted = None
        context_single_stage.beam_setup.probe_refractive = None
