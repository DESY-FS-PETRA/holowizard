import logging
from copy import deepcopy
from pathlib import Path

import holowizard.core
from holowizard.core.preprocessing.process_padding_options import (
    process_padding_options_noflipandpad,
)

from holowizard.core.reconstruction.utils.logging import log_preprocessed_params
from holowizard.core.reconstruction.controller.controller_projection_multistage import ControllerProjectionMultistage
from holowizard.core.reconstruction.contexts.context_projection_multistage import ContextProjectionMultistage
from holowizard.core.reconstruction.contexts.context_projection_singlestage import ContextProjectionSingleStage
from holowizard.core.utils.transform import resize_arrays_to_size

from holowizard.core.utils.memory import memory_breakdown


class ControllerProjectionMultistageLazy(ControllerProjectionMultistage):
    def __init__(self):
        return super().__init__()
    
    def set_stage(self, context_multi_stage: ContextProjectionMultistage, stage_index, verbose=True):
        self.check_successfull_initialization(context_multi_stage)
        self.check_valid_stage_index(context_multi_stage, stage_index)

        if context_multi_stage.current_stage == stage_index:
            return

        context_multi_stage.current_stage = stage_index
        if verbose is True:
            logging.comment(
                "Stage "
                + str(context_multi_stage.current_stage + 1)
                + "/"
                + str(len(context_multi_stage.reco_params.reco_options))
            )

        flag = False
        if context_multi_stage.current_projection_singlestage is None:
            flag = True
            context_multi_stage.current_projection_singlestage = ContextProjectionSingleStage(
                key=context_multi_stage.key, viewer=context_multi_stage.viewer
            )

        self.update_current_stage_options(context_multi_stage)

        if context_multi_stage.current_stage == 0 or (flag and context_multi_stage.current_stage > 0):
            self.create_effective_geometry_file(context_multi_stage)
            self.init_arrays(context_multi_stage)

        self.rescale_arrays(context_multi_stage)

        #print("debug")
        #memory_breakdown(context_multi_stage.current_projection_singlestage)

        self.update_propagator(context_multi_stage)

        if verbose is True:
            log_preprocessed_params(
                context_multi_stage.current_projection_singlestage.beam_setup,
                context_multi_stage.current_projection_singlestage.data_dimensions,
            )
    
    def init_arrays(self, context_multi_stage: ContextProjectionMultistage):
        self.init_arrays_for_shape(context_multi_stage, context_multi_stage.reco_params.data_dimensions.fov_size, init_nesterov_vt=False, init_probe_refractive=False)

    def rescale_arrays(self, context_multi_stage: ContextProjectionMultistage):

        (
            context_multi_stage.current_projection_singlestage.measurements,
            context_multi_stage.current_projection_singlestage.beam_setup,
            context_multi_stage.current_projection_singlestage.data_dimensions,
        ) = process_padding_options_noflipandpad(
            context_multi_stage.reco_params.measurements,
            context_multi_stage.reco_params.beam_setup,
            context_multi_stage.reco_params.data_dimensions,
            context_multi_stage.current_projection_singlestage.options.padding,
        )
 
        resized_arrays = resize_arrays_to_size(context_multi_stage.current_projection_singlestage.data_dimensions.fov_size,
            oref_predicted = context_multi_stage.current_projection_singlestage.oref_predicted,
            nesterov_vt = context_multi_stage.current_projection_singlestage.nesterov_vt,
        )

        context_multi_stage.current_projection_singlestage.oref_predicted = resized_arrays["oref_predicted"]
        context_multi_stage.current_projection_singlestage.nesterov_vt = None
        context_multi_stage.current_projection_singlestage.beam_setup.probe_refractive = None
        context_multi_stage.current_projection_singlestage.beam_setup.flat_field = None
