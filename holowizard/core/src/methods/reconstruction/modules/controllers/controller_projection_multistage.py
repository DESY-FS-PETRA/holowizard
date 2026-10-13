import numpy as np
import torch
from typing import List
from copy import deepcopy
import logging
from enum import Enum
import os

import holowizard.core
from holowizard.core.src.methods.preprocessing.process_padding_options import (
    process_padding_options,
)
from holowizard.core.src.parameters.measurement import Measurement
from holowizard.core.src.parameters.beam_setup import BeamSetup
from holowizard.core.src.parameters.reco_params import RecoParams
from holowizard.core.src.visualization.viewer.viewer import Viewer
from holowizard.core.src.methods.reconstruction.algorithms.asrm.utils.logging import Logger, log_preprocessed_params
from holowizard.core.src.misc.transform import resize_arrays_to_size
from holowizard.core.src.methods.reconstruction.modules.contexts.context_projection_singlestage import ContextProjectionSingleStage
from holowizard.core.src.methods.reconstruction.modules.contexts.context_projection_multistage import ContextProjectionMultistage
from holowizard.core.src.models.beam_setups.cone_beam import ConeBeam


class ControllerProjectionMultistage:
    def create_multistage_context(
        self, reco_params: RecoParams, viewer: List[Viewer] = None, torch_device=holowizard.core.torch_running_device
    ):
        context_multistage = ContextProjectionMultistage(
            current_projection_singlestage=None,
            key=str(reco_params.measurements[0].angle),
            reco_params=reco_params,
            viewer=viewer,
            current_global_beam_setup=None,
            current_options=None,
            current_data_dimensions=None,
            current_stage=-1,
            current_iter_offset=0,
            torch_device=torch_device,
            se_losses_all=None,
        )

        return context_multistage

    def create_effective_geometry_file(self, context_multi_stage: ContextProjectionMultistage):
        string_list = []

        string_list.append(
            f"{'z02':<17}{round(context_multi_stage.reco_params.beam_setup.z02, int(np.log10(BeamSetup.unit_z02()[1])))} "
            + BeamSetup.unit_z02()[0]
        )
        string_list.append(
            f"{'Energy':<17}{round(context_multi_stage.reco_params.beam_setup.energy, int(np.log10(BeamSetup.unit_energy()[1])))} "
            + BeamSetup.unit_energy()[0]
        )
        for i in range(len(context_multi_stage.reco_params.measurements)):
            lam, M, dx_eff, z_eff, fr_eff = ConeBeam.get_effective_geometry(
                context_multi_stage.reco_params.beam_setup, context_multi_stage.reco_params.measurements[i]
            )
            string_list.append("Distance " + str(i) + ":")
            string_list.append(
                f"{'z01':<17}{round(context_multi_stage.reco_params.measurements[i].z01, int(np.log10(Measurement.unit_z01()[1])))} "
                + Measurement.unit_z01()[0]
            )
            string_list.append(f"{'Magnification':<17}{round(M, 3)}")
            string_list.append(
                f"{'Effective dx':<17}{round(dx_eff, int(np.log10(BeamSetup.unit_px_size()[1])))} "
                + BeamSetup.unit_px_size()[0]
            )
            string_list.append(
                f"{'Effective z12':<17}{round(z_eff, int(np.log10(Measurement.unit_z01()[1])))} "
                + Measurement.unit_z01()[0]
            )
            string_list.append(f"{'Fresnel Number':<17}{fr_eff}")

        Logger.custom_string_file("geometry", string_list)

    def check_successfull_initialization(self, context_multi_stage: ContextProjectionMultistage):
        if None in [
            context_multi_stage.reco_params,
            context_multi_stage.reco_params.measurements,
            context_multi_stage.reco_params.beam_setup,
            context_multi_stage.reco_params.data_dimensions,
        ]:
            raise ValueError("Host context not complete.")

    def check_valid_stage_index(self, context_multi_stage: ContextProjectionMultistage, stage_index):
        if stage_index >= len(context_multi_stage.reco_params.reco_options):
            raise ValueError(
                "Invalid stage index " + str(stage_index),
                ". Can be at maximum " + str(len(context_multi_stage.reco_params.reco_options) - 1) + ".",
            )

    def update_current_stage_options(self, context_multi_stage: ContextProjectionMultistage):
        context_multi_stage.current_projection_singlestage.options = context_multi_stage.reco_params.reco_options[
            context_multi_stage.current_stage
        ]

        context_multi_stage.current_projection_singlestage.absorption_min = torch.tensor(
            context_multi_stage.current_projection_singlestage.options.regularization_object.values_min.imag,
            device=context_multi_stage.torch_device,
            dtype=torch.float,
        )
        context_multi_stage.current_projection_singlestage.absorption_max = torch.tensor(
            context_multi_stage.current_projection_singlestage.options.regularization_object.values_max.imag,
            device=context_multi_stage.torch_device,
            dtype=torch.float,
        )

        context_multi_stage.current_projection_singlestage.phaseshift_min = torch.tensor(
            context_multi_stage.current_projection_singlestage.options.regularization_object.values_min.real,
            device=context_multi_stage.torch_device,
            dtype=torch.float,
        )
        context_multi_stage.current_projection_singlestage.phaseshift_max = torch.tensor(
            context_multi_stage.current_projection_singlestage.options.regularization_object.values_max.real,
            device=context_multi_stage.torch_device,
            dtype=torch.float,
        )

    def update_loss_array_and_offset(self, context_multi_stage: ContextProjectionMultistage):
        extension_size = context_multi_stage.current_projection_singlestage.options.regularization_object.iterations
        
        context_multi_stage.current_projection_singlestage.se_losses_all = torch.cat(
            (
                context_multi_stage.current_projection_singlestage.se_losses_all,
                torch.zeros(
                    extension_size,
                    device=context_multi_stage.torch_device,
                    dtype=torch.float,
                ),
            )
        )


        context_multi_stage.current_projection_singlestage.current_iter_offset = len(context_multi_stage.current_projection_singlestage.se_losses_all ) - extension_size

    def update_propagator(self, context_multi_stage: ContextProjectionMultistage):
        context_multi_stage.current_projection_singlestage.fresnel_numbers = [
            ConeBeam.get_fr(
                context_multi_stage.current_projection_singlestage.beam_setup,
                context_multi_stage.current_projection_singlestage.measurements[distance],
                verbose=False,
            )
            for distance in range(len(context_multi_stage.current_projection_singlestage.measurements))
        ]

    def increment_block(self, context_multi_stage: ContextProjectionMultistage):
        
        self.update_loss_array_and_offset(context_multi_stage=context_multi_stage)

    def init_arrays(self, context_multi_stage: ContextProjectionMultistage):
        self.init_arrays_for_shape(context_multi_stage, context_multi_stage.reco_params.data_dimensions.total_size)

    def init_arrays_for_shape(self, context_multi_stage: ContextProjectionMultistage, shape, init_se_losses=True,init_oref_predicted=True, init_nesterov_vt=True, init_probe_refractive = True):
        if context_multi_stage.current_projection_singlestage.se_losses_all is None and init_se_losses:
            context_multi_stage.current_projection_singlestage.se_losses_all = torch.empty(
                0, device=context_multi_stage.torch_device
            )

        if context_multi_stage.current_projection_singlestage.oref_predicted is None and init_oref_predicted:
            if context_multi_stage.reco_params.initial_guess is not None:
                context_multi_stage.current_projection_singlestage.oref_predicted = (
                    context_multi_stage.reco_params.initial_guess
                )
            else:
                context_multi_stage.current_projection_singlestage.oref_predicted = torch.zeros(
                    shape,
                    device=context_multi_stage.torch_device,
                    dtype=torch.cfloat,
                )

        if context_multi_stage.current_projection_singlestage.nesterov_vt is None and init_nesterov_vt:
            context_multi_stage.current_projection_singlestage.nesterov_vt = torch.zeros(
                shape,
                device=context_multi_stage.torch_device,
                dtype=torch.cfloat,
            )

        if context_multi_stage.reco_params.beam_setup.probe_refractive is None and init_probe_refractive:
            context_multi_stage.reco_params.beam_setup.probe_refractive = torch.zeros(
                shape,
                device=context_multi_stage.torch_device,
                dtype=torch.cfloat,
            )

    def rescale_arrays(self, context_multi_stage: ContextProjectionMultistage):

        (
            context_multi_stage.current_projection_singlestage.measurements,
            context_multi_stage.current_projection_singlestage.beam_setup,
            context_multi_stage.current_projection_singlestage.data_dimensions,
        ) = process_padding_options(
            context_multi_stage.reco_params.measurements,
            context_multi_stage.reco_params.beam_setup,
            context_multi_stage.reco_params.data_dimensions,
            context_multi_stage.current_projection_singlestage.options.padding,
        )

        resized_arrays = resize_arrays_to_size(context_multi_stage.current_projection_singlestage.data_dimensions.total_size,
            oref_predicted = context_multi_stage.current_projection_singlestage.oref_predicted,
            nesterov_vt = context_multi_stage.current_projection_singlestage.nesterov_vt,
            probe_refractive = context_multi_stage.current_projection_singlestage.beam_setup.probe_refractive,
        )

        context_multi_stage.current_projection_singlestage.oref_predicted = resized_arrays["oref_predicted"]
        context_multi_stage.current_projection_singlestage.nesterov_vt = resized_arrays["nesterov_vt"]
        context_multi_stage.current_projection_singlestage.beam_setup.probe_refractive = resized_arrays["probe_refractive"]


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

        if context_multi_stage.current_projection_singlestage is None:
            context_multi_stage.current_projection_singlestage = ContextProjectionSingleStage(
                key=context_multi_stage.key, viewer=context_multi_stage.viewer
            )

        self.update_current_stage_options(context_multi_stage)

        if context_multi_stage.current_stage == 0:
            self.create_effective_geometry_file(context_multi_stage)
            self.init_arrays(context_multi_stage)

        self.rescale_arrays(context_multi_stage)

        self.update_propagator(context_multi_stage)

        if verbose is True:
            log_preprocessed_params(
                context_multi_stage.current_projection_singlestage.beam_setup,
                context_multi_stage.current_projection_singlestage.data_dimensions,
            )

    def prepare_next_reco_block(self,context_multi_stage:ContextProjectionMultistage):
        self.update_loss_array_and_offset(context_multi_stage)