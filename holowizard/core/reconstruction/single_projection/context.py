import numpy as np
import torch
from typing import List
from copy import deepcopy

import holowizard.core
from holowizard.core.preprocessing.process_padding_options import (
    process_padding_options,
)
from holowizard.core.parameters.measurement import Measurement
from holowizard.core.parameters.beam_setup import BeamSetup
from holowizard.core.parameters.data_dimensions import DataDimensions
from holowizard.core.parameters.options import Options
from holowizard.core.reconstruction.viewer.viewer import Viewer
from holowizard.core.reconstruction.utils import get_filter_kernels
from holowizard.core.reconstruction.logging import *
from holowizard.core.reconstruction.transformation import *
from holowizard.core.models.cone_beam import ConeBeam

if "cuda" in holowizard.core.torch_running_device_name:
    from holowizard.core.models.fresnel_propagator import FresnelPropagator
else:
    from holowizard.core.models.fresnel_propagator_torch import FresnelPropagator


class Context:
    def __init__(
        self,
        measurements: List[Measurement] = None,
        beam_setup: BeamSetup = None,
        options: List[Options] = None,
        data_dimensions: DataDimensions = None,
        viewer: List[Viewer] = None,
        oref_predicted=None,
        nesterov_vt=None,
    ):

        self.torch_device = holowizard.core.torch_running_device

        self.measurements_original = measurements
        self.beam_setup_original = beam_setup
        self.data_dimensions_original = data_dimensions
        self.options = options

        self.current_options = None
        self.se_losses_all = torch.empty(0, device=self.torch_device)

        self.filter_kernel_obj_phase = None
        self.filter_kernel_obj_absorption = None

        self.filter_kernel_vt = None

        self.absorption_min = None
        self.absorption_max = None
        self.phaseshift_min = None
        self.phaseshift_max = None

        self.measurements = None
        self.beam_setup = None
        self.data_dimensions = None
        self.current_options = None

        self.viewer = viewer
        self.oref_predicted = oref_predicted
        self.nesterov_vt = nesterov_vt

        self.current_stage = -1
        self.current_iter_offset = 0

        self.set_stage(0)

    def create_effective_geometry_file(self):
        string_list = []

        string_list.append(
            f"{'z02':<17}{round(self.beam_setup_original.z02,int(np.log10(BeamSetup.unit_z02()[1])))} " + BeamSetup.unit_z02()[0]
        )
        string_list.append(
            f"{'Energy':<17}{round(self.beam_setup_original.energy,int(np.log10(BeamSetup.unit_energy()[1])))} "
            + BeamSetup.unit_energy()[0]
        )
        for i in range(len(self.measurements_original)):
            lam, M, dx_eff, z_eff, fr_eff = ConeBeam.get_effective_geometry(
                self.beam_setup_original, self.measurements_original[i]
            )
            string_list.append("Distance " + str(i) + ":")
            string_list.append(
                f"{'z01':<17}{round(self.measurements_original[i].z01,int(np.log10(Measurement.unit_z01()[1])))} "
                + Measurement.unit_z01()[0]
            )
            string_list.append(f"{'Magnification':<17}{round(M, 3)}")
            string_list.append(
                f"{'Effective dx':<17}{round(dx_eff,int(np.log10(BeamSetup.unit_px_size()[1])))} " + BeamSetup.unit_px_size()[0]
            )
            string_list.append(
                f"{'Effective z12':<17}{round(z_eff,int(np.log10(Measurement.unit_z01()[1])))} " + Measurement.unit_z01()[0]
            )
            string_list.append(f"{'Fresnel Number':<17}{fr_eff}")

        Logger.custom_string_file("geometry", string_list)

    def check_successfull_initialization(self):
        if None in [
            self.measurements_original,
            self.beam_setup_original,
            self.options,
            self.data_dimensions_original,
        ]:
            raise ValueError("Host context not complete.")

    def check_valid_stage_index(self, stage_index):
        if stage_index >= len(self.options):
            raise ValueError(
                "Invalid stage index " + str(stage_index),
                ". Can be at maximum " + str(len(self.options) - 1) + ".",
            )

    def update_current_options(self):
        self.current_options = self.options[self.current_stage]

        self.absorption_min = torch.tensor(
            self.current_options.regularization_object.values_min.imag,
            device=self.torch_device,
            dtype=torch.float,
        )
        self.absorption_max = torch.tensor(
            self.current_options.regularization_object.values_max.imag,
            device=self.torch_device,
            dtype=torch.float,
        )

        self.phaseshift_min = torch.tensor(
            self.current_options.regularization_object.values_min.real,
            device=self.torch_device,
            dtype=torch.float,
        )
        self.phaseshift_max = torch.tensor(
            self.current_options.regularization_object.values_max.real,
            device=self.torch_device,
            dtype=torch.float,
        )

    def update_global_loss_array(self):
        self.se_losses_all = torch.cat(
            (
                self.se_losses_all,
                torch.zeros(
                    self.current_options.regularization_object.iterations,
                    device=self.torch_device,
                    dtype=torch.float,
                ),
            )
        )

    def update_filter_kernels(self):
        (
            self.filter_kernel_obj_phase,
            self.filter_kernel_obj_absorption,
        ) = get_filter_kernels(
            self.current_options.regularization_object.gaussian_filter_fwhm,
            self.data_dimensions.total_size,
            self.torch_device,
        )

        self.filter_kernel_vt, _ = get_filter_kernels(
            self.current_options.nesterov_object.gaussian_filter_fwhm,
            self.data_dimensions.total_size,
            self.torch_device,
        )

    def update_propagator(self):
        self.model = FresnelPropagator(
            [
                ConeBeam.get_fr(self.beam_setup, self.measurements[distance])
                for distance in range(len(self.measurements))
            ],
            self.data_dimensions.total_size,
            self.oref_predicted.device,
        )

    def rescale_arrays(self):
        if self.current_stage > 0:
            current_probe_refractive = deepcopy(self.beam_setup.probe_refractive)
        else:
            current_probe_refractive = deepcopy(
                self.beam_setup_original.probe_refractive
            )

        (
            self.measurements,
            self.beam_setup,
            self.data_dimensions,
        ) = process_padding_options(
            self.measurements_original,
            self.beam_setup_original,
            self.data_dimensions_original,
            self.current_options.padding,
        )

        if self.oref_predicted is None:
            self.oref_predicted = torch.zeros(
                self.data_dimensions.total_size,
                device=self.torch_device,
                dtype=torch.cfloat,
            )

            self.nesterov_vt = torch.zeros(
                self.data_dimensions.total_size,
                device=self.torch_device,
                dtype=torch.cfloat,
            )

            self.beam_setup.probe_refractive = torch.zeros_like(self.oref_predicted)

        else:
            (
                self.oref_predicted,
                self.nesterov_vt,
                self.beam_setup.probe_refractive,
            ) = resize_guess(
                self.oref_predicted,
                self.nesterov_vt,
                self.data_dimensions.total_size,
                current_probe_refractive,
            )

    def set_stage(self, stage_index):
        self.check_successfull_initialization()
        self.check_valid_stage_index(stage_index)

        if self.current_stage == stage_index:
            return

        self.current_stage = stage_index
        logging.comment(
            "Stage " + str(self.current_stage + 1) + "/" + str(len(self.options))
        )

        if self.current_stage == 0:
            self.create_effective_geometry_file()

        self.update_current_options()

        self.update_global_loss_array()

        self.rescale_arrays()

        self.update_filter_kernels()

        self.update_propagator()

        log_preprocessed_params(self.beam_setup, self.data_dimensions)
