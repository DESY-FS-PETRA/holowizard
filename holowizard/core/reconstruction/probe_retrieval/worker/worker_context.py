import os
import torch
from typing import List
from enum import Enum

import holowizard.core
from holowizard.core.serialization.params_serializer import ParamsSerializer
from holowizard.core.models.cone_beam import ConeBeam
from holowizard.core.reconstruction.utils import get_filter_kernels

from holowizard.core.models.fresnel_propagator_torch import FresnelPropagator

from holowizard.core.parameters.options import Options
from holowizard.core.parameters.beam_setup import BeamSetup
from holowizard.core.parameters.measurement import Measurement
from holowizard.core.parameters.data_dimensions import DataDimensions


class WorkerContext(object):
    class UpdateMode(Enum):
        OBJECT = 0
        PROBE = 1

    @staticmethod
    def from_dict(args_dict, update_mode: UpdateMode):
        index = int(args_dict["index"])
        working_dir = str(args_dict["working_dir"])
        update_oref = bool(args_dict["update_oref"])

        # JSON loads
        options = Options.from_json(args_dict["options"])

        # Pickle file loads
        measurements = [
            ParamsSerializer.deserialize(
                working_dir + "measurements_" + str(index) + ".pkl"
            )
        ]
        beam_setup = ParamsSerializer.deserialize(working_dir + "beam_setup.pkl")
        oref_predicted = ParamsSerializer.deserialize(
            working_dir + "oref_predicted_" + str(index) + ".pkl"
        )
        nesterov_vt = ParamsSerializer.deserialize(
            working_dir + "nesterov_vt_" + str(index) + ".pkl"
        )
        data_dimensions = ParamsSerializer.deserialize(
            working_dir + "data_dimensions.pkl"
        )

        context = WorkerContext(
            working_dir,
            index,
            update_oref,
            measurements,
            beam_setup,
            options,
            oref_predicted,
            nesterov_vt,
            data_dimensions,
            update_mode,
        )

        return context

    def __init__(
        self,
        working_dir: str,
        index: int,
        update_oref: bool,
        measurements: List[Measurement],
        beam_setup: BeamSetup,
        options: Options,
        oref_predicted: torch.Tensor,
        nesterov_vt: torch.Tensor,
        data_dimensions: DataDimensions,
        update_mode=UpdateMode.OBJECT,
    ):
        self.working_dir = working_dir
        self.index = index
        self.update_oref = update_oref
        self.measurements = measurements
        self.beam_setup = beam_setup
        self.options = options
        self.oref_predicted = oref_predicted
        self.nesterov_vt = nesterov_vt
        self.data_dimensions = data_dimensions
        self.update_mode = update_mode

        if update_mode == WorkerContext.UpdateMode.OBJECT:
            self.regularization_set = options.regularization_object
        else:
            self.regularization_set = options.regularization_probe
            self.regularization_set.iterations = 1

        # Copy to GPU
        self.beam_setup.probe = beam_setup.probe.to(device="cuda:0")
        self.measurements[0].data = measurements[0].data.to(device="cuda:0")
        self.oref_predicted = oref_predicted.to(device="cuda:0")
        self.nesterov_vt = nesterov_vt.to(device="cuda:0")
        self.data_dimensions.window = data_dimensions.window.to(device="cuda:0")

        # Init tensors
        self.se_loss_records = torch.zeros(
            self.regularization_set.iterations,
            device=oref_predicted.device,
            dtype=torch.float,
        )

        self.absorption_min = torch.tensor(
            self.regularization_set.values_min.imag,
            device=oref_predicted.device,
            dtype=torch.float,
        )
        self.absorption_max = torch.tensor(
            self.regularization_set.values_max.imag,
            device=oref_predicted.device,
            dtype=torch.float,
        )
        
        self.phaseshift_min = torch.tensor(
            self.regularization_set.values_min.real,
            device=oref_predicted.device,
            dtype=torch.float,
        )
        self.phaseshift_max = torch.tensor(
            self.regularization_set.values_max.real,
            device=oref_predicted.device,
            dtype=torch.float,
        )

        print("Phaseshift max: ", self.phaseshift_max)

        self.probe_refractive = beam_setup.probe_refractive

        # Init filter kernels
        (
            self.filter_kernel_obj_phase,
            self.filter_kernel_obj_absorption,
        ) = get_filter_kernels(
            self.regularization_set.gaussian_filter_fwhm,
            data_dimensions.total_size,
            holowizard.core.torch_running_device,
        )

        if self.update_mode == WorkerContext.UpdateMode.OBJECT:
            self.filter_kernel_vt, _ = get_filter_kernels(
                options.nesterov_object.gaussian_filter_fwhm,
                data_dimensions.total_size,
                holowizard.core.torch_running_device,
            )
        else:
            pass

        # Init model
        self.model = FresnelPropagator(
            [
                ConeBeam.get_fr(self.beam_setup, self.measurements[distance])
                for distance in range(len(self.measurements))
            ],
            self.data_dimensions.total_size,
            self.oref_predicted.device,
        )

        self.grad_probe = None

    def write_results(self):
        try:
            os.remove(self.working_dir + "se_loss_records_" + str(self.index) + ".pkl")
        except Exception as e:
            pass
        try:
            os.remove(self.working_dir + "grad_probe_" + str(self.index) + ".pkl")
        except Exception as e:
            pass
        if self.update_oref:
            os.remove(self.working_dir + "oref_predicted_" + str(self.index) + ".pkl")
            os.remove(self.working_dir + "nesterov_vt_" + str(self.index) + ".pkl")

        # Update probe and loss
        ParamsSerializer.serialize(
            self.se_loss_records,
            self.working_dir + "se_loss_records_" + str(self.index) + ".pkl",
        )
        ParamsSerializer.serialize(
            self.grad_probe, self.working_dir + "grad_probe_" + str(self.index) + ".pkl"
        )

        # Optional: Update only the probe and not the object
        if self.update_oref:
            ParamsSerializer.serialize(
                self.oref_predicted,
                self.working_dir + "oref_predicted_" + str(self.index) + ".pkl",
            )
            ParamsSerializer.serialize(
                self.nesterov_vt,
                self.working_dir + "nesterov_vt_" + str(self.index) + ".pkl",
            )
