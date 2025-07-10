import os
import time

import torch
from typing import List
from dask.distributed import as_completed
from copy import deepcopy

from holowizard.core.serialization.params_serializer import ParamsSerializer
from holowizard.core.dask_utils.dask_controller import DaskController
from holowizard.interface.parameters.dask_options import DaskOptions
from holowizard.core.preprocessing.process_padding_options import (
    process_padding_options,
)
from holowizard.interface.parameters.measurement import Measurement
from holowizard.interface.parameters.beam_setup import BeamSetup
from holowizard.interface.parameters.data_dimensions import DataDimensions
from holowizard.interface.parameters.options import Options
from holowizard.core.reconstruction.viewer.viewer import Viewer
from holowizard.core.reconstruction.utils import get_filter_kernels
from holowizard.core.reconstruction.logging import *
from holowizard.core.reconstruction.transformation import *


class HostContext:
    def __init__(
        self,
        dask_options: DaskOptions,
        dask_controller: DaskController,
        measurements: List[Measurement] = None,
        beam_setup: BeamSetup = None,
        options: List[Options] = None,
        data_dimensions: DataDimensions = None,
        viewer: List[Viewer] = None,
        oref_predicted=None,
        nesterov_vt=None,
    ):

        self.current_iter_offset = None
        self.current_options_index = None
        self.current_options = None
        self.se_losses_all = None

        self.filter_kernel_probe_phase = None
        self.filter_kernel_probe_absorption = None

        self.dask_options = dask_options
        self.dask_controller = dask_controller

        self.measurements_original = measurements
        self.beam_setup_original = beam_setup
        self.data_dimensions_original = data_dimensions
        self.options = options

        self.measurements = None
        self.beam_setup = None
        self.data_dimensions = None
        self.current_options = None

        self.viewer = viewer
        self.oref_predicted = oref_predicted
        self.nesterov_vt = nesterov_vt

        self.torch_device = torch.device("cpu")

        self.current_stage = -1
        self.current_iter_offset = -1
        self.se_losses_all = None

        self.dask_controller.start()
        self.set_stage(0)

    def finish(self):
        self.dask_controller.stop()

    def set_stage(self, stage_index):
        if None in [
            self.dask_options,
            self.dask_controller,
            self.measurements_original,
            self.beam_setup_original,
            self.options,
            self.data_dimensions_original,
        ]:
            raise ValueError("Host context not complete.")
        if stage_index >= len(self.options):
            raise ValueError(
                "Invalid stage index " + str(stage_index),
                ". Can be at maximum " + str(len(self.options) - 1) + ".",
            )

        if self.current_stage == stage_index:
            return

        if stage_index == 0:
            log_params(
                self.measurements_original,
                self.beam_setup_original,
                self.options,
                self.data_dimensions_original,
            )

        self.current_stage = stage_index
        logging.comment(
            "Stage " + str(self.current_stage + 1) + "/" + str(len(self.options))
        )

        self.current_options = self.options[self.current_stage]

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

        if self.current_stage == 0:
            self.oref_predicted = [
                torch.zeros(
                    self.data_dimensions.total_size,
                    device=self.torch_device,
                    dtype=torch.cfloat,
                )
                for i in range(len(self.measurements))
            ]

            self.nesterov_vt = [
                torch.zeros(
                    self.data_dimensions.total_size,
                    device=self.torch_device,
                    dtype=torch.cfloat,
                )
                for i in range(len(self.measurements))
            ]
        else:
            for i in range(len(self.measurements)):
                (
                    self.oref_predicted[i],
                    self.nesterov_vt[i],
                    self.beam_setup.probe_refractive,
                ) = resize_guess(
                    self.oref_predicted[i],
                    self.nesterov_vt[i],
                    self.data_dimensions.total_size,
                    current_probe_refractive,
                )

        (
            self.filter_kernel_probe_phase,
            self.filter_kernel_probe_absorption,
        ) = get_filter_kernels(
            self.current_options.regularization_probe.gaussian_filter_fwhm,
            self.data_dimensions.total_size,
            self.torch_device,
        )

        log_preprocessed_params(self.beam_setup, self.data_dimensions)

        HostContext.write_inputs(
            dask_options=self.dask_options,
            beam_setup=self.beam_setup,
            measurements=self.measurements,
            data_dimensions=self.data_dimensions,
            oref_predicted=self.oref_predicted,
            nesterov_vt=self.nesterov_vt,
        )

    def read_intermediate_results(self, results):

        grad_probe = torch.zeros(
            self.data_dimensions.total_size, device=self.torch_device
        )
        se_losses = []

        for completed in as_completed(results):
            current_result = completed.result()

            index = current_result["index"]

            grad_probe = grad_probe + ParamsSerializer.deserialize(
                self.dask_options.working_dir + "grad_probe_" + str(index) + ".pkl"
            ).to(self.torch_device) / len(results)

            se_losses.append(
                ParamsSerializer.deserialize(
                    self.dask_options.working_dir
                    + "se_loss_records_"
                    + str(index)
                    + ".pkl"
                ).to(self.torch_device)
            )

        return sum(se_losses) / len(results), grad_probe

    def read_final_results(self):
        for index in range(len(self.oref_predicted)):
            self.oref_predicted[index] = ParamsSerializer.deserialize(
                self.dask_options.working_dir + "oref_predicted_" + str(index) + ".pkl"
            )
            self.oref_predicted[index] = self.oref_predicted[index].to(
                self.torch_device
            )
            self.nesterov_vt[index] = ParamsSerializer.deserialize(
                self.dask_options.working_dir + "nesterov_vt_" + str(index) + ".pkl"
            )
            self.nesterov_vt[index] = self.nesterov_vt[index].to(self.torch_device)

    @staticmethod
    def write_inputs(
        dask_options,
        beam_setup=None,
        measurements=None,
        data_dimensions=None,
        oref_predicted=None,
        nesterov_vt=None,
    ):

        if beam_setup is not None:
            try:
                os.remove(dask_options.working_dir + "beam_setup.pkl")
            except Exception as e:
                pass
            ParamsSerializer.serialize(
                beam_setup, dask_options.working_dir + "beam_setup.pkl"
            )

        if data_dimensions is not None:
            try:
                os.remove(dask_options.working_dir + "data_dimensions.pkl")
            except Exception as e:
                pass
            ParamsSerializer.serialize(
                data_dimensions, dask_options.working_dir + "data_dimensions.pkl"
            )

        if measurements is not None:
            for j in range(len(measurements)):
                try:
                    os.remove(
                        dask_options.working_dir + +"measurements_" + str(j) + ".pkl"
                    )
                except Exception as e:
                    pass
                ParamsSerializer.serialize(
                    measurements[j],
                    dask_options.working_dir + "measurements_" + str(j) + ".pkl",
                )

        if oref_predicted is not None:
            for j in range(len(oref_predicted)):
                try:
                    os.remove(
                        dask_options.working_dir + "oref_predicted_" + str(j) + ".pkl"
                    )
                    # pass
                except Exception as e:
                    pass
                ParamsSerializer.serialize(
                    oref_predicted[j],
                    dask_options.working_dir + "oref_predicted_" + str(j) + ".pkl",
                )

        if nesterov_vt is not None:
            for j in range(len(nesterov_vt)):
                try:
                    os.remove(
                        dask_options.working_dir + "nesterov_vt_" + str(j) + ".pkl"
                    )
                except Exception as e:
                    pass
                ParamsSerializer.serialize(
                    nesterov_vt[j],
                    dask_options.working_dir + "nesterov_vt_" + str(j) + ".pkl",
                )

    # time.sleep(15)
