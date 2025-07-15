import numpy as np
import torch
import logging
import traceback
from typing import List

import holowizard.core
from holowizard.core.api.functions.single_projection.reconstruction_flatfieldcorrection import (
    reconstruct,
)

from holowizard.core.parameters.measurement import Measurement
from holowizard.core.utils import fileio
from holowizard.core.parameters.reco_params import RecoParams
from holowizard.core.parameters.flatfield_correction_params import FlatfieldCorrectionParams

from holowizard.core.reconstruction.viewer import Viewer
from holowizard.core.api.viewer import LossViewer


def read_measurements_data(measurements: List[Measurement]):
    for i in range(len(measurements)):
        logging.debug("Read measurements from " + measurements[i].data_path)
        measurement_data = torch.tensor(
            fileio.load_img_data(measurements[i].data_path),
            device=holowizard.core.torch_running_device,
            dtype=torch.float,
        )
        logging.debug("Take square root of measurements")
        measurements[i].data = torch.sqrt(measurement_data)

    return measurements


class Reconstruction:
    def __init__(self, viewer: List[Viewer] = None):
        self.viewer = [LossViewer()]

        if viewer is not None:
            self.viewer = self.viewer + viewer

    def reconstruct_x(
        self, flatfield_correction_params_serialized, reco_params_serialized
    ):
        try:
            logging.info("reconstruct_x called")

            logging.debug("Deserialize data")
            reco_params = RecoParams.from_json(reco_params_serialized)
            flatfield_correction_params = FlatfieldCorrectionParams.from_json(
                flatfield_correction_params_serialized
            )

            logging.info("Do reconstruction")

            logging.params("reco_params", reco_params)

            x_predicted, se_losses_all = reconstruct(
                flatfield_correction_params, reco_params, self.viewer
            )
            logging.debug("Write result to file system")
            result_phaseshift = np.float32(np.real(x_predicted.cpu().numpy()))
            result_absorption = np.float32(np.imag(x_predicted.cpu().numpy()))
            fileio.write_img_data(
                reco_params.output_path.split(".")[0] + "_phaseshift.tiff",
                result_phaseshift,
            )
            fileio.write_img_data(
                reco_params.output_path.split(".")[0] + "_absorption.tiff",
                result_absorption,
            )

            logging.info("reconstruct_x Finished")

        except:
            # printing stack trace
            traceback.print_exc()
            raise RuntimeError("Error in Server")

        return se_losses_all.cpu().numpy()
