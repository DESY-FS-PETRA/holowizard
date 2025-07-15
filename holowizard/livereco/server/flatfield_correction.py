import json
import pickle
import logging
import torch
from typing import List

import holowizard.core
from holowizard.core.preprocessing.calculate_flatfield_components import (
    calculate_flatfield_components,
)
from holowizard.core.preprocessing.correct_flatfield import correct_flatfield
from holowizard.core.parameters.flatfield_components_params import (
    FlatfieldComponentsParams,
)
from holowizard.core.parameters.flatfield_correction_params import (
    FlatfieldCorrectionParams,
)
from holowizard.core.reconstruction.viewer import Viewer

from holowizard.core.parameters.type_conversion.json_writable import JsonWritable


def log_files(files):
    for file in files:
        logging.info("Flatfield File: " + file)


def log_empties(empties):
    i = 0
    for empty in empties:
        logging.image_info("calculate_flatfield_components_input_" + str(i), empty)
        i = i + 1


class FlatfieldCorrection:
    def __init__(self, viewer: List[Viewer] = None):
        self.viewer = viewer
        self.flatfield_list = None

    def add_flatfield(self, measurement_serialized):
        if measurement_serialized is None:
            return

        numpy_array = JsonWritable.get_numpy_from_array(
            json.loads(measurement_serialized)
        )

        if self.flatfield_list is None:
            self.flatfield_list = [numpy_array]

        else:
            self.flatfield_list.append(numpy_array)

    def reset_flatfield_list(self):
        self.flatfield_list = None

    def calc_flatfield_components(self, flatfield_components_params_serialized):
        logging.info("calc_flatfield_components called")

        logging.debug("Deserialize")
        flatfield_components_params = FlatfieldComponentsParams.from_json(
            flatfield_components_params_serialized
        )

        flatfield_components_params.measurements = self.flatfield_list

        log_empties(flatfield_components_params.measurements)

        logging.debug("Calculate components")
        logging.info(
            "Using " + str(flatfield_components_params.num_components) + " components"
        )
        components_model = calculate_flatfield_components(
            flatfield_components_params.measurements,
            flatfield_components_params.num_components,
        )

        logging.debug(
            "Write components to file system: " + flatfield_components_params.save_path
        )

        with open(flatfield_components_params.save_path, "wb") as file:
            pickle.dump(components_model, file, pickle.HIGHEST_PROTOCOL)
            file.close()

        logging.info("calc_flatfield_components Finished")

        return True

    def correct_flatfield(self, flatfield_correction_params_serialized):
        logging.info("correct_flatfield called")

        logging.debug("Deserialize")
        flatfield_correction_params = FlatfieldCorrectionParams.from_json(
            flatfield_correction_params_serialized
        )

        logging.debug(
            "Load components from " + flatfield_correction_params.components_path
        )
        with open(flatfield_correction_params.components_path, "rb") as file:
            components_model = pickle.load(file)

        logging.debug("Load image from " + flatfield_correction_params.image)
        image_to_correct = torch.tensor(
            flatfield_correction_params.image,
            device=holowizard.core.torch_running_device,
        )
        logging.image_info("raw", image_to_correct.cpu().numpy())

        logging.debug("Correct flatfield")
        corrected_img_data = correct_flatfield(image_to_correct, components_model)
        logging.image_info("flatfield_corrected", corrected_img_data.cpu().numpy())

        logging.info("correct_flatfield finished")
        return corrected_img_data
