import logging
from typing import List
import traceback

from holowizard.core.api.functions.find_focus.find_focus_flatfieldcorrection import (
    find_focus,
)
from holowizard.core.api.viewer.loss_viewer import LossViewer
from holowizard.core.reconstruction.viewer import Viewer


class FindFocus:
    def __init__(self, viewer: List[Viewer] = None):
        self.viewer = [LossViewer()]

        if viewer is not None:
            self.viewer = self.viewer + viewer

    def find_focus(
        self, flatfield_correction_params_serialized, reco_params_serialized
    ):
        try:
            logging.info("reconstruct_z01 called")

            logging.debug("Deserialize data")
            reco_params = RecoParams.from_json(reco_params_serialized)
            flatfield_correction_params = FlatfieldCorrectionParams.from_json(
                flatfield_correction_params_serialized
            )

            logging.debug("Make focus step")
            z01_found, z01_history, loss_values_history = find_focus(
                flatfield_correction_params, reco_params, self.viewer
            )

            logging.info("Finished after " + str(len(loss_values_history)) + " steps")
            logging.info("Found z01 " + str(z01_found))
            logging.info("reconstruct_z01 Finished")

        except:
            # printing stack trace
            traceback.print_exc()
            raise RuntimeError("Error in Server")

        return z01_found
