import os
from holowizard.core.api.functions.find_focus.find_focus_flatfieldcorrection import find_focus 
from holowizard.core.logging.logger import Logger
from holowizard.core.utils import fileio
from holowizard.core.api.parameters import FlatfieldCorrectionParams, FlatfieldComponentsParams
from holowizard.pipe.scan import Scan
from holowizard.pipe.utils.reco_params import build_reco_params
import time
import datetime
from pathlib import Path

class FindFocusTask:
    """
    It ensures flatfield components are available, loads required runtime parameters, runs the optimization,
    updates the configuration, and optionally visualizes the result through plots.

    Typical usage:
        focus = FindFocus(config_path="/path/to/holopipe_config.json")
        focus.run_focus(img_index=0)
        focus.plot_results()
    """
    def __init__(self, scan: Scan, flatfield_params_save_path, viewer=[]):
        """
        Initialize the focus finder with a given holopipe configuration.

        Args:
            config_path (str): Path to the JSON configuration file.
        """
        self.scan = scan

        self.enabled = "find_focus" in scan.config.scan.tasks
        if not self.enabled:
          #  Logger.info(
          #      f"Find focus disabled; using default focus {scan.z01}",
   
   #         )
            return  # nothing else to set up
        self.viewer = viewer
        if "find_focus" not in scan.config.scan.tasks:
            Logger.log(f"Find focus is disabled in the configuration. Skipping find focus task. Using default focus {scan.z01}", level=Logger.level_num_info)
            return {"z01": scan.z01}

        self.reco_params_find_focus = build_reco_params(scan, scan.find_focus)
        self.flatfield_params = FlatfieldCorrectionParams(
                                                                image=scan["hologram", 0],  # Assuming the first hologram is used for focus finding
                                                                components_path=flatfield_params_save_path
                                                            )

    def __call__(self, scan: Scan, img_index=0):
        """
        Perform focus optimization by varying z01 and selecting the value that minimizes the loss.

        Args:
            img_index (int): Index of the image to be used from the config file list.
            is_batch (bool): Whether the function is being run in batch mode (disables plotting).
        """
        if not self.enabled:
            return {"z01": self.scan.z01}
        measurement = self.reco_params_find_focus.measurements[0]
        measurement.data = scan["hologram", img_index]


        Logger.current_log_level = Logger.level_num_image_info

        timestamp = time.time()
        log_time = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')

        self.log_subdir_name = f"{log_time}_find_focus_{img_index}"

        Logger.configure(
            session_name=self.log_subdir_name,
            working_dir=str(scan.path_log / Path(scan.config.paths.base_dir) / Path("log"))
        )

        z01_guess, z01_values_history, loss_values_history = find_focus(
            flatfield_correction_params=self.flatfield_params,
            reco_params=self.reco_params_find_focus,
            viewer=self.viewer
        )
        return {"z01": z01_guess, "z01_values_history": z01_values_history, "loss_values_history": loss_values_history}