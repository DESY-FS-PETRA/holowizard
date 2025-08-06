import os
import logging
from pathlib import Path
from holowizard.core.api.functions.flatfield_correction.calc_flatfield_components_o import (
    calculate_flatfield_components,
)
from holowizard.core.api.parameters import (
    FlatfieldComponentsParams,
)
from holowizard.pipe.scan import Scan
from holowizard.core.logging.logger import Logger


class FlatFieldTask:
    """
    This class handles the calculation of flatfield components using a JSON-based configuration.

    It checks if the PCA file exists, and if not, it calculates the flatfield components.

    Typical usage:
        flatfield_task = FlatFieldTask(config_path="/path/to/holopipe_config.json")
        flatfield_task.run()
    """

    def __init__(self, scan: Scan):
        """
        Initialize the FlatFieldTask with a given configuration.

        Args:
            config (dict): Configuration dictionary containing paths and parameters.
        """

        self.components_num = scan.config.flatfield.components_num
        self.save_path = scan.path_processed / Path(scan.config.paths.base_dir) / Path(scan.config.flatfield.save_path)
        os.makedirs(self.save_path.parent, exist_ok=True)
        self.save_path = str(self.save_path)
        self.flatfield = None

    def __call__(self, scan: Scan):
        Logger.current_log_level = Logger.level_num_loss

        Logger.configure(
            session_name="flatfield",
            working_dir=str(Path(scan.path_log) / Path(scan.config.paths.base_dir) / Path("log")),
        )
        # Check if the PCA file exists
        self.flatfield = FlatfieldComponentsParams(
            measurements=[scan["reference", i] for i in range(scan.length("reference"))],
            num_components=self.components_num,
            save_path=str(self.save_path),
        )
        try:
            if not os.path.isfile(self.flatfield.save_path):
                print(f"Calculating flatfield components and saving to {self.flatfield.save_path}")
                calculate_flatfield_components(self.flatfield)
            else:
                logging.info(f"Flatfield components already exist: {self.flatfield.save_path}")
        except Exception as e:
            print(f"Error calculating flatfield components: {e}")
            pass
