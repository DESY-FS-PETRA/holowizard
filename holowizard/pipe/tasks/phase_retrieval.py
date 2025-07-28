import os
import time
import datetime
from pathlib import Path
import numpy as np
import torch
import gc
from holowizard.core.api.functions.single_projection.reconstruction_flatfieldcorrection import (
    reconstruct,
)
from holowizard.core.logging.logger import Logger
from holowizard.core.utils import fileio
from holowizard.core.utils.transform import crop_center
from holowizard.core.api.parameters import FlatfieldCorrectionParams
from holowizard.pipe.scan import Scan
from holowizard.pipe.utils.reco_params import build_reco_params


# matplotlib.use("Qt5Agg")
class PhaseRetrievalTask:
    """
    Handles phase retrieval using the JSON config and builder pipeline.
    """

    def __init__(self, scan: Scan, flatfield_params_save_path, viewer=[], save_scratch=False):
        """
        Initialize the PhaseRetrieval runner.

        Args:
            config_path (str): Path to the JSON configuration file.
        """
        self.enabled = "reconstruction" in scan.config.scan.tasks
        if not self.enabled:
            # Logger.info(
            #    "Phase retrieval disabled",
            # )
            return
        self.viewer = viewer
        self.paths = scan.config["paths"]
        self.reco_params = build_reco_params(scan, scan.reconstruction)
        if save_scratch:
            self.phase_dir = os.path.join(scan.path_log, self.paths.base_dir, self.paths.phase_dir)
            self.absorption_dir = os.path.join(scan.path_log, self.paths.base_dir, self.paths.absorption_dir)

        else:
            self.phase_dir = os.path.join(scan.path_processed, self.paths.base_dir, self.paths.phase_dir)
            self.absorption_dir = os.path.join(scan.path_processed, self.paths.base_dir, self.paths.absorption_dir)
        os.makedirs(self.phase_dir, exist_ok=True)
        os.makedirs(self.absorption_dir, exist_ok=True)
        self.flatfield_params = FlatfieldCorrectionParams(
            image=scan.hologram_path[0],  # Assuming the first hologram is used for focus finding
            components_path=flatfield_params_save_path,
        )

    def __call__(self, scan, img_index=0):
        """
        Perform a single phase retrieval reconstruction.

        Args:
            img_index (int): Index of the image in the file list to reconstruct.
        """
        if not self.enabled:
            return
        self.reco_params.measurements[0].data = scan["hologram", img_index]

        filename = f"hologram_{img_index:04d}.tiff"

        self.reco_params.output_path = os.path.join(self.phase_dir, filename)
        absorption_output_path = os.path.join(self.absorption_dir, filename)

        Logger.current_log_level = Logger.level_num_loss

        timestamp = time.time()
        log_time = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")

        Logger.configure(
            session_name=f"{log_time}_reco_{filename}",
            working_dir=str(Path(scan.path_log) / Path(self.paths.base_dir) / Path("log")),
        )
        self.flatfield_params.image = scan.hologram_path[img_index]

        result, se_losses = reconstruct(
            flatfield_correction_params=self.flatfield_params,
            reco_params=self.reco_params,
            viewer=self.viewer,
        )

        self.result = result
        self.se_losses = se_losses
        self.result_phaseshift = np.float32(np.real(result.cpu().numpy()))
        self.result_absorption = np.float32(np.imag(result.cpu().numpy()))

        fileio.write_img_data(
            self.reco_params.output_path,
            crop_center(self.result_phaseshift, self.reco_params.data_dimensions.fov_size),
        )
        fileio.write_img_data(
            absorption_output_path,
            crop_center(self.result_absorption, self.reco_params.data_dimensions.fov_size),
        )
        gc.collect()
        torch.cuda.empty_cache()
        return 0
