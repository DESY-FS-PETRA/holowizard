import astra
import os
import numpy as np
from pathlib import Path
from holowizard.pipe.scan import Scan
from holowizard.core.logging.logger import Logger
from skimage.io import imread
from imageio import imwrite


## TODO Just a temporary fix for tigre
class TomographyTask:
    """
    Handles phase retrieval using the JSON config and builder pipeline.
    """

    def __init__(self, scan: Scan):
        """
        Initialize the PhaseRetrieval runner.

        Args:
            config_path (str): Path to the JSON configuration file.
        """
        self.enabled = "tomography" in scan.config.scan.tasks
        if not self.enabled:
            # Logger.info(
            #    "Tomography reconstruction disabled",
            # )
            return
        self.input_path = (
            Path(scan.path_processed) / Path(scan.config.paths.base_dir) / Path(scan.config.paths.phase_dir)
        )
        self.output_paths = (
            Path(scan.path_processed) / Path(scan.config.paths.base_dir) / Path(scan.config.paths.tomo_dir)
        )
        self.angles = scan.rotation_angles
        self.cfg = scan.config.tomography

    def __call__(self, scan):
        """
        Perform a single phase retrieval reconstruction.

        Args:
            img_index (int): Index of the image in the file list to reconstruct.
        """

        if not self.enabled:
            return
        try:
            ## load the input images
            input_images = [
                imread(os.path.join(self.input_path, f))
                for f in sorted(os.listdir(self.input_path))
                if f.endswith(".tiff")
            ]
            if not input_images:
                raise ValueError(f"No images found in {self.input_path}")
            volume = np.stack(input_images[1:], axis=0).astype(np.float32)
            os.makedirs(self.output_paths, exist_ok=True)
            for i in range(volume.shape[1]):
                vol_geom = astra.create_vol_geom(input_images[1].shape[1], input_images[1].shape[1])
                proj_geom = astra.create_proj_geom("parallel", 1.0, input_images[1].shape[0], self.angles)
                rec_id = astra.data2d.create("-vol", vol_geom)
                sinogram_id = astra.data2d.create(
                    "-sino", proj_geom, volume[:, i]
                )  # Move the first axis to the second position
                cfg = astra.astra_dict("FBP_CUDA")
                cfg["ReconstructionDataId"] = rec_id
                cfg["ProjectionDataId"] = sinogram_id
                alg_id = astra.algorithm.create(cfg)
                # Run 150 iterations of the algorithm
                astra.algorithm.run(alg_id)
                # Get the result
                reconstruction = astra.data2d.get(rec_id)
                astra.algorithm.delete(alg_id)
                astra.data2d.delete(rec_id)
                astra.data2d.delete(sinogram_id)
                output_path = self.output_paths / f"reconstruction_{i:04d}.tiff"
                imwrite(output_path, reconstruction)
        except Exception as e:
            print(f"Error during tomography reconstruction: {e}")
            pass
