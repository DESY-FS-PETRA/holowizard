from abc import ABC, abstractmethod
import datetime
import os
import uuid
import numpy as np
from skimage.io import imread
import yaml
from omegaconf import OmegaConf
from holowizard.pipe.utils.clean_yaml import remove_keys, to_clean_yaml
from holowizard.core.models.cone_beam import ConeBeam
from holowizard.core.api.parameters.beam_setup import BeamSetup
from holowizard.core.api.parameters.measurement import Measurement
from pathlib import Path
from fastapi.templating import Jinja2Templates

citations_path = Path(__file__).parent.parent / "citations.yaml"
with open(citations_path, "r") as f:
    citations = yaml.load(f, Loader=yaml.FullLoader)
    citations = citations["citations"]


def calculate_image_statistics(image_path: str) -> dict:
    """
    Calculate basic statistics from an image file.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: Dictionary with mean, std, min, max.
    """
    image = imread(image_path)
    return {
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
    }


class Scan(ABC):
    def __init__(
        self,
        name,
        energy,
        path_to_holograms,
        path_to_refs,
        path_processed,
        path_to_metadata,
        path_log,
        holo_name,
        ref_name,
        z01,
        z02,
        cfg,
        a0,
        rotation_angles,
    ):
        """
        Initialize the Scan object.

        Args:
            name (str): Name of the scan.
            year (int): Year of the scan.
            energy (float): Energy used in the scan.
            path_raw (str): Path to raw data.
            path_processed (str): Path to processed data.
        """
        self.name = name
        self.energy = energy
        self.path_processed = path_processed
        self.path_log = path_log
        self.hologram_key = holo_name
        self.reference_key = ref_name
        self.hologram_path = sorted(
            [os.path.join(path_to_holograms, f) for f in os.listdir(path_to_holograms) if holo_name in f]
        )
        self.reference_path = sorted([os.path.join(path_to_refs, f) for f in os.listdir(path_to_refs) if ref_name in f])
        stat = Path(self.hologram_path[0]).stat()
        dt = datetime.datetime.fromtimestamp(stat.st_ctime)
        self.date = dt.strftime("%Y-%m-%d %H:%M:%S")
        self.z01 = z01
        self.z02 = z02
        self.config = cfg
        self.reconstruction = cfg.reconstruction
        self.find_focus = cfg.find_focus
        self.meta_dict = self._load_metadata(path_to_metadata)
        self.a0 = a0
        self.rotation_angles = rotation_angles
        self.done = []
        beam_setup = BeamSetup(
            energy=self.energy,
            z02=self.z02,
            px_size=cfg.scan.px_size * cfg.reconstruction.stages[-1].padding.down_sampling_factor,
        )
        measurement = Measurement(self.z01)

        _, self.M, self.dx_eff, self.z_eff, self.fr_eff = ConeBeam.get_effective_geometry(beam_setup, measurement)
        self.binning = cfg.reconstruction.stages[-1].padding.down_sampling_factor
        self.key = str(uuid.uuid4())  # Unique identifier for the scan
        self.cancelled = False

    def __str__(self):
        """
        String representation of the Scan object.

        Returns:
            str: Formatted string with scan details.
        """
        """
        YAML representation of the Scan object.
        """
        data = {
            "name": self.name,
            **remove_keys(OmegaConf.to_container(self.config.beamtime, resolve=True)),
            "energy": self.energy,
            "a0": self.a0,
            "z01": float(self.z01),
            "z02": float(self.z02),
            **self.meta_dict,
            "tasks": remove_keys(OmegaConf.to_container(self.config.scan.tasks, resolve=True)),
            "path_processed": str(self.path_processed),
            "path_holograms": str(os.path.dirname(self.hologram_path[0])),
            "path_references": str(os.path.dirname(self.reference_path[0])),
            "hologram_key": str(self.hologram_key),
            "reference_key": str(self.reference_key),
            "further_paths": dict(self.config.paths),
            "flatfield": OmegaConf.to_container(self.config.flatfield),
            "config": {
                "reconstruction": remove_keys(OmegaConf.to_container(self.reconstruction, resolve=True)),
                "find_focus": remove_keys(OmegaConf.to_container(self.find_focus, resolve=True)),
            },
        }
        return yaml.dump(data, sort_keys=False)

    @abstractmethod
    def _load_metadata(self, path) -> dict:
        """
        Load and return metadata dictionary from the specified path.

        Args:
            path (str): Path to the metadata file.

        Returns:
            dict: Metadata dictionary.
        """
        pass

    def get_a0(self):
        """
        Compute the a0 parameter using the ratio of mean values from image and reference files.

        Args:
            data_path (str, optional): The path where image and reference files are stored.

        Returns:
            float: The calculated a0 parameter, or None if no valid images are found.
        """

        img_mean_list = [imread(image_path) for image_path in self.hologram_path]
        ref_mean_list = [imread(image_path) for image_path in self.reference_path]
        img_mean = np.mean(img_mean_list)
        ref_mean = np.mean(ref_mean_list)

        if ref_mean == 0:
            raise ValueError("Reference image mean is zero, cannot compute a0.")

        return np.sqrt(img_mean / ref_mean)

    def length(self, key: str) -> int:
        """
        Get the number of images for a given key.

        Args:
            key (str): 'hologram' or 'reference'.

        Returns:
            int: Number of images for the specified key.
        """
        pass  # This method should be implemented in subclasses to return the length of the specified key's images.

    def __getitem__(self, key):
        """
        Get either hologram or reference image by key and item.
        Args:
            key (str): 'hologram' or 'reference'.
            item (int): Index of the image to retrieve.
        Returns:
            np.ndarray: The image at the specified index.
        """
        pass  # This method should be implemented in subclasses to retrieve the image based on the key and item index.

    def get_number_of_holograms(self) -> int:
        """
        Get the number of images in the scan.

        Returns:
            int: Number of images.
        """
        if not hasattr(self, "image_files"):
            raise AttributeError("image_files attribute is not set.")
        return len(self.holograms_paths)

    def get_number_of_references(self) -> int:
        """
        Get the number of reference images in the scan.

        Returns:
            int: Number of reference images.
        """
        if not hasattr(self, "reference_files"):
            raise AttributeError("reference_files attribute is not set.")
        return len(self.reference_paths)

    def write_html(self):
        """
        Write the scan details to an HTML file using the provided template.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = base_dir + "/../templates"
        templates = Jinja2Templates(directory=template_dir)
        templates.env.filters["to_yaml"] = lambda x: to_clean_yaml(x)
        scan_html_template = templates.get_template("scan.html")

        html_str = scan_html_template.render(scan=self, scan_string=str(self), citations=citations)
        output_path = f"{self.path_processed}/{self.config.paths.base_dir}/README.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_str)

    def cancel(self):
        """
        Cancel the scan by clearing the hologram and reference paths.
        """
        self.cancelled = True
        print(f"Scan {self.name} has been cancelled.")
