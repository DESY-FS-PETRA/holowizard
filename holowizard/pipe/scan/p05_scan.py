import os
from pathlib import Path
import skimage.io as skio
import numpy as np
from holowizard.pipe.scan.scan import Scan
from holowizard.core.api.parameters.measurement import Measurement
from holowizard.core.api.parameters.beam_setup import BeamSetup
import pandas as pd


class P05Scan(Scan):
    """
    P05Scan class for handling P05 beamline scans.
    Inherits from the base Scan class.
    """

    def __init__(self, name, energy, holder, path_raw, path_processed, log_path, cfg, z01_new, a0):
        """
        Initialize the P05Scan object.

        Args:
            name (str): Name of the scan.
            energy (float): Energy used in the scan.
            holder (float): Holder length in mm.
        """
        self.path_raw = Path(path_raw) / Path(name)
        self.path_processed = Path(path_processed) / Path(name)
        self.path_log = Path(log_path) / Path(name)
        p05geo = P05Geometry(scan_path=self.path_raw, energy=energy, holder=holder, qp=True)

        z01, z02 = p05geo.compute_z_params()
        if z01_new:
            z01 = z01_new

        super().__init__(
            name,
            energy,
            self.path_raw,
            self.path_raw,
            self.path_processed,
            self.path_raw,
            self.path_log,
            "img",
            "ref",
            z01,
            z02,
            cfg=cfg,
            a0=a0,
            rotation_angles=p05geo.rotation_angle,
        )

    def _load_metadata(self, path):
        """
        Retrieve metadata from the scan's parameter file.

        Returns:
            dict: Metadata dictionary with scan parameters.
        """
        logfile = os.path.join(self.path_raw, f"{self.name}__ScanParam.txt")
        metadata = {}

        try:
            with open(logfile) as file:
                for line in file:
                    if ": " in line:
                        key, value = line.strip().split(": ", 1)
                        metadata[key] = value
        except FileNotFoundError:
            metadata[self.name] = "Metadata file not found."
        except Exception as e:
            metadata[self.name] = f"Metadata read error: {e}"

        return metadata

    def length(self, key):
        """
        Get the number of images for a given key.

        Args:
            key (str): 'hologram' or 'reference'.

        Returns:
            int: Number of images for the specified key.
        """
        if key == "hologram":
            return len(self.hologram_path)
        elif key == "reference":
            return len(self.reference_path)
        else:
            raise KeyError(f"Invalid key: {key}. Use 'hologram' or 'reference'.")

    def __getitem__(self, key):
        """
        Get either hologram or reference image by key and item.

        Args:
            key = (str, int): 'hologram' or 'reference' and index of the image to retrieve.

        Returns:
            np.ndarray: The image at the specified index.
        """
        key, item = key
        if key == "hologram":
            path = self.hologram_path[item]
        elif key == "reference":
            path = self.reference_path[item]
        else:
            raise KeyError(f"Invalid key: {key}. Use 'hologram' or 'reference'.")
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")
        return skio.imread(path)


def load_motor_log(scan_path):
    """
    Load motor positions from the scan's log file.

    Args:
        scan_path (str or Path): Path to the scan directory.

    Returns:
        dict: Mapping of motor names to float values.
    """
    scanname = Path(scan_path).name
    filename = Path(scan_path) / f"{scanname}__LogMotors.log"

    if not filename.exists():
        print(f"Motor log file not found: {filename}")
        return

    lines = filename.read_text().splitlines()

    # Probe format by looking at the first parameter block
    probe_block = lines[:60]
    first_param_block = [x for x in probe_block if ": " in x]

    # Detect Granite* variant
    granite_keys = [x.split(": ")[1] for x in first_param_block]
    if "GraniteSlider_1" in granite_keys:
        param_end = 50
        values_line = 55
        name_map = {
            "GraniteSlider_1": "GraniteSlab_1",
            "GraniteSlider_2": "GraniteSlab_2",
        }
    elif "GraniteSlab_1" in granite_keys:
        param_end = 53
        values_line = 58
        name_map = {}  # no renaming needed
    else:
        raise ValueError(f"Unknown motor name pattern in: {filename}\nPlease provide z01 and z02 manually.")

    try:
        param_names = lines[:param_end]
        motor_values = lines[values_line].split("\t")
        motor_pos = {}

        for i, line in enumerate(param_names):
            key = line.split(": ")[1]
            key = name_map.get(key, key)
            motor_pos[key] = motor_values[i]

    except Exception as e:
        raise ValueError(f"Failed to parse motor log from {filename}.\n{e}")

    filename = Path(scan_path) / f"{scanname}__LogScan.log"
    if not filename.exists():
        raise FileNotFoundError(f"Scan log file not found: {filename}")
    columns = [
        "Image Identifier",
        "InfoStr",
        "Image Number I",
        "Image Number II",
        "Current Number",
        "Timestamp",
        "PETRA Beam Current",
        "Rotation Stage Position",
    ]
    df = pd.read_csv(filename, comment="#", sep="\s+", names=columns)
    df = df[df["Image Identifier"] == "img"]
    df["Rotation Stage Position"] = df["Rotation Stage Position"].astype(float)
    return motor_pos, df["Rotation Stage Position"].to_numpy() * np.pi / 180


## TODO @marinhoa
class P05Geometry:
    """
    Handles P05-specific geometry calculation from scan motor logs.
    """

    def __init__(self, scan_path, energy, holder, qp=True):
        self.scan_path = scan_path
        self.energy = energy
        self.holder = holder
        self.qp = qp

        self.fzp_dr = 50  # nm
        self.fzp_d = 300  # um
        self.wl = 1.2398 / self.energy  # wavelength in nm

        self.motor_pos, self.rotation_angle = load_motor_log(self.scan_path)

    @property
    def fzp_f(self):
        """
        Focal length of the zone plate (mm).
        """
        return self.fzp_d * 1e-3 * self.fzp_dr * 1e-6 / (self.wl * 1e-6)  # in mm

    def compute_z_params(self):
        """
        Compute z01 and z02 based on motor positions and optics setup.

        Returns:
            tuple: (z01, z02) in nanometer.
        """
        try:
            o_stage_y = float(self.motor_pos["OpticsStage1_y"])
            slider1 = float(self.motor_pos["GraniteSlab_1"])
            slider2 = float(self.motor_pos["GraniteSlab_2"])
            sf1_y = float(self.motor_pos["OpticsSF1_y"])
        except KeyError as e:
            raise ValueError(f"Missing expected motor key: {e}. Cannot compute geometry.")

        detDistInit = 20413 if self.qp else 20264
        posInit = -80.77 if self.qp else -66

        offset = slider1 + o_stage_y + sf1_y + self.holder + self.fzp_f
        z02 = (detDistInit - offset) * 1e6
        z01 = (posInit + slider2 - offset) * 1e6

        return z01 / Measurement.unit_z01()[1], z02 / BeamSetup.unit_z02()[1]
