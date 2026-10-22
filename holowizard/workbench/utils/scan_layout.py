"""Layout of a raw scan folder: classic or multi-tomogram (e.g. ptychographic scans).

* classic: ``img_*`` and ``ref_*`` images directly in the scan folder.
* multi-tomogram: one subfolder per tomogram, ``tomo_<index>`` with ``img_*`` images, and one or more reference
  folders ``ref_<index>`` with ``ref_*`` images. Next to them, ``positions.txt`` lists the lateral position of each
  tomogram ("x y" in metres, one line per tomogram, in the order of the tomograms) and ``scan_parameters.txt`` the
  geometry. In the scan log, "image number II" is the index of the tomogram.

Kept free of heavy imports, so that the overview can use it.
"""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SUBSET_PATTERN = re.compile(r"(tomo|ref)_(\d+)")
POSITIONS_FILE = "positions.txt"


def split_subsets(names: Iterable[str]) -> Dict[str, List[str]]:
    """The tomogram and reference subfolders among the subfolder names of a scan, each sorted by index."""
    found: Dict[str, List[Tuple[int, str]]] = {"tomo": [], "ref": []}
    for name in names:
        m = SUBSET_PATTERN.fullmatch(name)
        if m:
            found[m.group(1)].append((int(m.group(2)), name))
    return {kind: [name for _, name in sorted(items)] for kind, items in found.items()}


def is_subset_name(name: str) -> bool:
    return SUBSET_PATTERN.fullmatch(name) is not None


def read_positions(scan_dir: Path) -> Optional[List[Tuple[float, float]]]:
    """Lateral positions (x, y in metres) of the tomograms from positions.txt, or None."""
    path = Path(scan_dir) / POSITIONS_FILE
    try:
        lines = path.read_text().split("\n")
    except OSError:
        return None
    positions = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                positions.append((float(parts[0]), float(parts[1])))
            except ValueError:
                return None
    return positions or None


SCAN_PARAMETERS_FILE = "scan_parameters.txt"
TOMOGRAM_MARKER = "tomogram.json"  # in the run folder of a phase retrieval of one tomogram


def subset_index(name: str) -> Optional[int]:
    m = SUBSET_PATTERN.fullmatch(name or "")
    return int(m.group(2)) if m else None


def reference_for(tomogram: str, references: List[str]) -> Optional[str]:
    """Reference folder of a tomogram: the last one recorded before it (largest index <= the tomogram's), else
    the first one."""
    if not references:
        return None
    index = subset_index(tomogram) or 0
    before = [r for r in references if (subset_index(r) or 0) <= index]
    return before[-1] if before else references[0]


def read_scan_parameters(scan_dir: Path) -> Dict[str, str]:
    """``key: value`` lines of scan_parameters.txt (empty if missing)."""
    try:
        lines = (Path(scan_dir) / SCAN_PARAMETERS_FILE).read_text().splitlines()
    except OSError:
        return {}
    return dict(line.split(": ", 1) for line in lines if ": " in line)


HC_KEV_M = 1.239841984e-9  # h*c in keV*m: energy (keV) = HC_KEV_M / wavelength (m)


def read_energy(scan_dir: Path) -> Optional[Dict]:
    """X-ray energy of a scan from the wavelength in scan_parameters.txt (``wavelength_m``, written by the
    ptychography scan scripts), or None. Classic holography scans do not record the energy in the raw folder."""
    try:
        wavelength = float(read_scan_parameters(scan_dir)["wavelength_m"])
    except (KeyError, ValueError):
        return None
    if not wavelength > 0:
        return None
    return {"energy_kev": round(HC_KEV_M / wavelength, 3), "wavelength_m": wavelength, "source": SCAN_PARAMETERS_FILE}


# --- Run folders of single tomograms ---
def write_tomogram_marker(run_folder: Path, info: Dict) -> None:
    import json

    run_folder = Path(run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    (run_folder / TOMOGRAM_MARKER).write_text(json.dumps(info, indent=2))


def read_tomogram_marker(run_folder: Path) -> Optional[Dict]:
    """``{"scan", "tomogram", "index", "reference", "series"}`` of a run of one tomogram, or None (classic run)."""
    import json

    try:
        data = json.loads((Path(run_folder) / TOMOGRAM_MARKER).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("tomogram") else None


# --- Shift of the object between the tomograms ---
# The tomograms of a scan are recorded at different lateral sample positions: the object (and its rotation axis) is
# shifted in the images. Shifts are given as the displacement of the object in the raw detector images relative to the
# first tomogram, (rows, columns) in raw pixels. The rotation axis lies along the raw image columns, so the rotation
# center (a column of the phase images, which are the raw images rotated by 90 degrees: phase column = raw row) moves
# with the row displacement. Measured on nano16462 (beamtime 11024767): the y position moves the object along the raw
# rows, x along the raw columns, with displacement = (-dy, +dx) / (detector pixel / magnification), within ~1 px.


def shifts_from_positions(scan_dir: Path, tomograms: List[str], detector_pixel_m: float) -> Optional[Dict]:
    """Object displacement per tomogram from positions.txt and the magnification in scan_parameters.txt, or None."""
    positions = read_positions(scan_dir)
    try:
        magnification = float(read_scan_parameters(scan_dir)["magnification"])
    except (KeyError, ValueError):
        return None
    if not positions or magnification <= 0 or len(positions) < len(tomograms):
        return None
    pixel = detector_pixel_m / magnification
    x0, y0 = positions[0]
    return {name: (-(positions[i][1] - y0) / pixel, (positions[i][0] - x0) / pixel) for i, name in enumerate(tomograms)}


def shifts_from_registration(scan_dir: Path, tomograms: List[str]) -> Dict:
    """Object displacement per tomogram by registering the same projection (same angle, the middle one) of each
    tomogram to the first one. The structured illumination, which stays fixed on the detector, is removed first by
    dividing by the mean over many angles of the same tomogram. Reads about 30 images per tomogram."""
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import tifffile
    from scipy.ndimage import gaussian_filter
    from skimage.registration import phase_cross_correlation

    scan_dir = Path(scan_dir)
    bin_ = 2

    def images(name):
        return sorted(
            p for p in (scan_dir / name).iterdir() if p.name.startswith("img") and p.suffix.lower() in (".tif", ".tiff")
        )

    def read(path):
        return tifffile.imread(str(path)).astype(np.float32)[::bin_, ::bin_]

    def object_part(name, index):
        files = images(name)
        with ThreadPoolExecutor(16) as ex:
            stationary = np.mean(list(ex.map(read, files[:: max(1, len(files) // 30)])), axis=0)
        a = read(files[min(index, len(files) - 1)]) / np.maximum(stationary, 1e-6)
        return a - gaussian_filter(a, 20)

    middle = len(images(tomograms[0])) // 2
    first = object_part(tomograms[0], middle)
    shifts = {tomograms[0]: (0.0, 0.0)}
    for name in tomograms[1:]:
        s, _, _ = phase_cross_correlation(first, object_part(name, middle), upsample_factor=4, normalization=None)
        shifts[name] = (-float(s[0]) * bin_, -float(s[1]) * bin_)
    return shifts
