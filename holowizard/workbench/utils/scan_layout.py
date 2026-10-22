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
