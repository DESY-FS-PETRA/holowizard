"""Overview pages (overview.html) of the beamtime's ``nano*`` data sets.

Two overviews share the tile grid and the image browser:

* ``raw``: the scan directories in ``path_raw``; each tile shows the first ref_ image and the first,
  middle and last img_ image.
* ``phase``: the phase retrieval results in ``path_processed/<scan>/<save prefix>/phase_retrieval``,
  one tile per phase retrieval run; each tile shows the first, middle and last image.
* ``tomo``: the tomographic reconstructions in ``path_processed/<scan>/recon_dir/<version>``, one tile
  per version that contains TIFF slices; each tile shows the first, middle and last slice.

A summary page combines all three per scan (raw data, phase retrieval runs, tomogram versions).

Directory listings are cached per directory mtime, so polling stays cheap while new data sets and
images are written.
"""

import io
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tifffile
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse
from PIL import Image

from holowizard.workbench.utils.scan_layout import is_subset_name, read_positions, split_subsets

DIR_PREFIX = "nano"
TIFF_SUFFIXES = (".tif", ".tiff")
MAX_IMAGE_SIZE = 4096
PHASE_SUBDIR = "phase_retrieval"
# Save prefix of the runs written by holowizard (paths.base_dir) and its former name: data in both are found everywhere
# (runs are listed by their phase_retrieval folder, whatever their name); these names are only preferred where one run
# has to be chosen, e.g. for the geometry of external tomograms.
DEFAULT_RUN = "holowizard"
LEGACY_RUNS = ("holopipe",)
RECON_SUBDIR = "recon_dir"

_listing_cache: Dict[str, Tuple[int, Any]] = {}
_listing_lock = threading.Lock()


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _cached_by_mtime(path: Path, compute: Callable[[Path], Any]) -> Optional[Any]:
    """``compute(path)``, recomputed only when the directory mtime changed; None if unreadable."""
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return None
    key = f"{compute.__name__}:{path}"
    with _listing_lock:
        cached = _listing_cache.get(key)
    if cached and cached[0] == mtime_ns:
        return cached[1]
    try:
        value = compute(path)
    except OSError:
        return None
    with _listing_lock:
        _listing_cache[key] = (mtime_ns, value)
    return value


def _list_tiffs(path: Path) -> List[str]:
    with os.scandir(path) as it:
        names = [e.name for e in it if e.name.lower().endswith(TIFF_SUFFIXES) and e.is_file()]
    return sorted(names, key=_natural_key)


def _list_subdirs(path: Path) -> List[str]:
    with os.scandir(path) as it:
        return sorted((e.name for e in it if e.is_dir()), key=_natural_key)


def _scan_dirs(root: Path) -> List[str]:
    return [d for d in _cached_by_mtime(root, _list_subdirs) or [] if d.startswith(DIR_PREFIX)]


def _first_middle_last(files: List[str], what: str) -> List[Dict[str, Optional[str]]]:
    return [
        {"label": f"{what} first", "file": files[0] if files else None},
        {"label": f"{what} middle", "file": files[len(files) // 2] if files else None},
        {"label": f"{what} last", "file": files[-1] if files else None},
    ]


# --- Data set types ---
@dataclass
class OverviewSource:
    key: str
    title: str
    root: Callable[[Any], Path]  # beamtime -> root folder
    datasets: Callable[[Path], List[str]]  # root -> data set ids, e.g. "nano123" or "nano123/holowizard"
    folder: Callable[[Path, str], Path]  # root, data set id -> folder with the TIFFs
    summarize: Callable[[List[str]], Tuple[str, List[Dict[str, Optional[str]]]]]  # files -> (meta, thumbnails)
    valid_id: Callable[[str], bool]
    unit: str  # what one tile is, for the counter ("3 / 5 runs")
    empty_text: str  # shown when there are no data sets yet
    rotate: bool = True  # detector images are shown rotated by 90 degrees, like the test image preview


def _raw_dataset_ok(ds: str) -> bool:
    """A scan folder ("nano123") or a tomogram / reference subfolder of a multi-tomogram scan ("nano123/tomo_00003")."""
    parts = ds.split("/")
    return _safe_parts(ds, 1) or (_safe_parts(ds, 2) and is_subset_name(parts[1]))


def _raw_summary(files):
    imgs = [f for f in files if f.startswith("img_")]
    refs = [f for f in files if f.startswith("ref_")]
    thumbs = [{"label": "ref first", "file": refs[0] if refs else None}] + _first_middle_last(imgs, "img")
    return f"{len(imgs)} img · {len(refs)} ref", thumbs


def _phase_datasets(root: Path) -> List[str]:
    ids = []
    for scan in _scan_dirs(root):
        for run in _cached_by_mtime(root / scan, _list_subdirs) or []:
            if (root / scan / run / PHASE_SUBDIR).is_dir():
                ids.append(f"{scan}/{run}")
    return ids


def _phase_summary(files):
    return f"{len(files)} images", _first_middle_last(files, "image")


def _tomo_datasets(root: Path) -> List[str]:
    ids = []
    for scan in _scan_dirs(root):
        recon = root / scan / RECON_SUBDIR
        for version in _cached_by_mtime(recon, _list_subdirs) or []:
            # recon_dir also holds folders without slices (e.g. Figures, FSC).
            if _cached_by_mtime(recon / version, _list_tiffs):
                ids.append(f"{scan}/{version}")
    return ids


def _tomo_summary(files):
    return f"{len(files)} slices", _first_middle_last(files, "slice")


def _safe_parts(dataset: str, n: int) -> bool:
    parts = dataset.split("/")
    return len(parts) == n and parts[0].startswith(DIR_PREFIX) and all(p and p not in (".", "..") for p in parts)


SOURCES = {
    "raw": OverviewSource(
        key="raw",
        title="Raw Data Overview",
        root=lambda bt: Path(bt.path_raw),
        datasets=_scan_dirs,
        folder=lambda root, ds: root / ds,
        summarize=_raw_summary,
        valid_id=_raw_dataset_ok,
        unit="directories",
        empty_text=f"No {DIR_PREFIX}* directories yet.",
    ),
    "phase": OverviewSource(
        key="phase",
        title="Phase Retrieval Overview",
        root=lambda bt: Path(bt.path_processed),
        datasets=_phase_datasets,
        folder=lambda root, ds: root / ds / PHASE_SUBDIR,
        summarize=_phase_summary,
        valid_id=lambda ds: _safe_parts(ds, 2),
        unit="runs",
        empty_text="No phase retrieval results yet.",
        # The phase retrieval stores its images already rotated by 90 degrees relative to the raw images,
        # i.e. in the orientation in which the raw overview shows the raw images.
        rotate=False,
    ),
    "tomo": OverviewSource(
        key="tomo",
        title="Tomogram Overview",
        root=lambda bt: Path(bt.path_processed),
        datasets=_tomo_datasets,
        folder=lambda root, ds: root / ds.split("/")[0] / RECON_SUBDIR / ds.split("/")[1],
        summarize=_tomo_summary,
        valid_id=lambda ds: _safe_parts(ds, 2),
        unit="reconstructions",
        empty_text="No tomographic reconstructions yet.",
        rotate=False,
    ),
}


def _multi_tomogram_summary(root: Path, scan: str, subsets: Dict[str, List[str]], with_files: bool) -> Optional[Dict]:
    """Raw scan with tomo_*/ref_* subfolders: the tile shows the first reference and the first tomogram; the
    subfolders (with their image counts) and the lateral positions are listed for the popup."""
    folder = root / scan
    listed = {
        name: _cached_by_mtime(folder / name, _list_tiffs) or [] for kind in ("tomo", "ref") for name in subsets[kind]
    }
    tomo0, refs0 = subsets["tomo"][0], (subsets["ref"][:1] or [None])[0]
    imgs = [f for f in listed[tomo0] if f.startswith("img_")]
    refs = [f for f in listed.get(refs0, []) if f.startswith("ref_")] if refs0 else []
    thumbs = [
        {"label": "ref first", "file": refs[0] if refs else None, "dataset": f"{scan}/{refs0}" if refs0 else None}
    ]
    thumbs += [dict(t, dataset=f"{scan}/{tomo0}") for t in _first_middle_last(imgs, "img")]
    n = len(subsets["tomo"])
    mtimes = [_mtime(folder)] + [_mtime(folder / name) for name in listed]
    summary = {
        "name": scan,
        "mtime": max(t for t in mtimes if t is not None) / 1e9,
        "meta": f"{n} tomogram{'s' if n != 1 else ''} · {len(imgs)} img · {len(refs)} ref",
        "thumbs": thumbs,
        "count": len(imgs),
        "subsets": {
            kind: [{"name": name, "count": len(listed[name])} for name in subsets[kind]] for kind in ("tomo", "ref")
        },
        "positions": _memo(
            ("positions", str(folder), _mtime(folder / "positions.txt")), lambda: read_positions(folder)
        ),
    }
    if with_files:
        summary["files"] = listed[tomo0]
    return summary


def _dataset_summary(source: OverviewSource, root: Path, dataset: str, with_files: bool = False) -> Optional[Dict]:
    if source.key == "raw" and "/" not in dataset:
        subsets = split_subsets(_cached_by_mtime(root / dataset, _list_subdirs) or [])
        if subsets["tomo"]:
            return _multi_tomogram_summary(root, dataset, subsets, with_files)
    folder = source.folder(root, dataset)
    files = _cached_by_mtime(folder, _list_tiffs)
    if files is None:
        return None
    meta, thumbs = source.summarize(files)
    summary = {"name": dataset, "mtime": folder.stat().st_mtime, "meta": meta, "thumbs": thumbs, "count": len(files)}
    if with_files:
        summary["files"] = files
    return summary


# --- Pixel sizes (for scale bars) ---
# The README.html of a phase retrieval run (written by Scan.write_html) lists the detector pixel size and the
# magnification. The effective pixel size of the raw detector is detector pixel / magnification; images with
# fewer pixels than the raw detector (binned phase retrieval, binned tomograms) have correspondingly larger
# pixels. Older READMEs give the detector pixel in micrometres although labelled "mm" (e.g. "6.5 mm").
README_FILE = "README.html"
# Results are kept until their inputs change: the keys contain the README mtime or the number of images.
_pixel_cache: Dict[Tuple, Any] = {}
_UNIT_UM = {"nm": 1e-3, "μm": 1.0, "µm": 1.0, "um": 1.0, "mm": 1e3, "cm": 1e4, "m": 1e6}


def _memo(key: Tuple, compute: Callable[[], Any]) -> Any:
    if key not in _pixel_cache:
        try:
            _pixel_cache[key] = compute()
        except Exception:
            _pixel_cache[key] = None
    return _pixel_cache[key]


def _mtime(path: Path) -> Optional[int]:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _readme_value(html: str, label: str) -> Tuple[Optional[float], Optional[str]]:
    m = re.search(re.escape(label) + r"\s*</strong>\s*([-0-9.eE+]+)\s*([^\s<]*)", html)
    return (float(m.group(1)), m.group(2) or None) if m else (None, None)


def _detector_pixel_nm(run_folder: Path) -> Optional[float]:
    """Effective pixel size (nm) of the raw detector images, from the README.html of a phase retrieval run:
    detector pixel / magnification. None if the geometry is not physical (e.g. a negative z01)."""
    html = (run_folder / README_FILE).read_text(errors="ignore")
    detector, unit = _readme_value(html, "Detector Pixel Size:")
    magnification, _ = _readme_value(html, "Magnification")
    if not detector or not magnification or detector <= 0 or magnification <= 1:
        return None
    if unit in _UNIT_UM:
        detector_um = detector * _UNIT_UM[unit]
    else:  # no or unknown unit: "6.5" means um, "0.0065" mm (detector pixels are a few um)
        detector_um = detector if detector > 0.5 else detector * 1000
    return detector_um * 1000 / magnification


def _first_tiff_width(folder: Path, prefix: str = "") -> Optional[int]:
    with os.scandir(folder) as it:
        names = sorted(e.name for e in it if e.name.startswith(prefix) and e.name.lower().endswith(TIFF_SUFFIXES))
    for name in names[:5]:
        try:
            with tifffile.TiffFile(str(folder / name)) as tif:
                return int(tif.pages[0].shape[1])
        except Exception:
            continue
    return None


def _image_size(folder: Path) -> Optional[List[int]]:
    with os.scandir(folder) as it:
        names = sorted(e.name for e in it if e.name.lower().endswith(TIFF_SUFFIXES))
    for name in names[:5]:
        try:
            with tifffile.TiffFile(str(folder / name)) as tif:
                h, w = tif.pages[0].shape[:2]
                return [int(w), int(h)]
        except Exception:
            continue
    return None


def _raw_width(beamtime, scan: str) -> Optional[int]:
    def compute():
        folder = Path(beamtime.path_raw) / scan
        width = _first_tiff_width(folder, "img")
        if width is None:  # multi-tomogram scan: the images are in tomo_* subfolders
            tomos = split_subsets(_list_subdirs(folder))["tomo"]
            width = _first_tiff_width(folder / tomos[0], "img") if tomos else None
        return width

    return _memo(("raw width", str(beamtime.path_raw), scan), compute)


def _phase_width(beamtime, scan: str, prefix: str) -> Optional[int]:
    folder = Path(beamtime.path_processed) / scan / prefix / PHASE_SUBDIR
    return _memo(("phase width", str(folder)), lambda: _first_tiff_width(folder))


def _phase_pixel_nm(beamtime, scan: str, prefix: str) -> Optional[float]:
    run_folder = Path(beamtime.path_processed) / scan / prefix
    readme_mtime = _mtime(run_folder / README_FILE)
    if readme_mtime is None:
        return None

    def compute():
        raw_pixel = _detector_pixel_nm(run_folder)
        raw_width, phase_width = _raw_width(beamtime, scan), _phase_width(beamtime, scan, prefix)
        if not (raw_pixel and raw_width and phase_width):
            return None
        return raw_pixel * raw_width / phase_width

    return _memo(("phase pixel", str(run_folder), readme_mtime), compute)


def _tomo_pixel(beamtime, scan: str, version: str, slice_width: Optional[int]) -> Tuple[Optional[float], str]:
    processed = Path(beamtime.path_processed)
    params = processed / scan / RECON_SUBDIR / version / "reconstruction.yaml"

    def compute():
        import yaml

        if params.exists():  # written by the tomography page
            p = yaml.safe_load(params.read_text())
            prefix = str(p["phase_retrieval_run"]).split("/")[1]
            pixel = _phase_pixel_nm(beamtime, scan, prefix)
            return (pixel * float(p.get("binning", 1)) if pixel else None), f"from {version}/reconstruction.yaml"
        # External reconstructions: geometry of the scan's phase retrieval (preferably the default run "holowizard",
        # then the legacy "holopipe", then any other run).
        preferred = [DEFAULT_RUN, *LEGACY_RUNS]
        runs = [r for r in preferred + _list_subdirs(processed / scan) if _phase_pixel_nm(beamtime, scan, r)]
        if not runs or not slice_width:
            return None, ""
        prefix = runs[0]
        pixel, phase_width = _phase_pixel_nm(beamtime, scan, prefix), _phase_width(beamtime, scan, prefix)
        # Slices are often cropped; the binning is the nearest whole number (2048 -> 1944: 1, 2048 -> 972: 2).
        binning = max(1, round(phase_width / slice_width))
        return pixel * binning, f"assumed from the {prefix} phase retrieval geometry, binning {binning}"

    key = (
        "tomo pixel",
        str(processed / scan),
        version,
        slice_width,
        _mtime(params),
        *(_mtime(processed / scan / run / README_FILE) for run in (DEFAULT_RUN, *LEGACY_RUNS)),
    )
    return _memo(key, compute) or (None, "")


def add_pixel_sizes(beamtime, source_key: str, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add ``pixel_size_nm``, ``pixel_note`` and ``image_size`` ([width, height]) to phase / tomogram summaries."""
    if source_key not in ("phase", "tomo"):
        return summaries
    src = SOURCES[source_key]
    root = src.root(beamtime)
    for s in summaries:
        if s is None or not s.get("count"):
            continue
        scan, sub = s["name"].split("/")
        folder = src.folder(root, s["name"])
        size = _memo(("size", str(folder), s["count"]), lambda: _image_size(folder))
        s["image_size"] = size
        if source_key == "phase":
            s["pixel_size_nm"] = _phase_pixel_nm(beamtime, scan, sub)
            s["pixel_note"] = f"from {sub}/{README_FILE}"
        else:
            s["pixel_size_nm"], s["pixel_note"] = _tomo_pixel(beamtime, scan, sub, size[0] if size else None)
            if size:
                w, h = size
                s["ortho"] = {
                    "xz": {"positions": ortho_positions(h), "size": [w, s["count"]]},
                    "yz": {"positions": ortho_positions(w), "size": [h, s["count"]]},
                }
    return summaries


def list_datasets(source: OverviewSource, root: Path) -> List[Dict[str, Any]]:
    summaries = (_dataset_summary(source, root, ds) for ds in source.datasets(root))
    return [s for s in summaries if s is not None]


def list_summary(beamtime) -> List[Dict[str, Any]]:
    """Per scan: the raw data set and all phase retrieval runs and tomogram versions (newest first)."""
    raw_src, phase_src, tomo_src = SOURCES["raw"], SOURCES["phase"], SOURCES["tomo"]
    raw_root, processed_root = raw_src.root(beamtime), phase_src.root(beamtime)
    raw = {d["name"]: d for d in list_datasets(raw_src, raw_root)} if raw_root.is_dir() else {}

    def by_scan(src: OverviewSource) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        if processed_root.is_dir():
            for d in list_datasets(src, processed_root):
                grouped.setdefault(d["name"].split("/")[0], []).append(d)
        for runs in grouped.values():
            runs.sort(key=lambda d: d["mtime"], reverse=True)
        return grouped

    phase, tomo = by_scan(phase_src), by_scan(tomo_src)
    for runs in phase.values():
        add_pixel_sizes(beamtime, "phase", runs)
    for versions in tomo.values():
        add_pixel_sizes(beamtime, "tomo", versions)
    summary = []
    for name in sorted(set(raw) | set(phase) | set(tomo), key=_natural_key):
        parts = [raw.get(name)] + phase.get(name, []) + tomo.get(name, [])
        summary.append(
            {
                "name": name,
                "mtime": max(d["mtime"] for d in parts if d),
                "raw": raw.get(name),
                "phase": phase.get(name, []),
                "tomo": tomo.get(name, []),
            }
        )
    return summary


# --- Image rendering ---
def _downsample(arr: np.ndarray, size: int) -> Tuple[np.ndarray, int]:
    factor = max(1, int(np.ceil(max(arr.shape) / size)))
    if factor == 1:
        return arr.astype(np.float32), 1
    h, w = (arr.shape[0] // factor) * factor, (arr.shape[1] // factor) * factor
    arr = arr[:h, :w].astype(np.float32)
    return arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3)), factor


def _render_png(path: str, mtime_ns: int, size: int, rotate: bool = True) -> Tuple[bytes, int, float, float]:
    """Downsampled, percentile-scaled 8 bit PNG of the first page of a TIFF.

    ``mtime_ns`` is only part of the cache key, so rewritten files are re-rendered.
    """
    with tifffile.TiffFile(path) as tif:
        arr = tif.pages[0].asarray()
    small, factor = _downsample(arr, size)
    # Same orientation as the image preview on the phase retrieval page.
    if rotate:
        small = np.rot90(small)
    # Phase maps are float and may contain NaN; those pixels are drawn black.
    lo, hi = np.nanpercentile(small, [1, 99]) if np.isfinite(small).any() else (0.0, 1.0)
    scaled = np.nan_to_num(np.clip((small - lo) / max(hi - lo, 1e-12), 0, 1))
    buf = io.BytesIO()
    Image.fromarray((scaled * 255).astype(np.uint8)).save(buf, format="PNG", compress_level=1)
    return buf.getvalue(), factor, float(lo), float(hi)


# --- Orthogonal sections of tomograms (XZ at fixed y, YZ at fixed x) ---
ORTHO_FRACTIONS = (0.25, 0.5, 0.75)
_ortho_cache: Dict[Tuple, Dict[str, List[np.ndarray]]] = {}
_ortho_locks: Dict[Tuple, threading.Lock] = {}
_ortho_lock = threading.Lock()
ORTHO_CACHE_SIZE = 64


def ortho_positions(n: int) -> List[int]:
    return [min(n - 1, int(n * f)) for f in ORTHO_FRACTIONS]


def _ortho_sections(folder: Path, files: List[str], size: int) -> Dict[str, List[np.ndarray]]:
    """XZ and YZ sections at 25/50/75 % of y and x, from every k-th slice (one pass; z down along the slices)."""
    step = max(1, int(np.ceil(len(files) / size)))
    chosen = files[::step]
    with tifffile.TiffFile(str(folder / chosen[0])) as tif:
        h, w = tif.pages[0].shape[:2]
    ys, xs = ortho_positions(h), ortho_positions(w)

    def read(name):
        try:
            try:
                img = tifffile.memmap(str(folder / name), mode="r")
            except Exception:  # compressed TIFF: read it completely
                img = tifffile.imread(str(folder / name))
            if img.shape[:2] != (h, w):
                return None
            return np.array(img[ys, :], dtype=np.float32), np.array(img[:, xs], dtype=np.float32).T
        except Exception:
            return None  # e.g. a slice that is still being written

    with ThreadPoolExecutor(32) as ex:
        parts = [r for r in ex.map(read, chosen) if r is not None]
    if not parts:
        raise ValueError("No readable slices")
    rows = np.stack([r for r, _ in parts], axis=1)  # (3, z, w)
    cols = np.stack([c for _, c in parts], axis=1)  # (3, z, h)

    def shrink(a):  # along the width to at most `size` pixels (block mean)
        f = max(1, int(np.ceil(a.shape[1] / size)))
        n = (a.shape[1] // f) * f
        return a[:, :n].reshape(a.shape[0], n // f, f).mean(axis=2)

    return {"xz": [shrink(r) for r in rows], "yz": [shrink(c) for c in cols]}


def ortho_png(folder: Path, files: List[str], plane: str, index: int, size: int) -> bytes:
    key = (str(folder), len(files), files[-1] if files else "", size)
    with _ortho_lock:
        sections = _ortho_cache.get(key)
        lock = _ortho_locks.setdefault(key, threading.Lock())
    if sections is None:
        with lock:  # the six thumbnails of a tile arrive together; compute once
            with _ortho_lock:
                sections = _ortho_cache.get(key)
            if sections is None:
                sections = _ortho_sections(folder, files, size)
                with _ortho_lock:
                    _ortho_cache[key] = sections
                    while len(_ortho_cache) > ORTHO_CACHE_SIZE:
                        _ortho_cache.pop(next(iter(_ortho_cache)))
                    _ortho_locks.pop(key, None)
    img = sections[plane][index]
    lo, hi = np.nanpercentile(img, [1, 99]) if np.isfinite(img).any() else (0.0, 1.0)
    scaled = np.nan_to_num(np.clip((img - lo) / max(hi - lo, 1e-12), 0, 1))
    buf = io.BytesIO()
    Image.fromarray((scaled * 255).astype(np.uint8)).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


# Thumbnails are small and requested on every page load, popup images are large.
_render_thumb = lru_cache(maxsize=4096)(_render_png)
_render_large = lru_cache(maxsize=64)(_render_png)
THUMB_MAX_SIZE = 256


# --- Routes ---
router = APIRouter(tags=["overview"])


def _source(source: str) -> OverviewSource:
    if source not in SOURCES:
        raise HTTPException(status_code=404, detail=f"Unknown overview: {source!r}")
    return SOURCES[source]


def _dataset_folder(request: Request, src: OverviewSource, dataset: str) -> Path:
    if not src.valid_id(dataset):
        raise HTTPException(status_code=400, detail=f"Invalid data set: {dataset!r}")
    folder = src.folder(src.root(request.app.state.beamtime), dataset)
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Data set not found: {dataset}")
    return folder


@router.get("/api/overview/summary")
async def overview_summary(request: Request):
    beamtime = request.app.state.beamtime
    path = f"{SOURCES['raw'].root(beamtime)}  ·  {SOURCES['phase'].root(beamtime)}"
    return {"path": path, "dirs": await run_in_threadpool(list_summary, beamtime)}


@router.get("/api/overview/{source}/dirs")
async def overview_dirs(source: str, request: Request):
    src = _source(source)
    root = src.root(request.app.state.beamtime)
    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {root}")
    beamtime = request.app.state.beamtime
    dirs = await run_in_threadpool(lambda: add_pixel_sizes(beamtime, source, list_datasets(src, root)))
    return {"path": str(root), "dirs": dirs}


@router.get("/api/overview/{source}/files")
async def overview_files(source: str, request: Request, dataset: str):
    src = _source(source)
    _dataset_folder(request, src, dataset)
    beamtime = request.app.state.beamtime
    summary = await run_in_threadpool(_dataset_summary, src, src.root(beamtime), dataset, True)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Data set not readable: {dataset}")
    await run_in_threadpool(add_pixel_sizes, beamtime, source, [summary])
    return summary


@router.get("/api/overview/tomo/ortho")
async def overview_tomo_ortho(
    request: Request,
    dataset: str,
    plane: str,
    index: int = Query(..., ge=0, le=2),
    size: int = Query(256, ge=32, le=1024),
):
    """XZ or YZ section (at 25/50/75 % of y or x) of a tomogram, as PNG."""
    if plane not in ("xz", "yz"):
        raise HTTPException(status_code=400, detail="plane must be xz or yz")
    src = SOURCES["tomo"]
    folder = _dataset_folder(request, src, dataset)
    files = await run_in_threadpool(_cached_by_mtime, folder, _list_tiffs)
    if not files:
        raise HTTPException(status_code=404, detail="No slices")
    try:
        png = await run_in_threadpool(ortho_png, folder, files, plane, index, size)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not read the sections: {e}")
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@router.get("/api/overview/{source}/image")
async def overview_image(
    source: str,
    request: Request,
    dataset: str,
    file: str,
    size: int = Query(256, ge=32, le=MAX_IMAGE_SIZE),
    rotate: Optional[bool] = None,  # default: as configured for the data type
):
    src = _source(source)
    if file != Path(file).name or not file.lower().endswith(TIFF_SUFFIXES):
        raise HTTPException(status_code=400, detail=f"Invalid file name: {file!r}")
    path = _dataset_folder(request, src, dataset) / file
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        raise HTTPException(status_code=404, detail=f"Image not found: {file}")
    try:
        png, factor, lo, hi = await run_in_threadpool(
            _render_thumb if size <= THUMB_MAX_SIZE else _render_large,
            str(path),
            mtime_ns,
            size,
            src.rotate if rotate is None else rotate,
        )
    except Exception as e:
        # Most likely a file that is still being written; the client retries later.
        raise HTTPException(status_code=503, detail=f"Could not read {file}: {e}")
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "Cache-Control": "max-age=3600",
            "X-Downsample": str(factor),
            "X-Range": f"{lo:g},{hi:g}",
            "Access-Control-Expose-Headers": "X-Downsample, X-Range",
        },
    )


@lru_cache(maxsize=4)
def _read_whole(path: str, mtime_ns: int) -> np.ndarray:
    """First page of a TIFF that cannot be memory-mapped (compressed); ``mtime_ns`` is only part of the cache key."""
    with tifffile.TiffFile(path) as tif:
        return tif.pages[0].asarray()


def _pixel_value(path: str, mtime_ns: int, x: int, y: int, rotate: bool) -> Optional[float]:
    """Value of the full resolution pixel at (x, y) of the image as displayed (rotated by 90 degrees if ``rotate``,
    like the rendered images); None outside the image."""
    try:
        img = tifffile.memmap(path, mode="r")
    except Exception:
        img = _read_whole(path, mtime_ns)
    if img.ndim > 2:
        img = img.reshape(-1, *img.shape[-2:])[0]
    h, w = img.shape
    # np.rot90(a)[i, j] == a[j, w - 1 - i]
    row, col = (x, w - 1 - y) if rotate else (y, x)
    if not (0 <= row < h and 0 <= col < w):
        return None
    return float(img[row, col])


@router.get("/api/overview/{source}/value")
async def overview_value(source: str, request: Request, dataset: str, file: str, x: int, y: int):
    """Pixel value at (x, y) in the coordinates of the displayed image (full resolution, default orientation)."""
    src = _source(source)
    if file != Path(file).name or not file.lower().endswith(TIFF_SUFFIXES):
        raise HTTPException(status_code=400, detail=f"Invalid file name: {file!r}")
    path = _dataset_folder(request, src, dataset) / file
    try:
        mtime_ns = path.stat().st_mtime_ns
        value = await run_in_threadpool(_pixel_value, str(path), mtime_ns, x, y, src.rotate)
    except OSError:
        raise HTTPException(status_code=404, detail=f"Image not found: {file}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not read {file}: {e}")
    # JSON has no NaN: report it separately (phase maps may contain NaN).
    is_nan = value is not None and value != value
    return {"value": None if is_nan else value, "nan": is_nan, "inside": value is not None}


# Views of the single overview page, in the order of its "View" dropdown; "raw" is the default.
VIEWS = {"raw": "Raw data", "phase": "Phase retrieval", "tomo": "Tomograms", "summary": "Summary"}


def _overview_page(request: Request, source: str):
    if source == "summary":
        title, unit, empty_text = "Summary Overview", "scans", f"No {DIR_PREFIX}* scans yet."
    else:
        src = SOURCES[source]
        title, unit, empty_text = src.title, src.unit, src.empty_text
    return request.app.state.templates.TemplateResponse(
        "overview.html",
        {
            "request": request,
            "prefix": DIR_PREFIX,
            "source": source,
            "title": title,
            "unit": unit,
            "empty_text": empty_text,
            "views": VIEWS,
        },
    )


@router.get("/overview")
async def overview_page(request: Request, view: str = "raw"):
    if view not in VIEWS:
        return RedirectResponse(url=str(request.url_for("overview_page")))
    return _overview_page(request, view)


# Former separate overview pages, kept so that bookmarks keep working.
@router.get("/phase-overview")
async def phase_overview_page(request: Request):
    return RedirectResponse(url=f"{request.url_for('overview_page')}?view=phase")


@router.get("/tomo-overview")
async def tomo_overview_page(request: Request):
    return RedirectResponse(url=f"{request.url_for('overview_page')}?view=tomo")


@router.get("/summary-overview")
async def summary_overview_page(request: Request):
    return RedirectResponse(url=f"{request.url_for('overview_page')}?view=summary")
