"""Rotation center page (rotation.html): find the rotation axis of a tomogram by eye.

For a phase retrieval run (``path_processed/<scan>/<save prefix>/phase_retrieval``) and one detector row
(the test slice), the sinogram is read from the projections and reconstructed with astra's FBP_CUDA for
every candidate center in a range (default: center pixel +/- 100 px). The chosen center is stored in
``rotation_center.yaml`` in the run folder.

Bad projections are left out before the reconstruction (filters of the P05 nanotomography reconstruction
script, applied in its order on whole images; the statistics are computed once per run and cached):

1. projections whose phase image contains NaN,
2. dark holograms: mean of the raw hologram < ``dark_factor`` * |mean of the remaining holograms|,
3. phase outliers: minimum of the phase image < ``min_factor`` * median of the remaining minima, or == 0,
4. optionally, on the test slice: projections that differ strongly from their neighbours in angle.

Conventions: the rotation center is a detector column index (0-based, pixel centres). Projection number
``k`` (last number in the file name) belongs to the row with image number ``k`` in the raw scan log
(``__LogScan.log``); images without a log row (the last image of P05 scans) are not used.
"""

import datetime
import io
import re
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tifffile
import yaml
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel

from holowizard.pipe.server.routes.overview import SOURCES, list_datasets

CENTER_FILE = "rotation_center.yaml"
CACHE_DIR = "rotation_center_cache"  # in the beamtime log_path
DEFAULT_RANGE = 100
MAX_CENTERS = 2001
# Fewer projections cannot give a usable tomogram (e.g. single test reconstructions of the phase retrieval page).
MIN_PROJECTIONS = 20
# Neighbour test on the test slice: projections that differ from their neighbours in angle by more than
# OUTLIER_FACTOR times the typical deviation, or whose largest deviation exceeds OUTLIER_PEAK_FACTOR times
# the usual largest deviation.
OUTLIER_FACTOR = 8.0
OUTLIER_PEAK_FACTOR = 20.0
OUTLIER_WINDOW = 5
MAX_OUTLIER_FRACTION = 0.25
READ_THREADS = 32
SINOGRAM_CACHE_SIZE = 8  # test slices kept in memory (one row of every projection each)
STATS_THREADS = 48
ALGORITHM = "FBP_CUDA"
PROJECTION_IDENTIFIER = re.compile(r"img(_y\d+)?")  # "img" (older scans) or "img_y0", ... (newer scans)

router = APIRouter(tags=["rotation"])
_gpu_lock = threading.Lock()  # astra/GPU work one reconstruction at a time
_jobs: Dict[str, "CenterScan"] = {}
_jobs_lock = threading.Lock()
_user_jobs: Dict[str, str] = {}


def _file_index(name: str) -> Optional[int]:
    m = re.search(r"(\d+)\D*$", name)
    return int(m.group(1)) if m else None


# --- Data set / geometry ---
def _run_folders(beamtime, dataset: str) -> Tuple[Path, Path, Path]:
    """(run folder, projection folder, raw scan folder) of a phase retrieval run "scan/prefix"."""
    src = SOURCES["phase"]
    if not src.valid_id(dataset):
        raise HTTPException(status_code=400, detail=f"Invalid data set: {dataset!r}")
    processed = src.root(beamtime)
    scan, prefix = dataset.split("/")
    projections = src.folder(processed, dataset)
    if not projections.is_dir():
        raise HTTPException(status_code=404, detail=f"Data set not found: {dataset}")
    return processed / scan / prefix, projections, Path(beamtime.path_raw) / scan


@lru_cache(maxsize=64)
def _scan_angles(raw_scan: str, mtime_ns: int) -> Dict[int, float]:
    """Rotation angle (radians) per image number, from the projection rows of the raw scan log."""
    import pandas as pd

    log = Path(raw_scan) / f"{Path(raw_scan).name}__LogScan.log"
    columns = ["identifier", "info", "number_1", "number_2", "number", "time", "current", "rotation"]
    df = pd.read_csv(log, comment="#", sep=r"\s+", names=columns, usecols=range(8))
    df = df[df["identifier"].astype(str).str.fullmatch(PROJECTION_IDENTIFIER)]
    angles: Dict[int, float] = {}
    for number, rotation in zip(df["number"].astype(int), df["rotation"].astype(float)):
        angles.setdefault(int(number), float(np.radians(rotation)))
    return angles


def _projections_and_angles(projections: Path, raw_scan: Path) -> Tuple[List[Path], np.ndarray, str]:
    files = sorted((p for p in projections.iterdir() if p.suffix.lower() in (".tif", ".tiff")), key=lambda p: p.name)
    if not files:
        raise ValueError("No projections in this run yet.")
    try:
        log = raw_scan / f"{raw_scan.name}__LogScan.log"
        angles = _scan_angles(str(raw_scan), log.stat().st_mtime_ns)
    except Exception as e:
        raise ValueError(f"Could not read the rotation angles from the raw scan log: {e}")
    used, theta = [], []
    for f in files:
        k = _file_index(f.name)
        if k is not None and k in angles:
            used.append(f)
            theta.append(angles[k])
    if len(used) < MIN_PROJECTIONS:
        raise ValueError(
            f"Only {len(used)} of {len(files)} projections have a rotation angle in the raw scan log; "
            f"at least {MIN_PROJECTIONS} are needed for a test reconstruction. "
            "Runs with a single image are test reconstructions, not tomography data."
        )
    return used, np.asarray(theta), f"{len(used)} projections, angles from {raw_scan.name}__LogScan.log"


def _image_shape(files: List[Path]) -> Tuple[int, int]:
    """Shape of the first readable projection (files of an interrupted run can be truncated)."""
    for f in files:
        try:
            with tifffile.TiffFile(str(f)) as tif:
                return tuple(tif.pages[0].shape[:2])
        except Exception:
            continue
    raise ValueError("None of the projections of this run can be read.")


# --- Projection quality (whole images, once per run) ---
@dataclass
class ProjectionStats:
    names: List[str]
    readable: np.ndarray  # bool: phase image readable with the expected shape
    has_nan: np.ndarray  # bool
    minimum: np.ndarray  # float, minimum of the phase image (NaN if unreadable)
    hologram_mean: np.ndarray  # float, mean of the raw hologram (NaN if missing)


_stats_cache: Dict[Tuple, ProjectionStats] = {}
_stats_locks: Dict[str, threading.Lock] = {}
_stats_locks_lock = threading.Lock()


def _raw_holograms(raw_scan: Path) -> Dict[int, Path]:
    holograms = {}
    for p in raw_scan.iterdir():
        if p.name.startswith("img") and p.suffix.lower() in (".tif", ".tiff"):
            k = _file_index(p.name)
            if k is not None:
                holograms[k] = p
    return holograms


def _compute_stats(files: List[Path], raw_scan: Path, shape: Tuple[int, int], progress=None) -> ProjectionStats:
    holograms = _raw_holograms(raw_scan)

    def one(f: Path):
        readable, has_nan, minimum, mean = False, False, np.nan, np.nan
        try:
            img = tifffile.memmap(str(f), mode="r")
            if img.shape[:2] == shape:
                readable = True
                has_nan = bool(np.isnan(img).any())
                minimum = float(np.nanmin(img)) if not np.isnan(img).all() else np.nan
        except Exception:
            pass
        hologram = holograms.get(_file_index(f.name))
        if hologram is not None:
            try:
                mean = float(tifffile.memmap(str(hologram), mode="r").mean(dtype=np.float64))
            except Exception:
                pass
        return readable, has_nan, minimum, mean

    results = []
    with ThreadPoolExecutor(STATS_THREADS) as ex:
        for i, r in enumerate(ex.map(one, files)):
            results.append(r)
            if progress and i % 32 == 0:
                progress(i)
    readable, has_nan, minimum, mean = (np.array(v) for v in zip(*results))
    return ProjectionStats([f.name for f in files], readable, has_nan, minimum.astype(float), mean.astype(float))


def projection_stats(
    beamtime, dataset: str, files: List[Path], raw_scan: Path, shape: Tuple[int, int], progress=None
) -> ProjectionStats:
    """Statistics of all projections of a run, cached in memory and in ``<log_path>/rotation_center_cache``."""
    folder = files[0].parent
    key = (str(folder), folder.stat().st_mtime_ns, len(files), str(raw_scan))
    with _stats_locks_lock:
        lock = _stats_locks.setdefault(str(folder), threading.Lock())
    with lock:  # one computation per run, also for concurrent scans
        if key in _stats_cache:
            return _stats_cache[key]
        cache_file = None
        log_path = getattr(beamtime, "log_path", None)
        if log_path is not None:
            cache_file = Path(log_path) / CACHE_DIR / f"{dataset.replace('/', '__')}.npz"
            try:
                with np.load(cache_file, allow_pickle=False) as c:
                    if tuple(str(v) for v in c["key"]) == tuple(str(v) for v in key):
                        stats = ProjectionStats(
                            list(c["names"]), c["readable"], c["has_nan"], c["minimum"], c["hologram_mean"]
                        )
                        _stats_cache[key] = stats
                        return stats
            except Exception:
                pass
        stats = _compute_stats(files, raw_scan, shape, progress)
        _stats_cache[key] = stats
        if cache_file is not None:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_file.with_name(f".{cache_file.stem}.{uuid.uuid4().hex}.npz")
                np.savez(
                    tmp,
                    key=np.array([str(v) for v in key]),
                    names=np.array(stats.names),
                    readable=stats.readable,
                    has_nan=stats.has_nan,
                    minimum=stats.minimum,
                    hologram_mean=stats.hologram_mean,
                )
                tmp.replace(cache_file)
            except OSError:
                pass  # the cache is optional
        return stats


class Filters(BaseModel):
    nan: bool = True
    dark: bool = True
    dark_factor: float = 0.5
    phase_min: bool = True
    min_factor: float = 5.0
    neighbours: bool = False


def apply_filters(stats: ProjectionStats, filters: Filters) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """Mask of the projections to keep, number removed per filter, and notes (in the reconstruction script's order)."""
    keep = stats.readable.copy()
    removed = {"unreadable": int((~keep).sum()), "nan": 0, "dark": 0, "phase_min": 0}
    notes: List[str] = []
    if filters.nan:
        drop = keep & stats.has_nan
        removed["nan"], keep = int(drop.sum()), keep & ~drop
    if filters.dark:
        means = stats.hologram_mean
        known = keep & np.isfinite(means)
        if known.any():
            threshold = filters.dark_factor * abs(float(np.mean(means[known])))
            drop = known & (means < threshold)
            removed["dark"], keep = int(drop.sum()), keep & ~drop
        else:
            notes.append("Dark hologram filter skipped: no raw holograms found.")
    if filters.phase_min:
        minima = stats.minimum
        known = keep & np.isfinite(minima)
        median = float(np.median(minima[known])) if known.any() else 0.0
        if median < 0:
            threshold = filters.min_factor * median
            drop = known & ((minima < threshold) | (minima == 0))
            removed["phase_min"], keep = int(drop.sum()), keep & ~drop
        else:
            notes.append("Phase minimum filter skipped: the median minimum is not negative.")
    return keep, removed, notes


# --- Sinogram and reconstruction ---
def _read_sinogram(files: List[Path], row: int, shape: Tuple[int, int], progress=None) -> Tuple[np.ndarray, np.ndarray]:
    """Row ``row`` of every projection; returns the sinogram (zeros for unreadable projections) and a mask of the
    projections that could be read."""

    def read(f):
        try:
            img = tifffile.memmap(str(f), mode="r")
            if img.shape[:2] != shape:
                return None
            return np.array(img[row], dtype=np.float32)
        except Exception:
            return None  # truncated or otherwise unreadable file

    rows = []
    with ThreadPoolExecutor(READ_THREADS) as ex:
        for i, r in enumerate(ex.map(read, files)):
            rows.append(r)
            if progress and i % 64 == 0:
                progress(i)
    ok = np.array([r is not None for r in rows])
    sino = np.zeros((len(files), shape[1]), dtype=np.float32)
    for i, r in enumerate(rows):
        if r is not None:
            sino[i] = r
    return np.nan_to_num(sino, nan=0.0, posinf=0.0, neginf=0.0), ok


_sinogram_cache: "OrderedDict[Tuple, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
_sinogram_locks: Dict[Tuple, threading.Lock] = {}
_sinogram_cache_lock = threading.Lock()


def cached_sinogram(
    files: List[Path], row: int, shape: Tuple[int, int], progress=None
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Like ``_read_sinogram`` for all projections of a run, but kept in memory for the last few slices, so a new
    scan of the same slice (other range, guess or filters) does not read the projections again. The cache entry
    is renewed when files are added to the run. Returns (sinogram, readable mask, read from cache)."""
    folder = files[0].parent
    key = (str(folder), folder.stat().st_mtime_ns, len(files), row, shape)
    with _sinogram_cache_lock:
        if key in _sinogram_cache:
            _sinogram_cache.move_to_end(key)
            return (*_sinogram_cache[key], True)
        lock = _sinogram_locks.setdefault(key, threading.Lock())
    with lock:  # concurrent scans of the same slice read it once
        with _sinogram_cache_lock:
            if key in _sinogram_cache:
                return (*_sinogram_cache[key], True)
        sino, ok = _read_sinogram(files, row, shape, progress)
        with _sinogram_cache_lock:
            _sinogram_cache[key] = (sino, ok)
            while len(_sinogram_cache) > SINOGRAM_CACHE_SIZE:
                _sinogram_cache.popitem(last=False)
            _sinogram_locks.pop(key, None)
        return sino, ok, False


def outlier_projections(sino: np.ndarray) -> np.ndarray:
    """Mask of projections that differ strongly from their neighbours (the sinogram is smooth along the angle)."""
    from scipy.ndimage import median_filter

    reference = median_filter(sino, size=(OUTLIER_WINDOW, 1), mode="nearest")
    deviation = np.abs(sino - reference)
    typical, peak = np.median(deviation, axis=1), deviation.max(axis=1)
    return (typical > OUTLIER_FACTOR * max(float(np.median(typical)), 1e-12)) | (
        peak > OUTLIER_PEAK_FACTOR * max(float(np.median(peak)), 1e-12)
    )


def fbp(sino: np.ndarray, angles: np.ndarray, center: float) -> np.ndarray:
    """FBP_CUDA reconstruction of one slice with the rotation axis at detector column ``center``."""
    import astra

    width = sino.shape[1]
    geometry = astra.create_proj_geom("parallel", 1.0, width, angles)
    # astra puts the axis at the detector centre; shift the detector so it lies at column `center`.
    geometry = astra.functions.geom_postalignment(geometry, (width - 1) / 2 - center)
    volume = astra.create_vol_geom(width, width)
    with _gpu_lock:
        sino_id = astra.data2d.create("-sino", geometry, sino)
        rec_id = astra.data2d.create("-vol", volume)
        try:
            cfg = astra.astra_dict(ALGORITHM)
            cfg["ReconstructionDataId"] = rec_id
            cfg["ProjectionDataId"] = sino_id
            alg_id = astra.algorithm.create(cfg)
            try:
                astra.algorithm.run(alg_id)
                return astra.data2d.get(rec_id)
            finally:
                astra.algorithm.delete(alg_id)
        finally:
            astra.data2d.delete([sino_id, rec_id])


def _to_png(img: np.ndarray, lo: float, hi: float) -> bytes:
    scaled = np.clip((img - lo) / max(hi - lo, 1e-12), 0, 1)
    buf = io.BytesIO()
    Image.fromarray((scaled * 255).astype(np.uint8)).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


# --- Center scan job ---
class CenterScan:
    """Filters the projections, reads the sinogram of one slice and reconstructs it for every center (in a thread)."""

    def __init__(
        self,
        beamtime,
        dataset: str,
        files: List[Path],
        angles: np.ndarray,
        raw_scan: Path,
        row: int,
        centers: np.ndarray,
        shape: Tuple[int, int],
        filters: Filters,
    ):
        self.id = uuid.uuid4().hex
        self.beamtime, self.dataset, self.raw_scan, self.filters = beamtime, dataset, raw_scan, filters
        self.files, self.angles, self.row, self.centers, self.shape = files, angles, row, centers, shape
        self.removed: Dict[str, int] = {}
        self.notes: List[str] = []
        self.used = 0
        self.from_cache = False
        self.status, self.message = "checking", "Checking projections…"
        self.check_count = 0
        self.read_count = 0
        self.images: Dict[float, bytes] = {}
        self.sino: Optional[np.ndarray] = None
        self.contrast: Optional[Tuple[float, float]] = None
        self.cancelled = False
        self._lock = threading.Lock()
        threading.Thread(target=self._run, name=f"center-scan-{self.id[:8]}", daemon=True).start()

    def _run(self):
        try:
            stats = projection_stats(
                self.beamtime,
                self.dataset,
                self.files,
                self.raw_scan,
                self.shape,
                progress=lambda i: setattr(self, "check_count", i),
            )
            self.check_count = len(self.files)
            keep, self.removed, self.notes = apply_filters(stats, self.filters)
            if keep.sum() < MIN_PROJECTIONS:
                raise ValueError(f"Only {int(keep.sum())} projections are left after filtering.")
            self.status, self.message = "reading", "Reading projections…"
            # The test slice of all projections is cached; the filters only select from it.
            sino, ok, self.from_cache = cached_sinogram(
                self.files, self.row, self.shape, progress=lambda i: setattr(self, "read_count", i)
            )
            use = keep & ok
            self.removed["unreadable"] += int((keep & ~ok).sum())
            if use.sum() < MIN_PROJECTIONS:
                raise ValueError(f"Only {int(use.sum())} projections are left after filtering.")
            sino, angles = sino[use], self.angles[use]
            if self.filters.neighbours:
                bad = outlier_projections(sino)
                if bad.sum() <= MAX_OUTLIER_FRACTION * len(bad):
                    sino, angles, self.removed["neighbours"] = sino[~bad], angles[~bad], int(bad.sum())
                else:
                    self.notes.append(f"Neighbour test flagged {int(bad.sum())} projections; too many, all are used.")
            self.read_count = len(self.files)
            self.sino, self.angles, self.used = sino, angles, len(angles)

            # Common contrast for all centers, from the reconstruction at the middle of the range.
            middle = fbp(self.sino, self.angles, float(self.centers[len(self.centers) // 2]))
            self.contrast = tuple(float(v) for v in np.percentile(middle, [0.5, 99.5]))
            self.status, self.message = "reconstructing", "Reconstructing…"
            for c in self.centers:
                if self.cancelled:
                    self.status, self.message = "cancelled", "Cancelled"
                    return
                self.image(float(c))
            self.status, self.message = "done", "Done"
        except Exception as e:
            self.status, self.message = "error", str(e)

    def image(self, center: float) -> bytes:
        """PNG of the slice for ``center`` (reconstructed now if it is not done yet)."""
        key = round(center, 3)
        with self._lock:
            png = self.images.get(key)
        if png is None:
            png = _to_png(fbp(self.sino, self.angles, center), *self.contrast)
            with self._lock:
                self.images[key] = png
        return png

    def state(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "checked": self.check_count,
            "read": self.read_count,
            "projections": len(self.files),
            "used": self.used,
            "from_cache": self.from_cache,
            "removed": self.removed,
            "notes": self.notes,
            "done": len(self.images),
            "total": len(self.centers),
            "centers": [float(c) for c in self.centers],
        }


# --- Routes ---
class ScanRequest(BaseModel):
    dataset: str
    slice: Optional[int] = None
    center: Optional[float] = None  # initial guess; default: center pixel
    range: float = DEFAULT_RANGE
    step: float = 1.0
    filters: Filters = Filters()


class SaveRequest(BaseModel):
    dataset: str
    center: float
    slice: int
    range: Optional[List[float]] = None
    filters: Optional[Filters] = None
    projections_used: Optional[int] = None


def _read_saved(run_folder: Path) -> Optional[Dict[str, Any]]:
    try:
        data = yaml.safe_load((run_folder / CENTER_FILE).read_text())
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


def _dataset_info(beamtime, dataset: str) -> Dict[str, Any]:
    run_folder, projections, raw_scan = _run_folders(beamtime, dataset)
    files, angles, angle_info = _projections_and_angles(projections, raw_scan)
    rows, cols = _image_shape(files)
    return {
        "dataset": dataset,
        "rows": rows,
        "cols": cols,
        "projections": len(files),
        "angle_range_deg": float(np.degrees(angles.max() - angles.min())),
        "angle_info": angle_info,
        "default_center": cols // 2,
        "saved": _read_saved(run_folder),
        "config_file": str(run_folder / CENTER_FILE),
    }


@router.get("/api/rotation/datasets")
async def rotation_datasets(request: Request):
    src = SOURCES["phase"]
    processed = src.root(request.app.state.beamtime)
    if not processed.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {processed}")
    runs = await run_in_threadpool(list_datasets, src, processed)
    # Only runs with enough projections for a tomogram (hides single test reconstructions).
    return {"datasets": [{"name": r["name"], "meta": r["meta"]} for r in runs if r["count"] >= MIN_PROJECTIONS]}


@router.get("/api/rotation/dataset")
async def rotation_dataset(request: Request, dataset: str):
    try:
        return await run_in_threadpool(_dataset_info, request.app.state.beamtime, dataset)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/api/rotation/scan")
async def start_center_scan(req: ScanRequest, request: Request):
    beamtime = request.app.state.beamtime
    try:
        info = await run_in_threadpool(_dataset_info, beamtime, req.dataset)
        _, projections, raw_scan = _run_folders(beamtime, req.dataset)
        files, angles, _ = await run_in_threadpool(_projections_and_angles, projections, raw_scan)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    row = info["rows"] // 2 if req.slice is None else req.slice
    if not 0 <= row < info["rows"]:
        raise HTTPException(status_code=400, detail=f"Slice must be between 0 and {info['rows'] - 1}")
    if req.step <= 0 or req.range < 0:
        raise HTTPException(status_code=400, detail="Range must be >= 0 and step > 0")
    guess = info["default_center"] if req.center is None else req.center
    centers = np.arange(guess - req.range, guess + req.range + req.step / 2, req.step)
    if len(centers) > MAX_CENTERS:
        raise HTTPException(status_code=400, detail=f"At most {MAX_CENTERS} centers per scan; increase the step")

    job = CenterScan(
        beamtime, req.dataset, files, angles, raw_scan, row, centers, (info["rows"], info["cols"]), req.filters
    )
    user = getattr(request.state, "user", None) or "anonymous"
    with _jobs_lock:
        # A new scan of the same user replaces the previous one.
        previous = _jobs.pop(_user_jobs.get(user, ""), None)
        if previous is not None:
            previous.cancelled = True
        _jobs[job.id] = job
        _user_jobs[user] = job.id
    return job.state()


def _job(job_id: str) -> CenterScan:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Scan not found (replaced by a newer scan or server restarted)")
    return job


@router.get("/api/rotation/scan/{job_id}")
async def center_scan_state(job_id: str):
    return _job(job_id).state()


@router.get("/api/rotation/scan/{job_id}/image")
async def center_scan_image(job_id: str, center: float):
    job = _job(job_id)
    if job.status == "error":
        raise HTTPException(status_code=500, detail=job.message)
    if job.sino is None or job.contrast is None:
        raise HTTPException(status_code=503, detail="Projections are still being read")
    png = await run_in_threadpool(job.image, center)
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.post("/api/rotation/center")
async def save_center(req: SaveRequest, request: Request):
    beamtime = request.app.state.beamtime
    run_folder, projections, raw_scan = _run_folders(beamtime, req.dataset)
    info = await run_in_threadpool(_dataset_info, beamtime, req.dataset)
    data = {
        "rotation_center": float(req.center),
        "convention": "detector column index (0-based, pixel centres) of the rotation axis",
        "detector_width": info["cols"],
        "slice": req.slice,
        "scan_range": [float(v) for v in req.range] if req.range else None,
        "algorithm": ALGORITHM,
        "projections": info["projections"],
        "projections_used": req.projections_used,
        "filters": req.filters.model_dump() if req.filters else None,
        "projection_folder": str(projections),
        "saved_by": getattr(request.state, "user", None),
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    path = run_folder / CENTER_FILE
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(yaml.safe_dump(data, sort_keys=False))
        tmp.replace(path)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not write {path}: {e}")
    return {"path": str(path), "saved": data}


@router.get("/rotation-center")
async def rotation_page(request: Request):
    return request.app.state.templates.TemplateResponse("rotation.html", {"request": request})
