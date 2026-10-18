"""Overview pages (overview.html) of the beamtime's ``nano*`` data sets.

Two overviews share the tile grid and the image browser:

* ``raw``: the scan directories in ``path_raw``; each tile shows the first ref_ image and the first,
  middle and last img_ image.
* ``phase``: the phase retrieval results in ``path_processed/<scan>/<save prefix>/phase_retrieval``,
  one tile per reconstruction run; each tile shows the first, middle and last image.
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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tifffile
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from PIL import Image

DIR_PREFIX = "nano"
TIFF_SUFFIXES = (".tif", ".tiff")
MAX_IMAGE_SIZE = 4096
PHASE_SUBDIR = "phase_retrieval"
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
    datasets: Callable[[Path], List[str]]  # root -> data set ids, e.g. "nano123" or "nano123/holopipe"
    folder: Callable[[Path, str], Path]  # root, data set id -> folder with the TIFFs
    summarize: Callable[[List[str]], Tuple[str, List[Dict[str, Optional[str]]]]]  # files -> (meta, thumbnails)
    valid_id: Callable[[str], bool]
    unit: str  # what one tile is, for the counter ("3 / 5 runs")
    empty_text: str  # shown when there are no data sets yet
    rotate: bool = True  # detector images are shown rotated by 90 degrees, like the test image preview


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
        valid_id=lambda ds: _safe_parts(ds, 1),
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


def _dataset_summary(source: OverviewSource, root: Path, dataset: str, with_files: bool = False) -> Optional[Dict]:
    folder = source.folder(root, dataset)
    files = _cached_by_mtime(folder, _list_tiffs)
    if files is None:
        return None
    meta, thumbs = source.summarize(files)
    summary = {"name": dataset, "mtime": folder.stat().st_mtime, "meta": meta, "thumbs": thumbs, "count": len(files)}
    if with_files:
        summary["files"] = files
    return summary


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
    return {"path": str(root), "dirs": await run_in_threadpool(list_datasets, src, root)}


@router.get("/api/overview/{source}/files")
async def overview_files(source: str, request: Request, dataset: str):
    src = _source(source)
    _dataset_folder(request, src, dataset)
    summary = await run_in_threadpool(_dataset_summary, src, src.root(request.app.state.beamtime), dataset, True)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Data set not readable: {dataset}")
    return summary


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
        },
    )


@router.get("/overview")
async def overview_page(request: Request):
    return _overview_page(request, "raw")


@router.get("/phase-overview")
async def phase_overview_page(request: Request):
    return _overview_page(request, "phase")


@router.get("/tomo-overview")
async def tomo_overview_page(request: Request):
    return _overview_page(request, "tomo")


@router.get("/summary-overview")
async def summary_overview_page(request: Request):
    return _overview_page(request, "summary")
