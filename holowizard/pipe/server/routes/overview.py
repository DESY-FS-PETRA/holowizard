"""Overview of the raw ``nano*`` scan directories of a beamtime.

Serves a tile grid (one tile per directory with thumbnails of the first ref_ image
and the first, middle and last img_ image) and a per-directory image browser.
Directory listings are cached per directory mtime, so polling stays cheap while
new directories and images are written.
"""

import io
import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tifffile
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from PIL import Image

DIR_PREFIX = "nano"
TIFF_SUFFIXES = (".tif", ".tiff")
IMAGE_PREFIXES = ("ref_", "img_")
MAX_IMAGE_SIZE = 4096

_listing_cache: Dict[str, Tuple[int, Dict[str, Any]]] = {}
_listing_lock = threading.Lock()


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _list_tiffs(path: Path) -> List[str]:
    with os.scandir(path) as it:
        names = [e.name for e in it if e.name.lower().endswith(TIFF_SUFFIXES) and e.is_file()]
    return sorted(names, key=_natural_key)


def _summarize_dir(path: Path, mtime_ns: int) -> Dict[str, Any]:
    files = _list_tiffs(path)
    imgs = [f for f in files if f.startswith("img_")]
    refs = [f for f in files if f.startswith("ref_")]
    thumbs = [
        {"label": "ref first", "file": refs[0] if refs else None},
        {"label": "img first", "file": imgs[0] if imgs else None},
        {"label": "img middle", "file": imgs[len(imgs) // 2] if imgs else None},
        {"label": "img last", "file": imgs[-1] if imgs else None},
    ]
    return {
        "name": path.name,
        "mtime": mtime_ns / 1e9,
        "n_img": len(imgs),
        "n_ref": len(refs),
        "n_files": len(files),
        "thumbs": thumbs,
        "files": files,
    }


def _dir_summary(path: Path) -> Optional[Dict[str, Any]]:
    """Summary of one directory, re-listed only when the directory mtime changed."""
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return None
    key = str(path)
    with _listing_lock:
        cached = _listing_cache.get(key)
    if cached and cached[0] == mtime_ns:
        return cached[1]
    try:
        summary = _summarize_dir(path, mtime_ns)
    except OSError:
        return None
    with _listing_lock:
        _listing_cache[key] = (mtime_ns, summary)
    return summary


def list_scan_dirs(raw: Path) -> List[Dict[str, Any]]:
    with os.scandir(raw) as it:
        dirs = [Path(e.path) for e in it if e.name.startswith(DIR_PREFIX) and e.is_dir()]
    summaries = (_dir_summary(d) for d in sorted(dirs, key=lambda p: _natural_key(p.name)))
    return [{k: v for k, v in s.items() if k != "files"} for s in summaries if s is not None]


def _downsample(arr: np.ndarray, size: int) -> Tuple[np.ndarray, int]:
    factor = max(1, int(np.ceil(max(arr.shape) / size)))
    if factor == 1:
        return arr.astype(np.float32), 1
    h, w = (arr.shape[0] // factor) * factor, (arr.shape[1] // factor) * factor
    arr = arr[:h, :w].astype(np.float32)
    return arr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3)), factor


def _render_png(path: str, mtime_ns: int, size: int) -> Tuple[bytes, int, float, float]:
    """Downsampled, percentile-scaled 8 bit PNG of the first page of a TIFF.

    ``mtime_ns`` is only part of the cache key, so rewritten files are re-rendered.
    """
    with tifffile.TiffFile(path) as tif:
        arr = tif.pages[0].asarray()
    small, factor = _downsample(arr, size)
    # Same orientation as the image preview on the parameter tuning page.
    small = np.rot90(small)
    lo, hi = np.percentile(small, [1, 99])
    scaled = np.clip((small - lo) / max(hi - lo, 1e-6), 0, 1)
    buf = io.BytesIO()
    Image.fromarray((scaled * 255).astype(np.uint8)).save(buf, format="PNG", compress_level=1)
    return buf.getvalue(), factor, float(lo), float(hi)


# Thumbnails are small and requested on every page load, popup images are large.
_render_thumb = lru_cache(maxsize=4096)(_render_png)
_render_large = lru_cache(maxsize=64)(_render_png)
THUMB_MAX_SIZE = 256


router = APIRouter(tags=["overview"])


def _scan_dir(raw: Path, name: str) -> Path:
    if not name.startswith(DIR_PREFIX) or name != Path(name).name:
        raise HTTPException(status_code=400, detail=f"Invalid directory name: {name!r}")
    path = raw / name
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {name}")
    return path


@router.get("/api/overview/dirs")
async def overview_dirs(request: Request):
    raw = Path(request.app.state.beamtime.path_raw)
    if not raw.is_dir():
        raise HTTPException(status_code=404, detail=f"Raw folder not found: {raw}")
    return {"path": str(raw), "dirs": await run_in_threadpool(list_scan_dirs, raw)}


@router.get("/api/overview/dirs/{name}")
async def overview_dir(name: str, request: Request):
    summary = await run_in_threadpool(_dir_summary, _scan_dir(Path(request.app.state.beamtime.path_raw), name))
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Directory not readable: {name}")
    return summary


@router.get("/api/overview/image/{name}/{file}")
async def overview_image(name: str, file: str, request: Request, size: int = Query(256, ge=32, le=MAX_IMAGE_SIZE)):
    if file != Path(file).name or not file.lower().endswith(TIFF_SUFFIXES):
        raise HTTPException(status_code=400, detail=f"Invalid file name: {file!r}")
    path = _scan_dir(Path(request.app.state.beamtime.path_raw), name) / file
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        raise HTTPException(status_code=404, detail=f"Image not found: {file}")
    try:
        png, factor, lo, hi = await run_in_threadpool(
            _render_thumb if size <= THUMB_MAX_SIZE else _render_large, str(path), mtime_ns, size
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


@router.get("/overview")
async def overview_page(request: Request):
    return request.app.state.templates.TemplateResponse("overview.html", {"request": request, "prefix": DIR_PREFIX})
