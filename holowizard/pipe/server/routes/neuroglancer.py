"""Neuroglancer page (neuroglancer.html): tomograms as OME-Zarr in the Neuroglancer viewer.

Tomograms (the TIFF slices in ``path_processed/<scan>/recon_dir/<version>``, as in the scan overview) are
converted to a multiscale OME-Zarr (zarr v2, chunks of 64³ voxels, blosc-zstd; every level is downsampled by 2
until the largest side is at most 128 voxels). Users can also upload a zarr (zipped). Everything is stored in
the cache folder ``<beamtime log_path>/neuroglancer`` (``scratch_cc/holopipe/neuroglancer``):

* ``<id>.zarr``: the data, served to the viewer under ``/neuroglancer/data/<id>/``
* ``<id>.json``: what it is (source tomogram or upload, user, time, comment, value range for the display)

Optionally, voxels outside the reconstruction cylinder (the circle inscribed in the slices) are masked: they get
a "no data" value (0 in 8 bit zarrs, whose data then use 1..255; the most negative float32 in 32 bit zarrs),
which is also the zarr fill value, so chunks outside the cylinder are not stored. The viewer's shader draws
them transparent and they are left out of the downsampling and the value statistics.

The Neuroglancer client is the prebuilt one shipped with the ``neuroglancer`` Python package (only its static
files are used, the package is not imported), so the viewer works without internet access. That build is the
client of Neuroglancer's Python integration: it is loaded from ``/neuroglancer/v/<token>/`` and talks to a
Python server next to it (a long-lived ``events`` stream and ``state``/``action`` posts). Those are answered
inside the browser by a small script added to its index.html, so no connection stays open (browsers allow only
about six per server, shared with all other pages). The page sets the viewer state itself (URL hash and
``window.viewer`` of the same-origin iframe).
"""

import datetime
import importlib.util
import json
import math
import os
import re
import shutil
import threading
import time
import uuid
import zipfile
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numcodecs
import numpy as np
import tifffile
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from holowizard.pipe.server.routes.overview import (
    SOURCES,
    _cached_by_mtime,
    _list_tiffs,
    add_pixel_sizes,
    list_datasets,
)

CACHE_SUBDIR = "neuroglancer"  # in beamtime.log_path
CHUNK = 64  # voxels per chunk side
TOP_LEVEL_SIZE = 128  # downsample until the largest side is at most this
COMPRESSOR = {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 1, "blocksize": 0}
DTYPES = ("uint8", "float32")
# Display range of the converted data: percentiles of a sample of slices. 8 bit copies map a wider range to
# 0..255, so that the contrast can still be changed in the viewer.
DISPLAY_PERCENTILES = (1, 99)
UINT8_PERCENTILES = (0.1, 99.9)
SAMPLE_SLICES = 24
READ_THREADS = 32
WRITE_THREADS = 16
MAX_RANGE_VOXELS = 64 * 1024**2  # largest zarr level read to find the display range of an upload
NO_DATA_FLOAT32 = float(np.finfo(np.float32).min)  # masked voxels in 32 bit zarrs
NO_DATA_UINT8 = 0  # masked voxels in 8 bit zarrs
DEFAULT_MASK_RADIUS = 0.97  # of the inscribed circle; FBP leaves a bright rim at the edge of the cylinder
STATS_MAX_VOXELS = 256**3  # contrast buttons: statistics of the finest level up to this size
STATS_PERCENTILES = {"minmax": (0, 100), "1-99": (1, 99), "5-95": (5, 95)}
ID_PATTERN = re.compile(r"[A-Za-z0-9_.+-]{1,200}")
UPLOAD_CHUNK = 8 * 1024**2

# Blosc's own threads are not safe with several Python threads compressing at once.
numcodecs.blosc.use_threads = False
_codec = numcodecs.get_codec(COMPRESSOR)

router = APIRouter(tags=["neuroglancer"])
_jobs: Dict[str, "ZarrJob"] = {}  # job id -> job
_cache_jobs: Dict[str, str] = {}  # cache id -> latest job id
_user_jobs: Dict[str, str] = {}
_jobs_lock = threading.Lock()


def client_dir() -> Optional[Path]:
    """Folder of the prebuilt Neuroglancer client in the installed ``neuroglancer`` package, or None."""
    spec = importlib.util.find_spec("neuroglancer")  # finds the package without importing it
    if spec is None or not spec.submodule_search_locations:
        return None
    folder = Path(list(spec.submodule_search_locations)[0]) / "static" / "client"
    return folder if (folder / "index.html").is_file() else None


def cache_dir(beamtime) -> Path:
    return Path(beamtime.log_path) / CACHE_SUBDIR


def _user(request: Request) -> str:
    return getattr(request.state, "user", None) or "anonymous"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", text).strip("._") or "data"


def _tomogram_cache_id(dataset: str) -> str:
    return "tomo__" + "__".join(_safe_name(p) for p in dataset.split("/"))


def _write_json(path: Path, data: Dict[str, Any]):
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _read_meta(root: Path, cache_id: str) -> Optional[Dict[str, Any]]:
    try:
        meta = json.loads((root / f"{cache_id}.json").read_text())
    except (OSError, ValueError):
        return None
    meta["id"] = cache_id
    return meta


def _entry_path(beamtime, cache_id: str) -> Path:
    if not ID_PATTERN.fullmatch(cache_id) or cache_id in (".", ".."):
        raise HTTPException(status_code=400, detail=f"Invalid id: {cache_id!r}")
    return cache_dir(beamtime) / f"{cache_id}.zarr"


def _entry_state(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Meta data plus the live conversion state; conversions cut off by a server restart are "interrupted"."""
    job = _jobs.get(_cache_jobs.get(meta["id"], ""))
    meta = dict(meta)
    if job is not None and job.status in ("reading", "converting"):
        meta["job"] = job.state()
    elif meta.get("status") == "converting":
        # Only a conversion this server process does not know any more (restart) is interrupted.
        meta["status"] = "interrupted" if job is None else job.status
    return meta


# --- OME-Zarr writing ---
def _level_shapes(shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    shapes = [tuple(shape)]
    while max(shapes[-1]) > TOP_LEVEL_SIZE:
        shapes.append(tuple((s + 1) // 2 for s in shapes[-1]))
    return shapes


def _downsample2(a: np.ndarray, masked: bool = False) -> np.ndarray:
    """Mean over 2x2x2 blocks; odd sides are padded with their last plane (so sizes are halved, rounded up).

    With ``masked``, NaN marks masked voxels: they are left out of the mean, and blocks of only masked voxels
    stay NaN."""
    pad = [(0, s % 2) for s in a.shape]
    if any(p for _, p in pad):
        a = np.pad(a, pad, mode="edge")
    z, y, x = a.shape
    shape = (z // 2, 2, y // 2, 2, x // 2, 2)
    if not masked:
        return a.reshape(shape).mean(axis=(1, 3, 5), dtype=np.float32)
    valid = ~np.isnan(a)
    total = np.where(valid, a, 0).reshape(shape).sum(axis=(1, 3, 5), dtype=np.float32)
    count = valid.reshape(shape).sum(axis=(1, 3, 5))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (total / count).astype(np.float32)  # 0 / 0 = NaN for fully masked blocks


def _cylinder_outside(shape2d: Tuple[int, int], fraction: float = 1.0) -> Tuple[np.ndarray, float]:
    """Pixels of a slice outside the reconstruction cylinder (the inscribed circle, its radius scaled by
    ``fraction`` to also cut off the artefacts at its edge), and the radius."""
    h, w = shape2d
    radius = fraction * min(h, w) / 2
    yy, xx = np.ogrid[:h, :w]
    return (yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2 > radius**2, radius


def _ome_attrs(name: str, n_levels: int, voxel_nm: Optional[float], window: Tuple[float, float], dtype: str):
    if voxel_nm:
        unit, voxel = "nanometer", float(voxel_nm)
    else:
        unit, voxel = None, 1.0
    axes = [{"name": a, "type": "space", **({"unit": unit} if unit else {})} for a in ("z", "y", "x")]
    lo, hi = window  # display range, in stored values
    vmin, vmax = (0.0, 255.0) if dtype == "uint8" else window
    return {
        "multiscales": [
            {
                "version": "0.4",
                "name": name,
                "axes": axes,
                "datasets": [
                    {
                        "path": str(level),
                        "coordinateTransformations": [{"type": "scale", "scale": [voxel * 2**level] * 3}],
                    }
                    for level in range(n_levels)
                ],
                "type": "mean",
            }
        ],
        "omero": {
            "channels": [
                {
                    "label": name,
                    "color": "FFFFFF",
                    "active": True,
                    "window": {"start": lo, "end": hi, "min": vmin, "max": vmax},
                }
            ],
            "rdefs": {"model": "greyscale"},
        },
    }


class ZarrJob:
    """Converts the TIFF slices of a tomogram (in a thread) into a multiscale OME-Zarr."""

    def __init__(
        self,
        beamtime,
        user: str,
        dataset: str,
        folder: Path,
        files: List[str],
        dtype: str,
        voxel_nm,
        note: str,
        mask: bool = True,
        mask_radius: float = DEFAULT_MASK_RADIUS,
    ):
        self.id = uuid.uuid4().hex
        self.user, self.dataset, self.folder, self.files, self.dtype = user, dataset, folder, files, dtype
        self.mask, self.mask_radius = mask, mask_radius
        self.outside: Optional[np.ndarray] = None  # pixels outside the cylinder, if masked
        self.voxel_nm, self.voxel_note = voxel_nm, note
        self.cache_id = _tomogram_cache_id(dataset)
        self.root = cache_dir(beamtime)
        self.output = self.root / f"{self.cache_id}.zarr"
        self.total, self.done, self.bad = len(files), 0, 0
        self.status, self.message = "reading", "Reading a sample of slices…"
        self.started, self.finished = time.time(), None
        self.cancelled = False
        self.meta: Dict[str, Any] = {}
        threading.Thread(target=self._run, name=f"zarr-{self.id[:8]}", daemon=True).start()

    def _read(self, name: str, shape) -> Optional[np.ndarray]:
        try:
            img = tifffile.imread(str(self.folder / name))
        except Exception:
            return None
        if img.ndim != 2 or img.shape != shape:
            return None
        return np.nan_to_num(img.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    def _window(self, shape) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        idx = np.unique(np.linspace(0, self.total - 1, min(SAMPLE_SLICES, self.total)).astype(int))
        with ThreadPoolExecutor(READ_THREADS) as ex:
            sample = [s[::4, ::4] for s in ex.map(lambda i: self._read(self.files[i], shape), idx) if s is not None]
        if not sample:
            raise ValueError("No readable slices")
        values = np.stack(sample)
        values = values[:, ~self.outside[::4, ::4]] if self.outside is not None else values.ravel()
        lo, hi = np.percentile(values, DISPLAY_PERCENTILES)
        lo8, hi8 = np.percentile(values, UINT8_PERCENTILES)
        fix = lambda a, b: (float(a), float(b) if b > a else float(a) + 1.0)  # noqa: E731
        return fix(lo, hi), fix(lo8, hi8)

    def _run(self):
        try:
            with tifffile.TiffFile(str(self.folder / self.files[0])) as tif:
                shape2d = tuple(tif.pages[0].shape[:2])
            radius = None
            if self.mask:
                self.outside, radius = _cylinder_outside(shape2d, self.mask_radius)
            display, (lo8, hi8) = self._window(shape2d)
            if self.mask:  # 8 bit: data in 1..255, 0 is "no data"; stored s = 255 (v - a) / (b - a)
                lo8 = lo8 - (hi8 - lo8) / 254
            self.scale8 = (lo8, hi8)
            self.no_data = (NO_DATA_UINT8 if self.dtype == "uint8" else NO_DATA_FLOAT32) if self.mask else None
            self.fill = NO_DATA_FLOAT32 if self.mask and self.dtype == "float32" else 0
            shape = (self.total, *shape2d)
            levels = _level_shapes(shape)
            if self.dtype == "uint8":  # display range in the stored 0..255 values
                window = tuple(255 * (v - lo8) / (hi8 - lo8) for v in display)
            else:
                window = display
            self.meta = {
                "kind": "tomogram",
                "name": self.dataset,
                "source": self.dataset,
                "source_folder": str(self.folder),
                "slices": self.total,
                "status": "converting",
                "format": "zarr2",
                "dtype": self.dtype,
                "shape": list(shape),
                "levels": [list(s) for s in levels],
                "chunk": CHUNK,
                "voxel_size_nm": self.voxel_nm,
                "voxel_size_note": self.voxel_note,
                "value_range": [round(float(v), 6) for v in window],
                "uint8_maps": [lo8, hi8] if self.dtype == "uint8" else None,  # original values of stored 0 and 255
                "mask": {"type": "cylinder", "radius_px": radius, "radius_fraction": self.mask_radius}
                if self.mask
                else None,
                "no_data_value": self.no_data,
                "created_by": self.user,
                "created_at": _now(),
            }
            self.root.mkdir(parents=True, exist_ok=True)
            if self.output.exists():
                shutil.rmtree(self.output)  # an older conversion of the same tomogram
            _write_json(self.root / f"{self.cache_id}.json", self.meta)
            self._write_metadata(levels)
            self.status, self.message = "converting", "Converting…"
            self._convert(levels)
            # The meta data file is written before the job reports the end, so that the listing never shows
            # a finished job whose file still says "converting".
            if self.cancelled:
                self.meta["status"] = "cancelled"
            else:
                self.message = "Finishing…"  # summing up the size of all chunk files takes a while on GPFS
                self.meta.update(status="done", unreadable_slices=self.bad, size_bytes=_folder_size(self.output))
            self.meta["finished_at"] = _now()
            _write_json(self.root / f"{self.cache_id}.json", self.meta)
            self.finished = time.time()
            self.status, self.message = ("cancelled", "Cancelled") if self.cancelled else ("done", "Done")
        except Exception as e:
            if self.meta:
                self.meta["status"] = f"error: {e}"
                try:
                    _write_json(self.root / f"{self.cache_id}.json", self.meta)
                except OSError:
                    pass
            self.status, self.message = "error", str(e)

    def _write_metadata(self, levels):
        self.output.mkdir(parents=True)
        (self.output / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        attrs = _ome_attrs(self.dataset, len(levels), self.voxel_nm, tuple(self.meta["value_range"]), self.dtype)
        attrs["holowizard"] = {
            k: self.meta[k]
            for k in ("source", "source_folder", "dtype", "uint8_maps", "mask", "no_data_value", "created_by")
        }
        (self.output / ".zattrs").write_text(json.dumps(attrs, indent=2))
        for level, shape in enumerate(levels):
            (self.output / str(level)).mkdir()
            zarray = {
                "zarr_format": 2,
                "shape": list(shape),
                "chunks": [CHUNK] * 3,
                "dtype": np.dtype(self.dtype).str,
                "compressor": COMPRESSOR,
                "fill_value": self.fill,
                "order": "C",
                "filters": None,
                "dimension_separator": "/",
            }
            (self.output / str(level) / ".zarray").write_text(json.dumps(zarray, indent=2))

    def _cast(self, block: np.ndarray) -> np.ndarray:
        """Stored values; masked voxels (NaN) become the "no data" value."""
        if self.dtype == "uint8":
            out = np.rint(np.clip(block, 1 if self.mask else 0, 255))
            if self.mask:
                out[np.isnan(block)] = NO_DATA_UINT8
            return out.astype(np.uint8)
        out = block.astype(np.float32, copy=True)
        if self.mask:
            out[np.isnan(block)] = NO_DATA_FLOAT32
        return out

    def _write_slab(self, writer: ThreadPoolExecutor, level: int, z0: int, slab: np.ndarray):
        """One layer of chunks (z0 is a multiple of CHUNK); chunks with only the fill value are left out."""
        zc = z0 // CHUNK

        def write(yx):
            yc, xc = yx
            block = self._cast(slab[:, yc * CHUNK : (yc + 1) * CHUNK, xc * CHUNK : (xc + 1) * CHUNK])
            if (block == self.fill).all():
                return
            if block.shape != (CHUNK,) * 3:  # edge chunks are stored in full size
                block = np.pad(block, [(0, CHUNK - s) for s in block.shape], constant_values=self.fill)
            path = self.output / str(level) / str(zc) / str(yc) / str(xc)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_codec.encode(np.ascontiguousarray(block)))

        ny, nx = math.ceil(slab.shape[1] / CHUNK), math.ceil(slab.shape[2] / CHUNK)
        list(writer.map(write, [(y, x) for y in range(ny) for x in range(nx)]))

    def _convert(self, levels):
        shape2d = levels[0][1:]
        buffers: List[Optional[np.ndarray]] = [None] * len(levels)
        written = [0] * len(levels)

        def push(writer, level: int, slab: Optional[np.ndarray], final: bool):
            # Collect planes per level and write whole chunk layers; each written layer, downsampled, feeds the
            # next level. ``final`` flushes the rest.
            if slab is not None and slab.shape[0]:
                buffers[level] = slab if buffers[level] is None else np.concatenate([buffers[level], slab])
            buf = buffers[level]
            while buf is not None and (buf.shape[0] >= CHUNK or (final and buf.shape[0])):
                part, buf = buf[:CHUNK], buf[CHUNK:]
                self._write_slab(writer, level, written[level], part)
                written[level] += part.shape[0]
                if level + 1 < len(levels):
                    push(writer, level + 1, _downsample2(part, self.mask), False)
            buffers[level] = buf
            if final and level + 1 < len(levels):
                push(writer, level + 1, None, True)

        def read_slab(first: int) -> np.ndarray:
            names = self.files[first : first + CHUNK]
            slab = np.empty((len(names), *shape2d), np.float32)  # filled in place, no copies of the slices

            def read_into(i: int) -> bool:
                plane = self._read(names[i], shape2d)
                ok = plane is not None
                if not ok:
                    slab[i] = 0
                elif self.dtype == "uint8":
                    lo, hi = self.scale8
                    np.multiply(plane - lo, 255.0 / (hi - lo), out=slab[i])
                else:
                    slab[i] = plane
                if self.outside is not None:
                    slab[i][self.outside] = np.nan
                return ok

            with ThreadPoolExecutor(READ_THREADS) as ex:
                self.bad += sum(not ok for ok in ex.map(read_into, range(len(names))))
            return slab

        starts = list(range(0, self.total, CHUNK))
        with ThreadPoolExecutor(1) as reader, ThreadPoolExecutor(WRITE_THREADS) as writer:
            pending = reader.submit(read_slab, starts[0])
            for i in range(len(starts)):
                slab = pending.result()
                if i + 1 < len(starts):
                    pending = reader.submit(read_slab, starts[i + 1])  # read ahead while writing
                if self.cancelled:
                    return
                push(writer, 0, slab, False)
                self.done = min(self.total, starts[i] + CHUNK)
            push(writer, 0, None, True)

    def state(self) -> Dict[str, Any]:
        elapsed = (self.finished or time.time()) - self.started
        rate = self.done / elapsed if self.done and self.status == "converting" else None
        return {
            "id": self.id,
            "cache_id": self.cache_id,
            "dataset": self.dataset,
            "dtype": self.dtype,
            "status": self.status,
            "message": self.message,
            "done": self.done,
            "total": self.total,
            "elapsed": round(elapsed),
            "eta": round((self.total - self.done) / rate) if rate else None,
        }


def _folder_size(path: Path) -> int:
    total = 0
    for folder, _, files in os.walk(path):
        for f in files:
            try:
                total += os.stat(os.path.join(folder, f)).st_size
            except OSError:
                pass
    return total


# --- Uploaded zarrs ---
def _zarr_root(names: List[str]) -> Optional[str]:
    """Prefix (folder inside the zip, '' or 'x.zarr/') of the shallowest zarr group or array in a zip listing."""
    markers = (".zgroup", ".zarray", "zarr.json")
    roots = [n[: -len(Path(n).name)] for n in names if Path(n).name in markers]
    return min(roots, key=lambda r: (r.count("/"), r)) if roots else None


def _read_zarr2_array(path: Path) -> np.ndarray:
    """A whole (small) zarr v2 array, for the display range."""
    meta = json.loads((path / ".zarray").read_text())
    shape, chunks = meta["shape"], meta["chunks"]
    if int(np.prod(shape)) > MAX_RANGE_VOXELS:
        raise ValueError("array too large")
    dtype, sep = np.dtype(meta["dtype"]), meta.get("dimension_separator") or "."
    codec = numcodecs.get_codec(meta["compressor"]) if meta.get("compressor") else None
    filters = [numcodecs.get_codec(f) for f in meta.get("filters") or []]
    fill = meta.get("fill_value")
    fill = 0.0 if fill is None else float(fill)  # also "NaN", "Infinity"
    out = np.full(shape, fill, dtype=np.float64)
    for idx in np.ndindex(*[math.ceil(s / c) for s, c in zip(shape, chunks)]):
        f = path / sep.join(map(str, idx))
        if not f.is_file():
            continue
        buf = f.read_bytes()
        if codec is not None:
            buf = codec.decode(buf)
        for flt in reversed(filters):
            buf = flt.decode(buf)
        chunk = np.frombuffer(buf, dtype=dtype).reshape(chunks, order=meta.get("order", "C"))
        region = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, shape))
        out[region] = chunk[tuple(slice(0, r.stop - r.start) for r in region)]
    return out


def _upload_info(path: Path) -> Dict[str, Any]:
    """Format, shape, voxel size and display range of an uploaded zarr (as far as they can be read)."""
    info: Dict[str, Any] = {"format": "zarr3" if (path / "zarr.json").is_file() else "zarr2"}
    if info["format"] == "zarr3":
        return info
    attrs = {}
    if (path / ".zattrs").is_file():
        attrs = json.loads((path / ".zattrs").read_text())
    multiscales = attrs.get("multiscales") or (attrs.get("ome") or {}).get("multiscales")
    if multiscales:
        datasets = multiscales[0]["datasets"]
        paths = [d["path"] for d in datasets]
        units = {a.get("unit") for a in multiscales[0].get("axes", []) if a.get("type") == "space"}
        scale = [t for t in datasets[0].get("coordinateTransformations", []) if t.get("type") == "scale"]
        if scale and units in ({"nanometer"}, {"micrometer"}):
            info["voxel_size_nm"] = scale[0]["scale"][-1] * (1000 if units == {"micrometer"} else 1)
    else:
        paths = [""]
    first = json.loads((path / paths[0] / ".zarray").read_text())
    info["shape"], info["dtype"] = first["shape"], np.dtype(first["dtype"]).name
    holowizard = attrs.get("holowizard") or {}
    for key in ("no_data_value", "mask", "uint8_maps"):  # zarrs converted here and uploaded again
        if holowizard.get(key) is not None:
            info[key] = holowizard[key]
    window = ((attrs.get("omero") or {}).get("channels") or [{}])[0].get("window")
    if window and window.get("end", 0) > window.get("start", 0):
        info["value_range"] = [window["start"], window["end"]]
    else:
        try:
            values = _read_zarr2_array(path / paths[-1])
            values = values[np.isfinite(values) & (values != info.get("no_data_value", np.nan))]
            lo, hi = np.percentile(values, DISPLAY_PERCENTILES)
            info["value_range"] = [float(lo), float(hi) if hi > lo else float(lo) + 1.0]
        except Exception:
            pass  # the viewer's own contrast controls still work
    return info


def _unpack_upload(zip_path: Path, output: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        prefix = _zarr_root(names)
        if prefix is None:
            raise ValueError("No zarr found in the zip (no .zgroup, .zarray or zarr.json)")
        output.mkdir(parents=True)
        for name in names:
            if not name.startswith(prefix):
                continue
            rel = Path(name[len(prefix) :])
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                raise ValueError(f"Invalid path in the zip: {name}")
            target = output / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, UPLOAD_CHUNK)


# --- Download as zip (stored, the chunks are compressed already) ---
class _ZipStream:
    """File-like sink for zipfile that hands the written bytes to a generator."""

    def __init__(self):
        self.parts: List[bytes] = []

    def write(self, b) -> int:
        self.parts.append(bytes(b))
        return len(b)

    def flush(self):
        pass

    def take(self) -> bytes:
        data, self.parts = b"".join(self.parts), []
        return data


def _zip_stream(folder: Path, arc_root: str) -> Iterator[bytes]:
    sink = _ZipStream()
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames.sort()
            for name in sorted(filenames):
                path = Path(dirpath) / name
                zf.write(path, f"{arc_root}/{path.relative_to(folder).as_posix()}")
                yield sink.take()
    yield sink.take()


# --- Routes ---
def _tomograms(beamtime) -> List[Dict[str, Any]]:
    src = SOURCES["tomo"]
    root = src.root(beamtime)
    return list_datasets(src, root) if root.is_dir() else []


def _list_entries(beamtime) -> Dict[str, Any]:
    root = cache_dir(beamtime)
    metas = {}
    if root.is_dir():
        for f in root.glob("*.json"):
            meta = _read_meta(root, f.stem)
            if meta is not None and (root / f"{f.stem}.zarr").is_dir():
                metas[f.stem] = _entry_state(meta)
    with _jobs_lock:  # conversions that have not written their meta data yet
        for cache_id, job_id in _cache_jobs.items():
            job = _jobs[job_id]
            if cache_id not in metas and job.status == "reading":
                metas[cache_id] = {
                    "id": cache_id,
                    "kind": "tomogram",
                    "source": job.dataset,
                    "status": "converting",
                    "job": job.state(),
                }
    tomograms = []
    for t in sorted(_tomograms(beamtime), key=lambda d: d["name"]):
        entry = metas.get(_tomogram_cache_id(t["name"]))
        tomograms.append({"dataset": t["name"], "meta": t["meta"], "count": t["count"], "zarr": entry})
    uploads = sorted(
        (m for m in metas.values() if m.get("kind") == "upload"), key=lambda m: m.get("created_at", ""), reverse=True
    )
    return {"cache_dir": str(root), "tomograms": tomograms, "uploads": uploads}


@router.get("/api/neuroglancer/datasets")
async def neuroglancer_datasets(request: Request):
    return await run_in_threadpool(_list_entries, request.app.state.beamtime)


class ConvertRequest(BaseModel):
    dataset: str
    dtype: str = "uint8"
    mask: bool = True  # mask the voxels outside the reconstruction cylinder
    mask_radius: float = DEFAULT_MASK_RADIUS  # fraction of the inscribed circle


@router.post("/api/neuroglancer/convert")
async def convert(req: ConvertRequest, request: Request):
    beamtime, user = request.app.state.beamtime, _user(request)
    if not 0.5 <= req.mask_radius <= 1.0:
        raise HTTPException(status_code=400, detail="The mask radius must lie between 50 and 100 %")
    if req.dtype not in DTYPES:
        raise HTTPException(status_code=400, detail=f"dtype must be one of {', '.join(DTYPES)}")
    src = SOURCES["tomo"]
    if not src.valid_id(req.dataset):
        raise HTTPException(status_code=400, detail=f"Invalid tomogram: {req.dataset!r}")
    folder = src.folder(src.root(beamtime), req.dataset)
    files = await run_in_threadpool(_cached_by_mtime, folder, _list_tiffs)
    if not files:
        raise HTTPException(status_code=404, detail=f"No slices found in {folder}")
    cache_id = _tomogram_cache_id(req.dataset)
    with _jobs_lock:
        for job_id in (_cache_jobs.get(cache_id), _user_jobs.get(user)):
            job = _jobs.get(job_id or "")
            if job is not None and job.status in ("reading", "converting"):
                what = "This tomogram is" if job.cache_id == cache_id else "You are"
                raise HTTPException(status_code=409, detail=f"{what} already being converted ({job.dataset}).")
    summary = {"name": req.dataset, "count": len(files)}
    await run_in_threadpool(add_pixel_sizes, beamtime, "tomo", [summary])
    job = ZarrJob(
        beamtime,
        user,
        req.dataset,
        folder,
        files,
        req.dtype,
        summary.get("pixel_size_nm"),
        summary.get("pixel_note", ""),
        req.mask,
        req.mask_radius,
    )
    with _jobs_lock:
        _jobs[job.id] = job
        _cache_jobs[cache_id] = job.id
        _user_jobs[user] = job.id
    return job.state()


@router.get("/api/neuroglancer/job/{job_id}")
async def job_state(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Conversion not found (server restarted?)")
    return job.state()


@router.post("/api/neuroglancer/job/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Conversion not found")
    if job.user != _user(request):
        raise HTTPException(status_code=403, detail="Only the user who started a conversion can cancel it")
    job.cancelled = True
    return job.state()


@router.post("/api/neuroglancer/upload")
async def upload(request: Request, filename: str, comment: str = ""):
    """Upload a zipped zarr as the raw request body (streamed to disk, then unpacked into the cache)."""
    beamtime, user = request.app.state.beamtime, _user(request)
    base = filename.rsplit("/", 1)[-1]
    for suffix in (".zip", ".zarr"):
        base = base[: -len(suffix)] if base.lower().endswith(suffix) else base
    name = _safe_name(base)
    root = cache_dir(beamtime)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f".upload_{uuid.uuid4().hex}.zip"
    try:
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                await run_in_threadpool(f.write, chunk)
        if not zipfile.is_zipfile(tmp):
            raise HTTPException(status_code=400, detail="Please upload the zarr folder as a .zip file")
        with _jobs_lock:  # unique id: name, name_2, name_3, ...
            cache_id, n = f"upload__{name}", 1
            while (root / f"{cache_id}.zarr").exists() or (root / f"{cache_id}.json").exists():
                n += 1
                cache_id = f"upload__{name}_{n}"
            (root / f"{cache_id}.json").write_text("{}")  # reserve the id
        output = root / f"{cache_id}.zarr"
        try:
            await run_in_threadpool(_unpack_upload, tmp, output)
            info = await run_in_threadpool(_upload_info, output)
        except (ValueError, OSError, KeyError, zipfile.BadZipFile) as e:
            shutil.rmtree(output, ignore_errors=True)
            (root / f"{cache_id}.json").unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=f"Could not read the zarr: {e}")
        meta = {
            "kind": "upload",
            "name": base,
            "filename": filename,
            "comment": comment.strip()[:200],
            "status": "done",
            "created_by": user,
            "created_at": _now(),
            "size_bytes": await run_in_threadpool(_folder_size, output),
            **info,
        }
        await run_in_threadpool(_write_json, root / f"{cache_id}.json", meta)
        meta["id"] = cache_id
        return meta
    finally:
        tmp.unlink(missing_ok=True)


# --- Contrast ranges (Min-Max, 1-99 %, 5-95 %) over the whole volume ---
_stats_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}


def _value_stats(folder: Path) -> Dict[str, Any]:
    """Ranges of the stored values, from the finest multiscale level with at most STATS_MAX_VOXELS voxels."""
    attrs = json.loads((folder / ".zattrs").read_text()) if (folder / ".zattrs").is_file() else {}
    multiscales = attrs.get("multiscales") or (attrs.get("ome") or {}).get("multiscales")
    paths = [d["path"] for d in multiscales[0]["datasets"]] if multiscales else [""]
    shapes = [json.loads((folder / p / ".zarray").read_text())["shape"] for p in paths]
    small = [i for i, shape in enumerate(shapes) if int(np.prod(shape)) <= STATS_MAX_VOXELS]
    level = small[0] if small else len(paths) - 1
    values = _read_zarr2_array(folder / paths[level]).ravel()
    no_data = (attrs.get("holowizard") or {}).get("no_data_value")
    finite = np.isfinite(values)
    values = values[finite & (values != no_data)] if no_data is not None else values[finite]
    if not values.size:
        raise ValueError("no finite values")
    keys = list(STATS_PERCENTILES)
    q = np.percentile(values, [p for k in keys for p in STATS_PERCENTILES[k]])
    ranges = {k: [float(q[2 * i]), float(q[2 * i + 1])] for i, k in enumerate(keys)}
    return {"ranges": ranges, "level": paths[level], "shape": shapes[level]}


@router.get("/api/neuroglancer/stats/{cache_id}")
async def value_stats(cache_id: str, request: Request):
    folder = _entry_path(request.app.state.beamtime, cache_id)
    meta_file = folder.parent / f"{cache_id}.json"
    try:
        key = (str(folder), meta_file.stat().st_mtime_ns)  # changes when the zarr is converted again
    except OSError:
        raise HTTPException(status_code=404, detail="Not found")
    if key not in _stats_cache:
        if (folder / "zarr.json").is_file():
            raise HTTPException(status_code=422, detail="Not available for zarr v3 data")
        try:
            _stats_cache[key] = await run_in_threadpool(_value_stats, folder)
        except (OSError, ValueError, KeyError, IndexError) as e:
            raise HTTPException(status_code=422, detail=f"Could not compute the value range: {e}")
    return _stats_cache[key]


@router.get("/api/neuroglancer/download/{cache_id}")
async def download(cache_id: str, request: Request):
    folder = _entry_path(request.app.state.beamtime, cache_id)
    meta = _read_meta(folder.parent, cache_id)
    if meta is None or not folder.is_dir():
        raise HTTPException(status_code=404, detail="Not found")
    if _entry_state(meta).get("status") != "done":
        raise HTTPException(status_code=409, detail="The conversion is not finished")
    arc_root = f"{_safe_name(meta.get('name', cache_id).replace('/', '_'))}.zarr"
    return StreamingResponse(
        _zip_stream(folder, arc_root),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{arc_root}.zip"'},
    )


@router.api_route("/neuroglancer/data/{cache_id}/{path:path}", methods=["GET", "HEAD"])
def zarr_file(cache_id: str, path: str, request: Request):
    """Files of a cached zarr, for the viewer; missing chunks are 404 (the viewer shows the fill value)."""
    folder = _entry_path(request.app.state.beamtime, cache_id)
    parts = Path(path).parts
    if not parts or any(p in ("..", ".") or p.startswith("/") for p in parts):
        raise HTTPException(status_code=404, detail="Not found")
    target = folder.joinpath(*parts)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    # Revalidated on every use (ETag), since a tomogram can be converted again under the same id.
    return FileResponse(target, headers={"Cache-Control": "no-cache"})


# --- Neuroglancer client and the server API of its Python integration build ---
CLIENT_TOKEN = re.compile(r"[A-Za-z0-9_-]{1,64}")
# Added to the client's index.html, before its scripts: the client's "Python server" (state updates pushed over
# an EventSource, state changes and actions posted back) is answered in the browser. A real EventSource would
# keep one of the browser's few connections per server open for as long as the viewer is open.
CLIENT_SHIM = """<script>
(() => {
  class SilentEventSource extends EventTarget {
    constructor(url) {
      super();
      this.url = String(url); this.readyState = 1; this.onopen = this.onmessage = this.onerror = null;
      setTimeout(() => { if (this.readyState === 1 && this.onopen) this.onopen(new Event('open')); }, 0);
    }
    close() { this.readyState = 2; }
  }
  window.EventSource = SilentEventSource;
  const realFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    if (/\\/neuroglancer\\/(state|action)\\//.test(url)) {
      let g = null;
      try { g = JSON.parse(init && init.body).g; } catch (e) { /* not a state update */ }
      return Promise.resolve(new Response(JSON.stringify({ g: String(g) }), { headers: { 'Content-Type': 'application/json' } }));
    }
    return realFetch(input, init);
  };
})();
</script>"""


# Patch of the client's main script: in the 4-panel layouts, Neuroglancer shows the scale bar only in the XY panel
# (the XZ and YZ panels get a fixed "off"); they get the viewer's scale bar setting like XY. Matches the minified
# code of the installed version; if a version differs, nothing is replaced and the client works as shipped.
SCALE_BAR_PATTERN = re.compile(
    r"(\w+)=\{\.\.\.(\w+)\((\w+)\),showScaleBar:\3\.showScaleBar\},"
    r"(\w+)=\{\.\.\.\2\(\3\),showScaleBar:new \w+\(!1,!1\)\}"
)
CLIENT_PATCH_VERSION = "1"  # in the script URL, so browsers do not use an unpatched cached copy


@lru_cache(maxsize=2)
def _patched_main_script(path: str, mtime_ns: int) -> bytes:
    """The client's main script with scale bars in all slice panels; ``mtime_ns`` is only part of the cache key."""
    text = Path(path).read_text()
    text = SCALE_BAR_PATTERN.sub(
        lambda m: (
            f"{m[1]}={{...{m[2]}({m[3]}),showScaleBar:{m[3]}.showScaleBar}},"
            f"{m[4]}={{...{m[2]}({m[3]}),showScaleBar:{m[3]}.showScaleBar}}"
        ),
        text,
    )
    return text.encode()


@router.get("/neuroglancer/v/{token}/{file:path}")
def client_file(token: str, file: str):
    folder = client_dir()
    if folder is None or not CLIENT_TOKEN.fullmatch(token):
        raise HTTPException(status_code=404, detail="Not found")
    name = file or "index.html"
    path = folder / name
    if name != Path(name).name or not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    if name == "index.html":
        html = path.read_text().replace("<head>", "<head>" + CLIENT_SHIM, 1)
        html = re.sub(r'(src="main\.[^"]+\.js)"', rf'\1?patch={CLIENT_PATCH_VERSION}"', html)
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})
    if name.startswith("main.") and name.endswith(".js"):
        script = _patched_main_script(str(path), path.stat().st_mtime_ns)
        return Response(script, media_type="text/javascript", headers={"Cache-Control": "max-age=86400"})
    media_type = "application/wasm" if name.endswith(".wasm") else None
    # The scripts have content hashes in their names, so they can be cached.
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "max-age=86400"})


@router.post("/neuroglancer/state/{token}")
async def client_state(request: Request):
    """Fallback for state changes that reach the server (normally answered in the browser); not stored."""
    try:
        generation = (await request.json()).get("g")
    except ValueError:
        generation = None
    return {"g": str(generation)}


@router.post("/neuroglancer/action/{token}")
async def client_action():
    return {}


@router.get("/neuroglancer")
async def neuroglancer_page(request: Request):
    return request.app.state.templates.TemplateResponse(
        "neuroglancer.html", {"request": request, "client_available": client_dir() is not None}
    )
