"""Conversion of a tomogram (TIFF slices) into a multiscale OME-Zarr in the cache folder."""

import json
import math
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tifffile

from holowizard.pipe.server.neuroglancer.array_helpers import cylinder_outside, downsample2
from holowizard.pipe.server.neuroglancer.cache_helpers import (
    cache_dir,
    folder_size,
    now,
    tomogram_cache_id,
    write_json,
)
from holowizard.pipe.server.neuroglancer.constants import (
    CHUNK,
    COMPRESSOR,
    DEFAULT_MASK_RADIUS,
    DISPLAY_PERCENTILES,
    NO_DATA_FLOAT32,
    NO_DATA_UINT8,
    READ_THREADS,
    SAMPLE_SLICES,
    UINT8_PERCENTILES,
    WRITE_THREADS,
)
from holowizard.pipe.server.neuroglancer.zarr_helpers import codec, level_shapes, ome_attrs


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
        self.cache_id = tomogram_cache_id(dataset)
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
                self.outside, radius = cylinder_outside(shape2d, self.mask_radius)
            display, (lo8, hi8) = self._window(shape2d)
            if self.mask:  # 8 bit: data in 1..255, 0 is "no data"; stored s = 255 (v - a) / (b - a)
                lo8 = lo8 - (hi8 - lo8) / 254
            self.scale8 = (lo8, hi8)
            self.no_data = (NO_DATA_UINT8 if self.dtype == "uint8" else NO_DATA_FLOAT32) if self.mask else None
            self.fill = NO_DATA_FLOAT32 if self.mask and self.dtype == "float32" else 0
            shape = (self.total, *shape2d)
            levels = level_shapes(shape)
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
                "created_at": now(),
            }
            self.root.mkdir(parents=True, exist_ok=True)
            if self.output.exists():
                shutil.rmtree(self.output)  # an older conversion of the same tomogram
            write_json(self.root / f"{self.cache_id}.json", self.meta)
            self._write_metadata(levels)
            self.status, self.message = "converting", "Converting…"
            self._convert(levels)
            # The meta data file is written before the job reports the end, so that the listing never shows
            # a finished job whose file still says "converting".
            if self.cancelled:
                self.meta["status"] = "cancelled"
            else:
                self.message = "Finishing…"  # summing up the size of all chunk files takes a while on GPFS
                self.meta.update(status="done", unreadable_slices=self.bad, size_bytes=folder_size(self.output))
            self.meta["finished_at"] = now()
            write_json(self.root / f"{self.cache_id}.json", self.meta)
            self.finished = time.time()
            self.status, self.message = ("cancelled", "Cancelled") if self.cancelled else ("done", "Done")
        except Exception as e:
            if self.meta:
                self.meta["status"] = f"error: {e}"
                try:
                    write_json(self.root / f"{self.cache_id}.json", self.meta)
                except OSError:
                    pass
            self.status, self.message = "error", str(e)

    def _write_metadata(self, levels):
        self.output.mkdir(parents=True)
        (self.output / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        attrs = ome_attrs(self.dataset, len(levels), self.voxel_nm, tuple(self.meta["value_range"]), self.dtype)
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
            path.write_bytes(codec.encode(np.ascontiguousarray(block)))

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
                    push(writer, level + 1, downsample2(part, self.mask), False)
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
