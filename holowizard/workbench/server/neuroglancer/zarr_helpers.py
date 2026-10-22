"""OME-Zarr helpers: levels and metadata of the written zarrs, reading (small) zarr v2 arrays and the value
statistics for the contrast buttons."""

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numcodecs
import numpy as np

from holowizard.workbench.server.neuroglancer.constants import (
    COMPRESSOR,
    MAX_RANGE_VOXELS,
    STATS_MAX_VOXELS,
    STATS_PERCENTILES,
    TOP_LEVEL_SIZE,
)

# Blosc's own threads are not safe with several Python threads compressing at once.
numcodecs.blosc.use_threads = False
codec = numcodecs.get_codec(COMPRESSOR)


# --- Writing ---
def level_shapes(shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """Shapes of the multiscale levels: halved (rounded up) until the largest side is at most TOP_LEVEL_SIZE."""
    shapes = [tuple(shape)]
    while max(shapes[-1]) > TOP_LEVEL_SIZE:
        shapes.append(tuple((s + 1) // 2 for s in shapes[-1]))
    return shapes


def ome_attrs(name: str, n_levels: int, voxel_nm: Optional[float], window: Tuple[float, float], dtype: str):
    """OME-NGFF 0.4 attributes (multiscales and omero display window) of a converted tomogram."""
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


# --- Reading ---
def read_zarr2_array(path: Path) -> np.ndarray:
    """A whole (small) zarr v2 array, for the value statistics."""
    meta = json.loads((path / ".zarray").read_text())
    shape, chunks = meta["shape"], meta["chunks"]
    if int(np.prod(shape)) > MAX_RANGE_VOXELS:
        raise ValueError("array too large")
    dtype, sep = np.dtype(meta["dtype"]), meta.get("dimension_separator") or "."
    compressor = numcodecs.get_codec(meta["compressor"]) if meta.get("compressor") else None
    filters = [numcodecs.get_codec(f) for f in meta.get("filters") or []]
    fill = meta.get("fill_value")
    fill = 0.0 if fill is None else float(fill)  # also "NaN", "Infinity"
    out = np.full(shape, fill, dtype=np.float64)
    for idx in np.ndindex(*[math.ceil(s / c) for s, c in zip(shape, chunks)]):
        f = path / sep.join(map(str, idx))
        if not f.is_file():
            continue
        buf = f.read_bytes()
        if compressor is not None:
            buf = compressor.decode(buf)
        for flt in reversed(filters):
            buf = flt.decode(buf)
        chunk = np.frombuffer(buf, dtype=dtype).reshape(chunks, order=meta.get("order", "C"))
        region = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, shape))
        out[region] = chunk[tuple(slice(0, r.stop - r.start) for r in region)]
    return out


# --- Contrast ranges (Min-Max, 1-99 %, 5-95 %) over the whole volume ---
def value_stats(folder: Path) -> Dict[str, Any]:
    """Ranges of the stored values, from the finest multiscale level with at most STATS_MAX_VOXELS voxels."""
    attrs = json.loads((folder / ".zattrs").read_text()) if (folder / ".zattrs").is_file() else {}
    multiscales = attrs.get("multiscales") or (attrs.get("ome") or {}).get("multiscales")
    paths = [d["path"] for d in multiscales[0]["datasets"]] if multiscales else [""]
    shapes = [json.loads((folder / p / ".zarray").read_text())["shape"] for p in paths]
    small = [i for i, shape in enumerate(shapes) if int(np.prod(shape)) <= STATS_MAX_VOXELS]
    level = small[0] if small else len(paths) - 1
    values = read_zarr2_array(folder / paths[level]).ravel()
    no_data = (attrs.get("holowizard") or {}).get("no_data_value")
    finite = np.isfinite(values)
    values = values[finite & (values != no_data)] if no_data is not None else values[finite]
    if not values.size:
        raise ValueError("no finite values")
    keys = list(STATS_PERCENTILES)
    q = np.percentile(values, [p for k in keys for p in STATS_PERCENTILES[k]])
    ranges = {k: [float(q[2 * i]), float(q[2 * i + 1])] for i, k in enumerate(keys)}
    return {"ranges": ranges, "level": paths[level], "shape": shapes[level]}


@lru_cache(maxsize=256)
def cached_value_stats(folder: str, meta_mtime_ns: int) -> Dict[str, Any]:
    """``value_stats``, kept until the entry's meta data file changes (``meta_mtime_ns`` is only part of the key,
    it changes when the zarr is converted again)."""
    return value_stats(Path(folder))
