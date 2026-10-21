"""The entries of the cache folder ``<beamtime log_path>/neuroglancer``: ``<id>.zarr`` and ``<id>.json``."""

import datetime
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from holowizard.pipe.server.neuroglancer.constants import CACHE_SUBDIR, ID_PATTERN
from holowizard.pipe.server.neuroglancer.job_registry import RUNNING, registry
from holowizard.pipe.server.routes.overview import SOURCES, list_datasets


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", text).strip("._") or "data"


def cache_dir(beamtime) -> Path:
    return Path(beamtime.log_path) / CACHE_SUBDIR


def tomogram_cache_id(dataset: str) -> str:
    return "tomo__" + "__".join(safe_name(p) for p in dataset.split("/"))


def write_json(path: Path, data: Dict[str, Any]):
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def read_meta(root: Path, cache_id: str) -> Optional[Dict[str, Any]]:
    try:
        meta = json.loads((root / f"{cache_id}.json").read_text())
    except (OSError, ValueError):
        return None
    meta["id"] = cache_id
    return meta


def folder_size(path: Path) -> int:
    total = 0
    for folder, _, files in os.walk(path):
        for f in files:
            try:
                total += os.stat(os.path.join(folder, f)).st_size
            except OSError:
                pass
    return total


def entry_path(beamtime, cache_id: str) -> Path:
    """The zarr folder of a cache entry; 400 for an invalid id."""
    if not ID_PATTERN.fullmatch(cache_id) or cache_id in (".", ".."):
        raise HTTPException(status_code=400, detail=f"Invalid id: {cache_id!r}")
    return cache_dir(beamtime) / f"{cache_id}.zarr"


def entry_file(folder: Path, path: str) -> Optional[Path]:
    """A file inside a zarr folder (``path`` relative to it, without "." or ".." parts), or None."""
    parts = Path(path).parts
    if not parts or any(p in ("..", ".") or p.startswith("/") for p in parts):
        return None
    target = folder.joinpath(*parts)
    return target if target.is_file() else None


def entry_state(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Meta data plus the live conversion state; conversions cut off by a server restart are "interrupted"."""
    job = registry.latest_for_cache(meta["id"])
    meta = dict(meta)
    if job is not None and job.status in RUNNING:
        meta["job"] = job.state()
    elif meta.get("status") == "converting":
        # Only a conversion this server process does not know any more (restart) is interrupted.
        meta["status"] = "interrupted" if job is None else job.status
    return meta


def tomograms(beamtime) -> List[Dict[str, Any]]:
    """The tomograms of the scan overview (TIFF slices in recon_dir)."""
    src = SOURCES["tomo"]
    root = src.root(beamtime)
    return list_datasets(src, root) if root.is_dir() else []


def list_entries(beamtime) -> Dict[str, Any]:
    """All tomograms, with their converted zarr (if any)."""
    root = cache_dir(beamtime)
    metas = {}
    if root.is_dir():
        for f in root.glob("*.json"):
            meta = read_meta(root, f.stem)
            if meta is not None and (root / f"{f.stem}.zarr").is_dir():
                metas[f.stem] = entry_state(meta)
    for cache_id, job in registry.latest_per_cache().items():  # conversions without meta data yet
        if cache_id not in metas and job.status == "reading":
            metas[cache_id] = {
                "id": cache_id,
                "kind": "tomogram",
                "source": job.dataset,
                "status": "converting",
                "job": job.state(),
            }
    entries = []
    for t in sorted(tomograms(beamtime), key=lambda d: d["name"]):
        entry = metas.get(tomogram_cache_id(t["name"]))
        entries.append({"dataset": t["name"], "meta": t["meta"], "count": t["count"], "zarr": entry})
    return {"cache_dir": str(root), "tomograms": entries}
