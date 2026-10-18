"""Tomography page (tomography.html): reconstruct all slices of a phase retrieval run.

Uses the rotation center saved on the rotation center page (``rotation_center.yaml``) and the same projection
filters, reconstructs every detector row with astra's FBP_CUDA and writes one float32 TIFF per slice to
``path_processed/<scan>/<save prefix>/tomograms/<name>/slice_<row>.tiff`` (the run folder of the phase
retrieval). ``path_processed/<scan>/recon_dir/<name>`` is a relative symbolic link to that folder, so the
scan overview ("Tomograms" view) shows the result. The parameters are stored next to the slices in
``reconstruction.yaml``.

Projections are read in blocks of detector rows (the next block is read while the current one is
reconstructed); optional binning averages ``binning x binning`` detector pixels before the reconstruction.
"""

import datetime
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tifffile
import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from holowizard.pipe.server.routes.overview import RECON_SUBDIR
from holowizard.pipe.server.routes.rotation import (
    ALGORITHM,
    MIN_PROJECTIONS,
    READ_THREADS,
    Filters,
    _dataset_info,
    _projections_and_angles,
    _run_folders,
    apply_filters,
    fbp,
    projection_stats,
)

PARAMETER_FILE = "reconstruction.yaml"
TOMOGRAM_SUBDIR = "tomograms"  # in the run folder of the phase retrieval
BLOCK_BYTES = 512 * 1024**2  # projection rows read at once (about), per block
NAME_PATTERN = re.compile(r"[A-Za-z0-9_.+-]{1,120}")

router = APIRouter(tags=["tomography"])
_jobs: Dict[str, "TomogramJob"] = {}
_user_jobs: Dict[str, str] = {}
_jobs_lock = threading.Lock()


def _read_rows(files: List[Path], first: int, last: int, binning: int) -> np.ndarray:
    """Detector rows ``first .. last-1`` of all projections, binned: (projections, rows/binning, cols/binning)."""

    def read(f: Path) -> np.ndarray:
        rows = np.array(tifffile.memmap(str(f), mode="r")[first:last], dtype=np.float32)
        if binning > 1:
            h, w = (rows.shape[0] // binning) * binning, (rows.shape[1] // binning) * binning
            rows = rows[:h, :w].reshape(h // binning, binning, w // binning, binning).mean(axis=(1, 3))
        return rows

    with ThreadPoolExecutor(READ_THREADS) as ex:
        block = np.stack(list(ex.map(read, files)))
    return np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)


class TomogramJob:
    """Reconstructs all slices of a run (in a thread) and writes them as TIFFs."""

    def __init__(
        self,
        beamtime,
        user: str,
        dataset: str,
        files: List[Path],
        angles: np.ndarray,
        raw_scan: Path,
        shape,
        center: float,
        filters: Filters,
        first_row: int,
        last_row: int,
        binning: int,
        output: Path,
        output_dataset: str,
        link: Path,
    ):
        self.id = uuid.uuid4().hex
        self.beamtime, self.user, self.dataset = beamtime, user, dataset
        self.files, self.angles, self.raw_scan, self.shape = files, angles, raw_scan, shape
        self.center, self.filters, self.binning = center, filters, binning
        self.first_row, self.last_row = first_row, last_row
        self.output, self.output_dataset, self.link = output, output_dataset, link
        self.total = (last_row - first_row + 1) // binning
        self.done = 0
        self.used = 0
        self.removed: Dict[str, int] = {}
        self.notes: List[str] = []
        self.status, self.message = "checking", "Checking projections…"
        self.check_count = 0
        self.started = time.time()
        self.finished: Optional[float] = None
        self.cancelled = False
        threading.Thread(target=self._run, name=f"tomogram-{self.id[:8]}", daemon=True).start()

    # binned row index -> file name
    def slice_name(self, index: int) -> str:
        return f"slice_{index:05d}.tiff"

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
            files = [f for f, k in zip(self.files, keep) if k]
            angles = self.angles[keep]
            self.used = len(files)

            self.output.mkdir(parents=True, exist_ok=True)
            self._create_link()
            self._write_parameters(status="running")
            b = self.binning
            # The rotation center in binned detector pixels (pixel centres).
            center = (self.center + 0.5) / b - 0.5
            width = self.shape[1] // b
            rows_per_block = max(b, (BLOCK_BYTES // max(1, len(files) * self.shape[1] * 4)) // b * b)
            starts = list(range(self.first_row, self.first_row + self.total * b, rows_per_block))

            self.status, self.message = "reconstructing", "Reconstructing…"
            with ThreadPoolExecutor(1) as reader:

                def block_at(i):
                    first = starts[i]
                    last = min(first + rows_per_block, self.first_row + self.total * b)
                    return reader.submit(_read_rows, files, first, last, b)

                pending = block_at(0)
                for i in range(len(starts)):
                    block = pending.result()
                    if i + 1 < len(starts):
                        pending = block_at(i + 1)  # read the next block while reconstructing this one
                    for j in range(block.shape[1]):
                        if self.cancelled:
                            self.status, self.message = "cancelled", "Cancelled"
                            self._write_parameters(status="cancelled")
                            return
                        rec = fbp(np.ascontiguousarray(block[:, j, :width]), angles, center)
                        index = (starts[i] - self.first_row) // b + j
                        tifffile.imwrite(self.output / self.slice_name(index), rec.astype(np.float32))
                        self.done += 1
            self.status, self.message = "done", "Done"
            self.finished = time.time()
            self._write_parameters(status="done")
        except Exception as e:
            self.status, self.message = "error", str(e)
            try:
                self._write_parameters(status=f"error: {e}")
            except Exception:
                pass

    def _create_link(self):
        """Relative symbolic link recon_dir/<name> -> ../<save prefix>/tomograms/<name> (for the scan overview)."""
        target = Path(os.path.relpath(self.output, self.link.parent))
        if self.link.is_symlink():
            if Path(os.readlink(self.link)) == target:
                return
            raise ValueError(f"{self.link} already links elsewhere; choose another name.")
        if self.link.exists():
            raise ValueError(f"{self.link} already exists; choose another name.")
        self.link.parent.mkdir(parents=True, exist_ok=True)
        self.link.symlink_to(target, target_is_directory=True)

    def _write_parameters(self, status: str):
        data = {
            "status": status,
            "slices_written": self.done,
            "slices": self.total,
            "algorithm": ALGORITHM,
            "rotation_center": float(self.center),
            "rotation_center_binned": float((self.center + 0.5) / self.binning - 0.5),
            "binning": self.binning,
            "detector_rows": [self.first_row, self.first_row + self.total * self.binning - 1],
            "slice_file": "slice_<binned row index from detector_rows[0]>.tiff, float32",
            "projections": len(self.files),
            "projections_used": self.used,
            "removed": self.removed,
            "filters": self.filters.model_dump(),
            "phase_retrieval_run": self.dataset,
            "projection_folder": str(self.files[0].parent),
            "recon_dir_link": str(self.link),
            "started_by": self.user,
            "started_at": datetime.datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "finished_at": datetime.datetime.fromtimestamp(self.finished).isoformat(timespec="seconds")
            if self.finished
            else None,
        }
        tmp = self.output / f".{PARAMETER_FILE}.{uuid.uuid4().hex}.tmp"
        tmp.write_text(yaml.safe_dump(data, sort_keys=False))
        tmp.replace(self.output / PARAMETER_FILE)

    def state(self) -> Dict[str, Any]:
        elapsed = (self.finished or time.time()) - self.started
        rate = self.done / elapsed if self.done and self.status == "reconstructing" else None
        return {
            "id": self.id,
            "dataset": self.dataset,
            "status": self.status,
            "message": self.message,
            "checked": self.check_count,
            "projections": len(self.files),
            "used": self.used,
            "removed": self.removed,
            "notes": self.notes,
            "done": self.done,
            "total": self.total,
            "center": self.center,
            "binning": self.binning,
            "output": str(self.output),
            "output_dataset": self.output_dataset,
            "elapsed": round(elapsed),
            "eta": round((self.total - self.done) / rate) if rate else None,
        }


# --- Routes ---
class TomogramRequest(BaseModel):
    dataset: str
    center: float
    name: str
    filters: Filters = Filters()
    first_row: Optional[int] = None
    last_row: Optional[int] = None
    binning: int = 1


def _user(request: Request) -> str:
    return getattr(request.state, "user", None) or "anonymous"


@router.post("/api/tomography/start")
async def start_tomogram(req: TomogramRequest, request: Request):
    beamtime = request.app.state.beamtime
    user = _user(request)
    with _jobs_lock:
        current = _jobs.get(_user_jobs.get(user, ""))
    if current is not None and current.status in ("checking", "reconstructing"):
        raise HTTPException(status_code=409, detail="You already have a tomogram reconstruction running.")
    if req.binning not in (1, 2, 4):
        raise HTTPException(status_code=400, detail="Binning must be 1, 2 or 4")
    if not NAME_PATTERN.fullmatch(req.name) or req.name in (".", ".."):
        raise HTTPException(status_code=400, detail="Use only letters, digits and . _ + - in the name")
    try:
        info = await run_in_threadpool(_dataset_info, beamtime, req.dataset)
        _, projections, raw_scan = _run_folders(beamtime, req.dataset)
        files, angles, _ = await run_in_threadpool(_projections_and_angles, projections, raw_scan)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    rows, cols = info["rows"], info["cols"]
    first = 0 if req.first_row is None else req.first_row
    last = rows - 1 if req.last_row is None else req.last_row
    if not 0 <= first <= last < rows:
        raise HTTPException(status_code=400, detail=f"Slices must lie between 0 and {rows - 1}")
    if (last - first + 1) // req.binning < 1:
        raise HTTPException(status_code=400, detail="The slice range is shorter than the binning")
    if not 0 <= req.center < cols:
        raise HTTPException(status_code=400, detail=f"The rotation center must lie between 0 and {cols - 1}")

    scan = req.dataset.split("/")[0]
    run_folder = _run_folders(beamtime, req.dataset)[0]
    output = run_folder / TOMOGRAM_SUBDIR / req.name
    link = Path(beamtime.path_processed) / scan / RECON_SUBDIR / req.name
    if output.exists() and any(p.suffix.lower() in (".tif", ".tiff") for p in output.iterdir()):
        raise HTTPException(
            status_code=409, detail=f"{output} already contains slices; choose another name (nothing is overwritten)."
        )
    if link.exists() or link.is_symlink():
        ours = link.is_symlink() and Path(os.readlink(link)) == Path(os.path.relpath(output, link.parent))
        if not ours:
            raise HTTPException(
                status_code=409,
                detail=f"{link} already exists in recon_dir; choose another name (nothing is replaced).",
            )

    job = TomogramJob(
        beamtime,
        user,
        req.dataset,
        files,
        angles,
        raw_scan,
        (rows, cols),
        req.center,
        req.filters,
        first,
        last,
        req.binning,
        output,
        f"{scan}/{req.name}",
        link,
    )
    with _jobs_lock:
        _jobs[job.id] = job
        _user_jobs[user] = job.id
    return job.state()


@router.get("/api/tomography/current")
async def current_tomogram(request: Request):
    """The latest reconstruction of this user (to reattach after a page reload), or null."""
    job = _jobs.get(_user_jobs.get(_user(request), ""))
    return job.state() if job else None


@router.get("/api/tomography/job/{job_id}")
async def tomogram_state(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Reconstruction not found (server restarted?)")
    return job.state()


@router.post("/api/tomography/job/{job_id}/cancel")
async def cancel_tomogram(job_id: str, request: Request):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Reconstruction not found")
    if job.user != _user(request):
        raise HTTPException(status_code=403, detail="Only the user who started a reconstruction can cancel it")
    job.cancelled = True
    return job.state()


@router.get("/tomography")
async def tomography_page(request: Request):
    return request.app.state.templates.TemplateResponse("tomography.html", {"request": request})
