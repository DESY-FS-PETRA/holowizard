"""Phase retrieval page (parameter.html, parameter_form.html): parameter sets, live preview and scan submission."""

import asyncio
import json
import os
import re
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import plotly.express as px
import tifffile
import yaml
import zmq
import zmq.asyncio
from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse
from omegaconf import OmegaConf
from plotly.utils import PlotlyJSONEncoder
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from holowizard.pipe.scan import P05Scan as Scan
from holowizard.pipe.scan.p05_scan import P05Geometry

UI_STATE_DIR = "ui_state"  # one JSON file per browser, inside beamtime.log_path
UI_CLIENT_COOKIE = "holowizard_client_id"
# Stages whose config is fixed and not selectable in the UI.
FIXED_STAGE_CONFIGS = {"find_focus": "defaults.yaml"}
# Tasks run by "Submit Phase Retrieval"; flatfield is always run first by the beamtime.
SUBMIT_TASKS = ["flatfield", "reconstruction"]
# Every edit made in the reconstruction parameter form goes into the parameter set of the logged-in user.
# Everyone can load it, only its user can write it.
CUSTOM_PREFIX = "custom_"
PARAMETER_SET_NAME = re.compile(r"[A-Za-z0-9_-]+")


def custom_set_name(user: str) -> str:
    return f"{CUSTOM_PREFIX}{user}.yaml"


_ui_state_lock = threading.Lock()

router = APIRouter(tags=["parameter"])


# --- Pydantic Model ---
class ScanConfig(BaseModel):
    scan_name: str
    holder: Optional[Union[str, float, int]] = None
    z01: Optional[float] = None
    z02: Optional[float] = None
    a0: Optional[float] = None
    energy: Optional[float] = None
    base_dir: Optional[Path] = None
    form_data: Optional[Dict[str, Any]] = None


class GeometryRequest(BaseModel):
    holder: int
    qp: bool = False
    energy: Optional[float] = None
    scan_name: str


# --- Utils ---
def _parse_val(val: str) -> Union[int, float, str]:
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            pass
    return val


def dict_from_form(form: Dict[str, Any]) -> Dict[str, Any]:
    data, temp = {"stages": []}, {}
    for k, v in form.items():
        if k in ("param_set_name", "stage"):
            continue
        parts, d = k.split("."), temp
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = _parse_val(v)
    for k in sorted(temp):
        data["stages"].append(temp[k])

    return data


# The test image plot is at most a few hundred pixels wide, so it is sent downsampled to this size.
PREVIEW_IMAGE_SIZE = 768


def _block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return arr
    h, w = (arr.shape[0] // factor) * factor, (arr.shape[1] // factor) * factor
    small = arr[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return np.rint(small).astype(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else small.astype(np.float32)


@lru_cache(maxsize=32)
def _image_figure(path: str, mtime_ns: int) -> Dict[str, Any]:
    """Plotly figure of a TIFF, downsampled for display; ``mtime_ns`` is only part of the cache key."""
    with tifffile.TiffFile(path) as tif:
        arr = tif.pages[0].asarray()
    # Contrast limits from the full-resolution image, so they do not depend on the downsampling.
    low5, high95 = np.percentile(arr, [5, 95])
    img = np.rot90(arr)
    factor = max(1, -(-max(img.shape) // PREVIEW_IMAGE_SIZE))
    small = _block_mean(img, factor)
    # Axes in full-resolution pixel coordinates (centre of each block).
    offset = (factor - 1) / 2
    fig = px.imshow(
        img=small,
        x=np.arange(small.shape[1]) * factor + offset,
        y=np.arange(small.shape[0]) * factor + offset,
        color_continuous_scale="gray",
        origin="upper",
        zmin=low5,
        zmax=high95,
    )
    return fig.to_dict()


def process_image(path: Path):
    return _image_figure(str(path), path.stat().st_mtime_ns)


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {path}")
    return yaml.safe_load(path.read_text())


def _read_ui_state(path: Path) -> Dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def save_ui_state(path: Path, page: str, page_state: Dict[str, Any]):
    """Merge the state of one page into the UI state file (atomic write)."""
    with _ui_state_lock:
        state = _read_ui_state(path)
        state[page] = page_state
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(tmp, path)


def load_stage_config(config_dir: Path, stage: str, opt: str) -> Dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(str(config_dir / stage / opt)), resolve=True)


def _ui_state_path(request: Request, response: Response) -> Path:
    """State file of the requesting browser; assigns a new client id cookie if it has none."""
    client_id = request.cookies.get(UI_CLIENT_COOKIE, "")
    if not re.fullmatch(r"[0-9a-f]{32}", client_id):
        client_id = uuid.uuid4().hex
        response.set_cookie(UI_CLIENT_COOKIE, client_id, max_age=10 * 365 * 24 * 3600, httponly=True, samesite="lax")
    return Path(request.app.state.beamtime.log_path) / UI_STATE_DIR / f"{client_id}.json"


# --- API routes ---
@router.post("/api/submit_scan")
async def submit_scan(cfg_in: ScanConfig, bg: BackgroundTasks, request: Request):
    state = request.app.state
    cfg = state.cfg.copy()
    energy = cfg_in.energy or state.cfg.scan.energy
    cfg.scan.tasks = SUBMIT_TASKS
    # The reconstruction always uses the parameters of the form on the phase retrieval page.
    reconstruction = dict_from_form(cfg_in.form_data or {})
    if not reconstruction["stages"]:
        raise HTTPException(status_code=400, detail="No reconstruction parameters; choose a parameter set first")
    cfg.reconstruction = reconstruction

    if cfg_in.base_dir is not None:
        cfg.paths.base_dir = cfg_in.base_dir

    holder_val = float(cfg_in.holder) if cfg_in.holder is not None else 0

    scan = Scan(
        name=cfg_in.scan_name,
        holder=holder_val,
        path_raw=state.beamtime.path_raw,
        path_processed=state.beamtime.path_processed,
        z01_new=cfg_in.z01,
        z02_new=cfg_in.z02,
        energy=energy,
        cfg=cfg,
        a0=cfg_in.a0,
        log_path=state.beamtime.log_path,
    )
    bg.add_task(_run_with_cluster, state.cluster, state.beamtime.new_scan, scan)
    return JSONResponse({"status": "submitted"})


def _run_with_cluster(cluster, func, *args, **kwargs):
    """Run ``func`` while holding the SLURM workers (they are started first if needed)."""
    with cluster.lease():
        return func(*args, **kwargs)


@router.post("/api/compute_z_params")
def api_compute_z_params(req: GeometryRequest, request: Request):
    state = request.app.state
    energy_cfg = OmegaConf.select(state.cfg, "scan.energy")
    energy_val = req.energy if req.energy is not None else energy_cfg
    if energy_val is None:
        raise HTTPException(
            status_code=400, detail="Energy is not set (request.energy and cfg.scan.energy are both None)"
        )
    try:
        energy = float(energy_val)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid energy value: {energy_val!r}")

    raw_dir = Path(state.beamtime.path_raw)
    scan_name = (req.scan_name or "").strip()
    if not scan_name:
        raise HTTPException(status_code=400, detail="scan_name is required")
    scan_dir = raw_dir / scan_name
    if not scan_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Scan folder not found: {scan_dir}")

    try:
        geo = P05Geometry(
            scan_path=str(scan_dir),
            energy=energy,
            holder=req.holder,
            qp=req.qp,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to init P05Geometry: {e}")

    try:
        z01, z02 = geo.compute_z_params()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"compute_z_params failed: {e}")

    try:
        return {"z01": float(z01), "z02": float(z02)}
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid z01/z02 returned: {z01}, {z02}")


@router.post("/api/parameter_sets/custom")
async def save_custom_parameters(form_data: Dict[str, str], request: Request):
    name = custom_set_name(request.state.user)
    out = request.app.state.config_dir / "reconstruction" / name
    data = dict_from_form(form_data)

    def _write():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
        os.replace(tmp, out)

    await run_in_threadpool(_write)
    return {"name": name}


@router.get("/api/ui_state/{page}")
async def get_ui_state(page: str, request: Request, response: Response):
    state = await run_in_threadpool(_read_ui_state, _ui_state_path(request, response))
    return state.get(page, {})


@router.post("/api/ui_state/{page}")
async def post_ui_state(page: str, page_state: Dict[str, Any], request: Request, response: Response):
    if not page.isidentifier():
        raise HTTPException(status_code=400, detail=f"Invalid page name: {page!r}")
    try:
        await run_in_threadpool(save_ui_state, _ui_state_path(request, response), page, page_state)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not save UI state: {e}")
    return {"status": "saved"}


# --- UI routes ---
@router.get("/parameter")
async def parameter_form(request: Request):
    state = request.app.state
    cfg = state.cfg.scan
    return state.templates.TemplateResponse(
        "parameter.html",
        {
            "request": request,
            "z01": cfg.z01,
            "z02": cfg.z02,
            "energy": cfg.energy,
            "basepath": state.cfg.paths.base_dir,
            "custom_set": custom_set_name(request.state.user),
        },
    )


@router.post("/parameter/{base_name}")
async def parameter_details(base_name: str, request: Request):
    state = request.app.state
    data = await request.json()
    dropdown = data.get("dropdown")
    path = state.config_dir / dropdown / base_name
    scans = load_yaml(path)
    return state.templates.TemplateResponse("parameter_form.html", {"request": request, "scans": scans})


@router.post("/tuning")
async def save_tuning(request: Request):
    form = await request.form()
    data = dict_from_form(form)
    stage = form.get("stage", "reconstruction")
    if stage in FIXED_STAGE_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Parameters of stage '{stage}' are fixed and cannot be saved")
    name: UploadFile | str = form.get("param_set_name", "unnamed")
    stem = name.split(".")[0]
    if not PARAMETER_SET_NAME.fullmatch(stem):
        raise HTTPException(status_code=400, detail="Use only letters, digits, '_' and '-' in the name")
    if stem.startswith(CUSTOM_PREFIX) and f"{stem}.yaml" != custom_set_name(request.state.user):
        raise HTTPException(status_code=403, detail=f"'{stem}' belongs to another user")
    out = request.app.state.config_dir / stage / f"{stem}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(data, out.open("w"), default_flow_style=False)
    return RedirectResponse(url="/parameter", status_code=303)


@router.get("/stage/{stage}")
async def list_stage(stage: str, request: Request):
    path = request.app.state.config_dir / stage
    if not path.exists():
        raise HTTPException(status_code=404, detail="Stage not found")
    return os.listdir(path)


## TODO that has to be changed to the beamtime object
@router.get("/folder")
async def list_folders(request: Request):
    path = Path(request.app.state.beamtime.path_raw)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    return sorted(os.listdir(path))


## TODO that has to be changed to the beamtime object
@router.get("/folder/{folder_name}")
async def list_folder(folder_name: str, request: Request):
    path = Path(request.app.state.beamtime.path_raw) / folder_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    return sorted([x for x in os.listdir(path) if "img" in x])


## TODO that has to be changed to a scan object
@router.get("/image/{scan}/{img}")
async def get_image(scan: str, img: str, request: Request):
    path = Path(request.app.state.beamtime.path_raw) / scan / img
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    fig_dict = await run_in_threadpool(process_image, path)
    return JSONResponse(content=fig_dict)


# --- Live preview websocket ---
# A preview run belongs to the logged-in user, not to one websocket connection: reloading the page (or a
# second tab) reattaches to the running preview. A run without any connected page is cancelled after
# PREVIEW_GRACE_SECONDS; "Stop" or starting a new preview cancels it right away.
PREVIEW_GRACE_SECONDS = 60


class PreviewRun:
    def __init__(self, session: str, scan, find_focus: bool):
        self.session = session
        self.scan = scan
        self.find_focus = find_focus
        self.task: Optional[asyncio.Task] = None
        self.pending_message: Optional[Dict[str, Any]] = None  # result that finished while no page was connected
        self.step: Optional[str] = None  # current step, reported by the beamtime ("flatfield", ...)
        self.started = time.monotonic()
        self.status: Optional[str] = None  # last status line sent to the page

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()


class PreviewConnection:
    def __init__(self, ws: WebSocket, sub: zmq.asyncio.Socket):
        self.ws = ws
        self.sub = sub
        self.topic: Optional[str] = None

    def subscribe(self, topic: Optional[str]):
        if self.topic:
            self.sub.setsockopt(zmq.UNSUBSCRIBE, self.topic.encode())
        if topic:
            self.sub.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self.topic = topic

    async def send(self, message: Dict[str, Any]) -> bool:
        try:
            await self.ws.send_text(json.dumps(message, cls=PlotlyJSONEncoder))
            return True
        except Exception:
            return False


class UserPreview:
    def __init__(self):
        self.run: Optional[PreviewRun] = None
        self.connections: set = set()
        self.cancel_timer: Optional[asyncio.Task] = None

    async def broadcast(self, message: Dict[str, Any]) -> bool:
        """Send to every connected page of the user; True if at least one received it."""
        results = [await c.send(message) for c in list(self.connections)]
        return any(results)


_user_previews: Dict[str, UserPreview] = {}


def _cancel_run(beamtime, run: PreviewRun):
    run.scan.cancelled = True
    try:
        beamtime.cancel_task(run.session)
    except Exception as e:
        print(f"cancel_task failed for {run.session}: {e}")
    if run.task is not None and not run.task.done():
        run.task.cancel()


# Status line on the page until the first image or plot arrives (the page hides it then): the step and whether a
# SLURM worker is connected yet. Starting workers on demand can take minutes (job queue, Python start on the node).
STEP_TEXT = {
    None: "Starting…",
    "flatfield": "Flatfield correction…",
    "find_focus": "Find focus: reconstructing the first z01 sampling point…",
    "reconstruction": "Reconstructing the test image…",
}


def _connected_workers(cluster) -> int:
    try:
        return cluster.status()["workers"]
    except Exception:
        return 0


async def _report_status(cluster, preview: UserPreview, run: PreviewRun):
    """Send the status line once a second while the run is busy."""
    while run.task is not None and not run.task.done() and preview.run is run:
        workers = await asyncio.to_thread(_connected_workers, cluster)
        if workers == 0 and run.step is not None:
            text = "Waiting for a SLURM worker to start (job in the queue or starting on the node)…"
        else:
            text = STEP_TEXT.get(run.step, "Running…")
        elapsed = int(time.monotonic() - run.started)
        run.status = f"{text} {elapsed // 60}:{elapsed % 60:02d}"
        await preview.broadcast({"status": run.status})
        await asyncio.sleep(1)


async def _execute_preview(beamtime, cluster, preview: UserPreview, run: PreviewRun, data: Dict[str, Any]):
    loop = asyncio.get_running_loop()

    def progress(step: str):  # called from the worker thread of the beamtime
        loop.call_soon_threadsafe(setattr, run, "step", step)

    # run the heavy lift in a thread, get a result back
    try:
        result = await asyncio.to_thread(
            _run_with_cluster,
            cluster,
            beamtime.phase_retrieval_single_holo,
            run.scan,
            run.session,
            img_name=data["img_name"],
            find_focus=run.find_focus,
            progress=progress,
        )
    except asyncio.CancelledError:
        return
    except Exception as e:
        message = {"error": f"Preview failed: {e}"}
    else:
        if not result:
            return
        message = {"z01": result.get("z01")}
    if preview.run is run and not await preview.broadcast(message):
        run.pending_message = message


async def _cancel_when_abandoned(beamtime, preview: UserPreview, run: PreviewRun):
    await asyncio.sleep(PREVIEW_GRACE_SECONDS)
    if not preview.connections and preview.run is run:
        _cancel_run(beamtime, run)
        preview.run = None


@router.websocket("/ws/preview")
async def websocket_preview(ws: WebSocket):
    await ws.accept()
    state = ws.app.state
    beamtime = state.beamtime
    beamtime.cluster.min_worker = 1
    user = getattr(ws.state, "user", None) or "anonymous"
    preview = _user_previews.setdefault(user, UserPreview())

    ctx = zmq.asyncio.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{os.getenv('HNAME')}:{os.getenv('PUB_PORT', '6001')}")
    conn = PreviewConnection(ws, sub)
    preview.connections.add(conn)
    forward = asyncio.create_task(_forward_zmq(ws, sub))

    # Reattach to a preview of this user that is still running (e.g. after a page reload).
    if preview.cancel_timer is not None:
        preview.cancel_timer.cancel()
        preview.cancel_timer = None
    run = preview.run
    if run is not None:
        conn.subscribe(run.session)
        if run.running:
            await conn.send({"running": True, "find_focus": run.find_focus, "status": run.status})
        if run.pending_message is not None and await conn.send(run.pending_message):
            run.pending_message = None

    cfg = state.cfg
    try:
        while True:
            data = await ws.receive_json()

            if data.get("action") == "cancel_preview":
                if preview.run is not None:
                    _cancel_run(beamtime, preview.run)
                    preview.run = None
                for c in preview.connections:
                    c.subscribe(None)
                await preview.broadcast({"stopped": True})
                continue

            form = dict_from_form(data.get("form_data", {}))
            cfg.reconstruction = form
            try:
                cfg.find_focus = load_stage_config(state.config_dir, "find_focus", FIXED_STAGE_CONFIGS["find_focus"])
            except Exception as e:
                await conn.send({"error": f"Could not load find focus config: {e}"})
                continue
            cfg.paths.base_dir = data.get("base_dir", cfg.paths.base_dir)

            if preview.run is not None and preview.run.running:
                _cancel_run(beamtime, preview.run)

            run_session = str(uuid.uuid4())
            scan = Scan(
                name=data["scan_name"],
                holder=0,
                path_raw=beamtime.path_raw,
                path_processed=beamtime.log_path,
                z01_new=data["z01"],
                z02_new=data["z02"],
                energy=data.get("energy"),
                cfg=cfg,
                a0=data["a0"],
                log_path=beamtime.log_path,
            )
            scan.key = run_session

            run = PreviewRun(run_session, scan, bool(data.get("find_focus", False)))
            preview.run = run
            for c in preview.connections:
                c.subscribe(run_session)
            # Other open pages of this user show that a preview is running.
            for c in list(preview.connections):
                if c is not conn:
                    await c.send({"running": True, "find_focus": run.find_focus})
            run.task = asyncio.create_task(_execute_preview(beamtime, state.cluster, preview, run, data))
            asyncio.create_task(_report_status(state.cluster, preview, run))

    except WebSocketDisconnect:
        pass
    finally:
        forward.cancel()
        sub.close()
        preview.connections.discard(conn)
        if not preview.connections and preview.run is not None and preview.run.running:
            preview.cancel_timer = asyncio.create_task(_cancel_when_abandoned(beamtime, preview, preview.run))
        if not preview.connections:
            beamtime.cluster.min_worker = 0


async def _forward_zmq(ws: WebSocket, sock: zmq.asyncio.Socket):
    try:
        while True:
            _, frame = await sock.recv_multipart()
            await ws.send_bytes(frame)
    except (WebSocketDisconnect, asyncio.CancelledError, RuntimeError, zmq.ZMQError):
        pass
