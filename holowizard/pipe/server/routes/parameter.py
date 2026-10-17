"""Phase retrieval page (parameter.html, parameter_form.html): parameter sets, live preview and scan submission."""

import asyncio
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse
from omegaconf import OmegaConf
from plotly.utils import PlotlyJSONEncoder
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile

from holowizard.pipe.scan import P05Scan as Scan
from holowizard.pipe.scan.p05_scan import P05Geometry

UI_STATE_FILE = "ui_state.json"
# Stages whose config is fixed and not selectable in the UI.
FIXED_STAGE_CONFIGS = {"find_focus": "defaults.yaml"}
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
    stages: List[str] = Field(default_factory=list)
    options: Dict[str, Any] = Field(default_factory=dict)
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


def process_image(path: Path):
    with tifffile.TiffFile(str(path)) as tif:
        arr = tif.pages[0].asarray()
    low5, high95 = np.percentile(arr.flatten(), [5, 95])
    fig = px.imshow(
        img=np.rot90(arr),
        color_continuous_scale="gray",
        origin="upper",
        zmin=low5,
        zmax=high95,
    )
    fig_json = fig.to_dict()
    return fig_json


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


def _ui_state_path(state) -> Path:
    return Path(state.beamtime.log_path) / UI_STATE_FILE


# --- API routes ---
@router.post("/api/submit_scan")
async def submit_scan(cfg_in: ScanConfig, bg: BackgroundTasks, request: Request):
    state = request.app.state
    cfg = state.cfg.copy()
    energy = cfg_in.energy or state.cfg.scan.energy
    if cfg_in.stages:
        cfg.scan.tasks = ["flatfield"] + cfg_in.stages
    for stage in cfg_in.stages:
        if stage == "tomography":
            continue
        opt = FIXED_STAGE_CONFIGS.get(stage) or cfg_in.options.get(stage, stage or "wire.yaml")
        if opt == "custom":
            setattr(cfg, stage, dict_from_form(cfg_in.form_data or {}))
        else:
            try:
                setattr(cfg, stage, load_stage_config(state.config_dir, stage, opt))
            except Exception:
                raise HTTPException(status_code=400, detail=f"Please set config for stage: '{stage}'")

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
    bg.add_task(state.beamtime.new_scan, scan)
    return JSONResponse({"status": "submitted"})


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


@router.get("/api/ui_state/{page}")
async def get_ui_state(page: str, request: Request):
    state = await run_in_threadpool(_read_ui_state, _ui_state_path(request.app.state))
    return state.get(page, {})


@router.post("/api/ui_state/{page}")
async def post_ui_state(page: str, page_state: Dict[str, Any], request: Request):
    if not page.isidentifier():
        raise HTTPException(status_code=400, detail=f"Invalid page name: {page!r}")
    try:
        await run_in_threadpool(save_ui_state, _ui_state_path(request.app.state), page, page_state)
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
    out = request.app.state.config_dir / stage / f"{name.split('.')[0]}.yaml"
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
async def worker(ws, beamtime, scan, session, data):
    # run the heavy lift in a thread, get a result back
    try:
        result = await asyncio.to_thread(
            beamtime.phase_retrieval_single_holo,
            scan,
            session,
            img_name=data["img_name"],
            find_focus=data.get("find_focus", False),
        )
    except asyncio.CancelledError:
        return
    if result:
        ret = dict(
            z01=result.get("z01"),
        )
        await ws.send_text(json.dumps(ret, cls=PlotlyJSONEncoder))


@router.websocket("/ws/preview")
async def websocket_preview(ws: WebSocket):
    await ws.accept()
    state = ws.app.state
    beamtime = state.beamtime
    beamtime.cluster.min_worker = 1

    current_session: Optional[str] = None
    current_task: Optional[asyncio.Task] = None
    current_scan: Optional[Scan] = None

    def _switch_subscription(new_session: str):
        nonlocal current_session, sub
        if current_session:
            sub.setsockopt(zmq.UNSUBSCRIBE, current_session.encode())
        sub.setsockopt(zmq.SUBSCRIBE, new_session.encode())
        current_session = new_session

    # session = str(uuid.uuid4())
    ctx = zmq.asyncio.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{os.getenv('HNAME')}:{os.getenv('PUB_PORT', '6001')}")
    # sub.setsockopt(zmq.SUBSCRIBE, session.encode())
    forward = asyncio.create_task(_forward_zmq(ws, sub))
    cfg = state.cfg
    try:
        while True:
            data = await ws.receive_json()

            if data.get("action") == "cancel_preview":
                if current_scan is not None:
                    current_scan.cancelled = True
                if current_session:
                    try:
                        beamtime.cancel_task(current_session)
                    except Exception as e:
                        print(f"cancel_task failed for {current_session}: {e}")
                if current_task is not None:
                    current_task.cancel()
                    current_task = None
                _switch_subscription("noop-" + uuid.uuid4().hex)
                current_scan = None

                try:
                    await ws.send_text(json.dumps({"stopped": True}))
                except Exception:
                    pass
                continue

            form = dict_from_form(data.get("form_data", {}))
            cfg.reconstruction = form
            try:
                cfg.find_focus = load_stage_config(state.config_dir, "find_focus", FIXED_STAGE_CONFIGS["find_focus"])
            except Exception as e:
                await ws.send_text(json.dumps({"error": f"Could not load find focus config: {e}"}))
                continue
            cfg.paths.base_dir = data.get("base_dir", cfg.paths.base_dir)

            run_session = str(uuid.uuid4())
            _switch_subscription(run_session)

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

            current_scan = scan
            current_task = asyncio.create_task(worker(ws, beamtime, scan, run_session, data))

    except WebSocketDisconnect:
        forward.cancel()
        sub.close()
        if current_scan is not None:
            current_scan.cancelled = True
        if current_task is not None:
            current_task.cancel()
        if current_session:
            try:
                beamtime.cancel_task(current_session)
            except Exception:
                pass
    finally:
        beamtime.cluster.min_worker = 0


async def _forward_zmq(ws: WebSocket, sock: zmq.asyncio.Socket):
    try:
        while True:
            _, frame = await sock.recv_multipart()
            await ws.send_bytes(frame)
    except WebSocketDisconnect:
        pass
