import os
import io
import socket
import uuid
import threading
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
import yaml
import markdown

from plotly.utils import PlotlyJSONEncoder
import zmq
import zmq.asyncio
import tifffile
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from fastapi import (
    FastAPI, APIRouter, BackgroundTasks, HTTPException, Request,
    WebSocket, WebSocketDisconnect, Depends
)
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from omegaconf import OmegaConf

import holowizard
from holowizard.pipe.cluster import SlurmCluster as Cluster
from holowizard.pipe.scan import P05Scan as Scan
from holowizard.pipe.beamtime import P05Beamtime as Beamtime
from holowizard.pipe.utils.clean_yaml import to_clean_yaml
import plotly.express as px
import json 

def find_free_port(host: str = "127.0.0.1") -> int:
    # Bind to port 0 tells the OS to pick an unused port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]
    
# --- Environment & Config ---
HOSTNAME = socket.gethostname()
sub_port = find_free_port()
pub_port = find_free_port()
os.environ.setdefault("HNAME", HOSTNAME)
os.environ.setdefault("SUB_PORT", str(sub_port))
os.environ.setdefault("PUB_PORT", str(pub_port))
env_path = Path('.') / '.env'
with env_path.open('w') as f:
    f.write(f"HNAME={HOSTNAME}\n")
    f.write(f"SUB_PORT={sub_port}\n")
    f.write(f"PUB_PORT={pub_port}\n")
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "configs"
CITATIONS_FILE = BASE_DIR.parent / "citations.yaml"

# Load citations
citations = OmegaConf.to_container(
    OmegaConf.load(str(CITATIONS_FILE)), resolve=True
).get("citations", {})



# --- Pydantic Model ---
class ScanConfig(BaseModel):
    scan_name: str
    holder: Union[str, float, int]
    z01: Optional[float] = None
    a0: Optional[float] = None
    energy: Optional[float] = None
    base_dir : Optional[Path] = None
    stages:  List[str]        = Field(default_factory=list)
    options: Dict[str,Any]    = Field(default_factory=dict)
    form_data: Optional[Dict[str, Any]] = None

# --- Utils ---
def _parse_val(val: str) -> Union[int, float, str]:
    for cast in (int, float):
        try: return cast(val)
        except ValueError: pass
    return val

def dict_from_form(form: Dict[str, Any]) -> Dict[str, Any]:
    data, temp = {"stages": []}, {}
    for k, v in form.items():
        if k in ("param_set_name", "stage"): continue
        parts, d = k.split('.'), temp
        for p in parts[:-1]: d = d.setdefault(p, {})
        d[parts[-1]] = _parse_val(v)
    for k in sorted(temp): data["stages"].append(temp[k])

    return data

def process_image(path: Path):
    with tifffile.TiffFile(str(path)) as tif:
        arr = tif.pages[0].asarray()
    low5, high95 = np.percentile(arr.flatten(), [5, 95])
    fig = px.imshow(img=np.rot90(arr), color_continuous_scale='gray',
                                origin="lower",zmin=low5, zmax=high95)
    fig_json = fig.to_dict()
    return fig_json

def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {path}")
    return yaml.safe_load(path.read_text())


# --- Lifespan & App Factory ---
@asynccontextmanager
async def _lifespan(app: FastAPI):
    cfg = getattr(app.state, "_initial_cfg",
                  OmegaConf.load(str(CONFIG_DIR / "defaults.yaml")))
    cluster = Cluster(cfg)
    app.state.beamtime = Beamtime(
        beamtime_name=cfg.beamtime.name,
        year=cfg.beamtime.year,
        cluster=cluster
    )
    app.state.cfg = cfg
    templates = Jinja2Templates(directory=str(BASE_DIR / ".." / "templates"))
    templates.env.filters['to_yaml'] = to_clean_yaml
    broker_thread = threading.Thread(target=run_broker, daemon=True)
    broker_thread.start()
    app.state.templates = templates
    yield
    del app.state.beamtime

def run_broker():
    ctx  = zmq.Context.instance()
    xsub = ctx.socket(zmq.XSUB)
    xsub.bind(f"tcp://0.0.0.0:{os.getenv('SUB_PORT', '6000')}")
    xpub = ctx.socket(zmq.XPUB)
    xpub.bind(f"tcp://0.0.0.0:{os.getenv('PUB_PORT', '6001')}")
    zmq.proxy(xsub, xpub)

# create_app factory
def create_app(cfg=None, config_dir: Union[str, Path]=None) -> FastAPI:
    global CONFIG_DIR
    if config_dir:
        CONFIG_DIR = Path(config_dir)
    app = FastAPI(lifespan=_lifespan)
    if cfg:
        app.state._initial_cfg = cfg
    _register_routes(app)
    return app

# --- Route registration ---
def _register_routes(app: FastAPI):
    api = APIRouter(prefix="/api", tags=["api"])
    ui  = APIRouter(tags=["ui"])

    # API routes
    @api.get("/config")
    async def get_config(cfg=Depends(lambda: app.state.cfg)):
        return OmegaConf.to_container(cfg, resolve=True)

    @api.get("/queue")
    async def queue_info():
        return app.state.beamtime.cluster.queue_info()

    @api.get("/progress")
    async def progress():
        scans, queue = app.state.beamtime.scans, app.state.beamtime.cluster.queue_info()
        prog = {}
        for scan in reversed(scans):
            key = scan.key
            q = [f for f in queue if key in f["key"]]
            tasks = {}
            for t in scan.config.scan.tasks:
                tq = [f for f in q if t in f["key"]]
                running = sum(1 for f in tq if f["state"] == "processing")
                total = len(scan.hologram_path) if t == "reconstruction" else 1
                if t == "reconstruction":
                    if t in [x["name"] for x in scan.done]:
                        done = total
                    elif sum(1 for f in queue if scan.key in f["key"] and "reconstruction" in f["key"]) > 0:
                        done = total -  sum(1 for f in queue if scan.key in f["key"]) 
                    else:
                        done = 0

                else:
                    done = int(any(x["name"] == t and x["status"] == "done" for x in scan.done))
                failed = int(any(x["name"]==t and x["status"]=="failed" for x in scan.done))
                tasks[t] = {"total": total, "done": done, "running": running, "failed": failed}
            prog[key] = {"name": scan.name, "base_dir": scan.config.paths.base_dir, "path": scan.path_processed, "tasks": tasks, "cancelled": scan.cancelled}
        return {"progress": prog, "num_workers": len(app.state.beamtime.cluster.client_scheduler.nthreads())}

    @api.post("/submit_scan")
    async def submit_scan(cfg_in: ScanConfig, bg: BackgroundTasks):
        cfg = app.state.cfg.copy()
        energy = cfg_in.energy or app.state.cfg.scan.energy
        if cfg_in.stages:
            cfg.scan.tasks = ["flatfield"] + cfg_in.stages
        for stage in cfg_in.stages:
            if stage == "tomography":
                continue
            opt = cfg_in.options.get(stage, stage or  "wire.yaml")
            if opt == "custom":
                setattr(cfg, stage, dict_from_form(cfg_in.form_data or {}))
            else:
                p = CONFIG_DIR / stage / opt
                setattr(cfg, stage, OmegaConf.to_container(OmegaConf.load(str(p)), resolve=True))
        
        if cfg_in.base_dir is not None:
            cfg.paths.base_dir = cfg_in.base_dir

        scan = Scan(
            name=cfg_in.scan_name,
            holder=cfg_in.holder,
            path_raw=app.state.beamtime.path_raw,
            path_processed=app.state.beamtime.path_processed,
            z01_new=cfg_in.z01,
            energy=energy,
            cfg=cfg,
            a0=cfg_in.a0,
            log_path=app.state.beamtime.log_path,
        )
        bg.add_task(app.state.beamtime.new_scan, scan)
        return JSONResponse({"status": "submitted"})

    @api.get("/cancel/{scan_id}")
    async def cancel_scan(scan_id: str):
        app.state.beamtime.cancel_task(scan_id)
        return {"status": "cancelled", "scan_id": scan_id}

    # UI routes
    @ui.get("/")
    async def home():
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        md_text = readme_path.read_text(encoding="utf-8")

        # 2. Convert to HTML
        html_readme = markdown.markdown(md_text)

        html_content = f"""
        <html>
        <head>
            <title>HoloServer</title>
        </head>
        <body>
            <h1>HoloServer is running for beamtime {app.state.beamtime.beamtime_name}</h1>
            <p>If you want to use the web interface, please open the following URL in your browser:</p>
            <ul>
            <li><a href="/dashboard">Dashboard</a> - see the queue status</li>
            <li><a href="/parameter">Parameter</a> - view and set scan parameters</li>
            </ul>
            <p>For further information and documentation read the following:</p>
            <div style="border: 1px solid #ccc; padding: 10px; background-color: #f9f9f9;">
                {html_readme}
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)
    
    @ui.get("/dashboard")
    async def dashboard(request: Request):
        data = await queue_info()
        return app.state.templates.TemplateResponse("dashboard.html", {"request": request, **await progress()})

    @ui.get("/scan/{name}", response_class=HTMLResponse)
    async def scan_detail(name: str, request: Request):
        scan = app.state.beamtime.get_scan(name)
        if not scan:
            raise HTTPException(status_code=404, detail="Scan not found")
        return app.state.templates.TemplateResponse("scan.html", {"request": request, "scan": scan, "citations": citations, "hw": {"version": holowizard.__version__}})

    @ui.get("/parameter")
    async def parameter_form(request: Request):
        cfg = app.state.cfg.scan
        return app.state.templates.TemplateResponse("parameter.html", {"request": request, "z01": cfg.z01, "energy": cfg.energy, "basepath": app.state.cfg.paths.base_dir})
    
    @ui.post("/parameter/{base_name}")
    async def parameter_details(base_name: str, request: Request):
        data = await request.json()
        dropdown = data.get("dropdown")
        path = CONFIG_DIR / dropdown / base_name
        scans = load_yaml(path)
        return app.state.templates.TemplateResponse("parameter_form.html", {"request": request, "scans": scans})
    
    @ui.post("/tuning")
    async def save_tuning(request: Request):
        form = await request.form()
        data = dict_from_form(form)
        stage = form.get("stage", "find_focus")
        name: UploadFile | str = form.get("param_set_name", "unnamed")
        out = CONFIG_DIR / stage / f"{name.split('.')[0]}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(data, out.open('w'), default_flow_style=False)
        return RedirectResponse(url=f"/parameter", status_code=303)

    @ui.get("/stage/{stage}")
    async def list_stage(stage: str):
        path = CONFIG_DIR / stage
        if not path.exists():
            raise HTTPException(status_code=404, detail="Stage not found")
        return os.listdir(path)

    ## TODO that has to be changed to the beamtime object
    @ui.get("/folder")
    async def list_folders():
        path = Path(app.state.beamtime.path_raw)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Folder not found")
        return sorted(os.listdir(path))
    
    ## TODO that has to be changed to the beamtime object 
    @ui.get("/folder/{folder_name}")
    async def list_folder(folder_name: str):
        path = Path(app.state.beamtime.path_raw) / folder_name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Folder not found")
        return sorted([x for x in os.listdir(path) if "img" in x])

    ## TODO that has to be changed to a scan object
    @ui.get("/image/{scan}/{img}")
    async def get_image(scan: str, img: str):
        path = Path(app.state.beamtime.path_raw) / scan / img
        if not path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        fig_dict = await run_in_threadpool(process_image, path)
        return JSONResponse(content=fig_dict)
    
    async def worker(ws, scan, session, data):
        # run the heavy lift in a thread, get a result back
        result = await asyncio.to_thread(
            app.state.beamtime.phase_retrieval_single_holo,
            scan, session,
            img_name=data["img_name"],
            find_focus=data.get("find_focus", False),
        )
        if result:
            fig = px.line(
                x=result.get("z01_values_history", []),
                y=result.get("loss_values_history", []),
                labels={"x": "z01", "y": "Loss"},
                title="Focus Optimization Results"
            )
            fig = fig.to_dict()
            ret = dict(
                data=fig, z01=result.get("z01"),)
            await ws.send_text(json.dumps(ret, cls=PlotlyJSONEncoder))


    @ui.websocket("/ws/preview")
    async def websocket_preview(ws: WebSocket):
        await ws.accept()
        app.state.beamtime.cluster.min_worker = 1
        session = str(uuid.uuid4())
        ctx = zmq.asyncio.Context.instance()
        sub = ctx.socket(zmq.SUB)
        sub.connect(f"tcp://{os.getenv('HNAME')}:{os.getenv('PUB_PORT', '6001')}")
        sub.setsockopt(zmq.SUBSCRIBE, session.encode())
        forward = asyncio.create_task(_forward_zmq(ws, sub))
        cfg = app.state.cfg
        try:
            while True:
                data = await ws.receive_json()
                form = dict_from_form(data.get("form_data", {}))
                cfg.reconstruction = form
                cfg.find_focus = form
                cfg.paths.base_dir = data.get("base_dir", cfg.paths.base_dir)
                scan = Scan(
                    name=data["scan_name"], holder=0,
                    path_raw=app.state.beamtime.path_raw,
                    path_processed=app.state.beamtime.log_path,
                    z01_new=data["z01"], energy=data.get("energy"),
                    cfg=cfg, a0=data["a0"],
                    log_path=app.state.beamtime.log_path
                )
                asyncio.create_task(worker(ws, scan, session, data))
                
        except WebSocketDisconnect:
            forward.cancel()
            sub.close()

        app.state.beamtime.cluster.min_worker = 0

    async def _forward_zmq(ws: WebSocket, sock: zmq.asyncio.Socket):
        try:
            while True:
                _, frame = await sock.recv_multipart()
                await ws.send_bytes(frame)
        except WebSocketDisconnect:
            pass
    
    app.include_router(api)
    app.include_router(ui)
