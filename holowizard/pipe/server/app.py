import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Union
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Import for side effects: initializes server env vars and writes .env.
import holowizard.pipe.server.config as config  # noqa: F401
import zmq
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from holowizard.core.src.hpc.slurm_cluster import SlurmCluster as Cluster
from holowizard.core.src.parameters.dask_options import DaskOptions
from holowizard.pipe.beamtime import P05Beamtime as Beamtime
from holowizard.pipe.utils.clean_yaml import to_clean_yaml
from holowizard.pipe.server.cluster_manager import OnDemandCluster
from holowizard.pipe.server.auth import LoginRequiredMiddleware, UserStore, ensure_default_user, users_file
from holowizard.pipe.server.routes import dashboard, login, neuroglancer, overview, parameter, rotation, scan, tomography


BASE_DIR = Path(__file__).resolve().parent
# Shipped configs; copied into the beamtime's own config folder on startup.
CONFIG_SOURCE = BASE_DIR.parent / "scripts" / "config"
# Configs that are only needed to start the server, not offered in the UI.
STARTUP_ONLY_CONFIGS = ("dask_options.yaml", "beamtime.yaml", "server.yaml")


def _sync_config_dir(source: Path, destination: Path):
    """Refresh the shipped configs, keeping files created from the UI (custom and saved parameter sets)."""
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    for name in STARTUP_ONLY_CONFIGS:
        (destination / name).unlink(missing_ok=True)


# --- Lifespan & App Factory ---
@asynccontextmanager
async def _lifespan(app: FastAPI):
    source = app.state.config_source
    cfg = getattr(app.state, "_initial_cfg", None)
    if cfg is None:
        with initialize_config_dir(version_base="1.2", config_dir=str(source)):
            cfg = compose(config_name="defaults")
    app.state.beamtime = Beamtime(beamtime_name=cfg.beamtime.id, year=cfg.beamtime.year, cluster=None)
    app.state.config_dir = app.state.beamtime.log_path / cfg.paths.pipe_config
    _sync_config_dir(source, app.state.config_dir)
    dask_dict = OmegaConf.to_container(cfg["dask_options"], resolve=True)
    dask_options = DaskOptions.from_dict(dask_dict)
    if dask_options is None:
        raise RuntimeError("Could not create dask options from hydra config")
    dask_options.working_dir = app.state.beamtime.log_path / dask_options.working_dir
    dask_options.python_env = sys.prefix
    app.state.beamtime.cluster = Cluster(dask_options=dask_options)
    # SLURM workers are started when work arrives and stopped again when idle.
    idle_minutes = OmegaConf.select(cfg, "server.cluster_idle_minutes", default=10)
    app.state.cluster = OnDemandCluster(app.state.beamtime.cluster, idle_seconds=float(idle_minutes) * 60)

    app.state.cfg = cfg
    templates = Jinja2Templates(directory=str(BASE_DIR / ".." / "templates"))
    templates.env.filters["to_yaml"] = to_clean_yaml
    broker_thread = threading.Thread(target=run_broker, daemon=True)
    broker_thread.start()
    app.state.templates = templates
    yield
    app.state.cluster.shutdown()
    del app.state.beamtime


def run_broker():
    ctx = zmq.Context.instance()
    xsub = ctx.socket(zmq.XSUB)
    xsub.bind(f"tcp://0.0.0.0:{os.getenv('SUB_PORT', '6000')}")
    xpub = ctx.socket(zmq.XPUB)
    xpub.bind(f"tcp://0.0.0.0:{os.getenv('PUB_PORT', '6001')}")
    zmq.proxy(xsub, xpub)


# create_app factory
def create_app(cfg=None, config_source: Union[str, Path] = None) -> FastAPI:
    """Create the server app.

    ``config_source`` is the folder with the shipped configs (default: ``scripts/config``). On startup it is
    copied into ``<beamtime log_path>/<paths.pipe_config>``, which is the folder the UI reads and writes.
    """
    app = FastAPI(lifespan=_lifespan)
    app.state.config_source = Path(config_source) if config_source else CONFIG_SOURCE
    if cfg:
        app.state._initial_cfg = cfg

    users = UserStore(users_file())
    ensure_default_user(users)
    app.state.users = users
    app.add_middleware(LoginRequiredMiddleware, store=users)

    @app.get("/", tags=["ui"])
    async def home(request: Request):
        return RedirectResponse(url=request.url_for("overview_page"))

    @app.get("/api/config", tags=["api"])
    async def get_config(request: Request):
        return OmegaConf.to_container(request.app.state.cfg, resolve=True)

    for module in (login, overview, dashboard, parameter, rotation, tomography, neuroglancer, scan):
        app.include_router(module.router)

    app.mount("/static", StaticFiles(directory=str(BASE_DIR / ".." / "templates/static")), name="static")

    return app
