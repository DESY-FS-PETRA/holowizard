import os
import sys
import threading
from pathlib import Path
from typing import Union
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
from holowizard.pipe.server.auth import LoginRequiredMiddleware, UserStore, ensure_default_user, users_file
from holowizard.pipe.server.routes import dashboard, login, overview, parameter, scan


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "configs"


# --- Lifespan & App Factory ---
@asynccontextmanager
async def _lifespan(app: FastAPI):
    cfg = getattr(app.state, "_initial_cfg", OmegaConf.load(str(app.state.config_dir / "defaults.yaml")))
    app.state.beamtime = Beamtime(beamtime_name=cfg.beamtime.id, year=cfg.beamtime.year, cluster=None)
    dask_dict = OmegaConf.to_container(cfg["dask_options"], resolve=True)
    dask_options = DaskOptions.from_dict(dask_dict)
    if dask_options is None:
        raise RuntimeError("Could not create dask options from hydra config")
    dask_options.working_dir = app.state.beamtime.log_path / dask_options.working_dir
    dask_options.python_env = sys.prefix
    app.state.beamtime.cluster = Cluster(dask_options=dask_options)
    app.state.beamtime.cluster.start_cluster()

    app.state.cfg = cfg
    templates = Jinja2Templates(directory=str(BASE_DIR / ".." / "templates"))
    templates.env.filters["to_yaml"] = to_clean_yaml
    broker_thread = threading.Thread(target=run_broker, daemon=True)
    broker_thread.start()
    app.state.templates = templates
    yield
    del app.state.beamtime


def run_broker():
    ctx = zmq.Context.instance()
    xsub = ctx.socket(zmq.XSUB)
    xsub.bind(f"tcp://0.0.0.0:{os.getenv('SUB_PORT', '6000')}")
    xpub = ctx.socket(zmq.XPUB)
    xpub.bind(f"tcp://0.0.0.0:{os.getenv('PUB_PORT', '6001')}")
    zmq.proxy(xsub, xpub)


# create_app factory
def create_app(cfg=None, config_dir: Union[str, Path] = None) -> FastAPI:
    app = FastAPI(lifespan=_lifespan)
    app.state.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
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

    for module in (login, overview, dashboard, parameter, scan):
        app.include_router(module.router)

    app.mount("/static", StaticFiles(directory=str(BASE_DIR / ".." / "templates/static")), name="static")

    return app
