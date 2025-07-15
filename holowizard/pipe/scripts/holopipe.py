import sys
import uvicorn
from hydra import initialize_config_dir, compose
import os
from holowizard.pipe.server.app import create_app
from holowizard.pipe.beamtime import P05Beamtime
import socket
from pathlib import Path
from omegaconf import OmegaConf
import shutil
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # directory where main.py lives
    config_dir = os.path.join(base_dir, "config")

    with initialize_config_dir(version_base="1.2", config_dir=config_dir):
        overrides = sys.argv[1:]
        cfg = compose(config_name="defaults", overrides=overrides)
        destination_dir = f"/asap3/petra3/gpfs/p05/{cfg.beamtime.year}/data/{cfg.beamtime.name}/processed/holowizard_config"
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir, exist_ok=True)
        shutil.copytree(config_dir, destination_dir, dirs_exist_ok=True) 
        app = create_app(cfg, destination_dir)

        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        uvicorn.run(app, host=ip_address, port=8000)

if __name__ == "__main__":
    main()