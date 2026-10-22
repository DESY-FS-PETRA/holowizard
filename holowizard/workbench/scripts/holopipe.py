import sys
import uvicorn
from hydra import initialize_config_dir, compose
import os
from holowizard.workbench.server.app import create_app
from holowizard.workbench.server.config import first_free_port
import socket


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # directory where main.py lives
    config_dir = os.path.join(base_dir, "config")

    with initialize_config_dir(version_base="1.2", config_dir=config_dir):
        overrides = sys.argv[1:]
        cfg = compose(config_name="defaults", overrides=overrides)

        # The server copies config_dir into the beamtime's config folder on startup.
        app = create_app(cfg, config_dir)

        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        port = first_free_port(ip_address, cfg.server.port, cfg.server.port_attempts)
        if port != cfg.server.port:
            print(f"Port {cfg.server.port} is in use, using port {port} instead.", flush=True)
        print(f"Server: http://{hostname}:{port}/", flush=True)
        uvicorn.run(app, host=ip_address, port=port)


if __name__ == "__main__":
    main()
