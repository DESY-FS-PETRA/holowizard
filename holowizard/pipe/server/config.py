import os
import socket
from pathlib import Path


def find_free_port(host: str = "127.0.0.1") -> int:
    # Bind to port 0 tells the OS to pick an unused port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def first_free_port(host: str, port: int, attempts: int = 1) -> int:
    """First port in ``port .. port + attempts - 1`` that can be bound on ``host``."""
    for candidate in range(port, port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Same option as uvicorn, so ports in TIME_WAIT count as free, but listening ones do not.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
            except OSError:
                continue
            return candidate
    raise RuntimeError(f"No free port in {port}-{port + attempts - 1} on {host}")


# --- Environment & Config ---
HOSTNAME = socket.gethostname()
sub_port = find_free_port()
pub_port = find_free_port()
os.environ.setdefault("HNAME", HOSTNAME)
os.environ.setdefault("SUB_PORT", str(sub_port))
os.environ.setdefault("PUB_PORT", str(pub_port))
env_path = Path(".") / ".env"

with env_path.open("w") as f:
    f.write(f"HNAME={HOSTNAME}\n")
    f.write(f"SUB_PORT={sub_port}\n")
    f.write(f"PUB_PORT={pub_port}\n")
