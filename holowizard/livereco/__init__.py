import sys

server_port = 8555
viewer_port = 8556

try:
    import torch
except Exception:
    pass

if "torch" in sys.modules:
    import subprocess

    torch_running_device_name = "cpu"

    try:
        subprocess.check_output("nvidia-smi")
        torch_running_device_name = "cuda:0"
        import cupy

        cupy.cuda.Device(0).use()
    except Exception:
        pass

    torch_running_device = torch.device(torch_running_device_name)
