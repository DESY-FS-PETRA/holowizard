import torch

import tempfile
import subprocess

tempfile.tempdir = "/tmp/"
current_logger = None

torch_running_device_name = "cpu"

try:
    subprocess.check_output("nvidia-smi")
    torch_running_device_name = "cuda:0"
    import cupy

    cupy.cuda.Device(0).use()
except Exception:
    pass

torch_running_device = torch.device(torch_running_device_name)

