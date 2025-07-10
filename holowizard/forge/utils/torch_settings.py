# standard libraries

# third party libraries
import torch
import numpy as np

# local libraries


__all__ = [
    "get_torch_device",
    "set_reproducibility",
]


def get_torch_device(device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def set_reproducibility(seed: int | float = 69) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
