# standard libraries

# third party libraries
import torch

# local libraries
from holowizard.forge.utils.datatypes import TensorFloat32, Pathlike, is_pathlike
from holowizard.forge.utils.torch_settings import get_torch_device
from holowizard.forge.utils import fileIO
from .beam_config import BeamConfig


__all__ = [
    "Probe",
]


class Probe:
    def __init__(self, probe: TensorFloat32 | Pathlike, size: int, beam_cfg: BeamConfig = None) -> None:
        if is_pathlike(probe):
            probe = torch.Tensor(fileIO.load_img(probe))
        elif type(probe) == TensorFloat32:
            pass
        else:
            raise ValueError

        self.probe = probe.to(device=get_torch_device())
        self.size = size
        self.shape = (size, size)
        self.beam_cfg = beam_cfg

    def __add__(self, other):
        if isinstance(other, Probe):
            new_probe = self.probe + other.probe
            return Probe(new_probe, self.size)
        elif isinstance(other, TensorFloat32):
            return self.probe + other

        else:
            raise ValueError("Unsupported operand type for '+'.")

    def __iadd__(self, other):
        if isinstance(other, Probe):
            self.probe += other.probe
            return self
        elif isinstance(other, TensorFloat32):
            self.probe += other
            return self
        else:
            raise ValueError("Unsupported operand type for '+='.")

    def __mul__(self, other):
        if isinstance(other, Probe):
            new_probe = self.probe * other.probe
            return Probe(new_probe, self.size)
        elif isinstance(other, TensorFloat32):
            return self.probe * other

        else:
            raise ValueError("Unsupported operand type for '*'.")

    def __imul__(self, other):
        if isinstance(other, Probe):
            self.probe *= other.probe
            return self
        elif isinstance(other, TensorFloat32):
            self.probe *= other
            return self
        else:
            raise ValueError("Unsupported operand type for '*='.")
