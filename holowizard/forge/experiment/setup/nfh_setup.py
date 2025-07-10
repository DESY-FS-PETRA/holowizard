# standard libraries
from abc import abstractmethod
from typing import Tuple

# third party libraries
import numpy as np
import torch
import torch.fft as fft

# local packages
from .setup import Setup
import holowizard.forge.utils.torch_settings as torch_settings
from holowizard.forge.utils.datatypes import TensorComplex64, TensorFloat32
from holowizard.forge.utils import calc_Fr


__all__ = [
    "NFHSetup",
]


class NFHSetup(Setup):
    def __init__(
        self,
        detector_size: int,
        detector_px_size: int,
        padding_factor: int,
        downsample_factor: int,
        energy: int,
    ):
        """Baseclass for all NFH Setups

        Args:
            detector_size (int): Number of pixels of detector.
            detector_px_size (int): Pixel size of the detector (in [nm]).
            padding_factor (int): Padding factor used in the forward simulation.
            downsampling_factor (int): Padding factor used in the forward simulation.
            energy (int): The beam's energy (in [eV]).
        """
        super().__init__(
            detector_size=detector_size,
            detector_px_size=detector_px_size,
            padding_factor=padding_factor,
            downsample_factor=downsample_factor,
            energy=energy,
        )

    @abstractmethod
    def get_distances(self, z01: float | Tuple[float, float], z02: float | Tuple[float, float]) -> Tuple[float, float]:
        """Will be called in the `self.kernel`-getter method to dynamically calculate the kernel.

        Put the random-logic in this method.
        Args:
            z01 (float | Tuple[float, float]): _description_
            z02 (float | Tuple[float, float]): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Tuple[float, float]: _description_
        """
        raise NotImplementedError("Method needs to be implemented in child class.")

    def create_kernel(self, Fr: float) -> TensorComplex64:
        sample_grid = torch.meshgrid(
            fft.fftfreq(
                self.probe_size,
                device=torch_settings.get_torch_device(),
                dtype=torch.float,
            ),
            fft.fftfreq(
                self.probe_size,
                device=torch_settings.get_torch_device(),
                dtype=torch.float,
            ),
            indexing="ij",
        )
        xi, eta = sample_grid
        kernel = torch.exp((-1j * np.pi) / Fr * (xi * xi + eta * eta)).type(torch.cfloat)
        return kernel

    @property
    def kernel(self) -> TensorComplex64:
        z01, z02 = self.get_distances()
        Fr = calc_Fr(self.energy, z01, z02, self.detector_px_size)
        return self.create_kernel(Fr=Fr)
