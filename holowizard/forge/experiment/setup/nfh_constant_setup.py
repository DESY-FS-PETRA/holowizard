# standard libraries
from typing import Tuple, Dict, Any

# third party libraries

# local packages
from .nfh_setup import NFHSetup
from holowizard.forge.utils.datatypes import TensorComplex64
from holowizard.forge.utils import calc_Fr


__all__ = [
    "NFHConstantDistSetup",
]


class NFHConstantDistSetup(NFHSetup):
    _instance = None  # implemented as singleton

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        detector_size: int,
        detector_px_size: int,
        padding_factor: int,
        downsample_factor: int,
        energy: int,
        z01: float,
        z02: float,
    ):
        """Setup parameters for a Near-Field Holography simulation with a FZP and constant distances z01 and z02.

        Args:
            detector_size (int): Number of pixels of detector.
            detector_px_size (int): Pixel size of the detector (in [nm]).
            padding_factor (int): Padding factor used in the forward simulation.
            downsampling_factor (int): Padding factor used in the forward simulation.
            energy (int): The beam's energy (in [eV]).
            z01 (float): The distance from the OSA to the object (in [cm]). Constant in this setup.
            z02 (float): The distance from the OSA to the detector (in [m]). Constant in this setup.
        """
        super().__init__(
            detector_size=detector_size,
            detector_px_size=detector_px_size,
            energy=energy,
            padding_factor=padding_factor,
            downsample_factor=downsample_factor,
        )
        # class properties specific for constant setup case
        self._z01 = z01
        self._z02 = z02
        self._Fr = calc_Fr(energy, z01, z02, detector_px_size * downsample_factor)
        self._kernel = self.create_kernel(self.Fr)

    def get_distances(self) -> Tuple[float, float]:
        return self.z01, self.z02

    def as_dict(self) -> Dict[str, Any]:
        d = super().as_dict() | dict(z01=self.z01, z02=self.z02, Fr=self.Fr)
        return d

    @property
    def kernel(self) -> TensorComplex64:
        return self._kernel

    @property
    def z01(self) -> float:
        return self._z01

    @property
    def z02(self) -> float:
        return self._z02

    @property
    def Fr(self) -> float:
        return self._Fr
