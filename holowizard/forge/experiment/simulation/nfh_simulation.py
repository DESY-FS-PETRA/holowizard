# standard libraries

# third party libraries
import torch
from torch import fft

# local libraries
from holowizard.forge.utils.utilities import crop_center
from holowizard.forge.utils.datatypes import TensorComplex64, TensorFloat32
from holowizard.forge.objects.phantom import Phantom
from holowizard.forge.experiment.flatfield.flatfield import FlatField

from .simulation import Simulation
from holowizard.forge.experiment.probe import Probe
from holowizard.forge.experiment.setup import NFHSetup


__all__ = [
    "NFHSimulation",
]


class NFHSimulation(Simulation):
    def __init__(self, setup: NFHSetup):
        self.setup = setup

    def forward(self, phantom: Phantom, probe: Probe, flatfield: FlatField | None = None) -> TensorFloat32:
        """Implements the forward model and creates a hologram for this simulation`s experiment setup.

        Args:
            phantom (Phantom): Object that is illuminated by the X-ray defined by its physical properties.
            probe (Probe): X-ray illuminating the object.
            flatfield (Flatfield, optional): Multiply hologram with this flatfield, if `flatfield` is not None.
                        Defaults to None.

        Returns:
            TensorFloat32: The (real-valued) hologram.
        """
        # TODO: might need to adjust forward pass test, since self.holo does not exist anymore
        self.probe = probe
        self.flatfield = flatfield
        self.psi_det = self._wave_field(phantom, probe)
        holo = self._make_hologram(self.psi_det, flatfield=flatfield)
        return holo

    def _wave_field(self, phantom: Phantom, probe: Probe) -> TensorComplex64:
        """Calculates the wavefield from the object to the detector.

        Returns:
            torch.Tensor: The wavefield at the detector (complex numbers).
        """
        padded_obj = phantom.pad_to(probe.size)
        self.obj = torch.exp(1j * padded_obj)
        self.psi_exit = probe * self.obj
        psi_det = fft.ifft2(fft.fft2(self.psi_exit) * self.setup.kernel)
        return psi_det

    def _make_hologram(self, psi_det: TensorComplex64, flatfield: FlatField = None) -> TensorFloat32:
        """Creates the hologram from the wavefield at the detector (just absolutes of complex values).

        Args:
            psi_det (TensorComplex64): Wavefield at detector.
            flatfield (FlatField, optional): Multiply the hologram by the given flatfield.

        Returns:
            TensorFloat32: The squared magnitude of the wave field at the detector, i.e. hologram.
        """
        holo = torch.abs(psi_det)
        holo = crop_center(holo, (self.setup.detector_size, self.setup.detector_size))
        self.gt_hologram = holo.clone()
        if flatfield is not None:
            holo *= flatfield.flatfield
        return holo

    @property
    def gt_hologram(self) -> TensorFloat32:
        return self._gt_hologram

    @gt_hologram.setter
    def gt_hologram(self, val: TensorFloat32) -> None:
        if not isinstance(val, TensorFloat32):
            raise AttributeError(f"hologram must be of type `TensorFloat32`, not {type(val)}")
        self._gt_hologram = val
