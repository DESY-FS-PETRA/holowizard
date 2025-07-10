# standard libraries
from typing import Tuple

# third party libraries
import numpy as np
import torch

# local libraries
from holowizard.forge.utils.torch_settings import get_torch_device
from holowizard.forge.utils.datatypes import TensorFloat32
from holowizard.forge.utils.noise import Noise
from .probe import Probe
from .beam_config import BeamConfig


__all__ = [
    "PolynomialProbe",
]


class PolynomialProbe(Probe):
    def __init__(self, beam_cfg: BeamConfig, size: int, center_beam: bool, noise: Noise) -> None:
        probe = self._create_probe(beam_cfg, size, center_beam=center_beam, noise=noise)
        super().__init__(probe=probe, size=size, beam_cfg=beam_cfg)

    def __str__(self) -> str:
        """Creates a representation when printing an object with `print()` of this class.

        Returns:
            str: Background illumination as function.
        """
        xx = f"{self.beam_cfg.xx:1.1e}*x^2 + " if self.beam_cfg.xx else ""
        xx = f"{self.beam_cfg.xy:1.1e}*x*y + " if self.beam_cfg.xy else ""
        yy = f"{self.beam_cfg.yy:1.1e}*y^2 + " if self.beam_cfg.yy else ""
        x = f"{self.beam_cfg.x:1.1e}*x + " if self.beam_cfg.x else ""
        y = f"{self.beam_cfg.y:1.1e}*y + " if self.beam_cfg.y else ""
        return f"I = {xx}{yy}{x}{y}{self.beam_cfg.C}"

    def _create_probe(self, beam_cfg: BeamConfig, size: int, center_beam: bool, noise: Noise) -> TensorFloat32:
        """Create the beam that is directed onto the object."""
        probe = torch.ones((size, size), dtype=torch.float32) * beam_cfg.C

        def add_probe_component(coeff: float, *, pow_x: int, pow_y: int) -> TensorFloat32:
            if coeff == 0:
                return 0
            c_x, c_y = self._get_origin(size, center_beam)
            coeff /= size / 2
            vec_x = torch.arange(size) - c_x
            vec_y = torch.arange(size) - c_y

            beam_component = coeff * torch.outer(vec_x**pow_x, vec_y**pow_y)
            return beam_component

        # if conditions for performance reasons
        probe += add_probe_component(beam_cfg.x, pow_x=1, pow_y=0)
        probe += add_probe_component(beam_cfg.y, pow_x=0, pow_y=1)
        probe += add_probe_component(beam_cfg.xx, pow_x=2, pow_y=0)
        probe += add_probe_component(beam_cfg.xy, pow_x=1, pow_y=1)
        probe += add_probe_component(beam_cfg.yy, pow_x=0, pow_y=2)
        if noise is not None:
            probe += noise.get_noise()  # add noise
        probe[probe < 0] = 0  # remove negative values
        return probe.to(get_torch_device())

    def _get_origin(self, size: int, center_beam: bool) -> Tuple[int, int]:
        """'Generate' origin that deviates from the actual center of the hologram/detector

        Returns:
            Tuple[int, int]: Center-coordinates that will be used as origin, when creating the beam.
        """
        beam_center_px = int(size / 2) - 1
        if not center_beam:
            return (beam_center_px, beam_center_px)

        c_x, c_y = np.random.normal(beam_center_px, beam_center_px / 20, 2)
        return round(c_x), round(c_y)
