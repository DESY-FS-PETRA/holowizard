from typing import Dict, Any
import numpy as np
import torch
import torchvision.transforms as ttf
import xraylib

# local libraries
from holowizard.forge.objects.phantom import Phantom
from holowizard.forge.objects.physical_properties import PhysicalProperties
from holowizard.forge.objects.placement import ObjectPlacement
import holowizard.forge.objects.shape_sampler as module_sampler
import holowizard.forge.utils.random as random
from holowizard.forge.utils.torch_settings import get_torch_device
from holowizard.forge.utils.utilities import randint_from_range
from holowizard.forge.configs.parse_config import ConfigParser
import holowizard.forge.experiment as experiment


__all__ = [
    "PhantomGenerator",
]


class PhantomGenerator:
    def __init__(self, config: ConfigParser) -> None:
        """Generate complex-valued phantoms.

        Args:
            config (ConfigParser): Parse configuration file.
        """
        # NOTE: Implemented in this way, so only the config needs to be passed to the constructor, but the remaining
        #       configuration-structure stays the same
        cfg_phantom_generator: Dict[str, Any] = config["phantom_generator"]["args"]

        self.position = cfg_phantom_generator["position"]
        self.materials = cfg_phantom_generator["materials"]
        self.num_shapes_min = cfg_phantom_generator["num_shapes_min"]
        self.num_shapes_max = cfg_phantom_generator["num_shapes_max"]
        self.thickness_min = cfg_phantom_generator["thickness_min"]
        self.thickness_max = cfg_phantom_generator["thickness_max"]

        self.absorption_min = cfg_phantom_generator.get("absorption_min", 0)
        self.absorption_max = cfg_phantom_generator.get("absorption_max", float("inf"))
        self.phaseshift_min = cfg_phantom_generator.get("phaseshift_min", -float("inf"))
        self.phaseshift_max = cfg_phantom_generator.get("phaseshift_max", 0)
        self.material_dict = {}

        self.size = experiment.GLOBAL_EXP_SETUP.detector_size
        self.shape_sampler: module_sampler.ShapeSampler = config.init_obj("shape_sampler", module_sampler)
        self.object_placer = ObjectPlacement(
            positioning=self.position,
            detector_size=experiment.GLOBAL_EXP_SETUP.detector_size,
            probe_size=experiment.GLOBAL_EXP_SETUP.probe_size,
        )

        if cfg_phantom_generator["smoothing_filter"]["type"] == "Gaussian":
            self.phantom_smoothing_filter = ttf.GaussianBlur(
                cfg_phantom_generator["smoothing_filter"]["args"]["kernel_size"],
                cfg_phantom_generator["smoothing_filter"]["args"]["sigma"],
            )
        else:
            self.phantom_smoothing_filter = None

        self.last_phantom = None

    def get_phantom_size(self):
        return self.size

    def smooth_phantom(self, phantom: Phantom):
        # Smoothing the amplitudes/intensities of the phantom. For this reason, the smoothing
        # has to be done in exponential space.

        absorption_exp = torch.exp(-phantom.phantom.imag)
        phaseshift_exp = torch.exp(phantom.phantom.real)

        absorption_exp = self.phantom_smoothing_filter(absorption_exp[None, None, :, :])[0, 0, :, :]
        phaseshift_exp = self.phantom_smoothing_filter(phaseshift_exp[None, None, :, :])[0, 0, :, :]

        phantom.phantom = torch.log(phaseshift_exp) - 1j * torch.log(absorption_exp)

        return phantom

    def create_phantom(self) -> Phantom:
        phantom_size = self.size
        composed_phantom = Phantom(
            torch.zeros((phantom_size, phantom_size), dtype=torch.cfloat, device=get_torch_device()), phantom_size
        )
        num_objects = randint_from_range((self.num_shapes_min, self.num_shapes_max))
        for _ in range(num_objects):
            phys_props = self._create_physical_properties()
            shape = self.shape_sampler.get_shape()
            center = self.object_placer.get_pos(shape.shape)

            physical_object = (phys_props.phaseshift + 1j * phys_props.absorption) * shape.shape
            composed_phantom += Phantom(physical_object, phantom_size, center)

        composed_phantom = self.smooth_phantom(composed_phantom)

        composed_phantom.set_limits_absorption(min=self.absorption_min, max=self.absorption_max)
        composed_phantom.set_limits_phaseshift(min=self.phaseshift_min, max=self.phaseshift_max)
        self.last_phantom = composed_phantom

        return composed_phantom

    def _create_physical_properties(self) -> PhysicalProperties:
        thickness = randint_from_range((self.thickness_min, self.thickness_max))
        material = self._get_material()
        energy = experiment.GLOBAL_EXP_SETUP.energy

        if material not in self.material_dict:
            self.material_dict[material] = xraylib.Refractive_Index(
                material, energy, xraylib.ElementDensity(xraylib.SymbolToAtomicNumber(material))
            )

        ref_index = self.material_dict[material]
        delta = 1 - ref_index.real
        beta = ref_index.imag

        lam = 1.2398 / energy
        k = 2 * np.pi / lam
        absorption = 2 * beta * k * thickness * 1e3
        phaseshift = -k * delta * thickness * 1e3

        return PhysicalProperties(
            material=material,
            energy=energy,
            thickness=thickness,
            delta=delta,
            beta=beta,
            absorption=absorption,
            phaseshift=phaseshift,
        )

    def _get_material(self) -> str:
        idx = random.get_random_idx(self.materials)
        material = self.materials[idx]
        return material

    @property
    def last_phantom(self) -> Phantom:
        return self._last_phantom

    @last_phantom.setter
    def last_phantom(self, value: Phantom) -> None:
        if value is not None and not isinstance(value, Phantom):
            raise AttributeError(f"Expecting `Phantom`-type, not {type(value)}.")
        self._last_phantom = value
