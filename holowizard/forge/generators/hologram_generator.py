# standard libraries

# third party libraries

# local libraries
from holowizard.forge.utils.datatypes import TensorFloat32
from holowizard.forge.objects.phantom import Phantom
import holowizard.forge.experiment as experiment
from holowizard.forge.experiment.flatfield import FlatField
from holowizard.forge.experiment.probe import Probe
from holowizard.forge.experiment.simulation import Simulation
import holowizard.forge.experiment.simulation as module_simulation
import holowizard.forge.utils.noise as module_noise
from holowizard.forge.configs.parse_config import ConfigParser
from holowizard.forge.utils.torch_settings import get_torch_device


__all__ = [
    "HologramGenerator",
]


class HologramGenerator:
    """Generates holograms for a specific simulation setup given in the config."""

    def __init__(self, config: ConfigParser) -> None:
        self.config = config
        self.simulation: Simulation = self.config.init_obj(
            "simulation", module_simulation, setup=experiment.GLOBAL_EXP_SETUP
        )

        self.probe_size = experiment.GLOBAL_EXP_SETUP.probe_size
        self.holo_size = experiment.GLOBAL_EXP_SETUP.detector_size

        if self.config.get("hologram_noise", None) is not None:
            self.noise: module_noise.Noise = self.config.init_obj("hologram_noise", module_noise, size=self.holo_size)
        else:
            self.noise = None

    def get_holo_size(self) -> int:
        return self.holo_size

    def create_hologram(self, phantom: Phantom, probe: Probe, flatfield: FlatField = None) -> TensorFloat32:
        hologram = self.simulation(phantom, probe, flatfield)
        if self.noise is not None:
            hologram += self.noise.get_noise().to(get_torch_device())
        hologram[hologram < 0] = 0
        return hologram

    @property
    def gt_hologram(self) -> TensorFloat32:
        return self.simulation.gt_hologram

    # @property
    # def last_simulation_setup(self) -> SimulationSetup:
    #     return self._last_simulation_setup

    # @last_simulation_setup.setter
    # def last_simulation_setup(self, value: SimulationSetup) -> None:
    #     if not isinstance(value, SimulationSetup) and value is not None:
    #         raise AttributeError(f"Expecting SimulationSetup. Actual: {type(value)}")
    #     self._last_simulation_setup = value
