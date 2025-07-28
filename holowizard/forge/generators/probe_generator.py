# standard libraries

# third party libraries

# local libraries
import holowizard.forge.utils.noise as module_noise
from holowizard.forge.experiment.probe.beam_config import get_random_beam_config
from holowizard.forge.experiment.probe import Probe, PolynomialProbe
from holowizard.forge.configs.parse_config import ConfigParser
import holowizard.forge.experiment as experiment


__all__ = [
    "ProbeGenerator",
]


class ProbeGenerator:
    """Generates polynomial probes."""

    def __init__(self, config: ConfigParser) -> None:
        # NOTE: Implemented in this way, so only the config needs to be passed to the constructor, but the remaining
        #       configuration-structure stays the same
        if config.get("probe_generator", None) is None:
            self.constant = 1
            self.linear = 0
            self.square = 0
            self.center_beam = True
        else:
            cfg_probe_generator = config["probe_generator"]["args"]
            self.constant = cfg_probe_generator["constant"]
            self.linear = cfg_probe_generator["linear"]
            self.square = cfg_probe_generator["square"]
            self.center_beam = cfg_probe_generator["center_beam"]
        self.probe_size = experiment.GLOBAL_EXP_SETUP.probe_size
        if config.get("probe_noise", None) is not None:
            self.probe_noise: module_noise.Noise = config.init_obj("probe_noise", module_noise, size=self.probe_size)
        else:
            self.probe_noise = None

    def create_probe(self) -> Probe:
        beam_cfg = get_random_beam_config(self.constant, self.linear, self.square)
        return PolynomialProbe(
            beam_cfg=beam_cfg,
            size=self.probe_size,
            center_beam=self.center_beam,
            noise=self.probe_noise,
        )
