# standard libraries
from pathlib import Path

# third party libraries
from tqdm import tqdm

# local libraries
import holowizard.forge.utils.labeller as module_labeller
import holowizard.forge.experiment as experiment
from holowizard.forge.utils.utilities import crop_center
from holowizard.forge.configs.parse_config import ConfigParser

from .hologram_generator import HologramGenerator
from .phantom_generator import PhantomGenerator
from .probe_generator import ProbeGenerator
from .flatfield_generator import BaseFlatFieldGenerator


__all__ = [
    "DataGenerator",
]


class DataGenerator:
    def __init__(
        self,
        output: str,
        num_samples: int,
        config: ConfigParser,
        hologram_generator: HologramGenerator,
        phantom_generator: PhantomGenerator,
        probe_generator: ProbeGenerator,
        flatfield_generator: BaseFlatFieldGenerator,
        override: bool = False,
    ) -> None:
        """Initialize training data generator.

        Args:
            output (str): Base output folder, where the generated data should be stored.
            TODO: docs

        """
        self.output = Path(output)
        self.num_samples = num_samples
        self.config = config

        self.hologram_generator = hologram_generator
        self.phantom_generator = phantom_generator
        self.probe_generator = probe_generator
        self.flatfield_generator = flatfield_generator

        self.cfg_data_generator = self.config["data_generator"]["args"]

        self.labeller = self._get_labeller(
            num_samples=self.num_samples,
            dataset_name=self.config["name"],
            override=override,
        )
        self.config.save(self.output / f"{config['name']}.json")

    def generate_data(self, num_samples: int = None) -> None:
        """Generates and stores `self.num_samples` holograms, corresponding labels and the metadata.

        Args:
            num_samples (int): Number of data examples that will be generated.
        """
        if num_samples is not None:
            self.labeller.update_dataset_size(num_samples)
            self.num_samples = num_samples

        for idx in tqdm(range(self.num_samples)):
            phantom = self.phantom_generator.create_phantom()
            probe = self.probe_generator.create_probe()
            flatfield = self.flatfield_generator.create_flatfield()
            holo = self.hologram_generator.create_hologram(phantom=phantom, probe=probe, flatfield=flatfield)
            raw_holo = self.hologram_generator.gt_hologram
            cropped_probe = (
                crop_center(probe.probe, holo.shape) if self.cfg_data_generator["crop_probe"] else probe.probe
            )

            beam_cfg = probe.beam_cfg
            setup = self.hologram_generator.simulation.setup
            self.labeller.cache(
                hologram=holo.cpu(),
                gt_hologram=raw_holo.cpu(),
                phantom=phantom.refractive_index.cpu(),
                probe=cropped_probe.cpu(),
                flatfield=(flatfield.flatfield.cpu() if flatfield is not None else None),
                condition=(
                    flatfield.condition.cpu() if flatfield is not None and flatfield.condition is not None else None
                ),
                beam_cfg=beam_cfg,
                setup=setup,
            )

        self.labeller.flush()  # explicitly flush the remaining cached elements

    def _get_labeller(self, num_samples: int, dataset_name: str, override: bool) -> module_labeller.Labeller:
        detector_size = experiment.GLOBAL_EXP_SETUP.detector_size
        probe_target_size = (
            detector_size if self.cfg_data_generator["crop_probe"] else experiment.GLOBAL_EXP_SETUP.probe_size
        )

        labeller: module_labeller.Labeller = self.config.init_obj(
            "labeller",
            module_labeller,
            output=self.output,
            num_samples=num_samples,
            detector_size=detector_size,
            probe_size=probe_target_size,
            dataset_name=dataset_name,
            override=override,
        )
        return labeller
