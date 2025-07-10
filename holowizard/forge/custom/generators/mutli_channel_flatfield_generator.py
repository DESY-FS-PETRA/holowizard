# standard libraries
from typing import Callable

# third party libraries

# local libraries
import holowizard.forge.datasets as module_data
import holowizard.forge.utils.random as random
from holowizard.forge.experiment.flatfield import MultiChannelFlatField
from holowizard.forge.configs.parse_config import ConfigParser
from holowizard.forge.generators import BaseFlatFieldGenerator


__all__ = [
    "MultiChannelFlatFieldGenerator",
]


class MultiChannelFlatFieldGenerator(BaseFlatFieldGenerator):
    def __init__(self, config: ConfigParser) -> None:
        super().__init__(config, constructor=self._get_constructor(config))

    def _get_constructor(self, config: ConfigParser) -> Callable[[], MultiChannelFlatField]:
        if config.get("flatfield_dataset", None) is None:
            self.dataset = None
            return lambda: None  # always return None

        self.dataset = config.init_obj("flatfield_dataset", module_data)

        def dataset_constructor() -> MultiChannelFlatField:
            idx = random.get_random_idx(self.dataset)
            flatfield, conditions = self.dataset[idx]
            return MultiChannelFlatField(flatfield, conditions)

        return dataset_constructor
