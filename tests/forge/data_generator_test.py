import os
from pathlib import Path

from holowizard.forge.configs.parse_config import ConfigParser
from holowizard.forge.generators import (
    DataGenerator,
    HologramGenerator,
    ProbeGenerator,
    PhantomGenerator,
)
import holowizard.forge.generators as module_generators


def data_generator_test() -> None:
    # alternatively using the CLI
    # python generate_data.py holowizard.forge/tests/output 3 --energy 11000 --override --config configs/custom/test_config.json
    root = Path(__file__).parent.resolve()
    print("Root Path: ", root)
    output = root / "test_output/data_generation_test"

    config = ConfigParser(root / "../../holowizard/forge/configs/test_config.json")
    output_file = output / (config["name"] + ".hdf5")

    num_samples = 10
    hologram_generator = HologramGenerator(config)
    phantom_generator = PhantomGenerator(config)
    probe_generator = ProbeGenerator(config)
    flatfield_generator = config.init_obj("flatfield_generator", module_generators, config=config)

    data_generator = DataGenerator(
        output=str(output),
        num_samples=num_samples,
        config=config,
        override=True,
        hologram_generator=hologram_generator,
        phantom_generator=phantom_generator,
        probe_generator=probe_generator,
        flatfield_generator=flatfield_generator,
    )

    try:
        os.remove(output_file)
    except:
        pass
    assert not output_file.exists()
    data_generator.generate_data()
    assert output_file.exists()
    os.remove(output_file)


data_generator_test()
