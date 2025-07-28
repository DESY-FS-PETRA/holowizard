# standard libraries
import argparse
from pathlib import Path

# third party libraries

# local libraries
from holowizard.forge.generators import (
    DataGenerator,
    HologramGenerator,
    ProbeGenerator,
    PhantomGenerator,
)
import holowizard.forge.generators as module_generators
from holowizard.forge.configs.parse_config import ConfigParser
import holowizard.forge.utils.torch_settings as torch_settings


def main():
    parser = argparse.ArgumentParser(
        description="Configuration how many holograms should be generated and where they will be stored."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to a json file containing custom parameters for the data generation process and the \
                            corresponing value generation.",
    )
    parser.add_argument("output_dir", type=str, help="Output folder where the generated data is stored.")
    parser.add_argument("num_samples", type=int, help="Number of data samples that should be generated.")
    parser.add_argument(
        "--override",
        action="store_true",
        help="Override the output folder if it already exists.",
    )
    parser.add_argument(
        "--seed",
        type=float,
        default=None,
        help="Set a custom seed for reproducibility.",
    )
    args = parser.parse_args()

    if seed := args.seed is not None:
        torch_settings.set_reproducibility(seed)

    config = ConfigParser(Path(args.config))

    hologram_generator = HologramGenerator(config)
    phantom_generator = PhantomGenerator(config)
    probe_generator = ProbeGenerator(config)
    flatfield_generator = config.init_obj("flatfield_generator", module_generators, config=config)

    data_generator = DataGenerator(
        output=args.output_dir,
        num_samples=args.num_samples,
        config=config,
        override=args.override,
        hologram_generator=hologram_generator,
        phantom_generator=phantom_generator,
        probe_generator=probe_generator,
        flatfield_generator=flatfield_generator,
    )
    data_generator.generate_data()


if __name__ == "__main__":
    main()
