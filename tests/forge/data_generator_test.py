# standard libraries

# third party libraries

# local libraries
from holoforge.generators import DataGenerator


def data_generator_test() -> None:
    # alternatively using the CLI
    # python generate_data.py holoforge/tests/output 3 --energy 11000 --override --config configs/custom/test_config.json
    config_path = '../holoforge/configs/test_config.json'
    energies = [11000, 17000]
    output = '../output/tests/data_generation_test'
    num_examples = 100
    for energy in energies:
        data_generator = DataGenerator(output, energy, config_path=config_path, override=True)
        data_generator.generate_data(num_examples)

data_generator_test()