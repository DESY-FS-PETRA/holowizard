# standard libraries
from pathlib import Path

# third party libraries

# local libraries


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "TEST_CONFIG_PATH",
]


DEFAULT_CONFIG_PATH = str(Path(__file__).parent.resolve() / "default.json")
TEST_CONFIG_PATH = str(Path(__file__).parent.resolve() / "test_config.json")
