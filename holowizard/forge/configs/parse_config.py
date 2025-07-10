# standard libraries
from typing import Any, Dict
from types import ModuleType
from pathlib import Path

# third party libraries

# local packages
import holowizard.forge.experiment.setup as module_setup
import holowizard.forge.experiment as experiment
from holowizard.forge.utils.datatypes import Pathlike
import holowizard.forge.utils.fileIO as fileIO
from holowizard.forge.configs import DEFAULT_CONFIG_PATH


__all__ = [
    "ConfigParser",
    "init_obj",
]


class ConfigParser:
    def __init__(self, config_path: Pathlike | None):
        """Parse json-configuration file."""
        self._config = self.create_config_dict(config_path)
        setup = self.init_obj("setup", module_setup)
        experiment.initialize_global_exp_setup(setup)

    def init_obj(self, name: str, module: ModuleType, *args, **kwargs) -> Any:
        """Initialize an object from a module based on its name.

        Finds a function handle with the name given as 'type' in `self.config`, and returns the
        instance initialized with corresponding arguments given. It first looks in the corresponding module and if the
        class was not found in it, it looks into the custom-package, were the definition might be found.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`

        Args:
            name (str): Name of the key in the config.
            module (ModuleType): Module, in which the class definition lies.

        Returns:
            Any: The initialized object.
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        return init_obj(module_name, module, module_args, *args, **kwargs)

    def __getitem__(self, name: Any):
        """Access items like ordinary dict."""
        return self.config[name]

    def get(self, key: Any, default: Any) -> Any:
        return self.config.get(key, default)

    def create_config_dict(self, config_path: Path | None) -> Dict[str, Any]:
        """Merge the custom
        Args:
            config_path (Path | None): The path to the file containing the custom configuration settings. If
                        `config_path` is `None` the default settings are used.

        Returns:
            Dict[str, Any]: The merged configuration dictionary.
        """
        default_config = fileIO.read_json(DEFAULT_CONFIG_PATH)
        if config_path is None:
            return default_config

        config_path = config_path.with_suffix(".json")
        custom_config = fileIO.read_json(config_path)

        return default_config | custom_config

    def save(self, path: Pathlike) -> None:
        fileIO.write_json(self.config, path)

    # setting read-only attributes
    @property
    def config(self) -> Dict[str, Any]:
        return self._config


def init_obj(module_name: str, module: ModuleType, module_args: Dict[str, Any], *args, **kwargs) -> Any:
    assert all([k not in module_args for k in kwargs]), "Overwriting kwargs given in config file is not allowed"
    module_args.update(kwargs)
    try:
        return getattr(module, module_name)(*args, **module_args)
    except AttributeError:
        import holowizard.forge.custom as module_custom

        return getattr(module_custom, module_name)(*args, **module_args)
