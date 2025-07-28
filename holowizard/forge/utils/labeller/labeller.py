# standard libraries
from pathlib import Path
from abc import ABC, abstractmethod
import shutil

# third party libraries

# local libraries


__all__ = [
    "Labeller",
]


class Labeller(ABC):
    def __init__(
        self,
        output: Path,
        store_hologram: bool,
        store_gt_hologram: bool,
        store_phantom: bool,
        store_probe: bool,
        store_flatfield: bool,
        store_polynomial: bool,
        store_setup: bool,
        override: bool,
    ) -> None:
        """Writes generated data as hdf5 file.

        Args:
            output (Path): Directory or file where the dataset is written to.
            store_hologram (bool): Store holograms.
            store_gt_hologram (bool): Store the raw hologram without any noise or flatfield multiplication.
            store_phantom (bool): Store phantoms.
            store_probe (bool): Store background illumination.
            store_flatfield (bool): Store multiplicative flatfield.
            store_polynomial (bool): Store polynomial.
            store_setup (bool): Store simulation setup.
            override (bool): If True and the output exists, override it. Otherwise raise an exception.
        """
        self._output = output
        self._store_hologram = store_hologram
        self._store_gt_hologram = store_gt_hologram
        self._store_phantom = store_phantom
        self._store_probe = store_probe
        self._store_flatfield = store_flatfield
        self._store_polynomial = store_polynomial
        self._store_setup = store_setup
        self._override = override
        self._check_output()

    @abstractmethod
    def flush(self) -> None: ...

    @abstractmethod
    def cache(self, *args) -> None: ...

    @abstractmethod
    def update_dataset_size(self, new_size: int) -> None: ...

    def _check_output(self) -> None:
        if self.output.exists() and not self.override:
            raise FileExistsError(f'Output {self.output} already exists. Use "--override" to override it.')
        if self.output.is_dir():
            shutil.rmtree(self.output)

        if "." in self.output.name:
            # assuming a file, since there is an an extension
            self.output.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.output.mkdir(parents=True)

    @property
    def store_hologram(self) -> bool:
        return self._store_hologram

    @property
    def store_gt_hologram(self) -> bool:
        return self._store_gt_hologram

    @property
    def store_phantom(self) -> bool:
        return self._store_phantom

    @property
    def store_probe(self) -> bool:
        return self._store_probe

    @property
    def store_flatfield(self) -> bool:
        return self._store_flatfield

    @property
    def store_polynomial(self) -> bool:
        return self._store_polynomial

    @property
    def store_setup(self) -> bool:
        return self._store_setup

    @property
    def output(self) -> Path:
        return self._output

    @property
    def override(self) -> Path:
        return self._override
