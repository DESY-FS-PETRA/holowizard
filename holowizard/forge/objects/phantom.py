# standard libraries
from typing import Any, Tuple
from pathlib import Path
import torch

# third party libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# local libraries
from holowizard.forge.utils.datatypes import Pathlike, TensorFloat32, TensorComplex64
from holowizard.forge.utils.torch_settings import get_torch_device
from .placement import ObjectPlacement


__all__ = [
    "Phantom",
]


class Phantom:
    def __init__(self, phys_obj: TensorComplex64, size: int, center: Tuple[int, int] = None) -> None:
        self.size = size
        self.phantom = self._embedd(phys_obj, size, center)

    def __add__(self, other):
        if isinstance(other, Phantom):
            composed_phantom = self.phantom + other.phantom
            return Phantom(composed_phantom, self.phantom.shape[0])
        else:
            raise ValueError("Unsupported operand type for '+'.")

    def __iadd__(self, other):
        if isinstance(other, Phantom):
            self.phantom += other.phantom
            return self
        else:
            raise ValueError("Unsupported operand type for '+='.")

    def _embedd(self, physical_object: TensorComplex64, new_size: int, center: Tuple[int, int]) -> TensorComplex64:
        """Embedds a canvas containing objects into an ambient space (usually the xray beam size).

        Args:
            physical_object (TensorComplex64): Complex array containing a shape already with physical properties.
            shape (int): Shape of the ambient space. Must be >= the size of this object's canvas.
            center (Tuple[int, int]): position on canvas, where the object`s center will be lying. c_x and c_y are
                    in [0, phantom_size - 1] (hence, '<' instead of '<=' for upper bound checks). Phantom size is the
                    maximal size of the resulting phantom (composed object).

        Returns:
            TensorComplex64: The embedded object (complex64). Might only contain zeros.
        """
        if center is None:
            if physical_object.shape[0] == new_size:  # already embedded
                return physical_object
            else:
                center = physical_object.shape[0] // 2, physical_object.shape[1] // 2

        if ObjectPlacement.object_fits(obj=physical_object, ambient_size=new_size, center=center):
            return ObjectPlacement.embedd(obj=physical_object, ambient_size=new_size, center=center)
        else:
            return torch.zeros((new_size, new_size), dtype=torch.cfloat, device=get_torch_device())

    def pad_to(self, new_size: int) -> TensorComplex64:
        if new_size < self.size:
            raise ValueError(f"New size ({new_size}) must be greater or equal the phantom`s size ({self.size}).")

        # return self._embedd(self.phantom, new_size=new_size, center=None)
        padding = int((new_size - self.size) / 2)
        padded = torch.nn.functional.pad(self.phantom, pad=[padding] * 4, mode="constant", value=0)
        return padded

    def pad(self, padding: int) -> TensorComplex64:
        if padding < 0:
            raise ValueError(f"Paddinng size ({padding}) must be >= 0.")

        padded = torch.nn.functional.pad(self.phantom, pad=[padding] * 4, mode="constant", value=0)
        return padded

    def set_limits_phaseshift(self, min: float = -torch.inf, max: float = 0) -> None:
        assert 0 >= max >= min, f"Must be: 0 >= {max=} >= {min=}"
        self.phantom.real = torch.clip(self.phantom.real, min=min, max=max)

    def set_limits_absorption(self, min: float, max: float) -> None:
        assert 0 <= min <= max, f"Must be: 0 <= {min=} <= {max=}"
        self.phantom.imag = torch.clip(self.phantom.imag, min=min, max=max)

    def to_tiff(self, output_folder: Pathlike, name: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.phaseshift).save(output_folder / f"{name}_phaseshift.tiff")
        Image.fromarray(self.absorption).save(output_folder / f"{name}_absorption.tiff")

    def show(self, phaseshift: bool = True, absorption: bool = True, suptitle: str = "") -> None:
        self._create_figure(phaseshift, absorption, suptitle)
        plt.show()

    def savefig(
        self,
        path: Pathlike,
        phaseshift: bool = True,
        absorption: bool = True,
        suptitle: str = "",
    ) -> None:
        self._create_figure(phaseshift, absorption, suptitle)
        plt.savefig(path)

    def _create_figure(self, phaseshift: bool, absorption: bool, suptitle: str = "Phantom") -> None:
        plt.close("all")
        plt.ioff()

        def create_subfigure(axis, img: np.ndarray, title: str) -> None:
            real = axis.imshow(img, cmap="gray", interpolation="None")
            axis.title.set_text(title)
            plt.colorbar(real, orientation="vertical", ax=axis)

        num_subplots = phaseshift + absorption
        figsize = (5 * num_subplots, 5)
        fig, axs = plt.subplots(1, num_subplots, figsize=figsize)

        if suptitle is not None:
            fig.suptitle(suptitle)

        axis = 0
        if phaseshift:
            create_subfigure(axs[axis], self.real, "Phaseshift / rad")
            axis += 1

        if absorption:
            create_subfigure(axs[axis], self.imag, "Absorption / A.U.")

    @property
    def phantom(self) -> TensorComplex64:
        return self._phantom

    @phantom.setter
    def phantom(self, value: TensorComplex64) -> None:
        if not (self.size, self.size) == value.shape:
            raise AttributeError(f"Shapes are not the same. Is: {value.shape}). Should be {self._phantom.shape}.")
        self._phantom = value

    @property
    def phaseshift(self) -> TensorFloat32:
        return torch.real(self._phantom)

    @phaseshift.setter
    def phaseshift(self, value: Any) -> None:
        raise AttributeError("Cannot set 'phaseshift' property directly. Modify 'phantom' instead.")

    @property
    def absorption(self) -> TensorFloat32:
        return torch.imag(self.phantom)

    @absorption.setter
    def absorption(self, value: Any) -> None:
        raise AttributeError("Cannot set 'absorption' property directly. Modify 'phantom' instead.")

    @property
    def real(self) -> TensorFloat32:
        return torch.real(self.phantom)

    @real.setter
    def real(self, value: Any) -> None:
        raise AttributeError("Cannot set 'real' property directly. Modify 'phantom' instead.")

    @property
    def imag(self) -> TensorFloat32:
        return torch.imag(self.phantom)

    @imag.setter
    def imag(self, value: Any) -> None:
        raise AttributeError("Cannot set 'imag' property directly. Modify 'phantom' instead.")

    @property
    def O_ref(self) -> TensorComplex64:
        return self.phantom

    @O_ref.setter
    def O_ref(self, value: TensorComplex64) -> None:
        self.phantom = value

    @property
    def physical_object(self) -> TensorComplex64:
        return self.phantom

    @physical_object.setter
    def physical_object(self, value: TensorComplex64) -> None:
        self.phantom = value

    @property
    def refractive_index(self) -> TensorComplex64:
        return self.phantom

    @refractive_index.setter
    def refractive_index(self, value: TensorComplex64) -> None:
        self.phantom = value
