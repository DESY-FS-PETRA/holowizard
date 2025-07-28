# standard libraries
from typing import Tuple

# third party libraries
import numpy as np
import torch

# local libraries
from holowizard.forge.utils.torch_settings import get_torch_device


__all__ = [
    "ObjectPlacement",
]


class ObjectPlacement:  # TODO: naming does not make any sense if it only returns the center pos
    def __init__(
        self,
        positioning: str,
        detector_size: int | None = None,
        probe_size: int | None = None,
    ) -> None:
        self.detector_size = detector_size
        self.probe_size = probe_size
        self._set_placer(positioning)

    def get_pos(self, obj: torch.Tensor) -> Tuple[int, int]:
        return self.placer(obj)

    def _set_placer(self, positioning: str):
        match positioning:
            case "center":
                self.size = self.detector_size
                self.placer = self.place_centered
            case "random":
                self.size = self.detector_size
                self.placer = self.place_anywhere
            # case "probe": # Also outside of field of view
            #     self.size = self.probe_size
            #     self.placer = self.place_anywhere
            # case "bottom":
            # place on the bottom (actually on the left side, since FFs are rotated") or also left/right/top
            case position:
                raise ValueError(f"Invalid option: {position}")

    def place_centered(self, obj: torch.Tensor) -> Tuple[int, int]:
        min_vertical, min_horizontal = obj.shape[0] // 2, obj.shape[1] // 2
        center = self.size // 2
        mu = (center, center)
        sigma = (min_vertical / 3, min_horizontal / 3)
        pos_x, pos_y = np.random.normal(mu, sigma)  # 3*sigma interval to have more "accepted" objects in boundaries
        return int(pos_x), int(pos_y)

    def place_anywhere(self, obj: torch.Tensor) -> Tuple[int, int]:
        low_x, high_x = obj.shape[0] // 2, self.size - obj.shape[0] // 2
        low_y, high_y = obj.shape[1] // 2, self.size - obj.shape[1] // 2
        pos = np.random.randint((low_x, low_y), (high_x, high_y))
        return pos

    @staticmethod
    def object_fits(obj: torch.Tensor, ambient_size: int, center: Tuple[int, int] | None) -> bool:  # TODO: incl. thr
        """Checks whether an object fits.

        Args:
            obj (torch.Tensor): The object that shall be embedded.
            ambient_size (int): Size of the ambient space. Assumed to be a squared 2D-Tensor.
            center (Tuple[int, int] | None): Center of the object in the ambient space.

        Returns:
            bool: True, if the object fits. False otherwise.
        """
        if center is None:
            return max(obj.shape) <= ambient_size

        cx, cy = center
        height, width = obj.shape
        half_height, half_width = height // 2, width // 2
        fits = (
            cx - half_height >= 0
            and cx + half_height < ambient_size
            and cy - half_width >= 0
            and cy + half_width < ambient_size
        )
        return fits

    @staticmethod
    def embedd(ambient_size: int, obj: torch.Tensor, center: Tuple[int, int]) -> torch.Tensor:
        cx, cy = center
        h, w = obj.shape
        start_x = cx - h // 2
        end_x = start_x + h
        start_y = cy - w // 2
        end_y = start_y + w

        ambient = torch.zeros((ambient_size, ambient_size), dtype=obj.dtype, device=get_torch_device())
        ambient[start_x:end_x, start_y:end_y] = obj
        return ambient
