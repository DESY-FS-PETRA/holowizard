"""Array operations of the conversion: downsampling and the mask of the reconstruction cylinder."""

from typing import Tuple

import numpy as np


def downsample2(a: np.ndarray, masked: bool = False) -> np.ndarray:
    """Mean over 2x2x2 blocks; odd sides are padded with their last plane (so sizes are halved, rounded up).

    With ``masked``, NaN marks masked voxels: they are left out of the mean, and blocks of only masked voxels
    stay NaN."""
    pad = [(0, s % 2) for s in a.shape]
    if any(p for _, p in pad):
        a = np.pad(a, pad, mode="edge")
    z, y, x = a.shape
    shape = (z // 2, 2, y // 2, 2, x // 2, 2)
    if not masked:
        return a.reshape(shape).mean(axis=(1, 3, 5), dtype=np.float32)
    valid = ~np.isnan(a)
    total = np.where(valid, a, 0).reshape(shape).sum(axis=(1, 3, 5), dtype=np.float32)
    count = valid.reshape(shape).sum(axis=(1, 3, 5))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (total / count).astype(np.float32)  # 0 / 0 = NaN for fully masked blocks


def cylinder_outside(shape2d: Tuple[int, int], fraction: float = 1.0) -> Tuple[np.ndarray, float]:
    """Pixels of a slice outside the reconstruction cylinder (the inscribed circle, its radius scaled by
    ``fraction`` to also cut off the artefacts at its edge), and the radius."""
    h, w = shape2d
    radius = fraction * min(h, w) / 2
    yy, xx = np.ogrid[:h, :w]
    return (yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2 > radius**2, radius
