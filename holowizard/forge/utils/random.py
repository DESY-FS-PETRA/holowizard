# standard libraries
from typing import Tuple, Any, Sequence
import random

# third party libraries
import torch
import numpy as np

# local libraries
from holowizard.forge.utils.datatypes import TensorFloat32


__all__ = [
    "choice",
    "get_random_idx",
    "randint",
    "set_seed",
    "normal",
    "poisson",
]


def set_seed(seed: float, numpy_seed: bool = True) -> None:
    if numpy_seed:
        np.random.seed(seed)
    else:
        torch.manual_seed(seed)


def choice(sequence: Sequence[Any]) -> Any:
    """Wrapper for `choice` from the standard library `random`.

    See:
        `get_random_idx`
        `random.choice`

    Args:
        sequence (Sequence[Any]): Sequence from which an element is drawn.

    Returns:
        Any: Random element from given sequence.
    """
    return random.choice(sequence)


def randint(val: int, incl_high: bool = False) -> int:
    """Sample a random integer from `[0, val)`

    Args:
        val (int): Defines the sample interval `[0, val)`.
        incl_high (bool, optional): If True, also include `val` in the sample space. Defaults to False.

    Returns:
        int: Random integer.
    """
    if incl_high:
        val += 1
    return np.random.randint(val)


def get_random_idx(array: Sequence) -> int:
    """Retrieves a random integer from in [0, `len(array)`).

    Args:
        array (Sequence): Object that has defined __len__().

    Returns:
        int: Random index for that list.
    """
    return np.random.randint(len(array))


def uniform(lower: float, upper: float, size: int | Tuple[int, ...] = 1) -> TensorFloat32:
    return np.random.uniform(lower, upper, size=size)


def normal(mean: float, std: float, size: int | Tuple[int, ...] = 1) -> torch.Tensor:
    if type(size) == int:
        size = (size,)
    return torch.normal(mean=mean, std=std, size=size)


def poisson(lam: float, size: int | Tuple[int, ...] = 1) -> torch.Tensor:
    rates = torch.ones(size) * lam
    return torch.poisson(rates)
