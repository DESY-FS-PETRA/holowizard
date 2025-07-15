# standard libraries

# third party libraries
import matplotlib.pyplot as plt

# local libraries
from holowizard.forge.utils import fileIO


__all__ = [
    "load_phantom_test",
]


def load_phantom_test(
    path: str = "holowizard.forge/tests/output/data/7000/train/phantoms/phantom_000001.pkl", output: str = None
) -> None:
    phantom = fileIO.load_phantom(path)

    plt.subplot(1, 2, 1)
    plt.imshow(phantom.phaseshift, cmap="gray")  # Use 'gray' colormap for grayscale images
    plt.title("Phaseshift")

    plt.subplot(1, 2, 2)
    plt.imshow(phantom.absorption, cmap="gray")
    plt.title("Absorption")

    if output is not None:
        plt.savefig(output)
    plt.show()
