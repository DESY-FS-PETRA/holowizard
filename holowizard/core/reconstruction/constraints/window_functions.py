import torch
import numpy as np


def hanning(x, width):
    return 0.5 * (1 - torch.cos(x * 2 * np.pi / (width - 1)))


def hamming(x, width):
    return 0.54 - 0.46 * torch.cos(x * 2 * np.pi / (width - 1))


def blackman(x, width):
    return 0.42 - 0.5 * torch.cos(x * 2 * np.pi / (width - 1)) + 0.08 * torch.cos(x * 4 * np.pi / (width - 1))
